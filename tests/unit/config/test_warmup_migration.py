"""Tests for the per-mode warmup grammar (SM8 R1): the D22 clean-break migration
of ``measurement.warmup`` into the ``offline:`` / ``server:`` mode namespaces.

Covers the SM8 verify charter's config half:
- ``measurement.warmup`` is a clean break with an actionable migration error;
- ``offline.warmup`` carries the migrated warmup verbatim and is OPTIONAL;
- ``server.warmup`` is the new composite/fixed protocol block with structural
  absence of a thermal-floor knob;
- both warmup blocks project into BOTH hash families (no new slo-only exclusion);
- the golden-hash shift for a default-offline config is deliberate;
- wrong-mode section placement is rejected in both directions.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.models import (
    ExperimentConfig,
    OfflineSection,
    ServerWarmupConfig,
    WarmupConfig,
)
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.domain.hashing import hash_config
from llenergymeasure.study.hashing import build_resolved_view


def _offline(**overrides) -> ExperimentConfig:
    base: dict = {"task": {"model": "gpt2"}, "engine": "vllm", "serving_mode": "offline"}
    base.update(overrides)
    return ExperimentConfig(**base)


def _server(**overrides) -> ExperimentConfig:
    server = {"traffic": {"rate": 10, "window_seconds": 60}}
    server.update(overrides.pop("server", {}))
    base: dict = {
        "task": {"model": "gpt2"},
        "engine": "vllm",
        "serving_mode": "server",
        "server": server,
    }
    base.update(overrides)
    return ExperimentConfig(**base)


# ---------------------------------------------------------------------------
# Clean-break migration
# ---------------------------------------------------------------------------


class TestMeasurementWarmupMigration:
    def test_measurement_warmup_rejected_with_actionable_error(self):
        with pytest.raises(ValueError) as exc:
            _offline(measurement={"warmup": {"n_prompts": 3}})
        msg = str(exc.value)
        assert "offline.warmup" in msg
        assert "measurement" in msg

    def test_measurement_still_accepts_baseline_and_sampler(self):
        cfg = _offline(measurement={"baseline": {"enabled": False}, "energy_sampler": "nvml"})
        assert cfg.measurement.baseline.enabled is False
        assert cfg.measurement.energy_sampler == "nvml"
        # MeasurementConfig no longer carries a warmup field at all.
        assert "warmup" not in type(cfg.measurement).model_fields


# ---------------------------------------------------------------------------
# offline.warmup namespace
# ---------------------------------------------------------------------------


class TestOfflineWarmup:
    def test_offline_section_optional_defaults_apply(self):
        cfg = _offline()
        assert cfg.offline is None
        # The accessor returns built-in defaults when the section is absent.
        assert cfg.offline_warmup().enabled is True
        assert cfg.offline_warmup().n_prompts == 5

    def test_offline_warmup_carried_verbatim(self):
        cfg = _offline(offline={"warmup": {"n_prompts": 9, "thermal_floor_seconds": 45.0}})
        assert cfg.offline_warmup().n_prompts == 9
        assert cfg.offline_warmup().thermal_floor_seconds == 45.0
        # thermal_floor_seconds keeps its offline idle-settling semantics + floor.
        assert isinstance(cfg.offline.warmup, WarmupConfig)

    def test_offline_warmup_thermal_floor_minimum_enforced(self):
        with pytest.raises((ValidationError, ValueError)):
            _offline(offline={"warmup": {"thermal_floor_seconds": 10.0}})

    def test_offline_section_default_factory(self):
        assert OfflineSection().warmup.n_prompts == 5


# ---------------------------------------------------------------------------
# server.warmup namespace (composite / fixed protocol)
# ---------------------------------------------------------------------------


class TestServerWarmup:
    def test_defaults_are_composite_gate(self):
        cfg = _server()
        assert cfg.server.warmup.mode == "composite"
        assert cfg.server.warmup.timeout_seconds == 900.0
        assert cfg.server.warmup.duration_seconds == 300.0

    def test_no_thermal_floor_knob(self):
        # Structural absence (R1): loaded equilibrium IS the server thermal posture.
        assert "thermal_floor_seconds" not in ServerWarmupConfig.model_fields

    def test_mode_is_closed_literal(self):
        with pytest.raises((ValidationError, ValueError)):
            _server(server={"warmup": {"mode": "adaptive"}})

    def test_timeout_must_be_positive(self):
        with pytest.raises((ValidationError, ValueError)):
            _server(server={"warmup": {"timeout_seconds": 0.0}})

    def test_fixed_duration_zero_allowed_skip(self):
        # duration_seconds ge=0: 0 is the explicit skip-warmup-traffic choice.
        cfg = _server(server={"warmup": {"mode": "fixed", "duration_seconds": 0.0}})
        assert cfg.server.warmup.duration_seconds == 0.0


# ---------------------------------------------------------------------------
# Identity: both warmup blocks enter BOTH hash families; slo stays sole exclusion
# ---------------------------------------------------------------------------


class TestWarmupIdentity:
    def test_offline_warmup_in_both_hash_families(self):
        a = _offline()
        b = _offline(offline={"warmup": {"n_prompts": 9}})
        assert compute_declared_config_hash(a) != compute_declared_config_hash(b)
        assert hash_config(build_resolved_view(a)) != hash_config(build_resolved_view(b))

    def test_server_warmup_in_both_hash_families(self):
        a = _server()
        b = _server(server={"warmup": {"mode": "fixed", "duration_seconds": 120}})
        assert compute_declared_config_hash(a) != compute_declared_config_hash(b)
        assert hash_config(build_resolved_view(a)) != hash_config(build_resolved_view(b))

    def test_server_mode_section_projects_warmup(self):
        cfg = _server()
        mode_section = cfg.mode_section_identity()
        assert set(mode_section) == {"traffic", "warmup", "cooldown_seconds"}
        assert mode_section["warmup"]["mode"] == "composite"

    def test_offline_mode_section_empty_when_no_section(self):
        # "empty for default-offline" (R1): the mode_section slot stays {}.
        assert _offline().mode_section_identity() == {}

    def test_offline_mode_section_projects_warmup_when_present(self):
        cfg = _offline(offline={"warmup": {"n_prompts": 7}})
        assert cfg.mode_section_identity() == {
            "warmup": cfg.offline.warmup.model_dump(mode="python")
        }


# ---------------------------------------------------------------------------
# Wrong-mode placement rejected in both directions; server still required
# ---------------------------------------------------------------------------


class TestModeSectionPlacement:
    def test_offline_section_under_server_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            _server(offline={"warmup": {"n_prompts": 3}})

    def test_server_section_under_offline_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            _offline(server={"traffic": {"rate": 5}})

    def test_server_mode_still_requires_section(self):
        with pytest.raises((ValidationError, ValueError)):
            ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="server")

    def test_bare_offline_valid(self):
        # offline: is optional - a bare offline config never requires the section.
        cfg = ExperimentConfig(task={"model": "gpt2"}, serving_mode="offline")
        assert cfg.offline is None
