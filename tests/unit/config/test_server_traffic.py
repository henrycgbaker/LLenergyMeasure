"""Tests for the server: mode namespace, traffic config, and the slo hash exclusion.

Covers the SM4 verify charter:
- the generic mode-section match validator (server: legal iff serving_mode=server);
- the dual-family slo exclusion (excluded from BOTH hash families, stamped in the dump);
- rate as a hashed identity axis in both families;
- mode-scoped sweep axes (server.traffic.rate) resolving through the existing
  dotted-path sweep machinery;
- offline configs round-tripping both hash pipelines unchanged (mode_section {}).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llenergymeasure.config.grid import expand_grid
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.domain.hashing import hash_config
from llenergymeasure.study.hashing import build_resolved_view


# These mode/traffic/hash tests are engine-agnostic in intent; they use vllm (a
# server-capable engine) so both the offline and server sides are admissible.
# transformers server mode is gated off as a fast-follow (E5), so a transformers
# server config would be rejected before these mechanics could be exercised.
def _offline(**overrides) -> ExperimentConfig:
    base: dict = {"task": {"model": "gpt2"}, "engine": "vllm", "serving_mode": "offline"}
    base.update(overrides)
    return ExperimentConfig(**base)


def _server(traffic: dict, **overrides) -> ExperimentConfig:
    base: dict = {
        "task": {"model": "gpt2"},
        "engine": "vllm",
        "serving_mode": "server",
        "server": {"traffic": traffic},
    }
    base.update(overrides)
    return ExperimentConfig(**base)


_TRAFFIC = {"rate": 10, "window_seconds": 60}


# ---------------------------------------------------------------------------
# Model shape + validators
# ---------------------------------------------------------------------------


class TestTrafficShape:
    def test_minimal_server_config_valid(self):
        cfg = _server(_TRAFFIC)
        assert cfg.serving_mode == "server"
        assert cfg.server is not None
        assert cfg.server.traffic.rate == 10.0
        assert cfg.server.traffic.arrival == "poisson"

    def test_rate_required(self):
        with pytest.raises((ValidationError, ValueError)):
            _server({"window_seconds": 60})

    def test_window_defaults_when_neither_set(self):
        # SM7 CONFIGURABILITY AMENDMENT: window duration is a config-exposed DEFAULT
        # (E2 floor 240s), not a required field - a server config may omit the window.
        cfg = _server({"rate": 10})
        assert cfg.server.traffic.window_seconds == 240.0
        assert cfg.server.traffic.window_requests is None

    def test_window_at_most_one_both_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="at most one"):
            _server({"rate": 10, "window_seconds": 60, "window_requests": 100})

    def test_ramp_exclusion_default_is_30s(self):
        assert _server(_TRAFFIC).server.traffic.ramp_exclusion_seconds == 30.0

    def test_ramp_exclusion_configurable_and_zero_allowed(self):
        assert (
            _server({**_TRAFFIC, "ramp_exclusion_seconds": 0}).server.traffic.ramp_exclusion_seconds
            == 0.0
        )
        assert (
            _server(
                {**_TRAFFIC, "ramp_exclusion_seconds": 45}
            ).server.traffic.ramp_exclusion_seconds
            == 45.0
        )

    def test_ramp_exclusion_negative_rejected(self):
        with pytest.raises((ValidationError, ValueError)):
            _server({**_TRAFFIC, "ramp_exclusion_seconds": -1})

    def test_cooldown_default_is_zero(self):
        assert _server(_TRAFFIC).server.cooldown_seconds == 0.0

    def test_cooldown_configurable(self):
        cfg = ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            serving_mode="server",
            server={"traffic": _TRAFFIC, "cooldown_seconds": 30},
        )
        assert cfg.server.cooldown_seconds == 30.0

    def test_window_requests_alone_valid(self):
        cfg = _server({"rate": 10, "window_requests": 500})
        assert cfg.server.traffic.window_requests == 500
        assert cfg.server.traffic.window_seconds is None

    def test_slo_shared_percentile_default(self):
        cfg = _server({**_TRAFFIC, "slo": {"ttft_ms": 200, "tpot_ms": 20}})
        assert cfg.server.traffic.slo.percentile == 0.99

    def test_traffic_forbids_extra_keys(self):
        with pytest.raises((ValidationError, ValueError)):
            _server({**_TRAFFIC, "not_a_field": 1})


class TestModeSectionMatch:
    """The generic mode-section match (mode's analogue of the engine-section match)."""

    def test_server_section_under_offline_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="section provided but"):
            ExperimentConfig(
                task={"model": "gpt2"}, serving_mode="offline", server={"traffic": _TRAFFIC}
            )

    def test_server_mode_without_section_rejected(self):
        with pytest.raises((ValidationError, ValueError), match="requires a server"):
            ExperimentConfig(task={"model": "gpt2"}, serving_mode="server")

    def test_offline_without_section_valid(self):
        # offline has no mandatory mode namespace: a plain offline config is valid.
        assert _offline().server is None

    def test_serving_mode_required_no_default(self):
        with pytest.raises((ValidationError, ValueError), match="serving_mode is required"):
            ExperimentConfig(task={"model": "gpt2"}, engine="transformers")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Dual-family slo exclusion (O5.3) + rate identity (C4)
# ---------------------------------------------------------------------------


class TestSloHashExclusion:
    def test_slo_excluded_from_declared_hash(self):
        # (a) two configs differing ONLY in slo bounds hash identically (declared).
        a = _server({**_TRAFFIC, "slo": {"ttft_ms": 200, "tpot_ms": 20}})
        b = _server({**_TRAFFIC, "slo": {"ttft_ms": 999, "tpot_ms": 99}})
        assert compute_declared_config_hash(a) == compute_declared_config_hash(b)

    def test_slo_excluded_from_resolved_hash(self):
        # (a) two configs differing ONLY in slo bounds hash identically (resolved/observed).
        a = _server({**_TRAFFIC, "slo": {"ttft_ms": 200, "tpot_ms": 20}})
        b = _server({**_TRAFFIC, "slo": {"ttft_ms": 999, "tpot_ms": 99}})
        assert hash_config(build_resolved_view(a)) == hash_config(build_resolved_view(b))

    def test_slo_present_vs_absent_hashes_identically(self):
        # slo is a pure overlay: adding it must not change identity in either family.
        without = _server(_TRAFFIC)
        with_slo = _server({**_TRAFFIC, "slo": {"ttft_ms": 200}})
        assert compute_declared_config_hash(without) == compute_declared_config_hash(with_slo)
        assert hash_config(build_resolved_view(without)) == hash_config(
            build_resolved_view(with_slo)
        )

    def test_slo_stamped_in_declared_dump_despite_exclusion(self):
        # (c) slo stays in the config sidecar dump (config.model_dump is what
        # harness.staging writes as the declared_config provenance block) even
        # though it is excluded from the identity hash. NOT Field(exclude=True).
        cfg = _server({**_TRAFFIC, "slo": {"ttft_ms": 200, "tpot_ms": 20, "percentile": 0.95}})
        dumped = cfg.model_dump(mode="json")
        assert dumped["server"]["traffic"]["slo"] == {
            "ttft_ms": 200.0,
            "tpot_ms": 20.0,
            "percentile": 0.95,
        }

    def test_slo_present_in_written_config_sidecar(self, tmp_path):
        # (c) companion, pinned at the artefact level: drive the real sidecar
        # writer end-to-end and assert slo survives into the written config.json
        # declared block, so a future edit to the sidecar call site cannot
        # silently drop the disclosure.
        import json
        from types import SimpleNamespace

        from llenergymeasure.domain.bundle_artefacts import CONFIG_SIDECAR_FILENAME
        from llenergymeasure.harness.result_assembly import _ConfigMethodology
        from llenergymeasure.harness.staging import write_config_sidecar

        cfg = _server({**_TRAFFIC, "slo": {"ttft_ms": 200, "tpot_ms": 20}})
        output = SimpleNamespace(
            extras={
                "observed_engine_params": {},
                "observed_sampling_params": {},
                "library_version": "test",
            }
        )
        result = SimpleNamespace(
            experiment_id="slo-sidecar-001", declared_config_hash="abc123def4567890"
        )
        methodology = _ConfigMethodology(
            measurement_methodology="total",
            steady_state_window=None,
            measurement_window_discard_fraction=None,
            steady_state_not_detected=False,
        )
        write_config_sidecar(
            output=output,
            config=cfg,
            result=result,
            engine_name="vllm",
            methodology=methodology,
            output_dir=tmp_path,
        )
        written = json.loads((tmp_path / CONFIG_SIDECAR_FILENAME).read_text())
        slo = written["declared_config"]["server"]["traffic"]["slo"]
        assert slo["ttft_ms"] == 200.0 and slo["tpot_ms"] == 20.0


class TestRateIdentity:
    def test_rate_distinct_declared_hash(self):
        # (b) differing only in rate -> distinct declared hash (C4).
        assert compute_declared_config_hash(_server({"rate": 2, "window_seconds": 60})) != (
            compute_declared_config_hash(_server({"rate": 10, "window_seconds": 60}))
        )

    def test_rate_distinct_resolved_hash(self):
        # (b) differing only in rate -> distinct resolved hash (C4).
        h2 = hash_config(build_resolved_view(_server({"rate": 2, "window_seconds": 60})))
        h10 = hash_config(build_resolved_view(_server({"rate": 10, "window_seconds": 60})))
        assert h2 != h10

    def test_arrival_and_window_are_identity(self):
        # non-slo traffic fields join identity: a sweep over them stays distinct.
        base = _server(_TRAFFIC)
        gamma = _server({**_TRAFFIC, "arrival": "gamma", "burstiness": 2.0})
        by_requests = _server({"rate": 10, "window_requests": 500})
        assert compute_declared_config_hash(base) != compute_declared_config_hash(gamma)
        assert compute_declared_config_hash(base) != compute_declared_config_hash(by_requests)


class TestWindowKnobIdentity:
    """SM7 window-duration / ramp / cooldown are identity in BOTH hash families.

    They follow the existing discipline with NO new exclusion (server.traffic.slo
    stays the sole declared-hash exclusion): user-set values enter the declared hash
    (wholesale dump) and the resolved/observed views (mode_section projection).
    """

    def test_ramp_exclusion_is_identity_both_families(self):
        a = _server({**_TRAFFIC, "ramp_exclusion_seconds": 30})
        b = _server({**_TRAFFIC, "ramp_exclusion_seconds": 45})
        assert compute_declared_config_hash(a) != compute_declared_config_hash(b)
        assert hash_config(build_resolved_view(a)) != hash_config(build_resolved_view(b))

    def _server_cd(self, cooldown: float):
        return ExperimentConfig(
            task={"model": "gpt2"},
            engine="vllm",
            serving_mode="server",
            server={"traffic": _TRAFFIC, "cooldown_seconds": cooldown},
        )

    def test_cooldown_is_identity_both_families(self):
        a = self._server_cd(0)
        b = self._server_cd(30)
        assert compute_declared_config_hash(a) != compute_declared_config_hash(b)
        assert hash_config(build_resolved_view(a)) != hash_config(build_resolved_view(b))

    def test_cooldown_projected_into_mode_section(self):
        proj = self._server_cd(30).mode_section_identity()
        assert proj["cooldown_seconds"] == 30.0
        assert "traffic" in proj

    def test_ramp_exclusion_in_mode_section_traffic(self):
        proj = _server({**_TRAFFIC, "ramp_exclusion_seconds": 45}).mode_section_identity()
        assert proj["traffic"]["ramp_exclusion_seconds"] == 45.0


# ---------------------------------------------------------------------------
# Offline stability (e) + mode_section projection
# ---------------------------------------------------------------------------


class TestOfflineStability:
    def test_offline_round_trips_both_pipelines(self):
        # (e) an offline config with no server section round-trips both hash
        # pipelines: server=None projects mode_section={}, so hashing is stable.
        cfg = _offline()
        assert cfg.mode_section_identity() == {}
        d1 = compute_declared_config_hash(cfg)
        d2 = compute_declared_config_hash(_offline())
        assert d1 == d2
        r1 = hash_config(build_resolved_view(cfg))
        r2 = hash_config(build_resolved_view(_offline()))
        assert r1 == r2

    def test_offline_and_server_never_dedup(self):
        assert compute_declared_config_hash(_offline()) != compute_declared_config_hash(
            _server(_TRAFFIC)
        )

    def test_mode_section_projects_traffic_minus_slo(self):
        cfg = _server({**_TRAFFIC, "slo": {"ttft_ms": 200}})
        projected = cfg.mode_section_identity()
        assert "traffic" in projected
        assert "slo" not in projected["traffic"]
        assert projected["traffic"]["rate"] == 10.0


# ---------------------------------------------------------------------------
# Mode-scoped sweep axis resolves through the existing dotted-path machinery
# ---------------------------------------------------------------------------


class TestModeScopedSweep:
    def test_server_traffic_rate_sweep_expands_and_hashes_distinctly(self):
        # server.traffic.rate: [2, 10] must resolve through the existing dotted-path
        # sweep machinery (no new machinery) into two independent experiments whose
        # resolved hashes differ - proving the axis is not a silent no-op (C4).
        raw_study = {
            "serving_mode": "server",
            "engine": "vllm",
            "task": {"model": "gpt2"},
            "server": {"traffic": {"rate": 2, "window_seconds": 60}},
            "sweep": {"server.traffic.rate": [2, 10]},
        }
        valid, skipped = expand_grid(raw_study)
        assert not skipped, skipped
        assert len(valid) == 2
        rates = sorted(c.server.traffic.rate for c in valid)
        assert rates == [2.0, 10.0]
        hashes = {hash_config(build_resolved_view(c)) for c in valid}
        assert len(hashes) == 2
