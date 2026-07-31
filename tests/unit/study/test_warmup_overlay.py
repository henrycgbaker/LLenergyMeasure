"""Tests for the R7W user-config server-warmup overlay wiring.

The overlay resolves the effective server warmup protocol through the R7 chain
(built-in defaults < user config < study YAML) and attaches the OUTPUT as
side-channel state on the ExperimentConfig. The declared-config hash is untouched
(it keeps naming the shareable study intent); the resolved/observed hashes carry
the realised protocol, so the dedup machinery treats one study run under two
different user-config warmups as distinct measurements.

The six charter pins:
  (a) same study YAML + two user-config warmups -> declared hashes identical,
      resolved hashes differ, and the DEDUP machinery treats them as distinct;
  (b) a study-YAML warmup value overrides the user config;
  (c) no user config -> resolved output byte-identical to today's behaviour;
  (d) offline configs + the declared-hash pin are untouched by the overlay;
  (e) a user config setting a field to the built-in default's VALUE still counts
      as user-supplied (fields_set semantics) - resolved reflects it, declared not;
  (f) a partial (one-field) user config overlays only that field.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from llenergymeasure.config.loader import load_study_config
from llenergymeasure.config.models import ExperimentConfig, ServerWarmupConfig
from llenergymeasure.config.precedence import apply_server_warmup_overlay
from llenergymeasure.config.user_config import (
    UserConfig,
    UserServerConfig,
    load_user_config,
)
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.domain.hashing import hash_config
from llenergymeasure.study.hashing import build_resolved_view
from llenergymeasure.study.library_resolution import resolve_library_effective
from llenergymeasure.study.loading import finalise_study

_TRAFFIC = {"rate": 10, "window_seconds": 60}


def _server(**server_overrides) -> ExperimentConfig:
    """A minimal vLLM server-mode config (transformers+server is rejected)."""
    server = {"traffic": dict(_TRAFFIC)}
    server.update(server_overrides)
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="server",
        server=server,
    )


def _user(**warmup_fields) -> UserConfig:
    """A UserConfig whose server.warmup sets exactly ``warmup_fields``."""
    return UserConfig(server=UserServerConfig(warmup=ServerWarmupConfig(**warmup_fields)))


def _declared(config: ExperimentConfig) -> str:
    return compute_declared_config_hash(config)


def _resolved(config: ExperimentConfig) -> str:
    return hash_config(build_resolved_view(config))


# ---------------------------------------------------------------------------
# UserConfig home (the R1 mode-grammar mirror)
# ---------------------------------------------------------------------------


def test_user_config_server_warmup_home_defaults_to_none() -> None:
    """A user config with no server section leaves the overlay unfed."""
    assert UserConfig().server is None


def test_user_config_server_warmup_loads_from_disk(tmp_path: Path) -> None:
    """The server.warmup home is a real YAML surface (not just a Python model)."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"server": {"warmup": {"mode": "fixed"}}}))
    user_config = load_user_config(cfg_path)
    assert user_config.server is not None
    assert user_config.server.warmup.mode == "fixed"
    # Only the field the user wrote is in fields_set (per-field overlay awareness).
    assert user_config.server.warmup.model_fields_set == {"mode"}


# ---------------------------------------------------------------------------
# (c) No user config -> byte-identical to today
# ---------------------------------------------------------------------------


def test_no_overlay_leaves_config_untouched() -> None:
    config = _server()
    declared_before, resolved_before = _declared(config), _resolved(config)
    # A user config with no server section is a no-op.
    apply_server_warmup_overlay(config, UserConfig())
    assert config._resolved_server_warmup is None
    assert _declared(config) == declared_before
    assert _resolved(config) == resolved_before
    # The seam still reports a warmup for the server session (the declared one).
    assert config.resolved_server_warmup() == config.server.warmup


def test_offline_config_is_not_a_warmup_target() -> None:
    """The overlay is server-only; offline configs are never touched (pin d)."""
    offline = ExperimentConfig(task={"model": "gpt2"}, engine="vllm", serving_mode="offline")
    declared_before, resolved_before = _declared(offline), _resolved(offline)
    apply_server_warmup_overlay(offline, _user(mode="fixed", duration_seconds=120))
    assert offline._resolved_server_warmup is None
    assert offline.resolved_server_warmup() is None
    assert _declared(offline) == declared_before
    assert _resolved(offline) == resolved_before


# ---------------------------------------------------------------------------
# (d) Declared hash untouched by the overlay
# ---------------------------------------------------------------------------


def test_overlay_never_shifts_the_declared_hash() -> None:
    """The declared hash names user intent - the overlay must not enter it (pin d)."""
    baseline = _declared(_server())
    for user in (
        _user(mode="fixed"),
        _user(duration_seconds=42),
        _user(mode="composite", timeout_seconds=123, duration_seconds=7),
    ):
        config = _server()
        apply_server_warmup_overlay(config, user)
        assert _declared(config) == baseline


# ---------------------------------------------------------------------------
# (e/f) Overlay resolves in the resolved view; per-field; fields_set semantics
# ---------------------------------------------------------------------------


def test_partial_user_config_overlays_only_that_field() -> None:
    """One user field fills a study-unset field; the rest stay at defaults (pin f)."""
    config = _server()
    apply_server_warmup_overlay(config, _user(duration_seconds=120))
    warmup = config.resolved_server_warmup()
    assert warmup.duration_seconds == 120.0  # user-supplied
    assert warmup.mode == "composite"  # built-in default (unfed)
    assert warmup.timeout_seconds == 900.0  # built-in default (unfed)
    # The resolved hash moved; the declared hash did not.
    assert _resolved(config) != _resolved(_server())
    assert _declared(config) == _declared(_server())


def test_user_field_set_to_default_value_counts_as_supplied() -> None:
    """A user field set to the built-in default's VALUE is still user-supplied (pin e).

    fields_set (not value inequality) governs what the overlay layer carries, so a
    user who explicitly writes mode=composite (the default) has supplied it - it
    wins over a study-unset field and appears in the resolved protocol.
    """
    user = _user(mode="composite", duration_seconds=250)
    # mode is set to its default VALUE but is still in fields_set.
    assert user.server is not None
    assert "mode" in user.server.warmup.model_fields_set

    config = _server()
    apply_server_warmup_overlay(config, user)
    warmup = config.resolved_server_warmup()
    assert warmup.mode == "composite"  # reflected from the user overlay
    assert warmup.duration_seconds == 250.0
    # Declared hash is unaffected regardless.
    assert _declared(config) == _declared(_server())


# ---------------------------------------------------------------------------
# (b) Study YAML wins over the user config
# ---------------------------------------------------------------------------


def test_study_yaml_warmup_overrides_user_config() -> None:
    """A field the study YAML wrote is never overlaid; study wins (pin b)."""
    # Study explicitly sets mode=composite; user config sets mode=fixed + a
    # study-unset duration.
    config = _server(warmup={"mode": "composite"})
    apply_server_warmup_overlay(config, _user(mode="fixed", duration_seconds=180))
    warmup = config.resolved_server_warmup()
    assert warmup.mode == "composite"  # study YAML wins the contested field
    assert warmup.duration_seconds == 180.0  # user fills the study-unset field


# ---------------------------------------------------------------------------
# (a) The dedup machinery binds on the resolved family
# ---------------------------------------------------------------------------


def test_side_channel_survives_sweep_dedup_deep_copy() -> None:
    """The overlay rides the dedup canonicalisation deep copy through to the runner."""
    config = _server()
    apply_server_warmup_overlay(config, _user(mode="fixed", duration_seconds=99))
    copied = config.model_copy(deep=True)
    assert copied.resolved_server_warmup().mode == "fixed"
    assert copied.resolved_server_warmup().duration_seconds == 99.0


def test_dedup_machinery_treats_different_overlays_as_distinct() -> None:
    """Pin (a): drive resolve_library_effective, not just a hash compare.

    Two declared-identical server configs under DIFFERENT user-config warmups must
    NOT collapse together; two under the SAME warmup MUST collapse.
    """
    user_a = _user(mode="fixed", duration_seconds=100)
    user_b = _user(mode="fixed", duration_seconds=200)

    cfg_a, cfg_b = _server(), _server()
    apply_server_warmup_overlay(cfg_a, user_a)
    apply_server_warmup_overlay(cfg_b, user_b)

    # Declared identity is shared (same study design)...
    assert _declared(cfg_a) == _declared(cfg_b)
    # ...but the dedup machinery keeps them as two distinct measurements.
    distinct = resolve_library_effective([cfg_a, cfg_b], deduplicate=True)
    assert len(distinct.groups) == 2
    assert not distinct.would_dedup
    assert len(distinct.canonical_configs) == 2

    # Control: identical overlays DO collapse to one run.
    cfg_c, cfg_d = _server(), _server()
    apply_server_warmup_overlay(cfg_c, user_a)
    apply_server_warmup_overlay(cfg_d, user_a)
    collapsed = resolve_library_effective([cfg_c, cfg_d], deduplicate=True)
    assert len(collapsed.groups) == 1
    assert collapsed.would_dedup
    assert len(collapsed.canonical_configs) == 1


# ---------------------------------------------------------------------------
# (a/c/d) End-to-end through finalise_study (the study composition point)
# ---------------------------------------------------------------------------


def _server_study(tmp_path: Path):
    study = {
        "study_name": "warmup_overlay",
        "serving_mode": "server",
        "engine": "vllm",
        "task": {"model": "gpt2"},
        "server": {"traffic": {"rate": 10, "window_seconds": 60}},
    }
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(study))
    return load_study_config(path)


def test_finalise_study_binds_dedup_on_the_overlay(tmp_path: Path) -> None:
    """Pin (a), end-to-end: two finalise_study runs of one study YAML under two
    user configs share declared identity but differ in the resolved family that
    dedup/resume key on."""
    raw_a = _server_study(tmp_path)
    raw_b = _server_study(tmp_path)

    finalised_a = finalise_study(raw_a, user_config=_user(mode="fixed", duration_seconds=100))
    finalised_b = finalise_study(raw_b, user_config=_user(mode="fixed", duration_seconds=200))

    # Declared identity (experiment_id) is byte-identical across the two runs.
    exp_a, exp_b = finalised_a.experiments[0], finalised_b.experiments[0]
    assert _declared(exp_a) == _declared(exp_b)
    # The resolved-config-hash family (what dedup/resume bind on) differs.
    assert (
        finalised_a.declared_resolved_config_hashes != finalised_b.declared_resolved_config_hashes
    )


def test_finalise_study_without_user_config_is_todays_behaviour(tmp_path: Path) -> None:
    """Pin (c), end-to-end: no user_config -> resolved family byte-identical."""
    raw_none = _server_study(tmp_path)
    raw_empty = _server_study(tmp_path)

    finalised_none = finalise_study(raw_none)  # user_config defaults to None
    finalised_empty = finalise_study(raw_empty, user_config=UserConfig())

    assert (
        finalised_none.declared_resolved_config_hashes
        == finalised_empty.declared_resolved_config_hashes
    )
    # And the config carries no overlay side-channel.
    assert finalised_none.experiments[0]._resolved_server_warmup is None
