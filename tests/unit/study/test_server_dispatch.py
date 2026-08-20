"""Tests for the server-mode wiring on the programmatic study-resolution path.

Covers the server-capable-entry-path contract: a study built from objects (no
YAML, no file touched) resolves through the same entry point as a YAML study, so
the ServerSession always reads the overlay-resolved warmup protocol. Offline
configs stay untouched, and a user config with no warmup layer leaves the
declared protocol in place.
"""

from __future__ import annotations

from llenergymeasure.config.models import (
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
    ServerWarmupConfig,
    StudyConfig,
)
from llenergymeasure.config.user_config import UserConfig, UserServerConfig
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.study.loading import resolve_study_objects


def _server_exp() -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="server",
        server={"traffic": {"rate": 10, "window_seconds": 60}},
    )


def _offline_exp() -> ExperimentConfig:
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="transformers",
        serving_mode="offline",
    )


def _user(mode: str = "fixed") -> UserConfig:
    return UserConfig(server=UserServerConfig(warmup=ServerWarmupConfig(mode=mode)))


def test_overlay_applies_to_a_server_study() -> None:
    """A server study built from objects resolves the overlaid warmup protocol."""
    study = StudyConfig(
        experiments=[_server_exp()],
        study_execution=ExecutionConfig(experiment_order="sequential"),
    )
    resolved_study = resolve_study_objects(study, user_config=_user("fixed"))

    resolved = resolved_study.experiments[0].resolved_server_warmup()
    assert resolved is not None and resolved.mode == "fixed"


def test_overlay_leaves_an_offline_study_untouched() -> None:
    """An offline study gets no warmup overlay, and its declared identity holds."""
    offline = _offline_exp()
    declared_before = compute_declared_config_hash(offline)

    resolved_study = resolve_study_objects(
        StudyConfig(experiments=[offline]), user_config=_user("fixed")
    )
    resolved_offline = resolved_study.experiments[0]

    assert resolved_offline.resolved_server_warmup() is None
    assert compute_declared_config_hash(resolved_offline) == declared_before


def test_resolution_is_stable_across_repeated_calls() -> None:
    """Resolving the same objects twice under one user config agrees exactly."""
    user = _user("fixed")
    first = resolve_study_objects(StudyConfig(experiments=[_server_exp()]), user_config=user)
    second = resolve_study_objects(StudyConfig(experiments=[_server_exp()]), user_config=user)

    warmup_first = first.experiments[0].resolved_server_warmup()
    warmup_second = second.experiments[0].resolved_server_warmup()
    assert warmup_first is not None and warmup_second is not None
    assert warmup_first.mode == warmup_second.mode == "fixed"
    assert first.study_design_hash == second.study_design_hash
    assert first.declared_resolved_config_hashes == second.declared_resolved_config_hashes


def test_offline_only_study_needs_no_overlay() -> None:
    offline = _offline_exp()
    resolved = resolve_study_objects(StudyConfig(experiments=[offline]), user_config=UserConfig())
    assert resolved.experiments[0].resolved_server_warmup() is None


def test_overlay_noop_when_user_config_has_no_warmup_layer() -> None:
    # A server experiment but no user-config warmup layer: resolved falls back to
    # the declared server.warmup (byte-identical to no-overlay behaviour).
    resolved = resolve_study_objects(
        StudyConfig(experiments=[_server_exp()]), user_config=UserConfig()
    )
    warmup = resolved.experiments[0].resolved_server_warmup()
    assert warmup is not None and warmup.mode == "composite"  # the built-in default


def test_resolution_from_objects_writes_no_file(tmp_path) -> None:
    """The programmatic path is pure in-memory: nothing lands on disk."""
    before = set(tmp_path.iterdir())
    resolved = resolve_study_objects(
        StudyConfig(experiments=[_server_exp()]), user_config=_user("fixed")
    )
    assert resolved.study_design_hash is not None
    assert set(tmp_path.iterdir()) == before


def test_output_config_survives_resolution() -> None:
    study = StudyConfig(
        experiments=[_offline_exp()],
        output=OutputConfig(results_dir="/tmp/somewhere"),
    )
    resolved = resolve_study_objects(study, user_config=UserConfig())
    assert resolved.output.results_dir == "/tmp/somewhere"
