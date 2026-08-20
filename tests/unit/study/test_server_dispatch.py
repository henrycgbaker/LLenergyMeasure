"""Tests for the server-mode wiring on the programmatic study-resolution path.

Covers the server-capable-entry-path contract: a study built from objects (no
YAML, no file touched) resolves through the same entry point as a YAML study, so
the ServerSession always reads the overlay-resolved warmup protocol. Offline
configs stay untouched, and a user config with no warmup layer leaves the
declared protocol in place.
"""

from __future__ import annotations

from llenergymeasure.config.loader import LoadedStudyRaw
from llenergymeasure.config.models import (
    ExecutionConfig,
    ExperimentConfig,
    OutputConfig,
    ServerWarmupConfig,
    StudyConfig,
)
from llenergymeasure.config.user_config import UserConfig, UserServerConfig
from llenergymeasure.domain.experiment import compute_declared_config_hash
from llenergymeasure.study.loading import resolve_study


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


def _resolve_from_objects(study: StudyConfig, user_config: UserConfig) -> StudyConfig:
    """Resolve a caller-built StudyConfig the way run_study does, without a file."""
    raw = LoadedStudyRaw(
        valid_experiments=list(study.experiments),
        skipped=[],
        study_name=study.study_name,
        output=study.output,
        execution=study.study_execution,
        runners=study.runners,
        images=study.images,
    )
    return resolve_study(raw, user_config=user_config)


def test_overlay_applies_to_server_leaves_offline_untouched() -> None:
    server, offline = _server_exp(), _offline_exp()
    # Sequential order keeps the resolved list in declared order.
    study = StudyConfig(
        experiments=[server, offline],
        study_execution=ExecutionConfig(experiment_order="sequential"),
    )
    offline_declared_before = compute_declared_config_hash(offline)

    resolved_study = _resolve_from_objects(study, _user("fixed"))
    resolved_server, resolved_offline = resolved_study.experiments

    # Server experiment now resolves the overlaid (fixed) protocol...
    resolved = resolved_server.resolved_server_warmup()
    assert resolved is not None and resolved.mode == "fixed"
    # ...while the offline experiment is untouched (no server warmup, hash stable).
    assert resolved_offline.resolved_server_warmup() is None
    assert compute_declared_config_hash(resolved_offline) == offline_declared_before


def test_resolution_is_stable_across_repeated_calls() -> None:
    """Resolving the same objects twice under one user config agrees exactly."""
    user = _user("fixed")
    first = _resolve_from_objects(StudyConfig(experiments=[_server_exp()]), user)
    second = _resolve_from_objects(StudyConfig(experiments=[_server_exp()]), user)

    warmup_first = first.experiments[0].resolved_server_warmup()
    warmup_second = second.experiments[0].resolved_server_warmup()
    assert warmup_first is not None and warmup_second is not None
    assert warmup_first.mode == warmup_second.mode == "fixed"
    assert first.study_design_hash == second.study_design_hash
    assert first.declared_resolved_config_hashes == second.declared_resolved_config_hashes


def test_offline_only_study_needs_no_overlay() -> None:
    offline = _offline_exp()
    resolved = _resolve_from_objects(StudyConfig(experiments=[offline]), UserConfig())
    assert resolved.experiments[0].resolved_server_warmup() is None


def test_overlay_noop_when_user_config_has_no_warmup_layer() -> None:
    # A server experiment but no user-config warmup layer: resolved falls back to
    # the declared server.warmup (byte-identical to no-overlay behaviour).
    resolved = _resolve_from_objects(StudyConfig(experiments=[_server_exp()]), UserConfig())
    warmup = resolved.experiments[0].resolved_server_warmup()
    assert warmup is not None and warmup.mode == "composite"  # the built-in default


def test_resolution_from_objects_writes_no_file(tmp_path) -> None:
    """The programmatic path is pure in-memory: nothing lands on disk."""
    before = set(tmp_path.iterdir())
    resolved = _resolve_from_objects(StudyConfig(experiments=[_server_exp()]), _user("fixed"))
    assert resolved.study_design_hash is not None
    assert set(tmp_path.iterdir()) == before


def test_output_config_survives_resolution() -> None:
    study = StudyConfig(
        experiments=[_offline_exp()],
        output=OutputConfig(results_dir="/tmp/somewhere"),
    )
    resolved = _resolve_from_objects(study, UserConfig())
    assert resolved.output.results_dir == "/tmp/somewhere"
