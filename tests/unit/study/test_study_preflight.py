"""Tests for study-level pre-flight checks (CM-10, DOCK-05).

Multi-engine Docker elevation is precedence-based: engines the user explicitly
pinned (env / YAML / user config) keep their runner, while engines whose runner
resolved from auto-detection or the default are elevated to Docker for
isolation. Engines pinned to local are checked for host importability; Docker is
only required when an auto-resolved engine actually needs elevating.
"""

from unittest.mock import MagicMock

import pytest

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.study.preflight import run_study_preflight
from llenergymeasure.utils.exceptions import PreFlightError


def test_single_engine_passes(monkeypatch):
    """Single-engine study passes pre-flight without error."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="transformers"),
        ]
    )
    run_study_preflight(study)  # should not raise


def test_single_engine_local_pin_not_import_checked(monkeypatch):
    """Single-engine studies are unaffected by the multi-engine import pre-flight.

    A single-engine local pin passes even if the engine is not importable on the
    host - per-experiment pre-flight runs later in the subprocess.
    """
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    # If the multi-engine import check ran, this would raise. It must not.
    monkeypatch.setattr(
        "llenergymeasure.harness.preflight._check_engine_installed", lambda engine: False
    )
    study = StudyConfig(experiments=[ExperimentConfig(task={"model": "m1"}, engine="vllm")])
    specs, overrides = run_study_preflight(study, yaml_runners={"vllm": "local"})
    assert specs["vllm"].mode == "local"
    assert specs["vllm"].source == "yaml"
    assert overrides == {}


def test_multi_engine_all_auto_without_docker_raises(monkeypatch):
    """Multi-engine all-auto study raises PreFlightError when Docker is unavailable."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    with pytest.raises(PreFlightError, match="Multi-engine"):
        run_study_preflight(study)


def test_multi_engine_error_mentions_docker(monkeypatch):
    """Error message directs user to Docker when an auto engine needs elevating."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    with pytest.raises(PreFlightError, match="Docker"):
        run_study_preflight(study)


def test_multi_engine_error_lists_engines(monkeypatch):
    """Docker-unavailable error names the engines that need elevating."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study)
    assert "transformers" in str(exc_info.value)
    assert "vllm" in str(exc_info.value)


def test_multi_engine_all_auto_elevates_to_docker(monkeypatch):
    """All-auto multi-engine study elevates every engine to Docker (unchanged)."""
    monkeypatch.setattr("llenergymeasure.infra.runner_resolution.is_docker_available", lambda: True)
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", lambda skip=False: None
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    specs, overrides = run_study_preflight(study)  # no explicit runner pins

    assert specs["transformers"].mode == "docker"
    assert specs["transformers"].source == "multi_engine_elevation"
    assert specs["vllm"].mode == "docker"
    assert specs["vllm"].source == "multi_engine_elevation"
    # Both engines recorded as elevated.
    assert overrides["runner.transformers"]["effective"] == "docker"
    assert overrides["runner.vllm"]["effective"] == "docker"
    assert "multi-engine" in overrides["runner.transformers"]["reason"]


def test_multi_engine_explicit_local_kept_auto_elevated(monkeypatch):
    """Explicit local pin is kept; auto-resolved engines are elevated to Docker."""
    monkeypatch.setattr("llenergymeasure.infra.runner_resolution.is_docker_available", lambda: True)
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", lambda skip=False: None
    )
    # transformers is pinned local and importable on the host.
    monkeypatch.setattr(
        "llenergymeasure.harness.preflight._check_engine_installed", lambda engine: True
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    specs, overrides = run_study_preflight(study, yaml_runners={"transformers": "local"})

    # Explicit local pin kept.
    assert specs["transformers"].mode == "local"
    assert specs["transformers"].source == "yaml"
    # Auto-resolved engine elevated.
    assert specs["vllm"].mode == "docker"
    assert specs["vllm"].source == "multi_engine_elevation"
    # Only the elevated engine appears in the overrides record.
    assert "runner.vllm" in overrides
    assert "runner.transformers" not in overrides


def test_multi_engine_explicit_local_missing_package_raises(monkeypatch):
    """Explicit local pin for an engine missing from the host raises a specific error."""
    monkeypatch.setattr("llenergymeasure.infra.runner_resolution.is_docker_available", lambda: True)
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", lambda skip=False: None
    )
    # tensorrt is pinned local but not importable on the host.
    monkeypatch.setattr(
        "llenergymeasure.harness.preflight._check_engine_installed", lambda engine: False
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="tensorrt"),
        ]
    )
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study, yaml_runners={"tensorrt": "local"})

    msg = str(exc_info.value)
    assert "tensorrt" in msg
    assert "tensorrt_llm" in msg  # the missing package, distinct from the engine name
    assert "pip install 'llenergymeasure[tensorrt]'" in msg  # fix 1: install the extra
    assert "drop the explicit" in msg  # fix 2: drop the local pin


def test_multi_engine_all_explicit_local_without_docker_passes(monkeypatch):
    """All-explicit-local multi-engine study passes without Docker."""
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    docker_preflight = MagicMock()
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", docker_preflight
    )
    # Both engines pinned local and importable on the host.
    monkeypatch.setattr(
        "llenergymeasure.harness.preflight._check_engine_installed", lambda engine: True
    )
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm"),
        ]
    )
    specs, overrides = run_study_preflight(
        study, yaml_runners={"transformers": "local", "vllm": "local"}
    )

    assert specs["transformers"].mode == "local"
    assert specs["transformers"].source == "yaml"
    assert specs["vllm"].mode == "local"
    assert specs["vllm"].source == "yaml"
    assert overrides == {}
    # No Docker runner resolved -> Docker pre-flight is never invoked.
    docker_preflight.assert_not_called()


def test_preflight_forwards_runner_context(monkeypatch):
    """run_study_preflight forwards yaml_runners and user_config to resolve_study_runners."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(engines, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        # Return local specs so no Docker preflight is triggered
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        return {b: RunnerSpec(mode="local", image=None, source="default") for b in engines}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    mock_user_config = MagicMock()
    study = StudyConfig(experiments=[ExperimentConfig(task={"model": "m1"}, engine="transformers")])

    run_study_preflight(study, yaml_runners={"transformers": "local"}, user_config=mock_user_config)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] == {"transformers": "local"}
    assert captured_calls[0]["user_config"] is mock_user_config


def test_preflight_defaults_to_auto_detect_without_context(monkeypatch):
    """Calling run_study_preflight without yaml_runners/user_config passes None for both."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(engines, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        from llenergymeasure.infra.runner_resolution import RunnerSpec

        return {b: RunnerSpec(mode="local", image=None, source="default") for b in engines}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    study = StudyConfig(experiments=[ExperimentConfig(task={"model": "m1"}, engine="transformers")])

    run_study_preflight(study)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] is None
    assert captured_calls[0]["user_config"] is None
