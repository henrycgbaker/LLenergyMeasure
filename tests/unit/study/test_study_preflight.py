"""Tests for study-level pre-flight checks (CM-10, DOCK-05).

Multi-engine Docker elevation is precedence-based: engines the user explicitly
pinned (env / YAML / user config) keep their runner, while engines whose runner
resolved from auto-detection or the default are elevated to Docker for
isolation. Engines pinned to process are checked for host importability; Docker is
only required when an auto-resolved engine actually needs elevating.
"""

import logging
from unittest.mock import MagicMock

import pytest

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.study.preflight import run_study_preflight
from llenergymeasure.utils.exceptions import PreFlightError

_ALL_LOCAL_CAUTION = "running every engine as a host process"


@pytest.fixture
def two_engine_study() -> StudyConfig:
    """A minimal multi-engine study (transformers + vllm)."""
    return StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers", serving_mode="offline"),
            ExperimentConfig(task={"model": "m2"}, engine="vllm", serving_mode="offline"),
        ]
    )


def patch_env(monkeypatch, *, docker: bool, importable: bool = True) -> MagicMock:
    """Patch the three preflight collaborators and return the docker-preflight mock.

    Args:
        docker: value returned by ``is_docker_available``.
        importable: value returned by the reused host-availability check
            (``harness.preflight.check_engine_installed``).

    Returns:
        The ``run_docker_preflight`` MagicMock, so a caller can assert whether it
        was invoked.
    """
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: docker
    )
    docker_preflight = MagicMock()
    monkeypatch.setattr(
        "llenergymeasure.infra.docker_preflight.run_docker_preflight", docker_preflight
    )
    monkeypatch.setattr(
        "llenergymeasure.harness.preflight.check_engine_installed", lambda engine: importable
    )
    return docker_preflight


def test_single_engine_passes(monkeypatch):
    """Single-engine study passes pre-flight without error."""
    patch_env(monkeypatch, docker=False)
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers", serving_mode="offline"),
            ExperimentConfig(task={"model": "m2"}, engine="transformers", serving_mode="offline"),
        ]
    )
    run_study_preflight(study)  # should not raise


def test_single_engine_local_pin_not_import_checked(monkeypatch):
    """Single-engine studies are unaffected by the multi-engine import pre-flight.

    A single-engine process pin passes even if the engine is not importable on the
    host - per-experiment pre-flight runs later in the subprocess.
    """
    # importable=False would trip the multi-engine import check; single engine must not run it.
    patch_env(monkeypatch, docker=False, importable=False)
    study = StudyConfig(
        experiments=[ExperimentConfig(task={"model": "m1"}, engine="vllm", serving_mode="offline")]
    )
    specs, overrides = run_study_preflight(study, yaml_runners={"vllm": "process"})
    assert specs["vllm"].mode == "process"
    assert specs["vllm"].source == "yaml"
    assert overrides == {}


def test_multi_engine_all_auto_without_docker_raises(monkeypatch, two_engine_study):
    """Multi-engine all-auto study raises PreFlightError when Docker is unavailable."""
    patch_env(monkeypatch, docker=False)
    with pytest.raises(PreFlightError, match="Multi-engine"):
        run_study_preflight(two_engine_study)


def test_multi_engine_error_mentions_docker(monkeypatch, two_engine_study):
    """Error message directs user to Docker when an auto engine needs elevating."""
    patch_env(monkeypatch, docker=False)
    with pytest.raises(PreFlightError, match="Docker"):
        run_study_preflight(two_engine_study)


def test_multi_engine_error_lists_engines(monkeypatch, two_engine_study):
    """Docker-unavailable error names the engines that need elevating."""
    patch_env(monkeypatch, docker=False)
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(two_engine_study)
    assert "transformers" in str(exc_info.value)
    assert "vllm" in str(exc_info.value)


def test_multi_engine_all_auto_elevates_to_docker(monkeypatch, two_engine_study):
    """All-auto multi-engine study elevates every engine to Docker (unchanged)."""
    patch_env(monkeypatch, docker=True)
    specs, overrides = run_study_preflight(two_engine_study)  # no explicit runner pins

    assert specs["transformers"].mode == "container"
    assert specs["transformers"].source == "multi_engine_elevation"
    assert specs["vllm"].mode == "container"
    assert specs["vllm"].source == "multi_engine_elevation"
    # Both engines recorded as elevated.
    assert overrides["runner.transformers"]["effective"] == "container"
    assert overrides["runner.vllm"]["effective"] == "container"
    assert "multi-engine" in overrides["runner.transformers"]["reason"]


def test_multi_engine_explicit_local_kept_auto_elevated(monkeypatch, two_engine_study):
    """Explicit process pin is kept; auto-resolved engines are elevated to Docker."""
    # transformers is pinned process and importable on the host.
    patch_env(monkeypatch, docker=True, importable=True)
    specs, overrides = run_study_preflight(
        two_engine_study, yaml_runners={"transformers": "process"}
    )

    # Explicit process pin kept.
    assert specs["transformers"].mode == "process"
    assert specs["transformers"].source == "yaml"
    # Auto-resolved engine elevated.
    assert specs["vllm"].mode == "container"
    assert specs["vllm"].source == "multi_engine_elevation"
    # Only the elevated engine appears in the overrides record.
    assert "runner.vllm" in overrides
    assert "runner.transformers" not in overrides


def test_multi_engine_explicit_process_missing_package_raises(monkeypatch):
    """Explicit process pin for an engine missing from the host raises a specific error."""
    # tensorrt is pinned process but not importable on the host.
    patch_env(monkeypatch, docker=True, importable=False)
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers", serving_mode="offline"),
            ExperimentConfig(task={"model": "m2"}, engine="tensorrt", serving_mode="offline"),
        ]
    )
    with pytest.raises(PreFlightError) as exc_info:
        run_study_preflight(study, yaml_runners={"tensorrt": "process"})

    msg = str(exc_info.value)
    assert "tensorrt" in msg
    assert "tensorrt_llm" in msg  # the missing package, distinct from the engine name
    assert "pip install 'llenergymeasure[tensorrt]'" in msg  # fix 1: install the extra
    # fix 2: the hint names the CANONICAL mode ("process"), never the pre-v0.7 "local".
    assert "drop the explicit 'tensorrt: process' runner pin" in msg
    assert "local" not in msg


def test_multi_engine_all_explicit_local_without_docker_passes(monkeypatch, two_engine_study):
    """All-explicit-process multi-engine study passes without Docker."""
    # Both engines pinned process and importable on the host.
    docker_preflight = patch_env(monkeypatch, docker=False, importable=True)
    specs, overrides = run_study_preflight(
        two_engine_study, yaml_runners={"transformers": "process", "vllm": "process"}
    )

    assert specs["transformers"].mode == "process"
    assert specs["transformers"].source == "yaml"
    assert specs["vllm"].mode == "process"
    assert specs["vllm"].source == "yaml"
    assert overrides == {}
    # No Docker runner resolved -> Docker pre-flight is never invoked.
    docker_preflight.assert_not_called()


def test_multi_engine_all_local_caution_fires_once(monkeypatch, two_engine_study, caplog):
    """An all-explicit-process multi-engine study warns once about lost isolation."""
    patch_env(monkeypatch, docker=False, importable=True)
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.preflight"):
        run_study_preflight(
            two_engine_study, yaml_runners={"transformers": "process", "vllm": "process"}
        )
    cautions = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and _ALL_LOCAL_CAUTION in r.message
    ]
    assert len(cautions) == 1


def test_single_engine_no_all_local_caution(monkeypatch, caplog):
    """The all-process caution is a multi-engine concern - single-engine must not fire it."""
    patch_env(monkeypatch, docker=False, importable=True)
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers", serving_mode="offline")
        ]
    )
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.preflight"):
        run_study_preflight(study, yaml_runners={"transformers": "process"})
    assert not [r for r in caplog.records if _ALL_LOCAL_CAUTION in r.message]


def test_multi_engine_mixed_no_all_local_caution(monkeypatch, two_engine_study, caplog):
    """A mixed process+elevated study is not all-process, so the caution must not fire."""
    patch_env(monkeypatch, docker=True, importable=True)
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.study.preflight"):
        run_study_preflight(two_engine_study, yaml_runners={"transformers": "process"})
    assert not [r for r in caplog.records if _ALL_LOCAL_CAUTION in r.message]


def test_preflight_forwards_runner_context(monkeypatch):
    """run_study_preflight forwards yaml_runners and user_config to resolve_study_runners."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(engines, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        # Return process specs so no Docker preflight is triggered
        from llenergymeasure.config.runner_spec import RunnerSpec

        return {b: RunnerSpec(mode="process", image=None, source="default") for b in engines}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    mock_user_config = MagicMock()
    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers", serving_mode="offline")
        ]
    )

    run_study_preflight(
        study, yaml_runners={"transformers": "process"}, user_config=mock_user_config
    )

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] == {"transformers": "process"}
    assert captured_calls[0]["user_config"] is mock_user_config


def test_preflight_defaults_to_auto_detect_without_context(monkeypatch):
    """Calling run_study_preflight without yaml_runners/user_config passes None for both."""
    captured_calls: list[dict] = []

    def mock_resolve_study_runners(engines, yaml_runners=None, user_config=None):
        captured_calls.append({"yaml_runners": yaml_runners, "user_config": user_config})
        from llenergymeasure.config.runner_spec import RunnerSpec

        return {b: RunnerSpec(mode="process", image=None, source="default") for b in engines}

    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.is_docker_available", lambda: False
    )
    monkeypatch.setattr(
        "llenergymeasure.infra.runner_resolution.resolve_study_runners",
        mock_resolve_study_runners,
    )

    study = StudyConfig(
        experiments=[
            ExperimentConfig(task={"model": "m1"}, engine="transformers", serving_mode="offline")
        ]
    )

    run_study_preflight(study)

    assert len(captured_calls) == 1
    assert captured_calls[0]["yaml_runners"] is None
    assert captured_calls[0]["user_config"] is None
