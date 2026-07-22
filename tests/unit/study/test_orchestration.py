"""Tests for study orchestration seams.

Covers the GPU-selector conflict warning choke point in
``_resolve_runner_specs``: precedence resolution (env>config) is silent, so the
orchestrator holds the single loud surface. The warning must fire exactly once
per dispatch when a Docker container will actually launch, and never for an
all-local study (GPU scoping only affects containers).
"""

from __future__ import annotations

from unittest.mock import patch

from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.config.ssot import RUNNER_DOCKER, RUNNER_LOCAL
from llenergymeasure.study.orchestration import _resolve_runner_specs

_WARN_TARGET = "llenergymeasure.utils.env_config.warn_on_gpu_selector_conflict"


def _study(gpu_indices: list[int] | None) -> StudyConfig:
    """A minimal study whose only relevant field is study_execution.gpu_indices."""
    return StudyConfig(
        experiments=[ExperimentConfig(task={"model": "m1"}, engine="vllm")],
        study_execution={"gpu_indices": gpu_indices},
    )


def _resolve(study: StudyConfig, runner_specs: dict[str, RunnerSpec]):
    """Drive the choke point with a preresolved plan (skips preflight entirely)."""
    return _resolve_runner_specs(
        study,
        user_config=None,
        preresolved=(runner_specs, {}),
        skip_preflight=True,
        progress=None,
    )


def test_gpu_selector_warn_fires_once_when_docker_present(monkeypatch):
    """One Docker runner + both selectors set -> warn called exactly once, with the config indices."""
    monkeypatch.setenv("LLEM_DOCKER_GPUS", "0")
    study = _study([0])
    specs = {"vllm": RunnerSpec(mode=RUNNER_DOCKER, image="img", source="yaml")}

    with patch(_WARN_TARGET) as mock_warn:
        _resolve(study, specs)

    mock_warn.assert_called_once_with([0])


def test_gpu_selector_warn_never_when_all_local(monkeypatch):
    """All-local study never triggers the warning, even with both selectors set."""
    monkeypatch.setenv("LLEM_DOCKER_GPUS", "0")
    study = _study([0])
    specs = {
        "vllm": RunnerSpec(mode=RUNNER_LOCAL, image=None, source="default"),
        "transformers": RunnerSpec(mode=RUNNER_LOCAL, image=None, source="default"),
    }

    with patch(_WARN_TARGET) as mock_warn:
        _resolve(study, specs)

    mock_warn.assert_not_called()


def test_gpu_selector_warn_not_duplicated_across_docker_runners(monkeypatch):
    """Multiple Docker runners still warn once (choke point is per-dispatch, not per-runner)."""
    monkeypatch.setenv("LLEM_DOCKER_GPUS", "0")
    study = _study([0])
    specs = {
        "vllm": RunnerSpec(mode=RUNNER_DOCKER, image="img-a", source="yaml"),
        "transformers": RunnerSpec(mode=RUNNER_DOCKER, image="img-b", source="yaml"),
    }

    with patch(_WARN_TARGET) as mock_warn:
        _resolve(study, specs)

    mock_warn.assert_called_once_with([0])
