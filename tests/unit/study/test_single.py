"""Unit tests for the single-experiment in-process execution seam.

run_single_experiment() is the shared body for single-experiment / n_cycles=1
studies. These tests verify it without GPU hardware (uses monkeypatching).
"""

from __future__ import annotations

from llenergymeasure import ExperimentConfig, ExperimentResult, StudyConfig
from llenergymeasure.study.single import run_single_experiment
from tests.conftest import make_result


class _MockBackend:
    """Minimal EnginePlugin for run_single_experiment() tests.

    Implements the 4-method EnginePlugin protocol. Tests that use this mock
    also patch MeasurementHarness.run to return the pre-built result directly,
    so only load_model/warmup/run_inference/cleanup stubs are needed here.
    """

    def __init__(self, result: ExperimentResult) -> None:
        self._result = result
        self.run_inference_calls: list[ExperimentConfig] = []

    @property
    def name(self) -> str:
        return "transformers"

    def load_model(self, config: ExperimentConfig, **kwargs):
        return object()  # Opaque model object

    def warmup(self, config: ExperimentConfig, model, prompts: list[str] | None = None):
        from llenergymeasure.domain.metrics import WarmupResult

        return WarmupResult(
            converged=True, final_cv=0.0, iterations_completed=0, target_cv=0.01, max_prompts=1
        )

    def run_inference(self, config: ExperimentConfig, model, prompts: list[str] | None = None):
        from llenergymeasure.engines.protocol import InferenceOutput

        self.run_inference_calls.append(config)
        return InferenceOutput(
            elapsed_time_sec=1.0,
            input_tokens=10,
            output_tokens=20,
            peak_memory_mb=0.0,
            model_memory_mb=0.0,
        )

    def cleanup(self, model) -> None:
        pass


def _patch_harness(monkeypatch, result: ExperimentResult) -> None:
    """Patch MeasurementHarness.run to return a pre-built result.

    Used by tests that verify execution wiring (preflight, get_engine) without
    running the actual measurement lifecycle.
    """
    import llenergymeasure.harness as harness_module

    monkeypatch.setattr(
        harness_module.MeasurementHarness, "run", lambda self, engine, config, **kw: result
    )


def test_run_single_experiment_calls_gpu_memory_check(monkeypatch, tmp_path):
    """run_single_experiment() calls check_gpu_memory_residual before running the experiment."""
    import llenergymeasure.engines as engines_module
    import llenergymeasure.harness.preflight as pf_module

    gpu_check_calls: list[int] = []

    def mock_gpu_check(device_index=0, threshold_mb=1024.0):
        gpu_check_calls.append(device_index)

    mock_result = make_result(experiment_id="gpu-check-test")
    mock_engine = _MockBackend(mock_result)

    monkeypatch.setattr(pf_module, "run_preflight", lambda config: None)
    monkeypatch.setattr(engines_module, "get_engine", lambda name: mock_engine)
    _patch_harness(monkeypatch, mock_result)
    monkeypatch.setattr(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual",
        mock_gpu_check,
    )
    monkeypatch.setattr(
        "llenergymeasure.results.persistence.save_result",
        lambda result, output_dir, **kw: tmp_path / "result.json",
    )

    from unittest.mock import MagicMock

    from llenergymeasure.study.manifest import ManifestWriter

    mock_manifest = MagicMock(spec=ManifestWriter)

    config = ExperimentConfig(task={"model": "gpt2"}, engine="transformers")
    study = StudyConfig(experiments=[config])

    run_single_experiment(study, mock_manifest, tmp_path, runner_specs=None)

    assert len(gpu_check_calls) == 1, (
        f"Expected check_gpu_memory_residual to be called once, got {len(gpu_check_calls)}"
    )
