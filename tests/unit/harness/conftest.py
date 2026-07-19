"""Shared fixtures and factories for harness/ tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

from llenergymeasure.domain.metrics import WarmupResult
from llenergymeasure.engines.protocol import InferenceOutput
from tests.conftest import TEST_POWER_MW


@dataclass
class FakeBackend:
    """Minimal EnginePlugin for testing MeasurementHarness lifecycle.

    All methods record calls into ``call_log`` (constructor-overridable so a test
    can thread a shared recorder through it) for order assertions.
    """

    engine_name: str = "fake"
    call_log: list[str] = field(default_factory=list)
    inference_output: InferenceOutput | None = None
    fail_on_run_inference: bool = False

    @property
    def name(self) -> str:
        return self.engine_name

    def load_model(self, config: Any, **kwargs: Any) -> dict:
        self.call_log.append("load_model")
        return {"model": "fake_model_object"}

    def warmup(self, config: Any, model: Any, prompts: list[str]) -> WarmupResult:
        self.call_log.append("warmup")
        return WarmupResult(
            converged=True,
            final_cv=0.0,
            iterations_completed=1,
            target_cv=0.01,
            max_prompts=10,
        )

    def run_warmup_prompt(self, config: Any, model: Any, prompt: str) -> float:
        self.call_log.append("run_warmup_prompt")
        # Return 0.0 = simple kernel warmup (no CV convergence loop needed in tests)
        return 0.0

    def run_inference(self, config: Any, model: Any, prompts: list[str]) -> InferenceOutput:
        self.call_log.append("run_inference")
        if self.fail_on_run_inference:
            raise RuntimeError("Fake inference failure")
        if self.inference_output is not None:
            return self.inference_output
        return InferenceOutput(
            elapsed_time_sec=1.0,
            input_tokens=10,
            output_tokens=20,
            peak_memory_mb=512.0,
            model_memory_mb=256.0,
            batch_times=[1.0],
        )

    def cleanup(self, model: Any) -> None:
        self.call_log.append("cleanup")


class FakeBackendWithCapture(FakeBackend):
    """FakeBackend that also records + returns observed params (post-window capture)."""

    def capture_observed_params(self, config: Any, model: Any, output: Any) -> dict:
        self.call_log.append("capture_observed_params")
        return {"engine": {}, "sampling": {}, "library_version": "test"}


def make_pynvml_mock(
    *,
    power_mw: int = TEST_POWER_MW,
    power_mw_values: list[int] | None = None,
) -> MagicMock:
    """Build a minimal pynvml mock for baseline/power tests.

    The thermal-throttle variant in test_power_thermal.py and the
    memory variant in test_gpu_memory.py remain local (different shape).

    Args:
        power_mw: Constant return value for nvmlDeviceGetPowerUsage.
        power_mw_values: Side-effect list (overrides power_mw).
    """
    mock = MagicMock()
    mock.NVMLError = Exception

    if power_mw_values is not None:
        mock.nvmlDeviceGetPowerUsage.side_effect = power_mw_values
    else:
        mock.nvmlDeviceGetPowerUsage.return_value = power_mw

    return mock
