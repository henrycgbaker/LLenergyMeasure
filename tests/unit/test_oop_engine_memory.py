"""Tests for out-of-process engine memory sourcing (NVML fallback).

vLLM V1 runs its model in the EngineCore child process and TRT-LLM in its
executor process, so torch's per-process allocator in the driver process reads a
silent 0.0 for peak/model memory. The fix sources those values from NVML
device-used memory whenever the torch reading is implausible (== 0.0), keeps
torch for in-process Transformers, and coerces any residual 0.0 to null at the
domain boundary so the fields are never a silently-wrong zero.

All tests run CPU-only: NVML and torch are mocked.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.harness.lifecycle import capture_model_memory_mb
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MB = 1024 * 1024


def _pynvml_mock(used_by_index: dict[int, int]) -> MagicMock:
    """A pynvml stand-in whose per-index handles report the given used bytes."""
    mock = MagicMock()
    mock.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: f"handle-{idx}"
    mock.nvmlDeviceGetMemoryInfo.side_effect = lambda handle: SimpleNamespace(
        used=used_by_index[int(str(handle).split("-")[1])]
    )
    return mock


def _fake_request_output(n_in: int = 3, n_out: int = 2) -> SimpleNamespace:
    """A minimal RequestOutput accepted by both vLLM and TRT-LLM token counters."""
    return SimpleNamespace(
        prompt_token_ids=list(range(n_in)),
        outputs=[SimpleNamespace(token_ids=list(range(n_out)))],
        metrics=None,  # vLLM _extract_request_stats -> empty
        metrics_dict=None,  # TRT-LLM _extract_metrics_dict -> empty
    )


class _FakeLLM:
    """Stand-in LLM whose generate() returns fake RequestOutputs."""

    def generate(self, prompts, sampling_params):
        return [_fake_request_output() for _ in prompts]

    def beam_search(self, prompts, params):  # pragma: no cover - not hit on host
        return [_fake_request_output() for _ in prompts]


# ---------------------------------------------------------------------------
# get_nvml_device_memory_mb - the shared process-agnostic fallback source
# ---------------------------------------------------------------------------


class TestNvmlDeviceMemoryHelper:
    def test_reads_used_memory_and_converts_to_mb(self):
        from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

        with patch.dict(sys.modules, {"pynvml": _pynvml_mock({0: 2048 * _MB})}):
            assert get_nvml_device_memory_mb([0]) == pytest.approx(2048.0)

    def test_defaults_to_index_zero(self):
        from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

        with patch.dict(sys.modules, {"pynvml": _pynvml_mock({0: 512 * _MB})}):
            assert get_nvml_device_memory_mb() == pytest.approx(512.0)

    def test_max_across_indices(self):
        """Matches the torch convention of peaking across tensor-parallel ranks."""
        from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

        with patch.dict(sys.modules, {"pynvml": _pynvml_mock({0: 1000 * _MB, 1: 3000 * _MB})}):
            assert get_nvml_device_memory_mb([0, 1]) == pytest.approx(3000.0)

    def test_returns_none_when_pynvml_unavailable(self):
        from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

        # sys.modules[name] = None makes `import name` raise ImportError.
        with patch.dict(sys.modules, {"pynvml": None}):
            assert get_nvml_device_memory_mb([0]) is None

    def test_returns_none_on_query_failure(self):
        from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

        broken = MagicMock()
        broken.nvmlDeviceGetHandleByIndex.side_effect = RuntimeError("no such device")
        with patch.dict(sys.modules, {"pynvml": broken}):
            assert get_nvml_device_memory_mb([0]) is None


# ---------------------------------------------------------------------------
# Harness model-memory baseline: torch for in-process, NVML for out-of-process
# ---------------------------------------------------------------------------


class TestHarnessModelMemoryCapture:
    def test_transformers_shaped_torch_value_preserved_no_fallback(self):
        """In-process engine: torch sees the weights (> 0), so NVML is never consulted."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 700 * _MB

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch(
                "llenergymeasure.engines._cuda.get_nvml_device_memory_mb",
                side_effect=AssertionError("NVML must not be consulted when torch reads > 0"),
            ),
        ):
            result = capture_model_memory_mb(gpu_indices=[0])

        assert result == pytest.approx(700.0)

    def test_oop_shaped_falls_back_to_nvml_when_torch_zero(self):
        """Out-of-process engine: torch reads 0.0, so NVML device memory is used."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 0  # child-process weights unseen

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch(
                "llenergymeasure.engines._cuda.get_nvml_device_memory_mb",
                return_value=8192.0,
            ),
        ):
            result = capture_model_memory_mb(gpu_indices=[0])

        assert result == pytest.approx(8192.0)

    def test_zero_when_neither_source_available(self):
        """torch reads 0.0 and NVML is unavailable -> 0.0 (nulled at the domain boundary)."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.max_memory_allocated.return_value = 0

        with (
            patch.dict(sys.modules, {"torch": mock_torch}),
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch(
                "llenergymeasure.engines._cuda.get_nvml_device_memory_mb",
                return_value=None,
            ),
        ):
            result = capture_model_memory_mb(gpu_indices=[0])

        assert result == 0.0


# ---------------------------------------------------------------------------
# Plugin peak-memory capture: vLLM- and TRT-shaped fallback on 0.0
# ---------------------------------------------------------------------------


class TestVllmPeakMemoryFallback:
    def _run(self, nvml_return):
        from llenergymeasure.engines.vllm import VLLMEngine

        config = make_config(engine="vllm", model="test-model")
        with (
            patch("llenergymeasure.engines._cuda.get_cuda_peak_memory_mb", return_value=0.0),
            patch(
                "llenergymeasure.engines._cuda.get_nvml_device_memory_mb",
                return_value=nvml_return,
            ),
        ):
            return VLLMEngine().run_inference(config, (_FakeLLM(), object()), ["hi"])

    def test_fallback_fires_when_torch_zero(self):
        """torch peak 0.0 (V1 out-of-process) -> peak sourced from NVML."""
        out = self._run(nvml_return=12345.0)
        assert out.peak_memory_mb == pytest.approx(12345.0)

    def test_stays_zero_when_nvml_unavailable(self):
        """torch 0.0 and NVML None -> 0.0 (never a fabricated number)."""
        out = self._run(nvml_return=None)
        assert out.peak_memory_mb == 0.0


class TestTensorRTPeakMemoryFallback:
    def test_fallback_fires_when_torch_zero(self):
        from llenergymeasure.engines.tensorrt import TensorRTEngine

        config = make_config(engine="tensorrt", model="test-model")
        with (
            patch("llenergymeasure.engines._cuda.get_cuda_peak_memory_mb", return_value=0.0),
            patch(
                "llenergymeasure.engines._cuda.get_nvml_device_memory_mb",
                return_value=6789.0,
            ),
        ):
            out = TensorRTEngine().run_inference(config, (_FakeLLM(), object()), ["hi"])

        assert out.peak_memory_mb == pytest.approx(6789.0)


class TestTransformersPeakNoFallback:
    def test_torch_value_preserved_no_nvml(self):
        """In-process Transformers: peak is the torch value, NVML never consulted."""
        pytest.importorskip("torch")
        from llenergymeasure.engines.transformers import TransformersEngine

        nvml_spy = MagicMock(return_value=99999.0)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.reset_peak_memory_stats"),
            patch("torch.cuda.max_memory_allocated", return_value=640 * _MB),
            patch("torch.manual_seed"),
            patch("llenergymeasure.engines._cuda.get_nvml_device_memory_mb", nvml_spy),
            patch.object(TransformersEngine, "_run_batch", return_value=(10, 20, 0.5, 0)),
        ):
            config = make_config(engine="transformers", model="test-model")
            out = TransformersEngine().run_inference(config, (object(), None), ["hi"])

        assert out.peak_memory_mb == pytest.approx(640.0)
        nvml_spy.assert_not_called()


# ---------------------------------------------------------------------------
# Cascade fields un-null when a real peak is present
# ---------------------------------------------------------------------------


class TestCascadeUnNulling:
    def test_cascade_populated_when_peak_real(self):
        """A real NVML-sourced peak un-nulls the derived memory metrics that the
        silent-0.0 bug left null for vLLM/TRT."""
        from llenergymeasure.domain.extended_metrics import _compute_memory_metrics

        mem = _compute_memory_metrics(
            2048,
            {"peak_mb": 2048.0, "total_vram_mb": 40960.0, "model_mb": 8192.0, "kv_cache_mb": 512.0},
        )
        assert mem.peak_memory_mb == pytest.approx(2048.0)
        assert mem.model_memory_mb == pytest.approx(8192.0)
        assert mem.tokens_per_gb_vram is not None
        assert mem.model_memory_utilisation == pytest.approx(8192.0 / 40960.0)
        assert mem.kv_cache_memory_ratio == pytest.approx(512.0 / 2048.0)
