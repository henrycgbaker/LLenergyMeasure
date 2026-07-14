"""Unit tests for opt-in streaming latency profiling capture.

All host-safe: no GPU, no torch, no engine library import. The transformers
``run_inference`` profiling path is exercised by monkeypatching ``_run_batch``
(which would otherwise need torch tensors) so it drives the streamer's ``put()``
and returns synthetic token counts.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llenergymeasure.domain.metrics import LatencyMeasurementMode
from llenergymeasure.engines.transformers.plugin import TransformersEngine, _TimingStreamer
from tests.conftest import make_config

# ---------------------------------------------------------------------------
# _TimingStreamer
# ---------------------------------------------------------------------------


class TestTimingStreamer:
    def test_first_put_is_prompt_and_dropped(self):
        s = _TimingStreamer()
        s.put("prompt_tensor")  # dropped (input_ids)
        assert s.token_times_ms == []

    def test_records_monotonic_token_times(self):
        s = _TimingStreamer()
        s.put("prompt")  # dropped
        s.put("tok1")
        s.put("tok2")
        s.put("tok3")
        assert len(s.token_times_ms) == 3
        # perf_counter is monotonic non-decreasing
        assert s.token_times_ms == sorted(s.token_times_ms)
        assert all(isinstance(t, float) for t in s.token_times_ms)

    def test_end_is_noop(self):
        s = _TimingStreamer()
        s.put("prompt")
        s.put("tok")
        s.end()
        assert len(s.token_times_ms) == 1


# ---------------------------------------------------------------------------
# Transformers run_inference profiling path (mocked _run_batch)
# ---------------------------------------------------------------------------


def _patch_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the cuda helpers + torch.manual_seed used by run_inference."""
    import llenergymeasure.engines._cuda as cuda_mod

    monkeypatch.setattr(cuda_mod, "reset_cuda_peak_memory", lambda: None)
    monkeypatch.setattr(cuda_mod, "get_cuda_peak_memory_mb", lambda: 0.0)

    # run_inference does `import torch as _torch; _torch.manual_seed(...)`.
    fake_torch = SimpleNamespace(manual_seed=lambda _seed: None)
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)


def _make_run_batch(n_tokens: int, batch_size_seen: list[int]):
    """Return a fake _run_batch that drives the streamer with n_tokens puts."""

    def _fake_run_batch(
        self: Any,
        model: Any,
        tokenizer: Any,
        config: Any,
        batch: list[str],
        generate_kwargs: dict[str, Any],
        streamer: Any | None = None,
    ) -> tuple[int, int, float, int]:
        batch_size_seen.append(len(batch))
        if streamer is not None:
            streamer.put("prompt")  # dropped
            for _ in range(n_tokens):
                streamer.put("tok")
        # (input_tokens, output_tokens, time_sec, padding_tokens)
        return 5, n_tokens, 0.1, 0

    return _fake_run_batch


class TestTransformersProfilingPath:
    def test_profiling_sets_mode_and_captures_itl(self, monkeypatch):
        _patch_cuda(monkeypatch)
        batch_sizes: list[int] = []
        monkeypatch.setattr(TransformersEngine, "_run_batch", _make_run_batch(6, batch_sizes))

        config = make_config(engine="transformers", latency_profiling=True)
        engine = TransformersEngine()
        out = engine.run_inference(config, (object(), object()), ["p1", "p2"])

        assert out.latency_measurement_mode == LatencyMeasurementMode.TRUE_STREAMING.value
        # 6 tokens -> 5 intervals + t0 boundary; trimmed ITL drops first/last.
        assert out.itl_ms  # non-empty
        assert len(out.ttft_ms) == 2  # one TTFT per request
        # batch forced to 1 (default batch unset -> already 1)
        assert all(b == 1 for b in batch_sizes)

    def test_profiling_forces_batch_size_one(self, monkeypatch):
        _patch_cuda(monkeypatch)
        batch_sizes: list[int] = []
        monkeypatch.setattr(TransformersEngine, "_run_batch", _make_run_batch(4, batch_sizes))

        # batch_size=4 should be forced to 1 under profiling.
        config = make_config(
            engine="transformers",
            latency_profiling=True,
            harness={"transformers": {"batch_size": 4}},
        )
        engine = TransformersEngine()
        out = engine.run_inference(config, (object(), object()), ["p1", "p2", "p3"])

        assert all(b == 1 for b in batch_sizes)
        assert out.extras.get("profiling_forced_batch_size") is True

    def test_beam_guard_falls_back_to_non_profiled(self, monkeypatch):
        _patch_cuda(monkeypatch)
        batch_sizes: list[int] = []
        # _run_batch must be called WITHOUT a streamer when falling back.
        seen_streamer: list[bool] = []

        def _fake_run_batch(self, model, tokenizer, config, batch, generate_kwargs, streamer=None):
            batch_sizes.append(len(batch))
            seen_streamer.append(streamer is not None)
            return 5, 4, 0.1, 0

        monkeypatch.setattr(TransformersEngine, "_run_batch", _fake_run_batch)

        config = make_config(
            engine="transformers",
            latency_profiling=True,
            transformers={"engine_params": {"num_beams": 4}},
        )
        engine = TransformersEngine()
        out = engine.run_inference(config, (object(), object()), ["p1", "p2"])

        # Beam guard -> profiling disabled: no streamer, no mode, no ITL/TTFT.
        assert out.latency_measurement_mode is None
        assert out.itl_ms == []
        assert out.ttft_ms == []
        assert not any(seen_streamer)

    def test_profiling_off_is_non_streaming(self, monkeypatch):
        _patch_cuda(monkeypatch)
        batch_sizes: list[int] = []
        monkeypatch.setattr(TransformersEngine, "_run_batch", _make_run_batch(4, batch_sizes))

        config = make_config(engine="transformers", latency_profiling=False)
        engine = TransformersEngine()
        out = engine.run_inference(config, (object(), object()), ["p1"])

        assert out.latency_measurement_mode is None
        assert out.itl_ms == []
        assert out.ttft_ms == []
        assert "profiling_forced_batch_size" not in out.extras


# ---------------------------------------------------------------------------
# vLLM mode-setting (logic exercised via extract_decode_itl + mode rules)
# ---------------------------------------------------------------------------


def _vllm_req(first_token, finished, n_out):
    metrics = SimpleNamespace(
        arrival_time=0.0,
        finished_time=finished,
        first_token_time=first_token,
    )
    return SimpleNamespace(
        metrics=metrics,
        outputs=[SimpleNamespace(token_ids=list(range(n_out)))],
    )


class TestVllmModeSelection:
    """Replicates the vLLM plugin's mode-selection branch (host-safe)."""

    @staticmethod
    def _select_mode(profiling: bool, outputs: list) -> tuple[list[float], str | None]:
        from llenergymeasure.engines._observed import (
            extract_decode_itl,
            extract_request_metrics,
        )

        _lat, ttft = extract_request_metrics(outputs)
        itl: list[float] = []
        mode: str | None = None
        if profiling:
            itl = extract_decode_itl(outputs)
            if ttft:
                mode = LatencyMeasurementMode.PROPORTIONAL_ESTIMATE.value
        elif ttft:
            mode = LatencyMeasurementMode.TRUE_STREAMING.value
        return itl, mode

    def test_profiling_on_is_proportional(self):
        outputs = [_vllm_req(first_token=0.1, finished=1.0, n_out=10)]
        itl, mode = self._select_mode(True, outputs)
        assert itl  # decode ITL captured
        assert mode == LatencyMeasurementMode.PROPORTIONAL_ESTIMATE.value

    def test_profiling_off_with_ttft_is_true_streaming(self):
        outputs = [_vllm_req(first_token=0.1, finished=1.0, n_out=10)]
        itl, mode = self._select_mode(False, outputs)
        assert itl == []  # no ITL without profiling
        assert mode == LatencyMeasurementMode.TRUE_STREAMING.value

    def test_no_ttft_no_mode(self):
        outputs = [SimpleNamespace(metrics=None, outputs=[])]
        itl, mode = self._select_mode(True, outputs)
        assert itl == []
        assert mode is None


# ---------------------------------------------------------------------------
# TensorRT metrics_dict extraction + return_perf_metrics opt-in
# ---------------------------------------------------------------------------


class _FakeMetricName:
    """Mimics TRT-LLM's MetricNames enum members (str(key.value) is the name)."""

    def __init__(self, value: str) -> None:
        self.value = value


def _trt_req(ttft: float | None, e2e: float | None, tpot: float | None) -> SimpleNamespace:
    """A fake TRT-LLM RequestOutput carrying a metrics_dict (values in seconds)."""
    md: dict[Any, Any] = {"finished_reason": "length"}
    if ttft is not None:
        md[_FakeMetricName("ttft")] = ttft
    if e2e is not None:
        md[_FakeMetricName("e2e")] = e2e
    if tpot is not None:
        md[_FakeMetricName("tpot")] = tpot
    return SimpleNamespace(metrics_dict=md, outputs=[])


class TestTensorrtMetricsDictExtraction:
    """_extract_metrics_dict: the TRT-LLM 1.x per-request metrics surface."""

    def test_extracts_seconds_as_ms(self):
        from llenergymeasure.engines.tensorrt.plugin import _extract_metrics_dict

        outputs = [_trt_req(ttft=0.05, e2e=0.25, tpot=0.01)]
        lat, ttft, itl = _extract_metrics_dict(outputs)
        assert lat == pytest.approx([250.0])
        assert ttft == pytest.approx([50.0])
        assert itl == pytest.approx([10.0])

    def test_empty_metrics_dict_contributes_nothing(self):
        from llenergymeasure.engines.tensorrt.plugin import _extract_metrics_dict

        outputs = [SimpleNamespace(metrics_dict={}, outputs=[]), SimpleNamespace(outputs=[])]
        lat, ttft, itl = _extract_metrics_dict(outputs)
        assert lat == [] and ttft == [] and itl == []

    def test_partial_keys_are_independent(self):
        from llenergymeasure.engines.tensorrt.plugin import _extract_metrics_dict

        outputs = [_trt_req(ttft=0.1, e2e=None, tpot=None)]
        lat, ttft, itl = _extract_metrics_dict(outputs)
        assert lat == [] and itl == []
        assert ttft == pytest.approx([100.0])

    def test_plain_string_keys_also_accepted(self):
        from llenergymeasure.engines.tensorrt.plugin import _extract_metrics_dict

        outputs = [SimpleNamespace(metrics_dict={"ttft": 0.2, "e2e": 0.4}, outputs=[])]
        lat, ttft, _ = _extract_metrics_dict(outputs)
        assert ttft == pytest.approx([200.0])
        assert lat == pytest.approx([400.0])


class TestTensorrtReturnPerfMetricsOptIn:
    """return_perf_metrics rides SamplingParams kwargs only under latency_profiling."""

    def test_profiling_on_sets_flag(self):
        from llenergymeasure.engines.tensorrt.plugin import TensorRTEngine

        config = make_config(engine="tensorrt", latency_profiling=True)
        kwargs = TensorRTEngine()._build_sampling_kwargs(config)
        assert kwargs.get("return_perf_metrics") is True

    def test_profiling_off_does_not_set_flag(self):
        from llenergymeasure.engines.tensorrt.plugin import TensorRTEngine

        config = make_config(engine="tensorrt", latency_profiling=False)
        kwargs = TensorRTEngine()._build_sampling_kwargs(config)
        assert "return_perf_metrics" not in kwargs
