"""Unit tests for per-engine extended-metrics capture helpers.

All host-safe: vLLM/TRT-LLM RequestOutputs are MagicMocks, the transformers
padding math is exercised through the real ``_run_batch`` with mocked tensors.
No GPU or engine library import occurs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenergymeasure.engines._observed import extract_decode_itl, extract_request_metrics
from llenergymeasure.engines.vllm.plugin import _capture_kv_cache_stats

# ---------------------------------------------------------------------------
# extract_request_metrics
# ---------------------------------------------------------------------------


def _req(arrival=None, finished=None, first_token=None, with_metrics=True):
    """Build a fake RequestOutput with a .metrics namespace (or metrics=None)."""
    if not with_metrics:
        return SimpleNamespace(metrics=None)
    metrics = SimpleNamespace(
        arrival_time=arrival,
        finished_time=finished,
        first_token_time=first_token,
    )
    return SimpleNamespace(metrics=metrics)


class TestExtractRequestMetrics:
    def test_full_metrics(self):
        outputs = [
            _req(arrival=1.0, finished=2.0, first_token=1.2),
            _req(arrival=10.0, finished=10.5, first_token=10.1),
        ]
        latencies, ttft = extract_request_metrics(outputs)
        assert latencies == pytest.approx([1000.0, 500.0])
        assert ttft == pytest.approx([200.0, 100.0])

    def test_metrics_none_skipped(self):
        outputs = [_req(with_metrics=False), _req(arrival=0.0, finished=1.0, first_token=0.1)]
        latencies, ttft = extract_request_metrics(outputs)
        assert latencies == pytest.approx([1000.0])
        assert ttft == pytest.approx([100.0])

    def test_partial_attrs(self):
        # arrival+finished present but no first_token -> latency captured, ttft skipped
        outputs = [_req(arrival=0.0, finished=2.0, first_token=None)]
        latencies, ttft = extract_request_metrics(outputs)
        assert latencies == pytest.approx([2000.0])
        assert ttft == []

    def test_empty_outputs(self):
        latencies, ttft = extract_request_metrics([])
        assert latencies == []
        assert ttft == []


# ---------------------------------------------------------------------------
# extract_decode_itl
# ---------------------------------------------------------------------------


def _req_decode(first_token=None, finished=None, n_out=0, with_metrics=True):
    """Build a fake RequestOutput for extract_decode_itl.

    ``n_out`` controls the total generated-token count across outputs (a single
    CompletionOutput whose ``token_ids`` has length ``n_out``).
    """
    if not with_metrics:
        return SimpleNamespace(metrics=None, outputs=[SimpleNamespace(token_ids=[0] * n_out)])
    metrics = SimpleNamespace(first_token_time=first_token, finished_time=finished)
    return SimpleNamespace(
        metrics=metrics,
        outputs=[SimpleNamespace(token_ids=list(range(n_out)))],
    )


class TestExtractDecodeItl:
    def test_decode_average_math(self):
        # finished-first_token = 0.9s over n_out-1 = 9 intervals -> 100 ms each.
        outputs = [_req_decode(first_token=1.0, finished=1.9, n_out=10)]
        itl = extract_decode_itl(outputs)
        assert itl == pytest.approx([100.0])

    def test_multi_request(self):
        outputs = [
            _req_decode(first_token=1.0, finished=1.9, n_out=10),  # 100 ms
            _req_decode(first_token=0.0, finished=0.2, n_out=3),  # 0.2s / 2 = 100 ms
        ]
        itl = extract_decode_itl(outputs)
        assert itl == pytest.approx([100.0, 100.0])

    def test_n_out_one_skipped(self):
        outputs = [_req_decode(first_token=1.0, finished=2.0, n_out=1)]
        assert extract_decode_itl(outputs) == []

    def test_n_out_zero_skipped(self):
        outputs = [_req_decode(first_token=1.0, finished=2.0, n_out=0)]
        assert extract_decode_itl(outputs) == []

    def test_missing_first_token_skipped(self):
        outputs = [_req_decode(first_token=None, finished=2.0, n_out=5)]
        assert extract_decode_itl(outputs) == []

    def test_missing_finished_skipped(self):
        outputs = [_req_decode(first_token=1.0, finished=None, n_out=5)]
        assert extract_decode_itl(outputs) == []

    def test_metrics_none_skipped(self):
        outputs = [_req_decode(with_metrics=False, n_out=5)]
        assert extract_decode_itl(outputs) == []

    def test_empty_outputs(self):
        assert extract_decode_itl([]) == []


def _req_decode_multi(first_token, finished, out_lens):
    """RequestOutput with N parallel CompletionOutputs (n>1 sampling).

    ``out_lens`` is a list of per-output token counts. The N completions stream
    concurrently and share the single ``(finished - first_token)`` wall-time.
    """
    metrics = SimpleNamespace(first_token_time=first_token, finished_time=finished)
    outputs = [SimpleNamespace(token_ids=list(range(n))) for n in out_lens]
    return SimpleNamespace(metrics=metrics, outputs=outputs)


class TestExtractDecodeItlParallelOutputs:
    """EN5: n>1 parallel outputs use per-sequence (max) decode length, not sum."""

    def test_n2_uses_max_not_sum(self):
        # Two parallel outputs of 10 tokens each over a 0.9s decode window.
        # Correct: max=10 -> 9 intervals -> 100 ms.
        # Buggy (sum): 20 tokens -> 19 intervals -> ~47.4 ms (understated ~1/N).
        outputs = [_req_decode_multi(first_token=1.0, finished=1.9, out_lens=[10, 10])]
        itl = extract_decode_itl(outputs)
        assert itl == pytest.approx([100.0])
        # Prove it is NOT the summed-denominator value.
        summed = 0.9 * 1000.0 / (20 - 1)
        assert itl[0] != pytest.approx(summed)

    def test_uneven_parallel_outputs_use_longest(self):
        # Outputs of length 8 and 5: decode length is the longest (8) -> 7 intervals.
        # 0.7s / 7 = 100 ms.
        outputs = [_req_decode_multi(first_token=0.0, finished=0.7, out_lens=[8, 5])]
        itl = extract_decode_itl(outputs)
        assert itl == pytest.approx([100.0])

    def test_n1_via_multi_helper_matches_single(self):
        # A single parallel output behaves exactly as the n=1 path: max == its len.
        outputs = [_req_decode_multi(first_token=1.0, finished=1.9, out_lens=[10])]
        itl = extract_decode_itl(outputs)
        assert itl == pytest.approx([100.0])

    def test_all_parallel_outputs_single_token_skipped(self):
        # Longest output has 1 token -> decode_len <= 1 -> skipped (no intervals).
        outputs = [_req_decode_multi(first_token=1.0, finished=2.0, out_lens=[1, 1])]
        assert extract_decode_itl(outputs) == []


# ---------------------------------------------------------------------------
# _capture_kv_cache_stats
# ---------------------------------------------------------------------------


def _make_llm(num_gpu_blocks=None, usage=None, hit_rate=None, block_size=None):
    """Build a MagicMock vLLM with controllable cache_config + stat_loggers."""
    llm = MagicMock()
    engine = llm.llm_engine
    if num_gpu_blocks is None and block_size is None:
        engine.cache_config = None
    else:
        engine.cache_config = SimpleNamespace(
            num_gpu_blocks=num_gpu_blocks,
            block_size=block_size,
        )
    if usage is None and hit_rate is None:
        engine.stat_loggers = None
    else:
        last = SimpleNamespace(
            gpu_cache_usage_sys=usage,
            gpu_prefix_cache_hit_rate=hit_rate,
        )
        engine.stat_loggers = {"prometheus": SimpleNamespace(last_local_log=last)}
    return llm


class TestCaptureKVCacheStats:
    def test_fully_readable(self):
        llm = _make_llm(num_gpu_blocks=200, usage=0.5, hit_rate=0.8, block_size=16)
        stats = _capture_kv_cache_stats(llm)
        assert stats is not None
        assert stats["blocks_total"] == 200
        assert stats["usage"] == pytest.approx(0.5)
        assert stats["hit_rate"] == pytest.approx(0.8)
        assert stats["blocks_used"] == 100

    def test_partially_readable_blocks_only(self):
        # cache_config present (blocks_total) but no stat loggers -> usage/hit_rate absent
        llm = _make_llm(num_gpu_blocks=128)
        stats = _capture_kv_cache_stats(llm)
        assert stats is not None
        assert stats["blocks_total"] == 128
        assert "usage" not in stats
        assert "hit_rate" not in stats
        assert "blocks_used" not in stats

    def test_unreadable_returns_none(self):
        llm = _make_llm()  # cache_config None, stat_loggers None
        assert _capture_kv_cache_stats(llm) is None

    def test_no_engine_returns_none(self):
        llm = MagicMock()
        llm.llm_engine = None
        assert _capture_kv_cache_stats(llm) is None
