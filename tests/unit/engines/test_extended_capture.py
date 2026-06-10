"""Unit tests for per-engine extended-metrics capture helpers.

All host-safe: vLLM/TRT-LLM RequestOutputs are MagicMocks, the transformers
padding math is exercised through the real ``_run_batch`` with mocked tensors.
No GPU or engine library import occurs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenergymeasure.engines._observed import extract_request_metrics
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
