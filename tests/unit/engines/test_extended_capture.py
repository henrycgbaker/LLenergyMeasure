"""Unit tests for per-engine extended-metrics capture helpers.

All host-safe: vLLM/TRT-LLM RequestOutputs are MagicMocks, the transformers
padding math is exercised through the real ``_run_batch`` with mocked tensors.
No GPU or engine library import occurs.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llenergymeasure.engines.vllm.plugin import _capture_kv_cache_stats, _extract_request_stats

# ---------------------------------------------------------------------------
# _extract_request_stats (vLLM V1 RequestStateStats)
# ---------------------------------------------------------------------------


def _req(ttft=None, first_ts=None, last_ts=None, out_lens=None, with_metrics=True):
    """Build a fake vLLM V1 RequestOutput with a ``RequestStateStats``-shaped ``.metrics``.

    ``ttft`` is ``first_token_latency`` (wall-clock seconds); ``first_ts`` /
    ``last_ts`` are engine-core monotonic timestamps; ``out_lens`` is a list of
    per-output token counts (N>1 = parallel sampling streams).
    """
    outputs = [SimpleNamespace(token_ids=list(range(n))) for n in (out_lens or [])]
    if not with_metrics:
        return SimpleNamespace(metrics=None, outputs=outputs)
    metrics = SimpleNamespace(
        first_token_latency=ttft,
        first_token_ts=first_ts,
        last_token_ts=last_ts,
    )
    return SimpleNamespace(metrics=metrics, outputs=outputs)


class TestExtractRequestStats:
    def test_full_metrics(self):
        # TTFT 0.2s; decode interval 100.9-100.0 = 0.9s over 9 intervals -> 100 ms.
        # E2E = (0.2 + 0.9) * 1000 = 1100 ms.
        outputs = [
            _req(ttft=0.2, first_ts=100.0, last_ts=100.9, out_lens=[10]),
            _req(ttft=0.1, first_ts=50.0, last_ts=50.2, out_lens=[3]),  # 0.2/2 = 100 ms
        ]
        lat, ttft, itl = _extract_request_stats(outputs)
        assert ttft == pytest.approx([200.0, 100.0])
        assert lat == pytest.approx([1100.0, 300.0])
        assert itl == pytest.approx([100.0, 100.0])

    def test_metrics_none_skipped(self):
        outputs = [
            _req(with_metrics=False, out_lens=[5]),
            _req(ttft=0.1, first_ts=0.0, last_ts=0.9, out_lens=[10]),
        ]
        lat, ttft, itl = _extract_request_stats(outputs)
        assert ttft == pytest.approx([100.0])
        assert lat == pytest.approx([1000.0])
        assert itl == pytest.approx([100.0])

    def test_ttft_without_timestamps_gives_ttft_only(self):
        # first_token_latency present but no engine-core timestamps: TTFT captured,
        # no decode interval -> no E2E, no ITL (never cross-subtract clocks).
        outputs = [_req(ttft=0.2, first_ts=None, last_ts=None, out_lens=[10])]
        lat, ttft, itl = _extract_request_stats(outputs)
        assert ttft == pytest.approx([200.0])
        assert lat == []
        assert itl == []

    def test_zero_ttft_skipped(self):
        # first_token_latency defaults to 0.0 when never recorded -> not a sample.
        outputs = [_req(ttft=0.0, first_ts=100.0, last_ts=100.9, out_lens=[10])]
        lat, ttft, itl = _extract_request_stats(outputs)
        assert ttft == []
        assert lat == []  # E2E requires a real TTFT
        assert itl == pytest.approx([100.0])  # decode interval still derivable

    def test_single_token_no_itl_but_e2e(self):
        # decode_len == 1 -> no ITL interval, but E2E is still TTFT + decode_s.
        outputs = [_req(ttft=0.2, first_ts=100.0, last_ts=100.0, out_lens=[1])]
        lat, ttft, itl = _extract_request_stats(outputs)
        assert ttft == pytest.approx([200.0])
        assert lat == pytest.approx([200.0])
        assert itl == []

    def test_empty_outputs(self):
        lat, ttft, itl = _extract_request_stats([])
        assert lat == [] and ttft == [] and itl == []


class TestExtractRequestStatsParallelOutputs:
    """n>1 parallel outputs use per-sequence (max) decode length, not the sum."""

    def test_n2_uses_max_not_sum(self):
        # Two parallel outputs of 10 tokens over a 0.9s decode window.
        # Correct: max=10 -> 9 intervals -> 100 ms. Summed (19 intervals) ~47.4 ms.
        outputs = [_req(ttft=0.1, first_ts=1.0, last_ts=1.9, out_lens=[10, 10])]
        _lat, _ttft, itl = _extract_request_stats(outputs)
        assert itl == pytest.approx([100.0])
        assert itl[0] != pytest.approx(0.9 * 1000.0 / (20 - 1))

    def test_uneven_parallel_outputs_use_longest(self):
        # Outputs of length 8 and 5: decode length is the longest (8) -> 7 intervals.
        outputs = [_req(ttft=0.0, first_ts=0.0, last_ts=0.7, out_lens=[8, 5])]
        _lat, _ttft, itl = _extract_request_stats(outputs)
        assert itl == pytest.approx([100.0])

    def test_all_parallel_outputs_single_token_skipped(self):
        # Longest output has 1 token -> decode_len <= 1 -> no ITL interval.
        outputs = [_req(ttft=0.1, first_ts=1.0, last_ts=2.0, out_lens=[1, 1])]
        _lat, _ttft, itl = _extract_request_stats(outputs)
        assert itl == []


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
