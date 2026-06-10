"""Unit tests for domain/extended_metrics.py - compute_extended_metrics and helpers."""

from __future__ import annotations

import pytest

from llenergymeasure.domain.extended_metrics import (
    _compute_batch_metrics,
    _compute_gpu_utilisation_metrics,
    _compute_kv_cache_metrics,
    _compute_memory_metrics,
    _compute_request_latency_metrics,
    compute_extended_metrics,
)
from llenergymeasure.domain.metrics import ExtendedEfficiencyMetrics

# ---------------------------------------------------------------------------
# TestComputeExtendedMetrics
# ---------------------------------------------------------------------------


class TestComputeExtendedMetrics:
    """Top-level compute_extended_metrics() orchestrator."""

    def test_zero_output_tokens_returns_defaults(self):
        result = compute_extended_metrics(
            output_tokens=0, total_energy_j=10.0, tokens_per_second=50.0
        )
        assert result.tpot_ms is None
        assert result.token_efficiency_index is None

    def test_tpot_from_itl_mean(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=10.0, tokens_per_second=50.0, itl_mean_ms=15.0
        )
        assert result.tpot_ms == pytest.approx(15.0)

    def test_tpot_none_when_no_itl(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=10.0, tokens_per_second=50.0
        )
        assert result.tpot_ms is None

    def test_tei_normal(self):
        result = compute_extended_metrics(
            output_tokens=200, total_energy_j=10.0, tokens_per_second=100.0, precision_factor=0.5
        )
        # TEI = tps * (output_tokens / energy) * precision_factor
        # = 100 * (200 / 10) * 0.5 = 100 * 20 * 0.5 = 1000.0
        assert result.token_efficiency_index == pytest.approx(1000.0)

    def test_tei_none_zero_throughput(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=10.0, tokens_per_second=0.0
        )
        assert result.token_efficiency_index is None

    def test_tei_none_zero_energy(self):
        result = compute_extended_metrics(
            output_tokens=100, total_energy_j=0.0, tokens_per_second=50.0
        )
        assert result.token_efficiency_index is None

    def test_all_optional_none_graceful(self):
        result = compute_extended_metrics(
            output_tokens=100,
            total_energy_j=10.0,
            tokens_per_second=50.0,
            itl_mean_ms=None,
            per_request_latencies_ms=None,
            gpu_utilisation_samples=None,
            memory_stats=None,
            batch_stats=None,
            kv_cache_stats=None,
        )
        assert isinstance(result, ExtendedEfficiencyMetrics)
        assert result.memory.peak_memory_mb == 0.0
        assert result.gpu_utilisation.sm_utilisation_mean is None


# ---------------------------------------------------------------------------
# TestComputeMemoryMetrics
# ---------------------------------------------------------------------------


class TestComputeMemoryMetrics:
    """_compute_memory_metrics() from memory_stats dict."""

    def test_none_memory_stats(self):
        mem = _compute_memory_metrics(100, None)
        assert mem.peak_memory_mb == 0.0
        assert mem.tokens_per_gb_vram is None

    def test_tokens_per_gb_vram(self):
        mem = _compute_memory_metrics(
            1024,
            {"peak_mb": 1024.0, "total_vram_mb": 40960.0, "model_mb": 8000.0},
        )
        # 1024 tokens / (1024 MB / 1024) = 1024 / 1.0 = 1024.0
        assert mem.tokens_per_gb_vram == pytest.approx(1024.0)

    def test_model_memory_utilisation(self):
        mem = _compute_memory_metrics(
            100,
            {"peak_mb": 1024.0, "total_vram_mb": 40960.0, "model_mb": 8192.0},
        )
        assert mem.model_memory_utilisation == pytest.approx(8192.0 / 40960.0)

    def test_kv_cache_memory_ratio(self):
        mem = _compute_memory_metrics(
            100,
            {"peak_mb": 2048.0, "total_vram_mb": 40960.0, "model_mb": 1000.0, "kv_cache_mb": 512.0},
        )
        assert mem.kv_cache_memory_ratio == pytest.approx(512.0 / 2048.0)

    def test_zero_peak_tokens_per_gb_none(self):
        mem = _compute_memory_metrics(
            100, {"peak_mb": 0.0, "total_vram_mb": 40960.0, "model_mb": 0.0}
        )
        assert mem.tokens_per_gb_vram is None

    def test_zero_total_vram_utilisation_none(self):
        mem = _compute_memory_metrics(
            100, {"peak_mb": 1024.0, "total_vram_mb": 0.0, "model_mb": 1000.0}
        )
        assert mem.model_memory_utilisation is None

    def test_kv_cache_mb_none_ratio_none(self):
        mem = _compute_memory_metrics(
            100, {"peak_mb": 1024.0, "total_vram_mb": 40960.0, "model_mb": 1000.0}
        )
        assert mem.kv_cache_memory_ratio is None


# ---------------------------------------------------------------------------
# TestComputeGPUUtilisationMetrics
# ---------------------------------------------------------------------------


class TestComputeGPUUtilisationMetrics:
    """_compute_gpu_utilisation_metrics() from sample list."""

    def test_none_input(self):
        gpu = _compute_gpu_utilisation_metrics(None)
        assert gpu.sm_utilisation_mean is None
        assert gpu.sm_utilisation_samples == 0

    def test_empty_list(self):
        gpu = _compute_gpu_utilisation_metrics([])
        assert gpu.sm_utilisation_mean is None

    def test_normal_samples(self):
        gpu = _compute_gpu_utilisation_metrics([50.0, 60.0, 70.0, 80.0])
        assert gpu.sm_utilisation_mean == pytest.approx(65.0)
        assert gpu.sm_utilisation_samples == 4

    def test_single_sample(self):
        gpu = _compute_gpu_utilisation_metrics([42.0])
        assert gpu.sm_utilisation_mean == pytest.approx(42.0)
        assert gpu.sm_utilisation_samples == 1

    def test_memory_bandwidth_mean(self):
        gpu = _compute_gpu_utilisation_metrics([50.0, 60.0], [30.0, 50.0])
        assert gpu.sm_utilisation_mean == pytest.approx(55.0)
        assert gpu.memory_bandwidth_utilisation == pytest.approx(40.0)

    def test_memory_bandwidth_none_when_absent(self):
        gpu = _compute_gpu_utilisation_metrics([50.0, 60.0], None)
        assert gpu.memory_bandwidth_utilisation is None

    def test_memory_bandwidth_without_sm(self):
        gpu = _compute_gpu_utilisation_metrics(None, [10.0, 20.0])
        assert gpu.sm_utilisation_mean is None
        assert gpu.memory_bandwidth_utilisation == pytest.approx(15.0)


# ---------------------------------------------------------------------------
# TestComputeBatchMetrics
# ---------------------------------------------------------------------------


class TestComputeBatchMetrics:
    """_compute_batch_metrics() from batch_stats dict."""

    def test_none_returns_empty(self):
        batch = _compute_batch_metrics(None)
        assert batch.effective_batch_size is None
        assert batch.batch_utilisation is None

    def test_batch_utilisation(self):
        batch = _compute_batch_metrics(
            {"effective_batch_size": 8.0, "configured_batch_size": 16, "padding_overhead": 0.1}
        )
        assert batch.batch_utilisation == pytest.approx(0.5)
        assert batch.padding_overhead == pytest.approx(0.1)

    def test_zero_configured_utilisation_none(self):
        batch = _compute_batch_metrics({"effective_batch_size": 8.0, "configured_batch_size": 0})
        assert batch.batch_utilisation is None

    def test_num_batches_cast_to_int(self):
        batch = _compute_batch_metrics({"num_batches": 5.0})
        assert batch.num_batches == 5
        assert isinstance(batch.num_batches, int)

    def test_padding_overhead_passthrough(self):
        batch = _compute_batch_metrics({"padding_overhead": 0.25})
        assert batch.padding_overhead == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# TestComputeKVCacheMetrics
# ---------------------------------------------------------------------------


class TestComputeKVCacheMetrics:
    """_compute_kv_cache_metrics() from kv_cache_stats dict."""

    def test_none_returns_empty(self):
        kv = _compute_kv_cache_metrics(None)
        assert kv.kv_cache_hit_rate is None
        assert kv.kv_cache_blocks_used is None

    def test_normal_case(self):
        kv = _compute_kv_cache_metrics({"hit_rate": 0.85, "blocks_used": 128, "blocks_total": 256})
        assert kv.kv_cache_hit_rate == pytest.approx(0.85)
        assert kv.kv_cache_blocks_used == 128
        assert kv.kv_cache_blocks_total == 256


# ---------------------------------------------------------------------------
# TestComputeRequestLatencyMetrics
# ---------------------------------------------------------------------------


class TestComputeRequestLatencyMetrics:
    """_compute_request_latency_metrics() from latency list."""

    def test_none_returns_empty(self):
        req = _compute_request_latency_metrics(None)
        assert req.e2e_latency_mean_ms is None
        assert req.e2e_latency_samples == 0

    def test_empty_list_returns_empty(self):
        req = _compute_request_latency_metrics([])
        assert req.e2e_latency_mean_ms is None

    def test_normal_latencies(self):
        req = _compute_request_latency_metrics([100.0, 200.0, 150.0, 180.0, 120.0])
        assert req.e2e_latency_mean_ms == pytest.approx(150.0)
        assert req.e2e_latency_median_ms == pytest.approx(150.0)
        assert req.e2e_latency_samples == 5

    def test_single_element(self):
        req = _compute_request_latency_metrics([42.0])
        assert req.e2e_latency_mean_ms == pytest.approx(42.0)
        assert req.e2e_latency_median_ms == pytest.approx(42.0)
        assert req.e2e_latency_p95_ms == pytest.approx(42.0)
        assert req.e2e_latency_p99_ms == pytest.approx(42.0)

    def test_ordering_p95_lte_p99(self):
        req = _compute_request_latency_metrics([10.0, 20.0, 30.0, 100.0, 200.0, 500.0])
        assert req.e2e_latency_p95_ms is not None
        assert req.e2e_latency_p99_ms is not None
        assert req.e2e_latency_p95_ms <= req.e2e_latency_p99_ms


# ---------------------------------------------------------------------------
# TestComputeExtendedMetricsIntegrated - full wiring through grouped sub-metrics
# ---------------------------------------------------------------------------


class TestComputeExtendedMetricsIntegrated:
    """compute_extended_metrics() wiring memory/batch/gpu/kv-cache together."""

    def test_full_population(self):
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=20.0,
            tokens_per_second=200.0,
            per_request_latencies_ms=[100.0, 120.0, 110.0],
            gpu_utilisation_samples=[80.0, 90.0],
            memory_bandwidth_samples=[40.0, 60.0],
            memory_stats={
                "peak_mb": 4096.0,
                "model_mb": 8192.0,
                "total_vram_mb": 40960.0,
                "kv_cache_mb": 1024.0,
            },
            batch_stats={
                "num_batches": 4,
                "effective_batch_size": 8.0,
                "configured_batch_size": 16,
                "padding_overhead": 0.2,
            },
            kv_cache_stats={"hit_rate": 0.5, "blocks_used": 100, "blocks_total": 200},
        )
        # Memory ratios
        assert result.memory.model_memory_utilisation == pytest.approx(8192.0 / 40960.0)
        assert result.memory.kv_cache_memory_ratio == pytest.approx(1024.0 / 4096.0)
        assert result.memory.tokens_per_gb_vram == pytest.approx(1000 / (4096.0 / 1024))
        # GPU utilisation incl mem-bw
        assert result.gpu_utilisation.sm_utilisation_mean == pytest.approx(85.0)
        assert result.gpu_utilisation.memory_bandwidth_utilisation == pytest.approx(50.0)
        # Batch
        assert result.batch.num_batches == 4
        assert result.batch.batch_utilisation == pytest.approx(0.5)
        assert result.batch.padding_overhead == pytest.approx(0.2)
        # KV cache
        assert result.kv_cache.kv_cache_hit_rate == pytest.approx(0.5)
        assert result.kv_cache.kv_cache_blocks_used == 100
        # Request latency
        assert result.request_latency.e2e_latency_samples == 3
        assert result.request_latency.e2e_latency_mean_ms == pytest.approx(110.0)
