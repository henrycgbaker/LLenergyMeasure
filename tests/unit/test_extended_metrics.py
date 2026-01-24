"""Unit tests for extended efficiency metrics computation and aggregation."""

from __future__ import annotations

import pytest

from llenergymeasure.core.extended_metrics import (
    aggregate_extended_metrics,
    compute_extended_metrics,
)
from llenergymeasure.domain.metrics import ExtendedEfficiencyMetrics


class TestComputeExtendedMetrics:
    """Tests for compute_extended_metrics function."""

    def test_basic_computation(self) -> None:
        """Test basic metric computation with valid inputs."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
        )

        assert isinstance(result, ExtendedEfficiencyMetrics)
        # TEI = throughput * tokens_per_joule * precision_factor
        # tokens_per_joule = 1000 / 100 = 10
        # TEI = 50 * 10 * 1.0 = 500
        assert result.token_efficiency_index == pytest.approx(500.0)
        assert result.tpot_ms is None  # No ITL data provided

    def test_tpot_from_itl(self) -> None:
        """Test TPOT computation from ITL mean."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            itl_mean_ms=20.0,
        )

        assert result.tpot_ms == pytest.approx(20.0)

    def test_precision_factor(self) -> None:
        """Test precision factor affects TEI."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            precision_factor=2.0,  # Higher precision = more efficient
        )

        # TEI = 50 * 10 * 2.0 = 1000
        assert result.token_efficiency_index == pytest.approx(1000.0)

    def test_memory_metrics(self) -> None:
        """Test memory efficiency metrics computation."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            memory_stats={
                "peak_mb": 8000.0,
                "total_vram_mb": 16000.0,
                "model_mb": 4000.0,
            },
        )

        # tokens_per_gb_vram = 1000 / (8000 / 1024) ≈ 128
        assert result.memory.tokens_per_gb_vram is not None
        assert result.memory.tokens_per_gb_vram > 0
        # model_memory_utilisation = 4000 / 16000 = 0.25
        assert result.memory.model_memory_utilisation == pytest.approx(0.25)

    def test_request_latency_metrics(self) -> None:
        """Test request latency statistics computation."""
        latencies = [100.0, 110.0, 120.0, 130.0, 140.0]
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            per_request_latencies_ms=latencies,
        )

        assert result.request_latency.e2e_latency_mean_ms == pytest.approx(120.0)
        assert result.request_latency.e2e_latency_samples == 5

    def test_gpu_utilisation_metrics(self) -> None:
        """Test GPU utilisation aggregation."""
        samples = [50.0, 60.0, 70.0, 80.0, 90.0]
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            gpu_utilisation_samples=samples,
        )

        assert result.gpu_utilisation.sm_utilisation_mean == pytest.approx(70.0)
        assert result.gpu_utilisation.sm_utilisation_samples == 5

    def test_null_when_no_energy(self) -> None:
        """Test graceful degradation when no energy data."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=0.0,  # No energy data
            tokens_per_second=50.0,
        )

        assert result.token_efficiency_index is None

    def test_null_when_no_tokens(self) -> None:
        """Test graceful degradation when no tokens."""
        result = compute_extended_metrics(
            output_tokens=0,  # No tokens
            total_energy_j=100.0,
            tokens_per_second=0.0,
        )

        assert result.token_efficiency_index is None
        assert result.memory.tokens_per_gb_vram is None

    def test_kv_cache_metrics(self) -> None:
        """Test KV cache efficiency metrics (vLLM-specific)."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            kv_cache_stats={
                "hit_rate": 0.75,
                "blocks_used": 1000,
                "blocks_total": 2000,
            },
        )

        assert result.kv_cache.kv_cache_hit_rate == pytest.approx(0.75)
        assert result.kv_cache.kv_cache_blocks_used == 1000
        assert result.kv_cache.kv_cache_blocks_total == 2000

    def test_batch_efficiency_metrics(self) -> None:
        """Test batch efficiency metrics."""
        result = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
            batch_stats={
                "effective_batch_size": 3.5,
                "num_batches": 10,
                "configured_batch_size": 4,
            },
        )

        assert result.batch.effective_batch_size == pytest.approx(3.5)
        assert result.batch.num_batches == 10
        # batch_utilisation = 3.5 / 4 = 0.875
        assert result.batch.batch_utilisation == pytest.approx(0.875)


class TestAggregateExtendedMetrics:
    """Tests for aggregate_extended_metrics function."""

    def test_aggregate_multiple_processes(self) -> None:
        """Test aggregation of metrics from multiple processes."""
        # Create two process results
        process1 = compute_extended_metrics(
            output_tokens=500,
            total_energy_j=50.0,
            tokens_per_second=50.0,
            per_request_latencies_ms=[100.0, 120.0],
            gpu_utilisation_samples=[50.0, 60.0],
        )
        process2 = compute_extended_metrics(
            output_tokens=500,
            total_energy_j=50.0,
            tokens_per_second=50.0,
            per_request_latencies_ms=[110.0, 130.0],
            gpu_utilisation_samples=[70.0, 80.0],
        )

        result = aggregate_extended_metrics(
            raw_extended_metrics=[process1, process2],
            all_request_latencies=[100.0, 120.0, 110.0, 130.0],
            all_gpu_samples=[50.0, 60.0, 70.0, 80.0],
            aggregated_output_tokens=1000,
            aggregated_energy_j=100.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=None,
        )

        assert isinstance(result, ExtendedEfficiencyMetrics)
        # Mean of all latencies: (100+120+110+130)/4 = 115
        assert result.request_latency.e2e_latency_mean_ms == pytest.approx(115.0)
        # Mean of all GPU samples: (50+60+70+80)/4 = 65
        assert result.gpu_utilisation.sm_utilisation_mean == pytest.approx(65.0)

    def test_aggregate_with_itl(self) -> None:
        """Test aggregation preserves ITL-based TPOT."""
        process1 = ExtendedEfficiencyMetrics()
        process2 = ExtendedEfficiencyMetrics()

        result = aggregate_extended_metrics(
            raw_extended_metrics=[process1, process2],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=1000,
            aggregated_energy_j=100.0,
            aggregated_tokens_per_sec=50.0,
            itl_mean_ms=25.0,  # From aggregated latency stats
        )

        assert result.tpot_ms == pytest.approx(25.0)

    def test_aggregate_empty_input(self) -> None:
        """Test aggregation with no raw data."""
        result = aggregate_extended_metrics(
            raw_extended_metrics=[],
            all_request_latencies=[],
            all_gpu_samples=[],
            aggregated_output_tokens=0,
            aggregated_energy_j=0.0,
            aggregated_tokens_per_sec=0.0,
            itl_mean_ms=None,
        )

        # Should return empty metrics without errors
        assert isinstance(result, ExtendedEfficiencyMetrics)
        assert result.token_efficiency_index is None


class TestExtendedMetricsSchema:
    """Tests for ExtendedEfficiencyMetrics schema consistency."""

    def test_default_values(self) -> None:
        """Test that default metrics have all fields as None or empty."""
        metrics = ExtendedEfficiencyMetrics()

        assert metrics.tpot_ms is None
        assert metrics.token_efficiency_index is None
        assert metrics.memory.tokens_per_gb_vram is None
        assert metrics.gpu_utilisation.sm_utilisation_mean is None
        assert metrics.request_latency.e2e_latency_mean_ms is None
        assert metrics.kv_cache.kv_cache_hit_rate is None
        assert metrics.batch.effective_batch_size is None

    def test_json_serialization(self) -> None:
        """Test that metrics serialize to JSON with consistent schema."""
        metrics = compute_extended_metrics(
            output_tokens=1000,
            total_energy_j=100.0,
            tokens_per_second=50.0,
        )

        json_dict = metrics.model_dump()

        # All top-level keys should be present
        assert "tpot_ms" in json_dict
        assert "token_efficiency_index" in json_dict
        assert "memory" in json_dict
        assert "gpu_utilisation" in json_dict
        assert "batch" in json_dict
        assert "kv_cache" in json_dict
        assert "request_latency" in json_dict

        # Nested keys should also be present
        assert "tokens_per_gb_vram" in json_dict["memory"]
        assert "sm_utilisation_mean" in json_dict["gpu_utilisation"]

    def test_null_values_in_json(self) -> None:
        """Test that null values are properly represented in JSON."""
        metrics = ExtendedEfficiencyMetrics()
        json_dict = metrics.model_dump()

        # Null values should be None (serializes to null in JSON)
        assert json_dict["tpot_ms"] is None
        assert json_dict["memory"]["tokens_per_gb_vram"] is None
