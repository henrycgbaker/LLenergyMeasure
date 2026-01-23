"""Tests for results aggregation functionality."""

from datetime import datetime, timedelta

import pytest

from llenergymeasure.domain.experiment import RawProcessResult, Timestamps
from llenergymeasure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llenergymeasure.exceptions import AggregationError
from llenergymeasure.results.aggregation import (
    aggregate_results,
    calculate_efficiency_metrics,
)


def make_raw_result(
    process_index: int,
    gpu_id: int,
    tokens: int = 100,
    energy_j: float = 10.0,
    tokens_per_second: float = 50.0,
    start_offset_sec: float = 0.0,
    duration_sec: float = 10.0,
) -> RawProcessResult:
    """Create a RawProcessResult for testing."""
    start = datetime(2024, 1, 1, 12, 0, 0) + timedelta(seconds=start_offset_sec)
    end = start + timedelta(seconds=duration_sec)

    return RawProcessResult(
        experiment_id="test_exp",
        process_index=process_index,
        gpu_id=gpu_id,
        config_name="test_config",
        model_name="test/model",
        timestamps=Timestamps(start=start, end=end, duration_sec=duration_sec),
        inference_metrics=InferenceMetrics(
            total_tokens=tokens,
            input_tokens=tokens // 2,
            output_tokens=tokens // 2,
            inference_time_sec=duration_sec,
            tokens_per_second=tokens_per_second,
            latency_per_token_ms=1000.0 / tokens_per_second,
        ),
        energy_metrics=EnergyMetrics(
            total_energy_j=energy_j,
            gpu_energy_j=energy_j * 0.8,
            cpu_energy_j=energy_j * 0.15,
            ram_energy_j=energy_j * 0.05,
            gpu_power_w=100.0,
            cpu_power_w=50.0,
            duration_sec=duration_sec,
            emissions_kg_co2=0.001,
            energy_per_token_j=energy_j / tokens,
        ),
        compute_metrics=ComputeMetrics(
            flops_total=1e12,
            flops_per_token=1e10,
            flops_per_second=1e11,
            peak_memory_mb=4096.0,
            model_memory_mb=2048.0,
            flops_method="ptflops",
            flops_confidence="medium",
            compute_precision="fp16",
        ),
    )


class TestAggregateResults:
    """Tests for aggregate_results function."""

    def test_aggregate_empty_raises_error(self) -> None:
        with pytest.raises(AggregationError, match="empty"):
            aggregate_results("test_exp", [])

    def test_aggregate_single_result(self) -> None:
        raw = make_raw_result(0, 0, tokens=100, energy_j=10.0)
        result = aggregate_results("test_exp", [raw])

        assert result.experiment_id == "test_exp"
        assert result.total_tokens == 100
        assert result.total_energy_j == 10.0
        assert result.aggregation.num_processes == 1

    def test_aggregate_multiple_results_sums_correctly(self) -> None:
        raw1 = make_raw_result(0, 0, tokens=100, energy_j=10.0)
        raw2 = make_raw_result(1, 1, tokens=150, energy_j=15.0)
        raw3 = make_raw_result(2, 2, tokens=200, energy_j=20.0)

        result = aggregate_results("test_exp", [raw1, raw2, raw3])

        # Tokens and energy should be summed
        assert result.total_tokens == 450
        assert result.total_energy_j == 45.0
        assert result.aggregation.num_processes == 3

    def test_aggregate_throughput_averaged(self) -> None:
        # All have 50 tok/s throughput
        raw1 = make_raw_result(0, 0, tokens_per_second=50.0)
        raw2 = make_raw_result(1, 1, tokens_per_second=50.0)

        result = aggregate_results("test_exp", [raw1, raw2])

        # Average of 50 and 50 = 50
        assert result.avg_tokens_per_second == 50.0

    def test_aggregate_energy_per_token_calculated(self) -> None:
        raw1 = make_raw_result(0, 0, tokens=100, energy_j=10.0)
        raw2 = make_raw_result(1, 1, tokens=100, energy_j=10.0)

        result = aggregate_results("test_exp", [raw1, raw2])

        # 20J / 200 tokens = 0.1 J/token
        assert result.avg_energy_per_token_j == pytest.approx(0.1)

    def test_aggregate_flops_summed(self) -> None:
        raw1 = make_raw_result(0, 0)  # 1e12 FLOPs
        raw2 = make_raw_result(1, 1)  # 1e12 FLOPs

        result = aggregate_results("test_exp", [raw1, raw2])

        assert result.total_flops == pytest.approx(2e12)

    def test_temporal_overlap_verification_concurrent(self) -> None:
        # Both run at the same time (offset=0)
        raw1 = make_raw_result(0, 0, start_offset_sec=0.0, duration_sec=10.0)
        raw2 = make_raw_result(1, 1, start_offset_sec=0.0, duration_sec=10.0)

        result = aggregate_results("test_exp", [raw1, raw2])

        assert result.aggregation.temporal_overlap_verified is True
        assert len(result.aggregation.warnings) == 0

    def test_temporal_overlap_verification_sequential(self) -> None:
        # Process 2 starts after process 1 ends (no overlap)
        raw1 = make_raw_result(0, 0, start_offset_sec=0.0, duration_sec=10.0)
        raw2 = make_raw_result(1, 1, start_offset_sec=15.0, duration_sec=10.0)

        result = aggregate_results("test_exp", [raw1, raw2])

        assert result.aggregation.temporal_overlap_verified is False
        assert any("concurrent" in w.lower() for w in result.aggregation.warnings)

    def test_gpu_attribution_unique_ids(self) -> None:
        raw1 = make_raw_result(0, 0)  # GPU 0
        raw2 = make_raw_result(1, 1)  # GPU 1

        result = aggregate_results("test_exp", [raw1, raw2])

        assert result.aggregation.gpu_attribution_verified is True

    def test_gpu_attribution_duplicate_ids(self) -> None:
        raw1 = make_raw_result(0, 0)  # GPU 0
        raw2 = make_raw_result(1, 0)  # Also GPU 0 - duplicate!

        result = aggregate_results("test_exp", [raw1, raw2])

        assert result.aggregation.gpu_attribution_verified is False
        assert any("double" in w.lower() for w in result.aggregation.warnings)

    def test_time_range_uses_extremes(self) -> None:
        raw1 = make_raw_result(0, 0, start_offset_sec=0.0, duration_sec=10.0)
        raw2 = make_raw_result(1, 1, start_offset_sec=5.0, duration_sec=20.0)

        result = aggregate_results("test_exp", [raw1, raw2])

        # Start from earliest (offset 0), end at latest (offset 5 + 20 = 25)
        assert result.start_time == raw1.timestamps.start
        assert result.end_time == raw2.timestamps.end

    def test_process_results_preserved(self) -> None:
        raw1 = make_raw_result(0, 0)
        raw2 = make_raw_result(1, 1)

        result = aggregate_results("test_exp", [raw1, raw2])

        assert len(result.process_results) == 2
        assert result.process_results[0] == raw1
        assert result.process_results[1] == raw2


class TestCalculateEfficiencyMetrics:
    """Tests for calculate_efficiency_metrics function."""

    def test_tokens_per_joule(self) -> None:
        raw = make_raw_result(0, 0, tokens=100, energy_j=10.0)
        result = aggregate_results("test_exp", [raw])

        metrics = calculate_efficiency_metrics(result)

        assert metrics["tokens_per_joule"] == pytest.approx(10.0)

    def test_joules_per_token(self) -> None:
        raw = make_raw_result(0, 0, tokens=100, energy_j=10.0)
        result = aggregate_results("test_exp", [raw])

        metrics = calculate_efficiency_metrics(result)

        assert metrics["joules_per_token"] == pytest.approx(0.1)

    def test_effective_batch_throughput(self) -> None:
        raw1 = make_raw_result(0, 0, tokens_per_second=50.0)
        raw2 = make_raw_result(1, 1, tokens_per_second=50.0)
        result = aggregate_results("test_exp", [raw1, raw2])

        metrics = calculate_efficiency_metrics(result)

        # 50 avg * 2 processes = 100 effective throughput
        assert metrics["effective_batch_throughput"] == pytest.approx(100.0)

    def test_zero_energy_handled(self) -> None:
        raw = make_raw_result(0, 0, tokens=100, energy_j=0.0)
        result = aggregate_results("test_exp", [raw])

        metrics = calculate_efficiency_metrics(result)

        assert metrics["tokens_per_joule"] == 0.0
        assert metrics["flops_per_joule"] == 0.0
