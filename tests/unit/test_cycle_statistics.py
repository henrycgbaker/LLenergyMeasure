"""Unit tests for cycle statistics module."""

from datetime import datetime, timedelta

import pytest

from llm_energy_measure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
)
from llm_energy_measure.results.cycle_statistics import (
    calculate_cv,
    calculate_statistics,
    create_cycle_metadata,
    create_cycle_statistics,
    create_multi_cycle_result,
)


class TestCalculateStatistics:
    """Tests for calculate_statistics function."""

    def test_empty_list(self) -> None:
        """Test with empty list returns zeros."""
        mean, std, ci_lo, ci_hi = calculate_statistics([])
        assert mean == 0.0
        assert std == 0.0
        assert ci_lo == 0.0
        assert ci_hi == 0.0

    def test_single_value(self) -> None:
        """Test with single value returns that value with zero std."""
        mean, std, ci_lo, ci_hi = calculate_statistics([42.0])
        assert mean == 42.0
        assert std == 0.0
        assert ci_lo == 42.0
        assert ci_hi == 42.0

    def test_two_values(self) -> None:
        """Test with two values."""
        mean, std, _, _ = calculate_statistics([10.0, 20.0])
        assert mean == 15.0
        # Sample std for [10, 20] = sqrt((25+25)/(2-1)) = sqrt(50) ≈ 7.07
        assert abs(std - 7.071) < 0.01

    def test_multiple_values(self) -> None:
        """Test with multiple values."""
        values = [100.0, 102.0, 98.0, 101.0, 99.0]
        mean, std, ci_lo, ci_hi = calculate_statistics(values)
        assert mean == 100.0  # Mean of [100, 102, 98, 101, 99]
        assert std > 0
        assert ci_lo < mean
        assert ci_hi > mean

    def test_confidence_interval_contains_mean(self) -> None:
        """Test that CI contains the mean."""
        values = [50.0, 52.0, 48.0, 51.0, 49.0]
        mean, _, ci_lo, ci_hi = calculate_statistics(values)
        assert ci_lo <= mean <= ci_hi


class TestCalculateCV:
    """Tests for coefficient of variation calculation."""

    def test_zero_mean(self) -> None:
        """Test with zero mean returns 0."""
        assert calculate_cv(0.0, 1.0) == 0.0

    def test_zero_std(self) -> None:
        """Test with zero std returns 0."""
        assert calculate_cv(10.0, 0.0) == 0.0

    def test_normal_case(self) -> None:
        """Test CV calculation."""
        # CV = std / |mean|
        assert calculate_cv(100.0, 10.0) == 0.1
        assert calculate_cv(100.0, 5.0) == 0.05

    def test_negative_mean(self) -> None:
        """Test CV with negative mean uses absolute value."""
        assert calculate_cv(-100.0, 10.0) == 0.1


class TestCreateCycleMetadata:
    """Tests for create_cycle_metadata function."""

    def test_minimal(self) -> None:
        """Test with minimal args."""
        meta = create_cycle_metadata(cycle_id=0)
        assert meta.cycle_id == 0
        assert meta.timestamp is not None
        assert meta.gpu_temperature_c is None
        assert meta.system_load is None

    def test_with_all_args(self) -> None:
        """Test with all args provided."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        meta = create_cycle_metadata(
            cycle_id=3,
            timestamp=ts,
            gpu_temperature_c=65.5,
            system_load=2.5,
        )
        assert meta.cycle_id == 3
        assert meta.timestamp == ts
        assert meta.gpu_temperature_c == 65.5
        assert meta.system_load == 2.5


def _create_mock_aggregated_result(
    experiment_id: str,
    total_energy_j: float = 100.0,
    avg_tokens_per_second: float = 50.0,
    total_tokens: int = 1000,
) -> AggregatedResult:
    """Create a mock AggregatedResult for testing."""
    now = datetime.now()
    return AggregatedResult(
        experiment_id=experiment_id,
        aggregation=AggregationMetadata(
            num_processes=1,
            temporal_overlap_verified=True,
            gpu_attribution_verified=True,
        ),
        total_tokens=total_tokens,
        total_energy_j=total_energy_j,
        total_inference_time_sec=total_tokens / avg_tokens_per_second,
        avg_tokens_per_second=avg_tokens_per_second,
        avg_energy_per_token_j=total_energy_j / total_tokens,
        total_flops=1e12,
        start_time=now,
        end_time=now + timedelta(seconds=20),
    )


class TestCreateCycleStatistics:
    """Tests for create_cycle_statistics function."""

    def test_empty_results_raises(self) -> None:
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate statistics"):
            create_cycle_statistics([])

    def test_single_result(self) -> None:
        """Test with single result."""
        result = _create_mock_aggregated_result("test-001", total_energy_j=100.0)
        stats = create_cycle_statistics([result])

        assert stats.num_cycles == 1
        assert stats.energy_mean_j == 100.0
        assert stats.energy_std_j == 0.0

    def test_multiple_results(self) -> None:
        """Test with multiple results."""
        results = [
            _create_mock_aggregated_result(
                "test-001_c0", total_energy_j=100.0, avg_tokens_per_second=50.0
            ),
            _create_mock_aggregated_result(
                "test-001_c1", total_energy_j=110.0, avg_tokens_per_second=48.0
            ),
            _create_mock_aggregated_result(
                "test-001_c2", total_energy_j=95.0, avg_tokens_per_second=52.0
            ),
        ]
        stats = create_cycle_statistics(results)

        assert stats.num_cycles == 3
        # Energy mean should be (100 + 110 + 95) / 3 ≈ 101.67
        assert abs(stats.energy_mean_j - 101.67) < 0.1
        assert stats.energy_std_j > 0
        # CI should contain mean
        assert stats.energy_ci_95_lower <= stats.energy_mean_j <= stats.energy_ci_95_upper

    def test_cv_calculation(self) -> None:
        """Test CV is properly calculated."""
        results = [
            _create_mock_aggregated_result("test-001_c0", total_energy_j=100.0),
            _create_mock_aggregated_result("test-001_c1", total_energy_j=100.0),
            _create_mock_aggregated_result("test-001_c2", total_energy_j=100.0),
        ]
        stats = create_cycle_statistics(results)

        # With identical values, CV should be 0 (or very close)
        assert stats.energy_cv < 0.001


class TestCreateMultiCycleResult:
    """Tests for create_multi_cycle_result function."""

    def test_empty_results_raises(self) -> None:
        """Test that empty results raises ValueError."""
        with pytest.raises(ValueError, match="Cannot create multi-cycle result"):
            create_multi_cycle_result(
                experiment_id="test-001",
                cycle_results=[],
                cycle_metadata=[],
            )

    def test_creates_valid_result(self) -> None:
        """Test creation of valid multi-cycle result."""
        results = [
            _create_mock_aggregated_result("test-001_c0", total_energy_j=100.0),
            _create_mock_aggregated_result("test-001_c1", total_energy_j=105.0),
            _create_mock_aggregated_result("test-001_c2", total_energy_j=95.0),
        ]
        metadata = [
            create_cycle_metadata(0),
            create_cycle_metadata(1),
            create_cycle_metadata(2),
        ]

        multi_result = create_multi_cycle_result(
            experiment_id="test-001",
            cycle_results=results,
            cycle_metadata=metadata,
            effective_config={"model_name": "test-model"},
        )

        assert multi_result.experiment_id == "test-001"
        assert multi_result.num_cycles == 3
        assert len(multi_result.cycle_results) == 3
        assert len(multi_result.cycle_metadata) == 3
        assert multi_result.statistics.num_cycles == 3
        assert multi_result.effective_config == {"model_name": "test-model"}

    def test_timestamps(self) -> None:
        """Test that start/end times are correctly calculated."""
        r1 = _create_mock_aggregated_result("test-001_c0")
        r2 = _create_mock_aggregated_result("test-001_c1")

        # Manually set timestamps for testing
        results = [r1, r2]

        multi_result = create_multi_cycle_result(
            experiment_id="test-001",
            cycle_results=results,
            cycle_metadata=[create_cycle_metadata(0), create_cycle_metadata(1)],
        )

        # Start time should be earliest, end time should be latest
        assert multi_result.start_time <= multi_result.end_time
        assert multi_result.total_duration_sec >= 0
