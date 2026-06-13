"""Unit tests for domain/metrics.py - precision, energy, inference, latency models."""

from __future__ import annotations

import pytest

from llenergymeasure.domain.metrics import (
    FlopsResult,
    LatencyMeasurementMode,
    LatencyMeasurements,
    collect_itl_measurements,
    compute_latency_statistics,
)
from tests.conftest import make_energy_metrics, make_inference_metrics

# ---------------------------------------------------------------------------
# TestFlopsResult
# ---------------------------------------------------------------------------


class TestFlopsResult:
    """is_valid property based on value."""

    def test_positive_is_valid(self):
        fr = FlopsResult(value=1e9, method="calflops", confidence="high", precision="fp16")
        assert fr.is_valid is True

    def test_zero_not_valid(self):
        fr = FlopsResult(value=0.0, method="calflops", confidence="low", precision="fp16")
        assert fr.is_valid is False

    def test_negative_not_valid(self):
        fr = FlopsResult(
            value=-1.0, method="parameter_estimate", confidence="low", precision="fp16"
        )
        assert fr.is_valid is False


# ---------------------------------------------------------------------------
# TestEnergyMetrics
# ---------------------------------------------------------------------------


class TestEnergyMetrics:
    """total_power_w property."""

    def test_total_power_w(self):
        em = make_energy_metrics(gpu_power_w=100.0, cpu_power_w=25.0)
        assert em.total_power_w == pytest.approx(125.0)


# ---------------------------------------------------------------------------
# TestInferenceMetrics
# ---------------------------------------------------------------------------


class TestInferenceMetrics:
    """throughput alias."""

    def test_throughput_alias(self):
        im = make_inference_metrics(tokens_per_second=42.5)
        assert im.throughput == im.tokens_per_second == 42.5


# ---------------------------------------------------------------------------
# TestComputeLatencyStatistics
# ---------------------------------------------------------------------------


class TestComputeLatencyStatistics:
    """compute_latency_statistics() from flat sample lists."""

    def test_empty_ttft_returns_none(self):
        assert compute_latency_statistics([]) is None

    def test_empty_ttft_with_itl_still_none(self):
        assert compute_latency_statistics([], itl_trimmed_ms=[5.0], itl_full_ms=[5.0]) is None

    def test_single_sample(self):
        stats = compute_latency_statistics([42.0])
        assert stats is not None
        assert stats.ttft_mean_ms == pytest.approx(42.0)
        assert stats.ttft_median_ms == pytest.approx(42.0)
        assert stats.ttft_p95_ms == pytest.approx(42.0)
        assert stats.ttft_p99_ms == pytest.approx(42.0)
        assert stats.ttft_min_ms == pytest.approx(42.0)
        assert stats.ttft_max_ms == pytest.approx(42.0)
        assert stats.ttft_samples == 1

    def test_multi_sample_percentiles(self):
        ttft = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = compute_latency_statistics(ttft)
        assert stats is not None
        assert stats.ttft_mean_ms == pytest.approx(30.0)
        assert stats.ttft_median_ms == pytest.approx(30.0)
        assert stats.ttft_min_ms == pytest.approx(10.0)
        assert stats.ttft_max_ms == pytest.approx(50.0)
        # p95 of 5 evenly spaced samples sits near the top
        assert stats.ttft_p95_ms == pytest.approx(48.0)
        assert stats.ttft_p99_ms == pytest.approx(49.6)
        assert stats.ttft_samples == 5

    def test_itl_none_leaves_itl_fields_none(self):
        stats = compute_latency_statistics([10.0, 20.0])
        assert stats is not None
        assert stats.itl_mean_ms is None
        assert stats.itl_median_ms is None
        assert stats.itl_p95_ms is None
        assert stats.itl_p99_ms is None
        assert stats.itl_samples == 0
        assert stats.itl_full_mean_ms is None
        assert stats.itl_full_p99_ms is None

    def test_itl_trimmed_computed(self):
        stats = compute_latency_statistics(
            [10.0], itl_trimmed_ms=[2.0, 4.0, 6.0, 8.0], itl_full_ms=[1.0, 2.0, 4.0, 6.0, 8.0, 9.0]
        )
        assert stats is not None
        assert stats.itl_mean_ms == pytest.approx(5.0)
        assert stats.itl_median_ms == pytest.approx(5.0)
        assert stats.itl_samples == 4
        assert stats.itl_full_mean_ms == pytest.approx(5.0)
        assert stats.itl_full_p99_ms is not None

    def test_measurement_mode_defaults_to_true_streaming(self):
        from llenergymeasure.domain.metrics import LatencyMeasurementMode

        stats = compute_latency_statistics([10.0])
        assert stats is not None
        assert stats.measurement_mode is LatencyMeasurementMode.TRUE_STREAMING

    def test_measurement_mode_round_trips(self):
        from llenergymeasure.domain.metrics import LatencyMeasurementMode

        stats = compute_latency_statistics(
            [10.0],
            itl_trimmed_ms=[5.0],
            measurement_mode=LatencyMeasurementMode.PROPORTIONAL_ESTIMATE,
        )
        assert stats is not None
        assert stats.measurement_mode is LatencyMeasurementMode.PROPORTIONAL_ESTIMATE


# ---------------------------------------------------------------------------
# TestCollectItlMeasurements
# ---------------------------------------------------------------------------


class TestCollectItlMeasurements:
    """collect_itl_measurements() from per-request token timestamps."""

    def test_normal_multi_request(self):
        timestamps = [
            [0.0, 10.0, 20.0, 30.0, 40.0],  # 4 intervals, trimmed: [10, 10]
            [0.0, 15.0, 25.0, 35.0, 45.0],  # 4 intervals, trimmed: [10, 10]
        ]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 8  # 4 + 4
        assert len(trimmed) == 4  # 2 + 2
        assert excluded == 4  # 2 per request

    def test_trimming_removes_first_last(self):
        timestamps = [[0.0, 100.0, 20.0, 30.0, 5.0]]  # 4 intervals
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        # full = [100, -80, 10, -25]
        assert len(full) == 4
        # trimmed = [-80, 10] (indices 1:-1)
        assert len(trimmed) == 2
        assert excluded == 2

    def test_short_request_two_tokens(self):
        """Two tokens = 1 interval; too short to trim -> excluded."""
        timestamps = [[0.0, 10.0]]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 1
        assert len(trimmed) == 0
        assert excluded == 1

    def test_short_request_three_tokens(self):
        """Three tokens = 2 intervals; still too short for 3-interval trim."""
        timestamps = [[0.0, 10.0, 20.0]]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 2
        assert len(trimmed) == 0
        assert excluded == 2

    def test_single_token_skipped(self):
        """Single token has no intervals."""
        timestamps = [[5.0]]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert full == []
        assert trimmed == []
        assert excluded == 0

    def test_empty_input(self):
        full, trimmed, excluded = collect_itl_measurements([])
        assert full == []
        assert trimmed == []
        assert excluded == 0

    def test_mixed_lengths(self):
        timestamps = [
            [0.0, 10.0, 20.0, 30.0, 40.0],  # 4 intervals, trimmed: 2
            [0.0, 5.0],  # 1 interval, not trimmable
            [0.0],  # 0 intervals, skipped
        ]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 5  # 4 + 1
        assert len(trimmed) == 2
        assert excluded == 3  # 2 from first + 1 from second

    def test_excluded_count_accuracy(self):
        """Excluded count must match intervals removed from full to get trimmed."""
        timestamps = [
            [0.0, 10.0, 20.0, 30.0],  # 3 intervals -> trim first/last -> 1 trimmed, 2 excluded
        ]
        full, trimmed, excluded = collect_itl_measurements(timestamps)
        assert len(full) == 3
        assert len(trimmed) == 1
        assert excluded == 2

    def test_four_token_request(self):
        """Four tokens = 3 intervals -> exactly trimmable."""
        timestamps = [[0.0, 10.0, 20.0, 30.0]]
        full, trimmed, _excluded = collect_itl_measurements(timestamps)
        assert len(full) == 3
        assert len(trimmed) == 1  # middle interval only
        assert trimmed[0] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# TestLatencyMeasurements
# ---------------------------------------------------------------------------


class TestLatencyMeasurements:
    """LatencyMeasurements defaults."""

    def test_default_measurement_mode(self):
        lm = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[5.0],
            request_count=1,
            total_output_tokens=10,
            excluded_tokens=0,
            streaming_mode=True,
            warmup_requests_excluded=0,
        )
        assert lm.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
