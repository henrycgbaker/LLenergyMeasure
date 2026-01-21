"""Tests for streaming latency measurement functionality.

Tests the shared ITL collection utility and LatencyMeasurements dataclass.
"""

import pytest

from llm_energy_measure.core.inference_backends.protocols import (
    LatencyMeasurementMode,
    LatencyMeasurements,
    collect_itl_measurements,
)


class TestCollectItlMeasurements:
    """Tests for the shared collect_itl_measurements utility."""

    def test_empty_input_returns_empty_lists(self):
        """Empty input should return empty results."""
        itl_full, itl_trimmed, excluded = collect_itl_measurements([])

        assert itl_full == []
        assert itl_trimmed == []
        assert excluded == 0

    def test_single_token_request_skipped(self):
        """Requests with single token (no intervals) should be skipped."""
        # Single timestamp = no intervals possible
        result = collect_itl_measurements([[100.0]])

        itl_full, itl_trimmed, excluded = result
        assert itl_full == []
        assert itl_trimmed == []
        assert excluded == 0

    def test_two_token_request_produces_one_interval(self):
        """Two tokens produce one interval, which gets excluded from trimmed."""
        timestamps = [[0.0, 50.0]]  # 50ms interval
        itl_full, itl_trimmed, excluded = collect_itl_measurements(timestamps)

        assert len(itl_full) == 1
        assert itl_full[0] == pytest.approx(50.0)
        assert len(itl_trimmed) == 0  # Too short to trim
        assert excluded == 1

    def test_three_token_request_produces_trimmed_interval(self):
        """Three tokens produce two intervals; only middle kept in trimmed."""
        timestamps = [[0.0, 30.0, 80.0]]  # 30ms and 50ms intervals
        itl_full, itl_trimmed, excluded = collect_itl_measurements(timestamps)

        assert len(itl_full) == 2
        assert itl_full[0] == pytest.approx(30.0)
        assert itl_full[1] == pytest.approx(50.0)
        # With 3 tokens (2 intervals), we trim first and last -> nothing left
        assert len(itl_trimmed) == 0
        assert excluded == 2

    def test_four_plus_tokens_produces_trimmed_intervals(self):
        """Four+ tokens produce meaningful trimmed intervals."""
        # 4 tokens = 3 intervals, trim first and last -> 1 middle interval
        timestamps = [[0.0, 25.0, 55.0, 90.0]]  # 25, 30, 35 ms intervals
        itl_full, itl_trimmed, excluded = collect_itl_measurements(timestamps)

        assert len(itl_full) == 3
        assert itl_full[0] == pytest.approx(25.0)
        assert itl_full[1] == pytest.approx(30.0)
        assert itl_full[2] == pytest.approx(35.0)

        # Trim first (25) and last (35), keep middle (30)
        assert len(itl_trimmed) == 1
        assert itl_trimmed[0] == pytest.approx(30.0)
        assert excluded == 2

    def test_multiple_requests_aggregated(self):
        """Multiple requests have their intervals aggregated."""
        timestamps = [
            [0.0, 20.0, 50.0, 90.0],  # 20, 30, 40 ms intervals
            [0.0, 15.0, 45.0, 80.0],  # 15, 30, 35 ms intervals
        ]
        itl_full, itl_trimmed, excluded = collect_itl_measurements(timestamps)

        # 3 intervals per request * 2 requests = 6 total
        assert len(itl_full) == 6

        # 1 trimmed interval per request * 2 requests = 2 total
        # (first and last excluded from each)
        assert len(itl_trimmed) == 2
        assert itl_trimmed[0] == pytest.approx(30.0)  # Middle of first request
        assert itl_trimmed[1] == pytest.approx(30.0)  # Middle of second request
        assert excluded == 4  # 2 excluded per request

    def test_handles_mixed_request_lengths(self):
        """Handles mix of short and long requests correctly."""
        timestamps = [
            [100.0],  # Single token - skipped
            [0.0, 50.0],  # 2 tokens - 1 interval, all excluded
            [0.0, 20.0, 50.0, 90.0, 140.0],  # 5 tokens - 4 intervals, 2 trimmed
        ]
        itl_full, itl_trimmed, excluded = collect_itl_measurements(timestamps)

        # 0 + 1 + 4 = 5 full intervals
        assert len(itl_full) == 5

        # 0 + 0 + 2 = 2 trimmed intervals
        assert len(itl_trimmed) == 2

        # 0 + 1 + 2 = 3 excluded
        assert excluded == 3


class TestLatencyMeasurements:
    """Tests for the LatencyMeasurements dataclass."""

    def test_default_measurement_method_is_true_streaming(self):
        """Default measurement_mode should be TRUE_STREAMING."""
        measurements = LatencyMeasurements(
            ttft_ms=[10.0, 12.0],
            itl_full_ms=[5.0, 6.0, 7.0],
            itl_trimmed_ms=[6.0],
            request_count=2,
            total_output_tokens=10,
            excluded_tokens=4,
            streaming_mode=True,
            warmup_requests_excluded=0,
        )

        assert measurements.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
        assert measurements.measurement_method == "true_streaming"  # Legacy property

    def test_measurement_mode_can_be_set(self):
        """measurement_mode can be explicitly set to other values."""
        measurements = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[],
            request_count=1,
            total_output_tokens=5,
            excluded_tokens=1,
            streaming_mode=False,
            warmup_requests_excluded=0,
            measurement_mode=LatencyMeasurementMode.PROPORTIONAL_ESTIMATE,
        )

        assert measurements.measurement_mode == LatencyMeasurementMode.PROPORTIONAL_ESTIMATE
        assert measurements.measurement_method == "proportional"  # Legacy property

    def test_measurement_mode_per_request_batch(self):
        """measurement_mode can be set to PER_REQUEST_BATCH."""
        measurements = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[],
            request_count=1,
            total_output_tokens=5,
            excluded_tokens=1,
            streaming_mode=False,
            warmup_requests_excluded=0,
            measurement_mode=LatencyMeasurementMode.PER_REQUEST_BATCH,
        )

        assert measurements.measurement_mode == LatencyMeasurementMode.PER_REQUEST_BATCH
        assert measurements.measurement_method == "per_request_batch"  # Legacy property

    def test_all_fields_stored_correctly(self):
        """Verify all fields are stored correctly."""
        measurements = LatencyMeasurements(
            ttft_ms=[10.0, 15.0, 12.0],
            itl_full_ms=[5.0, 6.0, 7.0, 8.0],
            itl_trimmed_ms=[6.0, 7.0],
            request_count=3,
            total_output_tokens=30,
            excluded_tokens=6,
            streaming_mode=True,
            warmup_requests_excluded=2,
            measurement_mode=LatencyMeasurementMode.TRUE_STREAMING,
        )

        assert measurements.ttft_ms == [10.0, 15.0, 12.0]
        assert measurements.itl_full_ms == [5.0, 6.0, 7.0, 8.0]
        assert measurements.itl_trimmed_ms == [6.0, 7.0]
        assert measurements.request_count == 3
        assert measurements.total_output_tokens == 30
        assert measurements.excluded_tokens == 6
        assert measurements.streaming_mode is True
        assert measurements.warmup_requests_excluded == 2
        assert measurements.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
        assert measurements.measurement_method == "true_streaming"  # Legacy property


class TestLatencyMeasurementsIntegration:
    """Integration tests for LatencyMeasurements with collect_itl_measurements."""

    def test_full_workflow(self):
        """Test typical workflow: collect measurements then create dataclass."""
        # Simulate token timestamps from multiple requests
        token_timestamps_per_request = [
            [0.0, 25.0, 55.0, 90.0, 130.0],  # 5 tokens, 4 intervals
            [0.0, 20.0, 45.0, 75.0],  # 4 tokens, 3 intervals
            [0.0, 30.0, 65.0, 100.0, 140.0, 185.0],  # 6 tokens, 5 intervals
        ]
        ttft_samples = [25.0, 20.0, 30.0]  # First token time for each request

        # Collect ITL measurements
        itl_full, itl_trimmed, excluded = collect_itl_measurements(token_timestamps_per_request)

        # Create LatencyMeasurements
        measurements = LatencyMeasurements(
            ttft_ms=ttft_samples,
            itl_full_ms=itl_full,
            itl_trimmed_ms=itl_trimmed,
            request_count=3,
            total_output_tokens=15,  # 5 + 4 + 6
            excluded_tokens=excluded,
            streaming_mode=True,
            warmup_requests_excluded=0,
            measurement_mode=LatencyMeasurementMode.TRUE_STREAMING,
        )

        # Verify results
        assert measurements.request_count == 3
        assert len(measurements.ttft_ms) == 3
        assert len(measurements.itl_full_ms) == 12  # 4 + 3 + 5
        # Trimmed: (4-2) + (3-2) + (5-2) = 2 + 1 + 3 = 6
        assert len(measurements.itl_trimmed_ms) == 6
        assert measurements.excluded_tokens == 6  # 2 per request * 3 requests
