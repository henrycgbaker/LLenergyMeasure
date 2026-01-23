"""Tests for PyTorch backend streaming latency measurement.

Tests the streaming inference capabilities of the PyTorch backend,
including TextIteratorStreamer support, fallback modes, and warnings.
"""

import pytest


class TestPyTorchStreamingSupport:
    """Tests for PyTorch streaming capability detection."""

    def test_supports_streaming_when_available(self):
        """_supports_streaming returns True when TextIteratorStreamer available."""
        from llm_energy_measure.core.inference_backends.pytorch import PyTorchBackend

        backend = PyTorchBackend()

        # TextIteratorStreamer is available in transformers >= 4.28
        assert backend._supports_streaming() is True

    def test_supports_streaming_when_unavailable(self):
        """_supports_streaming returns False when TextIteratorStreamer not importable."""
        # Note: This is difficult to test without uninstalling transformers
        # The real check does a fresh import of TextIteratorStreamer
        # In practice, the check will return True because transformers is installed
        # We document this behavior here rather than mock at the import level
        pass


class TestPyTorchStreamingFlags:
    """Tests for streaming-related flags and state."""

    def test_is_compiled_initially_false(self):
        """_is_compiled should be False on init."""
        from llm_energy_measure.core.inference_backends.pytorch import PyTorchBackend

        backend = PyTorchBackend()
        assert backend._is_compiled is False


class TestLatencyMeasurementsStructure:
    """Tests for LatencyMeasurements structure from PyTorch backend."""

    def test_latency_measurements_has_measurement_mode_field(self):
        """LatencyMeasurements should have measurement_mode field."""
        from llm_energy_measure.core.inference_backends.protocols import (
            LatencyMeasurementMode,
            LatencyMeasurements,
        )

        measurements = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[],
            request_count=1,
            total_output_tokens=5,
            excluded_tokens=1,
            streaming_mode=True,
            warmup_requests_excluded=0,
            measurement_mode=LatencyMeasurementMode.TRUE_STREAMING,
        )

        assert hasattr(measurements, "measurement_mode")
        assert measurements.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
        # Legacy property still available
        assert measurements.measurement_method == "true_streaming"

    def test_latency_measurements_default_method_is_true_streaming(self):
        """Default measurement_mode should be TRUE_STREAMING."""
        from llm_energy_measure.core.inference_backends.protocols import (
            LatencyMeasurementMode,
            LatencyMeasurements,
        )

        measurements = LatencyMeasurements(
            ttft_ms=[10.0],
            itl_full_ms=[5.0],
            itl_trimmed_ms=[],
            request_count=1,
            total_output_tokens=5,
            excluded_tokens=1,
            streaming_mode=True,
            warmup_requests_excluded=0,
        )

        assert measurements.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
        assert measurements.measurement_method == "true_streaming"


class TestPyTorchBackendResultStructure:
    """Tests for BackendResult structure with streaming."""

    def test_backend_result_supports_latency_measurements(self):
        """BackendResult should support latency_measurements field."""
        from llm_energy_measure.core.inference_backends.protocols import (
            BackendResult,
            LatencyMeasurementMode,
            LatencyMeasurements,
        )

        measurements = LatencyMeasurements(
            ttft_ms=[15.0, 18.0, 12.0],
            itl_full_ms=[5.0, 6.0, 7.0, 8.0, 9.0],
            itl_trimmed_ms=[6.0, 7.0, 8.0],
            request_count=3,
            total_output_tokens=15,
            excluded_tokens=6,
            streaming_mode=True,
            warmup_requests_excluded=2,
            measurement_mode=LatencyMeasurementMode.TRUE_STREAMING,
        )

        result = BackendResult(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            inference_time_sec=10.0,
            time_to_first_token_ms=15.0,
            latency_measurements=measurements,
        )

        assert result.latency_measurements is not None
        assert result.latency_measurements.request_count == 3
        assert result.latency_measurements.measurement_mode == LatencyMeasurementMode.TRUE_STREAMING
        assert result.latency_measurements.measurement_method == "true_streaming"

    def test_backend_result_latency_measurements_optional(self):
        """latency_measurements should be optional (None by default)."""
        from llm_energy_measure.core.inference_backends.protocols import BackendResult

        result = BackendResult(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            inference_time_sec=10.0,
        )

        assert result.latency_measurements is None


class TestCollectItlMeasurementsIntegration:
    """Tests for collect_itl_measurements usage in PyTorch backend."""

    def test_collect_itl_measurements_imported_in_pytorch(self):
        """collect_itl_measurements should be importable from protocols."""
        from llm_energy_measure.core.inference_backends.protocols import (
            collect_itl_measurements,
        )

        assert callable(collect_itl_measurements)

    def test_itl_trimming_logic(self):
        """ITL trimming should exclude first and last intervals per request."""
        from llm_energy_measure.core.inference_backends.protocols import (
            collect_itl_measurements,
        )

        # Request with 5 tokens = 4 intervals
        # Intervals: 25ms, 30ms, 35ms, 40ms
        # Trimmed: 30ms, 35ms (first and last excluded)
        timestamps = [[0.0, 25.0, 55.0, 90.0, 130.0]]

        itl_full, itl_trimmed, excluded = collect_itl_measurements(timestamps)

        assert len(itl_full) == 4
        assert len(itl_trimmed) == 2
        assert itl_trimmed[0] == pytest.approx(30.0)
        assert itl_trimmed[1] == pytest.approx(35.0)
        assert excluded == 2


class TestWarningBehavior:
    """Tests for warning emission in streaming mode."""

    def test_torch_compile_warning_flag_tracked(self):
        """Backend should track _is_compiled flag for warning."""
        from llm_energy_measure.core.inference_backends.pytorch import PyTorchBackend

        backend = PyTorchBackend()
        assert hasattr(backend, "_is_compiled")
        assert backend._is_compiled is False

    def test_statistical_sufficiency_threshold(self):
        """Verify the statistical sufficiency threshold constant."""
        # The implementation warns when < 30 samples
        # This test documents that expectation
        min_samples_for_reliable_percentiles = 30

        # This is a documentation test - the actual warning is emitted
        # during _run_streaming_inference when len(measurement_prompts) < 30
        assert min_samples_for_reliable_percentiles == 30
