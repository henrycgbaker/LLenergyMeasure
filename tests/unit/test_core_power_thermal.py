"""Tests for power/thermal sampling module.

Tests PowerThermalSampler with mocked NVML. All tests run without a GPU
by mocking pynvml imports inside the sampling thread.
"""

import time
from unittest.mock import patch

from llenergymeasure.core.power_thermal import (
    PowerThermalResult,
    PowerThermalSample,
    PowerThermalSampler,
)
from llenergymeasure.domain.metrics import ThermalThrottleInfo


class TestImports:
    """Verify module imports work correctly."""

    def test_imports(self):
        assert PowerThermalSampler is not None
        assert PowerThermalSample is not None
        assert PowerThermalResult is not None


class TestPowerThermalSample:
    """Tests for PowerThermalSample dataclass."""

    def test_fields_with_defaults(self):
        sample = PowerThermalSample(timestamp=1.0)
        assert sample.timestamp == 1.0
        assert sample.power_w is None
        assert sample.memory_used_mb is None
        assert sample.memory_total_mb is None
        assert sample.temperature_c is None
        assert sample.sm_utilisation is None
        assert sample.thermal_throttle is False
        assert sample.throttle_reasons == 0

    def test_fields_with_values(self):
        sample = PowerThermalSample(
            timestamp=1.5,
            power_w=250.0,
            memory_used_mb=4096.0,
            memory_total_mb=81920.0,
            temperature_c=72.0,
            sm_utilisation=95.0,
            thermal_throttle=True,
            throttle_reasons=0x40,
        )
        assert sample.power_w == 250.0
        assert sample.memory_used_mb == 4096.0
        assert sample.temperature_c == 72.0
        assert sample.sm_utilisation == 95.0
        assert sample.thermal_throttle is True
        assert sample.throttle_reasons == 0x40


class TestSamplerWithoutGPU:
    """Tests for PowerThermalSampler when no GPU is available (mocked)."""

    def test_sampler_start_stop_no_gpu(self):
        """Sampler should start/stop gracefully without pynvml."""
        # Force pynvml import to fail inside the sampling thread
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("mocked: no pynvml")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            sampler = PowerThermalSampler(device_index=0, sample_interval_ms=50)
            sampler.start()
            time.sleep(0.15)
            sampler.stop()

        assert sampler.get_samples() == []
        assert sampler.is_available is False

    def test_sample_count_zero_without_gpu(self):
        """Without pynvml, sample count should be 0."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            sampler = PowerThermalSampler()
            sampler.start()
            time.sleep(0.1)
            sampler.stop()

        assert sampler.sample_count == 0

    def test_get_mean_power_none_without_samples(self):
        sampler = PowerThermalSampler()
        # Never started — no samples
        assert sampler.get_mean_power() is None

    def test_get_power_samples_empty(self):
        sampler = PowerThermalSampler()
        assert sampler.get_power_samples() == []


class TestThermalThrottleInfoNoSamples:
    """Tests for get_thermal_throttle_info when there are no samples."""

    def test_returns_all_false(self):
        sampler = PowerThermalSampler()
        info = sampler.get_thermal_throttle_info()
        assert isinstance(info, ThermalThrottleInfo)
        assert info.detected is False
        assert info.thermal is False
        assert info.power is False
        assert info.throttle_duration_sec == 0.0
        assert info.max_temperature_c is None
        assert info.throttle_timestamps == []


class TestThermalThrottleDetection:
    """Tests for thermal throttle detection with synthetic samples (positive case)."""

    def _make_sampler_with_samples(self, samples: list[PowerThermalSample]) -> PowerThermalSampler:
        """Create a sampler and inject synthetic samples."""
        sampler = PowerThermalSampler(device_index=0, sample_interval_ms=100)
        sampler._samples = samples
        return sampler

    def test_thermal_throttle_detected_with_sw_thermal(self):
        """SW thermal slowdown bit set → detected=True, thermal=True."""
        # nvmlClocksThrottleReasonSwThermalSlowdown = 0x20
        sw_thermal_bit = 0x0000000000000020
        samples = [
            PowerThermalSample(
                timestamp=0.0,
                temperature_c=82.0,
                thermal_throttle=True,
                throttle_reasons=sw_thermal_bit,
            ),
            PowerThermalSample(
                timestamp=0.1,
                temperature_c=83.0,
                thermal_throttle=True,
                throttle_reasons=sw_thermal_bit,
            ),
            PowerThermalSample(
                timestamp=0.2, temperature_c=80.0, thermal_throttle=False, throttle_reasons=0
            ),
        ]
        sampler = self._make_sampler_with_samples(samples)

        # Mock pynvml constants for get_thermal_throttle_info
        import types

        mock_pynvml = types.ModuleType("pynvml")
        mock_pynvml.nvmlClocksThrottleReasonSwThermalSlowdown = 0x20
        mock_pynvml.nvmlClocksThrottleReasonHwThermalSlowdown = 0x40
        mock_pynvml.nvmlClocksThrottleReasonSwPowerCap = 0x04
        mock_pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 0x80

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            info = sampler.get_thermal_throttle_info()

        assert info.detected is True
        assert info.thermal is True  # sw_thermal bit set
        assert info.sw_thermal is True
        assert info.hw_thermal is False
        assert info.power is False
        assert info.hw_power is False
        assert info.max_temperature_c == 83.0
        assert info.throttle_duration_sec == 0.2  # 2 samples x 0.1s

    def test_thermal_throttle_hw_power_brake(self):
        """HW power brake bit set → detected=True, hw_power=True."""
        hw_power_bit = 0x80
        samples = [
            PowerThermalSample(
                timestamp=0.0, temperature_c=70.0, thermal_throttle=False, throttle_reasons=0
            ),
            PowerThermalSample(
                timestamp=0.1,
                temperature_c=71.0,
                thermal_throttle=True,
                throttle_reasons=hw_power_bit,
            ),
        ]
        sampler = self._make_sampler_with_samples(samples)

        import types

        mock_pynvml = types.ModuleType("pynvml")
        mock_pynvml.nvmlClocksThrottleReasonSwThermalSlowdown = 0x20
        mock_pynvml.nvmlClocksThrottleReasonHwThermalSlowdown = 0x40
        mock_pynvml.nvmlClocksThrottleReasonSwPowerCap = 0x04
        mock_pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 0x80

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            info = sampler.get_thermal_throttle_info()

        assert info.detected is True
        assert info.thermal is False
        assert info.hw_power is True
        assert info.throttle_duration_sec == 0.1  # 1 sample x 0.1s

    def test_multiple_throttle_reasons_combined(self):
        """Multiple throttle reason bits set across samples → all flagged."""
        sw_thermal = 0x20
        hw_thermal = 0x40
        sw_power = 0x04
        samples = [
            PowerThermalSample(
                timestamp=0.0,
                temperature_c=85.0,
                thermal_throttle=True,
                throttle_reasons=sw_thermal | hw_thermal,
            ),
            PowerThermalSample(
                timestamp=0.1, temperature_c=84.0, thermal_throttle=True, throttle_reasons=sw_power
            ),
        ]
        sampler = self._make_sampler_with_samples(samples)

        import types

        mock_pynvml = types.ModuleType("pynvml")
        mock_pynvml.nvmlClocksThrottleReasonSwThermalSlowdown = 0x20
        mock_pynvml.nvmlClocksThrottleReasonHwThermalSlowdown = 0x40
        mock_pynvml.nvmlClocksThrottleReasonSwPowerCap = 0x04
        mock_pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 0x80

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            info = sampler.get_thermal_throttle_info()

        assert info.detected is True
        assert info.sw_thermal is True
        assert info.hw_thermal is True
        assert info.power is True  # sw_power_cap
        assert info.hw_power is False
        assert info.max_temperature_c == 85.0

    def test_no_throttle_all_clear(self):
        """No throttle bits → detected=False, all fields False."""
        samples = [
            PowerThermalSample(timestamp=0.0, temperature_c=45.0, throttle_reasons=0),
            PowerThermalSample(timestamp=0.1, temperature_c=46.0, throttle_reasons=0),
        ]
        sampler = self._make_sampler_with_samples(samples)

        import types

        mock_pynvml = types.ModuleType("pynvml")
        mock_pynvml.nvmlClocksThrottleReasonSwThermalSlowdown = 0x20
        mock_pynvml.nvmlClocksThrottleReasonHwThermalSlowdown = 0x40
        mock_pynvml.nvmlClocksThrottleReasonSwPowerCap = 0x04
        mock_pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown = 0x80

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            info = sampler.get_thermal_throttle_info()

        assert info.detected is False
        assert info.thermal is False
        assert info.power is False
        assert info.max_temperature_c == 46.0
        assert info.throttle_timestamps == []


class TestContextManagerPattern:
    """Tests for using PowerThermalSampler as a context manager."""

    def test_context_manager_no_crash(self):
        """Using 'with' pattern should not crash (with or without GPU)."""
        with PowerThermalSampler(device_index=0, sample_interval_ms=50) as sampler:
            time.sleep(0.1)
        # After context exit, sampler is stopped — verify it returns a list
        assert isinstance(sampler.get_samples(), list)

    def test_context_manager_returns_self(self):
        with PowerThermalSampler() as sampler:
            assert isinstance(sampler, PowerThermalSampler)

    def test_context_manager_no_gpu_mocked(self):
        """Without pynvml, context manager returns empty samples."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pynvml":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            PowerThermalSampler(device_index=0, sample_interval_ms=50) as sampler,
        ):
            time.sleep(0.1)

        assert sampler.get_samples() == []
        assert sampler.is_available is False


class TestPowerThermalResult:
    """Tests for PowerThermalResult dataclass."""

    def test_from_empty_sampler(self):
        sampler = PowerThermalSampler()
        result = PowerThermalResult.from_sampler(sampler)
        assert result.power_samples == []
        assert result.memory_samples == []
        assert result.temperature_samples == []
        assert result.sample_count == 0
        assert result.available is False
        assert result.thermal_throttle_info.detected is False
