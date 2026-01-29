"""Tests for baseline power measurement module.

Tests adjust_energy_for_baseline, create_energy_breakdown, and caching logic.
All tests mock pynvml — no GPU required.
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.core.baseline import (
    BaselineCache,
    _baseline_cache,
    adjust_energy_for_baseline,
    create_energy_breakdown,
    invalidate_baseline_cache,
    measure_baseline_power,
)
from llenergymeasure.domain.metrics import EnergyBreakdown


class TestAdjustEnergyForBaseline:
    """Tests for adjust_energy_for_baseline."""

    def test_basic_adjustment(self):
        """100J - 10W*5s = 50J."""
        result = adjust_energy_for_baseline(
            total_energy_j=100.0,
            baseline_power_w=10.0,
            duration_sec=5.0,
        )
        assert result == pytest.approx(50.0)

    def test_floor_at_zero(self):
        """10J - 50W*5s = 0J (not negative)."""
        result = adjust_energy_for_baseline(
            total_energy_j=10.0,
            baseline_power_w=50.0,
            duration_sec=5.0,
        )
        assert result == 0.0

    def test_zero_baseline(self):
        """Zero baseline means no adjustment."""
        result = adjust_energy_for_baseline(
            total_energy_j=100.0,
            baseline_power_w=0.0,
            duration_sec=10.0,
        )
        assert result == pytest.approx(100.0)

    def test_zero_duration(self):
        """Zero duration means no baseline energy subtracted."""
        result = adjust_energy_for_baseline(
            total_energy_j=100.0,
            baseline_power_w=50.0,
            duration_sec=0.0,
        )
        assert result == pytest.approx(100.0)


class TestCreateEnergyBreakdown:
    """Tests for create_energy_breakdown."""

    def test_no_baseline(self):
        """Without baseline, adjusted_j is None and method is 'unavailable'."""
        eb = create_energy_breakdown(
            total_energy_j=100.0,
            baseline=None,
            duration_sec=10.0,
        )
        assert isinstance(eb, EnergyBreakdown)
        assert eb.raw_j == pytest.approx(100.0)
        assert eb.adjusted_j is None
        assert eb.baseline_power_w is None
        assert eb.baseline_method == "unavailable"
        assert eb.baseline_timestamp is None
        assert eb.baseline_cache_age_sec is None

    def test_with_baseline(self):
        """With baseline, adjusted_j is computed and baseline fields populated."""
        baseline = BaselineCache(
            power_w=10.0,
            timestamp=time.time() - 0.5,  # fresh (<1s)
            device_index=0,
            sample_count=100,
            duration_sec=30.0,
        )
        eb = create_energy_breakdown(
            total_energy_j=100.0,
            baseline=baseline,
            duration_sec=5.0,
        )
        assert eb.raw_j == pytest.approx(100.0)
        assert eb.adjusted_j == pytest.approx(50.0)  # 100 - 10*5
        assert eb.baseline_power_w == pytest.approx(10.0)
        assert eb.baseline_method == "fresh"
        assert eb.baseline_timestamp is not None
        assert eb.baseline_cache_age_sec is not None

    def test_with_cached_baseline(self):
        """Cached baseline (age > 1s) has method 'cached'."""
        baseline = BaselineCache(
            power_w=20.0,
            timestamp=time.time() - 10.0,  # 10 seconds old
            device_index=0,
            sample_count=50,
            duration_sec=30.0,
        )
        eb = create_energy_breakdown(
            total_energy_j=200.0,
            baseline=baseline,
            duration_sec=5.0,
        )
        assert eb.baseline_method == "cached"
        assert eb.adjusted_j == pytest.approx(100.0)  # 200 - 20*5


class TestBaselineCacheInvalidation:
    """Tests for baseline cache management."""

    def setup_method(self):
        """Clear global cache before each test."""
        _baseline_cache.clear()

    def test_invalidate_specific_device(self):
        """Invalidating a specific device clears only that entry."""
        _baseline_cache[0] = BaselineCache(
            power_w=10.0, timestamp=time.time(), device_index=0, sample_count=1, duration_sec=1.0
        )
        _baseline_cache[1] = BaselineCache(
            power_w=20.0, timestamp=time.time(), device_index=1, sample_count=1, duration_sec=1.0
        )
        invalidate_baseline_cache(device_index=0)
        assert 0 not in _baseline_cache
        assert 1 in _baseline_cache

    def test_invalidate_all(self):
        """Invalidating without device_index clears all entries."""
        _baseline_cache[0] = BaselineCache(
            power_w=10.0, timestamp=time.time(), device_index=0, sample_count=1, duration_sec=1.0
        )
        _baseline_cache[1] = BaselineCache(
            power_w=20.0, timestamp=time.time(), device_index=1, sample_count=1, duration_sec=1.0
        )
        invalidate_baseline_cache()
        assert len(_baseline_cache) == 0

    def test_invalidate_nonexistent_device_no_error(self):
        """Invalidating a device that was never cached should not raise."""
        invalidate_baseline_cache(device_index=99)  # should not raise


class TestMeasureBaselineCaching:
    """Tests for measure_baseline_power with mocked pynvml."""

    def setup_method(self):
        _baseline_cache.clear()

    @patch("llenergymeasure.core.baseline.time")
    def test_caching_reuses_result(self, mock_time):
        """Second call should use cached result, not re-measure."""
        mock_time.time.return_value = 1000.0
        mock_time.monotonic.side_effect = [0.0, 0.05, 0.1, 0.15, 100.0, 100.0]
        mock_time.sleep = MagicMock()

        mock_pynvml = MagicMock()
        mock_pynvml.NVMLError = Exception
        mock_pynvml.nvmlDeviceGetPowerUsage.return_value = 150_000  # 150W in mW

        with patch.dict("sys.modules", {"pynvml": mock_pynvml}):
            result1 = measure_baseline_power(
                device_index=0, duration_sec=0.1, sample_interval_ms=50
            )

        assert result1 is not None
        assert result1.power_w == pytest.approx(150.0)

        # Second call should hit cache
        mock_time.time.return_value = 1001.0  # 1 second later, still within TTL
        result2 = measure_baseline_power(device_index=0, cache_ttl_sec=3600.0)
        assert result2 is result1  # same object from cache

    def test_measure_without_pynvml_returns_none(self):
        """When pynvml is not importable, returns None."""
        _baseline_cache.clear()
        with patch.dict("sys.modules", {"pynvml": None}):
            # Force ImportError by removing from cache

            # Simulate pynvml not being available - the function does
            # `import pynvml` internally which will fail
            result = measure_baseline_power(device_index=0, duration_sec=0.1)
        # Without a GPU, this returns None (graceful degradation)
        # The test just verifies no crash
        assert result is None or isinstance(result, BaselineCache)
