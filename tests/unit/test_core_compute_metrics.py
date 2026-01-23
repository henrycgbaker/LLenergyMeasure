"""Tests for compute metrics collection."""

from unittest.mock import MagicMock, patch

from llenergymeasure.core.compute_metrics import (
    MemoryStats,
    UtilizationStats,
    get_gpu_utilization,
    get_utilization_stats,
)


class TestMemoryStats:
    """Tests for MemoryStats dataclass."""

    def test_creation(self):
        stats = MemoryStats(
            current_allocated_bytes=1000,
            max_allocated_bytes=2000,
            current_reserved_bytes=3000,
            max_reserved_bytes=4000,
        )

        assert stats.current_allocated_bytes == 1000
        assert stats.max_allocated_bytes == 2000
        assert stats.current_reserved_bytes == 3000
        assert stats.max_reserved_bytes == 4000


class TestUtilizationStats:
    """Tests for UtilizationStats dataclass."""

    def test_creation(self):
        stats = UtilizationStats(
            gpu_utilization_percent=[85.0, 90.0],
            cpu_usage_percent=45.5,
            cpu_memory_bytes=8_000_000_000,
        )

        assert stats.gpu_utilization_percent == [85.0, 90.0]
        assert stats.cpu_usage_percent == 45.5
        assert stats.cpu_memory_bytes == 8_000_000_000

    def test_creation_no_gpu(self):
        stats = UtilizationStats(
            gpu_utilization_percent=None,
            cpu_usage_percent=30.0,
            cpu_memory_bytes=4_000_000_000,
        )

        assert stats.gpu_utilization_percent is None


class TestGetGpuUtilization:
    """Tests for get_gpu_utilization."""

    def test_returns_utilization(self):
        mock_output = b"75\n80\n"

        with patch("subprocess.check_output", return_value=mock_output):
            result = get_gpu_utilization()

        assert result == [75.0, 80.0]

    def test_handles_subprocess_error(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = get_gpu_utilization()

        assert result is None

    def test_handles_empty_output(self):
        with patch("subprocess.check_output", return_value=b""):
            result = get_gpu_utilization()

        # Empty output returns None (consistent with error handling)
        assert result is None


class TestGetUtilizationStats:
    """Tests for get_utilization_stats."""

    def test_collects_stats(self):
        mock_process = MagicMock()
        mock_process.memory_info.return_value.rss = 1_000_000

        with (
            patch("llenergymeasure.core.compute_metrics.get_gpu_utilization", return_value=[50.0]),
            patch("psutil.cpu_percent", return_value=25.0),
            patch("psutil.Process", return_value=mock_process),
        ):
            result = get_utilization_stats()

        assert result.gpu_utilization_percent == [50.0]
        assert result.cpu_usage_percent == 25.0
        assert result.cpu_memory_bytes == 1_000_000

    def test_handles_psutil_error(self):
        with (
            patch("llenergymeasure.core.compute_metrics.get_gpu_utilization", return_value=None),
            patch("psutil.cpu_percent", side_effect=Exception("Error")),
        ):
            result = get_utilization_stats()

        assert result.cpu_usage_percent == 0.0
        assert result.cpu_memory_bytes == 0
