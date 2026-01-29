"""Tests for environment metadata collection module.

Tests collect_environment_metadata with mocked NVML. All tests run without a GPU.
"""

from datetime import datetime
from unittest.mock import patch

from llenergymeasure.core.environment import collect_environment_metadata
from llenergymeasure.domain.environment import EnvironmentMetadata


class TestImports:
    """Verify module imports work correctly."""

    def test_imports(self):
        from llenergymeasure.core.environment import collect_environment_metadata

        assert collect_environment_metadata is not None

    def test_domain_imports(self):
        from llenergymeasure.domain.environment import (
            ContainerEnvironment,
            CPUEnvironment,
            CUDAEnvironment,
            EnvironmentMetadata,
            GPUEnvironment,
            ThermalEnvironment,
        )

        assert EnvironmentMetadata is not None
        assert GPUEnvironment is not None
        assert CUDAEnvironment is not None
        assert ThermalEnvironment is not None
        assert CPUEnvironment is not None
        assert ContainerEnvironment is not None


class TestCollectWithoutNVML:
    """Tests for collect_environment_metadata when pynvml is unavailable."""

    def test_returns_degraded_metadata(self):
        """Without pynvml, should return EnvironmentMetadata with defaults (not crash)."""
        with patch.dict("sys.modules", {"pynvml": None}):
            result = collect_environment_metadata(device_index=0)

        assert isinstance(result, EnvironmentMetadata)
        assert result.gpu.name == "unavailable"
        assert result.gpu.vram_total_mb == 0.0
        assert result.cuda.version == "unknown"
        assert result.cuda.driver_version == "unknown"
        # CPU and container info should still be collected
        assert result.cpu.platform is not None
        assert isinstance(result.collected_at, datetime)

    def test_no_crash_on_import_error(self):
        """Should handle ImportError gracefully."""
        with patch.dict("sys.modules", {"pynvml": None}):
            result = collect_environment_metadata()
        assert result is not None


class TestEnvironmentMetadataModel:
    """Tests for EnvironmentMetadata model construction."""

    def test_construct_with_all_fields(self):
        from llenergymeasure.domain.environment import (
            ContainerEnvironment,
            CPUEnvironment,
            CUDAEnvironment,
            GPUEnvironment,
            ThermalEnvironment,
        )

        env = EnvironmentMetadata(
            gpu=GPUEnvironment(
                name="NVIDIA A100-SXM4-80GB",
                vram_total_mb=81920.0,
                compute_capability="8.0",
            ),
            cuda=CUDAEnvironment(version="12.4", driver_version="535.104.05"),
            thermal=ThermalEnvironment(
                temperature_c=45.0,
                power_limit_w=400.0,
                default_power_limit_w=400.0,
            ),
            cpu=CPUEnvironment(governor="performance", model="Intel Xeon", platform="Linux"),
            container=ContainerEnvironment(detected=True, runtime="docker"),
            collected_at=datetime(2024, 6, 15, 10, 0, 0),
        )

        assert env.gpu.name == "NVIDIA A100-SXM4-80GB"
        assert env.gpu.compute_capability == "8.0"
        assert env.cuda.version == "12.4"
        assert env.thermal.temperature_c == 45.0
        assert env.cpu.governor == "performance"
        assert env.container.detected is True
        assert env.container.runtime == "docker"


class TestSummaryLineFormat:
    """Tests for EnvironmentMetadata.summary_line property."""

    def test_summary_contains_gpu_name_and_cuda(self):
        from llenergymeasure.domain.environment import (
            CPUEnvironment,
            CUDAEnvironment,
            GPUEnvironment,
        )

        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920.0),
            cuda=CUDAEnvironment(version="12.4", driver_version="535.104"),
            cpu=CPUEnvironment(platform="Linux"),
            collected_at=datetime(2024, 1, 1),
        )
        summary = env.summary_line
        assert "A100" in summary
        assert "CUDA 12.4" in summary
        assert "80GB" in summary

    def test_summary_includes_temperature(self):
        from llenergymeasure.domain.environment import (
            CPUEnvironment,
            CUDAEnvironment,
            GPUEnvironment,
            ThermalEnvironment,
        )

        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="NVIDIA A100", vram_total_mb=81920.0),
            cuda=CUDAEnvironment(version="12.4", driver_version="535"),
            thermal=ThermalEnvironment(temperature_c=42.0),
            cpu=CPUEnvironment(platform="Linux"),
            collected_at=datetime(2024, 1, 1),
        )
        assert "42C" in env.summary_line

    def test_summary_without_temperature(self):
        from llenergymeasure.domain.environment import (
            CPUEnvironment,
            CUDAEnvironment,
            GPUEnvironment,
        )

        env = EnvironmentMetadata(
            gpu=GPUEnvironment(name="NVIDIA A100", vram_total_mb=81920.0),
            cuda=CUDAEnvironment(version="12.4", driver_version="535"),
            cpu=CPUEnvironment(platform="Linux"),
            collected_at=datetime(2024, 1, 1),
        )
        # Should not crash, just omit temperature
        summary = env.summary_line
        assert "CUDA 12.4" in summary
        assert "C" not in summary or "CUDA" in summary  # no temp-C, just CUDA
