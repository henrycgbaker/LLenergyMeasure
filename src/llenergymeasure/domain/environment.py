"""Environment metadata models for experiment reproducibility.

Captures the hardware and software environment at experiment time,
enabling post-hoc analysis of environmental factors affecting measurements.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llenergymeasure.domain.provenance import RunnerProvenance


class GPUEnvironment(BaseModel):
    """GPU hardware information."""

    name: str = Field(..., description="GPU model name (e.g., 'NVIDIA A100-SXM4-80GB')")
    vram_total_mb: float = Field(..., description="Total VRAM in MB")
    compute_capability: str | None = Field(
        default=None,
        description="CUDA compute capability (e.g., '8.0')",
    )


class CUDAEnvironment(BaseModel):
    """CUDA driver information reported by NVML."""

    driver_supported_version: str = Field(
        ...,
        description="Maximum CUDA version the installed NVIDIA driver supports, from NVML "
        "(nvmlSystemGetCudaDriverVersion; this is the 'CUDA Version' nvidia-smi prints in its "
        "header). A driver-side capability, distinct from the runtime CUDA version the software "
        "stack was actually built against - that is EnvironmentSnapshot.cuda_version.",
    )
    driver_version: str = Field(..., description="NVIDIA driver package version string")

    @model_validator(mode="before")
    @classmethod
    def _map_legacy_cuda_version(cls, data: Any) -> Any:
        """Read a legacy (bundle 1.0) CUDA block best-effort.

        Bundle 1.0 stored this driver-supported CUDA version under the ambiguous
        key ``version``. Map it onto ``driver_supported_version`` so an older
        system snapshot loads rather than failing the now-renamed required field.
        """
        if isinstance(data, dict) and "version" in data and "driver_supported_version" not in data:
            data = dict(data)
            data["driver_supported_version"] = data.pop("version")
        return data


class ThermalEnvironment(BaseModel):
    """GPU thermal state at experiment start."""

    temperature_c: float | None = Field(
        default=None,
        description="GPU temperature at experiment start in Celsius",
    )
    power_limit_w: float | None = Field(
        default=None,
        description="Configured GPU power limit in Watts",
    )
    default_power_limit_w: float | None = Field(
        default=None,
        description="Factory default GPU power limit in Watts",
    )


class CPUEnvironment(BaseModel):
    """CPU and OS information."""

    governor: str = Field(
        default="unknown",
        description="CPU frequency governor (e.g., 'performance', 'powersave')",
    )
    model: str | None = Field(
        default=None,
        description="CPU model string",
    )
    platform: str = Field(..., description="OS platform (e.g., 'Linux')")


class ContainerEnvironment(BaseModel):
    """Container runtime detection."""

    detected: bool = Field(
        default=False,
        description="Whether running inside a container",
    )
    runtime: str | None = Field(
        default=None,
        description="Container runtime (e.g., 'docker', 'podman')",
    )


class EnvironmentMetadata(BaseModel):
    """Complete environment metadata for an experiment.

    Captures GPU, CUDA, thermal, CPU, and container information
    at experiment time for reproducibility and post-hoc analysis.
    """

    gpu: GPUEnvironment = Field(..., description="GPU hardware information")
    cuda: CUDAEnvironment = Field(..., description="CUDA driver information")
    thermal: ThermalEnvironment = Field(
        default_factory=ThermalEnvironment,
        description="GPU thermal state at experiment start",
    )
    cpu: CPUEnvironment = Field(..., description="CPU and OS information")
    container: ContainerEnvironment = Field(
        default_factory=ContainerEnvironment,
        description="Container runtime detection",
    )
    collected_at: datetime = Field(..., description="When metadata was collected")


# ---------------------------------------------------------------------------
# EnvironmentSnapshot - full software + hardware context
# ---------------------------------------------------------------------------


class EnvironmentSnapshot(BaseModel):
    """Full software+hardware environment snapshot for experiment reproducibility.

    Contains per-experiment hardware and runtime metadata. Software package
    listings (installed_packages) are study-level constants and live in the
    study-level system.json artefact instead.
    """

    hardware: EnvironmentMetadata
    python_version: str
    tool_version: str
    cuda_version: str | None = Field(
        default=None,
        description="Runtime CUDA version the software stack was built against, detected via "
        "the torch/version.txt/nvcc fallback chain (cuda_version_source records which). Distinct "
        "from the driver-supported CUDA version in hardware.cuda.driver_supported_version.",
    )
    cuda_version_source: str | None = None  # "torch" | "version_txt" | "nvcc" | None
    runner: RunnerProvenance | None = Field(
        default=None,
        description="How the experiment was executed (docker vs local, image + digest + source, "
        "the unified runner-provenance model). None for older sidecars written before runner "
        "provenance was recorded.",
    )
