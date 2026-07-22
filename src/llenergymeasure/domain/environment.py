"""Environment metadata models for experiment reproducibility.

Captures the hardware and software environment at experiment time,
enabling post-hoc analysis of environmental factors affecting measurements.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class GPUEnvironment(BaseModel):
    """GPU hardware information."""

    name: str = Field(..., description="GPU model name (e.g., 'NVIDIA A100-SXM4-80GB')")
    vram_total_mb: float = Field(..., description="Total VRAM in MB")
    compute_capability: str | None = Field(
        default=None,
        description="CUDA compute capability (e.g., '8.0')",
    )
    pcie_gen: int | None = Field(
        default=None,
        description="PCIe generation",
    )
    mig_enabled: bool = Field(
        default=False,
        description="Whether MIG (Multi-Instance GPU) is enabled",
    )


class CUDAEnvironment(BaseModel):
    """CUDA runtime information."""

    version: str = Field(..., description="CUDA version (e.g., '12.4')")
    driver_version: str = Field(..., description="NVIDIA driver version string")
    cudnn_version: str | None = Field(
        default=None,
        description="cuDNN version string",
    )


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
    fan_speed_pct: float | None = Field(
        default=None,
        description="Fan speed as percentage (0-100)",
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


class RunnerEnvironment(BaseModel):
    """How an experiment was executed - containerized (docker) or on the host (local).

    Records the runner mode (docker vs local), the exact Docker image and its
    resolved registry digest (the reproducibility anchor pinning the full
    software stack: base image, CUDA, torch, patches), and the precedence
    source that selected the runner. The digest is None for local runs, and
    also None when it cannot be resolved (image built locally with no registry
    digest, docker unavailable, inspect error) - resolution is best-effort and
    never fails a run.

    Sibling of ``experiment.RunnerProvenance`` (which persists into result.json):
    both mirror the config-layer ``RunnerSpec``'s mode/image/source. They stay separate
    because their extra fields diverge - this one carries ``image_digest`` (the
    environment.json reproducibility anchor), RunnerProvenance carries
    ``image_source`` (result.json image-resolution provenance).
    """

    mode: Literal["docker", "local"] = Field(
        ..., description="Execution mode - 'docker' (containerized) or 'local' (host process)"
    )
    image: str | None = Field(
        default=None,
        description="Docker image reference used (None for local runs)",
    )
    image_digest: str | None = Field(
        default=None,
        description="Resolved image registry digest ('repo@sha256:...'). None for local "
        "runs or when the digest could not be resolved (e.g. locally-built image).",
    )
    source: str = Field(
        ...,
        description="RunnerSpec precedence source that selected the runner (e.g. 'env', "
        "'yaml', 'user_config', 'auto_detected', 'default', 'multi_engine_elevation', 'local')",
    )


class EnvironmentMetadata(BaseModel):
    """Complete environment metadata for an experiment.

    Captures GPU, CUDA, thermal, CPU, and container information
    at experiment time for reproducibility and post-hoc analysis.
    """

    gpu: GPUEnvironment = Field(..., description="GPU hardware information")
    cuda: CUDAEnvironment = Field(..., description="CUDA runtime information")
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
    study-level environment.json artefact instead.
    """

    hardware: EnvironmentMetadata
    python_version: str
    tool_version: str
    cuda_version: str | None = None
    cuda_version_source: str | None = None  # "torch" | "version_txt" | "nvcc" | None
    runner: RunnerEnvironment | None = Field(
        default=None,
        description="How the experiment was executed (docker vs local, image + digest, "
        "precedence source). None for older sidecars written before runner provenance "
        "was recorded.",
    )
