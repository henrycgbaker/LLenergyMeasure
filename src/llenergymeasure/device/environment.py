"""Environment metadata collection for experiment reproducibility.

Collects GPU, CUDA, driver, thermal, CPU, and container information via NVML.
Gracefully degrades when NVML is unavailable - returns EnvironmentMetadata
with reasonable defaults instead of crashing.
"""

import importlib.util
import logging
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from llenergymeasure.device.gpu_info import nvml_context
from llenergymeasure.domain.environment import (
    ContainerEnvironment,
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    GPUEnvironment,
    ThermalEnvironment,
)
from llenergymeasure.utils.formatting import bytes_to_mb

logger = logging.getLogger(__name__)


def collect_environment_metadata(device_index: int = 0) -> EnvironmentMetadata:
    """Collect full environment metadata for an experiment.

    Queries NVML for GPU, CUDA, driver, and thermal information. Falls back
    to reasonable defaults when NVML or specific queries are unavailable.

    Args:
        device_index: CUDA device index to query.

    Returns:
        EnvironmentMetadata with all available hardware/software info.
    """
    if importlib.util.find_spec("pynvml") is None:
        logger.debug("Environment: pynvml not available, returning defaults")
        return _unavailable_metadata()

    import pynvml

    result: EnvironmentMetadata | None = None

    with nvml_context():
        logger.debug("Environment: NVML initialised")
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:
            logger.debug("Environment: failed to get device handle: %s", e)
            return _unavailable_metadata()

        try:
            gpu = _collect_gpu(pynvml, handle)
            cuda = _collect_cuda(pynvml)
            thermal = _collect_thermal(pynvml, handle)
            cpu = _collect_cpu()
            container = _collect_container()

            result = EnvironmentMetadata(
                gpu=gpu,
                cuda=cuda,
                thermal=thermal,
                cpu=cpu,
                container=container,
                collected_at=datetime.now(),
            )
        except Exception as e:
            logger.debug("Environment: collection failed: %s", e)

    if result is not None:
        return result
    return _unavailable_metadata()


def _collect_gpu(pynvml: Any, handle: Any) -> GPUEnvironment:
    """Collect GPU hardware information."""
    name = "unknown"
    vram_total_mb = 0.0
    compute_capability = None

    try:
        raw_name = pynvml.nvmlDeviceGetName(handle)
        name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        logger.debug("Environment: GPU name = %s", name)
    except pynvml.NVMLError as e:
        logger.debug("Environment: failed to get GPU name: %s", e)

    try:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        vram_total_mb = bytes_to_mb(mem_info.total)
        logger.debug("Environment: VRAM = %.0f MB", vram_total_mb)
    except pynvml.NVMLError as e:
        logger.debug("Environment: failed to get memory info: %s", e)

    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_capability = f"{major}.{minor}"
        logger.debug("Environment: compute capability = %s", compute_capability)
    except (pynvml.NVMLError, AttributeError) as e:
        logger.debug("Environment: failed to get compute capability: %s", e)

    return GPUEnvironment(
        name=name,
        vram_total_mb=vram_total_mb,
        compute_capability=compute_capability,
    )


def _collect_cuda(pynvml: Any) -> CUDAEnvironment:
    """Collect CUDA driver information reported by NVML."""
    driver_supported_version = "unknown"
    driver_version = "unknown"

    try:
        raw_driver = pynvml.nvmlSystemGetDriverVersion()
        driver_version = (
            raw_driver.decode("utf-8") if isinstance(raw_driver, bytes) else str(raw_driver)
        )
        logger.debug("Environment: driver version = %s", driver_version)
    except pynvml.NVMLError as e:
        logger.debug("Environment: failed to get driver version: %s", e)

    try:
        cuda_driver_version = pynvml.nvmlSystemGetCudaDriverVersion()
        major = cuda_driver_version // 1000
        minor = (cuda_driver_version % 1000) // 10
        driver_supported_version = f"{major}.{minor}"
        logger.debug("Environment: driver-supported CUDA version = %s", driver_supported_version)
    except (pynvml.NVMLError, AttributeError) as e:
        logger.debug("Environment: failed to get driver-supported CUDA version: %s", e)

    return CUDAEnvironment(
        driver_supported_version=driver_supported_version,
        driver_version=driver_version,
    )


def _collect_thermal(pynvml: Any, handle: Any) -> ThermalEnvironment:
    """Collect GPU thermal state."""
    temperature_c: float | None = None
    power_limit_w: float | None = None
    default_power_limit_w: float | None = None

    try:
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        temperature_c = float(temp)
        logger.debug("Environment: temperature = %.0fC", temperature_c)
    except pynvml.NVMLError as e:
        logger.debug("Environment: failed to get temperature: %s", e)

    try:
        limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
        power_limit_w = limit_mw / 1000.0
        logger.debug("Environment: power limit = %.0fW", power_limit_w)
    except pynvml.NVMLError as e:
        logger.debug("Environment: failed to get power limit: %s", e)

    try:
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        default_power_limit_w = default_mw / 1000.0
        logger.debug("Environment: default power limit = %.0fW", default_power_limit_w)
    except pynvml.NVMLError as e:
        logger.debug("Environment: failed to get default power limit: %s", e)

    return ThermalEnvironment(
        temperature_c=temperature_c,
        power_limit_w=power_limit_w,
        default_power_limit_w=default_power_limit_w,
    )


def _collect_cpu() -> CPUEnvironment:
    """Collect CPU and OS information."""
    governor = "unknown"

    # Read CPU governor (Linux only)
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        governor = governor_path.read_text().strip()
        logger.debug("Environment: CPU governor = %s", governor)
    except (FileNotFoundError, PermissionError, OSError):
        logger.debug("Environment: CPU governor not available")

    cpu_model = platform.processor() or None
    logger.debug("Environment: CPU model = %s", cpu_model)

    return CPUEnvironment(
        governor=governor,
        model=cpu_model,
        platform=platform.system(),
    )


def _collect_container() -> ContainerEnvironment:
    """Detect container runtime."""
    detected = Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()
    runtime: str | None = None

    if detected:
        runtime = _detect_container_runtime()
        logger.debug("Environment: container detected, runtime = %s", runtime)
    else:
        # Also check cgroup for container detection
        try:
            cgroup_content = Path("/proc/1/cgroup").read_text()
            if "docker" in cgroup_content or "containerd" in cgroup_content:
                detected = True
                runtime = "docker"
            elif "lxc" in cgroup_content:
                detected = True
                runtime = "lxc"
            logger.debug("Environment: cgroup container check = %s", detected)
        except (FileNotFoundError, PermissionError, OSError):
            pass

    return ContainerEnvironment(
        detected=detected,
        runtime=runtime,
    )


def _detect_container_runtime() -> str | None:
    """Detect which container runtime is in use."""
    if Path("/.dockerenv").exists():
        return "docker"
    if Path("/run/.containerenv").exists():
        return "podman"
    return None


def _unavailable_metadata() -> EnvironmentMetadata:
    """Create metadata with reasonable defaults when NVML unavailable."""
    return EnvironmentMetadata(
        gpu=GPUEnvironment(name="unavailable", vram_total_mb=0.0),
        cuda=CUDAEnvironment(driver_supported_version="unknown", driver_version="unknown"),
        thermal=ThermalEnvironment(),
        cpu=_collect_cpu(),
        container=_collect_container(),
        collected_at=datetime.now(),
    )


# ---------------------------------------------------------------------------
# CUDA version detection - multi-source fallback chain
# ---------------------------------------------------------------------------


def detect_cuda_version_with_source() -> tuple[str | None, str | None]:
    """Detect the CUDA version using a fallback chain.

    Returns:
        Tuple of (version_string, source_name) where source_name is one of:
        "torch", "version_txt", "nvcc", or None if detection failed.
    """
    # Source 1: torch.version.cuda
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            cuda_ver = torch.version.cuda
            if cuda_ver:
                return cuda_ver, "torch"
        except Exception:
            logger.debug("CUDA version: torch source failed", exc_info=True)

    # Source 2: /usr/local/cuda/version.txt or version.json
    for version_file in (
        "/usr/local/cuda/version.txt",
        "/usr/local/cuda/version.json",
    ):
        try:
            with open(version_file) as f:
                content = f.read()
            match = re.search(r"(\d+\.\d+)", content)
            if match:
                return match.group(1), "version_txt"
        except Exception:
            pass

    # Source 3: nvcc --version
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        match = re.search(r"release (\d+\.\d+)", result.stdout)
        if match:
            return match.group(1), "nvcc"
    except Exception:
        logger.debug("CUDA version: nvcc source failed", exc_info=True)

    # Source 4: Give up
    return None, None
