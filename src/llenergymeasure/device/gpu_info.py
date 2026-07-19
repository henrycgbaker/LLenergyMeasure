"""GPU device detection helpers.

This module provides NVML lifecycle management, GPU index resolution for
experiments, and GPU architecture / compute-capability detection.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig

from llenergymeasure.config.ssot import ENGINES, Engine

logger = logging.getLogger(__name__)


@contextmanager
def nvml_context() -> Generator[None, None, None]:
    """Context manager for NVML init/shutdown lifecycle.

    Best-effort: silently ignores ImportError (pynvml not installed) and
    NVMLError (no NVIDIA GPU). Callers receive None on failure - handle gracefully.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            yield
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        # Fail-soft by design: caller proceeds without NVML. Leave a debug-level
        # trace so a real driver/permission failure is diagnosable, without
        # changing the swallow-and-continue behaviour.
        logger.debug("NVML unavailable; proceeding without it", exc_info=True)
        yield  # pynvml absent or nvmlInit failed - caller proceeds without NVML


def gpu_inventory() -> tuple[list[dict[str, Any]], str | None]:
    """Return ``(per-device info, driver version)`` from one NVML session.

    Each device dict carries ``name`` (str) and ``vram_gb`` (float). The list is
    empty when no NVIDIA GPU is visible; the driver string is None when it cannot
    be read. Best-effort: pynvml/NVML absent or failing yields ``([], None)`` and
    never raises. Opening a single ``nvml_context()`` covers both queries, so
    diagnostics (e.g. ``llem doctor``) do not pay two init/shutdown cycles.
    """
    gpus: list[dict[str, Any]] = []
    driver: str | None = None
    try:
        import pynvml

        with nvml_context():
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({"name": name, "vram_gb": mem.total / 1e9})
            raw = pynvml.nvmlSystemGetDriverVersion()
            driver = raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        return [], None
    return gpus, driver


def get_compute_capability(gpu_index: int = 0) -> tuple[int, int] | None:
    """Return (major, minor) SM version via pynvml, or None on failure.

    Args:
        gpu_index: NVML device index (default: 0).

    Returns:
        Tuple of (major, minor) SM version, e.g. (8, 0) for A100.
        Returns None if pynvml is unavailable or query fails.
    """
    try:
        import pynvml

        with nvml_context():
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            return (major, minor)
    except Exception:
        return None


def get_gpu_architecture(device_index: int = 0) -> str:
    """Get GPU compute capability string (e.g., "sm_80" for A100).

    This is the SSOT for GPU architecture detection. Use this instead of
    duplicating torch.cuda.get_device_properties() calls.

    Args:
        device_index: CUDA device index (default: 0).

    Returns:
        Architecture string like "sm_80" (A100) or "sm_89" (L40).
        Returns "unknown" if detection fails.
    """
    try:
        import torch

        if torch.cuda.is_available() and device_index < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(device_index)
            return f"sm_{props.major}{props.minor}"
    except Exception as e:
        logger.debug("Failed to get GPU architecture for device %d: %s", device_index, e)

    return "unknown"


def _resolve_gpu_indices(config: ExperimentConfig) -> list[int]:
    """Determine GPU indices to monitor for an experiment.

    Generic over each engine's ``ParallelismModel`` (from ``ssot.ENGINES``):

    - **multiply_fields** (vLLM ``tensor_parallel_size * pipeline_parallel_size``,
      TensorRT-LLM ``tensor_parallel_size``): the product of the named config
      fields (each defaulting to 1) is known before the harness runs, so
      gpu_indices = list(range(total)).
    - **all_visible_field** (transformers ``device_map``): a non-None value means
      the model shards across all NVML-visible GPUs. Sharding is decided at load
      time inside harness.run(), but gpu_indices must be passed *before* load, so
      measuring all visible GPUs is correct and safe.
    - **Neither / no engine section**: [0] (single-GPU default, backward compatible).

    Note: num_processes > 1 (data parallelism via Accelerate) is not handled here.
    For local runs this path is not yet implemented; for Docker each subprocess calls
    the harness independently.
    """
    engine = Engine(config.engine)
    section = getattr(config, engine.value, None)
    engine_params = getattr(section, "engine_params", None) if section is not None else None
    if engine_params is None:
        return [0]

    parallelism = ENGINES[engine].parallelism
    if parallelism.multiply_fields:
        total = 1
        for field in parallelism.multiply_fields:
            total *= getattr(engine_params, field, None) or 1
        if total > 1:
            return list(range(total))
    elif (
        parallelism.all_visible_field is not None
        and getattr(engine_params, parallelism.all_visible_field, None) is not None
    ):
        # Model will shard across all visible GPUs - measure all of them.
        # Best-effort: if pynvml is absent or no NVIDIA GPU, fall through to [0].
        try:
            import pynvml

            with nvml_context():
                count = pynvml.nvmlDeviceGetCount()
            if count > 1:
                return list(range(count))
        except Exception:
            pass  # pynvml absent or no NVIDIA GPU - fall through to [0]
    return [0]
