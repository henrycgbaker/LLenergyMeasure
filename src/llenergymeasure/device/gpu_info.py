"""GPU device detection helpers.

This module provides NVML lifecycle management, GPU index resolution for
experiments, translation between the physical and CUDA-visible index spaces, and
GPU architecture / compute-capability detection.

Index spaces, since two of them meet here: NVML addresses PHYSICAL device
indices and ignores ``CUDA_VISIBLE_DEVICES`` completely, while CUDA (and Zeus,
which builds on it) addresses LOGICAL indices into the visible set. llem's
monitoring indices are physical throughout - see :func:`_resolve_gpu_indices` -
and :func:`to_cuda_logical_indices` is the one translation into the other space.
"""

from __future__ import annotations

import logging
import os
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


def host_gpu_count() -> int | None:
    """Return the number of NVML-visible devices, or None when NVML cannot answer.

    Fail-soft by design: ``None`` means "unknown", not "zero". pynvml absent, no
    driver, or a remote Docker daemon (where the GPUs live on another host) all
    yield ``None``, so callers must treat it as "cannot check" rather than as a
    device count of nought.
    """
    try:
        import pynvml

        with nvml_context():
            return int(pynvml.nvmlDeviceGetCount())
    except Exception:
        logger.debug("NVML device count unavailable", exc_info=True)
        return None


def cuda_visible_physical_order() -> list[int] | None:
    """Return the physical device indices ``CUDA_VISIBLE_DEVICES`` makes visible.

    The returned list is in VISIBILITY order, so its positions are exactly the
    logical indices CUDA and Zeus address: ``CUDA_VISIBLE_DEVICES=3,1`` yields
    ``[3, 1]``, meaning logical 0 is physical 3 and logical 1 is physical 1.

    ``None`` when the variable is unset, empty, or names devices by UUID -
    i.e. whenever the physical-to-logical mapping is not integer-derivable here.
    An unset variable means logical == physical, which is the case every caller
    already handles as the identity mapping.
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        return None
    tokens = [tok.strip() for tok in raw.split(",") if tok.strip()]
    try:
        return [int(tok) for tok in tokens]
    except ValueError:
        return None  # GPU-<uuid> / MIG-<uuid> form: not integer-mappable.


def to_cuda_logical_indices(physical_indices: list[int]) -> list[int]:
    """Translate physical device indices into the CUDA-visible logical space.

    llem's monitoring indices are PHYSICAL device indices, because NVML addresses
    physical devices and ignores ``CUDA_VISIBLE_DEVICES`` entirely. Two consumers
    need the other space: torch (``torch.cuda.*(device=...)``) and Zeus
    (``ZeusMonitor(gpu_indices=...)``) both index into the visible set. Under a
    restricting ``CUDA_VISIBLE_DEVICES`` the two spaces differ, and handing a
    physical index to either addresses the wrong device or raises.

    With ``CUDA_VISIBLE_DEVICES=2,3``, physical ``[2, 3]`` becomes logical
    ``[0, 1]``. A physical index that is not visible at all is dropped: it has no
    logical counterpart, and asking torch about it is an error. Identity when
    ``CUDA_VISIBLE_DEVICES`` is unset or UUID-valued (nothing to translate).
    """
    visible = cuda_visible_physical_order()
    if visible is None:
        return list(physical_indices)
    return [visible.index(i) for i in physical_indices if i in visible]


def _resolve_gpu_indices(
    config: ExperimentConfig,
    allowed_gpu_indices: list[int] | None = None,
) -> list[int]:
    """Determine the PHYSICAL GPU indices to monitor for an experiment.

    Generic over each engine's ``ParallelismModel`` (from ``ssot.ENGINES``):

    - **multiply_fields** (vLLM ``tensor_parallel_size * pipeline_parallel_size``,
      TensorRT-LLM ``tensor_parallel_size``): the product of the named config
      fields (each defaulting to 1) is known before the harness runs, so
      gpu_indices = the first ``total`` allowed devices.
    - **all_visible_field** (transformers ``device_map``): a non-None value means
      the model shards across all visible GPUs. Sharding is decided at load
      time inside harness.run(), but gpu_indices must be passed *before* load, so
      measuring every device llem may use is correct and safe.
    - **Neither / no engine section**: one device (single-GPU default, backward
      compatible).

    ``allowed_gpu_indices`` is the physical device set llem may use (the resolved
    ``study_execution.gpu_indices``). When given, every branch draws from it
    instead of counting from zero, so the sampled devices are exactly the devices
    compute was placed on. When None, the historical behaviour stands: indices
    count from 0 and the census branch takes every NVML-visible device.

    Callers that run INSIDE a scoped container must pass None. There, docker has
    already restricted the device set and both CUDA and NVML re-enumerate from 0,
    so the in-container indices are already the right ones and applying a host
    allowlist on top would address devices the container cannot see.

    Note: num_processes > 1 (data parallelism via Accelerate) is not handled here.
    For local runs this path is not yet implemented; for Docker each subprocess calls
    the harness independently.
    """
    allowed = list(allowed_gpu_indices) if allowed_gpu_indices else None
    first = [allowed[0]] if allowed else [0]

    try:
        engine = Engine(config.engine)
    except ValueError:
        return first  # Unrecognised engine - single-GPU default (backward compatible).
    section = getattr(config, engine.value, None)
    engine_params = getattr(section, "engine_params", None) if section is not None else None
    if engine_params is None:
        return first

    parallelism = ENGINES[engine].parallelism
    if parallelism.multiply_fields:
        total = 1
        for field in parallelism.multiply_fields:
            total *= getattr(engine_params, field, None) or 1
        if total > 1:
            if allowed is None:
                return list(range(total))
            if total > len(allowed):
                logger.warning(
                    "Engine parallelism requests %d GPUs but only %s are allowed on this "
                    "machine; monitoring the allowed devices only. The engine itself will "
                    "fail to place %d ranks.",
                    total,
                    allowed,
                    total,
                )
            return allowed[:total]
    elif (
        parallelism.all_visible_field is not None
        and getattr(engine_params, parallelism.all_visible_field, None) is not None
    ):
        # Model will shard across all visible GPUs - measure all of them. Under an
        # allowlist "all visible" IS the allowed set, so no NVML census is needed.
        if allowed is not None:
            return allowed
        # Best-effort: if pynvml is absent or no NVIDIA GPU, fall through to [0].
        count = host_gpu_count()
        if count is not None and count > 1:
            return list(range(count))
    return first
