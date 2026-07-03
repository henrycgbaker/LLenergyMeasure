"""GPU device detection helpers.

This module provides NVML lifecycle management, GPU index resolution for
experiments, and GPU architecture / compute-capability detection.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig

from llenergymeasure.config.ssot import Engine

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
        yield  # pynvml absent or nvmlInit failed - caller proceeds without NVML


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

    Rules per engine:
    - **vLLM**: tensor_parallel_size * pipeline_parallel_size GPUs. Both are known
      from config before the harness runs, so gpu_indices = list(range(total)).
    - **Transformers**: device_map="auto" (or any non-None device_map) monitors all
      NVML-visible GPUs. Model sharding is determined at load time inside
      harness.run(), but gpu_indices must be passed *before* load. Using all
      visible GPUs is correct and safe.
    - **TensorRT-LLM**: tensor_parallel_size GPUs. Known from config before harness runs.
    - **Otherwise**: [0] (single-GPU default, backward compatible).

    Note: num_processes > 1 (data parallelism via Accelerate) is not handled here.
    For local runs this path is not yet implemented; for Docker each subprocess calls
    the harness independently.
    """
    if config.engine == Engine.VLLM and config.vllm is not None:
        tp = 1
        pp = 1
        engine_params = config.active_engine_params()
        if engine_params is not None:
            tp = engine_params.tensor_parallel_size or 1
            pp = engine_params.pipeline_parallel_size or 1
        total = tp * pp
        if total > 1:
            return list(range(total))
    elif config.engine == Engine.TENSORRT and config.tensorrt is not None:
        engine_params = config.active_engine_params()
        tp = (engine_params.tensor_parallel_size or 1) if engine_params is not None else 1
        if tp > 1:
            return list(range(tp))
    elif (
        config.engine == Engine.TRANSFORMERS
        and config.transformers is not None
        and getattr(config.active_engine_params(), "device_map", None) is not None
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
