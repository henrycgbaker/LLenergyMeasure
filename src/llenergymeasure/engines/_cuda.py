"""CUDA memory and warmup helpers for inference engine implementations.

Extracted from the repeated patterns in transformers.py, vllm.py, and
tensorrt.py to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

from llenergymeasure.utils.formatting import bytes_to_mb

logger = logging.getLogger(__name__)


def reset_cuda_peak_memory() -> None:
    """Reset CUDA peak memory stats before a measurement window.

    Best-effort - silently ignores failures (e.g. no CUDA, no torch).
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def get_cuda_peak_memory_mb() -> float:
    """Return peak GPU memory allocated in MB since last reset.

    torch's allocator tracks only THIS process. Returns 0.0 if torch/CUDA is
    unavailable, and also - silently - if the model runs out-of-process (vLLM
    V1's EngineCore, TRT-LLM's executor): the allocator in the calling process
    never sees the child's allocation. Out-of-process engines must treat a 0.0
    reading as "not measured here" and fall back to
    :func:`get_nvml_device_memory_mb`.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return bytes_to_mb(torch.cuda.max_memory_allocated())
    except Exception:
        pass
    return 0.0


def get_nvml_device_memory_mb(gpu_indices: list[int] | None = None) -> float | None:
    """Return whole-device used GPU memory in MB, max across the given indices.

    Process-agnostic fallback for out-of-process engines. vLLM V1 runs the model
    in its EngineCore child process and TRT-LLM in its executor process, so
    torch's per-process allocator in the driver process reports 0 and
    :func:`get_cuda_peak_memory_mb` / the harness torch baseline read a silent
    0.0. NVML reports device-level used memory, which counts the child's
    allocation.

    CAVEAT - whole-device reading: NVML ``used`` is the entire device's occupied
    memory: this process's CUDA context PLUS any other tenants on the device, not
    just the model under measurement. It is therefore an upper bound on the
    model's usage. Under ``LLEM_DOCKER_GPUS`` pinning the container sees only the
    pinned device(s) - the nvidia container runtime restricts NVML visibility too
    (not only CUDA), so indices ``0..N`` map to the experiment's own GPUs and
    contamination is bounded to co-tenants of those pinned devices. The
    peak/model deltas the harness derives (``inference_memory_mb``) cancel the
    shared context term, so relative figures stay meaningful even though the
    absolute values carry the whole-device overhead.

    Args:
        gpu_indices: NVML device indices to poll (defaults to ``[0]``). The max
            ``used`` across them is returned, matching the torch convention of
            peaking across tensor-parallel ranks.

    Returns:
        Used memory in MB, or ``None`` when NVML is unavailable, no GPU is
        present, or any query fails. Callers must treat ``None`` as "not
        measured" (null) - never coerce it to 0.0.
    """
    indices = gpu_indices if gpu_indices else [0]
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        with nvml_context():
            used_mb = [
                bytes_to_mb(float(pynvml.nvmlDeviceGetMemoryInfo(handle).used))
                for handle in (pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in indices)
            ]
        return max(used_mb) if used_mb else None
    except Exception:
        return None


def cleanup_model(model_obj: Any, *, use_gc: bool = True) -> None:
    """Release a model object from GPU memory and clear CUDA cache.

    Args:
        model_obj: The model object to delete (e.g. HF model, vLLM LLM, TRT-LLM LLM).
        use_gc: Whether to run gc.collect() after deletion. PyTorch/Transformers
            skips this (deterministic refcount cleanup); vLLM and TRT-LLM need it
            to break circular references in their engine internals.
    """
    del model_obj
    if use_gc:
        gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
    except Exception:
        logger.debug("CUDA cleanup failed", exc_info=True)


def warmup_single_token(
    llm: Any,
    prompts: list[str],
    sampling_params_cls: type,
    **sp_kwargs: Any,
) -> None:
    """Run a minimal single-token warmup generation.

    Used by vLLM and TRT-LLM engines to warm up the engine before the
    measurement window. Takes the first prompt and generates 1 token.

    Args:
        llm: The LLM engine object (vllm.LLM or tensorrt_llm.LLM).
        prompts: List of prompts (only the first is used).
        sampling_params_cls: The SamplingParams class to instantiate.
        **sp_kwargs: Keyword arguments for SamplingParams constructor.
            Defaults to temperature=0.0 if no kwargs provided.
    """
    if not sp_kwargs:
        sp_kwargs = {"temperature": 0.0}
    warmup_params = sampling_params_cls(**sp_kwargs)
    llm.generate(prompts[:1], warmup_params)
