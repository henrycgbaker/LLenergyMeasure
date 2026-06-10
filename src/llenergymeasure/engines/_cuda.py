"""CUDA memory and warmup helpers for inference engine implementations.

Extracted from the repeated patterns in transformers.py, vllm.py, and
tensorrt.py to reduce duplication while keeping engines thin.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

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

    Returns 0.0 if torch/CUDA is unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)  # type: ignore[no-any-return]
    except Exception:
        pass
    return 0.0


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
