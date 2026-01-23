"""Experiment lifecycle management.

This module provides setup and teardown operations for experiments,
including CUDA cache management and distributed process group cleanup.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from accelerate import Accelerator


def cleanup_cuda() -> None:
    """Clean up CUDA resources.

    Empties the CUDA cache, resets peak memory stats, and collects IPC handles.
    Safe to call even when CUDA is not available.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.ipc_collect()
            logger.debug("CUDA resources cleaned up")
    except Exception as e:
        logger.warning(f"Error during CUDA cleanup: {e}")


def cleanup_distributed() -> None:
    """Clean up distributed process group.

    Destroys the process group if one is initialized.
    Safe to call even when distributed is not available or initialized.
    """
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            logger.debug("Distributed process group destroyed")
    except Exception as e:
        logger.warning(f"Error during distributed cleanup: {e}")


def full_cleanup() -> None:
    """Perform complete cleanup of all resources.

    Cleans up in order: distributed, CUDA, garbage collection.
    """
    cleanup_distributed()
    cleanup_cuda()
    gc.collect()
    logger.info("Full cleanup completed")


@contextmanager
def experiment_lifecycle(accelerator: Accelerator | None = None) -> Iterator[None]:
    """Context manager for experiment lifecycle.

    Ensures proper cleanup of resources regardless of whether
    the experiment succeeds or fails.

    Args:
        accelerator: Optional Accelerator instance for coordinated cleanup.

    Yields:
        None

    Example:
        >>> with experiment_lifecycle(accelerator):
        ...     # Run experiment
        ...     result = model(input_ids)
    """
    try:
        yield
    finally:
        if accelerator is not None:
            # Coordinate cleanup across processes
            try:
                if accelerator.is_main_process:
                    logger.info("Main process initiating cleanup")
                full_cleanup()
            except Exception as e:
                logger.error(f"Error during lifecycle cleanup: {e}")
        else:
            full_cleanup()


def ensure_clean_start() -> None:
    """Ensure a clean starting state for a new experiment.

    Destroys any existing distributed process group and clears CUDA cache.
    Call this before starting a new experiment run.
    """
    if dist.is_available() and dist.is_initialized():
        try:
            dist.destroy_process_group()
            logger.debug("Previous process group destroyed")
        except Exception as e:
            logger.warning(f"Could not destroy previous process group: {e}")

    cleanup_cuda()
    logger.debug("Clean start ensured")


def warmup_model(
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    num_runs: int = 3,
    max_length: int = 128,
) -> None:
    """Perform warmup runs on the model.

    Runs a few inference passes to ensure the model is properly loaded
    and CUDA kernels are compiled.

    Args:
        model: The model to warm up.
        tokenizer: Tokenizer for creating dummy input.
        device: Device to run on.
        num_runs: Number of warmup iterations.
        max_length: Max length for dummy input.
    """
    dummy_text = "Hello world, this is a warmup test."

    try:
        dummy_input = tokenizer(  # type: ignore[operator]
            dummy_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).input_ids.to(device)

        with torch.no_grad():
            for i in range(num_runs):
                _ = model(dummy_input)
                logger.debug(f"Warmup run {i + 1}/{num_runs} complete")

        logger.info(f"Model warmup complete ({num_runs} runs)")

    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
