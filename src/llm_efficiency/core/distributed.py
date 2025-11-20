"""
Distributed computing setup using Hugging Face Accelerate.

Handles initialization of distributed environment, process synchronization,
and unique experiment ID generation.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator

logger = logging.getLogger(__name__)


def setup_accelerator(
    mixed_precision: str = "no",
    gradient_accumulation_steps: int = 1,
    log_with: Optional[str] = None,
) -> Accelerator:
    """
    Initialize Hugging Face Accelerator for distributed execution.

    Args:
        mixed_precision: Mixed precision training mode ("no", "fp16", "bf16")
        gradient_accumulation_steps: Number of gradient accumulation steps
        log_with: Logging integration (e.g., "tensorboard", "wandb")

    Returns:
        Initialized Accelerator instance
    """
    logger.info("Initializing Accelerator...")

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with=log_with,
    )

    logger.info("Accelerator initialized")
    logger.info(f"  Process index: {accelerator.process_index}")
    logger.info(f"  Total processes: {accelerator.num_processes}")
    logger.info(f"  Device: {accelerator.device}")
    logger.info(f"  Mixed precision: {mixed_precision}")
    logger.info(f"  Is main process: {accelerator.is_main_process}")

    return accelerator


def generate_experiment_id(
    accelerator: Accelerator,
    id_file: Path = Path("persistent_progress_trackers/experiment_id.txt"),
) -> str:
    """
    Generate unique experiment ID, synchronized across all processes.

    Only the main process reads/writes the ID file, then broadcasts
    to all other processes to ensure consistency.

    Args:
        accelerator: Accelerator instance
        id_file: Path to experiment ID counter file

    Returns:
        Experiment ID as zero-padded string (e.g., "0001", "4460")
    """
    # Only main process manages the ID file
    if accelerator.is_main_process:
        # Ensure directory exists
        id_file.parent.mkdir(parents=True, exist_ok=True)

        # Read current ID or initialize
        if id_file.exists():
            try:
                with open(id_file) as f:
                    current_id = int(f.read().strip())
            except (ValueError, IOError) as e:
                logger.warning(f"Failed to read experiment ID, starting from 1: {e}")
                current_id = 0
        else:
            logger.info("No experiment ID file found, starting from 1")
            current_id = 0

        # Increment and save
        next_id = current_id + 1
        with open(id_file, "w") as f:
            f.write(str(next_id))

        logger.info(f"Generated experiment ID: {next_id:04d}")

        # Prepare for broadcast (need list for gather)
        id_list = [next_id]
    else:
        # Non-main processes wait for broadcast
        id_list = [None]

    # Broadcast ID to all processes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(id_list, src=0)

    experiment_id = f"{id_list[0]:04d}"

    if not accelerator.is_main_process:
        logger.debug(f"Received experiment ID: {experiment_id}")

    return experiment_id


def synchronize_processes(accelerator: Accelerator, message: str = "") -> None:
    """
    Synchronize all processes at a barrier point.

    Args:
        accelerator: Accelerator instance
        message: Optional message for logging
    """
    if message:
        logger.debug(f"Synchronization point: {message}")

    accelerator.wait_for_everyone()

    if message:
        logger.debug(f"All processes synchronized: {message}")


def is_main_process(accelerator: Accelerator) -> bool:
    """
    Check if current process is the main process.

    Args:
        accelerator: Accelerator instance

    Returns:
        True if main process (rank 0), False otherwise
    """
    return accelerator.is_main_process


def get_process_info(accelerator: Accelerator) -> dict:
    """
    Get information about current process in distributed setup.

    Args:
        accelerator: Accelerator instance

    Returns:
        Dictionary with process information
    """
    return {
        "process_index": accelerator.process_index,
        "local_process_index": accelerator.local_process_index,
        "num_processes": accelerator.num_processes,
        "device": str(accelerator.device),
        "is_main_process": accelerator.is_main_process,
        "is_local_main_process": accelerator.is_local_main_process,
        "distributed_type": str(accelerator.distributed_type),
        "mixed_precision": str(accelerator.mixed_precision),
    }


def cleanup_distributed(accelerator: Accelerator) -> None:
    """
    Clean up distributed resources.

    Args:
        accelerator: Accelerator instance
    """
    logger.info("Cleaning up distributed resources...")

    # Wait for all processes
    accelerator.wait_for_everyone()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache")

    # Free accelerator
    accelerator.free_memory()

    logger.info("Distributed cleanup complete")
