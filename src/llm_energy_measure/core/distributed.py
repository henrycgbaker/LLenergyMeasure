"""Distributed computing utilities for multi-GPU experiments."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from loguru import logger

if TYPE_CHECKING:
    from accelerate import Accelerator

# Default timeout for synchronization barriers
DEFAULT_BARRIER_TIMEOUT = 600


def get_accelerator(
    gpu_list: list[int] | None = None,
    num_processes: int | None = None,
) -> Accelerator:
    """Create and configure an Accelerator for distributed training.

    Args:
        gpu_list: List of GPU indices to use. If None, uses all available.
        num_processes: Number of processes to launch. Defaults to len(gpu_list).

    Returns:
        Configured Accelerator instance.

    Raises:
        ConfigurationError: If GPU configuration is invalid.
    """
    # Disable CLI config to let script settings take priority
    os.environ["ACCELERATE_CONFIG_FILE"] = ""

    if gpu_list is not None:
        # Only set CUDA_VISIBLE_DEVICES if not already set to MIG/GPU UUIDs
        existing_cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if "MIG-" not in existing_cuda_env and "GPU-" not in existing_cuda_env:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_list)

        if num_processes is not None:
            available = len(gpu_list)
            if num_processes > available:
                logger.warning(
                    f"num_processes ({num_processes}) exceeds available GPUs ({available}). "
                    f"Using {available} processes instead."
                )
                num_processes = available
            os.environ["ACCELERATE_NUM_PROCESSES"] = str(num_processes)

    # Import AFTER setting environment variables
    from accelerate import Accelerator

    accelerator = Accelerator(device_placement=True)
    logger.info(
        f"Accelerator initialized: device={accelerator.device}, "
        f"num_processes={accelerator.num_processes}"
    )
    return accelerator


def get_persistent_unique_id(id_file: Path | None = None) -> str:
    """Retrieve and increment a persistent unique experiment ID.

    Args:
        id_file: Path to store the ID counter. Defaults to state_dir/experiment_id.txt.

    Returns:
        Zero-padded 4-digit unique ID string.
    """
    if id_file is None:
        from llm_energy_measure.constants import DEFAULT_STATE_DIR

        id_file = DEFAULT_STATE_DIR / "experiment_id.txt"

    id_file.parent.mkdir(parents=True, exist_ok=True)

    last_id = 0
    if id_file.exists():
        try:
            last_id = int(id_file.read_text().strip())
        except ValueError:
            logger.warning(f"Invalid ID in {id_file}, resetting to 0")

    new_id = last_id + 1
    id_file.write_text(str(new_id))

    return f"{new_id:04d}"


def get_shared_unique_id(accelerator: Accelerator, id_file: Path | None = None) -> str:
    """Generate a unique ID on main process and broadcast to all workers.

    Uses torch.distributed.broadcast_object_list to ensure all processes
    get the same unique ID.

    Args:
        accelerator: The Accelerator instance.
        id_file: Path to store the ID counter.

    Returns:
        Unique experiment ID, same across all processes.
    """
    unique_id_list = [""]

    if accelerator.is_main_process:
        unique_id_list[0] = get_persistent_unique_id(id_file)

    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(unique_id_list, src=0)

    return unique_id_list[0]


def get_original_generate_method(model: Any) -> Any | None:
    """Recursively find the generate method through model wrappers.

    Useful for finding the original generate method before model is
    wrapped with DataParallel or FSDP.

    Args:
        model: The model, possibly wrapped.

    Returns:
        The original generate method if found, None otherwise.
    """
    if hasattr(model, "generate") and callable(model.generate):
        return model.generate
    elif hasattr(model, "module"):
        return get_original_generate_method(model.module)
    return None


def safe_wait(
    accelerator: Accelerator,
    description: str = "",
    timeout: int = DEFAULT_BARRIER_TIMEOUT,
) -> bool:
    """Wait for all processes with timeout protection.

    Args:
        accelerator: The Accelerator instance.
        description: Description of the barrier for logging.
        timeout: Maximum seconds to wait.

    Returns:
        True if barrier completed, False if timeout occurred.
    """
    logger.debug(f"Entering wait barrier: {description}")

    completed = threading.Event()

    def wait_func() -> None:
        try:
            accelerator.wait_for_everyone()
            completed.set()
        except Exception as e:
            logger.error(f"Error during wait_for_everyone at {description}: {e}")

    thread = threading.Thread(target=wait_func, daemon=True)
    thread.start()
    thread.join(timeout)

    if not completed.is_set():
        logger.warning(
            f"Timeout: wait_for_everyone did not complete within {timeout}s " f"for {description}"
        )
        return False

    logger.debug(f"Completed wait barrier: {description}")
    return True


def cleanup_distributed() -> None:
    """Clean up distributed process group and CUDA cache."""
    try:
        torch.cuda.empty_cache()
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
        logger.info("Distributed cleanup completed")
    except Exception as e:
        logger.warning(f"Error during distributed cleanup: {e}")


class MinimalAccelerator:
    """Minimal Accelerator-like object for backends that manage their own distribution.

    vLLM and other backends that handle their own CUDA context and multiprocessing
    don't need the full Accelerator. This provides a minimal compatible interface.
    """

    def __init__(self, device: torch.device) -> None:
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def num_processes(self) -> int:
        return 1

    @property
    def process_index(self) -> int:
        return 0

    @property
    def is_main_process(self) -> bool:
        return True

    def wait_for_everyone(self) -> None:
        """No-op for single process."""
        pass


def create_minimal_accelerator(device: torch.device) -> MinimalAccelerator:
    """Create a minimal Accelerator-like object for vLLM and similar backends.

    Args:
        device: The torch device to use.

    Returns:
        MinimalAccelerator instance.
    """
    return MinimalAccelerator(device)
