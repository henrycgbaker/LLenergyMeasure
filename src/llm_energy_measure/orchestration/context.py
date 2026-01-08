"""Experiment context management for LLM Bench.

This module provides the ExperimentContext dataclass and context manager
for managing experiment lifecycle, including distributed computing setup
and resource cleanup.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.core.distributed import (
    cleanup_distributed,
    get_accelerator,
    get_shared_unique_id,
)

if TYPE_CHECKING:
    from accelerate import Accelerator


@dataclass
class ExperimentContext:
    """Context object containing experiment state and resources.

    Encapsulates all runtime state for an experiment, including the
    Accelerator instance, device information, and process identifiers.
    Created via the `experiment_context` context manager or `create` factory.

    Attributes:
        experiment_id: Unique identifier for this experiment run.
        config: The experiment configuration.
        accelerator: Accelerate instance for distributed computing.
        device: The torch device for this process.
        is_main_process: Whether this is the main (rank 0) process.
        process_index: Index of this process (0 to num_processes-1).
        start_time: When the experiment started.
        effective_config: Full resolved config for reproducibility.
        cli_overrides: CLI parameters that overrode config values.
    """

    experiment_id: str
    config: ExperimentConfig
    accelerator: Accelerator
    device: torch.device
    is_main_process: bool
    process_index: int
    start_time: datetime = field(default_factory=datetime.now)
    effective_config: dict[str, Any] = field(default_factory=dict)
    cli_overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        id_file: Path | None = None,
        effective_config: dict[str, Any] | None = None,
        cli_overrides: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentContext:
        """Factory method to create context from config.

        Sets up the Accelerator, generates a unique experiment ID,
        and determines process-specific information.

        Args:
            config: The experiment configuration.
            id_file: Optional path for storing persistent ID counter.
            effective_config: Full resolved config for reproducibility.
            cli_overrides: CLI parameters that overrode config values.
            experiment_id: Optional pre-generated experiment ID (from CLI).

        Returns:
            Fully initialized ExperimentContext.
        """
        accelerator = get_accelerator(
            gpu_list=config.gpu_list,
            num_processes=config.num_processes,
        )

        # Use provided experiment_id if available, otherwise generate one
        if experiment_id is None:
            experiment_id = get_shared_unique_id(accelerator, id_file)

        logger.info(
            f"Created ExperimentContext: id={experiment_id}, "
            f"process={accelerator.process_index}/{accelerator.num_processes}, "
            f"device={accelerator.device}"
        )

        return cls(
            experiment_id=experiment_id,
            config=config,
            accelerator=accelerator,
            device=accelerator.device,
            is_main_process=accelerator.is_main_process,
            process_index=accelerator.process_index,
            start_time=datetime.now(),
            effective_config=effective_config or {},
            cli_overrides=cli_overrides or {},
        )

    def cleanup(self) -> None:
        """Cleanup resources (GPU memory, distributed state).

        Frees GPU memory and destroys the distributed process group.
        Safe to call multiple times.
        """
        logger.debug(f"Cleaning up ExperimentContext {self.experiment_id}")
        cleanup_distributed()

    @property
    def elapsed_time(self) -> float:
        """Return elapsed time since experiment start in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def num_processes(self) -> int:
        """Return total number of processes."""
        return int(self.accelerator.num_processes)


@contextmanager
def experiment_context(
    config: ExperimentConfig,
    id_file: Path | None = None,
    effective_config: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
    experiment_id: str | None = None,
) -> Iterator[ExperimentContext]:
    """Context manager for experiment lifecycle.

    Creates an ExperimentContext, yields it, and ensures cleanup
    happens regardless of whether the experiment succeeds or fails.

    Args:
        config: The experiment configuration.
        id_file: Optional path for storing persistent ID counter.
        effective_config: Full resolved config for reproducibility.
        cli_overrides: CLI parameters that overrode config values.
        experiment_id: Optional pre-generated experiment ID (from CLI).

    Yields:
        ExperimentContext instance.

    Example:
        >>> with experiment_context(config) as ctx:
        ...     model = load_model(ctx.config, ctx.device)
        ...     results = run_inference(model, ctx)
    """
    ctx = ExperimentContext.create(config, id_file, effective_config, cli_overrides, experiment_id)
    try:
        yield ctx
    finally:
        ctx.cleanup()
