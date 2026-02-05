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

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.core.distributed import (
    cleanup_distributed,
    get_accelerator,
    get_shared_unique_id,
)

if TYPE_CHECKING:
    from accelerate import Accelerator


def _get_orchestrator_may_call_cuda(backend_name: str) -> bool:
    """Check if orchestration layer may safely call torch.cuda.* functions.

    Uses the backend's RuntimeCapabilities to determine if the orchestration
    layer can safely initialize CUDA, or if the backend manages CUDA internally.

    Args:
        backend_name: Name of the backend (e.g., "pytorch", "vllm").

    Returns:
        True if orchestrator may call torch.cuda.*, False if backend manages CUDA.
    """
    try:
        from llenergymeasure.core.inference_backends import get_backend

        backend = get_backend(backend_name)
        capabilities = backend.get_runtime_capabilities()
        return capabilities.orchestrator_may_call_cuda
    except Exception as e:
        # If we can't load the backend, fall back to safe defaults
        # vLLM and tensorrt must NOT have CUDA initialized before them
        logger.debug(f"Could not get capabilities for {backend_name}: {e}")
        return backend_name not in ("vllm", "tensorrt")


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
    config_warnings: list[str] = field(default_factory=list)
    parameter_provenance: dict[str, dict[str, Any]] = field(default_factory=dict)
    preset_chain: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        config: ExperimentConfig,
        id_file: Path | None = None,
        effective_config: dict[str, Any] | None = None,
        cli_overrides: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        config_warnings: list[str] | None = None,
        parameter_provenance: dict[str, dict[str, Any]] | None = None,
        preset_chain: list[str] | None = None,
    ) -> ExperimentContext:
        """Factory method to create context from config.

        Sets up the Accelerator, generates a unique experiment ID,
        and determines process-specific information.

        For backends that manage their own CUDA context (vLLM, TensorRT-LLM),
        skips Accelerator initialization to avoid CUDA fork issues.

        Args:
            config: The experiment configuration.
            id_file: Optional path for storing persistent ID counter.
            effective_config: Full resolved config for reproducibility.
            cli_overrides: CLI parameters that overrode config values.
            experiment_id: Optional pre-generated experiment ID (from CLI).
            config_warnings: Config validation warnings to embed in results.
            parameter_provenance: Full provenance tracking for each parameter.
            preset_chain: Presets applied in order.

        Returns:
            Fully initialized ExperimentContext.
        """
        # Query backend capabilities to determine CUDA management
        backend_name = getattr(config, "backend", "pytorch")
        orchestrator_may_call_cuda = _get_orchestrator_may_call_cuda(backend_name)

        if not orchestrator_may_call_cuda:
            # Backend manages CUDA context - use minimal context
            # CRITICAL: Do NOT call torch.cuda.* functions here - they initialize CUDA
            from llenergymeasure.core.distributed import get_persistent_unique_id

            if experiment_id is None:
                experiment_id = get_persistent_unique_id(id_file)

            # Use a placeholder device - backend manages its own device selection
            # We don't check torch.cuda.is_available() as it can initialize CUDA
            device = torch.device("cuda:0")  # Backend will handle actual device

            logger.info(
                f"Created ExperimentContext ({backend_name} mode): id={experiment_id}, "
                f"device={device} (backend manages CUDA)"
            )

            # Create minimal accelerator-like object for compatibility
            from llenergymeasure.core.distributed import create_minimal_accelerator

            minimal_accelerator = create_minimal_accelerator(device)

            return cls(
                experiment_id=experiment_id,
                config=config,
                accelerator=minimal_accelerator,
                device=device,
                is_main_process=True,  # Backend runs single process from our perspective
                process_index=0,
                start_time=datetime.now(),
                effective_config=effective_config or {},
                cli_overrides=cli_overrides or {},
                config_warnings=config_warnings or [],
                parameter_provenance=parameter_provenance or {},
                preset_chain=preset_chain or [],
            )

        # Standard path: orchestrator manages CUDA via Accelerator
        # gpus = available device IDs, num_processes from backend-native config
        from llenergymeasure.orchestration.launcher import get_launcher_process_count

        num_processes = get_launcher_process_count(config)
        accelerator = get_accelerator(gpus=config.gpus, num_processes=num_processes)

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
            config_warnings=config_warnings or [],
            parameter_provenance=parameter_provenance or {},
            preset_chain=preset_chain or [],
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
    config_warnings: list[str] | None = None,
    parameter_provenance: dict[str, dict[str, Any]] | None = None,
    preset_chain: list[str] | None = None,
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
        config_warnings: Config validation warnings to embed in results.
        parameter_provenance: Full provenance tracking for each parameter.
        preset_chain: Presets applied in order.

    Yields:
        ExperimentContext instance.

    Example:
        >>> with experiment_context(config) as ctx:
        ...     model = load_model(ctx.config, ctx.device)
        ...     results = run_inference(model, ctx)
    """
    ctx = ExperimentContext.create(
        config,
        id_file,
        effective_config,
        cli_overrides,
        experiment_id,
        config_warnings,
        parameter_provenance,
        preset_chain,
    )
    try:
        yield ctx
    finally:
        ctx.cleanup()
