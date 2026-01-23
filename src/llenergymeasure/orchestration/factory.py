"""Factory for creating experiment components with DI wiring.

Supports backend selection via config.backend field. Supported backends:
- pytorch: HuggingFace Transformers + Accelerate (default)
- vllm: vLLM with PagedAttention and continuous batching
- tensorrt: TensorRT-LLM with compiled execution plans
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from llenergymeasure.core.inference_backends.protocols import InferenceBackend
    from llenergymeasure.orchestration.context import ExperimentContext
    from llenergymeasure.orchestration.runner import ExperimentOrchestrator
    from llenergymeasure.protocols import (
        EnergyBackend,
        InferenceEngine,
        MetricsCollector,
        ModelLoader,
        ResultsRepository,
    )


@dataclass
class ExperimentComponents:
    """Container for all experiment components."""

    model_loader: ModelLoader
    inference_engine: InferenceEngine
    metrics_collector: MetricsCollector
    energy_backend: EnergyBackend
    repository: ResultsRepository
    backend_name: str = "pytorch"
    backend_version: str | None = None


def _validate_backend(backend_name: str) -> None:
    """Validate that the requested backend is available.

    Args:
        backend_name: Name of the backend to validate.

    Raises:
        ConfigurationError: If backend is not available.
    """
    from llenergymeasure.core.inference_backends import get_backend, is_backend_available
    from llenergymeasure.exceptions import ConfigurationError

    if not is_backend_available(backend_name):
        # Try to get the backend to produce a helpful error message
        try:
            get_backend(backend_name)
        except ConfigurationError:
            raise
        raise ConfigurationError(f"Backend '{backend_name}' is not available on this system.")


def create_components(
    ctx: ExperimentContext, results_dir: Path | None = None
) -> ExperimentComponents:
    """Create all experiment components wired for the given context.

    Validates the requested backend and creates appropriate components.
    Supports 'pytorch', 'vllm', and 'tensorrt' backends.

    Args:
        ctx: Experiment context with accelerator and config.
        results_dir: Optional results directory (None uses default from constants/env).

    Returns:
        ExperimentComponents with all dependencies wired.

    Raises:
        ConfigurationError: If requested backend is not available or not yet supported.
    """
    from llenergymeasure.core.inference_backends import get_backend as get_inference_backend

    # Get backend name from config (default to pytorch)
    backend_name = getattr(ctx.config, "backend", "pytorch")

    # Validate backend is available
    _validate_backend(backend_name)

    # Get backend instance
    backend = get_inference_backend(backend_name)
    logger.debug(f"Using inference backend: {backend.name} ({backend.version})")

    # Create components using unified function
    return _create_backend_components(ctx, backend, results_dir)


def _create_backend_components(
    ctx: ExperimentContext,
    backend: InferenceBackend,
    results_dir: Path | None = None,
) -> ExperimentComponents:
    """Create experiment components for any backend.

    Uses RuntimeCapabilities to determine whether the backend manages its own
    CUDA context and device allocation, adjusting the runtime context accordingly.

    Args:
        ctx: Experiment context with device/process info.
        backend: The inference backend instance.
        results_dir: Directory for results persistence.

    Returns:
        ExperimentComponents with all adapters configured.
    """
    from llenergymeasure.core.energy_backends import get_backend as get_energy_backend
    from llenergymeasure.core.inference_backends.adapters import (
        BackendInferenceEngineAdapter,
        BackendMetricsCollectorAdapter,
        BackendModelLoaderAdapter,
    )
    from llenergymeasure.core.inference_backends.protocols import BackendRuntime
    from llenergymeasure.results.repository import FileSystemRepository

    # Query backend capabilities to determine device/accelerator handling
    capabilities = backend.get_runtime_capabilities()
    manages_own_devices = not capabilities.orchestrator_may_call_cuda

    # Create runtime context - backends that manage their own CUDA don't need device/accelerator
    runtime = BackendRuntime(
        device=None if manages_own_devices else ctx.device,
        process_index=ctx.process_index,
        num_processes=ctx.num_processes,
        is_main_process=ctx.is_main_process,
        accelerator=None if manages_own_devices else ctx.accelerator,
    )

    # Create adapters that wrap the backend
    model_loader = BackendModelLoaderAdapter(backend, runtime)
    inference_engine = BackendInferenceEngineAdapter(backend)
    metrics_collector = BackendMetricsCollectorAdapter(backend)

    return ExperimentComponents(
        model_loader=model_loader,
        inference_engine=inference_engine,
        metrics_collector=metrics_collector,
        energy_backend=get_energy_backend("codecarbon"),
        repository=FileSystemRepository(results_dir),
        backend_name=backend.name,
        backend_version=backend.version,
    )


def create_orchestrator(
    ctx: ExperimentContext, results_dir: Path | None = None
) -> ExperimentOrchestrator:
    """Create a fully wired ExperimentOrchestrator.

    Convenience function that creates components and instantiates orchestrator.

    Args:
        ctx: Experiment context.
        results_dir: Optional results directory (None uses default from constants/env).

    Returns:
        Ready-to-use ExperimentOrchestrator.
    """
    from llenergymeasure.orchestration.runner import ExperimentOrchestrator

    components = create_components(ctx, results_dir)
    return ExperimentOrchestrator(
        model_loader=components.model_loader,
        inference_engine=components.inference_engine,
        metrics_collector=components.metrics_collector,
        energy_backend=components.energy_backend,
        repository=components.repository,
        backend_name=components.backend_name,
        backend_version=components.backend_version,
    )
