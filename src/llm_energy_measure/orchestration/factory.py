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
    from llm_energy_measure.core.inference_backends.protocols import InferenceBackend
    from llm_energy_measure.orchestration.context import ExperimentContext
    from llm_energy_measure.orchestration.runner import ExperimentOrchestrator
    from llm_energy_measure.protocols import (
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
    from llm_energy_measure.core.inference_backends import get_backend, is_backend_available
    from llm_energy_measure.exceptions import ConfigurationError

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
    from llm_energy_measure.core.inference_backends import get_backend as get_inference_backend
    from llm_energy_measure.exceptions import ConfigurationError

    # Get backend name from config (default to pytorch)
    backend_name = getattr(ctx.config, "backend", "pytorch")

    # Validate backend is available
    _validate_backend(backend_name)

    # Get backend instance
    backend = get_inference_backend(backend_name)
    logger.debug(f"Using inference backend: {backend.name} ({backend.version})")

    # Create backend-specific components
    if backend_name == "pytorch":
        return _create_pytorch_components(ctx, backend, results_dir)
    elif backend_name == "vllm":
        return _create_vllm_components(ctx, backend, results_dir)
    elif backend_name == "tensorrt":
        return _create_tensorrt_components(ctx, backend, results_dir)
    else:
        raise ConfigurationError(
            f"Backend '{backend_name}' is not yet integrated with the orchestrator. "
            f"Supported backends: pytorch, vllm, tensorrt."
        )


def _create_pytorch_components(
    ctx: ExperimentContext, backend: InferenceBackend, results_dir: Path | None = None
) -> ExperimentComponents:
    """Create components for PyTorch/Transformers backend.

    Uses the adapter pattern to route through PyTorchBackend.run_inference(),
    which supports streaming latency measurement (TTFT/ITL).

    Args:
        ctx: Experiment context.
        backend: PyTorch backend instance.
        results_dir: Optional results directory (None uses default).

    Returns:
        ExperimentComponents configured for PyTorch.
    """
    from llm_energy_measure.core.energy_backends import get_backend as get_energy_backend
    from llm_energy_measure.core.inference_backends.adapters import (
        BackendInferenceEngineAdapter,
        BackendMetricsCollectorAdapter,
        BackendModelLoaderAdapter,
    )
    from llm_energy_measure.core.inference_backends.protocols import BackendRuntime
    from llm_energy_measure.results.repository import FileSystemRepository

    # Create runtime context for PyTorch
    runtime = BackendRuntime(
        device=ctx.device,
        process_index=ctx.process_index,
        num_processes=ctx.num_processes,
        is_main_process=ctx.is_main_process,
        accelerator=ctx.accelerator,
    )

    # Create adapters that wrap the PyTorch backend
    # This enables streaming latency measurement via backend.run_inference()
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


def _create_vllm_components(
    ctx: ExperimentContext, backend: InferenceBackend, results_dir: Path | None = None
) -> ExperimentComponents:
    """Create components for vLLM backend.

    vLLM manages its own model loading and distribution, so we use adapters
    to integrate with the existing orchestrator architecture.

    Args:
        ctx: Experiment context.
        backend: vLLM backend instance.
        results_dir: Optional results directory (None uses default).

    Returns:
        ExperimentComponents configured for vLLM.
    """
    from llm_energy_measure.core.energy_backends import get_backend as get_energy_backend
    from llm_energy_measure.core.inference_backends.adapters import (
        BackendInferenceEngineAdapter,
        BackendMetricsCollectorAdapter,
        BackendModelLoaderAdapter,
    )
    from llm_energy_measure.core.inference_backends.protocols import BackendRuntime
    from llm_energy_measure.results.repository import FileSystemRepository

    # Create runtime context for vLLM
    # vLLM manages its own distribution, so we don't pass the accelerator
    runtime = BackendRuntime(
        device=None,  # vLLM manages devices
        process_index=ctx.process_index,
        num_processes=ctx.num_processes,
        is_main_process=ctx.is_main_process,
        accelerator=None,  # vLLM doesn't use Accelerate
    )

    # Create adapters that wrap the vLLM backend
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


def _create_tensorrt_components(
    ctx: ExperimentContext, backend: InferenceBackend, results_dir: Path | None = None
) -> ExperimentComponents:
    """Create components for TensorRT-LLM backend.

    TensorRT-LLM manages its own model loading, CUDA context, and batching.
    Uses the same adapter pattern as vLLM.

    Args:
        ctx: Experiment context.
        backend: TensorRT backend instance.
        results_dir: Optional results directory (None uses default).

    Returns:
        ExperimentComponents configured for TensorRT-LLM.
    """
    from llm_energy_measure.core.energy_backends import get_backend as get_energy_backend
    from llm_energy_measure.core.inference_backends.adapters import (
        BackendInferenceEngineAdapter,
        BackendMetricsCollectorAdapter,
        BackendModelLoaderAdapter,
    )
    from llm_energy_measure.core.inference_backends.protocols import BackendRuntime
    from llm_energy_measure.results.repository import FileSystemRepository

    # Create runtime context for TensorRT
    # TensorRT manages its own distribution, so we don't pass the accelerator
    runtime = BackendRuntime(
        device=None,  # TensorRT manages devices
        process_index=ctx.process_index,
        num_processes=ctx.num_processes,
        is_main_process=ctx.is_main_process,
        accelerator=None,  # TensorRT doesn't use Accelerate
    )

    # Create adapters that wrap the TensorRT backend
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
    from llm_energy_measure.orchestration.runner import ExperimentOrchestrator

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
