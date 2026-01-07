"""Factory for creating experiment components with DI wiring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def create_components(ctx: ExperimentContext) -> ExperimentComponents:
    """Create all experiment components wired for the given context.

    Args:
        ctx: Experiment context with accelerator and config.

    Returns:
        ExperimentComponents with all dependencies wired.
    """
    from llm_energy_measure.core.energy_backends import get_backend
    from llm_energy_measure.core.implementations import (
        HuggingFaceModelLoader,
        ThroughputMetricsCollector,
        TransformersInferenceEngine,
    )
    from llm_energy_measure.results.repository import FileSystemRepository

    return ExperimentComponents(
        model_loader=HuggingFaceModelLoader(),
        inference_engine=TransformersInferenceEngine(ctx.accelerator),
        metrics_collector=ThroughputMetricsCollector(ctx.accelerator),
        energy_backend=get_backend("codecarbon"),
        repository=FileSystemRepository(),
    )


def create_orchestrator(ctx: ExperimentContext) -> ExperimentOrchestrator:
    """Create a fully wired ExperimentOrchestrator.

    Convenience function that creates components and instantiates orchestrator.

    Args:
        ctx: Experiment context.

    Returns:
        Ready-to-use ExperimentOrchestrator.
    """
    from llm_energy_measure.orchestration.runner import ExperimentOrchestrator

    components = create_components(ctx)
    return ExperimentOrchestrator(
        model_loader=components.model_loader,
        inference_engine=components.inference_engine,
        metrics_collector=components.metrics_collector,
        energy_backend=components.energy_backend,
        repository=components.repository,
    )
