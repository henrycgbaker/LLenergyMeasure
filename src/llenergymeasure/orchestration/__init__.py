"""Experiment orchestration for LLM Bench.

This module provides experiment lifecycle management including:
- ExperimentContext: Runtime state container
- experiment_context: Context manager for experiment lifecycle
- ExperimentOrchestrator: Main experiment runner
- Lifecycle utilities: Setup/teardown, warmup

Note: launcher.py is meant to be run via `python -m` or `accelerate launch -m`,
not imported directly. Import from llenergymeasure.orchestration.launcher if needed.
"""

from llenergymeasure.orchestration.context import (
    ExperimentContext,
    experiment_context,
)
from llenergymeasure.orchestration.factory import (
    ExperimentComponents,
    create_components,
    create_orchestrator,
)
from llenergymeasure.orchestration.lifecycle import (
    cleanup_cuda,
    cleanup_distributed,
    ensure_clean_start,
    experiment_lifecycle,
    full_cleanup,
    warmup_model,
)
from llenergymeasure.orchestration.runner import ExperimentOrchestrator

__all__ = [
    "ExperimentComponents",
    "ExperimentContext",
    "ExperimentOrchestrator",
    "cleanup_cuda",
    "cleanup_distributed",
    "create_components",
    "create_orchestrator",
    "ensure_clean_start",
    "experiment_context",
    "experiment_lifecycle",
    "full_cleanup",
    "warmup_model",
]
