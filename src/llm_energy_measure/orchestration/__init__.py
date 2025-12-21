"""Experiment orchestration for LLM Bench.

This module provides experiment lifecycle management including:
- ExperimentContext: Runtime state container
- experiment_context: Context manager for experiment lifecycle
- ExperimentOrchestrator: Main experiment runner
- Lifecycle utilities: Setup/teardown, warmup
- Launcher utilities: Accelerate CLI launching
"""

from llm_energy_measure.orchestration.context import (
    ExperimentContext,
    experiment_context,
)
from llm_energy_measure.orchestration.launcher import (
    launch_experiment_accelerate,
    log_failed_experiment,
    run_from_config,
)
from llm_energy_measure.orchestration.lifecycle import (
    cleanup_cuda,
    cleanup_distributed,
    ensure_clean_start,
    experiment_lifecycle,
    full_cleanup,
    warmup_model,
)
from llm_energy_measure.orchestration.runner import ExperimentOrchestrator

__all__ = [
    "ExperimentContext",
    "ExperimentOrchestrator",
    "cleanup_cuda",
    "cleanup_distributed",
    "ensure_clean_start",
    "experiment_context",
    "experiment_lifecycle",
    "full_cleanup",
    "launch_experiment_accelerate",
    "log_failed_experiment",
    "run_from_config",
    "warmup_model",
]
