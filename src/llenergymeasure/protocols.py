"""Protocol definitions for LLM Bench framework.

These protocols define the interfaces for pluggable components,
enabling dependency injection and easier testing.

Note: Model and tokenizer parameters use `Any` because different backends
use different types (torch.nn.Module, vLLM engine, etc.). The protocols
define the interface contract, not the concrete types.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.experiment import AggregatedResult, RawProcessResult
    from llenergymeasure.domain.metrics import CombinedMetrics, EnergyMetrics


@runtime_checkable
class ModelLoader(Protocol):
    """Protocol for loading models and tokenizers."""

    def load(self, config: "ExperimentConfig") -> tuple[Any, Any]:
        """Load model and tokenizer from config.

        Args:
            config: Experiment configuration.

        Returns:
            Tuple of (model, tokenizer). Types depend on backend.
        """
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol for running model inference."""

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        config: "ExperimentConfig",
    ) -> Any:
        """Run inference on prompts.

        Args:
            model: Loaded model (type depends on backend).
            tokenizer: Loaded tokenizer (type depends on backend).
            prompts: List of prompts to process.
            config: Experiment configuration.

        Returns:
            Inference result (type depends on backend, typically InferenceMetrics
            or BackendInferenceResult).
        """
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for collecting metrics from inference."""

    def collect(
        self,
        model: Any,
        inference_result: Any,
        config: "ExperimentConfig",
    ) -> "CombinedMetrics":
        """Collect metrics from model and inference result.

        Args:
            model: The model used for inference (type depends on backend).
            inference_result: Result from inference engine (type depends on backend).
            config: Experiment configuration.

        Returns:
            Combined metrics (inference, energy, compute).
        """
        ...


@runtime_checkable
class EnergyBackend(Protocol):
    """Protocol for energy measurement backends.

    Implementations include CodeCarbon, NVML, RAPL, etc.
    """

    @property
    def name(self) -> str:
        """Backend name for identification."""
        ...

    def start_tracking(self) -> Any:
        """Start energy tracking.

        Returns:
            Tracker handle to pass to stop_tracking.
        """
        ...

    def stop_tracking(self, tracker: Any) -> "EnergyMetrics":
        """Stop energy tracking and return metrics.

        Args:
            tracker: Handle from start_tracking.

        Returns:
            Energy metrics from the tracking period.
        """
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system.

        Returns:
            True if the backend can be used.
        """
        ...


@runtime_checkable
class ResultsRepository(Protocol):
    """Protocol for persisting experiment results.

    Supports the late aggregation pattern where raw per-process
    results are saved separately and aggregated later.
    """

    def save_raw(self, experiment_id: str, result: "RawProcessResult") -> Path:
        """Save raw per-process result.

        Args:
            experiment_id: Unique experiment identifier.
            result: Raw result from a single process.

        Returns:
            Path to the saved result file.
        """
        ...

    def list_raw(self, experiment_id: str) -> list[Path]:
        """List all raw result files for an experiment.

        Args:
            experiment_id: Unique experiment identifier.

        Returns:
            List of paths to raw result files.
        """
        ...

    def load_raw(self, path: Path) -> "RawProcessResult":
        """Load a raw result from file.

        Args:
            path: Path to the result file.

        Returns:
            Loaded RawProcessResult.
        """
        ...

    def save_aggregated(self, result: "AggregatedResult") -> Path:
        """Save aggregated experiment result.

        Args:
            result: Aggregated result to save.

        Returns:
            Path to the saved result file.
        """
        ...
