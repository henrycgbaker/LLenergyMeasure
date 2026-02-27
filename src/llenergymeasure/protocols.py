"""Protocol definitions for llenergymeasure v2.0.

These protocols define the interfaces for pluggable components,
enabling dependency injection and easier testing.

Note: Model and tokenizer parameters use `Any` because different backends
use different types (torch.nn.Module, vLLM engine, etc.). The protocols
define the interface contract, not the concrete types.

The v2.0 domain types (ExperimentConfig, ExperimentResult) are imported
under TYPE_CHECKING only — they will be created in Phase 2 and Phase 6.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.experiment import ExperimentResult


@runtime_checkable
class ModelLoader(Protocol):
    """Protocol for loading models and tokenizers."""

    def load(self, config: ExperimentConfig) -> tuple[Any, Any]:
        """Load model and tokenizer. Returns (model, tokenizer)."""
        ...


@runtime_checkable
class InferenceEngine(Protocol):
    """Protocol for running model inference."""

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        config: ExperimentConfig,
    ) -> Any:
        """Run inference on prompts. Returns backend-specific result."""
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Protocol for collecting metrics from inference."""

    def collect(
        self,
        model: Any,
        inference_result: Any,
        config: ExperimentConfig,
    ) -> Any:
        """Collect metrics from inference. Returns metrics object."""
        ...


@runtime_checkable
class EnergyBackend(Protocol):
    """Protocol for energy measurement backends.

    Implementations include Zeus, NVML, CodeCarbon, etc.
    """

    @property
    def name(self) -> str:
        """Backend name for identification."""
        ...

    def start_tracking(self) -> Any:
        """Start energy tracking. Returns tracker handle."""
        ...

    def stop_tracking(self, tracker: Any) -> Any:
        """Stop energy tracking and return metrics."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...


@runtime_checkable
class ResultsRepository(Protocol):
    """Protocol for persisting experiment results."""

    def save(self, result: ExperimentResult, output_dir: Path) -> Path:
        """Save experiment result to output directory."""
        ...

    def load(self, path: Path) -> ExperimentResult:
        """Load experiment result from path."""
        ...
