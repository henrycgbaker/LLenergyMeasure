"""InferenceBackend Protocol — contract all backends must satisfy."""

from typing import Protocol, runtime_checkable

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult


@runtime_checkable
class InferenceBackend(Protocol):
    """Contract for all inference backends (PyTorch, vLLM, TRT-LLM).

    Each backend owns its entire lifecycle internally:
    model load, warmup, measurement, cleanup. This matches the
    lm-eval LM subclass pattern — one method, one contract.
    """

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'pytorch', 'vllm', 'tensorrt')."""
        ...

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a complete inference experiment and return aggregated results.

        Args:
            config: Fully resolved experiment configuration.

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            BackendError: If model loading or inference fails.
            PreFlightError: If pre-flight checks fail before GPU allocation.
        """
        ...
