"""Protocol implementations wrapping existing functions.

These classes adapt existing function-based implementations to the
protocol interfaces for dependency injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import CombinedMetrics


class HuggingFaceModelLoader:
    """Model loader using HuggingFace Transformers.

    Implements ModelLoader protocol by wrapping load_model_tokenizer().
    """

    def load(self, config: ExperimentConfig) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load model and tokenizer from HuggingFace.

        Args:
            config: Experiment configuration with model settings.

        Returns:
            Tuple of (model, tokenizer).
        """
        from llenergymeasure.core.model_loader import load_model_tokenizer

        return load_model_tokenizer(config)


class ThroughputMetricsCollector:
    """Metrics collector for throughput and compute metrics.

    Implements MetricsCollector protocol by wrapping collect_compute_metrics()
    and combining with inference metrics from the inference result.
    """

    def __init__(self, accelerator: Accelerator) -> None:
        """Initialize with accelerator for device access.

        Args:
            accelerator: HuggingFace Accelerator instance.
        """
        self._accelerator = accelerator

    def collect(
        self,
        model: Any,
        inference_result: Any,
        config: ExperimentConfig,
    ) -> CombinedMetrics:
        """Collect metrics from model and inference result.

        Args:
            model: The model used for inference.
            inference_result: Result from inference engine.
            config: Experiment configuration.

        Returns:
            Combined metrics (inference, energy placeholder, compute).
        """
        from llenergymeasure.core.compute_metrics import collect_compute_metrics
        from llenergymeasure.domain.metrics import CombinedMetrics, EnergyMetrics

        compute = collect_compute_metrics(
            model=model,
            device=self._accelerator.device,
            input_ids=inference_result.input_ids,
            accelerator=self._accelerator,
            config=config,
        )

        # Return with placeholder energy - orchestrator uses energy from backend directly
        return CombinedMetrics(
            inference=inference_result.metrics,
            energy=EnergyMetrics.placeholder(),
            compute=compute,
        )
