"""Protocol implementations wrapping existing functions.

These classes adapt existing function-based implementations to the
protocol interfaces for dependency injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from accelerate import Accelerator
    from transformers import PreTrainedModel, PreTrainedTokenizer

    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.core.inference import InferenceResult
    from llm_energy_measure.domain.metrics import CombinedMetrics


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
        from llm_energy_measure.core.model_loader import load_model_tokenizer

        return load_model_tokenizer(config)


class TransformersInferenceEngine:
    """Inference engine using HuggingFace Transformers.

    Implements InferenceEngine protocol by wrapping run_inference().
    Requires accelerator at construction time for distributed setup.
    """

    def __init__(self, accelerator: Accelerator) -> None:
        """Initialize with accelerator for distributed inference.

        Args:
            accelerator: HuggingFace Accelerator instance.
        """
        self._accelerator = accelerator

    def run(
        self,
        model: Any,
        tokenizer: Any,
        prompts: list[str],
        config: ExperimentConfig,
    ) -> InferenceResult:
        """Run inference on prompts.

        Args:
            model: Loaded model.
            tokenizer: Loaded tokenizer.
            prompts: List of prompts to process.
            config: Experiment configuration.

        Returns:
            InferenceResult with metrics and outputs.
        """
        from llm_energy_measure.core.inference import run_inference

        # Note: run_inference has param order (model, config, prompts, tokenizer, accelerator)
        return run_inference(model, config, prompts, tokenizer, self._accelerator)


class ThroughputMetricsCollector:
    """Metrics collector for throughput and compute metrics.

    Implements MetricsCollector protocol by wrapping collect_compute_metrics()
    and combining with inference metrics from InferenceResult.
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
        inference_result: InferenceResult,
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
        from llm_energy_measure.core.compute_metrics import collect_compute_metrics
        from llm_energy_measure.domain.metrics import CombinedMetrics, EnergyMetrics

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
            energy=EnergyMetrics(
                total_energy_j=0.0,
                gpu_energy_j=0.0,
                cpu_energy_j=0.0,
                ram_energy_j=0.0,
                gpu_power_w=0.0,
                cpu_power_w=0.0,
                duration_sec=0.0,
                emissions_kg_co2=0.0,
                energy_per_token_j=0.0,
            ),
            compute=compute,
        )
