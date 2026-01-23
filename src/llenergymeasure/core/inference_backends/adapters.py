"""Adapters for integrating inference backends with the orchestrator.

These adapters wrap InferenceBackend implementations (like VLLMBackend) to
implement the protocols expected by the ExperimentOrchestrator (ModelLoader,
InferenceEngine, MetricsCollector).

This adapter pattern allows the orchestrator to remain unchanged while
supporting different backends with unified interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llenergymeasure.core.inference_backends.protocols import (
    BackendResult,
    BackendRuntime,
    InferenceBackend,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import CombinedMetrics, InferenceMetrics


@dataclass
class BackendInferenceResult:
    """Adapter result that bridges BackendResult to orchestrator expectations.

    The orchestrator expects an inference result with:
    - metrics: InferenceMetrics
    - input_ids: tensor (for FLOPs estimation)

    For backends like vLLM that don't expose input_ids, we use None and
    the metrics collector handles this gracefully.
    """

    metrics: InferenceMetrics
    input_ids: Any  # None for vLLM, torch.Tensor for PyTorch
    backend_result: BackendResult  # Original result for metadata access


class BackendModelLoaderAdapter:
    """Adapter that implements ModelLoader protocol using InferenceBackend.

    For backends like vLLM that manage their own model loading, this adapter:
    1. Initialises the backend during load()
    2. Returns (None, None) since the backend manages model internally
    3. Stores the backend reference for the inference adapter to use

    The inference adapter must be initialised with the same backend instance.
    """

    def __init__(self, backend: InferenceBackend, runtime: BackendRuntime) -> None:
        """Initialise with backend and runtime context.

        Args:
            backend: The inference backend instance.
            runtime: Runtime context for initialisation.
        """
        self._backend = backend
        self._runtime = runtime
        self._initialised = False

    def load(self, config: ExperimentConfig) -> tuple[None, None]:
        """Initialise the backend (model loading happens internally).

        Args:
            config: Experiment configuration.

        Returns:
            (None, None) - backend manages model internally.
        """
        if not self._initialised:
            self._backend.initialize(config, self._runtime)
            self._initialised = True
        return (None, None)

    @property
    def backend(self) -> InferenceBackend:
        """Access to the initialised backend."""
        return self._backend


class BackendInferenceEngineAdapter:
    """Adapter that implements InferenceEngine protocol using InferenceBackend.

    Wraps the backend's run_inference() method and converts BackendResult
    to the format expected by the metrics collector.
    """

    def __init__(self, backend: InferenceBackend) -> None:
        """Initialise with backend.

        Args:
            backend: The (already initialised) inference backend.
        """
        self._backend = backend

    def run(
        self,
        model: Any,  # Ignored - backend manages model
        tokenizer: Any,  # Ignored - backend manages tokenizer
        prompts: list[str],
        config: ExperimentConfig,
    ) -> BackendInferenceResult:
        """Run inference via the backend.

        Args:
            model: Ignored (backend manages model internally).
            tokenizer: Ignored (backend manages tokenizer internally).
            prompts: List of prompts to process.
            config: Experiment configuration.

        Returns:
            BackendInferenceResult with metrics and original result.
        """
        from llenergymeasure.domain.metrics import InferenceMetrics

        result = self._backend.run_inference(prompts, config)

        # Convert BackendResult to InferenceMetrics
        metrics = InferenceMetrics(
            total_tokens=result.total_tokens,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            inference_time_sec=result.inference_time_sec,
            tokens_per_second=result.tokens_per_second,
            latency_per_token_ms=result.latency_per_token_ms,
            time_to_first_token_ms=result.time_to_first_token_ms,
            latency_measurements=result.latency_measurements,
        )

        return BackendInferenceResult(
            metrics=metrics,
            input_ids=None,  # vLLM doesn't expose input_ids
            backend_result=result,
        )


class BackendMetricsCollectorAdapter:
    """Adapter that implements MetricsCollector protocol for backends.

    Unlike PyTorch metrics collection which needs the model for FLOPs,
    this adapter extracts metrics directly from BackendResult and uses
    parameter-based FLOPs estimation when input_ids aren't available.
    """

    def __init__(self, backend: InferenceBackend) -> None:
        """Initialise with backend for model info access.

        Args:
            backend: The inference backend for model metadata.
        """
        self._backend = backend

    def collect(
        self,
        model: Any,  # Ignored - backend manages model
        inference_result: BackendInferenceResult,
        config: ExperimentConfig,
    ) -> CombinedMetrics:
        """Collect metrics from backend inference result.

        Args:
            model: Ignored (backend manages model internally).
            inference_result: Result from BackendInferenceEngineAdapter.
            config: Experiment configuration.

        Returns:
            Combined metrics (inference, energy placeholder, compute).
        """
        from llenergymeasure.domain.metrics import (
            CombinedMetrics,
            ComputeMetrics,
            EnergyMetrics,
        )

        # Get model info for FLOPs estimation
        model_info = self._backend.get_model_info()

        # Estimate FLOPs from parameters using 2*P*T rule
        # This is a low-confidence estimate but always available
        param_count = model_info.num_parameters
        total_tokens = inference_result.metrics.total_tokens
        flops_total = 2.0 * param_count * total_tokens if param_count > 0 else 0.0
        flops_method = "parameter_estimate"
        flops_confidence = "low"

        compute = ComputeMetrics(
            flops_total=flops_total,
            flops_per_token=flops_total / max(total_tokens, 1),
            flops_per_second=flops_total / max(inference_result.metrics.inference_time_sec, 0.001),
            peak_memory_mb=0.0,  # Not easily accessible from vLLM
            model_memory_mb=0.0,
            flops_method=flops_method,
            flops_confidence=flops_confidence,
            compute_precision=config.fp_precision,
        )

        # Return with placeholder energy - orchestrator fills this from energy backend
        return CombinedMetrics(
            inference=inference_result.metrics,
            energy=EnergyMetrics.placeholder(
                duration_sec=inference_result.metrics.inference_time_sec
            ),
            compute=compute,
        )
