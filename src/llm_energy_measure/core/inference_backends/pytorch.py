"""PyTorch/Transformers inference backend.

This backend wraps the existing HuggingFace Transformers + Accelerate implementation,
providing backward compatibility while conforming to the InferenceBackend protocol.

Design:
- Composes existing ModelLoader and InferenceEngine implementations
- Converts internal InferenceResult to protocol-agnostic BackendResult
- Uses Accelerate for distributed execution and device management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

from llm_energy_measure.core.inference_backends.protocols import (
    BackendResult,
    BackendRuntime,
    ConfigWarning,
)
from llm_energy_measure.exceptions import (
    BackendInferenceError,
    BackendInitializationError,
)

if TYPE_CHECKING:
    from llm_energy_measure.config.models import ExperimentConfig
    from llm_energy_measure.domain.model_info import ModelInfo


# Parameters supported by PyTorch/Transformers backend
_SUPPORTED_PARAMS: set[str] = {
    # Core
    "model_name",
    "fp_precision",
    "max_input_tokens",
    "max_output_tokens",
    "min_output_tokens",
    "random_seed",
    # Batching
    "batch_size",
    "batching.batch_size",
    "batching.strategy",
    "batching.max_tokens_per_batch",
    # Decoder/Generation
    "decoder.preset",
    "decoder.temperature",
    "decoder.do_sample",
    "decoder.top_p",
    "decoder.top_k",
    "decoder.min_p",
    "decoder.repetition_penalty",
    "decoder.no_repeat_ngram_size",
    # Quantization (BitsAndBytes)
    "quantization.load_in_4bit",
    "quantization.load_in_8bit",
    "quantization.quantization",
    # Sharding (via Accelerate)
    "sharding.strategy",
    "sharding.num_shards",
    # Traffic simulation
    "traffic_simulation.enabled",
    "traffic_simulation.mode",
    "traffic_simulation.target_qps",
    # Other
    "save_outputs",
    "num_input_prompts",
    "gpus",
    "num_processes",
}


class PyTorchBackend:
    """HuggingFace Transformers inference backend.

    This backend composes existing implementations:
    - HuggingFaceModelLoader for model/tokenizer loading
    - TransformersInferenceEngine for inference
    - ThroughputMetricsCollector for metrics

    It provides full backward compatibility with existing experiments while
    conforming to the InferenceBackend protocol.
    """

    def __init__(self) -> None:
        """Initialize backend (model loaded lazily in initialize())."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._accelerator: Any = None
        self._config: ExperimentConfig | None = None
        self._runtime: BackendRuntime | None = None

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "pytorch"

    @property
    def version(self) -> str:
        """Backend version string."""
        import transformers

        return f"transformers={transformers.__version__}, torch={torch.__version__}"

    def is_available(self) -> bool:
        """Check if PyTorch and Transformers are available."""
        try:
            import accelerate  # noqa: F401
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def initialize(self, config: ExperimentConfig, runtime: BackendRuntime) -> None:
        """Load model and prepare for inference.

        Args:
            config: Experiment configuration.
            runtime: Runtime context with accelerator.

        Raises:
            BackendInitializationError: If model loading fails.
        """
        from llm_energy_measure.core.implementations import HuggingFaceModelLoader

        self._config = config
        self._runtime = runtime
        self._accelerator = runtime.accelerator

        if self._accelerator is None:
            raise BackendInitializationError(
                "PyTorch backend requires an Accelerator instance in BackendRuntime. "
                "This backend uses HuggingFace Accelerate for distributed execution."
            )

        try:
            loader = HuggingFaceModelLoader()
            self._model, self._tokenizer = loader.load(config)
            logger.info(f"Model loaded: {config.model_name}")
        except Exception as e:
            raise BackendInitializationError(
                f"Failed to load model '{config.model_name}': {e}"
            ) from e

    def run_inference(self, prompts: list[str], config: ExperimentConfig) -> BackendResult:
        """Run inference using Transformers.

        Args:
            prompts: List of input prompts.
            config: Experiment configuration.

        Returns:
            BackendResult with token counts and timing.

        Raises:
            BackendInferenceError: If inference fails.
        """
        from llm_energy_measure.core.inference import run_inference

        if self._model is None or self._tokenizer is None:
            raise BackendInferenceError("Backend not initialized. Call initialize() first.")

        try:
            # Use existing inference implementation
            result = run_inference(
                model=self._model,
                config=config,
                prompts=prompts,
                tokenizer=self._tokenizer,
                accelerator=self._accelerator,
            )

            # Convert to BackendResult
            return BackendResult(
                total_tokens=result.metrics.total_tokens,
                input_tokens=result.metrics.input_tokens,
                output_tokens=result.metrics.output_tokens,
                inference_time_sec=result.metrics.inference_time_sec,
                batch_latencies_ms=[],  # TODO: Extract from result if available
                output_texts=None,  # TODO: Decode if save_outputs is True
                backend_metadata={
                    "backend": self.name,
                    "version": self.version,
                    "tokens_per_second": result.metrics.tokens_per_second,
                    "latency_per_token_ms": result.metrics.latency_per_token_ms,
                },
            )
        except Exception as e:
            raise BackendInferenceError(f"Inference failed: {e}") from e

    def cleanup(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.debug("PyTorch backend cleaned up")

    def get_model_info(self) -> ModelInfo:
        """Return model metadata.

        Returns:
            ModelInfo with parameter count, architecture details.
        """
        from llm_energy_measure.domain.model_info import ModelInfo, QuantizationSpec

        if self._model is None or self._config is None:
            raise BackendInferenceError("Backend not initialized. Call initialize() first.")

        model_config = self._model.config

        # Determine quantization
        quant_config = self._config.quantization_config
        if quant_config.load_in_4bit:
            quant = QuantizationSpec(
                enabled=True, bits=4, method="bitsandbytes", compute_dtype="float16"
            )
        elif quant_config.load_in_8bit:
            quant = QuantizationSpec(
                enabled=True, bits=8, method="bitsandbytes", compute_dtype="float16"
            )
        else:
            quant = QuantizationSpec(
                enabled=False, bits=None, method="none", compute_dtype=self._config.fp_precision
            )

        return ModelInfo(
            name=self._config.model_name,
            revision=None,
            num_parameters=sum(p.numel() for p in self._model.parameters()),
            num_layers=getattr(model_config, "num_hidden_layers", 0),
            hidden_size=getattr(model_config, "hidden_size", 0),
            num_attention_heads=getattr(model_config, "num_attention_heads", 0),
            vocab_size=getattr(model_config, "vocab_size", 0),
            model_type=getattr(model_config, "model_type", "unknown"),
            torch_dtype=str(self._model.dtype),
            quantization=quant,
        )

    def get_supported_params(self) -> set[str]:
        """Return parameters supported by this backend."""
        return _SUPPORTED_PARAMS.copy()

    def validate_config(self, config: ExperimentConfig) -> list[ConfigWarning]:
        """Validate config compatibility with PyTorch backend.

        Only returns warnings for actual incompatibilities or problems,
        not informational notes about normal backend behaviour.

        Args:
            config: Configuration to validate.

        Returns:
            List of warnings/errors for config problems. Empty for valid configs.
        """
        # PyTorch/Transformers backend supports all standard config options.
        # No warnings needed for normal usage.
        return []
