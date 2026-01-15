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
    CudaManagement,
    LaunchMode,
    RuntimeCapabilities,
)
from llm_energy_measure.exceptions import (
    BackendInferenceError,
    BackendInitializationError,
)

if TYPE_CHECKING:
    from llm_energy_measure.config.backend_configs import PyTorchConfig
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

    PyTorch-specific configuration (config.pytorch) controls:
    - Attention implementation (flash_attention_2, sdpa, eager)
    - torch.compile optimization
    - BetterTransformer conversion
    - KV caching, memory management
    - Assisted generation (speculative decoding)
    """

    def __init__(self) -> None:
        """Initialize backend (model loaded lazily in initialize())."""
        self._model: Any = None
        self._tokenizer: Any = None
        self._accelerator: Any = None
        self._config: ExperimentConfig | None = None
        self._runtime: BackendRuntime | None = None
        self._assistant_model: Any = None  # For assisted generation

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

    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        """Return PyTorch/Accelerate runtime requirements.

        The orchestration layer manages CUDA context via Accelerate.
        torch.cuda.* calls are safe before this backend initializes.
        """
        return RuntimeCapabilities(
            launch_mode=LaunchMode.ACCELERATE,
            cuda_management=CudaManagement.ORCHESTRATOR,
            supports_tensor_parallel=True,
            supports_pipeline_parallel=False,
            manages_own_batching=False,
        )

    def _build_model_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs for model loading from PyTorchConfig.

        Args:
            config: Experiment configuration.

        Returns:
            Dict of kwargs for AutoModelForCausalLM.from_pretrained().
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        kwargs: dict[str, Any] = {}

        if pytorch_cfg is None:
            return kwargs

        # Attention implementation
        if pytorch_cfg.attn_implementation != "sdpa":
            kwargs["attn_implementation"] = pytorch_cfg.attn_implementation

        # Memory management
        if pytorch_cfg.low_cpu_mem_usage:
            kwargs["low_cpu_mem_usage"] = True
        if pytorch_cfg.max_memory:
            kwargs["max_memory"] = pytorch_cfg.max_memory

        # Escape hatch
        if pytorch_cfg.extra:
            kwargs.update(pytorch_cfg.extra)

        return kwargs

    def _build_generation_kwargs(self, config: ExperimentConfig) -> dict[str, Any]:
        """Build kwargs for model.generate() from PyTorchConfig.

        Args:
            config: Experiment configuration.

        Returns:
            Dict of additional kwargs for generate().
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        kwargs: dict[str, Any] = {}

        if pytorch_cfg is None:
            return kwargs

        # KV caching
        if not pytorch_cfg.use_cache:
            kwargs["use_cache"] = False

        # Beam search
        if pytorch_cfg.num_beams > 1:
            kwargs["num_beams"] = pytorch_cfg.num_beams
            if pytorch_cfg.early_stopping:
                kwargs["early_stopping"] = True
            if pytorch_cfg.length_penalty != 1.0:
                kwargs["length_penalty"] = pytorch_cfg.length_penalty

        # Output configuration
        if pytorch_cfg.output_scores:
            kwargs["output_scores"] = True
        if pytorch_cfg.return_dict_in_generate:
            kwargs["return_dict_in_generate"] = True

        # Assisted generation (speculative decoding)
        if pytorch_cfg.assisted_generation and pytorch_cfg.assisted_generation.model:
            # The assistant model is loaded separately and passed to generate()
            # This is handled in run_inference, not here
            pass

        return kwargs

    def _apply_torch_compile(self, config: ExperimentConfig) -> None:
        """Apply torch.compile to model if configured.

        Args:
            config: Experiment configuration.
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        if pytorch_cfg is None or not pytorch_cfg.torch_compile:
            return

        if self._model is None:
            return

        # Determine compilation mode
        mode = "default"
        if isinstance(pytorch_cfg.torch_compile, str):
            mode = pytorch_cfg.torch_compile

        logger.info(f"Applying torch.compile with mode='{mode}'")

        try:
            self._model = torch.compile(self._model, mode=mode)
            logger.info("torch.compile applied successfully")
        except Exception as e:
            logger.warning(f"torch.compile failed (non-fatal): {e}")

    def _apply_bettertransformer(self, config: ExperimentConfig) -> None:
        """Convert model to BetterTransformer if configured.

        Args:
            config: Experiment configuration.
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        if pytorch_cfg is None or not pytorch_cfg.use_bettertransformer:
            return

        if self._model is None:
            return

        logger.info("Converting model to BetterTransformer")

        try:
            self._model = self._model.to_bettertransformer()
            logger.info("BetterTransformer conversion successful")
        except Exception as e:
            logger.warning(f"BetterTransformer conversion failed (non-fatal): {e}")

    def _load_assistant_model(self, config: ExperimentConfig) -> None:
        """Load assistant model for assisted generation if configured.

        Args:
            config: Experiment configuration.
        """
        pytorch_cfg: PyTorchConfig | None = config.pytorch
        if pytorch_cfg is None:
            return

        assisted_cfg = pytorch_cfg.assisted_generation
        if assisted_cfg is None or not assisted_cfg.model:
            return

        from transformers import AutoModelForCausalLM

        logger.info(f"Loading assistant model for speculative decoding: {assisted_cfg.model}")

        try:
            self._assistant_model = AutoModelForCausalLM.from_pretrained(
                assisted_cfg.model,
                torch_dtype=getattr(torch, config.fp_precision),
                device_map="auto",
                trust_remote_code=True,
            )
            logger.info(f"Assistant model loaded: {assisted_cfg.model}")
        except Exception as e:
            logger.warning(f"Failed to load assistant model (non-fatal): {e}")
            self._assistant_model = None

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
            # Build model loading kwargs from PyTorchConfig
            model_kwargs = self._build_model_kwargs(config)
            if model_kwargs:
                logger.info(f"PyTorch config: {model_kwargs}")

            loader = HuggingFaceModelLoader()
            # Note: HuggingFaceModelLoader.load() currently doesn't accept extra kwargs
            # TODO: Pass model_kwargs to loader when supported
            self._model, self._tokenizer = loader.load(config)
            logger.info(f"Model loaded: {config.model_name}")

            # Apply post-load optimizations
            self._apply_bettertransformer(config)
            self._apply_torch_compile(config)

            # Load assistant model for speculative decoding if configured
            self._load_assistant_model(config)

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

        if self._assistant_model is not None:
            del self._assistant_model
            self._assistant_model = None

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
        warnings: list[ConfigWarning] = []

        pytorch_cfg = config.pytorch
        if pytorch_cfg is None:
            return warnings

        # Check for flash_attention_2 - requires flash-attn package
        if pytorch_cfg.attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                warnings.append(
                    ConfigWarning(
                        param="pytorch.attn_implementation",
                        message=(
                            "flash_attention_2 requires the flash-attn package. "
                            "Falling back to sdpa."
                        ),
                        severity="warning",
                        suggestion="pip install flash-attn or use attn_implementation='sdpa'",
                    )
                )

        # BetterTransformer is deprecated with sdpa/flash_attention
        if pytorch_cfg.use_bettertransformer and pytorch_cfg.attn_implementation != "eager":
            warnings.append(
                ConfigWarning(
                    param="pytorch.use_bettertransformer",
                    message=(
                        "BetterTransformer is deprecated and may conflict with "
                        f"attn_implementation='{pytorch_cfg.attn_implementation}'. "
                        "Consider removing use_bettertransformer."
                    ),
                    severity="warning",
                )
            )

        # torch.compile with BetterTransformer is problematic
        if pytorch_cfg.torch_compile and pytorch_cfg.use_bettertransformer:
            warnings.append(
                ConfigWarning(
                    param="pytorch.torch_compile",
                    message="torch.compile and BetterTransformer together may cause issues.",
                    severity="warning",
                )
            )

        return warnings
