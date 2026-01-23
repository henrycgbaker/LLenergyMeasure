"""Shared utilities and constants for inference backends.

This module provides common functionality used across all inference backends,
reducing code duplication and ensuring consistent behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from llm_energy_measure.config.models import ExperimentConfig


# Core parameters supported by all backends (Tier 1 universal params)
# Individual backends extend this with backend-specific params from their sections
CORE_SUPPORTED_PARAMS: frozenset[str] = frozenset(
    {
        # Core config
        "model_name",
        "fp_precision",
        "max_input_tokens",
        "max_output_tokens",
        "min_output_tokens",
        "random_seed",
        # Decoder/Generation (universal)
        "decoder.preset",
        "decoder.temperature",
        "decoder.top_p",
        "decoder.do_sample",
        "decoder.repetition_penalty",
        # Execution
        "num_input_prompts",
        "gpus",
        "save_outputs",
        # Streaming
        "streaming",
        "streaming_warmup_requests",
    }
)


def check_statistical_sufficiency(
    num_samples: int,
    measurement_type: str = "latency",
    min_recommended: int = 10,
) -> str | None:
    """Check if sample size is statistically sufficient.

    Args:
        num_samples: Number of samples collected.
        measurement_type: Type of measurement for warning message.
        min_recommended: Minimum recommended sample size.

    Returns:
        Warning message if sample size is too low, None otherwise.
    """
    if num_samples < min_recommended:
        return (
            f"Low sample size for {measurement_type} measurement "
            f"({num_samples} < {min_recommended}). "
            "Results may not be statistically reliable."
        )
    return None


def estimate_ttft_from_request_time(
    total_request_time_sec: float,
    num_output_tokens: int,
) -> float:
    """Estimate TTFT when streaming is not available.

    Uses a simple heuristic: first token takes ~10% of total time
    for a typical generation. This is a rough estimate only.

    Args:
        total_request_time_sec: Total time for the request.
        num_output_tokens: Number of tokens generated.

    Returns:
        Estimated TTFT in seconds.
    """
    if num_output_tokens <= 1:
        return total_request_time_sec

    # Rough heuristic: first token latency is higher than average
    # Typical pattern is TTFT takes ~10-20% of total time
    avg_per_token = total_request_time_sec / num_output_tokens
    estimated_ttft = avg_per_token * 2.0  # First token typically slower

    return min(estimated_ttft, total_request_time_sec * 0.5)


def log_warmup_progress(
    current: int,
    total: int,
    measurement_type: str = "warmup",
) -> None:
    """Log warmup progress.

    Args:
        current: Current warmup iteration (1-indexed).
        total: Total warmup iterations.
        measurement_type: Type of warmup for logging.
    """
    if current == 1:
        logger.info(f"Running {total} {measurement_type} iterations...")
    elif current == total:
        logger.info(f"{measurement_type.capitalize()} complete")


def validate_streaming_config(config: ExperimentConfig) -> str | None:
    """Validate streaming configuration.

    Args:
        config: Experiment configuration.

    Returns:
        Warning message if configuration is problematic, None otherwise.
    """
    if config.streaming:
        if config.streaming_warmup_requests < 1:
            return "streaming_warmup_requests must be >= 1 when streaming is enabled"

        # Check batch_size from backend-specific config
        batch_size = 1
        if config.pytorch and config.pytorch.batch_size:
            batch_size = config.pytorch.batch_size

        if batch_size > 1:
            return (
                "Streaming latency measurement with batch_size > 1 may not "
                "accurately measure per-request TTFT/ITL"
            )

    return None


def get_precision_dtype_str(precision: str) -> str:
    """Convert precision string to dtype string.

    Args:
        precision: Precision string (e.g., "float16", "bfloat16").

    Returns:
        Dtype string suitable for backends.
    """
    precision_map = {
        "float32": "float32",
        "float16": "float16",
        "bfloat16": "bfloat16",
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    return precision_map.get(precision.lower(), "float16")


def create_precision_metadata(
    config: ExperimentConfig,
    backend: str,
    actual_compute_dtype: str | None = None,
) -> PrecisionMetadata:
    """Create PrecisionMetadata from config and runtime info.

    Args:
        config: Experiment configuration.
        backend: Backend name ('pytorch', 'vllm', 'tensorrt').
        actual_compute_dtype: Actual compute dtype if different from config.

    Returns:
        PrecisionMetadata with populated fields.
    """
    from llm_energy_measure.domain.metrics import PrecisionMetadata

    # Normalise precision string to metric format (must match PrecisionMetadata Literals)
    precision_str = config.fp_precision.lower()
    if precision_str in ("float32", "fp32"):
        precision_str = "fp32"
    elif precision_str in ("float16", "fp16"):
        precision_str = "fp16"
    elif precision_str in ("bfloat16", "bf16"):
        precision_str = "bf16"

    # Determine weights precision (affected by quantization)
    quantization_method: str | None = None
    weights_precision = precision_str

    # Check backend-specific quantization configs
    pytorch_cfg = config.pytorch
    vllm_cfg = config.vllm
    tensorrt_cfg = config.tensorrt

    if backend == "pytorch" and pytorch_cfg:
        if pytorch_cfg.load_in_4bit:
            weights_precision = "int4"
            quantization_method = "bitsandbytes"
        elif pytorch_cfg.load_in_8bit:
            weights_precision = "int8"
            quantization_method = "bitsandbytes"
    elif backend == "vllm" and vllm_cfg and vllm_cfg.quantization:
        quantization_method = vllm_cfg.quantization
        # GPTQ/AWQ models use int4/int8 weights with fp16 compute
        if vllm_cfg.quantization in ("gptq", "awq", "marlin", "squeezellm"):
            weights_precision = "int4"
    elif backend == "tensorrt" and tensorrt_cfg and tensorrt_cfg.quantization != "none":
        quantization_method = tensorrt_cfg.quantization
        if tensorrt_cfg.quantization in ("int4_awq", "int4_gptq"):
            weights_precision = "int4"
        elif tensorrt_cfg.quantization in ("int8_sq", "int8_weight_only"):
            weights_precision = "int8"

    # Helper to normalise dtype string to PrecisionMetadata format
    def normalise_dtype(dtype: str) -> str:
        dtype = dtype.lower().replace("torch.", "")
        if dtype in ("float32", "fp32"):
            return "fp32"
        if dtype in ("float16", "fp16"):
            return "fp16"
        if dtype in ("bfloat16", "bf16"):
            return "bf16"
        return dtype

    # Compute precision - for BitsAndBytes, compute happens at fp16 after dequant
    if actual_compute_dtype:
        compute_precision = normalise_dtype(actual_compute_dtype)
    elif quantization_method == "bitsandbytes" and pytorch_cfg:
        # BitsAndBytes dequantizes to fp16 for compute
        compute_precision = normalise_dtype(pytorch_cfg.bnb_4bit_compute_dtype)
    else:
        compute_precision = precision_str

    # Activations typically match compute precision
    activations_precision = compute_precision

    return PrecisionMetadata(
        weights=weights_precision,  # type: ignore[arg-type]
        activations=activations_precision,  # type: ignore[arg-type]
        compute=compute_precision,  # type: ignore[arg-type]
        quantization_method=quantization_method,
    )


# Type alias for typing import
if TYPE_CHECKING:
    from llm_energy_measure.domain.metrics import PrecisionMetadata
