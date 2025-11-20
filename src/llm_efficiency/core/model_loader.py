"""
Model and tokenizer loading with support for various precision and quantization methods.

Features:
- Automatic retry for network failures
- Comprehensive error handling and validation
- Support for multiple precision types (float32, float16, bfloat16, float8)
- 4-bit and 8-bit quantization with bitsandbytes
- Informative logging and error messages
"""

import logging
import warnings
from typing import Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from packaging import version

from llm_efficiency.config import ExperimentConfig
from llm_efficiency.utils.exceptions import ModelLoadingError, QuantizationError
from llm_efficiency.utils.retry import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


def detect_quantization_support() -> dict:
    """
    Detect supported quantization types based on bitsandbytes version.

    Returns:
        Dictionary with support information and default types
    """
    try:
        import bitsandbytes as bnb

        bnb_version = getattr(bnb, "__version__", "0.0.0")
        parsed_version = version.parse(bnb_version)

        supports_4bit = parsed_version >= version.parse("0.39.0")
        supports_8bit = parsed_version >= version.parse("0.38.0")

        return {
            "supports_4bit": supports_4bit,
            "supports_8bit": supports_8bit,
            "default_4bit_type": "nf4" if supports_4bit else None,
            "default_8bit_type": "int8" if supports_8bit else None,
            "version": bnb_version,
        }
    except ImportError:
        warnings.warn("bitsandbytes not installed. Quantization disabled.")
        return {
            "supports_4bit": False,
            "supports_8bit": False,
            "default_4bit_type": None,
            "default_8bit_type": None,
            "version": None,
        }


def load_model_and_tokenizer(
    config: ExperimentConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load model and tokenizer from Hugging Face with specified configuration.

    Args:
        config: Experiment configuration

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ValueError: If configuration is invalid
        OSError: If model cannot be loaded
    """
    logger.info("Loading model: %s", config.model_name)
    logger.info("Precision: %s", config.precision)
    logger.info("Quantization: %s", config.quantization.enabled)

    # Load tokenizer
    tokenizer = _load_tokenizer(config.model_name)

    # Load model with appropriate settings
    if config.quantization.enabled:
        model = _load_quantized_model(config)
    else:
        model = _load_standard_model(config)

    logger.info("Model loaded successfully")
    logger.info("Model parameters: %s", sum(p.numel() for p in model.parameters()))

    return model, tokenizer


@retry_with_exponential_backoff(
    max_retries=4,
    initial_delay=2.0,
    exceptions=(OSError, ConnectionError, TimeoutError),
)
def _load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load tokenizer from Hugging Face with automatic retry.

    Automatically retries on network failures with exponential backoff.

    Args:
        model_name: Model name or path

    Returns:
        Tokenizer instance

    Raises:
        ModelLoadingError: If tokenizer cannot be loaded after retries
    """
    logger.debug("Loading tokenizer: %s", model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token")

        # Set padding side to left (better for generation)
        tokenizer.padding_side = "left"

        logger.info("Tokenizer loaded successfully")
        return tokenizer

    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}': {e}")
        raise ModelLoadingError(
            f"Could not load tokenizer '{model_name}'. "
            f"Check model name is correct and you have network access. Error: {e}"
        ) from e


@retry_with_exponential_backoff(
    max_retries=4,
    initial_delay=2.0,
    exceptions=(OSError, ConnectionError, TimeoutError),
)
def _load_standard_model(config: ExperimentConfig) -> PreTrainedModel:
    """
    Load model without quantization.

    Automatically retries on network failures with exponential backoff.

    Args:
        config: Experiment configuration

    Returns:
        Model instance

    Raises:
        ModelLoadingError: If model cannot be loaded after retries
    """
    # Map precision to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
    }

    dtype = dtype_map.get(config.precision, torch.float16)

    if config.precision not in dtype_map:
        logger.warning(
            f"Unknown precision '{config.precision}', using float16. "
            f"Valid options: {list(dtype_map.keys())}"
        )

    if config.precision == "float8" and not hasattr(torch, "float8_e4m3fn"):
        logger.warning("float8 not supported in this PyTorch version, falling back to float16")

    logger.debug("Loading with dtype: %s", dtype)

    try:
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        logger.debug(f"Using device_map: {device_map}")

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map=device_map,
        )

        logger.info("Model loaded successfully")
        return model

    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU out of memory during model loading")
        raise ModelLoadingError(
            f"GPU ran out of memory loading {config.model_name}. "
            f"Try using a smaller model, quantization, or CPU device. Error: {e}"
        ) from e

    except Exception as e:
        logger.error(f"Failed to load model '{config.model_name}': {e}")
        raise ModelLoadingError(
            f"Could not load model '{config.model_name}'. "
            f"Check model name is correct, you have network access, "
            f"and sufficient memory/disk space. Error: {e}"
        ) from e


@retry_with_exponential_backoff(
    max_retries=4,
    initial_delay=2.0,
    exceptions=(OSError, ConnectionError, TimeoutError),
)
def _load_quantized_model(config: ExperimentConfig) -> PreTrainedModel:
    """
    Load model with quantization.

    Automatically retries on network failures with exponential backoff.

    Args:
        config: Experiment configuration

    Returns:
        Quantized model instance

    Raises:
        QuantizationError: If quantization configuration is invalid or not supported
        ModelLoadingError: If model cannot be loaded after retries
    """
    quant_support = detect_quantization_support()

    if not quant_support["supports_4bit"] and not quant_support["supports_8bit"]:
        raise QuantizationError(
            "Quantization requested but bitsandbytes not installed or version too old. "
            f"Install with: pip install bitsandbytes>=0.39.0"
        )

    # Check for CUDA availability
    if not torch.cuda.is_available():
        raise QuantizationError(
            "Quantization requires CUDA/GPU. CPU-only quantization is not supported by bitsandbytes."
        )

    # Validate quantization settings
    if config.quantization.load_in_4bit and not quant_support["supports_4bit"]:
        raise QuantizationError(
            f"4-bit quantization not supported (bitsandbytes={quant_support['version']}). "
            "Requires bitsandbytes>=0.39.0"
        )

    if config.quantization.load_in_8bit and not quant_support["supports_8bit"]:
        raise QuantizationError(
            f"8-bit quantization not supported (bitsandbytes={quant_support['version']}). "
            "Requires bitsandbytes>=0.38.0"
        )

    # Build quantization config
    bnb_config_dict = {
        "load_in_4bit": config.quantization.load_in_4bit,
        "load_in_8bit": config.quantization.load_in_8bit,
    }

    # Add 4-bit specific settings
    if config.quantization.load_in_4bit:
        compute_dtype = (
            torch.float16
            if config.quantization.compute_dtype == "float16"
            else torch.bfloat16
        )
        bnb_config_dict.update(
            {
                "bnb_4bit_compute_dtype": compute_dtype,
                "bnb_4bit_quant_type": config.quantization.quant_type
                or quant_support["default_4bit_type"],
                "bnb_4bit_use_double_quant": True,  # Better quality
            }
        )
        logger.info("4-bit quantization: type=%s", bnb_config_dict["bnb_4bit_quant_type"])

    # Add 8-bit specific settings
    if config.quantization.load_in_8bit:
        compute_dtype = (
            torch.float16
            if config.quantization.compute_dtype == "float16"
            else torch.bfloat16
        )
        bnb_config_dict.update(
            {
                "llm_int8_enable_fp32_cpu_offload": False,
            }
        )
        logger.info("8-bit quantization enabled")

    bnb_config = BitsAndBytesConfig(**bnb_config_dict)

    logger.debug("Loading quantized model with config: %s", bnb_config_dict)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto",
        )

        logger.info("Quantized model loaded successfully")
        return model

    except torch.cuda.OutOfMemoryError as e:
        logger.error("GPU out of memory during quantized model loading")
        raise ModelLoadingError(
            f"GPU ran out of memory loading quantized {config.model_name}. "
            f"Try using 8-bit instead of 4-bit, or a smaller model. Error: {e}"
        ) from e

    except ImportError as e:
        logger.error("Missing dependency for quantization")
        raise QuantizationError(
            f"Missing required library for quantization. "
            f"Install with: pip install bitsandbytes>=0.39.0. Error: {e}"
        ) from e

    except Exception as e:
        logger.error(f"Failed to load quantized model '{config.model_name}': {e}")
        raise ModelLoadingError(
            f"Could not load quantized model '{config.model_name}'. "
            f"Check model name, network access, and GPU compatibility. Error: {e}"
        ) from e


# Backward compatibility wrapper for v1.0
def load_model_tokenizer(configs: dict) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Legacy wrapper for v1.0 compatibility.

    Args:
        configs: Dictionary configuration (v1.0 format)

    Returns:
        Tuple of (model, tokenizer)

    .. deprecated:: 2.0
        Use :func:`load_model_and_tokenizer` with ExperimentConfig instead.
    """
    warnings.warn(
        "load_model_tokenizer with dict config is deprecated. "
        "Use load_model_and_tokenizer with ExperimentConfig instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from llm_efficiency.config import ExperimentConfig

    # Convert dict to config if needed
    if isinstance(configs, dict):
        config = ExperimentConfig.from_legacy_dict(configs)
    else:
        config = configs

    return load_model_and_tokenizer(config)
