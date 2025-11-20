"""
Model and tokenizer loading with support for various precision and quantization methods.
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


def _load_tokenizer(model_name: str) -> PreTrainedTokenizer:
    """
    Load tokenizer from Hugging Face.

    Args:
        model_name: Model name or path

    Returns:
        Tokenizer instance
    """
    logger.debug("Loading tokenizer: %s", model_name)

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

    return tokenizer


def _load_standard_model(config: ExperimentConfig) -> PreTrainedModel:
    """
    Load model without quantization.

    Args:
        config: Experiment configuration

    Returns:
        Model instance
    """
    # Map precision to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float8": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.float16,
    }

    dtype = dtype_map.get(config.precision, torch.float16)

    if config.precision == "float8" and not hasattr(torch, "float8_e4m3fn"):
        logger.warning("float8 not supported, falling back to float16")

    logger.debug("Loading with dtype: %s", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    return model


def _load_quantized_model(config: ExperimentConfig) -> PreTrainedModel:
    """
    Load model with quantization.

    Args:
        config: Experiment configuration

    Returns:
        Quantized model instance

    Raises:
        ValueError: If quantization configuration is invalid
    """
    quant_support = detect_quantization_support()

    if not quant_support["supports_4bit"] and not quant_support["supports_8bit"]:
        raise ValueError(
            "Quantization requested but bitsandbytes not installed or version too old. "
            f"Install with: pip install bitsandbytes>=0.39.0"
        )

    # Validate quantization settings
    if config.quantization.load_in_4bit and not quant_support["supports_4bit"]:
        raise ValueError(
            f"4-bit quantization not supported (bitsandbytes={quant_support['version']}). "
            "Requires bitsandbytes>=0.39.0"
        )

    if config.quantization.load_in_8bit and not quant_support["supports_8bit"]:
        raise ValueError(
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

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )

    return model


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
