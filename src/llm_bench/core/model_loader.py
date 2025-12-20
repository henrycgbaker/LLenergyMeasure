"""Model and tokenizer loading utilities for LLM Bench."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from packaging import version

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_bench.config.models import ExperimentConfig, QuantizationConfig
from llm_bench.exceptions import ConfigurationError


@dataclass
class QuantizationSupport:
    """Information about supported quantization types."""

    supports_4bit: bool
    supports_8bit: bool
    default_4bit_quant_type: str | None
    default_8bit_quant_type: str | None


class ModelWrapper(torch.nn.Module):  # type: ignore[misc]
    """Wrapper for HuggingFace models to ensure standard nn.Module interface.

    Some models from HuggingFace transformers aren't in standard nn.Module format.
    This wrapper ensures compatibility with tools like ptflops.
    """

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> Any:
        return self.model(input_ids=input_ids)


def detect_quantization_support() -> QuantizationSupport:
    """Detect available quantization types based on bitsandbytes version.

    Returns:
        QuantizationSupport with available quantization options.
    """
    try:
        bnb = importlib.import_module("bitsandbytes")
        bnb_version = getattr(bnb, "__version__", "0.0.0")
    except ImportError:
        warnings.warn(
            "bitsandbytes is not installed. Defaulting to no quantization.",
            stacklevel=2,
        )
        return QuantizationSupport(
            supports_4bit=False,
            supports_8bit=False,
            default_4bit_quant_type=None,
            default_8bit_quant_type=None,
        )

    parsed_version = version.parse(bnb_version)

    supports_4bit = parsed_version >= version.parse("0.39.0")  # QLoRA support
    supports_8bit = parsed_version >= version.parse("0.38.0")

    quant_type_4bit = "nf4" if supports_4bit else None
    quant_type_8bit = "fp8" if supports_8bit else "int8"

    logger.debug(f"bitsandbytes {bnb_version}: 4bit={supports_4bit}, 8bit={supports_8bit}")

    return QuantizationSupport(
        supports_4bit=supports_4bit,
        supports_8bit=supports_8bit,
        default_4bit_quant_type=quant_type_4bit,
        default_8bit_quant_type=quant_type_8bit,
    )


def get_torch_dtype(fp_precision: str) -> torch.dtype:
    """Convert precision string to torch dtype.

    Args:
        fp_precision: One of 'float8', 'float16', 'bfloat16', 'float32'.

    Returns:
        Corresponding torch.dtype.
    """
    dtype_map = {
        "float8": torch.float8_e4m3fn,  # Most common float8 variant
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(fp_precision, torch.float16)


def load_model_tokenizer(
    config: ExperimentConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a model and tokenizer from HuggingFace.

    Args:
        config: Experiment configuration with model settings.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ConfigurationError: If model loading fails.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = config.model_name
    fp_precision = config.fp_precision
    quant_config = config.quantization_config

    logger.info(f"Loading model: {model_name}")

    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    except Exception as e:
        raise ConfigurationError(f"Failed to load tokenizer for {model_name}: {e}") from e

    # Determine dtype
    dtype = get_torch_dtype(fp_precision)

    # Load model with optional quantization
    try:
        if quant_config and quant_config.quantization:
            model = _load_quantized_model(model_name, quant_config)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
            )
    except Exception as e:
        raise ConfigurationError(f"Failed to load model {model_name}: {e}") from e

    logger.info(
        f"Model loaded: {model_name}, dtype={model.dtype}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    return model, tokenizer


def _load_quantized_model(
    model_name: str,
    quant_config: QuantizationConfig,
) -> PreTrainedModel:
    """Load a model with BitsAndBytes quantization.

    Args:
        model_name: HuggingFace model name or path.
        quant_config: Quantization configuration.

    Returns:
        Quantized model.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    qsupport = detect_quantization_support()

    bnb_kwargs: dict[str, Any] = {}

    if quant_config.load_in_4bit:
        if not qsupport.supports_4bit:
            raise ConfigurationError(
                "4-bit quantization requested but not supported by bitsandbytes version"
            )
        bnb_kwargs["load_in_4bit"] = True
        bnb_kwargs["bnb_4bit_compute_dtype"] = torch.float16
        bnb_kwargs["bnb_4bit_quant_type"] = qsupport.default_4bit_quant_type
        logger.info(f"Using 4-bit quantization with {qsupport.default_4bit_quant_type}")

    if quant_config.load_in_8bit:
        if not qsupport.supports_8bit:
            raise ConfigurationError(
                "8-bit quantization requested but not supported by bitsandbytes version"
            )
        bnb_kwargs["load_in_8bit"] = True
        bnb_kwargs["bnb_8bit_compute_dtype"] = torch.float16
        bnb_kwargs["bnb_8bit_quant_type"] = qsupport.default_8bit_quant_type
        logger.info(f"Using 8-bit quantization with {qsupport.default_8bit_quant_type}")

    bnb_config = BitsAndBytesConfig(**bnb_kwargs)

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
    )
