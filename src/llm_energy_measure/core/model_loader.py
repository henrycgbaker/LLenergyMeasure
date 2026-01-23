"""Model and tokenizer loading utilities for LLM Bench."""

from __future__ import annotations

import importlib
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger
from packaging import version
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer

from llm_energy_measure.config.backend_configs import PyTorchConfig
from llm_energy_measure.config.models import ExperimentConfig
from llm_energy_measure.exceptions import ConfigurationError


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

    For PyTorch backend, supports optional BitsAndBytes quantization via
    the pytorch config section.

    Args:
        config: Experiment configuration with model settings.

    Returns:
        Tuple of (model, tokenizer).

    Raises:
        ConfigurationError: If model loading fails.
    """
    model_name = config.model_name
    fp_precision = config.fp_precision
    pytorch_cfg = config.pytorch

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

    # Build model kwargs
    model_kwargs: dict[str, Any] = {"device_map": "auto"}

    # Merge PyTorch-specific model kwargs (if config.pytorch is set)
    if pytorch_cfg is not None:
        # Attention implementation
        if pytorch_cfg.attn_implementation != "sdpa":
            model_kwargs["attn_implementation"] = pytorch_cfg.attn_implementation

        # Memory management
        if pytorch_cfg.low_cpu_mem_usage:
            model_kwargs["low_cpu_mem_usage"] = True
        if pytorch_cfg.max_memory:
            model_kwargs["max_memory"] = pytorch_cfg.max_memory

        # Escape hatch for extra kwargs
        if pytorch_cfg.extra:
            model_kwargs.update(pytorch_cfg.extra)

    # Load model with optional quantization
    try:
        # Check for BitsAndBytes quantization in pytorch config
        use_quantization = pytorch_cfg and (pytorch_cfg.load_in_4bit or pytorch_cfg.load_in_8bit)

        if use_quantization and pytorch_cfg:
            # Quantization uses its own loading path
            model = _load_quantized_model(model_name, pytorch_cfg)
        else:
            # Standard loading with dtype
            model_kwargs["torch_dtype"] = dtype
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
    except Exception as e:
        raise ConfigurationError(f"Failed to load model {model_name}: {e}") from e

    # Load and merge LoRA adapter if specified
    if config.adapter:
        model = _load_and_merge_adapter(model, config.adapter, dtype)

    # Get model dtype defensively (some wrappers may not expose .dtype directly)
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is None:
        try:
            model_dtype = next(model.parameters()).dtype
        except StopIteration:
            model_dtype = "unknown"

    logger.info(
        f"Model loaded: {model_name}, dtype={model_dtype}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    return model, tokenizer


def _load_quantized_model(
    model_name: str,
    pytorch_cfg: PyTorchConfig,
) -> PreTrainedModel:
    """Load a model with BitsAndBytes quantization.

    Args:
        model_name: HuggingFace model name or path.
        pytorch_cfg: PyTorch backend configuration with quantization settings.

    Returns:
        Quantized model.
    """
    qsupport = detect_quantization_support()

    bnb_kwargs: dict[str, Any] = {}

    if pytorch_cfg.load_in_4bit:
        if not qsupport.supports_4bit:
            raise ConfigurationError(
                "4-bit quantization requested but not supported by bitsandbytes version"
            )
        bnb_kwargs["load_in_4bit"] = True
        # Use compute dtype from config, default to float16
        compute_dtype_str = pytorch_cfg.bnb_4bit_compute_dtype
        compute_dtype = torch.bfloat16 if compute_dtype_str == "bfloat16" else torch.float16
        bnb_kwargs["bnb_4bit_compute_dtype"] = compute_dtype
        bnb_kwargs["bnb_4bit_quant_type"] = pytorch_cfg.bnb_4bit_quant_type
        if pytorch_cfg.bnb_4bit_use_double_quant:
            bnb_kwargs["bnb_4bit_use_double_quant"] = True
        logger.info(f"Using 4-bit quantization with {pytorch_cfg.bnb_4bit_quant_type}")

    if pytorch_cfg.load_in_8bit:
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
        device_map="auto",
    )


def _load_and_merge_adapter(
    model: PreTrainedModel,
    adapter_path: str,
    dtype: torch.dtype,
) -> PreTrainedModel:
    """Load LoRA adapter and merge weights into base model.

    Uses PEFT library to load adapter, then merges weights permanently.
    After merging, the model behaves as a standard HuggingFace model.

    Args:
        model: Base model to apply adapter to.
        adapter_path: HuggingFace Hub ID or local path to adapter.
        dtype: Torch dtype for adapter loading.

    Returns:
        Model with merged adapter weights.

    Raises:
        ConfigurationError: If PEFT not installed or adapter load fails.
    """
    try:
        from peft import PeftModel
    except ImportError as e:
        raise ConfigurationError(
            "LoRA adapter loading requires the 'peft' package. " "Install with: pip install peft"
        ) from e

    logger.info(f"Loading LoRA adapter: {adapter_path}")

    try:
        # Load adapter weights
        model = PeftModel.from_pretrained(  # type: ignore[assignment]
            model,
            adapter_path,
            torch_dtype=dtype,
        )

        # Merge and unload - returns standard HF model
        model = model.merge_and_unload()  # type: ignore[operator]

        logger.info(f"LoRA adapter merged: {adapter_path}")
        return model

    except Exception as e:
        raise ConfigurationError(f"Failed to load LoRA adapter '{adapter_path}': {e}") from e
