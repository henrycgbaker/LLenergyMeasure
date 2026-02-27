"""Backend-specific configuration models (v2.0 schema, M1 minimal).

Each backend section uses None-as-default: all fields default to None, meaning
"use the backend's own default at execution time". This makes it explicit when
a researcher has set a value versus when the backend's built-in default applies.

full parameter completeness audit is Phase 4.1 — this file covers only the
fields the M1 PyTorch backend will actively use.

Usage in YAML:
    backend: pytorch
    pytorch:
      batch_size: 4
      load_in_4bit: true

    backend: vllm
    vllm:
      tensor_parallel_size: 2
      gpu_memory_utilization: 0.85

    backend: tensorrt
    tensorrt:
      tp_size: 2
      quantization: fp8
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# PyTorch Backend Configuration (M1 minimal)
# =============================================================================


class PyTorchConfig(BaseModel):
    """PyTorch/Transformers backend configuration.

    All fields default to None — None means "use the backend's own default".
    This distinguishes explicit researcher choices from backend defaults,
    which is important for result reproducibility and experiment attribution.

    M1 scope: fields used by the M1 PyTorch backend implementation.
    Phase 4.1 will audit and expand based on what researchers actually need.
    """

    model_config = {"extra": "forbid"}

    batch_size: int | None = Field(default=None, ge=1, description="Batch size (None -> 1)")
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] | None = Field(
        default=None, description="Attention implementation (None -> sdpa)"
    )
    torch_compile: bool | None = Field(
        default=None, description="Enable torch.compile (None -> False)"
    )
    load_in_4bit: bool | None = Field(default=None, description="BitsAndBytes 4-bit quantization")
    load_in_8bit: bool | None = Field(default=None, description="BitsAndBytes 8-bit quantization")
    num_processes: int | None = Field(
        default=None, ge=1, description="Data parallel processes via Accelerate (None -> 1)"
    )

    @model_validator(mode="after")
    def validate_quantization(self) -> PyTorchConfig:
        """4-bit and 8-bit quantization are mutually exclusive."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "Cannot use both load_in_4bit=True and load_in_8bit=True simultaneously"
            )
        return self


# =============================================================================
# vLLM Backend Configuration (M1 minimal)
# =============================================================================


class VLLMConfig(BaseModel):
    """vLLM backend configuration.

    All fields default to None — None means "use vLLM's own default".
    vLLM uses continuous batching (max_num_seqs) rather than static batch_size.

    M1 scope: fields used by the M1 vLLM backend implementation.
    Phase 4.1 will audit and expand.
    """

    model_config = {"extra": "forbid"}

    max_num_seqs: int | None = Field(
        default=None, ge=1, description="Max concurrent sequences per iteration (None -> 256)"
    )
    tensor_parallel_size: int | None = Field(
        default=None, ge=1, description="Tensor parallel degree (None -> 1)"
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        ge=0.1,
        le=1.0,
        description="GPU memory fraction for KV cache (None -> 0.9)",
    )
    enable_prefix_caching: bool | None = Field(
        default=None, description="Automatic prefix caching for repeated prompts (None -> False)"
    )
    quantization: Literal["awq", "gptq", "fp8"] | None = Field(
        default=None,
        description="Quantization method. Requires pre-quantized model.",
    )


# =============================================================================
# TensorRT-LLM Backend Configuration (M1 minimal)
# =============================================================================


class TensorRTConfig(BaseModel):
    """TensorRT-LLM backend configuration.

    All fields default to None — None means "use TRT-LLM's own default".
    TensorRT requires engine compilation; max_batch_size is a compile-time constant.

    M1 scope: fields used by the M1 TensorRT backend implementation.
    Phase 4.1 will audit and expand.
    """

    model_config = {"extra": "forbid"}

    max_batch_size: int | None = Field(
        default=None, ge=1, description="Max batch size (compile-time constant, None -> 8)"
    )
    tp_size: int | None = Field(default=None, ge=1, description="Tensor parallel size (None -> 1)")
    quantization: Literal["int8_sq", "int4_awq", "fp8"] | None = Field(
        default=None,
        description="Quantization method",
    )
    engine_path: str | None = Field(
        default=None, description="Pre-compiled engine path (skip compilation)"
    )
