"""Backend-specific configuration models (v2.0 schema).

Each backend section uses None-as-default: all fields default to None, meaning
"use the backend's own default at execution time". This makes it explicit when
a researcher has set a value versus when the backend's built-in default applies.

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
# PyTorch Backend Configuration (v2.0)
# =============================================================================


class PyTorchConfig(BaseModel):
    """PyTorch/Transformers backend configuration.

    All fields default to None — None means "use the backend's own default".
    This distinguishes explicit researcher choices from backend defaults,
    which is important for result reproducibility and experiment attribution.

    Fields cover the complete researcher-useful parameter space for
    AutoModelForCausalLM.from_pretrained() and model.generate().
    """

    model_config = {"extra": "forbid"}

    # -------------------------------------------------------------------------
    # Batching
    # -------------------------------------------------------------------------

    batch_size: int | None = Field(default=None, ge=1, description="Batch size (None -> 1)")

    # -------------------------------------------------------------------------
    # Attention implementation
    # -------------------------------------------------------------------------

    attn_implementation: (
        Literal["sdpa", "flash_attention_2", "flash_attention_3", "eager"] | None
    ) = Field(default=None, description="Attention implementation (None -> sdpa)")

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------

    torch_compile: bool | None = Field(
        default=None, description="Enable torch.compile (None -> False)"
    )
    torch_compile_mode: str | None = Field(
        default=None,
        description="torch.compile mode: 'default', 'reduce-overhead', 'max-autotune' (None -> 'default')",
    )
    torch_compile_backend: str | None = Field(
        default=None, description="torch.compile backend (None -> 'inductor')"
    )

    # -------------------------------------------------------------------------
    # BitsAndBytes quantization
    # -------------------------------------------------------------------------

    load_in_4bit: bool | None = Field(default=None, description="BitsAndBytes 4-bit quantization")
    load_in_8bit: bool | None = Field(default=None, description="BitsAndBytes 8-bit quantization")
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] | None = Field(
        default=None,
        description="Compute dtype for 4-bit (None -> float32, usually want bfloat16)",
    )
    bnb_4bit_quant_type: Literal["nf4", "fp4"] | None = Field(
        default=None, description="4-bit quantization type (None -> 'nf4')"
    )
    bnb_4bit_use_double_quant: bool | None = Field(
        default=None, description="Double quantization saves ~0.4 bits/param (None -> False)"
    )

    # -------------------------------------------------------------------------
    # KV caching
    # -------------------------------------------------------------------------

    use_cache: bool | None = Field(
        default=None, description="Use KV cache during generation (None -> True)"
    )
    cache_implementation: Literal["static", "offloaded_static", "sliding_window"] | None = Field(
        default=None,
        description="KV cache strategy; 'static' enables CUDA graphs (None -> dynamic)",
    )

    # -------------------------------------------------------------------------
    # Beam search
    # -------------------------------------------------------------------------

    num_beams: int | None = Field(
        default=None, ge=1, description="Beam search width (None -> 1, greedy/sampling)"
    )
    early_stopping: bool | None = Field(
        default=None, description="Stop beam search when all beams hit EOS (None -> False)"
    )
    length_penalty: float | None = Field(
        default=None,
        description="Beam length penalty: >1 shorter, <1 longer (None -> 1.0)",
    )

    # -------------------------------------------------------------------------
    # N-gram repetition
    # -------------------------------------------------------------------------

    no_repeat_ngram_size: int | None = Field(
        default=None, ge=0, description="Prevent n-gram repetition (None -> 0, disabled)"
    )

    # -------------------------------------------------------------------------
    # Speculative decoding (prompt-lookup — draft model via passthrough_kwargs)
    # -------------------------------------------------------------------------

    prompt_lookup_num_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Prompt-lookup speculative decoding tokens (None -> disabled)",
    )

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    device_map: str | None = Field(
        default=None, description="Device placement strategy (None -> 'auto')"
    )
    max_memory: dict | None = Field(
        default=None,
        description="Per-device memory limits, e.g. {0: '10GiB', 'cpu': '50GiB'}",
    )
    revision: str | None = Field(
        default=None, description="Model revision/commit hash for reproducibility"
    )
    trust_remote_code: bool | None = Field(
        default=None, description="Trust remote code in model repo (None -> True)"
    )

    # -------------------------------------------------------------------------
    # Data parallelism
    # -------------------------------------------------------------------------

    num_processes: int | None = Field(
        default=None, ge=1, description="Data parallel processes via Accelerate (None -> 1)"
    )

    # -------------------------------------------------------------------------
    # Cross-validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_quantization(self) -> PyTorchConfig:
        """4-bit and 8-bit quantization are mutually exclusive."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError(
                "Cannot use both load_in_4bit=True and load_in_8bit=True simultaneously"
            )
        return self

    @model_validator(mode="after")
    def validate_torch_compile_options(self) -> PyTorchConfig:
        """torch_compile_mode/torch_compile_backend require torch_compile=True."""
        if (
            self.torch_compile_mode is not None or self.torch_compile_backend is not None
        ) and self.torch_compile is not True:
            raise ValueError("torch_compile_mode/torch_compile_backend requires torch_compile=True")
        return self

    @model_validator(mode="after")
    def validate_bnb_4bit_options(self) -> PyTorchConfig:
        """bnb_4bit_* fields require load_in_4bit=True."""
        if (
            self.bnb_4bit_compute_dtype is not None
            or self.bnb_4bit_quant_type is not None
            or self.bnb_4bit_use_double_quant is not None
        ) and self.load_in_4bit is not True:
            raise ValueError("bnb_4bit_* fields require load_in_4bit=True")
        return self

    @model_validator(mode="after")
    def validate_cache_options(self) -> PyTorchConfig:
        """cache_implementation requires use_cache to be True or None (not explicitly False)."""
        if self.cache_implementation is not None and self.use_cache is False:
            raise ValueError(
                "cache_implementation requires use_cache to be True or None (not explicitly False)"
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
