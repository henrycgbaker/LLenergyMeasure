"""Backend-specific configuration models.

This module defines Pydantic models for backend-specific parameters that
cannot be abstracted into the shared ExperimentConfig. Each backend has
its own config section with full type safety and validation.

Usage in YAML:
    backend: vllm
    vllm:
      max_num_seqs: 256
      enable_prefix_caching: true

    backend: pytorch
    pytorch:
      attn_implementation: flash_attention_2
      torch_compile: reduce-overhead
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# vLLM Backend Configuration
# =============================================================================


class VLLMAttentionConfig(BaseModel):
    """vLLM attention backend configuration.

    Controls which attention kernel implementation vLLM uses.
    """

    backend: Literal["auto", "FLASH_ATTN", "FLASHINFER", "TORCH_SDPA"] = Field(
        default="auto",
        description="Attention backend (auto=let vLLM decide)",
    )
    flash_version: Literal[2, 3] | None = Field(
        default=None,
        description="Flash Attention version (3 for H100/Hopper GPUs)",
    )
    disable_sliding_window: bool = Field(
        default=False,
        description="Disable sliding window attention (for models like Mistral)",
    )


class VLLMSpeculativeConfig(BaseModel):
    """vLLM speculative decoding configuration.

    Speculative decoding uses a small draft model to propose multiple tokens,
    then the main model verifies them in a single forward pass. Can provide
    2-3x latency improvement for compatible model pairs.

    Energy Impact:
        Better n-gram tuning improves speculation hit rate → fewer wasted
        draft computations → lower energy per token.
    """

    model: str | None = Field(
        default=None,
        description="Draft model name/path for speculative decoding",
    )
    num_tokens: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of tokens to speculate per step",
    )
    method: Literal["ngram", "eagle", "eagle3", "medusa", "mlp", "lookahead"] = Field(
        default="ngram",
        description="Speculation method (ngram=prompt lookup, eagle/medusa=learned)",
    )
    # Ngram lookup bounds (Phase 3.1 - energy impacting)
    prompt_lookup_min: int = Field(
        default=1,
        ge=1,
        description="Minimum n-gram window for prompt lookup speculation",
    )
    prompt_lookup_max: int | None = Field(
        default=None,
        ge=1,
        description="Maximum n-gram window for prompt lookup speculation",
    )
    # Legacy aliases
    ngram_min: int = Field(
        default=1,
        ge=1,
        description="[Deprecated] Use prompt_lookup_min instead",
    )
    ngram_max: int | None = Field(
        default=None,
        description="[Deprecated] Use prompt_lookup_max instead",
    )
    draft_tp_size: int = Field(
        default=1,
        ge=1,
        description="Tensor parallel size for draft model",
    )


class VLLMLoRAConfig(BaseModel):
    """vLLM LoRA adapter configuration.

    Enables serving multiple LoRA adapters concurrently.
    """

    enabled: bool = Field(
        default=False,
        description="Enable LoRA adapter support",
    )
    max_loras: int = Field(
        default=1,
        ge=1,
        description="Maximum number of concurrent LoRA adapters",
    )
    max_rank: int = Field(
        default=16,
        ge=1,
        description="Maximum LoRA rank supported",
    )
    extra_vocab_size: int = Field(
        default=256,
        ge=0,
        description="Extra vocabulary size for LoRA adapters",
    )


class VLLMConfig(BaseModel):
    """vLLM backend configuration.

    Exposes vLLM-specific optimization parameters for the LLM() constructor
    and SamplingParams. These parameters control memory allocation, KV cache
    behaviour, parallelism, and advanced features like speculative decoding.

    Note: Parameters that overlap with ExperimentConfig (e.g., temperature,
    tensor_parallel_size) should be set in the shared config, not here.
    """

    # -------------------------------------------------------------------------
    # Memory & Batching
    # -------------------------------------------------------------------------
    max_num_seqs: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum concurrent sequences per iteration",
    )
    max_num_batched_tokens: int | None = Field(
        default=None,
        description="Maximum tokens per iteration (None=auto). "
        "Lower values improve latency, higher values improve throughput.",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.5,
        le=0.99,
        description="Fraction of GPU memory for KV cache (0.5-0.99)",
    )
    swap_space: float = Field(
        default=4.0,
        ge=0,
        description="CPU swap space per GPU in GiB (for best_of sampling)",
    )
    cpu_offload_gb: float = Field(
        default=0.0,
        ge=0,
        description="CPU memory for model weight offloading in GiB",
    )

    # -------------------------------------------------------------------------
    # KV Cache Configuration
    # -------------------------------------------------------------------------
    enable_prefix_caching: bool = Field(
        default=False,
        description="Enable automatic prefix caching for repeated prompts. "
        "Can improve throughput 30-50% for similar prompts.",
    )
    enable_chunked_prefill: bool = Field(
        default=False,
        description="Chunk large prefills and batch with decode requests. "
        "Improves latency for mixed workloads.",
    )
    kv_cache_dtype: Literal["auto", "float16", "bfloat16", "fp8"] = Field(
        default="auto",
        description="KV cache precision (fp8 saves ~50% memory)",
    )
    block_size: Literal[8, 16, 32] = Field(
        default=16,
        description="KV cache block size in tokens (PagedAttention)",
    )

    # -------------------------------------------------------------------------
    # Context & Sequence Length
    # -------------------------------------------------------------------------
    max_model_len: int | None = Field(
        default=None,
        description="Maximum context length (None=use model's native max)",
    )
    max_seq_len_to_capture: int | None = Field(
        default=None,
        description="Maximum sequence length for CUDA graph capture",
    )

    # -------------------------------------------------------------------------
    # Execution Mode
    # -------------------------------------------------------------------------
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graphs (for debugging or compatibility)",
    )

    # -------------------------------------------------------------------------
    # Parallelism (supplements shared sharding config)
    # -------------------------------------------------------------------------
    distributed_backend: Literal["mp", "ray"] = Field(
        default="mp",
        description="Distributed executor backend (mp=multiprocessing, ray=Ray cluster)",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom NCCL AllReduce kernel",
    )

    # -------------------------------------------------------------------------
    # Nested Configurations
    # -------------------------------------------------------------------------
    attention: VLLMAttentionConfig | None = Field(
        default=None,
        description="Attention backend configuration",
    )
    speculative: VLLMSpeculativeConfig | None = Field(
        default=None,
        description="Speculative decoding configuration",
    )
    lora: VLLMLoRAConfig | None = Field(
        default=None,
        description="LoRA adapter configuration",
    )

    # -------------------------------------------------------------------------
    # Quantization (supplements shared quant config)
    # -------------------------------------------------------------------------
    quantization_method: str | None = Field(
        default=None,
        description="Explicit quantization method (gptq, awq, fp8, marlin, etc.)",
    )
    load_format: Literal["auto", "pt", "safetensors", "gguf"] = Field(
        default="auto",
        description="Weight loading format",
    )

    # -------------------------------------------------------------------------
    # Advanced Sampling (vLLM-specific extensions to decoder config)
    # Note: Beam search params moved to DecoderConfig.beam_search
    # -------------------------------------------------------------------------
    best_of: int | None = Field(
        default=None,
        ge=1,
        description="Generate N sequences, return best (requires swap_space)",
    )
    logprobs: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Return top-k log probabilities per token",
    )
    logit_bias: dict[int, float] | None = Field(
        default=None,
        description="Per-token logit adjustments {token_id: bias}",
    )

    # -------------------------------------------------------------------------
    # Escape Hatch
    # -------------------------------------------------------------------------
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed directly to vLLM LLM()",
    )


# =============================================================================
# PyTorch Backend Configuration
# =============================================================================


class PyTorchAssistedGenerationConfig(BaseModel):
    """PyTorch assisted generation (speculative decoding) configuration.

    Uses a small assistant model to propose tokens that the main model
    verifies in batch. PyTorch's equivalent to vLLM speculative decoding.
    """

    model: str | None = Field(
        default=None,
        description="Assistant/draft model name for speculative decoding",
    )
    num_tokens: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of tokens to speculate per step",
    )


class PyTorchConfig(BaseModel):
    """PyTorch/Transformers backend configuration.

    Exposes PyTorch-specific optimization parameters for model loading
    and generation. These parameters control attention implementation,
    compilation, memory management, and advanced generation features.

    Note: Parameters that overlap with ExperimentConfig (e.g., temperature,
    tensor_parallel_size) should be set in the shared config, not here.
    """

    # -------------------------------------------------------------------------
    # Attention Configuration
    # -------------------------------------------------------------------------
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa",
        description="Attention implementation: "
        "sdpa (PyTorch native), flash_attention_2 (fastest), eager (compatible)",
    )

    # -------------------------------------------------------------------------
    # Compilation
    # -------------------------------------------------------------------------
    torch_compile: bool | Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default=False,
        description="Enable torch.compile: False, True/'default', "
        "'reduce-overhead' (best for small batches), 'max-autotune' (slowest compile)",
    )

    # -------------------------------------------------------------------------
    # Legacy Optimizations
    # -------------------------------------------------------------------------
    use_bettertransformer: bool = Field(
        default=False,
        description="Convert to BetterTransformer (pre-PyTorch 2.0 optimization)",
    )

    # -------------------------------------------------------------------------
    # KV Caching
    # -------------------------------------------------------------------------
    use_cache: bool = Field(
        default=True,
        description="Enable KV caching during generation (faster but uses memory)",
    )
    # Phase 3.3: Cache implementation (energy impacting)
    cache_implementation: Literal["dynamic", "static", "hybrid", "sliding_window"] | None = Field(
        default=None,
        description="KV cache implementation: 'static' enables CUDA graphs (lower energy), "
        "'dynamic' (default), 'hybrid' (balance), 'sliding_window' (long context)",
    )

    # -------------------------------------------------------------------------
    # Memory Management
    # -------------------------------------------------------------------------
    low_cpu_mem_usage: bool = Field(
        default=True,
        description="Memory-efficient model loading (load directly to GPU)",
    )
    max_memory: dict[str, str] | None = Field(
        default=None,
        description='Per-device memory limits, e.g., {"0": "10GiB", "cpu": "30GiB"}',
    )

    # -------------------------------------------------------------------------
    # Assisted Generation (Speculative Decoding)
    # -------------------------------------------------------------------------
    assisted_generation: PyTorchAssistedGenerationConfig | None = Field(
        default=None,
        description="Assisted generation (speculative decoding) configuration",
    )

    # -------------------------------------------------------------------------
    # Generation Configuration
    # Note: Beam search params moved to DecoderConfig.beam_search
    # -------------------------------------------------------------------------
    output_scores: bool = Field(
        default=False,
        description="Return generation scores/logprobs",
    )
    return_dict_in_generate: bool = Field(
        default=False,
        description="Return GenerateOutput dict instead of tensor",
    )

    # -------------------------------------------------------------------------
    # Escape Hatch
    # -------------------------------------------------------------------------
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to model.generate()",
    )


# =============================================================================
# TensorRT-LLM Backend Configuration
# =============================================================================


class TensorRTCalibrationConfig(BaseModel):
    """TensorRT INT8/INT4 calibration configuration.

    Required for post-training quantization (PTQ) with INT8 or INT4.
    Calibration data is used to determine optimal scaling factors.
    """

    dataset: str = Field(
        default="wikitext",
        description="HuggingFace dataset name or local path for calibration data",
    )
    split: str = Field(
        default="train",
        description="Dataset split to use for calibration",
    )
    num_samples: int = Field(
        default=512,
        ge=64,
        le=4096,
        description="Number of samples for calibration (512-1024 typical)",
    )
    max_length: int = Field(
        default=2048,
        ge=128,
        description="Maximum sequence length for calibration samples",
    )


class TensorRTQuantizationConfig(BaseModel):
    """TensorRT-LLM quantization configuration.

    Controls quantization method for TensorRT engine compilation.
    Some methods require calibration data for optimal performance.

    Methods:
        none: No quantization (FP16/BF16)
        fp8: FP8 quantization (Hopper+ GPUs, fast, minimal accuracy loss)
        int8_sq: INT8 SmoothQuant (requires calibration)
        int8_weight_only: INT8 weights, FP16 compute
        int4_awq: INT4 AWQ (pre-quantized checkpoint)
        int4_gptq: INT4 GPTQ (pre-quantized checkpoint)
    """

    method: Literal["none", "fp8", "int8_sq", "int8_weight_only", "int4_awq", "int4_gptq"] = Field(
        default="none",
        description="Quantization method for TRT engine",
    )
    calibration: TensorRTCalibrationConfig | None = Field(
        default=None,
        description="Calibration config (required for int8_sq)",
    )


class TensorRTConfig(BaseModel):
    """TensorRT-LLM backend configuration.

    TensorRT-LLM provides high-performance inference through compiled
    execution plans. Engines can be pre-compiled or built on-demand.

    Key concepts:
    - Engine: Compiled inference plan optimised for specific GPU + config
    - Build config: Compile-time settings (max batch/sequence lengths, etc.)
    - Runtime config: Execution settings (KV cache, batching behaviour)

    Usage in YAML:
        backend: tensorrt
        tensorrt:
          max_batch_size: 8
          builder_opt_level: 3
          quantization:
            method: fp8
          enable_chunked_context: true
    """

    # -------------------------------------------------------------------------
    # Engine Source
    # -------------------------------------------------------------------------
    engine_path: str | None = Field(
        default=None,
        description="Path to pre-compiled TRT engine directory. "
        "If not set, engine will be built from HuggingFace checkpoint.",
    )

    # -------------------------------------------------------------------------
    # Build Configuration (used when compiling from HF checkpoint)
    # -------------------------------------------------------------------------
    max_batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Maximum batch size for compiled engine",
    )
    max_input_len: int | None = Field(
        default=None,
        description="Maximum input sequence length. Defaults to model's max_position_embeddings.",
    )
    max_output_len: int | None = Field(
        default=None,
        description="Maximum output tokens per request. Defaults to config.max_output_tokens.",
    )
    builder_opt_level: int = Field(
        default=3,
        ge=0,
        le=5,
        description="TensorRT builder optimization level (0-5). "
        "Higher = slower build, faster inference.",
    )
    strongly_typed: bool = Field(
        default=True,
        description="Enable strong typing for FP8 precision (recommended for Hopper+)",
    )

    # -------------------------------------------------------------------------
    # Quantization
    # -------------------------------------------------------------------------
    quantization: TensorRTQuantizationConfig = Field(
        default_factory=TensorRTQuantizationConfig,
        description="Quantization configuration",
    )

    # -------------------------------------------------------------------------
    # Parallelism (uses config.parallelism for degree)
    # Note: tp_size removed - use parallelism.degree with strategy=tensor_parallel
    # -------------------------------------------------------------------------
    pp_size: int = Field(
        default=1,
        ge=1,
        description="Pipeline parallel size (for very large models)",
    )

    # -------------------------------------------------------------------------
    # Build Optimisation (Phase 3.2 - energy impacting)
    # -------------------------------------------------------------------------
    multiple_profiles: bool = Field(
        default=False,
        description="Build with multiple TensorRT profiles for different input shapes. "
        "Enables better kernel selection per input shape (moderate energy impact).",
    )

    # -------------------------------------------------------------------------
    # Runtime Options
    # -------------------------------------------------------------------------
    kv_cache_type: Literal["paged", "continuous"] = Field(
        default="paged",
        description="KV cache management: paged (memory efficient) or continuous",
    )
    enable_chunked_context: bool = Field(
        default=True,
        description="Enable chunked context for long sequences",
    )
    max_num_tokens: int | None = Field(
        default=None,
        description="Maximum tokens per iteration (for inflight batching)",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.5,
        le=0.99,
        description="Fraction of GPU memory for KV cache (0.5-0.99)",
    )
    # Phase 3.2: KV cache reuse (energy impacting - high impact)
    enable_kv_cache_reuse: bool = Field(
        default=False,
        description="Enable KV cache reuse for prefix caching. "
        "Avoids recomputing attention for shared prefixes (high energy impact).",
    )

    # -------------------------------------------------------------------------
    # Cache Control
    # -------------------------------------------------------------------------
    engine_cache_dir: str | None = Field(
        default=None,
        description="Directory for caching compiled engines. "
        "Defaults to ~/.cache/llm-energy-measure/tensorrt-engines/",
    )
    force_rebuild: bool = Field(
        default=False,
        description="Force engine rebuild even if cached version exists",
    )

    # -------------------------------------------------------------------------
    # Speculative Decoding
    # -------------------------------------------------------------------------
    draft_model: str | None = Field(
        default=None,
        description="Draft model for speculative decoding",
    )
    num_draft_tokens: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of tokens to speculate per step",
    )

    # -------------------------------------------------------------------------
    # Escape Hatches
    # -------------------------------------------------------------------------
    extra_build_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to trtllm-build",
    )
    extra_runtime_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to TRT-LLM runtime/executor",
    )
