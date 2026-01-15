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
    ngram_min: int = Field(
        default=1,
        ge=1,
        description="Minimum n-gram window for prompt lookup",
    )
    ngram_max: int | None = Field(
        default=None,
        description="Maximum n-gram window for prompt lookup",
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
    # Advanced Sampling (extends decoder config)
    # -------------------------------------------------------------------------
    best_of: int | None = Field(
        default=None,
        ge=1,
        description="Generate N sequences, return best (requires swap_space)",
    )
    use_beam_search: bool = Field(
        default=False,
        description="Enable beam search instead of sampling",
    )
    length_penalty: float = Field(
        default=1.0,
        description="Length penalty for beam search",
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
    # -------------------------------------------------------------------------
    num_beams: int = Field(
        default=1,
        ge=1,
        description="Beam search width (1=greedy/sampling)",
    )
    early_stopping: bool = Field(
        default=False,
        description="Stop beam search when N best sequences complete",
    )
    length_penalty: float = Field(
        default=1.0,
        description="Exponential length penalty for beam search",
    )
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
