"""Backend-specific configuration models.

This module defines Pydantic models for backend-specific parameters. Following
the backend-native architecture, each backend section contains ALL parameters
specific to that backend, using native parameter names from the official APIs.

Architecture:
    - Tier 1 (Universal): Defined in models.py (config_name, model_name, decoder, etc.)
    - Tier 2 (Backend-Native): Defined here (batching, quantization, parallelism, decoder extensions)

Note: top_k is now a universal parameter in DecoderConfig (models.py) since all
backends support it with identical semantics. The only difference is the "disabled"
convention: PyTorch/TensorRT use 0, vLLM uses -1. Backends handle this conversion
internally.

Usage in YAML:
    decoder:
      top_k: 40  # Universal - all backends
      temperature: 0.7

    backend: vllm
    vllm:
      max_num_seqs: 256
      tensor_parallel_size: 2
      quantization: awq
      min_p: 0.05  # vLLM-specific

    backend: pytorch
    pytorch:
      batch_size: 4
      batching_strategy: sorted_dynamic
      load_in_4bit: true
      min_p: 0.05  # PyTorch-specific
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# =============================================================================
# PyTorch Backend Configuration
# =============================================================================


class PyTorchBeamSearchConfig(BaseModel):
    """PyTorch beam search configuration.

    Beam search explores multiple hypotheses in parallel, keeping the top-k
    at each step. Generally produces higher quality but slower generation.
    """

    enabled: bool = Field(
        default=False,
        description="Enable beam search (disables sampling)",
    )
    num_beams: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of beams (1 = greedy, >1 = beam search)",
    )
    length_penalty: float = Field(
        default=1.0,
        description="Length penalty: >1.0 favours longer, <1.0 favours shorter",
    )
    early_stopping: bool = Field(
        default=False,
        description="Stop when num_beams complete sentences found",
    )
    no_repeat_ngram_size: int = Field(
        default=0,
        ge=0,
        description="Prevent n-gram repetition (0 = disabled)",
    )


class PyTorchAssistedGenerationConfig(BaseModel):
    """PyTorch assisted generation (speculative decoding) configuration.

    Uses a small assistant model to propose tokens that the main model
    verifies in batch. PyTorch's equivalent to vLLM speculative decoding.

    Native HuggingFace params: assistant_model, num_assistant_tokens
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

    Contains ALL PyTorch-specific parameters using native HuggingFace naming.
    Parameters are organised by functional category.

    Native API sources:
        - transformers.GenerationConfig
        - AutoModelForCausalLM.from_pretrained()
        - model.generate()
        - BitsAndBytesConfig
        - torch.compile()
    """

    # =========================================================================
    # Batching (Application-Level)
    # =========================================================================
    # Note: HuggingFace model.generate() does NOT have native batching -
    # batching is handled by our application layer.
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for static/sorted_static strategies",
    )
    batching_strategy: Literal["static", "dynamic", "sorted_static", "sorted_dynamic"] = Field(
        default="static",
        description="Batching strategy: "
        "static (fixed batch), dynamic (token budget), "
        "sorted_* (length-sorted for efficiency)",
    )
    max_tokens_per_batch: int | None = Field(
        default=None,
        ge=1,
        description="Token budget per batch (only for dynamic/sorted_dynamic)",
    )

    # =========================================================================
    # Parallelism
    # =========================================================================
    parallelism_strategy: Literal["none", "tensor_parallel", "data_parallel"] = Field(
        default="none",
        description="Parallelism strategy: "
        "none (single GPU), tensor_parallel (split layers), "
        "data_parallel (replicate model)",
    )
    parallelism_degree: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor/data parallelism",
    )
    # Note: pipeline_parallel NOT supported for PyTorch generate()

    # =========================================================================
    # Quantization (BitsAndBytes)
    # =========================================================================
    # Native BitsAndBytesConfig parameters
    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit quantization (QLoRA-style)",
    )
    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit quantization",
    )
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16"] = Field(
        default="float16",
        description="Compute dtype for 4-bit quantization",
    )
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = Field(
        default="nf4",
        description="4-bit quantization type (nf4 = NormalFloat4, recommended)",
    )
    bnb_4bit_use_double_quant: bool = Field(
        default=False,
        description="Use nested quantization for memory efficiency",
    )

    # =========================================================================
    # Attention Configuration
    # =========================================================================
    # Native from_pretrained() parameter
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = Field(
        default="sdpa",
        description="Attention implementation: "
        "sdpa (PyTorch native), flash_attention_2 (fastest), eager (compatible)",
    )

    # =========================================================================
    # Compilation
    # =========================================================================
    # Native torch.compile() parameters
    torch_compile: bool | Literal["default", "reduce-overhead", "max-autotune"] = Field(
        default=False,
        description="Enable torch.compile: False, True/'default', "
        "'reduce-overhead' (best for small batches), 'max-autotune' (slowest compile)",
    )
    torch_compile_backend: Literal["inductor", "cudagraphs", "onnxrt", "aot_eager"] | None = Field(
        default=None,
        description="torch.compile backend: 'inductor' (default, Triton-based), "
        "'cudagraphs' (captures ops as CUDA graphs), 'onnxrt' (ONNX Runtime), "
        "'aot_eager' (AOT compilation). Only applies when torch_compile is enabled.",
    )

    # =========================================================================
    # KV Caching
    # =========================================================================
    # Native GenerationConfig parameters
    use_cache: bool = Field(
        default=True,
        description="Enable KV caching during generation (faster but uses memory)",
    )
    cache_implementation: Literal["dynamic", "static", "hybrid", "sliding_window"] | None = Field(
        default=None,
        description="KV cache implementation: 'static' enables CUDA graphs (lower energy), "
        "'dynamic' (default), 'hybrid' (balance), 'sliding_window' (long context)",
    )

    # =========================================================================
    # Memory Management
    # =========================================================================
    # Native from_pretrained() parameters
    low_cpu_mem_usage: bool = Field(
        default=True,
        description="Memory-efficient model loading (load directly to GPU)",
    )
    max_memory: dict[str, str] | None = Field(
        default=None,
        description='Per-device memory limits, e.g., {"0": "10GiB", "cpu": "30GiB"}',
    )

    # =========================================================================
    # Decoder Extensions (PyTorch-specific sampling params)
    # =========================================================================
    # These extend the universal decoder config in models.py
    # Note: top_k is now in universal DecoderConfig (all backends support it)
    # Native GenerationConfig parameters
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for sampling",
    )
    no_repeat_ngram_size: int = Field(
        default=0,
        ge=0,
        description="Prevent n-gram repetition (0 = disabled)",
    )
    output_scores: bool = Field(
        default=False,
        description="Return generation scores/logprobs",
    )
    return_dict_in_generate: bool = Field(
        default=False,
        description="Return GenerateOutput dict instead of tensor",
    )

    # =========================================================================
    # Beam Search
    # =========================================================================
    beam_search: PyTorchBeamSearchConfig = Field(
        default_factory=PyTorchBeamSearchConfig,
        description="Beam search configuration",
    )

    # =========================================================================
    # Assisted Generation (Speculative Decoding)
    # =========================================================================
    assisted_generation: PyTorchAssistedGenerationConfig | None = Field(
        default=None,
        description="Assisted generation (speculative decoding) configuration",
    )

    # =========================================================================
    # Legacy Optimizations
    # =========================================================================
    use_bettertransformer: bool = Field(
        default=False,
        description="Convert to BetterTransformer (pre-PyTorch 2.0 optimization)",
    )

    # =========================================================================
    # Escape Hatch
    # =========================================================================
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to model.generate()",
    )

    # =========================================================================
    # Validators
    # =========================================================================
    @model_validator(mode="after")
    def validate_quantization(self) -> PyTorchConfig:
        """Validate quantization settings are mutually exclusive."""
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot enable both 4-bit and 8-bit quantization")
        return self


# =============================================================================
# vLLM Backend Configuration
# =============================================================================


class VLLMAttentionConfig(BaseModel):
    """vLLM attention backend configuration.

    Controls which attention kernel implementation vLLM uses.
    Set via VLLM_ATTENTION_BACKEND environment variable internally.
    """

    backend: Literal["auto", "FLASH_ATTN", "FLASHINFER"] = Field(
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

    Native vLLM params: speculative_model, num_speculative_tokens,
    ngram_prompt_lookup_min, ngram_prompt_lookup_max,
    speculative_draft_tensor_parallel_size
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
    draft_tp_size: int = Field(
        default=1,
        ge=1,
        description="Tensor parallel size for draft model",
    )


class VLLMLoRAConfig(BaseModel):
    """vLLM LoRA adapter configuration.

    Enables serving multiple LoRA adapters concurrently.

    Native vLLM params: enable_lora, max_loras, max_lora_rank, lora_extra_vocab_size
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

    Contains ALL vLLM-specific parameters using native vLLM naming.
    Parameters are organised by functional category.

    Native API sources:
        - vLLM LLM() constructor
        - vLLM SamplingParams
        - VLLM_ATTENTION_BACKEND environment variable

    Key difference from PyTorch:
        - vLLM uses continuous batching (max_num_seqs) instead of static batch_size
        - No external batching strategy - vLLM handles batching internally
    """

    # =========================================================================
    # Memory & Concurrency (vLLM's batching equivalent)
    # =========================================================================
    # Note: vLLM uses continuous batching - max_num_seqs is the concurrency limit
    max_num_seqs: int = Field(
        default=256,
        ge=1,
        le=1024,
        description="Maximum concurrent sequences per iteration (continuous batching)",
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

    # =========================================================================
    # Parallelism
    # =========================================================================
    # Native vLLM LLM() parameters
    tensor_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for tensor parallelism (split layers horizontally)",
    )
    pipeline_parallel_size: int = Field(
        default=1,
        ge=1,
        description="Number of GPUs for pipeline parallelism (split model vertically)",
    )
    distributed_backend: Literal["mp", "ray"] = Field(
        default="mp",
        description="Distributed executor backend (mp=multiprocessing, ray=Ray cluster)",
    )
    disable_custom_all_reduce: bool = Field(
        default=False,
        description="Disable custom NCCL AllReduce kernel",
    )

    # =========================================================================
    # KV Cache Configuration
    # =========================================================================
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

    # =========================================================================
    # Context & Sequence Length
    # =========================================================================
    max_model_len: int | None = Field(
        default=None,
        description="Maximum context length (None=use model's native max)",
    )
    max_seq_len_to_capture: int | None = Field(
        default=None,
        description="Maximum sequence length for CUDA graph capture",
    )

    # =========================================================================
    # Execution Mode
    # =========================================================================
    enforce_eager: bool = Field(
        default=False,
        description="Disable CUDA graphs (for debugging or compatibility)",
    )

    # =========================================================================
    # Attention
    # =========================================================================
    attention: VLLMAttentionConfig | None = Field(
        default=None,
        description="Attention backend configuration",
    )

    # =========================================================================
    # Quantization
    # =========================================================================
    # Native vLLM LLM() parameter
    quantization: Literal["awq", "gptq", "fp8", "marlin", "bitsandbytes", "squeezellm"] | None = (
        Field(
            default=None,
            description="Quantization method for pre-quantized models",
        )
    )
    load_format: Literal["auto", "pt", "safetensors", "gguf"] = Field(
        default="auto",
        description="Weight loading format",
    )

    # =========================================================================
    # Decoder Extensions (vLLM SamplingParams)
    # =========================================================================
    # These extend the universal decoder config in models.py
    # Note: top_k is now in universal DecoderConfig (all backends support it)
    # vLLM uses -1 for disabled, we convert from decoder.top_k=0 internally
    min_p: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for sampling",
    )
    # Note: no_repeat_ngram_size NOT supported by vLLM

    # =========================================================================
    # Advanced Sampling (vLLM-specific)
    # =========================================================================
    # Note: best_of was removed in vLLM v1 (use beam search or repetition instead)
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

    # =========================================================================
    # Speculative Decoding
    # =========================================================================
    speculative: VLLMSpeculativeConfig | None = Field(
        default=None,
        description="Speculative decoding configuration",
    )

    # =========================================================================
    # LoRA
    # =========================================================================
    lora: VLLMLoRAConfig | None = Field(
        default=None,
        description="LoRA adapter configuration",
    )

    # =========================================================================
    # Escape Hatch
    # =========================================================================
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed directly to vLLM LLM()",
    )


# =============================================================================
# TensorRT-LLM Backend Configuration
# =============================================================================


class TensorRTCalibrationConfig(BaseModel):
    """TensorRT INT8/INT4 calibration configuration.

    Required for post-training quantization (PTQ) with INT8 SmoothQuant.
    Calibration data is used to determine optimal scaling factors.

    Native trtllm-build params: calib_dataset, calib_size, calib_max_seq_len
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


class TensorRTConfig(BaseModel):
    """TensorRT-LLM backend configuration.

    Contains ALL TensorRT-specific parameters using native TRT-LLM naming.
    Parameters are organised by functional category.

    Native API sources:
        - trtllm-build command
        - TensorRT-LLM LLM() runtime
        - TensorRT-LLM Executor

    Key concepts:
        - Engine: Compiled inference plan optimised for specific GPU + config
        - Build config: Compile-time settings (max batch/sequence lengths, etc.)
        - Runtime config: Execution settings (KV cache, batching behaviour)
    """

    # =========================================================================
    # Engine Source
    # =========================================================================
    engine_path: str | None = Field(
        default=None,
        description="Path to pre-compiled TRT engine directory. "
        "If not set, engine will be built from HuggingFace checkpoint.",
    )
    force_rebuild: bool = Field(
        default=False,
        description="Force engine rebuild even if cached version exists",
    )
    engine_cache_dir: str | None = Field(
        default=None,
        description="Directory for caching compiled engines. "
        "Defaults to ~/.cache/lem/tensorrt-engines/",
    )

    # =========================================================================
    # Build Configuration (compile-time)
    # =========================================================================
    # Native trtllm-build parameters
    max_batch_size: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Maximum batch size for compiled engine (compile-time)",
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
        description="Enable strong typing for FP8 precision (required for Hopper+)",
    )
    multiple_profiles: bool = Field(
        default=False,
        description="Build with multiple TensorRT profiles for different input shapes. "
        "Enables better kernel selection per input shape.",
    )

    # =========================================================================
    # Parallelism
    # =========================================================================
    # Native trtllm-build parameters
    tp_size: int = Field(
        default=1,
        ge=1,
        description="Tensor parallel size (split layers across GPUs)",
    )
    pp_size: int = Field(
        default=1,
        ge=1,
        description="Pipeline parallel size (for very large models)",
    )

    # =========================================================================
    # Quantization
    # =========================================================================
    # Flattened from nested config - maps to trtllm-build flags
    quantization: Literal["none", "fp8", "int8_sq", "int8_weight_only", "int4_awq", "int4_gptq"] = (
        Field(
            default="none",
            description="Quantization method: "
            "none (FP16/BF16), fp8 (Hopper+ fast), int8_sq (SmoothQuant, needs calibration), "
            "int8_weight_only, int4_awq (pre-quantized), int4_gptq (pre-quantized)",
        )
    )
    calibration: TensorRTCalibrationConfig | None = Field(
        default=None,
        description="Calibration config (required for int8_sq)",
    )

    # =========================================================================
    # Runtime Options
    # =========================================================================
    # Native TRT-LLM executor parameters
    kv_cache_type: Literal["paged", "continuous"] = Field(
        default="paged",
        description="KV cache management: paged (memory efficient) or continuous",
    )
    enable_chunked_context: bool = Field(
        default=True,
        description="Enable chunked context for long sequences",
    )
    enable_kv_cache_reuse: bool = Field(
        default=False,
        description="Enable KV cache reuse for prefix caching. "
        "Avoids recomputing attention for shared prefixes.",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        ge=0.5,
        le=0.99,
        description="Fraction of GPU memory for KV cache (kv_cache_free_gpu_memory_fraction)",
    )
    max_num_tokens: int | None = Field(
        default=None,
        description="Maximum tokens per iteration (for inflight batching)",
    )

    # =========================================================================
    # Decoder Extensions (TensorRT-specific)
    # =========================================================================
    # Note: TensorRT-LLM has limited sampling support
    # - top_k: now in universal DecoderConfig (all backends support it)
    # - min_p: NOT supported by TensorRT-LLM
    # - no_repeat_ngram_size: NOT supported by TensorRT-LLM

    # =========================================================================
    # Speculative Decoding
    # =========================================================================
    # Native TRT-LLM parameters
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

    # =========================================================================
    # Escape Hatches
    # =========================================================================
    extra_build_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to trtllm-build",
    )
    extra_runtime_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs passed to TRT-LLM runtime/executor",
    )
