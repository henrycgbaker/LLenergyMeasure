"""vLLM backend parameter specifications.

Defines ParamSpecs for all vLLM-specific parameters, covering memory,
batching, KV cache, parallelism, LoRA, and sampling configurations.
"""

from __future__ import annotations

from ...registry import (
    HardwareRequirement,
    ParamSpec,
    VerificationType,
    register_all,
)

# =============================================================================
# Memory & Batching Parameters
# =============================================================================

MEMORY_BATCHING_SPECS = [
    ParamSpec(
        name="max_num_seqs",
        backend="vllm",
        config_path="vllm.max_num_seqs",
        test_values=[64, 128, 256],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.scheduler_config.max_num_seqs",
        description="Maximum concurrent sequences per iteration",
        category="memory",
    ),
    ParamSpec(
        name="max_num_batched_tokens",
        backend="vllm",
        config_path="vllm.max_num_batched_tokens",
        test_values=[512, 1024, 2048],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.scheduler_config.max_num_batched_tokens",
        description="Maximum tokens per iteration",
        category="memory",
    ),
    ParamSpec(
        name="gpu_memory_utilization",
        backend="vllm",
        config_path="vllm.gpu_memory_utilization",
        test_values=[0.5, 0.7, 0.9],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.cache_config.gpu_memory_utilization",
        description="Fraction of GPU memory for KV cache",
        category="memory",
        energy_impact=True,
    ),
    ParamSpec(
        name="swap_space",
        backend="vllm",
        config_path="vllm.swap_space",
        test_values=[2.0, 4.0, 8.0],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.cache_config.swap_space_bytes",
        description="CPU swap space per GPU in GiB",
        category="memory",
        # Transform GB to bytes for comparison
        expected_transform=lambda x: int(x * (1024**3)),
    ),
    ParamSpec(
        name="cpu_offload_gb",
        backend="vllm",
        config_path="vllm.cpu_offload_gb",
        test_values=[0.0, 2.0],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.load_config.cpu_offload_gb",
        description="CPU memory for model weight offloading",
        category="memory",
        skip_reason="vLLM cpu_offload path varies by version",
    ),
]

# =============================================================================
# KV Cache Parameters
# =============================================================================

KV_CACHE_SPECS = [
    ParamSpec(
        name="enable_prefix_caching",
        backend="vllm",
        config_path="vllm.enable_prefix_caching",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.cache_config.enable_prefix_caching",
        description="Enable automatic prefix caching",
        category="kv_cache",
        energy_impact=True,
    ),
    ParamSpec(
        name="enable_chunked_prefill",
        backend="vllm",
        config_path="vllm.enable_chunked_prefill",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.scheduler_config.chunked_prefill_enabled",
        description="Chunk large prefills and batch with decode",
        category="kv_cache",
        energy_impact=True,
    ),
    ParamSpec(
        name="kv_cache_dtype",
        backend="vllm",
        config_path="vllm.kv_cache_dtype",
        test_values=[
            "auto",
            "float16",
        ],  # FP8 tested separately in GPU tests with Hopper requirement
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.cache_config.cache_dtype",
        description="KV cache precision",
        category="kv_cache",
        energy_impact=True,
        skip_reason="kv_cache_dtype not supported in vLLM v1 (VLLM_USE_V1=1)",
    ),
    ParamSpec(
        name="block_size",
        backend="vllm",
        config_path="vllm.block_size",
        test_values=[8, 16, 32],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.cache_config.block_size",
        description="KV cache block size in tokens",
        category="kv_cache",
    ),
]

# =============================================================================
# Context & Sequence Length Parameters
# =============================================================================

CONTEXT_SPECS = [
    ParamSpec(
        name="max_model_len",
        backend="vllm",
        config_path="vllm.max_model_len",
        test_values=[128, 256, 512],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.model_config.max_model_len",
        description="Maximum context length",
        category="context",
    ),
    ParamSpec(
        name="max_seq_len_to_capture",
        backend="vllm",
        config_path="vllm.max_seq_len_to_capture",
        test_values=[256, 512],  # Values <= max_model_len (512 in tests)
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.model_config.max_seq_len_to_capture",
        description="Maximum sequence length for CUDA graph capture",
        category="context",
    ),
]

# =============================================================================
# Execution Mode Parameters
# =============================================================================

EXECUTION_SPECS = [
    ParamSpec(
        name="enforce_eager",
        backend="vllm",
        config_path="vllm.enforce_eager",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.model_config.enforce_eager",
        description="Disable CUDA graphs",
        category="execution",
        energy_impact=True,
    ),
]

# =============================================================================
# Parallelism Parameters (vLLM-specific, supplements shared parallelism config)
# Note: Main parallelism settings (strategy, degree) are in shared/param_specs.py
# =============================================================================

PARALLELISM_SPECS = [
    ParamSpec(
        name="distributed_executor_backend",
        backend="vllm",
        config_path="vllm.distributed_executor_backend",
        test_values=["mp"],  # ray requires additional setup
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={
            HardwareRequirement.GPU,
            HardwareRequirement.VLLM,
            HardwareRequirement.MULTI_GPU,
        },
        # vLLM v1 path: vllm_config.parallel_config
        passthrough_path="llm_engine.vllm_config.parallel_config.distributed_executor_backend",
        description="vLLM distributed executor backend (mp=multiprocessing, ray=Ray cluster)",
        category="parallelism",
    ),
    ParamSpec(
        name="disable_custom_all_reduce",
        backend="vllm",
        config_path="vllm.disable_custom_all_reduce",
        test_values=[True, False],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.parallel_config.disable_custom_all_reduce",
        description="Disable vLLM's custom NCCL AllReduce kernel",
        category="parallelism",
    ),
]

# =============================================================================
# LoRA Parameters
# =============================================================================

LORA_SPECS = [
    ParamSpec(
        name="lora.enabled",
        backend="vllm",
        config_path="vllm.lora.enabled",
        test_values=[True],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        # vLLM v1 path: vllm_config.lora_config
        passthrough_path="llm_engine.vllm_config.lora_config",
        description="Enable LoRA adapter support",
        category="lora",
        # Custom checker since we just need to verify lora_config exists
        passthrough_checker=lambda llm, _: (
            llm.llm_engine.vllm_config.lora_config is not None,
            "LoRA config present"
            if llm.llm_engine.vllm_config.lora_config
            else "LoRA config is None",
        ),
    ),
    ParamSpec(
        name="lora.max_loras",
        backend="vllm",
        config_path="vllm.lora.max_loras",
        test_values=[1, 2],  # Keep small for faster tests
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        # vLLM v1 path: vllm_config.lora_config
        passthrough_path="llm_engine.vllm_config.lora_config.max_loras",
        description="Maximum number of concurrent LoRA adapters",
        category="lora",
    ),
    ParamSpec(
        name="lora.max_rank",
        backend="vllm",
        config_path="vllm.lora.max_rank",
        test_values=[8, 16],  # Keep small for faster tests
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        # vLLM v1 path: vllm_config.lora_config
        passthrough_path="llm_engine.vllm_config.lora_config.max_lora_rank",
        description="Maximum LoRA rank supported",
        category="lora",
    ),
]

# =============================================================================
# Sampling Parameters
# =============================================================================

SAMPLING_SPECS = [
    ParamSpec(
        name="best_of",
        backend="vllm",
        config_path="vllm.best_of",
        test_values=[1, 3],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        description="Generate N sequences, return best",
        category="sampling",
    ),
    ParamSpec(
        name="logprobs",
        backend="vllm",
        config_path="vllm.logprobs",
        test_values=[1, 5],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        description="Return top-k log probabilities per token",
        category="sampling",
    ),
]

# =============================================================================
# Quantization Parameters
# =============================================================================

QUANTIZATION_SPECS = [
    ParamSpec(
        name="load_format",
        backend="vllm",
        config_path="vllm.load_format",
        test_values=["auto", "safetensors"],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        passthrough_path="llm_engine.load_config.load_format",
        description="Weight loading format",
        category="quantization",
    ),
]

# =============================================================================
# Attention Parameters
# =============================================================================

ATTENTION_SPECS = [
    ParamSpec(
        name="attention.backend",
        backend="vllm",
        config_path="vllm.attention.backend",
        test_values=["auto", "FLASH_ATTN", "TORCH_SDPA"],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        description="Attention backend implementation",
        category="attention",
        skip_reason="Attention backend passthrough path varies by vLLM version",
    ),
]

# =============================================================================
# Speculative Decoding Parameters
# =============================================================================

SPECULATIVE_SPECS = [
    ParamSpec(
        name="speculative.method",
        backend="vllm",
        config_path="vllm.speculative.method",
        test_values=["ngram"],
        verification_type=VerificationType.BEHAVIOUR,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        description="Speculation method",
        category="speculative",
        skip_reason="Speculative decoding test requires specific model setup",
    ),
    ParamSpec(
        name="speculative.num_tokens",
        backend="vllm",
        config_path="vllm.speculative.num_tokens",
        test_values=[3, 5],
        verification_type=VerificationType.PASSTHROUGH,
        hardware_requirements={HardwareRequirement.GPU, HardwareRequirement.VLLM},
        description="Number of tokens to speculate per step",
        category="speculative",
        skip_reason="Speculative decoding test requires specific model setup",
    ),
]


# =============================================================================
# Registry
# =============================================================================

VLLM_PARAM_SPECS = (
    MEMORY_BATCHING_SPECS
    + KV_CACHE_SPECS
    + CONTEXT_SPECS
    + EXECUTION_SPECS
    + PARALLELISM_SPECS
    + LORA_SPECS
    + SAMPLING_SPECS
    + QUANTIZATION_SPECS
    + ATTENTION_SPECS
    + SPECULATIVE_SPECS
)


def register_vllm_params() -> None:
    """Register all vLLM parameter specs with the global registry."""
    register_all(VLLM_PARAM_SPECS)
