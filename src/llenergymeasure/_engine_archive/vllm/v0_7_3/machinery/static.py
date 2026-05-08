"""Static-miner LANDMARKS for vLLM 0.7.3.

These are the dotted attribute paths the probe verifies under the
installed ``vllm`` package before the static miner runs. Mirrors the
``_AST_TARGETS`` registry in ``scripts/engine_miners/vllm_static_miner.py``
at the dotted-path level, plus class targets needed for AST walking.

Frozen at vllm 0.7.3. Subsequent versions live alongside this file as
sibling subpackages (``v0_16_0``, ``v0_18_1``, ...) populated by the
chunk PRs that bump ``current_version`` in the SSOT.
"""

from __future__ import annotations

LANDMARKS: tuple[str, ...] = (
    "vllm.sampling_params.SamplingParams",
    "vllm.sampling_params.SamplingParams._verify_args",
    "vllm.sampling_params.SamplingParams.__post_init__",
    "vllm.sampling_params.SamplingParams._verify_greedy_sampling",
    "vllm.sampling_params.StructuredOutputsParams",
    "vllm.sampling_params.StructuredOutputsParams.__post_init__",
    "vllm.config.parallel.ParallelConfig",
    "vllm.config.parallel.ParallelConfig._validate_parallel_config",
    "vllm.config.parallel.ParallelConfig._verify_args",
    "vllm.config.parallel.ParallelConfig.__post_init__",
    "vllm.config.parallel.EPLBConfig",
    "vllm.config.parallel.EPLBConfig._validate_eplb_config",
    "vllm.config.lora.LoRAConfig",
    "vllm.config.lora.LoRAConfig._validate_lora_config",
    "vllm.config.multimodal.MultiModalConfig",
    "vllm.config.multimodal.MultiModalConfig._validate_multimodal_config",
    "vllm.config.structured_outputs.StructuredOutputsConfig",
    "vllm.config.structured_outputs.StructuredOutputsConfig._validate_structured_output_config",
    "vllm.config.cache.CacheConfig",
    "vllm.config.cache.CacheConfig._validate_cache_dtype",
    "vllm.config.model.ModelConfig",
    "vllm.config.model.ModelConfig.__post_init__",
    "vllm.config.compilation.CompilationConfig",
    "vllm.config.compilation.CompilationConfig.__post_init__",
    "vllm.config.scheduler.SchedulerConfig",
    "vllm.config.scheduler.SchedulerConfig.__post_init__",
)
