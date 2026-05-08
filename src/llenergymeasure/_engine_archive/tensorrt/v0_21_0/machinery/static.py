"""Static-miner LANDMARKS for TensorRT-LLM 0.21.0.

The static miner walks an extracted source tree under
``/tmp/trt-llm-0.21.0/`` rather than the installed library; LANDMARKS
nevertheless reference the live ``tensorrt_llm`` package because that
is the seam Renovate's library bumps shift first. A missing landmark
under the installed library trips the probe; a divergent source tree
trips ``MinerLandmarkMissingError`` inside the miner.
"""

from __future__ import annotations

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_dtype",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_model",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_model_format_misc",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.set_runtime_knobs_from_build_config",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_build_config_with_runtime_params",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_build_config_remaining",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_speculative_config",
    "tensorrt_llm.llmapi.llm_args.BaseLlmArgs.validate_lora_config_consistency",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.validate_enable_build_cache",
    "tensorrt_llm.llmapi.llm_args.LookaheadDecodingConfig",
    "tensorrt_llm.llmapi.llm_args.LookaheadDecodingConfig.validate_positive_values",
    "tensorrt_llm.llmapi.llm_args.CalibConfig",
    "tensorrt_llm.llmapi.llm_args.BatchingType",
    "tensorrt_llm.llmapi.llm_args.CapacitySchedulerPolicy",
    "tensorrt_llm.llmapi.llm_args.ContextChunkingPolicy",
    "tensorrt_llm.builder.Builder",
    "tensorrt_llm.builder.Builder.build_engine",
)
