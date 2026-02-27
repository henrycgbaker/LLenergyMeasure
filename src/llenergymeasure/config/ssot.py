"""Single source of truth for backend capability constants.

These dicts define which precision modes and decoding strategies each backend
supports. They are consumed by:
- ExperimentConfig cross-validators (structural validation)
- config/introspection.py (backend capability metadata)
- Future CLI help generation (Phase 7)

Do not inline these values in validators — always import from here.
"""

# Precision modes supported by each backend.
# "fp32" = full precision, "fp16" = half, "bf16" = bfloat16.
# Note: fp16/bf16 require GPU. The cpu backend (future) would be fp32-only.
# GPU detection and cpu-precision cross-validation is Phase 4 (pre-flight).
PRECISION_SUPPORT: dict[str, list[str]] = {
    "pytorch": ["fp32", "fp16", "bf16"],
    "vllm": ["fp16", "bf16"],  # vLLM does not support fp32 inference
    "tensorrt": ["fp16", "bf16"],  # TRT-LLM does not support fp32 inference
}

# Decoding strategies supported by each backend.
# "sampling" = do_sample=True path; "greedy" = do_sample=False path.
DECODING_SUPPORT: dict[str, list[str]] = {
    "pytorch": ["greedy", "sampling"],  # full HuggingFace generate() support
    "vllm": ["greedy", "sampling"],  # vLLM supports both via SamplingParams
    "tensorrt": ["greedy", "sampling"],  # TRT-LLM supports both
}

# Backends that support the full DecoderConfig temperature/top_k/top_p fields.
# All current backends support these — this dict exists to make future
# backend additions explicit rather than implicit.
DECODER_PARAM_SUPPORT: dict[str, list[str]] = {
    "pytorch": ["temperature", "top_k", "top_p", "repetition_penalty"],
    "vllm": ["temperature", "top_k", "top_p", "repetition_penalty"],
    "tensorrt": ["temperature", "top_k", "top_p"],  # TRT-LLM: repetition_penalty support varies
}

__all__ = ["DECODER_PARAM_SUPPORT", "DECODING_SUPPORT", "PRECISION_SUPPORT"]
