"""Schema-introspector LANDMARKS for TensorRT-LLM 0.21.0.

The introspector runs inside ``nvcr.io/nvidia/tensorrt-llm/release:0.21.0``
and lifts engine parameter specs from ``TrtLlmArgs`` (Pydantic v2) and
sampling parameter specs from ``SamplingParams`` (dataclass).
"""

from __future__ import annotations

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.SamplingParams",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.model_json_schema",
)
