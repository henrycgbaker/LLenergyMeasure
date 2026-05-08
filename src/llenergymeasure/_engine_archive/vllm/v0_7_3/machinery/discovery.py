"""Schema-introspector LANDMARKS for vLLM 0.7.3.

Read by the probe before the introspector runs inside the
``vllm/vllm-openai:v0.7.3`` image. The introspector extracts engine
parameter specs from ``EngineArgs`` (dataclass) and sampling parameter
specs from ``SamplingParams`` (msgspec-typed).
"""

from __future__ import annotations

LANDMARKS: tuple[str, ...] = (
    "vllm.SamplingParams",
    "vllm.engine.arg_utils.EngineArgs",
)
