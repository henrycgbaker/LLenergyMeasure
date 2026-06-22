"""Frozen TensorRT-LLM 1.0.0 machinery + outputs (the 0.x -> 1.x major).

Vendored snapshot for the engine-update-workflow replay leg crossing
TensorRT-LLM's first major. The 1.0.0 pin reflects the CUDA 13.x
compatibility envelope (0.21.0 was CUDA 12.6.x).

The C++ -> pydantic migration the mining study predicted across this major
DID land, but as imperative ``@field_validator`` / ``@model_validator``
bodies, not declarative ``Field(ge=...)`` bounds - so the imperative floor
walker (not the declarative walker) recovers the new constraints. Full
measurement: ``.product/research/tensorrt-leg-findings-2026-06-14.md``.

LANDMARKS for the static miner reference symbols in the extracted source
tree at ``/tmp/trt-llm-1.0.0/``; LANDMARKS for the discovery introspector
reference the live ``tensorrt_llm`` package inside
``nvcr.io/nvidia/tensorrt-llm/release:1.0.0`` (Phase-B / GPU container).
"""
