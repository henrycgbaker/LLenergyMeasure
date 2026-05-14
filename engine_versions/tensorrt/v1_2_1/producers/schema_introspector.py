"""TensorRT-LLM 1.2.1 schema introspector machinery.

Runs inside ``nvcr.io/nvidia/tensorrt-llm/release:1.2.1``. Introspects
``TrtLlmArgs`` and ``SamplingParams`` and writes a JSON schema file with
the common envelope.

Confirmed structure in 1.2.1 (from in-container probe):
  - TrtLlmArgs is a Pydantic v2 BaseModel with model_json_schema() (~64 fields)
  - In 1.2.x, ``LlmArgs`` is ``TorchLlmArgs`` (not ``TrtLlmArgs``); this
    introspector targets ``TrtLlmArgs`` directly via its module-qualified path.
  - BuildConfig is NOT Pydantic -> appears as Optional[object] in the schema
  - KvCacheConfig / SchedulerConfig / CalibConfig / BuildCacheConfig are
    Pydantic (fallback path, unused because primary path works)
  - SamplingParams is a dataclass with 54+ public fields
  - tensorrt_llm.__commit__ is not exposed (null)
  - 1.x-only classes ``StrictBaseModel``, ``MoeConfig``, ``CudaGraphConfig``
    surface as ``$ref`` types when referenced by TrtLlmArgs fields.

LANDMARKS for the introspector reference the live ``tensorrt_llm`` package
inside the 1.2.1 NGC image. A missing landmark at probe time means the
installed library no longer exposes a structure this introspector expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    make_envelope,
)

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.SamplingParams",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.model_json_schema",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM 1.2.1 engine and sampling schemas.

    engine_params:   TrtLlmArgs.model_json_schema() properties (with description + deprecated)
    sampling_params: dataclasses.fields(SamplingParams)
    """
    import tensorrt_llm  # type: ignore[import-not-found]
    from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []

    raw_schema = TrtLlmArgs.model_json_schema()
    engine_params: dict[str, Any] = {}
    for name, spec in raw_schema.get("properties", {}).items():
        if name.startswith("_"):
            continue
        type_repr: Any = spec.get("type")
        if type_repr is None and "anyOf" in spec:
            parts: list[str] = []
            for sub in spec["anyOf"]:
                if "type" in sub:
                    part = "None" if sub["type"] == "null" else str(sub["type"])
                elif "$ref" in sub:
                    part = str(sub["$ref"]).rsplit("/", 1)[-1]
                else:
                    continue
                if part not in parts:  # dedupe string | string etc.
                    parts.append(part)
            type_repr = " | ".join(parts) if parts else "unknown"
        elif type_repr is None and "$ref" in spec:
            type_repr = str(spec["$ref"]).rsplit("/", 1)[-1]
        if isinstance(type_repr, list):
            type_repr = " | ".join("None" if t == "null" else str(t) for t in type_repr)
        elif type_repr == "null":
            type_repr = "None"
        engine_params[name] = {
            "type": type_repr or "unknown",
            "default": spec.get("default"),
            "description": spec.get("description"),
            "deprecated": spec.get("deprecated", False),
        }

    limitations.append(
        {
            "section": "engine_params",
            "fields": ["build_config"],
            "reason": "BuildConfig is not a Pydantic model; appears as Optional[object] in the schema",
        }
    )

    sampling_params = dataclass_fields_to_specs(SamplingParams, skip_private=True)

    limitations.append(
        {
            "section": "sampling_params",
            "fields": [],
            "reason": "SamplingParams is a dataclass; no per-field descriptions",
        }
    )

    # tensorrt-llm runs inside the NGC release image; the workflow passes
    # its concrete reference via --image-ref. No first-party Dockerfile
    # to read.
    return make_envelope(
        engine="tensorrt",
        engine_version=tensorrt_llm.__version__,
        engine_commit_sha=getattr(tensorrt_llm, "__commit__", None),
        image_ref=image_ref,
        base_image_ref=None,
        discovery_method="TrtLlmArgs.model_json_schema() + dataclasses.fields(SamplingParams)",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
