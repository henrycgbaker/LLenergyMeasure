"""TensorRT-LLM 0.21.0 schema introspector machinery.

Runs inside ``nvcr.io/nvidia/tensorrt-llm/release:0.21.0``. Introspects
``TrtLlmArgs`` and ``SamplingParams`` and writes a JSON schema file with
the common envelope.

Spike (2026-04-13, TRT-LLM 0.21.0 in pristine NGC image):
  - TrtLlmArgs is a Pydantic v2 BaseModel with model_json_schema() (61 fields)
  - LlmArgs is an alias for TrtLlmArgs
  - BuildConfig is NOT Pydantic -> appears as Optional[object] in the schema
  - KvCacheConfig / SchedulerConfig / CalibConfig / BuildCacheConfig are
    Pydantic (fallback path, unused because primary path works)
  - SamplingParams is a dataclass with 47+ public fields
  - tensorrt_llm.__commit__ is not exposed (null)

LANDMARKS for the introspector reference the live ``tensorrt_llm`` package
inside the 0.21.0 NGC image. A missing landmark at probe time means the
installed library no longer exposes a structure this introspector expects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    make_envelope,
    pydantic_properties_to_specs,
)

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.SamplingParams",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.model_json_schema",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM 0.21.0 engine and sampling schemas.

    engine_params:   TrtLlmArgs Pydantic JSON Schema properties + nested $defs
                     (preserves enum, minimum, maximum, $ref, description,
                     deprecated)
    sampling_params: dataclasses.fields(SamplingParams) (Literal annotations
                     surface as ``enum`` on the spec dict)
    """
    import tensorrt_llm  # type: ignore[import-not-found]
    from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []

    engine_params, engine_defs = pydantic_properties_to_specs(TrtLlmArgs)

    limitations.append(
        {
            "section": "engine_params",
            "fields": ["build_config"],
            "reason": "BuildConfig is not a Pydantic model; appears as Optional[object] in the schema",
        }
    )

    sampling_params = dataclass_fields_to_specs(SamplingParams, skip_private=True)

    # TRT-LLM's SamplingParams dataclass doesn't set field.metadata['description']
    # for any field, so the dataclass_fields_to_specs description path stays empty
    # here. Literal-annotated fields still surface ``enum`` via the helper.
    if not any(spec.get("description") for spec in sampling_params.values()):
        limitations.append(
            {
                "section": "sampling_params",
                "fields": [],
                "reason": "SamplingParams is a dataclass with no per-field "
                "field(metadata={'description': ...}) metadata; descriptions unavailable",
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
        engine_params_defs=engine_defs,
    )
