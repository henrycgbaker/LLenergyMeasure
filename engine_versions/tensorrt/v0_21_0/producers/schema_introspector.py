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

import importlib
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    discover_validation_collections,
    make_envelope,
    merge_validation_collections,
)

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.SamplingParams",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.model_json_schema",
)

# Validation-collection source modules. v0.21 uses
# ``_torch/pyexecutor/model_engine.py`` for ``_VALID_KV_CACHE_DTYPES``;
# v1.x moved it to ``model_loader.py`` (handled in the v1 vendored
# introspectors). Per-version filter: modules absent in this version
# raise ImportError / AttributeError and are skipped silently.
_VALIDATION_COLLECTION_MODULES: tuple[tuple[str, str], ...] = (
    ("tensorrt_llm._torch.pyexecutor.model_engine", "engine_params"),
    ("tensorrt_llm.quantization.quantize_by_modelopt", "engine_params"),
    ("tensorrt_llm.bench.utils", "engine_params"),
    ("tensorrt_llm.bench.benchmark.utils.general", "engine_params"),
    ("tensorrt_llm.lora_manager", "engine_params"),
)


def _collect_validation_collections() -> dict[str, dict[str, dict[str, Any]]]:
    """Return ``{section: {field_name: enum_spec}}`` lifted from validator gates.

    See :data:`_VALIDATION_COLLECTION_MODULES` for sources. Per-version
    filter: modules absent in this tensorrt-llm version are skipped silently.
    """
    by_section: dict[str, dict[str, dict[str, Any]]] = {
        "engine_params": {},
        "sampling_params": {},
    }
    for module_path, section in _VALIDATION_COLLECTION_MODULES:
        try:
            module = importlib.import_module(module_path)
        except (ImportError, AttributeError):
            continue
        try:
            collections = discover_validation_collections(module)
        except Exception:
            continue
        for field_name, spec in collections.items():
            by_section[section].setdefault(field_name, spec)
    return by_section


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM 0.21.0 engine and sampling schemas.

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

    # PR-0.5: lift module-scope validation-collection constants (e.g.
    # ``_VALID_KV_CACHE_DTYPES``, ``KV_QUANT_CFG_CHOICES``) into ``enum``
    # entries on matching fields.
    collections = _collect_validation_collections()
    merge_validation_collections(engine_params, collections["engine_params"])
    merge_validation_collections(sampling_params, collections["sampling_params"])

    # tensorrt-llm runs inside the NGC release image; the workflow passes
    # its concrete reference via --image-ref. No first-party Dockerfile
    # to read.
    return make_envelope(
        engine="tensorrt",
        engine_version=tensorrt_llm.__version__,
        engine_commit_sha=getattr(tensorrt_llm, "__commit__", None),
        image_ref=image_ref,
        base_image_ref=None,
        discovery_method="TrtLlmArgs.model_json_schema() + dataclasses.fields(SamplingParams) "
        "+ discover_validation_collections()",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
