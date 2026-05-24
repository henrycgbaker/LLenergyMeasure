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

import importlib
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    discover_validation_collections,
    jsonschema_property_to_canonical,
    make_envelope,
    merge_validation_collections,
)

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.SamplingParams",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.model_json_schema",
)

# Validation-collection source modules. v1.x relocates the KV-cache-dtype
# gate from ``model_engine.py`` (v0.21) to ``model_loader.py``; per-version
# filter handles either path via try/except. Per-version filter: modules
# absent in this version raise ImportError / AttributeError and are skipped.
_VALIDATION_COLLECTION_MODULES: tuple[tuple[str, str], ...] = (
    ("tensorrt_llm._torch.pyexecutor.model_loader", "engine_params"),
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
        engine_params[name] = jsonschema_property_to_canonical(spec)

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
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
