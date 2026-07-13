"""TensorRT-LLM 1.2.1 schema introspector machinery.

Runs inside ``nvcr.io/nvidia/tensorrt-llm/release:1.2.1``. Introspects BOTH
backend args classes - ``TorchLlmArgs`` (pytorch backend) and ``TrtLlmArgs``
(trt backend) - plus ``SamplingParams``, and writes a JSON schema file with the
common envelope.

Per-field backend applicability (the union-format extension)
------------------------------------------------------------
The two backend legs are selected by class at construction: backend
``pytorch``/None -> ``tensorrt_llm.LLM`` (validates ``TorchLlmArgs``); backend
``trt`` -> ``tensorrt_llm._tensorrt_engine.LLM`` (validates ``TrtLlmArgs``). A
field can live on one args class, the other, or both. We mine BOTH classes and
UNION them into one ``engine_params`` surface, tagging each field with a
``backends`` list naming the backend(s) whose args class carries it:
``["pytorch", "trt"]`` (shared), ``["trt"]`` (trt-only, e.g. ``fast_build`` /
``quant_config`` / ``enable_build_cache``), or ``["pytorch"]`` (pytorch-only,
e.g. ``cuda_graph_config`` / ``stream_interval`` / ``batch_wait_*``).

``backends`` is DESCRIPTIVE schema metadata: codegen keeps one ``EngineParams``
and ignores it; cross-backend applicability is enforced by loud validation
rules in the corpus ("field X requires backend Y"), never by silently dropping
a field. Absence of the ``backends`` key means "all backends" - the convention
the single-class transformers / vllm schemas rely on and that ``sampling_params``
(a backend-shared class) keeps.

Merge precedence for a SHARED field's spec (type / default / description /
deprecated): the trt (``TrtLlmArgs``) spec wins, so the previously-committed
63-field ``TrtLlmArgs`` surface stays byte-continuous and the ``TorchLlmArgs``-
only fields append after it. Any shared field whose two classes disagree on
spec is recorded as a discovery limitation for review (at 1.2.1 the only
divergence is ``load_format``).

Confirmed structure in 1.2.1 (from in-container probe):
  - ``TorchLlmArgs`` (82 fields) and ``TrtLlmArgs`` (63 fields) are Pydantic v2
    BaseModels with ``model_json_schema()``; the union is 95 fields (50 shared,
    32 pytorch-only, 13 trt-only).
  - ``BuildConfig`` is NOT Pydantic -> ``build_config`` appears as
    ``Optional[object]`` in the schema.
  - ``SamplingParams`` is a dataclass, backend-shared (no per-field backends).
  - 1.x-only classes ``StrictBaseModel``, ``MoeConfig``, ``CudaGraphConfig``
    surface as ``$ref`` type names when referenced by a field.

LANDMARKS reference the live ``tensorrt_llm`` package inside the 1.2.1 NGC
image. A missing landmark at probe time means the installed library no longer
exposes a structure this introspector expects.
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
    "tensorrt_llm.llmapi.llm_args.TorchLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TorchLlmArgs.model_json_schema",
)

# Exposed backend value -> the args class whose field surface it selects. The
# trt leg is listed FIRST so a shared field's spec precedence (trt wins) and
# byte-continuity with the prior TrtLlmArgs-only schema fall out of iteration
# order: the 63 TrtLlmArgs fields keep their order, the TorchLlmArgs-only
# fields append after. These are exactly the two backends curated.yaml narrows
# the ``backend`` field to.
_BACKEND_ARGS_CLASSES: tuple[tuple[str, str], ...] = (
    ("trt", "TrtLlmArgs"),
    ("pytorch", "TorchLlmArgs"),
)

_SPEC_KEYS: tuple[str, ...] = ("type", "default", "description", "deprecated")


def _field_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """Render one ``model_json_schema()`` property into the envelope field shape."""
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
    return {
        "type": type_repr or "unknown",
        "default": spec.get("default"),
        "description": spec.get("description"),
        "deprecated": spec.get("deprecated", False),
    }


def _extract_params(raw_schema: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract ``{name: field_spec}`` from one args class's ``model_json_schema()``."""
    out: dict[str, dict[str, Any]] = {}
    for name, spec in raw_schema.get("properties", {}).items():
        if name.startswith("_"):
            continue
        out[name] = _field_spec(spec)
    return out


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM 1.2.1 engine and sampling schemas across both backends.

    engine_params:   union of ``TorchLlmArgs`` + ``TrtLlmArgs``
                     ``model_json_schema()`` properties, each tagged with a
                     per-field ``backends`` applicability list.
    sampling_params: ``dataclasses.fields(SamplingParams)`` (backend-shared, so
                     no ``backends`` key - absence means "all backends").
    """
    import tensorrt_llm  # type: ignore[import-not-found]
    from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]
    from tensorrt_llm.llmapi import llm_args  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []

    # Mine each backend's args class up front. Iterating trt first below keeps
    # its field order and gives its spec precedence on shared fields.
    per_backend: dict[str, dict[str, dict[str, Any]]] = {
        backend: _extract_params(getattr(llm_args, cls_name).model_json_schema())
        for backend, cls_name in _BACKEND_ARGS_CLASSES
    }

    # Union into one surface, recording per-field applicability and any
    # shared-field spec divergence between the two classes.
    engine_params: dict[str, dict[str, Any]] = {}
    conflicts: set[str] = set()
    for backend, _cls_name in _BACKEND_ARGS_CLASSES:
        for name, spec in per_backend[backend].items():
            existing = engine_params.get(name)
            if existing is None:
                field = dict(spec)
                field["backends"] = [backend]
                engine_params[name] = field
            else:
                existing["backends"].append(backend)
                if any(existing.get(k) != spec.get(k) for k in _SPEC_KEYS):
                    conflicts.add(name)
    # Deterministic applicability ordering; ``backends`` stays the last key.
    for field in engine_params.values():
        field["backends"] = sorted(set(field["backends"]))

    limitations.append(
        {
            "section": "engine_params",
            "fields": ["build_config"],
            "reason": "BuildConfig is not a Pydantic model; appears as Optional[object] in the schema",
        }
    )
    if conflicts:
        limitations.append(
            {
                "section": "engine_params",
                "fields": sorted(conflicts),
                "reason": (
                    "shared field: TorchLlmArgs and TrtLlmArgs disagree on "
                    "type/default/description; the trt (TrtLlmArgs) spec is "
                    "recorded and the field is tagged applicable to both backends"
                ),
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

    # tensorrt-llm runs inside the NGC release image; the workflow passes its
    # concrete reference via --image-ref. No first-party Dockerfile to read.
    return make_envelope(
        engine="tensorrt",
        engine_version=tensorrt_llm.__version__,
        engine_commit_sha=getattr(tensorrt_llm, "__commit__", None),
        image_ref=image_ref,
        base_image_ref=None,
        discovery_method=(
            "TorchLlmArgs+TrtLlmArgs.model_json_schema() union with per-field "
            "backends + dataclasses.fields(SamplingParams)"
        ),
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
