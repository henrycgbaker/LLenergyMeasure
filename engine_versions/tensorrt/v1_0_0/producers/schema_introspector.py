"""Schema introspector for TensorRT-LLM 1.0.0.

Runs inside ``nvcr.io/nvidia/tensorrt-llm/release:1.0.0`` (CUDA 13.x).
Introspects ``TrtLlmArgs`` and ``SamplingParams`` and writes a JSON schema file
with the common envelope. Import-driven (needs the installed library): this
introspector is a GPU-container artefact, not runnable on the host.

Surface contract:
  - TrtLlmArgs is a Pydantic v2 BaseModel with model_json_schema()
  - LlmArgs is an alias for TrtLlmArgs
  - BuildConfig is a stdlib dataclass (folded via a field->class hint)
  - KvCacheConfig / SchedulerConfig / CalibConfig / BuildCacheConfig are Pydantic
    (rendered via TrtLlmArgs' own $defs)
  - SamplingParams is a dataclass with public fields
  - tensorrt_llm.__commit__ is not exposed (null)

The ``LANDMARKS`` tuple is the probe contract (``scripts/_drift.py`` resolves
each dotted path under the installed library before discovery runs); a missing
entry flips the probe verdict to ``fail``.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    fold_dict_typed_subconfig,
    make_envelope,
    merge_source_constraints,
)

LANDMARKS: tuple[str, ...] = (
    "tensorrt_llm.SamplingParams",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs",
    "tensorrt_llm.llmapi.llm_args.TrtLlmArgs.model_json_schema",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM 1.0.0 engine and sampling schemas.

    engine_params:   TrtLlmArgs.model_json_schema() properties (with description + deprecated)
    sampling_params: dataclasses.fields(SamplingParams)
    """
    import tensorrt_llm  # type: ignore[import-not-found]
    from tensorrt_llm import SamplingParams  # type: ignore[import-not-found]
    from tensorrt_llm.builder import BuildConfig  # type: ignore[import-not-found]
    from tensorrt_llm.llmapi.llm_args import TrtLlmArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []

    raw_schema = TrtLlmArgs.model_json_schema()
    # ``model_json_schema()`` returns the nested-class $defs (KvCacheConfig /
    # SchedulerConfig / ...); seed the accumulator with them so the dataclass-lift
    # folds below (BuildConfig etc.) land in the same $defs namespace.
    defs: dict[str, Any] = dict(raw_schema.get("$defs") or {})
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

    # build_config / decoding_config / cp_config are non-Pydantic on TrtLlmArgs, so
    # model_json_schema() renders them as bare Optional[object]/Optional[dict] with no
    # $def, burying their leaves. BuildConfig is a stdlib dataclass -> fold it via the
    # field->class hint so the build-time levers (max_batch_size / max_num_tokens /
    # max_seq_len caps, plugin_config, weight_streaming, kv_cache_type, max_draft_len, ...)
    # become discoverable rather than hidden behind one opaque object.
    fold_dict_typed_subconfig(engine_params, "build_config", BuildConfig, defs)

    # decoding_config is also non-Pydantic; fold it if it resolves to an introspectable
    # class, else record the gap honestly (it is largely superseded by the richly-mined
    # speculative_config decoding union). DecodingConfig is a plain class on 1.0.0 (not a
    # dataclass/Pydantic model), so the fold no-ops and we record the limitation; a future
    # version that makes it a dataclass would fold automatically.
    decoding_reason: str | None = None
    try:
        from tensorrt_llm.llmapi.llm_args import (  # type: ignore[import-not-found]
            DecodingConfig,
        )

        if not fold_dict_typed_subconfig(engine_params, "decoding_config", DecodingConfig, defs):
            decoding_reason = (
                "DecodingConfig is a plain (non-Pydantic, non-dataclass) class; not "
                "introspectable by the dataclass-lift, and superseded by the mined "
                "speculative_config decoding union"
            )
    except Exception as exc:  # pragma: no cover - defensive: class-path drift
        decoding_reason = f"DecodingConfig not importable ({exc!r}); left as Optional[object]"
    if decoding_reason is not None:
        limitations.append(
            {"section": "engine_params", "fields": ["decoding_config"], "reason": decoding_reason}
        )

    # cp_config is a bare Optional[dict] (context-parallel settings: cp_type / cp_size /
    # ...) with NO schema class to introspect, so its leaves can only come from an
    # overlay. Recorded so the gap is explicit rather than silently absent from the
    # discovered surface.
    limitations.append(
        {
            "section": "engine_params",
            "fields": ["cp_config"],
            "reason": "cp_config is a bare dict (context-parallel settings) with no "
            "Pydantic/dataclass schema class; leaves are not introspectable and must "
            "come from curated overrides if exposed",
        }
    )

    sampling_params = dataclass_fields_to_specs(SamplingParams, skip_private=True)

    # Fold source-text Field(...) bounds + Literal[...] membership onto the
    # discovered SamplingParams fields. engine_params already carries pydantic's
    # own bounds/enums via model_json_schema(), so the walk targets the dataclass
    # sampling source where field metadata is otherwise absent - near-zero on
    # 1.0.0; the wiring carries forward to later sampling surfaces.
    sp_source = inspect.getsourcefile(SamplingParams)
    if sp_source is not None:
        merge_source_constraints(sampling_params, [Path(sp_source)])

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
        # The pydantic $defs (KvCacheConfig / SchedulerConfig / ...) plus the
        # dataclass-lift folds (BuildConfig / DecodingConfig) accumulated above.
        defs=defs,
    )
