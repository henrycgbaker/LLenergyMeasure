"""TensorRT-LLM engine + sampling schema introspector.

Runs inside ``nvcr.io/nvidia/tensorrt-llm/release:<tag>``. Introspects
``TrtLlmArgs`` and ``SamplingParams`` and writes a JSON schema file with
the common envelope.

Spike (2026-04-13, TRT-LLM 0.21.0 in pristine NGC image):
  - TrtLlmArgs is a Pydantic v2 BaseModel with model_json_schema() (61 fields)
  - LlmArgs is an alias for TrtLlmArgs
  - BuildConfig is NOT Pydantic -> appears as Optional[object] in the schema
  - KvCacheConfig / SchedulerConfig / CalibConfig / BuildCacheConfig are
    Pydantic (fallback path, unused because primary path works)
  - SamplingParams is a dataclass with 47 public fields
  - tensorrt_llm.__commit__ is not exposed (null)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.engine_introspectors._common import (
    dataclass_fields_to_specs,
    make_envelope,
)
from scripts.engine_miners._ssot import load_ssot

# Symbols this introspector relies on inside the live ``tensorrt_llm``
# package. Read by ``scripts._probe`` before discovery runs; a missing
# landmark flips the probe verdict to ``fail`` and skips downstream
# discovery.
#
# Sourced from the version-pinned archive at
# ``llenergymeasure._engine_archive.tensorrt.<safe_version>.machinery.discovery``.
# PEP 562 ``__getattr__`` defers the archive import + SSOT parse until
# accessed.


def _get_landmarks() -> tuple[str, ...]:
    """Resolve LANDMARKS for the current SSOT ``library.current_version``."""
    cached = globals().get("LANDMARKS")
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    from llenergymeasure._engine_archive._dispatcher import load_machinery

    ssot = load_ssot("tensorrt")
    library = ssot.get("library")
    if not isinstance(library, dict) or "current_version" not in library:
        raise ValueError(
            "engine_versions/tensorrt.yaml is missing library.current_version; "
            "cannot resolve archived LANDMARKS."
        )
    landmarks = load_machinery(
        engine="tensorrt",
        version=str(library["current_version"]),
        producer="discovery",
    ).LANDMARKS
    globals()["LANDMARKS"] = landmarks
    return landmarks  # type: ignore[no-any-return]


def __getattr__(name: str) -> object:
    """PEP 562 hook: lazy LANDMARKS export from the per-version archive."""
    if name == "LANDMARKS":
        return _get_landmarks()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover TensorRT-LLM engine and sampling schemas.

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
