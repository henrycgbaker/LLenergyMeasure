"""vLLM engine + sampling schema introspector.

Runs inside ``vllm/vllm-openai:<tag>``. Introspects ``EngineArgs`` and
``SamplingParams`` and writes a JSON schema file with the common envelope.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.engine_introspectors._common import (
    dataclass_fields_to_specs,
    make_envelope,
)

# Symbols this introspector relies on inside the live ``vllm`` package.
# Read by ``scripts._probe`` before discovery runs; a missing landmark
# flips the probe verdict to ``fail`` and skips downstream discovery.
LANDMARKS: tuple[str, ...] = (
    "vllm.SamplingParams",
    "vllm.engine.arg_utils.EngineArgs",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover vLLM engine and sampling schemas.

    engine_params:   dataclasses.fields(EngineArgs)  (~86 fields)
    sampling_params: msgspec.json.schema(SamplingParams)  (~28 fields)
    """
    import vllm  # type: ignore[import-not-found]
    from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []
    engine_params = dataclass_fields_to_specs(EngineArgs)

    sampling_params: dict[str, Any] = {}
    try:
        import msgspec  # type: ignore[import-not-found]

        raw_schema = msgspec.json.schema(vllm.SamplingParams)
        props = raw_schema.get("properties")
        if not props:
            defs = raw_schema.get("$defs") or raw_schema.get("definitions") or {}
            sp_def: Any = defs.get("SamplingParams") or next(iter(defs.values()), {})
            props = sp_def.get("properties", {}) if isinstance(sp_def, dict) else {}
        for name, spec in (props or {}).items():
            type_repr: Any = spec.get("type", "unknown")
            if isinstance(type_repr, list):
                type_repr = " | ".join(str(t) for t in type_repr)
            sampling_params[name] = {
                "type": type_repr,
                "default": spec.get("default"),
            }
    except Exception as exc:
        limitations.append(
            {
                "section": "sampling_params",
                "fields": [],
                "reason": f"msgspec.json.schema(SamplingParams) failed: {exc!r}",
            }
        )

    limitations.append(
        {
            "section": "sampling_params",
            "fields": [],
            "reason": "constraints (e.g. temperature>=0, top_p in (0,1]) live in imperative "
            "_verify_args() and are not introspectable from field metadata",
        }
    )
    limitations.append(
        {
            "section": "engine_params",
            "fields": [],
            "reason": "per-field descriptions unavailable (vLLM EngineArgs has only a class docstring)",
        }
    )

    # vllm runs inside the upstream vllm/vllm-openai image; the workflow
    # passes its concrete reference via --image-ref. No first-party
    # Dockerfile to read.
    return make_envelope(
        engine="vllm",
        engine_version=vllm.__version__,
        engine_commit_sha=getattr(vllm, "__commit__", None),
        image_ref=image_ref,
        base_image_ref=None,
        discovery_method="dataclasses.fields(EngineArgs) + msgspec.json.schema(SamplingParams)",
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
    )
