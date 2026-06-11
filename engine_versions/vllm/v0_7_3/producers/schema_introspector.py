"""Schema-introspector machinery for vLLM 0.7.3.

Vendored machinery: the full ``discover`` body lives here, version-pinned.
The global ``scripts/engine_producers/vllm_introspector.py`` is a
thin dispatcher that resolves ``library.current_version`` from the SSOT
and delegates to this module.

vLLM 0.7.3 exposes the same runtime-introspection surface contract as
later versions: ``vllm.engine.arg_utils.EngineArgs`` is a stdlib
dataclass (~104 fields) and ``vllm.SamplingParams`` is msgspec-typed.
``msgspec.json.schema(SamplingParams)`` returns a ``$ref`` envelope with
the concrete definition under ``$defs.SamplingParams``; the property-
discovery loop below handles both shapes.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from scripts.engine_producers._common import (
    dataclass_fields_to_specs,
    make_envelope,
    merge_source_constraints,
)

LANDMARKS: tuple[str, ...] = (
    "vllm.SamplingParams",
    "vllm.engine.arg_utils.EngineArgs",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover vLLM 0.7.3 engine and sampling schemas.

    engine_params:   dataclasses.fields(EngineArgs)
    sampling_params: msgspec.json.schema(SamplingParams)
    """
    import vllm  # type: ignore[import-not-found]
    from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]

    limitations: list[dict[str, Any]] = []
    # Shared ``$defs`` accumulator: EngineArgs' Pydantic-typed sub-configs and
    # SamplingParams' nested msgspec definitions both land here so every ``$ref``
    # in the envelope resolves (2026-05-24 ``$defs`` resolution).
    defs: dict[str, Any] = {}
    engine_params = dataclass_fields_to_specs(EngineArgs, defs=defs)

    sampling_params: dict[str, Any] = {}
    try:
        import msgspec  # type: ignore[import-not-found]

        raw_schema = msgspec.json.schema(vllm.SamplingParams)
        raw_defs = raw_schema.get("$defs") or raw_schema.get("definitions") or {}
        props = raw_schema.get("properties")
        if not props:
            sp_def: Any = raw_defs.get("SamplingParams") or next(iter(raw_defs.values()), {})
            props = sp_def.get("properties", {}) if isinstance(sp_def, dict) else {}
        # Carry the nested sub-definitions (SamplingParams' own root is flattened
        # into ``sampling_params`` above, so skip it to avoid a redundant def).
        for def_name, def_body in raw_defs.items():
            if def_name != "SamplingParams":
                defs.setdefault(def_name, def_body)
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

    # D3: fold source-text Field(...) bounds + Literal[...] membership onto the
    # discovered fields. vLLM's numeric/value constraints live in imperative
    # _verify_args() (not field metadata), and EngineArgs is a flat stdlib
    # dataclass, so the declarative walk surfaces near-zero here on 0.7.3 - the
    # wiring is what carries forward to pins whose config classes use Field().
    try:
        import vllm.config as _vllm_config  # type: ignore[import-not-found]
        import vllm.sampling_params as _vllm_sp  # type: ignore[import-not-found]

        for module, fields in (
            (_vllm_config, engine_params),
            (_vllm_sp, sampling_params),
        ):
            src = inspect.getsourcefile(module)
            if src is not None:
                merge_source_constraints(fields, [Path(src)])
    except Exception:  # pragma: no cover - defensive: discovery proceeds without bounds
        pass

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
        defs=defs,
    )
