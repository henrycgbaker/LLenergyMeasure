"""Schema-introspector machinery for vLLM 0.19.1.

Vendored machinery: the full ``discover`` body lives here, version-pinned.
The global ``scripts/engine_producers/vllm_introspector.py`` is a
thin dispatcher that resolves ``library.current_version`` from the SSOT
and delegates to this module.

vLLM 0.19.1 keeps the same runtime-introspection surface contract as
0.7.3 - ``vllm.engine.arg_utils.EngineArgs`` is a stdlib dataclass and
``vllm.SamplingParams`` is msgspec-typed - so the dataclass/msgspec
discovery path is unchanged. What moved is the *declarative constraint
surface*: 0.7.3's flat ``vllm/config.py`` became the ``vllm/config/*.py``
subpackage (cache, scheduler, parallel, model, ...) and the per-concern
config classes migrated to pydantic-dataclass ``Field(ge/le/...)`` bounds
and ``Literal[...]`` membership sets. The source-constraint overlay below
therefore globs the whole ``config/`` subpackage (P5) rather than the
single flat file, so the declarative walker (P3 + P4) reaches the bounds
and enums that no longer live in the imperative ``_verify_*`` bodies.
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
from scripts.engine_producers._source_walker import expand_files

LANDMARKS: tuple[str, ...] = (
    "vllm.SamplingParams",
    "vllm.engine.arg_utils.EngineArgs",
)


def discover(repo_root: Path, image_ref: str | None) -> dict[str, Any]:
    """Discover vLLM 0.19.1 engine and sampling schemas.

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

    # D3 + P5: fold source-text Field(...) bounds + Literal[...] membership onto
    # the discovered fields. On 0.19.1 the per-concern config classes (CacheConfig,
    # SchedulerConfig, ParallelConfig, ...) live in the ``vllm/config/*.py``
    # subpackage and carry their constraints declaratively, so the overlay globs
    # the WHOLE subpackage rather than the single ``config/__init__.py`` that
    # ``inspect.getsourcefile(vllm.config)`` resolves to. The flat single-file
    # walk (0.7.3) would miss every per-concern module - this is the recall cliff
    # the subpackage glob recovers.
    try:
        import vllm.config as _vllm_config  # type: ignore[import-not-found]
        import vllm.sampling_params as _vllm_sp  # type: ignore[import-not-found]

        config_init = inspect.getsourcefile(_vllm_config)
        config_paths = (
            expand_files(Path(config_init).parent, ["*.py"]) if config_init is not None else []
        )
        merge_source_constraints(engine_params, config_paths)
        sp_src = inspect.getsourcefile(_vllm_sp)
        if sp_src is not None:
            merge_source_constraints(sampling_params, [Path(sp_src)])
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
        discovery_method=(
            "dataclasses.fields(EngineArgs) + msgspec.json.schema(SamplingParams) "
            "+ config/*.py declarative-constraint overlay"
        ),
        discovery_limitations=limitations,
        engine_params=engine_params,
        sampling_params=sampling_params,
        defs=defs,
    )
