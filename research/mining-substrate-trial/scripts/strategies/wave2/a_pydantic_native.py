"""Wave 2d a-pydantic-native: framework-reflection-only deterministic substrate.

See WAVE2_PROTOCOL.md section 3 Tier A (row W2-a-pydantic-native): lean on
the engine framework's OWN introspection surface (Pydantic ``model_fields``,
msgspec ``json.schema``, stdlib dataclass ``__dataclass_fields__``,
``inspect.signature``) instead of AST-walking validator method bodies.

Hypothesis under test: framework-native reflection cheaply produces high
schema recall, but only structural (NOT body-derived) invariants. That gap
IS the empirical finding being measured; this module deliberately emits
``severity="structural"`` invariants with an empty ``match.fields`` so the
scorer can quantify the missing-predicate cost.

Scope: vllm 0.7.3 only. Single-engine pilot per protocol section 6 step 4.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone
from typing import Any

import yaml

from strategies.wave2._base import CellOutput, standard_out_dir

# Stdlib-dataclass config classes under ``vllm.config`` whose fields should
# surface as ``$defs`` entries in the schema envelope. Each is also scanned
# for ``_verify_*`` + ``__post_init__`` methods to emit structural invariants.
_VLLM_CONFIG_CLASSES: tuple[str, ...] = (
    "ModelConfig",
    "CacheConfig",
    "ParallelConfig",
    "SchedulerConfig",
    "DeviceConfig",
    "LoadConfig",
    "LoRAConfig",
    "PromptAdapterConfig",
    "ObservabilityConfig",
)

# Method-name patterns that mark a validator-style entry point on a
# stdlib-dataclass config. Reflection only sees the method exists; body
# predicates are out of scope for this substrate (that's the gap we measure).
_VALIDATOR_METHOD_PREFIXES: tuple[str, ...] = ("_verify_",)
_VALIDATOR_METHOD_EXACT: frozenset[str] = frozenset({"__post_init__"})

_SOURCE_TAG = "framework_native_reflection"
_STRATEGY_ID = "w2-a-pydantic-native"
_SCHEMA_VERSION = "2.0.0"
_INVARIANTS_SCHEMA_VERSION = "1.0.0"


def _json_schema_type_for(py_type: Any) -> str | dict[str, Any]:
    """Best-effort mapping from a Python annotation to a JSON Schema type.

    Reflection-only: we read ``dataclasses.Field.type`` (which may be a
    string forward-ref or a real type) and map primitives. Anything we
    can't classify becomes the catch-all ``"any"`` rather than guessing.
    """
    if py_type is None or py_type is type(None):
        return "null"
    name = getattr(py_type, "__name__", None) or str(py_type)
    name = name.strip("'\"")
    primitive_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "bytes": "string",
        "list": "array",
        "tuple": "array",
        "dict": "object",
        "set": "array",
        "NoneType": "null",
    }
    if name in primitive_map:
        return primitive_map[name]
    return {"type": "any", "x-py-annotation": name}


def _reflect_dataclass_fields(cls: type) -> dict[str, dict[str, Any]]:
    """Return ``{field_name: {name, json_schema_type, default, x-source}}``.

    Pure ``dataclasses.fields()`` reflection. No AST walking. Defaults that
    are sentinels (``MISSING`` / ``dataclass_factory``) collapse to ``None``.
    """
    out: dict[str, dict[str, Any]] = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = None
        else:
            default = None
        out[f.name] = {
            "name": f.name,
            "json_schema_type": _json_schema_type_for(f.type),
            "default": default,
            "x-source": _SOURCE_TAG,
        }
    return out


def _reflect_msgspec_schema(cls: type) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(properties, $defs)`` from ``msgspec.json.schema(cls)``.

    msgspec emits a JSON-Schema envelope where the root schema lives under
    ``$defs[ClassName]`` and the top-level is a ``$ref``. We unpack the
    properties block and pass through any sibling defs.
    """
    import msgspec  # type: ignore[import-not-found]

    raw = msgspec.json.schema(cls)
    defs = raw.get("$defs") or raw.get("definitions") or {}
    props = raw.get("properties")
    if not props:
        root = defs.get(cls.__name__) or next(iter(defs.values()), {})
        props = root.get("properties", {}) if isinstance(root, dict) else {}
    canon: dict[str, Any] = {}
    for name, spec in (props or {}).items():
        canon[name] = {
            "name": name,
            "json_schema_type": spec.get("type") or spec.get("$ref") or "any",
            "default": spec.get("default"),
            "x-source": _SOURCE_TAG,
        }
    return canon, dict(defs)


def _enumerate_validator_methods(cls: type) -> list[str]:
    """Return the names of methods that look like validator entry points.

    Reflection only: we inspect ``vars(cls)`` for callables whose names
    match the validator-prefix / exact-name patterns. No body inspection.
    """
    found: list[str] = []
    for name, member in vars(cls).items():
        if not callable(member):
            continue
        if name in _VALIDATOR_METHOD_EXACT or any(
            name.startswith(p) for p in _VALIDATOR_METHOD_PREFIXES
        ):
            found.append(name)
    return sorted(found)


def _structural_invariant(*, cls: type, method_name: str) -> dict[str, Any]:
    """Emit ONE structural invariant record for a (class, method) pair.

    ``match.fields`` is deliberately empty: this substrate cannot derive
    predicates from method bodies, and recording that gap is the point.
    """
    fqn = f"{cls.__module__}.{cls.__qualname__}"
    return {
        "id": f"vllm_{cls.__name__.lower()}_{method_name}_structural",
        "engine": "vllm",
        "library": "vllm",
        "native_type": fqn,
        "miner_source": {"method": method_name},
        "severity": "structural",
        "match": {"engine": "vllm", "fields": {}},
        "added_by": "wave2_pydantic_native",
    }


def run_cell(*, engine: str, version: str) -> CellOutput:
    """Execute the Wave 2d a-pydantic-native cell for vllm 0.7.3."""
    if engine != "vllm" or version not in {"v0.7.3", "0.7.3"}:
        raise NotImplementedError(
            f"a_pydantic_native is vllm 0.7.3-only; got engine={engine!r} version={version!r}"
        )

    try:
        import vllm  # type: ignore[import-not-found]
        import vllm.config as vllm_config  # type: ignore[import-not-found]
        from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]
    except ImportError as e:
        raise RuntimeError(
            "a_pydantic_native requires vllm installed in the trial venv. "
            "Install via `uv add vllm==0.7.3` in the wave2 venv."
        ) from e

    version_slug = "v0_7_3"
    out_dir = standard_out_dir(_STRATEGY_ID, engine, version_slug)
    observations: list[str] = []
    wall: dict[str, float] = {}

    # Phase 1: schema reflection.
    t0 = time.perf_counter()
    engine_params = _reflect_dataclass_fields(EngineArgs)
    sampling_params: dict[str, Any] = {}
    all_defs: dict[str, Any] = {}
    try:
        sampling_params, sampling_defs = _reflect_msgspec_schema(vllm.SamplingParams)
        for name, spec in sampling_defs.items():
            if name == "SamplingParams":
                continue
            all_defs.setdefault(name, spec)
    except Exception as exc:
        observations.append(f"sampling_params reflection failed: {exc!r}")

    config_classes: list[type] = []
    for cls_name in _VLLM_CONFIG_CLASSES:
        cls = getattr(vllm_config, cls_name, None)
        if cls is None:
            observations.append(f"vllm.config.{cls_name} not found in this version")
            continue
        if not dataclasses.is_dataclass(cls):
            observations.append(f"vllm.config.{cls_name} is not a stdlib dataclass; skipped")
            continue
        config_classes.append(cls)
        all_defs.setdefault(
            cls_name,
            {"type": "object", "properties": _reflect_dataclass_fields(cls)},
        )
    wall["schema"] = time.perf_counter() - t0

    # Phase 2: structural invariant reflection.
    t1 = time.perf_counter()
    invariants: list[dict[str, Any]] = []
    invariant_targets: list[type] = [vllm.SamplingParams, *config_classes]
    for cls in invariant_targets:
        for method_name in _enumerate_validator_methods(cls):
            invariants.append(_structural_invariant(cls=cls, method_name=method_name))
    wall["invariants"] = time.perf_counter() - t1

    # Phase 3: write envelopes.
    t2 = time.perf_counter()
    discovered_at = datetime.now(timezone.utc).isoformat()
    schema_envelope: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "discovered_at": discovered_at,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if all_defs:
        schema_envelope["$defs"] = all_defs
    schema_path = out_dir / "schema.json"
    import json

    schema_path.write_text(json.dumps(schema_envelope, indent=2, default=str))

    invariants_envelope: dict[str, Any] = {
        "schema_version": _INVARIANTS_SCHEMA_VERSION,
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "mined_at": discovered_at,
        "invariants": invariants,
    }
    invariants_path = out_dir / "invariants.proposed.yaml"
    invariants_path.write_text(yaml.safe_dump(invariants_envelope, sort_keys=False))
    wall["write"] = time.perf_counter() - t2
    wall["total"] = sum(wall.values())

    return CellOutput(
        strategy_id=_STRATEGY_ID,
        engine=engine,
        engine_version=version,
        schema_path=schema_path,
        invariants_path=invariants_path,
        observations=observations,
        wall_sec=wall,
        extra={
            "engine_param_count": len(engine_params),
            "sampling_param_count": len(sampling_params),
            "defs_count": len(all_defs),
            "invariant_count": len(invariants),
        },
    )
