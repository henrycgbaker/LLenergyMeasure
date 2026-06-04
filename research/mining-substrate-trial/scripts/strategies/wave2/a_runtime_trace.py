"""Wave 2d a-runtime-trace: dynamic invariant discovery via monkey-patched validators.

See WAVE2_PROTOCOL.md section 3 Tier A (row W2-a-runtime-trace). Instead of
statically walking validator method bodies (the W2-a baseline) OR reading
framework reflection (W2-a-pydantic-native), this substrate INSTRUMENTS the
engine at import time by wrapping each validator method with a trace
decorator. Synthetic input is driven through each config class; the wrapper
records ``(args, kwargs)`` and the resulting exception (if any). Captured
(input -> raise) observations are clustered into canonical invariant records
using the same predicate_kind vocabulary the static miner emits.

Hypothesis: dynamic discovery survives upstream refactor better than static
walking - rename ``_verify_args`` to ``_validate`` and the static walker
breaks, but the runtime trace still hits the new method via the dispatch
chain. Recall/precision tradeoff vs the static baseline is the measurement.

Scope: vllm 0.7.3 only. Single-engine pilot per protocol section 3 Tier A.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import yaml

from strategies.wave2._base import CellOutput, standard_out_dir

_STRATEGY_ID = "w2-a-runtime-trace"
_SOURCE_TAG = "wave2_runtime_trace"
_SCHEMA_VERSION = "2.0.0"
_INVARIANTS_SCHEMA_VERSION = "1.0.0"

# Target surface mirrors the static miner's _CLASS_TARGETS registry for
# vllm 0.7.3. SamplingParams + EngineArgs live outside vllm.config and are
# appended at runtime.
_VLLM_CONFIG_CLASS_NAMES: tuple[str, ...] = (
    "ModelConfig",
    "CacheConfig",
    "ParallelConfig",
    "SchedulerConfig",
    "LoRAConfig",
    "PromptAdapterConfig",
    "TokenizerPoolConfig",
    "DecodingConfig",
    "SpeculativeConfig",
)

# Validator-method-name patterns. Same convention as the static miner's
# landmarks: ``_verify_*`` prefix OR exact ``__post_init__``.
_VALIDATOR_METHOD_PREFIXES: tuple[str, ...] = ("_verify_",)
_VALIDATOR_METHOD_EXACT: frozenset[str] = frozenset({"__post_init__"})


@dataclasses.dataclass
class _TraceRecord:
    """One observed (input -> outcome) pair from a wrapped validator call."""

    cls_name: str
    method_name: str
    bound_args: dict[str, Any]
    exc_type: str | None
    exc_message: str | None


def _type_name(field_type: Any) -> str:
    """Coarse type bucket for a stdlib-dataclass field annotation.

    Annotations may be real ``type`` objects, ``typing`` forms, or string
    forward-references (PEP 563). For perturbation seeding a substring
    match on the annotation repr is enough.
    """
    if field_type is None or field_type is type(None):
        return "NoneType"
    raw = getattr(field_type, "__name__", None) or str(field_type)
    lowered = raw.lower().strip("'\"")
    for bucket in ("bool", "int", "float", "str", "list", "tuple", "dict"):
        if bucket in lowered:
            return bucket
    return "any"


def _perturbations_for_type(field_type: Any, default: Any) -> list[Any]:
    """Return 3-5 trial values for a field of the given declared type.

    Buckets:
      - bool: True, False (closed domain).
      - int: default, 0, -1, 1, 2.
      - float: default, 0.0, -1.0, 1.0, 1.5.
      - str: default, "" (empty), "__nope__" (catches enum-shaped str).
      - list / tuple: default, empty container, None.
      - dict: default, empty dict, None.
      - any / unknown: default, None, "__nope__", 0.
    Including the default lets the host short-circuit on known-good input,
    which helps the post-hoc clustering attribute a raise to the right
    perturbed field.
    """
    name = _type_name(field_type)
    if name == "bool":
        return [True, False]
    if name == "int":
        return [default, 0, -1, 1, 2]
    if name == "float":
        return [default, 0.0, -1.0, 1.0, 1.5]
    if name == "str":
        return [default, "", "__nope__"]
    if name in {"list", "tuple"}:
        empty: Any = [] if name == "list" else ()
        return [default, empty, None]
    if name == "dict":
        return [default, {}, None]
    return [default, None, "__nope__", 0]


def _safe_repr(value: Any) -> Any:
    """YAML/JSON-safe shadow of ``value`` for trace records."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_repr(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_repr(v) for k, v in value.items()}
    return repr(value)


def _make_trace_wrapper(
    cls_name: str,
    method_name: str,
    original: Callable[..., Any],
    sink: list[_TraceRecord],
) -> Callable[..., Any]:
    """Wrap ``original`` to record (bound args, exception or None) into sink.

    Exceptions propagate after recording so multi-stage validators
    (__post_init__ -> _verify_args) still fail-fast normally.
    """
    try:
        sig = inspect.signature(original)
    except (TypeError, ValueError):
        sig = None

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound: dict[str, Any] = {}
        if sig is not None:
            try:
                ba = sig.bind(*args, **kwargs)
                ba.apply_defaults()
                bound = {k: _safe_repr(v) for k, v in ba.arguments.items() if k != "self"}
            except TypeError:
                bound = {"_bind_error": True}
        try:
            result = original(*args, **kwargs)
        except Exception as exc:
            sink.append(
                _TraceRecord(
                    cls_name=cls_name,
                    method_name=method_name,
                    bound_args=bound,
                    exc_type=type(exc).__name__,
                    exc_message=str(exc),
                )
            )
            raise
        sink.append(
            _TraceRecord(
                cls_name=cls_name,
                method_name=method_name,
                bound_args=bound,
                exc_type=None,
                exc_message=None,
            )
        )
        return result

    wrapper.__name__ = getattr(original, "__name__", method_name)
    wrapper.__qualname__ = getattr(original, "__qualname__", method_name)
    return wrapper


def _enumerate_validator_methods(cls: type) -> list[str]:
    """Validator method names declared directly on ``cls`` (no MRO walk).

    Only directly-declared methods are patched so restoration via setattr
    is clean and reversible.
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


def _install_patches(
    classes: list[type],
    sink: list[_TraceRecord],
) -> list[tuple[type, str, Callable[..., Any]]]:
    """Install trace wrappers; return originals for restore."""
    originals: list[tuple[type, str, Callable[..., Any]]] = []
    for cls in classes:
        for method_name in _enumerate_validator_methods(cls):
            original = vars(cls).get(method_name)
            if original is None or not callable(original):
                continue
            originals.append((cls, method_name, original))
            setattr(
                cls,
                method_name,
                _make_trace_wrapper(cls.__name__, method_name, original, sink),
            )
    return originals


def _restore_patches(originals: list[tuple[type, str, Callable[..., Any]]]) -> None:
    """Undo every installed patch, in reverse install order."""
    for cls, method_name, original in reversed(originals):
        setattr(cls, method_name, original)


def _baseline_kwargs(cls: type) -> dict[str, Any]:
    """Kwargs dict from a stdlib-dataclass's declared defaults."""
    out: dict[str, Any] = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            out[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                out[f.name] = f.default_factory()  # type: ignore[misc]
            except Exception:
                continue
        else:
            out[f.name] = None
    return out


def _drive_perturbations(cls: type, sink: list[_TraceRecord]) -> tuple[int, int]:
    """For each field, try baseline + perturbations. Returns (tried, raised).

    Each attempt is independent: construct ``cls(**baseline_with_one_field_overridden)``.
    All exceptions are caught - the wrapped validators record into ``sink``
    BEFORE re-raising, so failing constructions are the signal, not noise.
    """
    if not dataclasses.is_dataclass(cls):
        return 0, 0
    baseline = _baseline_kwargs(cls)
    tried = 0
    raised = 0
    tried += 1
    try:
        cls(**baseline)
    except Exception:
        raised += 1
    for f in dataclasses.fields(cls):
        baseline_value = baseline.get(f.name)
        for perturbed in _perturbations_for_type(f.type, baseline_value):
            kwargs = dict(baseline)
            kwargs[f.name] = perturbed
            tried += 1
            try:
                cls(**kwargs)
            except Exception:
                raised += 1
    return tried, raised


def _predicate_from_perturbation(
    field_name: str,
    perturbed_value: Any,
    baseline_value: Any,
) -> tuple[str, Any]:
    """Map a raising perturbation to (predicate_kind, predicate_value).

    Vocabulary mirrors trial_scoring's catalogue:
    ``exact | not_equal | gt | lt | ge | le | not_in | type_is_not | present``.
    Heuristic:
      - perturbed None vs non-None default -> ``present``.
      - non-str handed to str field -> ``type_is_not``.
      - numeric perturbed < default -> ``lt``; > default -> ``gt``.
      - bool flipped -> ``not_equal``.
      - str perturbed (non-empty) -> ``not_in`` over the default value.
      - str perturbed "" -> ``exact`` "" (empty).
      - ambiguous -> ``exact``.
    """
    del field_name  # signature kept for caller readability; not used by heuristic
    if perturbed_value is None and baseline_value is not None:
        return "present", True
    if isinstance(baseline_value, str) and not isinstance(perturbed_value, str):
        return "type_is_not", "str"
    if isinstance(baseline_value, bool):
        return "not_equal", baseline_value
    if isinstance(baseline_value, (int, float)) and isinstance(perturbed_value, (int, float)):
        if isinstance(perturbed_value, bool):
            return "not_equal", baseline_value
        if perturbed_value < baseline_value:
            return "lt", baseline_value
        if perturbed_value > baseline_value:
            return "gt", baseline_value
        return "exact", perturbed_value
    if isinstance(baseline_value, str) and isinstance(perturbed_value, str):
        if perturbed_value == "":
            return "exact", ""
        return "not_in", [baseline_value]
    return "exact", perturbed_value


def _cluster_traces_to_invariants(
    records: list[_TraceRecord],
    *,
    baselines_by_class: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Distil captured traces into canonical-envelope invariants.

    For each raising record: locate the bound arg that differs from the
    class baseline (that's the perturbed field), derive a predicate from
    the diff, deduplicate on (cls, method, field, predicate_kind).
    Records with no identifiable diff are surfaced as ``exception_class``
    invariants so the scorer can choose to ignore them.
    """
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for rec in records:
        if rec.exc_type is None:
            continue
        baseline = baselines_by_class.get(rec.cls_name, {})
        suspect_field: str | None = None
        suspect_value: Any = None
        for fname, fval in rec.bound_args.items():
            if fname in baseline and baseline[fname] != fval:
                suspect_field = fname
                suspect_value = fval
                break
        if suspect_field is None:
            suspect_field = "__unknown__"
            predicate_kind = "exception_class"
            predicate_value: Any = rec.exc_type
        else:
            predicate_kind, predicate_value = _predicate_from_perturbation(
                suspect_field, suspect_value, baseline.get(suspect_field)
            )
        identity = (rec.cls_name, rec.method_name, suspect_field, predicate_kind)
        if identity in seen:
            continue
        seen.add(identity)
        inv_id = f"vllm_{rec.cls_name.lower()}_{suspect_field}_{predicate_kind}".replace("__", "_")
        out.append(
            {
                "id": inv_id,
                "engine": "vllm",
                "library": "vllm",
                "native_type": f"vllm.{rec.cls_name}",
                "native_field": suspect_field,
                "miner_source": {"method": rec.method_name},
                "severity": "error",
                "predicate_kind": predicate_kind,
                "predicate_value": predicate_value,
                "match": {
                    "engine": "vllm",
                    "fields": {f"vllm.{suspect_field}": {predicate_kind: predicate_value}},
                },
                "message_template": rec.exc_message,
                "added_by": _SOURCE_TAG,
            }
        )
    return out


def _structural_schema_for_class(cls: type) -> dict[str, Any]:
    """Minimal ``{type: object, properties: ...}`` fragment for the envelope.

    Intentionally not a re-implementation of schema_introspector; just
    enough envelope shape for the scorer's schema-recall lane.
    """
    props: dict[str, Any] = {}
    if not dataclasses.is_dataclass(cls):
        return {"type": "object", "properties": props}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            default: Any = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = f.default_factory()  # type: ignore[misc]
            except Exception:
                default = None
        else:
            default = None
        props[f.name] = {
            "name": f.name,
            "type": _type_name(f.type),
            "default": default,
            "x-source": _SOURCE_TAG,
        }
    return {"type": "object", "properties": props}


def _write_empty_invariants(out_dir: Any, version: str) -> Any:
    """Drop a valid empty-invariants envelope (used when vllm import fails)."""
    envelope = {
        "schema_version": _INVARIANTS_SCHEMA_VERSION,
        "engine": "vllm",
        "engine_version": version,
        "mined_at": datetime.now(timezone.utc).isoformat(),
        "invariants": [],
    }
    path = out_dir / "invariants.proposed.yaml"
    path.write_text(yaml.safe_dump(envelope, sort_keys=False))
    return path


def run_cell(*, engine: str, version: str) -> CellOutput:
    """Execute the Wave 2d a-runtime-trace cell for vllm 0.7.3."""
    if engine != "vllm" or version not in {"v0.7.3", "0.7.3"}:
        raise NotImplementedError(
            f"a_runtime_trace is vllm 0.7.3-only; got engine={engine!r} version={version!r}"
        )

    version_slug = "v0_7_3"
    out_dir = standard_out_dir(_STRATEGY_ID, engine, version_slug)
    observations: list[str] = []
    wall: dict[str, float] = {}

    # Phase 0: import vllm. Handled as an observation rather than a crash
    # so the runner can record "vllm not installed" as a per-cell outcome.
    t_import = time.perf_counter()
    try:
        import vllm  # type: ignore[import-not-found]
        import vllm.config as vllm_config  # type: ignore[import-not-found]
        from vllm.engine.arg_utils import EngineArgs  # type: ignore[import-not-found]
        from vllm.sampling_params import SamplingParams  # type: ignore[import-not-found]
    except ImportError as exc:
        observations.append(f"vllm import failed: {exc!r}")
        invariants_path = _write_empty_invariants(out_dir, version)
        wall["import"] = time.perf_counter() - t_import
        wall["total"] = wall["import"]
        return CellOutput(
            strategy_id=_STRATEGY_ID,
            engine=engine,
            engine_version=version,
            schema_path=None,
            invariants_path=invariants_path,
            observations=observations,
            wall_sec=wall,
            extra={"perturbations_tried": 0, "perturbations_raised": 0, "invariant_count": 0},
        )

    # Phase 1: resolve target classes.
    target_classes: list[type] = []
    for cls_name in _VLLM_CONFIG_CLASS_NAMES:
        cls = getattr(vllm_config, cls_name, None)
        if cls is None:
            observations.append(f"vllm.config.{cls_name} not found in this version")
            continue
        target_classes.append(cls)
    target_classes.append(SamplingParams)
    target_classes.append(EngineArgs)

    # Phase 2: monkey-patch setup. Originals captured for guaranteed restore.
    t_patch = time.perf_counter()
    sink: list[_TraceRecord] = []
    originals = _install_patches(target_classes, sink)
    wall["monkey_patch_setup"] = time.perf_counter() - t_patch

    perturbations_tried = 0
    perturbations_raised = 0
    baselines_by_class: dict[str, dict[str, Any]] = {}

    try:
        # Phase 3: perturbation generation + execution (folded together).
        t_exec = time.perf_counter()
        for cls in target_classes:
            if not dataclasses.is_dataclass(cls):
                observations.append(f"{cls.__name__} is not a stdlib dataclass; skipped driver")
                continue
            baselines_by_class[cls.__name__] = _baseline_kwargs(cls)
            tried, raised = _drive_perturbations(cls, sink)
            perturbations_tried += tried
            perturbations_raised += raised
        wall["perturbation_generation"] = 0.0
        wall["exec_trace"] = time.perf_counter() - t_exec
    finally:
        # Phase 4: restore patches. ALWAYS runs, even on driver crash.
        _restore_patches(originals)

    # Phase 5: cluster traces and emit envelopes.
    t_emit = time.perf_counter()
    invariants = _cluster_traces_to_invariants(sink, baselines_by_class=baselines_by_class)
    discovered_at = datetime.now(timezone.utc).isoformat()

    schema_defs: dict[str, Any] = {}
    for cls in target_classes:
        if dataclasses.is_dataclass(cls):
            schema_defs[cls.__name__] = _structural_schema_for_class(cls)
    schema_envelope: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "engine": "vllm",
        "engine_version": vllm.__version__,
        "discovered_at": discovered_at,
        "$defs": schema_defs,
    }
    schema_path = out_dir / "schema.json"
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
    wall["emit"] = time.perf_counter() - t_emit
    wall["import"] = t_patch - t_import
    wall["total"] = sum(v for k, v in wall.items() if k != "total")

    observations.append(f"perturbations: tried={perturbations_tried} raised={perturbations_raised}")
    observations.append(f"validator methods wrapped: {len(originals)}")
    observations.append(f"trace records captured: {len(sink)}")

    return CellOutput(
        strategy_id=_STRATEGY_ID,
        engine=engine,
        engine_version=version,
        schema_path=schema_path,
        invariants_path=invariants_path,
        observations=observations,
        wall_sec=wall,
        extra={
            "perturbations_tried": perturbations_tried,
            "perturbations_raised": perturbations_raised,
            "wrapped_method_count": len(originals),
            "trace_record_count": len(sink),
            "invariant_count": len(invariants),
            "target_class_count": len(target_classes),
        },
    )
