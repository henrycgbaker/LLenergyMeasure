"""AST static miner for vLLM 0.19.1.

Vendored machinery: full miner body lives here, version-pinned. The
global ``scripts/engine_producers/vllm_static_miner.py`` is a thin
dispatcher that resolves ``library.current_version`` from the SSOT and
delegates to this module.

vLLM 0.19.1 split the flat 0.7.3 ``vllm/config.py`` into the
``vllm/config/*.py`` per-concern subpackage, so the imperative landmarks
move from ``vllm.config.<Class>`` to ``vllm.config.<module>.<Class>``
(ParallelConfig -> ``config.parallel``, ModelConfig -> ``config.model``,
SchedulerConfig -> ``config.scheduler``, SpeculativeConfig ->
``config.speculative``); ``GuidedDecodingParams`` renamed to
``StructuredOutputsParams`` and ``DecodingConfig`` was removed. Most
per-field constraints migrated from imperative ``_verify_*`` /
``__post_init__`` bodies to declarative pydantic-dataclass ``Field(...)``
bounds, ``Literal[...]`` membership, and validator hooks; CacheConfig and
LoRAConfig no longer carry walkable imperative field-predicates at all
(recovered by the declarative walker in the schema introspector). The
ast_targets registry in ``landmarks.yaml`` reflects the imperative residue
that survives at 0.19.1; the walker / detector / kwargs-synth machinery
below is engine-version-agnostic and byte-identical to the 0.7.3 cut.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import inspect
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.engine_producers._base import (
    COMPARE_OP_NAMES,
    INVERSE_OPS,
    InvariantCandidate,
    MinerLandmarkMissingError,
    MinerSource,
    call_func_path,
    candidate_to_dict,
    find_class,
    find_method,
    first_string_arg,
    literal_value,
    self_attr_name,
)
from scripts.engine_producers._current import current_version
from scripts.engine_producers._landmarks import load_landmarks
from scripts.engine_producers._msgspec_lift import recover_field_types
from scripts.engine_producers._section_classifier import load_curated_sections

# ---------------------------------------------------------------------------
# Engine + namespace conventions
# ---------------------------------------------------------------------------

ENGINE = "vllm"
LIBRARY = "vllm"

# Landmark DATA is externalised to engine_versions/vllm/v<safe>/landmarks.yaml
# and loaded for the pinned library version (with <=target fallback, mirroring
# the producer dispatcher). The walking algorithm below stays version-stable;
# only these data bindings change across versions.
_LANDMARKS = load_landmarks(ENGINE, current_version(ENGINE))


# Drift-tool contract landmarks. Read by ``scripts._drift`` before the
# miner runs; a missing landmark flips the probe verdict to ``fail``. Kept
# as a module attribute because scripts/_drift._read_landmarks reads
# ``module.LANDMARKS``.
LANDMARKS: tuple[str, ...] = _LANDMARKS.probe_landmarks


# ---------------------------------------------------------------------------
# AST landmark registry - what to walk and where
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ASTTarget:
    """One method to AST-walk on a given class/module pair."""

    module_attr: str
    method: str
    namespace: str
    native_type: str

    @property
    def class_name(self) -> str:
        return self.module_attr.rsplit(".", 1)[-1]

    @property
    def module_path(self) -> str:
        return f"vllm.{self.module_attr.rsplit('.', 1)[0]}"


# Built from the externalised ast_targets. The walker iterates this tuple
# exactly as before; reconstructing the per-version sibling's _ASTTarget shape
# keeps the walk byte-identical.
_CLASS_TARGETS: tuple[_ASTTarget, ...] = tuple(
    _ASTTarget(
        module_attr=t.module_attr,
        method=t.method,
        namespace=t.namespace,
        native_type=t.native_type,
    )
    for t in _LANDMARKS.ast_targets
)


def probe_uncovered_validators() -> tuple[list[str], list[str]]:
    """Drift-completeness probe: live raise-bearing validators NOT in ast_targets.

    Imports the live vLLM config tree (in-container) and returns
    (validators_uncovered, validators_unanalyzable). Uncovered = raise-bearing
    validators on flat-hoisted config classes reachable from the seed set that
    the miner's ast_targets covered set misses (the relocation gap). Unanalyzable
    = inherited/wrapped validators that cannot be statically analyzed (surfaced,
    not dropped). Engine-specific: vLLM is import-based, seeded on the ast_targets
    classes + SamplingParams + StructuredOutputsParams; EngineArgs supplies the
    flat-hoist field set (blob-only exclusion). Called by scripts._drift.run().
    """
    import dataclasses

    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import SamplingParams

    from scripts.engine_producers import _completeness

    covered = {(t.class_name, t.method) for t in _CLASS_TARGETS}
    seeds: list[type] = []
    seen_seed: set[type] = set()
    for t in _CLASS_TARGETS:
        module = __import__(t.module_path, fromlist=[t.class_name])
        cls = getattr(module, t.class_name, None)
        if isinstance(cls, type) and cls not in seen_seed:
            seen_seed.add(cls)
            seeds.append(cls)
    for extra in (SamplingParams, EngineArgs):
        if extra not in seen_seed:
            seen_seed.add(extra)
            seeds.append(extra)
    try:
        from vllm.sampling_params import StructuredOutputsParams

        if StructuredOutputsParams not in seen_seed:
            seeds.append(StructuredOutputsParams)
    except Exception:
        pass
    hoist_fields = {f.name for f in dataclasses.fields(EngineArgs)}
    seed_names = {c.__name__ for c in seeds}
    return _completeness.compute_uncovered_validators(
        roots=seeds,
        covered=covered,
        hoist_fields=hoist_fields,
        seed_names=seed_names,
    )


# ---------------------------------------------------------------------------
# Detected pattern + invariant emission
# ---------------------------------------------------------------------------


@dataclass
class _Detected:
    """One detected raise/dormancy site within an ``if`` body."""

    severity: str  # "error" | "warn" | "dormant"
    outcome: str  # "error" | "warn" | "dormant_announced" | "dormant_silent"
    emission_channel: str
    affected_field: str | None
    message_template: str | None
    detail: str
    line: int


def _detect_raise(stmt: ast.stmt) -> _Detected | None:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return None
    if isinstance(stmt.exc, ast.Call):
        func = stmt.exc.func
        if isinstance(func, ast.Name):
            exc_type = func.id
        elif isinstance(func, ast.Attribute):
            exc_type = func.attr
        else:
            exc_type = "Exception"
        msg = first_string_arg(stmt.exc)
        return _Detected(
            severity="error",
            outcome="error",
            emission_channel="none",
            affected_field=None,
            message_template=msg,
            detail=f"raise {exc_type}",
            line=stmt.lineno,
        )
    return None


def _detect_self_assign(stmt: ast.stmt) -> _Detected | None:
    """``self.X = Y`` inside an ``if`` body - silent normalisation."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return None
    rhs = ast.unparse(stmt.value)
    return _Detected(
        severity="dormant",
        outcome="dormant_silent",
        emission_channel="none",
        affected_field=target.attr,
        message_template=None,
        detail=f"self.{target.attr} = {rhs}",
        line=stmt.lineno,
    )


def _detect_logger_warning(stmt: ast.stmt) -> _Detected | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    path = call_func_path(stmt.value)
    if path is None or len(path) != 2 or path[0] != "logger":
        return None
    method = path[-1]
    if method not in {"warning", "warning_once", "error"}:
        return None
    severity = "error" if method == "error" else "warn"
    outcome = "error" if method == "error" else "warn"
    channel = "logger_warning_once" if method == "warning_once" else "logger_warning"
    return _Detected(
        severity=severity,
        outcome=outcome,
        emission_channel=channel,
        affected_field=None,
        message_template=first_string_arg(stmt.value),
        detail=f"logger.{method}",
        line=stmt.lineno,
    )


def _detect_warnings_warn(stmt: ast.stmt) -> _Detected | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    path = call_func_path(stmt.value)
    if path != ["warnings", "warn"]:
        return None
    return _Detected(
        severity="warn",
        outcome="warn",
        emission_channel="warnings_warn",
        affected_field=None,
        message_template=first_string_arg(stmt.value),
        detail="warnings.warn",
        line=stmt.lineno,
    )


_DETECTORS = (
    _detect_raise,
    _detect_logger_warning,
    _detect_warnings_warn,
    _detect_self_assign,
)


def _detect_body_stmts(body: list[ast.stmt]) -> list[_Detected]:
    out: list[_Detected] = []
    for stmt in body:
        for det in _DETECTORS:
            result = det(stmt)
            if result is not None:
                out.append(result)
                break
    return out


# ---------------------------------------------------------------------------
# Predicate translation (AST condition -> match.fields shape)
# ---------------------------------------------------------------------------


@dataclass
class _Predicate:
    """One self-attribute predicate distilled from an ``if`` condition."""

    field: str
    op: str
    rhs: Any  # literal or ``"@<field>"`` cross-field reference
    confidence_penalty: int = 0


_FLIPPED_OPS: dict[str, str] = {
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
    "==": "==",
    "!=": "!=",
}


def _self_method_name(node: ast.expr) -> str | None:
    """Return ``X`` for a ``self.X(...)`` call expression, else None.

    Used to follow a validator that factors a cross-field check into a sibling
    method called from ``__post_init__`` (e.g. vLLM 0.19's
    ``self.verify_max_model_len(...)``), so the relocated raise is still mined.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
    ):
        return node.func.attr
    return None


def _rhs_value(node: ast.expr, field_refs: frozenset[str]) -> tuple[bool, Any]:
    ok, v = literal_value(node)
    if ok:
        return True, v
    name = self_attr_name(node)
    if name is not None:
        return True, f"@{name}"
    if isinstance(node, ast.Name) and node.id in field_refs:
        # A bare reference to a config field / InitVar (e.g. a validator helper
        # comparing self.<field> against the max_model_len InitVar parameter)
        # resolves to the same ``@`` cross-field reference as ``self.<field>``.
        return True, f"@{node.id}"
    return False, None


def _extract_compare(cmp: ast.Compare, field_refs: frozenset[str]) -> list[_Predicate]:
    preds: list[_Predicate] = []
    operands = [cmp.left, *cmp.comparators]
    for left, op, right in zip(operands, cmp.ops, cmp.comparators, strict=False):
        if isinstance(op, (ast.Is, ast.IsNot)):
            field_name = self_attr_name(left)
            ok, rhs = literal_value(right)
            if field_name is not None and ok and rhs is None:
                preds.append(
                    _Predicate(
                        field=field_name,
                        op="absent" if isinstance(op, ast.Is) else "present",
                        rhs=True,
                    )
                )
            continue
        op_name = COMPARE_OP_NAMES.get(type(op))
        if op_name is None:
            continue
        left_field = self_attr_name(left)
        right_field = self_attr_name(right)
        if left_field is not None:
            ok, rhs = _rhs_value(right, field_refs)
            if ok:
                preds.append(_Predicate(field=left_field, op=op_name, rhs=rhs))
        elif right_field is not None:
            flipped = _FLIPPED_OPS.get(op_name)
            ok, rhs = _rhs_value(left, field_refs)
            if flipped is not None and ok:
                preds.append(_Predicate(field=right_field, op=flipped, rhs=rhs))
    return preds


def _extract_call(call: ast.Call) -> list[_Predicate]:
    path = call_func_path(call)
    if path is None:
        return []
    head = path[-1]
    if head == "isinstance" and len(call.args) == 2:
        target = call.args[0]
        type_arg = call.args[1]
        field_name = self_attr_name(target)
        if field_name is None:
            return []
        names: list[str] = []
        if isinstance(type_arg, ast.Name):
            names = [type_arg.id]
        elif isinstance(type_arg, ast.Attribute):
            names = [type_arg.attr]
        elif isinstance(type_arg, (ast.Tuple, ast.List)):
            for elt in type_arg.elts:
                if isinstance(elt, ast.Name):
                    names.append(elt.id)
                elif isinstance(elt, ast.Attribute):
                    names.append(elt.attr)
        if not names:
            return []
        rhs: Any = names[0] if len(names) == 1 else names
        return [_Predicate(field=field_name, op="type_is", rhs=rhs)]
    return []


def _extract_predicates(condition: ast.expr, field_refs: frozenset[str]) -> list[_Predicate]:
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        out: list[_Predicate] = []
        for value in condition.values:
            out.extend(_extract_predicates(value, field_refs))
        return out
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or):
        return []
    if isinstance(condition, ast.Compare):
        return _extract_compare(condition, field_refs)
    if isinstance(condition, ast.Call):
        return _extract_call(condition)
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        inner = _extract_predicates(condition.operand, field_refs)
        if len(inner) == 1 and inner[0].op in INVERSE_OPS:
            p = inner[0]
            return [_Predicate(field=p.field, op=INVERSE_OPS[p.op], rhs=p.rhs)]
        return []
    field_name = self_attr_name(condition)
    if field_name is not None:
        return [_Predicate(field=field_name, op="present", rhs=True, confidence_penalty=1)]
    return []


# ---------------------------------------------------------------------------
# kwargs synthesis (positive = trips the invariant; negative = doesn't)
# ---------------------------------------------------------------------------


def _value_satisfying(op: str, rhs: Any) -> Any:
    if op == "present" and rhs is True:
        return 1
    if op == "absent":
        return None
    if op == "==":
        return rhs
    if op == "!=":
        if isinstance(rhs, bool):
            return not rhs
        if isinstance(rhs, (int, float)):
            return rhs + 1
        if isinstance(rhs, str):
            return rhs + "_x"
        return None
    if op == "<":
        if isinstance(rhs, bool):
            return False
        if isinstance(rhs, int):
            return rhs - 1
        if isinstance(rhs, float):
            return rhs - 1.0
        return rhs
    if op == "<=":
        return rhs
    if op == ">":
        if isinstance(rhs, bool):
            return True
        if isinstance(rhs, int):
            return rhs + 1
        if isinstance(rhs, float):
            return rhs + 1.0
        return rhs
    if op == ">=":
        return rhs
    if op == "in":
        if isinstance(rhs, (list, tuple, set)) and rhs:
            return next(iter(rhs))
        return rhs
    if op == "not_in":
        return "__vllm_static_synth__"
    if op == "type_is":
        return _type_label_default(rhs)
    if op == "type_is_not":
        return _other_type_default(rhs)
    return rhs


def _type_label_default(label: Any) -> Any:
    label_str = label if isinstance(label, str) else (label[0] if label else "str")
    return {
        "bool": True,
        "int": 1,
        "float": 1.0,
        "str": "x",
        "list": [],
        "dict": {},
        "tuple": (),
    }.get(label_str)


def _other_type_default(label: Any) -> Any:
    label_str = label if isinstance(label, str) else (label[0] if label else "str")
    if label_str == "str":
        return 1
    if label_str in {"int", "float"}:
        return "x"
    if label_str == "bool":
        return 1
    return "x"


def _minimal_value_for_type(type_str: str) -> tuple[bool, Any]:
    """Map a recovered field-type string to a minimal VALID non-None value.

    Returns ``(known, value)``. The value only needs to be valid-typed and
    non-None so that the W1.1 ``msgspec.convert`` gate constructs it cleanly
    and the dormancy correctly does NOT fire on the negative probe. A union
    ``X | None`` collapses to its non-None arm; genuinely-opaque types yield
    ``(False, None)`` so the caller keeps the generic probe value.
    """
    t = type_str.strip()
    if "|" in t:
        for arm in (a.strip() for a in t.split("|")):
            if arm and arm != "None":
                known, value = _minimal_value_for_type(arm)
                if known:
                    return True, value
        return False, None
    if t.startswith("list["):
        known, elem = _minimal_value_for_type(t[len("list[") : -1])
        return True, [elem if known else "x"]
    if t.startswith("dict["):
        inner = t[len("dict[") : -1]
        key_str, _, val_str = inner.partition(",")
        k_known, key = _minimal_value_for_type(key_str)
        v_known, val = _minimal_value_for_type(val_str)
        return True, {(key if k_known else "x"): (val if v_known else "x")}
    if t.startswith("Literal["):
        members = [m.strip() for m in t[len("Literal[") : -1].split(",") if m.strip()]
        if members:
            try:
                return True, ast.literal_eval(members[0])
            except (ValueError, SyntaxError):
                return False, None
        return False, None
    scalar = {"bool": True, "int": 1, "float": 1.0, "str": "x", "bytes": b"x"}.get(t)
    if scalar is not None:
        return True, scalar
    return False, None


def _typed_present_value(target: _ASTTarget, field_name: str) -> tuple[bool, Any]:
    """Resolve a type-valid non-None value for a dormant field's negative probe.

    Recovers ``{field: type_str}`` from the target's msgspec Struct (W1.2's
    ``recover_field_types``) and maps the dormant field's type to a minimal
    valid value. Returns ``(False, None)`` for non-Struct targets, unknown
    fields, or opaque types so the caller falls back to the generic probe.
    """
    try:
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name)
    except (ImportError, AttributeError):
        return False, None
    type_str = recover_field_types(cls).get(field_name)
    if type_str is None:
        return False, None
    return _minimal_value_for_type(type_str)


def _synthesise_kwargs(preds: list[_Predicate]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in preds:
        if isinstance(p.rhs, str) and p.rhs.startswith("@"):
            companion = p.rhs[1:].split(".")[-1]
            out.setdefault(companion, 2)
            out.setdefault(p.field, _value_satisfying(p.op, out[companion]))
        else:
            out.setdefault(p.field, _value_satisfying(p.op, p.rhs))
    return out


def _negate_predicates(preds: list[_Predicate]) -> list[_Predicate]:
    if not preds:
        return []
    last = preds[-1]
    flipped_op = INVERSE_OPS.get(last.op, last.op)
    return [
        *preds[:-1],
        _Predicate(field=last.field, op=flipped_op, rhs=last.rhs),
    ]


# ---------------------------------------------------------------------------
# match_fields construction
# ---------------------------------------------------------------------------


def _build_match_fields(preds: list[_Predicate], namespace: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    for p in preds:
        path = f"{namespace}.{p.field}"
        spec = grouped.setdefault(path, {})
        spec[p.op] = p.rhs
    out: dict[str, Any] = {}
    for path, spec in grouped.items():
        if len(spec) == 1 and "==" in spec:
            out[path] = spec["=="]
        else:
            out[path] = spec
    return out


# ---------------------------------------------------------------------------
# Miner proper
# ---------------------------------------------------------------------------


@dataclass
class _Frame:
    predicates: list[_Predicate] = field(default_factory=list)


def _walk_function(
    func: ast.FunctionDef,
    *,
    target: _ASTTarget,
    field_names: frozenset[str],
    class_methods: dict[str, ast.FunctionDef],
    target_methods: frozenset[str],
    rel_source_path: str,
    today: str,
) -> list[InvariantCandidate]:
    invariants: list[InvariantCandidate] = []
    seen_ids: set[str] = set()
    visited_helpers: set[str] = set()

    def descend(body: list[ast.stmt], frame: _Frame) -> None:
        for stmt in body:
            if isinstance(stmt, ast.If):
                _handle_if(stmt, frame)
            elif isinstance(stmt, ast.For):
                descend(stmt.body, frame)
            elif isinstance(stmt, ast.Expr):
                # Follow a same-class validator helper called from this method
                # (e.g. self.verify_max_model_len(...)) so a raise that upstream
                # relocated out of the walked method is still mined. One level
                # per helper (visited guard) to avoid cycles. Skip a helper that
                # is ITSELF a direct ast_target on this class - it is walked
                # directly, so following it here would discover its rules twice
                # and shift their provenance onto the calling method.
                helper = _self_method_name(stmt.value)
                if (
                    helper is not None
                    and helper in class_methods
                    and helper not in target_methods
                    and helper not in visited_helpers
                ):
                    visited_helpers.add(helper)
                    descend(class_methods[helper].body, frame)

    def _handle_if(if_node: ast.If, frame: _Frame) -> None:
        own_preds = _extract_predicates(if_node.test, field_names)
        if field_names and not any(p.field in field_names for p in own_preds):
            descend(if_node.body, frame)
            return
        local = _Frame(predicates=[*frame.predicates, *own_preds])
        for det in _detect_body_stmts(if_node.body):
            invariant = _build_rule(
                target=target,
                preds=local.predicates,
                detected=det,
                field_names=field_names,
                rel_source_path=rel_source_path,
                today=today,
            )
            if invariant is None:
                continue
            if invariant.id in seen_ids:
                suffix = 2
                while f"{invariant.id}__{suffix}" in seen_ids:
                    suffix += 1
                invariant.id = f"{invariant.id}__{suffix}"
            seen_ids.add(invariant.id)
            invariants.append(invariant)
        descend(if_node.body, local)
        for sub in if_node.orelse:
            if isinstance(sub, ast.If):
                _handle_if(sub, frame)
            elif isinstance(sub, (ast.Assign, ast.Raise, ast.Expr)):
                continue

    descend(func.body, _Frame())
    return invariants


def _build_rule(
    *,
    target: _ASTTarget,
    preds: list[_Predicate],
    detected: _Detected,
    field_names: frozenset[str],
    rel_source_path: str,
    today: str,
) -> InvariantCandidate | None:
    effective_preds = list(preds)
    subject_field = detected.affected_field
    if subject_field is not None and not any(p.field == subject_field for p in effective_preds):
        effective_preds.append(_Predicate(field=subject_field, op="present", rhs=True))

    if not effective_preds:
        return None

    # A predicate keyed on a name absent from the class's settable surface
    # addresses a computed ``@property`` (e.g. ``ParallelConfig.world_size_across_dp``,
    # which is TPxPPxDP), never a user-set knob. Its config-time match path cannot
    # resolve against the generated Config and a probe cannot set it, so the
    # candidate is not a faithful config-time invariant - drop it here, where the
    # miner knows the settable surface. ``assert_path_resolves`` stays fail-loud
    # for a *settable* field that unexpectedly fails to resolve (genuine drift).
    # ``field_names`` is empty only when introspection failed, in which case the
    # filter degrades to no-op (recall over precision, as the rest of the miner).
    if field_names and any(p.field not in field_names for p in effective_preds):
        return None

    match_fields = _build_match_fields(effective_preds, target.namespace)
    if not match_fields:
        return None

    # Internals guard: an invariant whose every match field is a private
    # (underscore-prefixed) attribute addresses internal library state the public
    # config surface never exposes (e.g. ParallelConfig._api_process_rank, set
    # internally, never by a user). Such a rule is empirically real but can never
    # fire against a user config, so it is inert clutter. The comprehensive-mining
    # intent records every validated invariant EXCEPT these internals.
    if all(path.rsplit(".", 1)[-1].startswith("_") for path in match_fields):
        return None

    kwargs_pos = _synthesise_kwargs(effective_preds)
    kwargs_neg = _synthesise_kwargs(_negate_predicates(effective_preds))
    if kwargs_pos == kwargs_neg:
        kwargs_neg = _force_distinct(kwargs_pos, effective_preds)

    # Type-aware dormancy negative probe. A dormant-when-absent rule synthesises
    # its negative as the field "present" with a generic non-None ``1``; under
    # the W1.1 msgspec.convert gate that ``1`` fails the field's declared type
    # (e.g. bad_words: list[str] | None) and silently quarantines a real rule.
    # Pick a valid-typed non-None value so the negative constructs cleanly and
    # the dormancy correctly does NOT fire.
    last_neg = _negate_predicates(effective_preds)[-1]
    if detected.outcome.startswith("dormant") and last_neg.op == "present":
        typed_known, typed_val = _typed_present_value(target, last_neg.field)
        if typed_known:
            kwargs_neg = {**kwargs_neg, last_neg.field: typed_val}

    last_pred = effective_preds[-1]
    known, default = _field_default_from_target(target, last_pred.field)
    if known and default != kwargs_pos.get(last_pred.field):
        kwargs_neg = {**kwargs_neg, last_pred.field: default}

    invariant_id = _make_invariant_id(target=target, preds=effective_preds, detected=detected)

    invariant_under_test = _describe_rule(target=target, preds=effective_preds, detected=detected)

    return InvariantCandidate(
        id=invariant_id,
        engine=ENGINE,
        library=LIBRARY,
        invariant_under_test=invariant_under_test,
        severity=detected.severity,  # type: ignore[arg-type]
        native_type=target.native_type,
        miner_source=MinerSource(
            path=rel_source_path,
            method=target.method,
            line_at_scan=detected.line,
        ),
        match_fields=match_fields,
        kwargs_positive=kwargs_pos,
        kwargs_negative=kwargs_neg,
        expected_outcome={
            "outcome": detected.outcome,
            "emission_channel": detected.emission_channel,
            "normalised_fields": (
                [subject_field]
                if subject_field is not None and detected.outcome.startswith("dormant")
                else []
            ),
        },
        message_template=detected.message_template,
        references=[f"{rel_source_path}:{detected.line} ({target.native_type}.{target.method})"],
        added_by="static_miner",
        added_at=today,
    )


def _force_distinct(pos: dict[str, Any], preds: list[_Predicate]) -> dict[str, Any]:
    if not preds:
        return pos
    last = preds[-1]
    out = dict(pos)
    cur = out.get(last.field)
    if isinstance(cur, bool):
        out[last.field] = not cur
    elif isinstance(cur, (int, float)):
        out[last.field] = cur + 1
    elif isinstance(cur, str):
        out[last.field] = cur + "_neg"
    elif cur is None:
        out[last.field] = 0
    else:
        out[last.field] = None
    return out


_OP_WORDS = {
    "==": "eq",
    "!=": "ne",
    "<": "lt",
    "<=": "le",
    ">": "gt",
    ">=": "ge",
    "in": "in",
    "not_in": "not_in",
    "present": "set",
    "absent": "unset",
    "type_is": "type",
    "type_is_not": "not_type",
}


def _slug(value: Any) -> str:
    s = str(value).replace(".", "p").replace("-", "neg").replace(" ", "_")
    return "".join(ch for ch in s if ch.isalnum() or ch == "_").lower()[:24]


def _make_invariant_id(
    *,
    target: _ASTTarget,
    preds: list[_Predicate],
    detected: _Detected,
) -> str:
    severity_tag = {"error": "raises", "warn": "warns", "dormant": "dormant"}.get(
        detected.severity, "invariant"
    )
    short = target.class_name.lower()
    parts: list[str] = ["vllm", short, severity_tag]
    last = preds[-1]
    parts.append(last.field)
    parts.append(_OP_WORDS.get(last.op, last.op))
    if isinstance(last.rhs, str) and last.rhs.startswith("@"):
        parts.append("ref_" + last.rhs[1:])
    elif isinstance(last.rhs, (int, float, bool, str)):
        parts.append(_slug(last.rhs))
    rid = "_".join(p for p in parts if p)
    while "__" in rid:
        rid = rid.replace("__", "_")
    return rid.strip("_")


def _describe_rule(*, target: _ASTTarget, preds: list[_Predicate], detected: _Detected) -> str:
    pred_str = " AND ".join(f"{p.field} {p.op} {p.rhs}" for p in preds)
    sev_word = {"error": "raises", "warn": "warns", "dormant": "marks dormant"}.get(
        detected.severity, "fires"
    )
    return f"{target.class_name}.{target.method}: {sev_word} when {pred_str}"


# ---------------------------------------------------------------------------
# Public-field discovery for filtering
# ---------------------------------------------------------------------------


def _initvar_names(cls: type) -> set[str]:
    """Names declared as ``InitVar`` on the class (or its MRO).

    InitVars are passed to ``__post_init__`` but not stored as fields, yet a
    validator compares config knobs against them (e.g. SchedulerConfig's
    ``max_model_len: InitVar[int]``). Detected from annotations whether or not
    the defining module uses ``from __future__ import annotations`` (which
    leaves the annotation a string).
    """
    import dataclasses as _dc

    out: set[str] = set()
    for klass in getattr(cls, "__mro__", [cls]):
        for name, ann in getattr(klass, "__annotations__", {}).items():
            if name.startswith("_"):
                continue
            if (
                isinstance(ann, _dc.InitVar)
                or ann is _dc.InitVar
                or (isinstance(ann, str) and "InitVar" in ann)
            ):
                out.add(name)
    return out


def _public_field_names_from_target(target: _ASTTarget) -> frozenset[str]:
    """Return the set of public field names of the target class.

    vLLM mixes stdlib dataclasses (``ParallelConfig``, ``LoRAConfig``,
    ``SchedulerConfig``, ``DecodingConfig``) with bare classes (``CacheConfig``,
    ``ModelConfig``, ``SpeculativeConfig``). The dataclass / msgspec / pydantic
    ladder below catches the dataclasses; the bare-class fallback constructs an
    instance with no args to enumerate ``vars()``. InitVar-declared names are
    folded in too: they are not stored fields but they ARE config knobs a
    validator compares against, so a bare reference to one resolves to an ``@``
    field-ref. On any failure return an empty set so the public-field filter
    degrades to "no filter" (recall over precision).
    """
    try:
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name)
    except (ImportError, AttributeError):
        return frozenset()

    import dataclasses as _dc

    names: set[str] = set()
    pyd_fields = getattr(cls, "__pydantic_fields__", None)
    struct_fields = getattr(cls, "__struct_fields__", None)
    if pyd_fields:
        names.update(pyd_fields.keys())
    elif struct_fields:
        names.update(n for n in struct_fields if not n.startswith("_"))
    elif _dc.is_dataclass(cls):
        names.update(f.name for f in _dc.fields(cls) if not f.name.startswith("_"))
    else:
        try:
            instance = cls()
        except Exception:
            return frozenset()
        names.update(n for n in vars(instance) if not n.startswith("_"))
    names.update(_initvar_names(cls))
    return frozenset(names)


def _field_default_from_target(target: _ASTTarget, field_name: str) -> tuple[bool, Any]:
    """Return ``(known, default_value)`` from runtime introspection.

    At 0.19.1 the per-concern configs are pydantic-dataclasses, so the
    pydantic branch is the primary default source; the stdlib-dataclass
    branch picks up ``ParallelConfig`` / ``SchedulerConfig`` defaults when
    the last predicate's field is dataclass-bound. Falls open (known=False)
    only when neither introspection path resolves the field.
    """
    try:
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name)
    except (ImportError, AttributeError):
        return False, None
    pyd_fields = getattr(cls, "__pydantic_fields__", None)
    if pyd_fields and field_name in pyd_fields:
        info = pyd_fields[field_name]
        default = getattr(info, "default", None)
        if default is not None and type(default).__name__ == "PydanticUndefinedType":
            return False, None
        return True, default
    import dataclasses as _dc

    if _dc.is_dataclass(cls):
        for f in _dc.fields(cls):
            if f.name == field_name:
                if f.default is not _dc.MISSING:
                    return True, f.default
                if f.default_factory is not _dc.MISSING:  # type: ignore[misc]
                    try:
                        return True, f.default_factory()
                    except Exception:
                        return False, None
                return False, None
    return False, None


# ---------------------------------------------------------------------------
# Source location + landmark verification
# ---------------------------------------------------------------------------


def _site_packages_relative(abs_path: str) -> str:
    marker = "site-packages/"
    idx = abs_path.find(marker)
    if idx >= 0:
        return abs_path[idx + len(marker) :]
    return Path(abs_path).name


def _check_landmarks() -> tuple[str, dict[str, str]]:
    try:
        import vllm  # type: ignore
    except ImportError as exc:
        raise MinerLandmarkMissingError("vllm.__init__", detail="vllm not importable") from exc

    paths: dict[str, str] = {}
    for target in _CLASS_TARGETS:
        try:
            module = __import__(target.module_path, fromlist=[target.class_name])
        except ImportError as exc:
            raise MinerLandmarkMissingError(
                target.module_path, detail=f"module not importable: {exc}"
            ) from exc
        cls = getattr(module, target.class_name, None)
        if cls is None:
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}",
                detail="class symbol missing",
            )
        if not hasattr(cls, target.method):
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}.{target.method}",
                detail="method missing on class",
            )
        abs_path = inspect.getsourcefile(module)
        if abs_path is None:
            raise MinerLandmarkMissingError(target.module_path, detail="source file unavailable")
        paths[target.module_path] = abs_path
        module_ast = ast.parse(Path(abs_path).read_text())
        cls_ast = find_class(module_ast, target.class_name)
        if cls_ast is None:
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}",
                detail="ClassDef not in AST",
            )
        method_ast = find_method(cls_ast, target.method)
        if method_ast is None:
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}.{target.method}",
                detail="FunctionDef not in AST",
            )

    return vllm.__version__, paths


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def walk_vllm_static(*, today: str | None = None) -> tuple[list[InvariantCandidate], str]:
    installed_version, abs_paths = _check_landmarks()

    if today is None:
        frozen = os.environ.get("LLENERGY_MINER_FROZEN_AT")
        today = frozen if frozen else dt.date.today().isoformat()

    candidates: list[InvariantCandidate] = []
    ast_cache: dict[str, ast.Module] = {}
    for target in _CLASS_TARGETS:
        abs_path = abs_paths[target.module_path]
        rel_path = _site_packages_relative(abs_path)
        if abs_path not in ast_cache:
            ast_cache[abs_path] = ast.parse(Path(abs_path).read_text())
        module_ast = ast_cache[abs_path]
        cls_ast = find_class(module_ast, target.class_name)
        if cls_ast is None:
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}", detail="vanished mid-walk"
            )
        method_ast = find_method(cls_ast, target.method)
        if method_ast is None:
            raise MinerLandmarkMissingError(
                f"{target.module_path}.{target.class_name}.{target.method}",
                detail="vanished mid-walk",
            )
        field_names = _public_field_names_from_target(target)
        class_methods = {
            node.name: node for node in cls_ast.body if isinstance(node, ast.FunctionDef)
        }
        # Methods of THIS class that are themselves direct ast_targets: a helper
        # call into one of these is skipped during call-following because it is
        # walked directly (avoids duplicate-discovery provenance churn).
        target_methods = frozenset(
            t.method for t in _CLASS_TARGETS if t.class_name == target.class_name
        )
        candidates.extend(
            _walk_function(
                method_ast,
                target=target,
                field_names=field_names,
                class_methods=class_methods,
                target_methods=target_methods,
                rel_source_path=rel_path,
                today=today,
            )
        )

    return candidates, installed_version


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def emit_yaml(candidates: list[InvariantCandidate], engine_version: str) -> str:
    import yaml

    sorted_candidates = sorted(candidates, key=lambda c: (c.miner_source.method, c.id))
    frozen = os.environ.get("LLENERGY_MINER_FROZEN_AT")
    mined_at = frozen if frozen else dt.date.today().isoformat()
    curated_sections = load_curated_sections(ENGINE)
    rows = [
        d
        for c in sorted_candidates
        if (d := candidate_to_dict(c, ENGINE, curated_sections)) is not None
    ]
    skipped = len(sorted_candidates) - len(rows)
    if skipped:
        print(
            f"[{ENGINE}] skipped {skipped} candidate(s) with unreachable match paths",
            file=sys.stderr,
        )
    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": engine_version,
        "mined_at": mined_at,
        "invariants": rows,
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/llenergymeasure/engines/vllm/_staging/vllm_static_miner.yaml"),
        help="Where to write the staging YAML.",
    )
    args = parser.parse_args(argv)

    candidates, version = walk_vllm_static()
    text = emit_yaml(candidates, version)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} vLLM static-miner invariant candidates to {args.out}",
        file=sys.stderr,
    )
    return 0
