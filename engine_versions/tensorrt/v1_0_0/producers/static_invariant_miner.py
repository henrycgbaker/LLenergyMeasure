"""TensorRT-LLM 1.0.0 static-miner machinery.

Vendored per-version snapshot of the source-driven AST extractor for
``tensorrt_llm`` 1.0.0 (the 0.x -> 1.x MAJOR). Walks the extracted source
tarball at ``/tmp/trt-llm-1.0.0/`` and emits validation-invariant
candidates for the AST validators on :class:`BaseLlmArgs`,
:class:`TrtLlmArgs`, :class:`TorchLlmArgs`, :class:`CudaGraphConfig`,
:class:`TorchCompileConfig`, and :class:`LookaheadDecodingConfig`, plus
``Literal[...]``-typed field allowlists and ``StrEnum``-typed field
allowlists.

What the 0.x -> 1.x major moved (measured 2026-06-14; leg findings at
``.product/research/tensorrt-leg-findings-2026-06-14.md``)
-------------------------------------------------------------------------
The C++-to-pydantic migration the study predicted DID land, but the new
Python-visible constraints arrived as IMPERATIVE ``@field_validator`` /
``@model_validator`` bodies (``if v < 0: raise``), not as DECLARATIVE
``Field(ge=...)`` bounds. Recall therefore rises through the imperative
floor walker, not the declarative walker: the new mineable validators
land on the pytorch-backend config surface (``CudaGraphConfig``,
``TorchCompileConfig``, ``TorchLlmArgs.stream_interval``). All ten 0.21.0
validator-method landmarks survive verbatim; this cut ADDS the migrated
ones to the landmark set rather than retargeting.

Why source-driven, not import-driven
------------------------------------
The TRT-LLM library at 1.0.0 (CUDA 13.x) cannot be imported on the host
(no CUDA). The miner therefore reads the extracted source tree and never
imports the installed library.

Schema lift via AST
-------------------
``_pydantic_lift`` is the standard sub-library lift for Pydantic v2 models,
but it requires a live class import (``model_fields`` is a runtime attribute).
For TRT-LLM we cannot import the library on host, so this miner extracts
``Literal[...]`` allowlists on Pydantic ``Field``-defined attributes from
class-body AST. The shape it emits is byte-identical to
``_pydantic_lift``'s output for the equivalent fields.

No dynamic miner
----------------
TRT-LLM's ``TrtLlmArgs(...)`` constructor is permissive at construction.
All cross-field invariants fire at engine build inside C++, out of reach
for Python-side construction probing (adversarial review decision #8).

LANDMARKS for the static miner reference the live ``tensorrt_llm`` package
because that is the seam Renovate's library bumps shift first. A missing
landmark under the installed library trips the probe; a divergent source
tree trips ``MinerLandmarkMissingError`` inside the miner.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scripts.engine_producers._base import (
    InvariantCandidate,
    MinerLandmarkMissingError,
    MinerSource,
    call_func_path,
    find_class,
    find_method,
    first_string_arg,
)
from scripts.engine_producers._current import current_version
from scripts.engine_producers._landmarks import load_landmarks
from scripts.engine_producers._section_classifier import (
    load_curated_sections,
    relabel_match_fields,
)
from scripts.engine_producers._source_walker import _literal_values

# ---------------------------------------------------------------------------
# Version-pinned configuration (1.0.0)
# ---------------------------------------------------------------------------

ENGINE = "tensorrt"
LIBRARY = "tensorrt_llm"
NAMESPACE = "tensorrt"

# Landmark DATA is externalised to engine_versions/tensorrt/v<safe>/landmarks.yaml
# and loaded for the pinned library version (with <=target fallback, mirroring
# the producer dispatcher). The walking algorithm below stays version-stable;
# only these data bindings change across versions.
_LANDMARKS = load_landmarks(ENGINE, current_version(ENGINE))

# Default source root. Overridable via ``--source-root`` for tests / CI.
_DEFAULT_SOURCE_ROOT = _LANDMARKS.source_root

# Files we AST-walk.
LLM_ARGS_REL = _LANDMARKS.llm_args_rel
BUILDER_REL = _LANDMARKS.builder_rel

# Probe LANDMARKS: dotted attribute paths the probe resolves under
# ``import tensorrt_llm``. Mirrors the validator surface this miner expects to
# find when the installed library matches the pin. Kept as a module attribute
# because scripts/_drift._read_landmarks reads ``module.LANDMARKS``.
LANDMARKS: tuple[str, ...] = _LANDMARKS.probe_landmarks

# Class-level AST targets the miner expects to find inside source.
# Each entry is ``(class_name, file_relative_path)``.
_CLASS_TARGETS: tuple[tuple[str, Path], ...] = _LANDMARKS.class_targets

# Method-level landmarks: ``(class_name, method_name)``. The methods whose
# bodies are walked for invariant extraction.
_METHOD_LANDMARKS: tuple[tuple[str, str], ...] = _LANDMARKS.method_landmarks

# StrEnum classes whose members are the allowlist for a particular
# TrtLlmArgs field. Mapping is ``(enum class name, top-level field name)``.
_STRENUM_FIELDS: tuple[tuple[str, str], ...] = _LANDMARKS.strenum_fields

# Pydantic classes whose ``Literal[...]``-typed fields are lifted as
# allowlist invariants via class-body AST. Intentional in-code per-version
# constant (deliberately NOT externalised to landmarks.yaml): review it on each
# tensorrt bump in case the lift surface gains or loses a config class.
_LITERAL_LIFT_CLASSES: tuple[str, ...] = ("BaseLlmArgs", "TrtLlmArgs", "CalibConfig")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class _Predicate:
    """Single ``self.<field>`` op rhs predicate extracted from a condition."""

    field: str
    op: str
    rhs: Any


@dataclass
class _DetectedBody:
    """The interesting statement inside an ``if`` body."""

    severity: str  # "error" | "warn"
    outcome: str  # "error" | "warn"
    emission_channel: str
    message_template: str | None
    detail: str


@dataclass
class _Frame:
    """Conditions accumulated from enclosing ``if`` statements."""

    predicates: list[_Predicate] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Predicate extraction
# ---------------------------------------------------------------------------


_COMPARE_OP_NAMES: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not_in",
}

# Ops that have a clean inverse for kwargs_negative synthesis.
_INVERSION_MAP: dict[str, str] = {
    "==": "!=",
    "!=": "==",
    "<": ">=",
    "<=": ">",
    ">": "<=",
    ">=": "<",
    "in": "not_in",
    "not_in": "in",
    "present": "absent",
    "absent": "present",
    "type_is": "type_is_not",
    "type_is_not": "type_is",
}


def _self_attr(node: ast.expr) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


def _literal_value(node: ast.expr) -> tuple[bool, Any]:
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ok, v = _literal_value(node.operand)
        if ok and isinstance(v, (int, float)):
            return True, -v
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: list[Any] = []
        for elt in node.elts:
            ok, v = _literal_value(elt)
            if not ok:
                return False, None
            out.append(v)
        return True, list(out)
    return False, None


def _isinstance_type_names(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Name):
                names.append(elt.id)
            elif isinstance(elt, ast.Attribute):
                names.append(elt.attr)
            else:
                return []
        return names
    return []


def _extract_predicates(condition: ast.expr) -> list[_Predicate]:
    """Translate a condition AST into a list of ``self.<field>`` predicates.

    Only AND-combined predicates are extracted into the invariant's match.fields;
    OR / opaque calls / non-self conditions are silently dropped (recall-first
    - the invariant still emits, just without those preconditions).
    """
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        out: list[_Predicate] = []
        for v in condition.values:
            out.extend(_extract_predicates(v))
        return out
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        # ``not isinstance(self.x, T)`` -> type_is_not predicate.
        if isinstance(condition.operand, ast.Call):
            preds = _extract_call_predicate(condition.operand)
            inverted: list[_Predicate] = []
            for p in preds:
                if p.op in _INVERSION_MAP:
                    inverted.append(_Predicate(field=p.field, op=_INVERSION_MAP[p.op], rhs=p.rhs))
            return inverted
        # ``not self.x`` -> absent.
        attr = _self_attr(condition.operand)
        if attr is not None:
            return [_Predicate(field=attr, op="absent", rhs=True)]
        return []
    if isinstance(condition, ast.Compare):
        return _extract_compare(condition)
    if isinstance(condition, ast.Call):
        return _extract_call_predicate(condition)
    # Bare ``self.x`` -> truthiness == "present".
    attr = _self_attr(condition)
    if attr is not None:
        return [_Predicate(field=attr, op="present", rhs=True)]
    return []


def _extract_compare(cmp: ast.Compare) -> list[_Predicate]:
    out: list[_Predicate] = []
    operands = [cmp.left, *cmp.comparators]
    for left, op, right in zip(operands, cmp.ops, cmp.comparators, strict=False):
        # ``is`` / ``is not`` -> absent / present (only the ``None`` rhs case).
        if isinstance(op, (ast.Is, ast.IsNot)):
            attr = _self_attr(left)
            ok, rhs = _literal_value(right)
            if attr is not None and ok and rhs is None:
                out.append(
                    _Predicate(
                        field=attr,
                        op="absent" if isinstance(op, ast.Is) else "present",
                        rhs=True,
                    )
                )
            continue
        op_name = _COMPARE_OP_NAMES.get(type(op))
        if op_name is None:
            continue
        left_field = _self_attr(left)
        right_field = _self_attr(right)
        if left_field is not None:
            ok, rhs = _literal_value(right)
            if ok:
                out.append(_Predicate(field=left_field, op=op_name, rhs=rhs))
            elif right_field is not None:
                # Cross-field compare: ``self.a OP self.b`` -> ``a OP @b``.
                out.append(_Predicate(field=left_field, op=op_name, rhs=f"@{right_field}"))
        elif right_field is not None:
            flipped = {
                "<": ">",
                "<=": ">=",
                ">": "<",
                ">=": "<=",
                "==": "==",
                "!=": "!=",
            }.get(op_name)
            ok, rhs = _literal_value(left)
            if flipped is not None and ok:
                out.append(_Predicate(field=right_field, op=flipped, rhs=rhs))
    return out


def _extract_call_predicate(call: ast.Call) -> list[_Predicate]:
    path = call_func_path(call)
    if path is None:
        return []
    head = path[-1]
    if head == "isinstance" and len(call.args) == 2:
        attr = _self_attr(call.args[0])
        if attr is None:
            return []
        names = _isinstance_type_names(call.args[1])
        if not names:
            return []
        rhs: Any = names[0] if len(names) == 1 else names
        return [_Predicate(field=attr, op="type_is", rhs=rhs)]
    return []


# ---------------------------------------------------------------------------
# Body detectors
# ---------------------------------------------------------------------------


def _detect_raise(stmt: ast.stmt) -> _DetectedBody | None:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return None
    exc_type = "Exception"
    msg: str | None = None
    if isinstance(stmt.exc, ast.Call):
        if isinstance(stmt.exc.func, ast.Name):
            exc_type = stmt.exc.func.id
        msg = first_string_arg(stmt.exc)
    return _DetectedBody(
        severity="error",
        outcome="error",
        emission_channel="none",
        message_template=msg,
        detail=f"raise {exc_type}",
    )


def _detect_logger_warning(stmt: ast.stmt) -> _DetectedBody | None:
    if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
        return None
    path = call_func_path(stmt.value)
    if path is None or len(path) != 2 or path[0] != "logger":
        return None
    method = path[-1]
    if method == "warning":
        return _DetectedBody(
            severity="warn",
            outcome="warn",
            emission_channel="logger_warning",
            message_template=first_string_arg(stmt.value),
            detail="logger.warning",
        )
    if method == "warning_once":
        return _DetectedBody(
            severity="warn",
            outcome="warn",
            emission_channel="logger_warning_once",
            message_template=first_string_arg(stmt.value),
            detail="logger.warning_once",
        )
    return None


_DETECTORS = (_detect_raise, _detect_logger_warning)


def _detect_body(stmt: ast.stmt) -> _DetectedBody | None:
    for det in _DETECTORS:
        result = det(stmt)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# AST traversal - model_validator / field_validator method bodies
# ---------------------------------------------------------------------------


def _flatten_if_chain(if_node: ast.If) -> list[tuple[ast.expr | None, list[ast.stmt]]]:
    branches: list[tuple[ast.expr | None, list[ast.stmt]]] = []
    cur: ast.If | None = if_node
    while cur is not None:
        branches.append((cur.test, cur.body))
        if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
        else:
            if cur.orelse:
                branches.append((None, cur.orelse))
            break
    return branches


def _walk_if_block(if_node: ast.If, frame: _Frame, emit: _Emitter) -> None:
    """Walk an ``if/elif/else`` chain; emit one invariant per detected body stmt."""
    for cond, body in _flatten_if_chain(if_node):
        preds = _extract_predicates(cond) if cond is not None else []
        local_frame = _Frame(predicates=[*frame.predicates, *preds])
        for stmt in body:
            detected = _detect_body(stmt)
            if detected is not None:
                emit.emit(detected, local_frame, line=getattr(stmt, "lineno", if_node.lineno))
            if isinstance(stmt, ast.If):
                _walk_if_block(stmt, local_frame, emit)
            elif isinstance(stmt, ast.For):
                _walk_for_block(stmt, local_frame, emit)


def _walk_for_block(for_node: ast.For, frame: _Frame, emit: _Emitter) -> None:
    """Walk a ``for key in [literal, ...]:`` loop; parameterise invariants per literal.

    For loops over a literal list/tuple of strings (the
    ``set_runtime_knobs_from_build_config`` pattern), expand the loop into
    one invariant per literal - the loop variable's bound value gets recorded as
    extra context on each emitted invariant.
    """
    literals = _literal_iterable(for_node.iter)
    target_name = for_node.target.id if isinstance(for_node.target, ast.Name) else None
    if literals is None or target_name is None:
        # Unparameterisable loop - descend into body anyway.
        for stmt in for_node.body:
            detected = _detect_body(stmt)
            if detected is not None:
                emit.emit(detected, frame, line=getattr(stmt, "lineno", for_node.lineno))
            if isinstance(stmt, ast.If):
                _walk_if_block(stmt, frame, emit)
            elif isinstance(stmt, ast.For):
                _walk_for_block(stmt, frame, emit)
        return

    for literal in literals:
        loop_frame = _Frame(predicates=list(frame.predicates))
        emit.push_loop_var(target_name, literal)
        try:
            for stmt in for_node.body:
                detected = _detect_body(stmt)
                if detected is not None:
                    emit.emit(detected, loop_frame, line=getattr(stmt, "lineno", for_node.lineno))
                if isinstance(stmt, ast.If):
                    _walk_if_block(stmt, loop_frame, emit)
                elif isinstance(stmt, ast.For):
                    _walk_for_block(stmt, loop_frame, emit)
        finally:
            emit.pop_loop_var()


def _literal_iterable(iter_node: ast.expr) -> list[Any] | None:
    if not isinstance(iter_node, (ast.List, ast.Tuple)):
        return None
    out: list[Any] = []
    for elt in iter_node.elts:
        if isinstance(elt, ast.Constant):
            out.append(elt.value)
        else:
            return None
    return out


# ---------------------------------------------------------------------------
# Invariant emission
# ---------------------------------------------------------------------------


@dataclass
class _Emitter:
    """Stateful collector for invariants produced from one validator method."""

    native_type: str
    method_name: str
    rel_source_path: str
    today: str
    invariants: list[InvariantCandidate] = field(default_factory=list)
    _seen_ids: set[str] = field(default_factory=set)
    _id_counter: dict[str, int] = field(default_factory=dict)
    _loop_vars: list[tuple[str, Any]] = field(default_factory=list)

    def push_loop_var(self, name: str, value: Any) -> None:
        self._loop_vars.append((name, value))

    def pop_loop_var(self) -> None:
        self._loop_vars.pop()

    def emit(self, detected: _DetectedBody, frame: _Frame, *, line: int) -> None:
        # Resolve a "subject field" for the invariant. If a loop variable holds a
        # string, treat it as the affected field (set_runtime_knobs pattern).
        subject_field: str | None = None
        loop_context: dict[str, Any] = {}
        for name, value in self._loop_vars:
            loop_context[name] = value
            if isinstance(value, str) and subject_field is None:
                subject_field = value

        preds = list(frame.predicates)
        if subject_field is not None and not any(p.field == subject_field for p in preds):
            preds.append(_Predicate(field=subject_field, op="present", rhs=True))

        if not preds:
            # No predicates at all - the invariant has nothing to match on. Skip
            # to avoid fingerprint collisions in the merger.
            return

        match_fields = self._build_match_fields(preds)
        kwargs_pos = self._synthesise_kwargs(preds, sense="positive")
        kwargs_neg = self._synthesise_kwargs(self._negate(preds), sense="negative")
        if kwargs_pos == kwargs_neg:
            kwargs_neg = self._force_distinct(kwargs_pos, preds)

        invariant_id = self._make_id(preds, detected, subject_field)
        invariant_under_test = self._describe(preds, detected)

        invariant = InvariantCandidate(
            id=invariant_id,
            engine=ENGINE,
            library=LIBRARY,
            invariant_under_test=invariant_under_test,
            severity=detected.severity,
            native_type=self.native_type,
            miner_source=MinerSource(
                path=self.rel_source_path,
                method=self.method_name,
                line_at_scan=line,
            ),
            match_fields=match_fields,
            kwargs_positive=kwargs_pos,
            kwargs_negative=kwargs_neg,
            expected_outcome={
                "outcome": detected.outcome,
                "emission_channel": detected.emission_channel,
                "normalised_fields": [],
            },
            message_template=detected.message_template,
            references=[
                f"{self.rel_source_path}:{line} ({self.native_type}.{self.method_name})",
            ],
            added_by="static_miner",
            added_at=self.today,
        )
        # De-dup id collisions (same predicate shape on two different lines).
        if invariant.id in self._seen_ids:
            self._id_counter[invariant.id] = self._id_counter.get(invariant.id, 1) + 1
            invariant.id = f"{invariant.id}__{self._id_counter[invariant.id]}"
        self._seen_ids.add(invariant.id)
        self.invariants.append(invariant)

    # -- helpers --------------------------------------------------------

    def _build_match_fields(self, preds: list[_Predicate]) -> dict[str, Any]:
        grouped: dict[str, dict[str, Any]] = {}
        for p in preds:
            path = f"{NAMESPACE}.{p.field}"
            spec = grouped.setdefault(path, {})
            spec[p.op] = p.rhs
        out: dict[str, Any] = {}
        for path, spec in grouped.items():
            if len(spec) == 1 and "==" in spec:
                out[path] = spec["=="]
            else:
                out[path] = spec
        return out

    def _negate(self, preds: list[_Predicate]) -> list[_Predicate]:
        if not preds:
            return []
        last = preds[-1]
        new_op = _INVERSION_MAP.get(last.op, last.op)
        return [
            *preds[:-1],
            _Predicate(field=last.field, op=new_op, rhs=last.rhs),
        ]

    def _synthesise_kwargs(self, preds: list[_Predicate], *, sense: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for p in preds:
            out.setdefault(p.field, _value_for(p, out))
            if isinstance(p.rhs, str) and p.rhs.startswith("@"):
                ref_field = p.rhs[1:].split(".")[-1]
                out.setdefault(ref_field, _companion_value(p))
        return out

    def _force_distinct(self, pos: dict[str, Any], preds: list[_Predicate]) -> dict[str, Any]:
        if not preds:
            return pos
        out = dict(pos)
        last = preds[-1]
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

    def _make_id(
        self,
        preds: list[_Predicate],
        detected: _DetectedBody,
        subject_field: str | None,
    ) -> str:
        sev = {"error": "raises", "warn": "warns"}[detected.severity]
        last = preds[-1]
        op_word = {
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
        }.get(last.op, last.op.replace(" ", "_"))
        rhs_part: str = ""
        if isinstance(last.rhs, (int, float)):
            rhs_part = str(last.rhs).replace("-", "neg").replace(".", "p")
        elif isinstance(last.rhs, str):
            tail = last.rhs.replace("@", "ref_").replace("-", "neg")
            rhs_part = "".join(ch for ch in tail if ch.isalnum() or ch == "_").lower()
        elif isinstance(last.rhs, bool):
            rhs_part = "true" if last.rhs else "false"
        method_slug = self.method_name.replace("validate_", "")
        parts = ["tensorrt", sev, last.field, op_word]
        if rhs_part:
            parts.append(rhs_part[:24])
        parts.append(method_slug)
        rid = "_".join(parts)
        while "__" in rid:
            rid = rid.replace("__", "_")
        return rid.strip("_")

    def _describe(self, preds: list[_Predicate], detected: _DetectedBody) -> str:
        pred_str = " AND ".join(f"{p.field} {p.op} {p.rhs}" for p in preds)
        sev_word = {"error": "raises", "warn": "warns"}[detected.severity]
        cls_short = self.native_type.rsplit(".", 1)[-1]
        return f"{cls_short}.{self.method_name} {sev_word} when {pred_str} (via {detected.detail})"


def _value_for(p: _Predicate, others: dict[str, Any]) -> Any:
    """Pick a concrete value for ``p.field`` that satisfies the predicate."""
    if isinstance(p.rhs, str) and p.rhs.startswith("@"):
        ref = p.rhs[1:].split(".")[-1]
        companion = others.get(ref)
        if companion is None:
            companion = 2
        return _value_satisfying(p.op, companion)
    return _value_satisfying(p.op, p.rhs)


def _companion_value(p: _Predicate) -> Any:
    if p.op in {">", ">=", "<", "<="}:
        return 2
    return 1


def _value_satisfying(op: str, rhs: Any) -> Any:
    if op == "present":
        return rhs if rhs is not True else "x"
    if op == "absent":
        return None
    if op == "type_is":
        return _type_default(rhs)
    if op == "type_is_not":
        return _other_type_default(rhs)
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
        if isinstance(rhs, int) and not isinstance(rhs, bool):
            return rhs - 1
        if isinstance(rhs, float):
            return rhs - 1.0
        return rhs
    if op == "<=":
        return rhs
    if op == ">":
        if isinstance(rhs, int) and not isinstance(rhs, bool):
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
        return "__static_miner_synth__"
    return rhs


def _type_default(label: Any) -> Any:
    s = label if isinstance(label, str) else (label[0] if label else "str")
    return {
        "bool": True,
        "int": 1,
        "float": 1.0,
        "str": "x",
        "list": [],
        "Path": "/tmp/synthetic",
    }.get(s, "x")


def _other_type_default(label: Any) -> Any:
    s = label if isinstance(label, str) else (label[0] if label else "str")
    if s in {"str", "Path"}:
        return 1
    if s in {"int", "float"}:
        return "x"
    if s == "bool":
        return 1
    return "x"


# ---------------------------------------------------------------------------
# Validator-method miner - public entry
# ---------------------------------------------------------------------------


def _walk_method(
    method: ast.FunctionDef,
    *,
    native_type: str,
    method_name: str,
    rel_source_path: str,
    today: str,
) -> list[InvariantCandidate]:
    """Walk the body of one validator method and return invariant candidates."""
    emitter = _Emitter(
        native_type=native_type,
        method_name=method_name,
        rel_source_path=rel_source_path,
        today=today,
    )
    # Field validators are ``def validate_X(cls, v, info=None)`` - they raise on
    # ``v`` (the new value), not ``self.X``. For these, we emit at most one
    # invariant per top-level raise, treating the field name as the implicit
    # subject. Detect by signature: first arg is ``cls`` and second is ``v``.
    is_field_validator = _is_field_validator(method)
    if is_field_validator:
        emitter._loop_vars = []
        return _walk_field_validator(method, emitter, native_type, method_name)

    # model_validator(mode="after") body - descend with no preconditions.
    frame = _Frame()
    for stmt in method.body:
        if isinstance(stmt, ast.If):
            _walk_if_block(stmt, frame, emitter)
        elif isinstance(stmt, ast.For):
            _walk_for_block(stmt, frame, emitter)
    return emitter.invariants


def _is_field_validator(method: ast.FunctionDef) -> bool:
    for deco in method.decorator_list:
        if (
            isinstance(deco, ast.Call)
            and isinstance(deco.func, ast.Name)
            and deco.func.id == "field_validator"
        ):
            return True
    return False


def _field_validator_targets(method: ast.FunctionDef) -> list[str]:
    """Return the list of field names a ``@field_validator(...)`` targets."""
    for deco in method.decorator_list:
        if (
            isinstance(deco, ast.Call)
            and isinstance(deco.func, ast.Name)
            and deco.func.id == "field_validator"
        ):
            names: list[str] = []
            for arg in deco.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    names.append(arg.value)
            return names
    return []


def _walk_field_validator(
    method: ast.FunctionDef,
    emitter: _Emitter,
    native_type: str,
    method_name: str,
) -> list[InvariantCandidate]:
    """Emit one invariant per (target field, raise/warn site) pair.

    Field validators take ``(cls, v, info=None)``. Raises inside the body
    fire when ``v`` violates a check. We translate each detected raise/warn
    into a invariant keyed on every target field declared in the decorator.

    Conditions like ``if v <= 0: raise`` are translated to the equivalent
    predicate on the *target field*: ``self.<field> <= 0``. This is the same
    invariant from the corpus loader's perspective.
    """
    targets = _field_validator_targets(method)
    if not targets:
        return []
    invariants: list[InvariantCandidate] = []
    for target in targets:
        sub_emitter = _Emitter(
            native_type=native_type,
            method_name=method_name,
            rel_source_path=emitter.rel_source_path,
            today=emitter.today,
        )
        for stmt in method.body:
            _walk_field_validator_stmt(stmt, target, sub_emitter)
        invariants.extend(sub_emitter.invariants)
    return invariants


def _walk_field_validator_stmt(stmt: ast.stmt, target: str, emitter: _Emitter) -> None:
    """Walk one statement inside a field_validator body, rebinding ``v`` to ``target``.

    The detector emits a invariant whose predicates are derived from the condition,
    with all references to the local parameter ``v`` rewritten as
    ``self.<target>``. Other patterns (assignments to ``v`` etc.) are ignored.
    """
    if isinstance(stmt, ast.If):
        preds = _extract_v_predicates(stmt.test, target)
        for body_stmt in stmt.body:
            detected = _detect_body(body_stmt)
            if detected is not None:
                emitter.emit(
                    detected,
                    _Frame(predicates=preds),
                    line=getattr(body_stmt, "lineno", stmt.lineno),
                )
            if isinstance(body_stmt, ast.If):
                # Nested if: combine preds.
                inner_preds = _extract_v_predicates(body_stmt.test, target)
                for inner_stmt in body_stmt.body:
                    inner_detected = _detect_body(inner_stmt)
                    if inner_detected is not None:
                        emitter.emit(
                            inner_detected,
                            _Frame(predicates=[*preds, *inner_preds]),
                            line=getattr(inner_stmt, "lineno", body_stmt.lineno),
                        )


def _extract_v_predicates(condition: ast.expr, target: str) -> list[_Predicate]:
    """Translate a field_validator condition over ``v`` to predicates on ``self.<target>``."""
    if isinstance(condition, ast.Compare) and len(condition.ops) == 1:
        op = condition.ops[0]
        op_name = _COMPARE_OP_NAMES.get(type(op))
        # ``v op literal`` - the canonical shape.
        if isinstance(condition.left, ast.Name) and condition.left.id == "v" and op_name:
            ok, rhs = _literal_value(condition.comparators[0])
            if ok:
                return [_Predicate(field=target, op=op_name, rhs=rhs)]
        # ``literal op v`` - flip.
        if (
            isinstance(condition.comparators[0], ast.Name)
            and condition.comparators[0].id == "v"
            and op_name
        ):
            ok, rhs = _literal_value(condition.left)
            if ok:
                flipped = {
                    "<": ">",
                    "<=": ">=",
                    ">": "<",
                    ">=": "<=",
                    "==": "==",
                    "!=": "!=",
                }.get(op_name)
                if flipped:
                    return [_Predicate(field=target, op=flipped, rhs=rhs)]
    if (
        isinstance(condition, ast.UnaryOp)
        and isinstance(condition.op, ast.Not)
        and isinstance(condition.operand, ast.Call)
    ):
        path = call_func_path(condition.operand)
        if (
            path
            and path[-1] == "isinstance"
            and len(condition.operand.args) == 2
            and isinstance(condition.operand.args[0], ast.Name)
            and condition.operand.args[0].id == "v"
        ):
            names = _isinstance_type_names(condition.operand.args[1])
            if names:
                rhs = names[0] if len(names) == 1 else names
                return [_Predicate(field=target, op="type_is_not", rhs=rhs)]
    return []


# ---------------------------------------------------------------------------
# Pydantic schema lift - source-driven
# ---------------------------------------------------------------------------


def _walk_literal_fields(
    cls_node: ast.ClassDef,
    *,
    native_type: str,
    rel_source_path: str,
    today: str,
) -> list[InvariantCandidate]:
    """Emit one allowlist invariant per ``Literal[...]``-typed Pydantic field.

    Equivalent to ``_pydantic_lift._from_literal`` but reads the AST instead
    of importing the live class - needed because TRT-LLM isn't importable
    on the host (the design's "source-driven" contract).
    """
    out: list[InvariantCandidate] = []
    for stmt in cls_node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        field_name = stmt.target.id
        annotation = stmt.annotation
        # Shared deterministic-floor walker. Byte-identical to the former bespoke
        # ``_literal_args`` for the bare ``Literal[...]`` shapes in the 1.0.0
        # source; a strict superset for the ``Optional[Literal[...]]`` /
        # ``Annotated[Literal[...], ...]`` wrappers this version's source does
        # not yet use.
        values = _literal_values(annotation)
        if values is None:
            continue
        out.append(
            _make_literal_rule(
                field_name=field_name,
                values=values,
                native_type=native_type,
                rel_source_path=rel_source_path,
                line=stmt.lineno,
                today=today,
            )
        )
    return out


def _make_literal_rule(
    *,
    field_name: str,
    values: list[Any],
    native_type: str,
    rel_source_path: str,
    line: int,
    today: str,
) -> InvariantCandidate:
    cls_short = native_type.rsplit(".", 1)[-1]
    rid = f"tensorrt_{cls_short.lower()}_{field_name}_in_{len(values)}_values"
    sample_invalid = "<invalid_static_miner_probe>"
    sample_valid = values[0]
    return InvariantCandidate(
        id=rid,
        engine=ENGINE,
        library=LIBRARY,
        invariant_under_test=(f"{cls_short}.{field_name} must be one of {list(values)!r}"),
        severity="error",
        native_type=native_type,
        miner_source=MinerSource(
            path=rel_source_path,
            method="<literal_field>",
            line_at_scan=line,
        ),
        match_fields={f"{NAMESPACE}.{field_name}": {"in": list(values)}},
        kwargs_positive={field_name: sample_invalid},
        kwargs_negative={field_name: sample_valid},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=(f"`{field_name}` must be one of {list(values)!r}"),
        references=[
            f"{rel_source_path}:{line} ({native_type}.{field_name} Literal annotation)",
        ],
        added_by="static_miner",
        added_at=today,
    )


def _walk_strenum(
    cls_node: ast.ClassDef,
    *,
    native_type: str,
    field_name: str,
    rel_source_path: str,
    today: str,
) -> InvariantCandidate | None:
    """Lift a ``StrEnum`` class to a single allowlist invariant for the named field.

    The enum class itself is the underlying type for one particular field on
    ``TrtLlmArgs`` (e.g. ``BatchingType`` is the type of ``batching_type``).
    The invariant constrains the field's value to the enum's string members.
    """
    values: list[str] = []
    for item in cls_node.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
            if (
                isinstance(target, ast.Name)
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ):
                values.append(item.value.value)
    if not values:
        return None
    cls_short = cls_node.name
    rid = f"tensorrt_{field_name}_in_{len(values)}_values"
    return InvariantCandidate(
        id=rid,
        engine=ENGINE,
        library=LIBRARY,
        invariant_under_test=(
            f"{native_type.split('.')[-1]}.{field_name} must be one of "
            f"{cls_short} members {values!r}"
        ),
        severity="error",
        native_type=native_type,
        miner_source=MinerSource(
            path=rel_source_path,
            method="<strenum>",
            line_at_scan=cls_node.lineno,
        ),
        match_fields={f"{NAMESPACE}.{field_name}": {"in": list(values)}},
        kwargs_positive={field_name: "<invalid_static_miner_probe>"},
        kwargs_negative={field_name: values[0]},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=(f"`{field_name}` must be one of {cls_short} members: {values!r}"),
        references=[
            f"{rel_source_path}:{cls_node.lineno} ({cls_short} StrEnum)",
        ],
        added_by="static_miner",
        added_at=today,
    )


# ---------------------------------------------------------------------------
# Source loading / orchestration
# ---------------------------------------------------------------------------


@dataclass
class _SourceTree:
    """Loaded ASTs for the files this miner reads."""

    llm_args: ast.Module
    builder: ast.Module
    llm_args_rel: str
    builder_rel: str


def _read_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


def _load_source(source_root: Path) -> _SourceTree:
    """Load 1.0.0 source ASTs and verify class-level landmarks."""
    if not source_root.is_dir():
        raise MinerLandmarkMissingError(
            "tensorrt_llm source root",
            detail=f"{source_root} not found - extract 1.0.0 source first",
        )
    llm_args_path = source_root / LLM_ARGS_REL
    builder_path = source_root / BUILDER_REL
    if not llm_args_path.is_file():
        raise MinerLandmarkMissingError(
            "tensorrt_llm/llmapi/llm_args.py",
            detail=f"missing at {llm_args_path}",
        )
    if not builder_path.is_file():
        raise MinerLandmarkMissingError(
            "tensorrt_llm/builder.py", detail=f"missing at {builder_path}"
        )
    llm_args = _read_module(llm_args_path)
    builder = _read_module(builder_path)

    # Class landmarks - fail-loud if any are missing.
    for cls_name, rel in _CLASS_TARGETS:
        module = llm_args if rel == LLM_ARGS_REL else builder
        if find_class(module, cls_name) is None:
            raise MinerLandmarkMissingError(
                f"{rel}::{cls_name}",
                detail="class missing in 1.0.0 source - has the library upgraded?",
            )

    return _SourceTree(
        llm_args=llm_args,
        builder=builder,
        llm_args_rel=str(LLM_ARGS_REL),
        builder_rel=str(BUILDER_REL),
    )


def _verify_method_landmarks(tree: _SourceTree) -> None:
    """Fail-loud if any expected validator method is missing in 1.0.0 source."""
    for cls_name, method_name in _METHOD_LANDMARKS:
        cls = find_class(tree.llm_args, cls_name)
        # Class-existence already checked in _load_source.
        assert cls is not None
        if find_method(cls, method_name) is None:
            raise MinerLandmarkMissingError(
                f"{cls_name}.{method_name}",
                detail="validator method missing in 1.0.0 source",
            )


def probe_uncovered_validators(
    source_root: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Drift-completeness probe: raise-bearing validators NOT in ``_METHOD_LANDMARKS``.

    AST-source based (tensorrt cannot be imported on host - no CUDA): parses the
    extracted 1.0.0 source tree and returns ``(validators_uncovered,
    validators_unanalyzable)``. The covered set is the validator methods the
    miner actually walks (:data:`_METHOD_LANDMARKS`); ``class_targets`` /
    ``strenum_fields`` / :data:`_LITERAL_LIFT_CLASSES` are field-type lifts, not
    validator walks, so they are NOT treated as covered. Seeds (the
    genuinely-flat config surface that escapes the blob-only exclusion) = the
    ``class_targets`` classes; the flat-hoist field set = ``TrtLlmArgs``'s
    PUBLIC fields including those inherited from ``BaseLlmArgs`` (so the blob
    escape can auto-discover relocations into engine-arg-shaped classes that are
    not yet named). Every class in the source files is scanned so a validator
    relocated into a brand-new config class is still caught. Called best-effort
    by :func:`scripts._drift.run`; a missing source tree degrades to a single
    ``<prober-error>`` marker rather than a false completeness gap.
    """
    from scripts.engine_producers import _completeness

    root: Path = source_root if source_root is not None else _DEFAULT_SOURCE_ROOT
    tree = _load_source(root)
    classdefs: dict[str, ast.ClassDef] = {}
    collisions: list[str] = []
    for module in (tree.llm_args, tree.builder):
        for node in ast.iter_child_nodes(module):
            if isinstance(node, ast.ClassDef):
                if node.name in classdefs:
                    # Same short name in both source files: the covered/seed sets
                    # key on short names, so a silent drop could hide validators.
                    # Surface it (never silently lose a class).
                    collisions.append(f"{node.name}:<short-name collision across source files>")
                    continue
                classdefs[node.name] = node

    covered = {(cls, method) for cls, method in _METHOD_LANDMARKS}
    seed_names = {cls for cls, _rel in _CLASS_TARGETS}
    # Inheritance-aware + public-only hoist surface (the blob-escape measures
    # overlap of PUBLIC knobs; private PrivateAttr fields are not user knobs).
    hoist_fields = {
        f
        for f in _completeness._settable_fields_ast_inherited("TrtLlmArgs", classdefs)
        if not f.startswith("_")
    }
    uncovered, unanalyzable = _completeness.compute_uncovered_validators_ast(
        classdefs=classdefs,
        covered=covered,
        hoist_fields=hoist_fields,
        seed_names=seed_names,
    )
    return uncovered, sorted(set(unanalyzable) | set(collisions))


def walk_tensorrt(source_root: Path | None = None) -> tuple[list[InvariantCandidate], str, str]:
    """Walk TRT-LLM source and return ``(candidates, version, llm_args_rel_path)``.

    ``source_root`` defaults to ``/tmp/trt-llm-1.0.0/tensorrt_llm`` -
    overridable for tests / CI extraction in different paths.
    """
    root = source_root if source_root is not None else _DEFAULT_SOURCE_ROOT
    tree = _load_source(root)
    _verify_method_landmarks(tree)

    # Read the version from the source tree so the orchestrator can pin
    # against the SSOT miner range without importing the library.
    version = _read_source_version(root)

    today = dt.date.today().isoformat()
    candidates: list[InvariantCandidate] = []

    # 1) Validator-method walks.
    for cls_name, method_name in _METHOD_LANDMARKS:
        cls = find_class(tree.llm_args, cls_name)
        assert cls is not None
        method = find_method(cls, method_name)
        assert method is not None
        candidates.extend(
            _walk_method(
                method,
                native_type=f"{LIBRARY}.{cls_name}",
                method_name=method_name,
                rel_source_path=tree.llm_args_rel,
                today=today,
            )
        )

    # 2) Source-driven Literal-field lift on Pydantic configs.
    for cls_name in _LITERAL_LIFT_CLASSES:
        cls = find_class(tree.llm_args, cls_name)
        assert cls is not None
        candidates.extend(
            _walk_literal_fields(
                cls,
                native_type=f"{LIBRARY}.{cls_name}",
                rel_source_path=tree.llm_args_rel,
                today=today,
            )
        )

    # 3) StrEnum-typed fields on TrtLlmArgs (BatchingType / etc.).
    for enum_cls, field_name in _STRENUM_FIELDS:
        cls = find_class(tree.llm_args, enum_cls)
        if cls is None:
            continue
        invariant = _walk_strenum(
            cls,
            native_type=f"{LIBRARY}.TrtLlmArgs",
            field_name=field_name,
            rel_source_path=tree.llm_args_rel,
            today=today,
        )
        if invariant is not None:
            candidates.append(invariant)

    return candidates, version, tree.llm_args_rel


def _read_source_version(source_root: Path) -> str:
    """Read ``__version__`` from ``tensorrt_llm/version.py`` without importing."""
    version_path = source_root / "version.py"
    if not version_path.is_file():
        raise MinerLandmarkMissingError(
            "tensorrt_llm/version.py", detail=f"missing at {version_path}"
        )
    text = version_path.read_text()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and target.id == "__version__"
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                return node.value.value
    raise MinerLandmarkMissingError(
        "tensorrt_llm.__version__", detail="version.py exists but no __version__ literal"
    )


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def _candidate_to_dict(c: InvariantCandidate, curated_sections: dict[str, str]) -> dict[str, Any]:
    return {
        "id": c.id,
        "engine": c.engine,
        "library": c.library,
        "invariant_under_test": c.invariant_under_test,
        "severity": c.severity,
        "native_type": c.native_type,
        "miner_source": {
            "path": c.miner_source.path,
            "method": c.miner_source.method,
            "line_at_scan": c.miner_source.line_at_scan,
        },
        "match": {
            "engine": c.engine,
            # D2: re-key onto classified {engine}.{section}.{field} paths
            # (curation first, then llm_args-side / sampling-side native origin).
            "fields": relabel_match_fields(
                c.match_fields,
                engine=ENGINE,
                native_type=c.native_type,
                curated_sections=curated_sections,
            ),
        },
        "kwargs_positive": c.kwargs_positive,
        "kwargs_negative": c.kwargs_negative,
        "expected_outcome": c.expected_outcome,
        "message_template": c.message_template,
        "references": c.references,
        "added_by": c.added_by,
        "added_at": c.added_at,
    }


def emit_yaml(
    candidates: list[InvariantCandidate],
    *,
    engine_version: str,
    rel_path: str,
) -> str:
    import yaml

    sorted_candidates = sorted(candidates, key=lambda c: (c.miner_source.method, c.id))
    frozen = os.environ.get("LLENERGY_MINER_FROZEN_AT")
    mined_at = (
        frozen
        if frozen
        else dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    curated_sections = load_curated_sections(ENGINE)
    doc: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": engine_version,
        "mined_at": mined_at,
        "invariants": [_candidate_to_dict(c, curated_sections) for c in sorted_candidates],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("src/llenergymeasure/engines/tensorrt/_staging/tensorrt_static_miner.yaml"),
        help=(
            "Where to write the staging YAML (default: "
            "src/llenergymeasure/engines/tensorrt/_staging/tensorrt_static_miner.yaml)"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_DEFAULT_SOURCE_ROOT,
        help=(
            "Path to the extracted tensorrt_llm 1.0.0 source tree (default: "
            f"{_DEFAULT_SOURCE_ROOT})"
        ),
    )
    args = parser.parse_args(argv)

    candidates, engine_version, rel_path = walk_tensorrt(args.source_root)
    text = emit_yaml(candidates, engine_version=engine_version, rel_path=rel_path)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(
        f"Wrote {len(candidates)} candidate invariants to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
