"""AST static miner for vLLM 0.16.0.

Vendored machinery: full miner body lives here, version-pinned. The
global ``scripts/engine_producers/vllm_static_miner.py`` is a thin
dispatcher that resolves ``library.current_version`` from the SSOT and
delegates to the per-version archive's module-level attributes.

vLLM 0.16.0 introduced the per-concern config subpackage layout
(``vllm.config.parallel``, ``vllm.config.lora``, ...) and renamed most
imperative validators to ``_validate_*`` methods. The :data:`_CLASS_TARGETS`
registry below reflects that surface; the walker / detector / kwargs-
synth machinery is engine-version-agnostic.
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
    InvariantCandidate,
    MinerLandmarkMissingError,
    MinerSource,
    call_func_path,
    find_class,
    find_method,
    first_string_arg,
)
from scripts.engine_producers._current import current_outputs_dir

# ---------------------------------------------------------------------------
# Engine + namespace conventions
# ---------------------------------------------------------------------------

ENGINE = "vllm"
LIBRARY = "vllm"

NS_SAMPLING = "vllm.sampling"
NS_STRUCTURED = "vllm.sampling.structured_outputs"
NS_ENGINE = "vllm.engine"


# Probe-contract landmarks. Read by ``scripts._probe`` before the miner
# runs; a missing landmark flips the probe verdict to ``fail``.
LANDMARKS: tuple[str, ...] = (
    "vllm.sampling_params.SamplingParams",
    "vllm.sampling_params.SamplingParams._verify_args",
    "vllm.sampling_params.SamplingParams.__post_init__",
    "vllm.sampling_params.SamplingParams._verify_greedy_sampling",
    "vllm.sampling_params.StructuredOutputsParams",
    "vllm.sampling_params.StructuredOutputsParams.__post_init__",
    "vllm.config.parallel.ParallelConfig",
    "vllm.config.parallel.ParallelConfig._validate_parallel_config",
    "vllm.config.parallel.ParallelConfig._verify_args",
    "vllm.config.parallel.ParallelConfig.__post_init__",
    "vllm.config.parallel.EPLBConfig",
    "vllm.config.parallel.EPLBConfig._validate_eplb_config",
    "vllm.config.lora.LoRAConfig",
    "vllm.config.lora.LoRAConfig._validate_lora_config",
    "vllm.config.multimodal.MultiModalConfig",
    "vllm.config.multimodal.MultiModalConfig._validate_multimodal_config",
    "vllm.config.structured_outputs.StructuredOutputsConfig",
    "vllm.config.structured_outputs.StructuredOutputsConfig._validate_structured_output_config",
    "vllm.config.cache.CacheConfig",
    "vllm.config.cache.CacheConfig._validate_cache_dtype",
    "vllm.config.model.ModelConfig",
    "vllm.config.model.ModelConfig.__post_init__",
    "vllm.config.compilation.CompilationConfig",
    "vllm.config.compilation.CompilationConfig.__post_init__",
    "vllm.config.scheduler.SchedulerConfig",
    "vllm.config.scheduler.SchedulerConfig.__post_init__",
)


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


_CLASS_TARGETS: tuple[_ASTTarget, ...] = (
    # SamplingParams family - sampling_params.py
    _ASTTarget(
        module_attr="sampling_params.SamplingParams",
        method="_verify_args",
        namespace=NS_SAMPLING,
        native_type="vllm.SamplingParams",
    ),
    _ASTTarget(
        module_attr="sampling_params.SamplingParams",
        method="__post_init__",
        namespace=NS_SAMPLING,
        native_type="vllm.SamplingParams",
    ),
    _ASTTarget(
        module_attr="sampling_params.SamplingParams",
        method="_verify_greedy_sampling",
        namespace=NS_SAMPLING,
        native_type="vllm.SamplingParams",
    ),
    _ASTTarget(
        module_attr="sampling_params.StructuredOutputsParams",
        method="__post_init__",
        namespace=NS_STRUCTURED,
        native_type="vllm.sampling_params.StructuredOutputsParams",
    ),
    # vllm.config.* family (subpackage layout introduced in 0.16.0).
    _ASTTarget(
        module_attr="config.parallel.ParallelConfig",
        method="_validate_parallel_config",
        namespace=NS_ENGINE,
        native_type="vllm.config.ParallelConfig",
    ),
    _ASTTarget(
        module_attr="config.parallel.ParallelConfig",
        method="_verify_args",
        namespace=NS_ENGINE,
        native_type="vllm.config.ParallelConfig",
    ),
    _ASTTarget(
        module_attr="config.parallel.ParallelConfig",
        method="__post_init__",
        namespace=NS_ENGINE,
        native_type="vllm.config.ParallelConfig",
    ),
    _ASTTarget(
        module_attr="config.parallel.EPLBConfig",
        method="_validate_eplb_config",
        namespace=NS_ENGINE,
        native_type="vllm.config.EPLBConfig",
    ),
    _ASTTarget(
        module_attr="config.lora.LoRAConfig",
        method="_validate_lora_config",
        namespace=NS_ENGINE,
        native_type="vllm.config.LoRAConfig",
    ),
    _ASTTarget(
        module_attr="config.multimodal.MultiModalConfig",
        method="_validate_multimodal_config",
        namespace=NS_ENGINE,
        native_type="vllm.config.MultiModalConfig",
    ),
    _ASTTarget(
        module_attr="config.structured_outputs.StructuredOutputsConfig",
        method="_validate_structured_output_config",
        namespace=NS_ENGINE,
        native_type="vllm.config.StructuredOutputsConfig",
    ),
    _ASTTarget(
        module_attr="config.cache.CacheConfig",
        method="_validate_cache_dtype",
        namespace=NS_ENGINE,
        native_type="vllm.config.CacheConfig",
    ),
    _ASTTarget(
        module_attr="config.model.ModelConfig",
        method="__post_init__",
        namespace=NS_ENGINE,
        native_type="vllm.config.ModelConfig",
    ),
    _ASTTarget(
        module_attr="config.compilation.CompilationConfig",
        method="__post_init__",
        namespace=NS_ENGINE,
        native_type="vllm.config.CompilationConfig",
    ),
    _ASTTarget(
        module_attr="config.scheduler.SchedulerConfig",
        method="__post_init__",
        namespace=NS_ENGINE,
        native_type="vllm.config.SchedulerConfig",
    ),
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


_COMPARE_OPS: dict[type[ast.cmpop], str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not_in",
}

_FLIPPED_OPS: dict[str, str] = {
    "<": ">",
    "<=": ">=",
    ">": "<",
    ">=": "<=",
    "==": "==",
    "!=": "!=",
}

_INVERSE_OPS: dict[str, str] = {
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


def _literal(node: ast.expr) -> tuple[bool, Any]:
    if isinstance(node, ast.Constant):
        return True, node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        ok, v = _literal(node.operand)
        if ok and isinstance(v, (int, float)):
            return True, -v
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: list[Any] = []
        for elt in node.elts:
            ok, v = _literal(elt)
            if not ok:
                return False, None
            out.append(v)
        return True, list(out)
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        return True, {"True": True, "False": False, "None": None}[node.id]
    return False, None


def _rhs_value(node: ast.expr) -> tuple[bool, Any]:
    ok, v = _literal(node)
    if ok:
        return True, v
    name = _self_attr(node)
    if name is not None:
        return True, f"@{name}"
    return False, None


def _extract_compare(cmp: ast.Compare) -> list[_Predicate]:
    preds: list[_Predicate] = []
    operands = [cmp.left, *cmp.comparators]
    for left, op, right in zip(operands, cmp.ops, cmp.comparators, strict=False):
        if isinstance(op, (ast.Is, ast.IsNot)):
            field_name = _self_attr(left)
            ok, rhs = _literal(right)
            if field_name is not None and ok and rhs is None:
                preds.append(
                    _Predicate(
                        field=field_name,
                        op="absent" if isinstance(op, ast.Is) else "present",
                        rhs=True,
                    )
                )
            continue
        op_name = _COMPARE_OPS.get(type(op))
        if op_name is None:
            continue
        left_field = _self_attr(left)
        right_field = _self_attr(right)
        if left_field is not None:
            ok, rhs = _rhs_value(right)
            if ok:
                preds.append(_Predicate(field=left_field, op=op_name, rhs=rhs))
        elif right_field is not None:
            flipped = _FLIPPED_OPS.get(op_name)
            ok, rhs = _rhs_value(left)
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
        field_name = _self_attr(target)
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


def _extract_predicates(condition: ast.expr) -> list[_Predicate]:
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        out: list[_Predicate] = []
        for value in condition.values:
            out.extend(_extract_predicates(value))
        return out
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or):
        return []
    if isinstance(condition, ast.Compare):
        return _extract_compare(condition)
    if isinstance(condition, ast.Call):
        return _extract_call(condition)
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        inner = _extract_predicates(condition.operand)
        if len(inner) == 1 and inner[0].op in _INVERSE_OPS:
            p = inner[0]
            return [_Predicate(field=p.field, op=_INVERSE_OPS[p.op], rhs=p.rhs)]
        return []
    field_name = _self_attr(condition)
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
    flipped_op = _INVERSE_OPS.get(last.op, last.op)
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
    rel_source_path: str,
    today: str,
) -> list[InvariantCandidate]:
    invariants: list[InvariantCandidate] = []
    seen_ids: set[str] = set()

    public_fields = _public_field_names_from_target(target)

    def descend(body: list[ast.stmt], frame: _Frame) -> None:
        for stmt in body:
            if isinstance(stmt, ast.If):
                _handle_if(stmt, frame)
            elif isinstance(stmt, ast.For):
                descend(stmt.body, frame)

    def _handle_if(if_node: ast.If, frame: _Frame) -> None:
        own_preds = _extract_predicates(if_node.test)
        if public_fields and not any(p.field in public_fields for p in own_preds):
            descend(if_node.body, frame)
            return
        local = _Frame(predicates=[*frame.predicates, *own_preds])
        for det in _detect_body_stmts(if_node.body):
            invariant = _build_rule(
                target=target,
                preds=local.predicates,
                detected=det,
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
    rel_source_path: str,
    today: str,
) -> InvariantCandidate | None:
    effective_preds = list(preds)
    subject_field = detected.affected_field
    if subject_field is not None and not any(p.field == subject_field for p in effective_preds):
        effective_preds.append(_Predicate(field=subject_field, op="present", rhs=True))

    if not effective_preds:
        return None

    match_fields = _build_match_fields(effective_preds, target.namespace)
    if not match_fields:
        return None

    kwargs_pos = _synthesise_kwargs(effective_preds)
    kwargs_neg = _synthesise_kwargs(_negate_predicates(effective_preds))
    if kwargs_pos == kwargs_neg:
        kwargs_neg = _force_distinct(kwargs_pos, effective_preds)

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


def _public_field_names_from_target(target: _ASTTarget) -> frozenset[str]:
    try:
        module = __import__(target.module_path, fromlist=[target.class_name])
        cls = getattr(module, target.class_name)
    except (ImportError, AttributeError):
        return frozenset()

    pyd_fields = getattr(cls, "__pydantic_fields__", None)
    if pyd_fields:
        return frozenset(pyd_fields.keys())
    struct_fields = getattr(cls, "__struct_fields__", None)
    if struct_fields:
        return frozenset(n for n in struct_fields if not n.startswith("_"))
    import dataclasses as _dc

    if _dc.is_dataclass(cls):
        return frozenset(f.name for f in _dc.fields(cls) if not f.name.startswith("_"))
    try:
        instance = cls()
        return frozenset(n for n in vars(instance) if not n.startswith("_"))
    except Exception:
        return frozenset()


def _field_default_from_target(target: _ASTTarget, field_name: str) -> tuple[bool, Any]:
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
        candidates.extend(
            _walk_function(
                method_ast,
                target=target,
                rel_source_path=rel_path,
                today=today,
            )
        )

    return candidates, installed_version


# ---------------------------------------------------------------------------
# YAML emission
# ---------------------------------------------------------------------------


def _candidate_to_dict(c: InvariantCandidate) -> dict[str, Any]:
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
            "fields": c.match_fields,
        },
        "kwargs_positive": c.kwargs_positive,
        "kwargs_negative": c.kwargs_negative,
        "expected_outcome": c.expected_outcome,
        "message_template": c.message_template,
        "references": c.references,
        "added_by": c.added_by,
        "added_at": c.added_at,
    }


def emit_yaml(candidates: list[InvariantCandidate], engine_version: str) -> str:
    import yaml

    sorted_candidates = sorted(candidates, key=lambda c: (c.miner_source.method, c.id))
    frozen = os.environ.get("LLENERGY_MINER_FROZEN_AT")
    mined_at = frozen if frozen else dt.date.today().isoformat()
    doc = {
        "schema_version": "1.0.0",
        "engine": ENGINE,
        "engine_version": engine_version,
        "mined_at": mined_at,
        "invariants": [_candidate_to_dict(c) for c in sorted_candidates],
    }
    return yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=current_outputs_dir("vllm") / "_staging" / "vllm_static_invariant_miner.yaml",
        help="Where to write the staging YAML (default: archive outputs/_staging/).",
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
