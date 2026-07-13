#!/usr/bin/env python3
"""Deterministic cross-field constraint extractor - a standing candidate source.

One non-vendored, engine-generic, deterministic proposer that mines cross-field
(and single-field) validation constraint CANDIDATES from pinned engine source and
feeds them into the absorb pool-union alongside the analyst cold read and manual
seeds. It PROPOSES; the verification ladder (the engine itself) adjudicates. It
never writes the shipped corpus.

This is the value core of the retired per-version invariant miners, extracted and
generalised: an AST walk over validator method bodies that turns ``if <condition>:
raise`` and ``if <condition>: self.x = <normalised>`` sites into match specs on the
declared config surface. What was per-version vendoring superstructure (one full
miner body per engine-version) collapses to a small per-engine descriptor table
(:data:`TARGETS`) plus this single walker.

Deterministic and byte-stable: no LLM, no network, no engine import. It reads the
extracted source tree on the host (``--source-root`` = the engine's top-level
package dir at the pinned version, the same contract as
:mod:`scripts.analyst_cold_read`), so it runs even for engines that bind CUDA at
import (tensorrt).

Run: python scripts/cross_field_extractor.py --engine vllm --source-root SRC/
Output: ``<pool-root>/<engine>/v<version>/candidates/cross_field_extractor.yaml``
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts._candidate_pool import pool_path, slug, unverified_verdict, write_pool  # noqa: E402
from scripts.analyst_cold_read import resolve_version  # noqa: E402

SOURCE = "deterministic_extractor"

_POOL_HEADER = (
    "# Deterministic cross-field extractor pool: cross- and single-field constraint\n"
    "# candidates mined from pinned engine source by a static AST walk over validator\n"
    "# bodies. UNVERIFIED candidates for the verification ladder; never a shipped corpus.\n"
)


# ---------------------------------------------------------------------------
# Per-engine descriptor table (the only per-engine knowledge)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One validator method to AST-walk, addressed within the source tree.

    ``rel_file`` is source-root-relative (e.g. ``config/compilation.py``);
    ``namespace`` prefixes the emitted field paths (absorb canonicalises them to
    the shipped ``engine.section.leaf`` form by leaf, so only the leaf matters
    downstream - the prefix is kept for provenance/debuggability).
    """

    rel_file: str
    cls: str
    method: str
    namespace: str
    native_type: str


# The classes/methods whose bodies carry declared-config validation. Mirrors the
# validator surface the retired per-version miners walked, ported to one table.
# Missing entries at a given pin are skipped with a warning (drift is the bump
# probe's job, not this proposer's); the gate filters anything spurious.
TARGETS: dict[str, tuple[Target, ...]] = {
    "vllm": (
        Target(
            "sampling_params.py",
            "SamplingParams",
            "_verify_args",
            "vllm.sampling",
            "vllm.SamplingParams",
        ),
        Target(
            "sampling_params.py",
            "SamplingParams",
            "__post_init__",
            "vllm.sampling",
            "vllm.SamplingParams",
        ),
        Target(
            "sampling_params.py",
            "SamplingParams",
            "_verify_greedy_sampling",
            "vllm.sampling",
            "vllm.SamplingParams",
        ),
        Target(
            "sampling_params.py",
            "StructuredOutputsParams",
            "__post_init__",
            "vllm.sampling",
            "vllm.sampling_params.StructuredOutputsParams",
        ),
        Target(
            "config/parallel.py",
            "ParallelConfig",
            "_validate_parallel_config",
            "vllm.engine",
            "vllm.config.ParallelConfig",
        ),
        Target(
            "config/parallel.py",
            "ParallelConfig",
            "_verify_args",
            "vllm.engine",
            "vllm.config.ParallelConfig",
        ),
        Target(
            "config/parallel.py",
            "ParallelConfig",
            "__post_init__",
            "vllm.engine",
            "vllm.config.ParallelConfig",
        ),
        Target(
            "config/parallel.py",
            "EPLBConfig",
            "_validate_eplb_config",
            "vllm.engine",
            "vllm.config.EPLBConfig",
        ),
        Target(
            "config/lora.py",
            "LoRAConfig",
            "_validate_lora_config",
            "vllm.engine",
            "vllm.config.LoRAConfig",
        ),
        Target(
            "config/multimodal.py",
            "MultiModalConfig",
            "_validate_multimodal_config",
            "vllm.engine",
            "vllm.config.MultiModalConfig",
        ),
        Target(
            "config/structured_outputs.py",
            "StructuredOutputsConfig",
            "_validate_structured_output_config",
            "vllm.engine",
            "vllm.config.StructuredOutputsConfig",
        ),
        Target(
            "config/cache.py",
            "CacheConfig",
            "_validate_cache_dtype",
            "vllm.engine",
            "vllm.config.CacheConfig",
        ),
        Target(
            "config/model.py",
            "ModelConfig",
            "__post_init__",
            "vllm.engine",
            "vllm.config.ModelConfig",
        ),
        Target(
            "config/compilation.py",
            "CompilationConfig",
            "__post_init__",
            "vllm.engine",
            "vllm.config.CompilationConfig",
        ),
        Target(
            "config/scheduler.py",
            "SchedulerConfig",
            "__post_init__",
            "vllm.engine",
            "vllm.config.SchedulerConfig",
        ),
    ),
    "transformers": (
        Target(
            "generation/configuration_utils.py",
            "GenerationConfig",
            "validate",
            "transformers.sampling",
            "transformers.GenerationConfig",
        ),
        Target(
            "generation/configuration_utils.py",
            "WatermarkingConfig",
            "__post_init__",
            "transformers.sampling.watermarking_config",
            "transformers.WatermarkingConfig",
        ),
        Target(
            "generation/configuration_utils.py",
            "SynthIDTextWatermarkingConfig",
            "validate",
            "transformers.sampling.watermarking_config",
            "transformers.SynthIDTextWatermarkingConfig",
        ),
        Target(
            "utils/quantization_config.py",
            "BitsAndBytesConfig",
            "post_init",
            "transformers.quant",
            "transformers.BitsAndBytesConfig",
        ),
    ),
    "tensorrt": (
        Target(
            "llmapi/llm_args.py",
            "BaseLlmArgs",
            "validate_dtype",
            "tensorrt",
            "tensorrt_llm.BaseLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "BaseLlmArgs",
            "validate_model",
            "tensorrt",
            "tensorrt_llm.BaseLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "BaseLlmArgs",
            "validate_lora_config_consistency",
            "tensorrt",
            "tensorrt_llm.BaseLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "TrtLlmArgs",
            "validate_enable_build_cache",
            "tensorrt",
            "tensorrt_llm.TrtLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "TrtLlmArgs",
            "validate_model_format_misc",
            "tensorrt",
            "tensorrt_llm.TrtLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "TrtLlmArgs",
            "validate_build_config_with_runtime_params",
            "tensorrt",
            "tensorrt_llm.TrtLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "TrtLlmArgs",
            "validate_build_config_remaining",
            "tensorrt",
            "tensorrt_llm.TrtLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "TrtLlmArgs",
            "validate_speculative_config",
            "tensorrt",
            "tensorrt_llm.TrtLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "TrtLlmArgs",
            "validate_kv_cache_dtype",
            "tensorrt",
            "tensorrt_llm.TrtLlmArgs",
        ),
        Target(
            "llmapi/llm_args.py",
            "LookaheadDecodingConfig",
            "validate_positive_values",
            "tensorrt",
            "tensorrt_llm.LookaheadDecodingConfig",
        ),
    ),
}


# ---------------------------------------------------------------------------
# AST primitives (self-contained: no dependency on the retired miner _base)
# ---------------------------------------------------------------------------


def find_class(module: ast.Module, class_name: str) -> ast.ClassDef | None:
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def find_method(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return item
    return None


def call_func_path(call: ast.Call) -> list[str] | None:
    """Dotted path for a Call's func (``logger.warning`` -> ``[logger, warning]``)."""
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _render_joinedstr(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            inner = value.value
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
            ):
                parts.append(f"{{{inner.attr}}}")
            else:
                parts.append(f"{{{ast.unparse(inner)}}}")
    return "".join(parts)


def first_string_arg(call: ast.Call) -> str | None:
    """First string-template positional arg of a Call, as a substitution template.

    Avoids leaking literal Python source: f-strings render to ``{name}``
    placeholders and ``"lit {x}".format(...)`` returns the literal LHS.
    """
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.JoinedStr):
            return _render_joinedstr(arg)
        if (
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "format"
            and isinstance(arg.func.value, ast.Constant)
            and isinstance(arg.func.value.value, str)
        ):
            return arg.func.value.value
    return None


# ---------------------------------------------------------------------------
# Predicate extraction (AST condition -> self.<field> op rhs predicates)
# ---------------------------------------------------------------------------


@dataclass
class _Predicate:
    """One ``self.<field>`` op rhs predicate distilled from an ``if`` condition.

    ``rhs`` is a literal or the cross-field reference ``"@<field>"`` when the
    comparand is another ``self`` attribute.
    """

    field: str
    op: str
    rhs: Any


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
        ok, value = _literal(node.operand)
        if ok and isinstance(value, (int, float)):
            return True, -value
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: list[Any] = []
        for elt in node.elts:
            ok, value = _literal(elt)
            if not ok:
                return False, None
            out.append(value)
        return True, out
    if isinstance(node, ast.Name) and node.id in {"True", "False", "None"}:
        return True, {"True": True, "False": False, "None": None}[node.id]
    return False, None


def _rhs_value(node: ast.expr) -> tuple[bool, Any]:
    ok, value = _literal(node)
    if ok:
        return True, value
    name = _self_attr(node)
    if name is not None:
        return True, f"@{name}"
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
    if path is None or path[-1] != "isinstance" or len(call.args) != 2:
        return []
    field_name = _self_attr(call.args[0])
    if field_name is None:
        return []
    names = _isinstance_type_names(call.args[1])
    if not names:
        return []
    rhs: Any = names[0] if len(names) == 1 else names
    return [_Predicate(field=field_name, op="type_is", rhs=rhs)]


def extract_predicates(condition: ast.expr) -> list[_Predicate]:
    """Translate an ``if`` condition into ``self.<field>`` predicates.

    AND-combined predicates all extract (they co-constrain the match); OR / opaque
    calls / non-self conditions drop silently (recall-first: the constraint still
    emits, minus the undecidable precondition).
    """
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.And):
        out: list[_Predicate] = []
        for value in condition.values:
            out.extend(extract_predicates(value))
        return out
    if isinstance(condition, ast.BoolOp) and isinstance(condition.op, ast.Or):
        return []
    if isinstance(condition, ast.UnaryOp) and isinstance(condition.op, ast.Not):
        inner = extract_predicates(condition.operand)
        if len(inner) == 1 and inner[0].op in _INVERSE_OPS:
            p = inner[0]
            return [_Predicate(field=p.field, op=_INVERSE_OPS[p.op], rhs=p.rhs)]
        return []
    if isinstance(condition, ast.Compare):
        return _extract_compare(condition)
    if isinstance(condition, ast.Call):
        return _extract_call(condition)
    field_name = _self_attr(condition)
    if field_name is not None:
        return [_Predicate(field=field_name, op="present", rhs=True)]
    return []


# ---------------------------------------------------------------------------
# Body detectors (raise -> error, self.x = y -> dormant normalisation)
# ---------------------------------------------------------------------------


@dataclass
class _Detected:
    severity: str  # "error" | "dormant"
    affected_field: str | None
    message_template: str | None


def _detect_raise(stmt: ast.stmt) -> _Detected | None:
    if not isinstance(stmt, ast.Raise) or stmt.exc is None:
        return None
    msg = first_string_arg(stmt.exc) if isinstance(stmt.exc, ast.Call) else None
    return _Detected(severity="error", affected_field=None, message_template=msg)


def _detect_self_assign(stmt: ast.stmt) -> _Detected | None:
    """``self.X = Y`` inside an ``if`` body: silent normalisation (dormant)."""
    if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    field_name = _self_attr(target)
    if field_name is None:
        return None
    return _Detected(severity="dormant", affected_field=field_name, message_template=None)


_DETECTORS = (_detect_raise, _detect_self_assign)


def _detect(stmt: ast.stmt) -> _Detected | None:
    for detector in _DETECTORS:
        result = detector(stmt)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# Match construction + candidate emission
# ---------------------------------------------------------------------------


def _build_match_fields(preds: list[_Predicate], namespace: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for p in preds:
        path = f"{namespace}.{p.field}"
        if path not in grouped:
            grouped[path] = {}
            order.append(path)
        grouped[path][p.op] = p.rhs
    out: dict[str, Any] = {}
    for path in order:
        spec = grouped[path]
        out[path] = spec["=="] if len(spec) == 1 and "==" in spec else spec
    return out


def _digest(engine: str, severity: str, match_fields: dict[str, Any]) -> str:
    """Stable short hash of the canonical claim, so re-proposals collapse on id."""
    claim = json.dumps(
        {"engine": engine, "severity": severity, "fields": match_fields},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha1(claim.encode()).hexdigest()[:8]


@dataclass
class _Emitter:
    engine: str
    version: str
    run_date: str
    rel_file: str
    target: Target
    seen_ids: set[str] = field(default_factory=set)
    candidates: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, preds: list[_Predicate], detected: _Detected, line: int) -> None:
        effective = list(preds)
        subject = detected.affected_field
        if subject is not None and not any(p.field == subject for p in effective):
            effective.append(_Predicate(field=subject, op="present", rhs=True))
        if not effective:
            return  # nothing to match on
        match_fields = _build_match_fields(effective, self.target.namespace)
        seen_fields: list[str] = []
        for p in effective:
            if p.field not in seen_fields:
                seen_fields.append(p.field)
        leaves = "_".join(seen_fields)
        digest = _digest(self.engine, detected.severity, match_fields)
        cid = f"{self.engine}_extractor_{detected.severity}_{slug(leaves)}_{digest}"
        if cid in self.seen_ids:
            return  # same claim reached twice in this method; keep the first
        self.seen_ids.add(cid)
        candidate: dict[str, Any] = {
            "id": cid,
            "engine": self.engine,
            "engine_version": self.version,
            "severity": detected.severity,
            "match": {"fields": match_fields},
        }
        if detected.severity == "dormant":
            candidate["normalised_fields"] = [subject] if subject is not None else []
        candidate["citation"] = {"file": self.rel_file, "lines": [line]}
        candidate["provenance"] = {
            "source": SOURCE,
            "engine_version": self.version,
            "date": self.run_date,
            "native_type": self.target.native_type,
            "method": self.target.method,
        }
        if detected.message_template:
            candidate["provenance"]["predicate"] = detected.message_template
        candidate["verdict"] = unverified_verdict()
        self.candidates.append(candidate)


def _walk_body(body: list[ast.stmt], frame: list[_Predicate], emitter: _Emitter) -> None:
    """Descend statements, accumulating enclosing ``if`` predicates in the frame."""
    for stmt in body:
        if isinstance(stmt, ast.If):
            _walk_if_chain(stmt, frame, emitter)
        elif isinstance(stmt, ast.For):
            _walk_body(stmt.body, frame, emitter)


def _walk_if_chain(if_node: ast.If, frame: list[_Predicate], emitter: _Emitter) -> None:
    node: ast.If | None = if_node
    while node is not None:
        preds = extract_predicates(node.test)
        local = [*frame, *preds]
        for stmt in node.body:
            detected = _detect(stmt)
            if detected is not None:
                emitter.emit(local, detected, getattr(stmt, "lineno", node.lineno))
            if isinstance(stmt, ast.If):
                _walk_if_chain(stmt, local, emitter)
            elif isinstance(stmt, ast.For):
                _walk_body(stmt.body, local, emitter)
        # Follow ``elif`` chains with the enclosing (not the if-branch) frame.
        if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
        else:
            if node.orelse:
                _walk_body(node.orelse, frame, emitter)
            node = None


def _walk_target(module: ast.Module, target: Target, rel_file: str, emitter: _Emitter) -> None:
    cls = find_class(module, target.cls)
    if cls is None:
        return
    method = find_method(cls, target.method)
    if method is None:
        return
    _walk_body(method.body, [], emitter)


def extract(engine: str, source_root: Path, version: str, run_date: str) -> list[dict[str, Any]]:
    """Walk the pinned source tree and return sorted, deduped pool candidates."""
    candidates: list[dict[str, Any]] = []
    ast_cache: dict[str, ast.Module] = {}
    for target in TARGETS.get(engine, ()):
        abs_path = source_root / target.rel_file
        if not abs_path.is_file():
            print(
                f"cross_field_extractor: {engine} source missing {target.rel_file}; skipping",
                file=sys.stderr,
            )
            continue
        module = ast_cache.get(target.rel_file)
        if module is None:
            module = ast.parse(abs_path.read_text())
            ast_cache[target.rel_file] = module
        emitter = _Emitter(
            engine=engine,
            version=version,
            run_date=run_date,
            rel_file=target.rel_file,
            target=target,
        )
        _walk_target(module, target, target.rel_file, emitter)
        candidates.extend(emitter.candidates)
    # Byte-stable output; collapse any claim two targets both reached.
    by_id: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_id.setdefault(str(candidate["id"]), candidate)
    return [by_id[cid] for cid in sorted(by_id)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=sorted(TARGETS))
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Engine top-level package dir at the pinned version (the citation root).",
    )
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=_PROJECT_ROOT / "engine_versions",
        help="Root of the version-scoped candidate pool.",
    )
    parser.add_argument("--date", help="Provenance date (default: today; set for reproducibility).")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.source_root.is_dir():
        print(f"error: --source-root is not a directory: {args.source_root}", file=sys.stderr)
        return 2
    engine = args.engine
    version = resolve_version(engine)
    run_date = args.date or date.today().isoformat()
    candidates = extract(engine, args.source_root, version, run_date)
    out = pool_path(args.pool_root, engine, version, "cross_field_extractor.yaml")
    write_pool(
        out,
        source=SOURCE,
        engine=engine,
        version=version,
        generated_at=run_date,
        candidates=candidates,
        header=_POOL_HEADER,
    )
    print(f"Wrote {len(candidates)} extractor candidates to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
