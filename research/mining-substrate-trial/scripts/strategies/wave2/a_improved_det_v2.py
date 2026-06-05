"""Wave 2.5 Tier A improved-det V2 = improved-det's 7 primitives + Primitive 8.

Primitive 8 is the declarative-constraint extractor. It is the trial's #1
engineering recommendation under test (Wave 2.1 bump-survivability deliverable,
DECISIONS_LOG 0.2): the static substrate's invariant recall vs ground truth
COLLAPSES across the vllm v0.7.3 -> v0.19.1 bump (improved-det tolerant
0.513 -> 0.147) because v0.19.1 moved many invariants OUT of imperative
``if x: raise`` bodies and INTO declarative pydantic ``Field(ge=, gt=, le=,
lt=, ...)`` / ``Literal[...]`` constraints and per-platform checks that the
grep/AST-for-raise primitives (1-7) never parse.

Primitive 8 extracts, in the canonical ``match.fields`` shape that the
tolerant GT matcher keys on (``leaf_field`` + coarse predicate bucket):

* pydantic field constraints - ``x: T = Field(ge=N, gt=N, le=N, lt=N,
  multiple_of=N, min_length=N, max_length=N)`` -> one numeric/range invariant
  per bound (predicate_kind ge/gt/le/lt; length bounds map to ge/le on the
  field). Also ``Annotated[int, Field(gt=0)]`` and ``conint(ge=...)`` /
  ``confloat(...)`` forms.
* Literal / enum field types - ``x: Literal["a","b"]`` or ``x: SomeEnum`` ->
  a membership (not_in) invariant: the value must lie in the allowed set.
* msgspec ``Meta`` - ``Annotated[int, Meta(ge=...)]`` (vllm SamplingParams /
  per-field msgspec constraints) - same numeric forms.

pydantic v2 ``@field_validator`` / ``@model_validator`` bodies are ALREADY
covered by Primitive 6; Primitive 8 dedupes against the prior 7-primitive pass
by ``(leaf_field, coarse_predicate_bucket)`` identity so a field constrained
BOTH declaratively and in a validator is not double-counted.

This module does NOT mutate the validated improved-det. It imports its tree-
sitter infra + emission helpers and runs Primitive 8 as an additional pass,
appending to the same invariants envelope.

Tasks mirror a_improved_det: ``schema`` | ``invariants`` | ``both`` (default).
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import yaml

from strategies.wave2 import a_improved_det as base
from strategies.wave2._base import CellOutput, standard_out_dir

_STRATEGY_ID = "w2-a-improved-det-v2"
_ADDED_BY = "wave2_a_improved_det_v2_p8"


# ---------------------------------------------------------------------------
# Primitive 8 file targets. Declarative constraints concentrate in the config
# subpackage (vllm v0.19.1 split the flat config.py into config/*.py) and the
# sampling/llm-args modules. We discover files via globs against the resolved
# source root so the same config works for the flat (v0.7.3) and split
# (v0.19.1) layouts.
# ---------------------------------------------------------------------------


_PRIMITIVE8_GLOBS: dict[str, list[str]] = {
    "vllm": [
        "config.py",
        "config/*.py",
        "sampling_params.py",
        "pooling_params.py",
        "engine/arg_utils.py",
    ],
    "transformers": [
        "generation/configuration_utils.py",
        "utils/quantization_config.py",
    ],
    "tensorrt": [
        "llmapi/llm_args.py",
        "llmapi/build_cache.py",
        "sampling_params.py",
    ],
}


def _primitive8_files(engine: str, source_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in _PRIMITIVE8_GLOBS.get(engine, []):
        if "*" in pattern:
            for p in sorted(source_root.glob(pattern)):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    out.append(p)
        else:
            p = source_root / pattern
            if p.is_file() and p not in seen:
                seen.add(p)
                out.append(p)
    return out


# ---------------------------------------------------------------------------
# Declarative-constraint parsing (tree-sitter class-body annotated assignments).
# ---------------------------------------------------------------------------


# Numeric bound kw -> scorer predicate kind. Length bounds collapse onto the
# same field as ge/le (a minimum/maximum on the field's measured size); the
# tolerant matcher buckets all of ge/gt/le/lt as "numeric" so the exact choice
# only affects the strict score, where ge/le is the honest reading of a
# length bound.
_BOUND_KIND = {
    "ge": "ge",
    "gt": "gt",
    "le": "le",
    "lt": "lt",
    "multiple_of": "ge",  # divisibility is a numeric constraint; coarse-bucket numeric
    "min_length": "ge",
    "max_length": "le",
    "min_items": "ge",
    "max_items": "le",
}

# Match a single kw=value inside a Field(...)/Meta(...)/conint(...) call. Value
# can be a number, a negative number, or a simple name (e.g. a module constant).
_KW_RE = re.compile(r"\b(ge|gt|le|lt|multiple_of|min_length|max_length|min_items|max_items)\s*=")


def _is_field_like_call(call_text: str) -> bool:
    head = call_text.split("(", 1)[0].strip()
    leaf = head.rpartition(".")[2]
    return leaf in {"Field", "Meta", "conint", "confloat", "PositiveInt", "PositiveFloat"}


def _bounds_in_call(call_text: str) -> list[str]:
    """Return the list of bound-kind keys present in a Field/Meta/conint call."""
    if not _is_field_like_call(call_text):
        return []
    kinds: list[str] = []
    for m in _KW_RE.finditer(call_text):
        kw = m.group(1)
        kind = _BOUND_KIND.get(kw)
        if kind is not None and kind not in kinds:
            kinds.append(kind)
    # conint/confloat with no explicit kw but PositiveInt/PositiveFloat alias.
    head = call_text.split("(", 1)[0].strip().rpartition(".")[2]
    if head in {"PositiveInt", "PositiveFloat"} and "gt" not in kinds:
        kinds.append("gt")
    return kinds


def _annotation_constraints(type_text: str) -> tuple[list[str], bool]:
    """Inspect an annotation string for declarative constraints.

    Returns ``(numeric_kinds, is_membership)``:
    * numeric_kinds - bound kinds from any Field/Meta/conint/confloat call
      nested in an ``Annotated[...]`` (or a bare ``conint(...)`` annotation).
    * is_membership - True when the annotation is a ``Literal[...]`` or a
      bare enum-like name (CamelCase, not a builtin/typing container), i.e.
      the value is constrained to a closed allowed set.
    """
    text = type_text.strip()
    numeric: list[str] = []
    # Annotated[T, Field(...)] / Annotated[int, Meta(ge=...)]
    if "Annotated[" in text or "conint(" in text or "confloat(" in text:
        # Pull each Field(...)/Meta(...)/conint(...)/confloat(...) call body.
        for m in re.finditer(r"(Field|Meta|conint|confloat)\s*\(", text):
            start = m.start()
            depth = 0
            i = m.end() - 1
            for j in range(i, len(text)):
                if text[j] == "(":
                    depth += 1
                elif text[j] == ")":
                    depth -= 1
                    if depth == 0:
                        call_text = text[start : j + 1]
                        numeric.extend(k for k in _bounds_in_call(call_text) if k not in numeric)
                        break
    is_membership = False
    # Strip a leading Optional / | None so `Literal[...] | None` still reads.
    head_text = text
    if "Literal[" in head_text:
        is_membership = True
    else:
        # Bare enum-like annotation: a single CamelCase name (optionally
        # Optional-wrapped) that is not a typing container / builtin. This is
        # a heuristic; the tolerant matcher only needs the leaf+bucket, so a
        # false positive costs precision, not a crash. We are conservative:
        # require the WHOLE (None-stripped) annotation to be one bare name
        # ending in a config-enum suffix to avoid flagging every typed field.
        stripped = head_text.replace("| None", "").replace("Optional[", "").rstrip("]").strip()
        if re.fullmatch(r"[A-Z][A-Za-z0-9]*", stripped or ""):
            if stripped.endswith(("Mode", "Type", "Backend", "Policy", "Method", "Format", "Role")):
                is_membership = True
    return numeric, is_membership


def _class_level_fields(source: bytes, class_node: Any) -> list[tuple[str, str, str | None]]:
    """Return [(field_name, annotation_text, rhs_call_text|None)] for every
    annotated class-body assignment ``name: TYPE`` or ``name: TYPE = RHS``.

    RHS call text is the dotted call head + args when the RHS is a call (so we
    can spot ``= Field(...)``); None otherwise.
    """
    out: list[tuple[str, str, str | None]] = []
    body = base._class_body(class_node)
    if body is None:
        return out
    for stmt in body.named_children:
        if stmt.type != "expression_statement":
            continue
        for child in stmt.named_children:
            if child.type != "assignment":
                continue
            name_node = child.child_by_field_name("left")
            type_node = child.child_by_field_name("type")
            right_node = child.child_by_field_name("right")
            if name_node is None or name_node.type != "identifier" or type_node is None:
                continue
            fname = base._node_text(source, name_node)
            if fname.startswith("_"):
                continue
            type_text = base._node_text(source, type_node)
            rhs = None
            if right_node is not None and right_node.type == "call":
                rhs = base._node_text(source, right_node)
            out.append((fname, type_text, rhs))
    return out


def _primitive8_invariants(
    *,
    engine: str,
    cfg: base._EngineCfg,
    source: bytes,
    rel_path: str,
    parser: Any,
) -> list[dict[str, Any]]:
    """Run Primitive 8 over one source file: emit a declarative invariant per
    (field, bound) and per (Literal/enum field) membership."""
    root = parser.parse(source).root_node
    invs: list[dict[str, Any]] = []
    for class_node in base._class_nodes(root):
        cname = base._class_name(source, class_node)
        if cname is None or cname.startswith("_"):
            continue
        namespace = base._resolve_namespace(cname, cfg)
        for fname, type_text, rhs in _class_level_fields(source, class_node):
            numeric_kinds: list[str] = []
            is_membership = False
            # RHS Field(...)/conint(...) call bounds.
            if rhs is not None:
                numeric_kinds.extend(_bounds_in_call(rhs))
            # Annotation-side constraints (Annotated[...]/conint(...)/Literal).
            ann_numeric, ann_member = _annotation_constraints(type_text)
            for k in ann_numeric:
                if k not in numeric_kinds:
                    numeric_kinds.append(k)
            is_membership = ann_member
            line = class_node.start_point[0] + 1
            for kind in numeric_kinds:
                invs.append(
                    base._make_invariant(
                        engine=engine,
                        class_name=cname,
                        method_name="__declarative__",
                        namespace=namespace,
                        primary_field=fname,
                        kind=kind,
                        secondary_field="",
                        rel_path=rel_path,
                        line=line,
                        severity="error",
                        fired_by="p8",
                    )
                )
            if is_membership:
                invs.append(
                    base._make_invariant(
                        engine=engine,
                        class_name=cname,
                        method_name="__declarative__",
                        namespace=namespace,
                        primary_field=fname,
                        kind="not_in",
                        secondary_field="",
                        rel_path=rel_path,
                        line=line,
                        severity="error",
                        fired_by="p8",
                    )
                )
    return invs


def _coarse_bucket(kind: str) -> str:
    if kind in {"ge", "gt", "le", "lt"}:
        return "numeric"
    if kind in {"not_in", "not_equal"}:
        return "membership"
    if kind == "type_is_not":
        return "type"
    if kind == "exact":
        return "exact"
    return "presence"


def _tolerant_identity(inv: dict[str, Any]) -> tuple[str, str]:
    """(leaf_field, coarse_bucket) - the key the GT tolerant matcher uses.

    Dedupes Primitive 8 against the prior 7-primitive pass so a field
    constrained both declaratively and in a validator counts once.
    """
    fields = (inv.get("match") or {}).get("fields") or {}
    if not fields:
        return ("", "presence")
    primary_key = next(iter(fields))
    leaf = primary_key.rpartition(".")[2]
    value = fields[primary_key]
    kind = "exact"
    if isinstance(value, dict):
        keys = set(value.keys())
        if "not_in" in keys:
            kind = "not_in"
        elif ">" in keys:
            kind = "gt"
        elif "<" in keys:
            kind = "lt"
        elif ">=" in keys:
            kind = "ge"
        elif "<=" in keys:
            kind = "le"
        elif "type_is_not" in keys:
            kind = "type_is_not"
        elif "present" in keys:
            kind = "present"
    return (leaf, _coarse_bucket(kind))


def run_cell(
    *,
    engine: str,
    version: str,
    engine_source_root: Path | None = None,
    task: str = "both",
) -> CellOutput:
    """Execute one Wave 2.5 improved-det-v2 cell: the validated 7-primitive
    pass plus Primitive 8 (declarative constraints), scored as one envelope."""
    if engine not in base._ENGINE_CFG:
        raise NotImplementedError(
            f"a_improved_det_v2 config missing for engine={engine!r}; "
            f"known: {sorted(base._ENGINE_CFG)}"
        )
    if task not in {"schema", "invariants", "both"}:
        raise ValueError(f"task must be schema|invariants|both; got {task!r}")

    # Run the unmodified 7-primitive pass first (writes its own out dir; we
    # re-emit under the v2 strategy id below).
    inner = base.run_cell(
        engine=engine,
        version=version,
        engine_source_root=engine_source_root,
        task=task,
    )

    cfg = base._ENGINE_CFG[engine]
    version_slug = (
        version
        if version.startswith("v") and "." not in version
        else f"v{version.lstrip('v').replace('.', '_')}"
    )
    source_root = base._resolve_source_root(engine, version_slug, engine_source_root)
    out_dir = standard_out_dir(_STRATEGY_ID, engine, version_slug)

    observations = list(inner.observations)
    wall = dict(inner.wall_sec)
    extra = dict(inner.extra)

    # Load the 7-primitive invariants envelope back from disk (inner wrote it).
    inv_env: dict[str, Any] = {}
    if inner.invariants_path and inner.invariants_path.exists():
        inv_env = yaml.safe_load(inner.invariants_path.read_text()) or {}
    base_invs: list[dict[str, Any]] = list(inv_env.get("invariants", []))

    # Primitive 8 pass.
    p8_added = 0
    p8_skipped_dup = 0
    if task in {"invariants", "both"} and source_root is not None:
        t = time.perf_counter()
        _language, parser = base._load_parser()
        existing_keys = {_tolerant_identity(inv) for inv in base_invs}
        seen_p8: set[tuple[str, str]] = set()
        for path in _primitive8_files(engine, source_root):
            src = base._read_source(path)
            if not src:
                continue
            rel = str(path.relative_to(source_root))
            try:
                cand = _primitive8_invariants(
                    engine=engine, cfg=cfg, source=src, rel_path=rel, parser=parser
                )
            except Exception as exc:  # degrade gracefully; never crash the cell
                observations.append(f"primitive8: parse-degraded on {rel}: {exc}")
                continue
            for inv in cand:
                ident = _tolerant_identity(inv)
                if ident in existing_keys or ident in seen_p8:
                    p8_skipped_dup += 1
                    continue
                seen_p8.add(ident)
                inv["added_by"] = _ADDED_BY
                base_invs.append(inv)
                p8_added += 1
        wall["primitive8"] = time.perf_counter() - t
        if p8_added:
            fired = extra.get("primitives_fired", [])
            if "p8" not in fired:
                fired = sorted([*fired, "p8"])
            extra["primitives_fired"] = fired
    extra["p8_added"] = p8_added
    extra["p8_skipped_dup"] = p8_skipped_dup

    inv_env["invariants"] = base_invs

    # Re-emit both artefacts under the v2 strategy id (schema is unchanged from
    # the inner pass; copy it through so scoring has both files co-located).
    schema_path = out_dir / "schema.json"
    invariants_path = out_dir / "invariants.proposed.yaml"
    if inner.schema_path and inner.schema_path.exists():
        schema_path.write_text(inner.schema_path.read_text())
    invariants_path.write_text(yaml.safe_dump(inv_env, sort_keys=False))

    extra["strategy_id"] = _STRATEGY_ID
    extra["invariant_count"] = len(base_invs)

    return CellOutput(
        strategy_id=_STRATEGY_ID,
        engine=engine,
        engine_version=version,
        schema_path=schema_path,
        invariants_path=invariants_path,
        observations=observations,
        wall_sec=wall,
        extra=extra,
    )


__all__ = ["run_cell"]
