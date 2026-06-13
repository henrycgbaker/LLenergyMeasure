"""Static source-level walkers for the deterministic mining floor.

This module is the AST (source-level) companion to the live-class lifts
(:mod:`scripts.engine_producers._pydantic_lift`,
:mod:`scripts.engine_producers._dataclass_lift`). The lifts read constraints
off *imported* classes; these walkers read the same constraints straight from
*source text* so they remain reachable when a class cannot be imported or
constructed (the engine isn't installed on the host, a sub-config moved to a
subpackage, construction needs unavailable runtime state).

Two concerns live here, both pattern-matched (no location-pinned landmarks -
the study found citation pinning survives none of the observed bumps):

1. **Declarative-constraint walker** (carry-forward P3 + P4, study Primitive 8).
   Extracts pydantic ``Field(ge/gt/le/lt/...)`` keyword bounds, ``Literal[...]``
   membership sets (capturing the allowed *values*), and enum-typed fields from
   class-body annotated assignments. Field-level facts (``enum`` / ``minimum`` /
   ``maximum`` / ...) are returned as canonical JSON Schema fragments for the
   schema; the per-engine miner routes them.

2. **Class-surface enumeration** (carry-forward P5 + P7). A generalised
   subpackage glob and an AST ``ClassDef`` walk discovering config classes beyond
   entry-point reachability (sibling and nested classes). These surface the
   20-80+ extra config classes per cell the study's ground truth found beyond
   flat entry-point introspection.

All functions are pure (AST + text in, data out): no probing, no imports of the
target engine, no time-based seeds. Suitable for CI floors that must stay
affordable on every upstream bump.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Generalised subpackage glob (P5)
# ---------------------------------------------------------------------------


def expand_files(source_root: Path, patterns: list[str]) -> list[Path]:
    """Expand path patterns (literal or glob) against ``source_root``.

    De-duped and order-preserving. Globbing de-pins the file list so subpackage
    refactors (vllm ``config.py`` -> ``config/*.py``, a ``plugin/`` fan-out) are
    picked up across bumps WITHOUT editing the pattern list. Patterns that match
    nothing (the surface does not exist on this version - vllm 0.7.3 has no
    ``config/`` subpackage) simply contribute no files; the walker degrades
    gracefully rather than failing.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        matches = sorted(source_root.glob(pattern)) if "*" in pattern else [source_root / pattern]
        for path in matches:
            if path.is_file() and path not in seen:
                seen.add(path)
                out.append(path)
    return out


# ---------------------------------------------------------------------------
# AST class enumeration (P7)
# ---------------------------------------------------------------------------


def iter_config_classes(
    module: ast.Module, *, suffixes: tuple[str, ...] = ("Config", "Params", "Args")
) -> list[ast.ClassDef]:
    """Return config-like ``ClassDef`` nodes anywhere in ``module``.

    Walks the whole module tree (not just top-level), so sibling classes the
    entry-point introspector never reaches AND nested classes are discovered.
    A class is config-like when its name ends in one of ``suffixes`` (the study
    GT's naming convention across all three engines) and is not private. Order
    is source order; duplicates by identity are not possible (one node each).
    """
    classes: list[ast.ClassDef] = []
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name.startswith("_"):
            continue
        if node.name.endswith(suffixes):
            classes.append(node)
    return classes


# ---------------------------------------------------------------------------
# Declarative-constraint walker (P3 + P4, study Primitive 8)
# ---------------------------------------------------------------------------

# pydantic / annotated-types numeric keyword -> canonical JSON Schema key.
# exclusive variants follow JSON Schema 2020-12 (a numeric value, not a bool).
_BOUND_TO_JSONSCHEMA: dict[str, str] = {
    "ge": "minimum",
    "gt": "exclusiveMinimum",
    "le": "maximum",
    "lt": "exclusiveMaximum",
    "multiple_of": "multipleOf",
    "min_length": "minLength",
    "max_length": "maxLength",
    "min_items": "minItems",
    "max_items": "maxItems",
}

_FIELD_LIKE_CALLS = {"Field", "Meta", "conint", "confloat", "PositiveInt", "PositiveFloat"}


def walk_declarative_constraints(
    module: ast.Module,
    *,
    suffixes: tuple[str, ...] = ("Config", "Params", "Args"),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract per-class, per-field declarative constraints as JSON Schema fragments.

    Returns ``{class_name: {field_name: {<jsonschema constraint keys>}}}``. Each
    field fragment may carry numeric bounds (``minimum`` / ``maximum`` /
    ``exclusiveMinimum`` / ``exclusiveMaximum`` / ``multipleOf`` / length / items
    bounds) lifted from a ``Field(...)`` / ``Meta(...)`` / ``conint(...)`` call,
    and/or an ``enum`` membership list lifted from a ``Literal[...]`` annotation.

    These are *field-level* facts; the per-engine miner merges them onto the
    matching schema field. Cross-field facts (one field constrains another) are
    NOT produced here - those come from the imperative validator walkers in
    :mod:`scripts.engine_producers._base`, which already route to the invariant
    proposals. This keeps the declarative walker's output schema-shaped, matching
    how the live ``_pydantic_lift`` splits its work.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cls in iter_config_classes(module, suffixes=suffixes):
        fields = _class_field_constraints(cls)
        if fields:
            out[cls.name] = fields
    return out


def _class_field_constraints(cls: ast.ClassDef) -> dict[str, dict[str, Any]]:
    fields: dict[str, dict[str, Any]] = {}
    for stmt in cls.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name.startswith("_"):
            continue
        fragment: dict[str, Any] = {}
        # Numeric / length bounds from a Field(...) RHS or an Annotated[...] call.
        bounds = _bounds_in_annotation(stmt.annotation)
        if isinstance(stmt.value, ast.Call):
            bounds.update(_bounds_in_call(stmt.value))
        for key, value in bounds.items():
            fragment.setdefault(key, value)
        # Membership: Literal[...] / Optional[Literal[...]] annotation.
        members = _membership_values(stmt.annotation)
        if members:
            fragment["enum"] = members
        if fragment:
            fields[name] = fragment
    return fields


def _bounds_in_call(call: ast.Call) -> dict[str, Any]:
    """Numeric/length bounds from a ``Field(...)``-like call's keyword arguments."""
    if not _is_field_like(call):
        return {}
    bounds: dict[str, Any] = {}
    head = _call_head(call)
    if head in {"PositiveInt", "PositiveFloat"}:
        bounds["exclusiveMinimum"] = 0
    for kw in call.keywords:
        if kw.arg is None:
            continue
        json_key = _BOUND_TO_JSONSCHEMA.get(kw.arg)
        if json_key is None:
            continue
        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, (int, float)):
            bounds[json_key] = kw.value.value
    return bounds


def _bounds_in_annotation(annotation: ast.expr) -> dict[str, Any]:
    """Bounds from ``Annotated[T, Field(...)]`` / ``conint(...)`` inside an annotation."""
    bounds: dict[str, Any] = {}
    for node in ast.walk(annotation):
        if isinstance(node, ast.Call) and _is_field_like(node):
            bounds.update(_bounds_in_call(node))
    return bounds


def _membership_values(annotation: ast.expr) -> list[Any] | None:
    """Allowed values when the annotation is (or wraps) a ``Literal[...]``.

    Unwraps ``Optional[...]`` / ``X | None`` wrappers. Returns the value list,
    or ``None`` when the annotation is not a closed membership set.
    """
    return _literal_values(annotation)


# ---------------------------------------------------------------------------
# Small AST helpers
# ---------------------------------------------------------------------------


def _literal_values(node: ast.expr) -> list[Any] | None:
    """Closed value set if ``node`` is (or wraps) a ``Literal[...]``, else None.

    Handles bare ``Literal[a, b]``, ``Optional[Literal[...]]``, ``Literal[...] |
    None``, and ``Annotated[Literal[...], ...]`` by descending into the
    annotation tree. Returns ``None`` when there is no Literal; an empty list is
    never returned (a Literal with a non-constant member -> not-closed -> ``None``).
    """
    for sub in ast.walk(node):
        if isinstance(sub, ast.Subscript):
            values = _literal_subscript_values(sub)
            if values is not None:
                return values
    return None


def _literal_subscript_values(node: ast.expr) -> list[Any] | None:
    """Closed value set if ``node`` is *directly* a ``Literal[...]`` subscript.

    Unlike :func:`_literal_values` this does not descend into wrappers - the node
    itself must be the ``Literal`` subscript. ``None`` if not a Literal or any
    member is non-constant.
    """
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "Literal"
    ):
        return None
    elts = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    values: list[Any] = []
    for elt in elts:
        if isinstance(elt, ast.Constant):
            values.append(elt.value)
        else:
            return None
    return values


def _is_field_like(call: ast.Call) -> bool:
    return _call_head(call) in _FIELD_LIKE_CALLS


def _call_head(call: ast.Call) -> str:
    """Rightmost name of a call's callee (``pydantic.Field`` -> ``"Field"``)."""
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


__all__ = [
    "expand_files",
    "iter_config_classes",
    "walk_declarative_constraints",
]
