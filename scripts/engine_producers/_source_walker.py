"""Source-text walkers for schema discovery.

The schema introspectors read a type's constraints off the *imported* class
(via runtime introspection). These walkers read the same constraints straight
from *source text*, so they stay reachable when a class carries its bounds and
enums declaratively in source that runtime introspection alone does not expose
(a pydantic ``Field(ge=...)`` keyword, a ``Literal[...]`` annotation on a
sub-config that has moved into a subpackage).

Two concerns live here, both pattern-matched (no file:line pinning - upstream
refactors move both):

1. **Declarative-constraint walk.** Extracts pydantic ``Field(ge/gt/le/lt/...)``
   keyword bounds, ``Literal[...]`` membership sets (the allowed values), and
   enum-typed fields from class-body annotated assignments. Field-level facts
   (``enum`` / ``minimum`` / ``maximum`` / ...) are returned as canonical JSON
   Schema fragments; the introspector folds them onto the matching schema field.

2. **Class-surface enumeration.** A subpackage glob plus an AST ``ClassDef``
   walk that discovers config classes beyond entry-point reachability (sibling
   and nested classes), so a config surface split across ``config/*.py`` modules
   is still reached.

All functions are pure (AST + text in, data out): no probing, no imports of the
target engine, no time-based seeds. Suitable for the host-side schema leg that
must stay affordable on every upstream bump.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Subpackage glob
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
# AST class enumeration
# ---------------------------------------------------------------------------


def iter_config_classes(
    module: ast.Module, *, suffixes: tuple[str, ...] = ("Config", "Params", "Args")
) -> list[ast.ClassDef]:
    """Return config-like ``ClassDef`` nodes anywhere in ``module``.

    Walks the whole module tree (not just top-level), so sibling classes the
    entry-point introspector never reaches AND nested classes are discovered.
    A class is config-like when its name ends in one of ``suffixes`` (the naming
    convention shared across all three engines) and is not private. Order is
    source order; duplicates by identity are not possible (one node each).
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
# Declarative-constraint walk
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

    These are *field-level* facts; the introspector merges them onto the matching
    schema field. The output stays schema-shaped by design (bounds and membership
    sets keyed by field name), which is all the schema product needs.
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
