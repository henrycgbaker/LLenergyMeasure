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

2. **Class-surface enumeration** (carry-forward P5 + P7 + P8). A generalised
   subpackage glob, an AST ``ClassDef`` walk discovering config classes beyond
   entry-point reachability (sibling and nested classes), and module-level
   collection lifts (module-scope ``Literal`` aliases, enum globals, lookup-map
   allowlists). These surface the 20-80+ extra config classes per cell the
   study's ground truth found beyond flat entry-point introspection.

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
# Module-level collection lifts (P8)
# ---------------------------------------------------------------------------


def module_literal_aliases(module: ast.Module) -> dict[str, list[Any]]:
    """Module-scope ``Name = Literal[...]`` aliases mapped to their VALUES.

    TRT-LLM's plugin config declares e.g. ``DefaultPluginDtype = Literal[...]``
    and types many fields ``Optional[DefaultPluginDtype]``; resolving the alias
    to its values lets those fields emit a probeable membership allowlist (the
    plugin-literal fold). Only closed (all-constant) Literals are captured -
    an alias with a non-constant member yields no entry.
    """
    aliases: dict[str, list[Any]] = {}
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name):
            continue
        # The RHS must itself be a ``Literal[...]`` subscript - not merely contain
        # one buried in a larger expression - to count as a type alias.
        values = _literal_subscript_values(stmt.value)
        if values is not None:
            aliases[target.id] = values
    return aliases


def module_enum_globals(module: ast.Module) -> dict[str, list[Any]]:
    """Module-scope ``enum.Enum`` subclasses mapped to their member VALUES.

    Matches a ``ClassDef`` whose bases name an ``Enum`` family
    (``Enum`` / ``IntEnum`` / ``StrEnum`` / ``Flag`` / ``IntFlag``). Members are
    ``NAME = <constant>`` class-body assignments; the constant values become the
    membership allowlist. Auto-valued members (``auto()``) and non-constant
    members are skipped, so a partially-dynamic enum surfaces only its constants.
    """
    enum_bases = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
    enums: dict[str, list[Any]] = {}
    for node in ast.walk(module):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {_base_name(b) for b in node.bases}
        if not (base_names & enum_bases):
            continue
        values: list[Any] = []
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
                continue
            tgt = stmt.targets[0]
            if not isinstance(tgt, ast.Name) or tgt.id.startswith("_"):
                continue
            if isinstance(stmt.value, ast.Constant):
                values.append(stmt.value.value)
        if values:
            enums[node.name] = values
    return enums


def module_lookup_maps(module: ast.Module) -> dict[str, list[Any]]:
    """Module-scope dict-literal lookup maps mapped to their KEY sets.

    Captures allowlist-style constants such as vllm's
    ``STR_DTYPE_TO_TORCH_DTYPE = {"float16": ..., "bfloat16": ...}`` - the keys
    are the accepted string values for the field the map gates. Only string-keyed
    dict literals with at least two constant keys are captured (avoids flagging
    small config dicts); values are ignored (often opaque torch/runtime objects).
    """
    maps: dict[str, list[Any]] = {}
    for stmt in module.body:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if not isinstance(target, ast.Name) or not isinstance(stmt.value, ast.Dict):
            continue
        keys: list[Any] = []
        ok = True
        for key in stmt.value.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.append(key.value)
            else:
                ok = False
                break
        if ok and len(keys) >= 2:
            maps[target.id] = keys
    return maps


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
    literal_aliases: dict[str, list[Any]] | None = None,
    suffixes: tuple[str, ...] = ("Config", "Params", "Args"),
) -> dict[str, dict[str, dict[str, Any]]]:
    """Extract per-class, per-field declarative constraints as JSON Schema fragments.

    Returns ``{class_name: {field_name: {<jsonschema constraint keys>}}}``. Each
    field fragment may carry numeric bounds (``minimum`` / ``maximum`` /
    ``exclusiveMinimum`` / ``exclusiveMaximum`` / ``multipleOf`` / length / items
    bounds) lifted from a ``Field(...)`` / ``Meta(...)`` / ``conint(...)`` call,
    and/or an ``enum`` membership list lifted from a ``Literal[...]`` annotation,
    a module-level Literal alias, or - resolved by the caller - an enum global.

    These are *field-level* facts; the per-engine miner merges them onto the
    matching schema field. Cross-field facts (one field constrains another) are
    NOT produced here - those come from the imperative validator walkers in
    :mod:`scripts.engine_producers._base`, which already route to the invariant
    proposals. This keeps the declarative walker's output schema-shaped, matching
    how the live ``_pydantic_lift`` splits its work.
    """
    aliases = literal_aliases or {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for cls in iter_config_classes(module, suffixes=suffixes):
        fields = _class_field_constraints(cls, aliases)
        if fields:
            out[cls.name] = fields
    return out


def _class_field_constraints(
    cls: ast.ClassDef, aliases: dict[str, list[Any]]
) -> dict[str, dict[str, Any]]:
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
        # Membership: Literal[...] / alias / enum-typed annotation.
        members = _membership_values(stmt.annotation, aliases)
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


def _membership_values(annotation: ast.expr, aliases: dict[str, list[Any]]) -> list[Any] | None:
    """Allowed values when the annotation is a Literal / Literal-alias.

    Unwraps ``Optional[...]`` / ``X | None`` wrappers. Returns the value list,
    or ``None`` when the annotation is not a closed membership set. A
    module-level Literal alias (``DefaultPluginDtype``) resolves to its values.
    """
    direct = _literal_values(annotation)
    if direct is not None:
        return direct
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in aliases:
            return list(aliases[node.id])
    return None


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
    itself must be the ``Literal`` subscript (used for module-level alias RHS
    matching). ``None`` if not a Literal or any member is non-constant.
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


def _base_name(node: ast.expr) -> str:
    """Rightmost name of a class base (``enum.IntEnum`` -> ``"IntEnum"``)."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


__all__ = [
    "expand_files",
    "iter_config_classes",
    "module_enum_globals",
    "module_literal_aliases",
    "module_lookup_maps",
    "walk_declarative_constraints",
]
