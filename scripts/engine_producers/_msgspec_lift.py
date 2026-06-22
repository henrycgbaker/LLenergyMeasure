"""msgspec ``Struct`` → ``InvariantCandidate[]`` lift.

Sub-library type-system lift consumed by per-engine miners. Walks
:class:`msgspec.Struct` subclasses via :func:`msgspec.inspect.type_info` and
emits one invariant candidate per ``Meta(ge=, le=, ...)`` constraint or
``Literal[...]`` allowlist.

Used by:

- vLLM ``SamplingParams`` (msgspec.Struct). Per
  ``research-vllm-extractor.md``, the live class ships **zero**
  ``Meta`` annotations - that's expected; this lift returns ``[]`` and the
  static miner picks up the slack via AST parsing of ``_verify_args``.
  The lift exists so the day vLLM (or another msgspec user) starts
  annotating ``Meta(ge=...)`` we capture it for free.

Determinism
-----------
No randomness, no library probing, no time-based seeds. Per
``feedback_miner_pipeline_deterministic.md``.
"""

from __future__ import annotations

import inspect
from typing import Any

import msgspec

from scripts.engine_producers._base import (
    InvariantCandidate,
    MinerSource,
    satisfies_numeric,
    slug,
    violates_numeric,
)

# msgspec maps annotated-types-style constraints onto these attribute names
# on its ``Constraints`` info object. We mirror the pydantic lift's op-key
# vocabulary so the corpus stays uniform across lifts.
_NUMERIC_OPS: tuple[tuple[str, str], ...] = (
    ("gt", ">"),
    ("ge", ">="),
    ("lt", "<"),
    ("le", "<="),
    ("multiple_of", "multiple_of"),
)

_LENGTH_OPS: tuple[tuple[str, str], ...] = (
    ("min_length", "min_len"),
    ("max_length", "max_len"),
)


def recover_field_types(target_type: type) -> dict[str, str]:
    """Recover ``{field: type_str}`` from a ``msgspec.Struct`` via type inspection.

    ``msgspec.json.schema`` drops to an untyped ``anyOf`` (no top-level ``type``
    key) for union / enum / nested-struct fields, so the schema introspector
    renders them ``"unknown"``. ``msgspec.inspect.type_info`` still resolves the
    concrete field type for those same fields, so this lift recovers them - the
    same inspection :func:`lift` already runs, exposed for the schema product
    (design 3: "wire ``_msgspec_lift`` into the sampling introspector to recover
    the ``Any | None`` sampling-param types").

    Returns ``{}`` for non-Struct types. Fields whose type is genuinely opaque
    (a bare ``Any``) are omitted so the caller keeps its existing label rather
    than overwriting ``unknown`` with another non-informative token.
    """
    if not (isinstance(target_type, type) and issubclass(target_type, msgspec.Struct)):
        return {}
    info = msgspec.inspect.type_info(target_type)
    if not isinstance(info, msgspec.inspect.StructType):
        return {}
    recovered: dict[str, str] = {}
    for field in info.fields:
        type_str = _render_type(field.type)
        if type_str is not None:
            recovered[field.name] = type_str
    return recovered


def _render_type(node: Any) -> str | None:
    """Render a ``msgspec.inspect`` type node to a compact type string, or None.

    Returns ``None`` only for a bare ``AnyType`` (no information to add). Unions
    render their members joined by `` | `` with ``None`` last, mirroring the
    introspector's existing ``X | None`` convention; a union that is *only*
    ``Any | None`` collapses to ``None`` (still no type information beyond
    nullability) so the caller does not overwrite ``unknown`` with ``Any | None``.
    """
    inspect_mod = msgspec.inspect
    if isinstance(node, inspect_mod.AnyType):
        return None
    if isinstance(node, inspect_mod.UnionType):
        members = [m for m in node.types if not isinstance(m, inspect_mod.NoneType)]
        has_none = len(members) < len(node.types)
        rendered = [_render_type(m) for m in members]
        parts = [r for r in rendered if r is not None]
        if not parts:
            return None
        if has_none:
            parts.append("None")
        return " | ".join(parts)
    return _render_scalar(node)


def _render_scalar(node: Any) -> str:
    """Render a non-union ``msgspec.inspect`` node to a type token."""
    inspect_mod = msgspec.inspect
    simple = {
        inspect_mod.NoneType: "None",
        inspect_mod.BoolType: "bool",
        inspect_mod.IntType: "int",
        inspect_mod.FloatType: "float",
        inspect_mod.StrType: "str",
        inspect_mod.BytesType: "bytes",
    }
    for cls, token in simple.items():
        if isinstance(node, cls):
            return token
    if isinstance(node, inspect_mod.ListType):
        return f"list[{_render_scalar_or_any(node.item_type)}]"
    if isinstance(node, inspect_mod.DictType):
        key = _render_scalar_or_any(node.key_type)
        val = _render_scalar_or_any(node.value_type)
        return f"dict[{key}, {val}]"
    if isinstance(node, inspect_mod.LiteralType):
        return f"Literal[{', '.join(repr(v) for v in node.values)}]"
    if isinstance(node, inspect_mod.EnumType):
        return node.cls.__name__
    if isinstance(node, (inspect_mod.StructType, inspect_mod.DataclassType)):
        return node.cls.__name__
    # Fall back to the node's own class name with the trailing "Type" stripped
    # (CustomType, DateTimeType, ...) so an unmapped node still carries a token.
    return type(node).__name__.removesuffix("Type").lower()


def _render_scalar_or_any(node: Any) -> str:
    """Like :func:`_render_scalar` but renders a bare Any item as ``Any``."""
    if isinstance(node, msgspec.inspect.AnyType):
        return "Any"
    rendered = _render_type(node)
    return rendered if rendered is not None else "Any"


def lift(
    target_type: type,
    *,
    namespace: str,
    today: str,
    source_path: str,
) -> list[InvariantCandidate]:
    """Extract validation-invariant candidates from a ``msgspec.Struct`` subclass.

    Parameters
    ----------
    target_type:
        A subclass of :class:`msgspec.Struct`. Other types yield an empty list.
    namespace:
        Engine field-namespace prefix used in ``match_fields``.
    today:
        ISO-8601 date string for ``added_at``.
    source_path:
        Source-file path recorded on each invariant's :class:`MinerSource`.

    Returns
    -------
    list[InvariantCandidate]
        One candidate per ``Meta(...)`` numeric/length constraint or per
        ``Literal[...]`` allowlist. Empty if the struct has no annotations.
    """
    if not (isinstance(target_type, type) and issubclass(target_type, msgspec.Struct)):
        return []

    library = target_type.__module__.split(".", 1)[0]
    type_name = target_type.__name__
    method = "<msgspec_lift>"
    try:
        line = inspect.getsourcelines(target_type)[1]
    except (OSError, TypeError):
        line = 0

    info = msgspec.inspect.type_info(target_type)
    if not isinstance(info, msgspec.inspect.StructType):
        return []

    candidates: list[InvariantCandidate] = []
    for field in info.fields:
        candidates.extend(
            _candidates_for_field(
                field,
                library=library,
                type_name=type_name,
                method=method,
                line=line,
                source_path=source_path,
                namespace=namespace,
                today=today,
            )
        )
    return candidates


def _candidates_for_field(
    field: msgspec.inspect.Field,
    *,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> list[InvariantCandidate]:
    """Emit candidates for one ``msgspec`` struct field's type info."""
    out: list[InvariantCandidate] = []
    field_type = field.type
    field_name = field.name

    # Numeric / length constraints sit on IntType, FloatType, StrType, etc.
    for attr, op_key in _NUMERIC_OPS:
        threshold = getattr(field_type, attr, None)
        if threshold is None:
            continue
        out.append(
            _build_numeric(
                field_name,
                op_key,
                threshold,
                library,
                type_name,
                method,
                line,
                source_path,
                namespace,
                today,
            )
        )

    for attr, op_key in _LENGTH_OPS:
        threshold = getattr(field_type, attr, None)
        if threshold is None:
            continue
        out.append(
            _build_length(
                field_name,
                op_key,
                threshold,
                library,
                type_name,
                method,
                line,
                source_path,
                namespace,
                today,
            )
        )

    # Literal allowlists - msgspec exposes these as LiteralType with .values.
    if isinstance(field_type, msgspec.inspect.LiteralType):
        out.append(
            _build_literal(
                field_name,
                field_type.values,
                library,
                type_name,
                method,
                line,
                source_path,
                namespace,
                today,
            )
        )

    return out


def _build_numeric(
    field_name: str,
    op_key: str,
    threshold: Any,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> InvariantCandidate:
    """Materialise a numeric-constraint invariant candidate."""
    return InvariantCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_{op_key.replace('>', 'gt').replace('<', 'lt').replace('=', 'e')}_{slug(threshold)}",
        engine=library,
        library=library,
        invariant_under_test=(f"{type_name}.{field_name} requires {op_key} {threshold!r}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {op_key: threshold}},
        kwargs_positive={field_name: violates_numeric(op_key, threshold)},
        kwargs_negative={field_name: satisfies_numeric(op_key, threshold)},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` must satisfy {op_key} {threshold!r}",
        references=[f"{library}.{type_name} - msgspec.Meta constraint"],
        added_by="msgspec_lift",
        added_at=today,
    )


def _build_length(
    field_name: str,
    op_key: str,
    threshold: int,
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> InvariantCandidate:
    """Materialise a length-constraint invariant candidate."""
    bad = "" if op_key == "min_len" else "x" * (threshold + 1)
    good = "x" * threshold
    return InvariantCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_{op_key}_{threshold}",
        engine=library,
        library=library,
        invariant_under_test=(f"{type_name}.{field_name} length must satisfy {op_key} {threshold}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {op_key: threshold}},
        kwargs_positive={field_name: bad},
        kwargs_negative={field_name: good},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` length must satisfy {op_key} {threshold}",
        references=[f"{library}.{type_name} - msgspec.Meta length constraint"],
        added_by="msgspec_lift",
        added_at=today,
    )


def _build_literal(
    field_name: str,
    values: tuple[Any, ...],
    library: str,
    type_name: str,
    method: str,
    line: int,
    source_path: str,
    namespace: str,
    today: str,
) -> InvariantCandidate:
    """Materialise a Literal-allowlist invariant candidate."""
    return InvariantCandidate(
        id=f"{library}_{type_name.lower()}_{field_name}_in_{len(values)}_values",
        engine=library,
        library=library,
        invariant_under_test=(f"{type_name}.{field_name} must be one of {list(values)!r}"),
        severity="error",
        native_type=f"{library}.{type_name}",
        miner_source=MinerSource(path=source_path, method=method, line_at_scan=line),
        match_fields={f"{namespace}.{field_name}": {"in": list(values)}},
        kwargs_positive={field_name: "<invalid_msgspec_lift_probe>"},
        kwargs_negative={field_name: values[0]},
        expected_outcome={
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        message_template=f"`{field_name}` must be one of {list(values)!r}",
        references=[f"{library}.{type_name} - msgspec Literal annotation"],
        added_by="msgspec_lift",
        added_at=today,
    )


__all__ = ["lift", "recover_field_types"]
