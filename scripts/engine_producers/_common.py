"""Shared helpers for per-engine schema introspectors.

Introspectors run inside an environment where the target engine package is
installed (typically a Docker container). For each engine, they introspect the
native Python API surface and write a JSON schema file with a common envelope.
The envelope is versioned separately from the engines (see
:data:`SCHEMA_VERSION`); major bumps are breaking and SchemaLoader rejects
them. Minor bumps add envelope keys; downstream loaders are expected to be
forward-compatible.

This module is the schema-discovery substrate: runtime-introspection helpers
(``dataclass_fields_to_specs``, ``recover_field_types``, ``make_envelope``) plus
the fold that overlays source-text bounds and enums (via
:mod:`scripts.engine_producers._source_walker`) onto discovered fields. It
imports no engine library and no validation-mining machinery.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import os
import re
import types
import typing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Union, get_args, get_origin

from scripts.engine_producers._source_walker import walk_declarative_constraints

SCHEMA_VERSION = "1.0.0"

# Declarative-constraint keys the source walker emits (JSON Schema fragments)
# that the schema introspectors fold onto discovered fields. Type / default
# come from runtime introspection; these bounds + membership sets come from the
# source-text walk (Field(ge/le/...) and Literal[...] annotations).
_CONSTRAINT_KEYS: tuple[str, ...] = (
    "enum",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
)

# Only the transformers engine has a first-party Dockerfile. vllm and
# tensorrt run inside upstream images (vllm/vllm-openai,
# nvcr.io/nvidia/tensorrt-llm/release) that introspectors receive via
# the workflow-supplied ``--image-ref`` flag rather than reading from a
# local Dockerfile.
TRANSFORMERS_DOCKERFILE = "docker/Dockerfile.transformers"

DEFAULT_OUTPUT_DIR = "src/llenergymeasure/engines"
DEFAULT_SCHEMA_FILENAME = "schema.discovered.json"


def annotation_to_type_str(annotation: Any) -> str:
    """Render a type annotation as a compact readable string.

    Handles None, Optional[X], X | None, Union, Literal, generics, forward
    refs, and inspect.Parameter.empty. Falls back to str(annotation) for
    anything unrecognised so discovery never raises on exotic types.
    """
    if annotation is type(None):
        return "None"
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        return "unknown"

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        return getattr(annotation, "__name__", str(annotation))

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        parts = [annotation_to_type_str(a) for a in non_none]
        if has_none:
            parts.append("None")
        return " | ".join(parts)

    origin_str = str(origin)
    if "Literal" in origin_str:
        vals = ", ".join(repr(a) for a in args)
        return f"Literal[{vals}]"

    origin_name = getattr(origin, "__name__", origin_str)
    arg_strs = ", ".join(annotation_to_type_str(a) for a in args)
    return f"{origin_name}[{arg_strs}]"


# A HuggingFace-style docstring ``Args:`` entry: ``name (`type`, *optional*, ...):``.
# The first backticked token after the field name is the documented type. HF uses
# this convention universally; for classes that ship no real field annotations
# (GenerationConfig is ``__init__(self, **kwargs)`` with every default ``None``),
# the docstring is the only machine-readable type source. ``or`` unions
# (``bool` or `str``) take the first member, matching the value-inference baseline
# the older pins produced when defaults were still concrete.
_DOCSTRING_ARG_TYPE = re.compile(r"^\s*(\w+)\s*\(\s*`([A-Za-z_][\w.]*)`")

# Documented docstring type tokens mapped to the introspector's scalar vocabulary
# (the same surface ``type(value).__name__`` emits for concrete defaults). Tokens
# outside this set (``torch.dtype``, ``Dict``, ...) are left unmapped so the
# field falls through to default-inference rather than inventing a scalar type.
_DOC_TYPE_TO_SCALAR: dict[str, str] = {
    "bool": "bool",
    "int": "int",
    "float": "float",
    "str": "str",
}


def docstring_arg_types(obj: Any) -> dict[str, str]:
    """Map ``{field: scalar_type}`` from an object's docstring ``Args:`` block.

    Reads HuggingFace's documented-arg convention as a type-annotation source for
    classes that carry no real field annotations. Only scalar tokens in
    :data:`_DOC_TYPE_TO_SCALAR` are returned; non-scalar or undocumented fields are
    omitted so the caller can fall back to default-value inference. Returns an empty
    dict when the object has no docstring.
    """
    doc = inspect.getdoc(obj) or ""
    out: dict[str, str] = {}
    for line in doc.splitlines():
        match = _DOCSTRING_ARG_TYPE.match(line)
        if match is None:
            continue
        scalar = _DOC_TYPE_TO_SCALAR.get(match.group(2))
        if scalar is not None:
            out.setdefault(match.group(1), scalar)
    return out


def read_dockerfile_from(dockerfile: Path) -> str:
    """Extract the FROM tag from a Dockerfile, expanding the default ARG value.

    For multi-stage Dockerfiles, prefers the ``AS runtime`` stage (convention
    used by all llenergymeasure Dockerfiles). Falls back to the first FROM
    line that references an external image (not a prior stage name). Only
    default ARG values are substituted - no environment overrides.

    Returns e.g. ``"vllm/vllm-openai:v0.7.3"`` for a Dockerfile with
    ``ARG VLLM_VERSION=v0.7.3`` and ``FROM vllm/vllm-openai:${VLLM_VERSION}``.
    """
    text = dockerfile.read_text()
    arg_defaults: dict[str, str] = {}
    from_lines: list[tuple[str, str | None]] = []  # (ref, stage_alias)

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("ARG "):
            m = re.match(r"ARG\s+(\w+)(?:=(.+))?", stripped)
            if m:
                arg_defaults[m.group(1)] = (m.group(2) or "").strip()
            continue
        if stripped.startswith("FROM "):
            m = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?", stripped, re.IGNORECASE)
            if m:
                from_lines.append((m.group(1), m.group(2)))

    if not from_lines:
        raise ValueError(f"No FROM directive found in {dockerfile}")

    stage_names = {alias for _, alias in from_lines if alias}

    def _expand(ref: str) -> str:
        return re.sub(
            r"\$\{(\w+)\}",
            lambda match: arg_defaults.get(match.group(1), match.group(0)),
            ref,
        )

    for ref, alias in from_lines:
        if alias == "runtime":
            return _expand(ref)

    for ref, _ in from_lines:
        if ref not in stage_names:
            return _expand(ref)

    # All FROM lines reference prior stages - shouldn't happen in a valid Dockerfile
    return _expand(from_lines[0][0])


def _sorted_stable(items: list[Any]) -> list[Any]:
    """Sort *items* deterministically.

    Falls back to a type-stable key when the elements are not mutually orderable
    (e.g. opaque values sanitised to ``None`` mixed with primitives, or a
    genuinely heterogeneous set). Without the fallback ``sorted`` would raise a
    ``TypeError`` and abort the whole discovery on such a default.
    """
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=lambda x: (x is None, str(x)))


def _coerce_json(value: Any, *, opaque: Any) -> Any:
    """Recursively coerce *value* into json-safe data.

    Shared traversal for :func:`jsonable` and :func:`exposable_default`: both
    pass through primitives, recurse structures (list/tuple/set/dict), and render
    a ``type`` as its name - they differ ONLY in how an opaque leaf is handled,
    supplied here as ``opaque(value)`` (``str`` to stringify, ``lambda _: None``
    to drop it).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_json(v, opaque=opaque) for v in value]
    if isinstance(value, set):
        return _sorted_stable([_coerce_json(v, opaque=opaque) for v in value])
    if isinstance(value, dict):
        return {str(k): _coerce_json(v, opaque=opaque) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return opaque(value)


def jsonable(value: Any) -> Any:
    """Coerce a value into something json.dumps can handle without default=str.

    Handles primitives, lists, tuples, dicts, sets, enums, and falls back to
    str(value) for anything else. This keeps the output deterministic and
    free of object repr noise.
    """
    return _coerce_json(value, opaque=str)


def exposable_default(value: Any) -> Any:
    """Sanitise a field default for the discovered schema.

    Like :func:`jsonable` for clean structures (primitives, lists, dicts, sets),
    but an opaque non-serialisable object becomes ``None`` rather than its repr
    string. A complex default (e.g. a pydantic sub-config instance) must not
    leak into the schema as a stringified blob: codegen would emit it as a
    non-None default and forward that bogus value to the engine on every unset
    run. Recording ``None`` makes the field omit cleanly (exclude_none) unless
    the user sets it. Pin this contract in tests so the str() regression can't
    return silently.
    """
    return _coerce_json(value, opaque=lambda _value: None)


def _single_candidate(annotation: Any) -> Any | None:
    """Return the sole non-``None`` member of ``annotation`` (unwrapping ``X | None``), else None.

    Multi-member unions and bare generics (``list[SubConfig]``) have no single
    class to ``$ref`` and yield None.
    """
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        candidates = tuple(a for a in get_args(annotation) if a is not type(None))
    elif origin is None:
        candidates = (annotation,)
    else:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _resolve_pydantic_type(annotation: Any) -> type | None:
    """Return the Pydantic model OR pydantic-dataclass in ``annotation``, else None.

    Recognises ``pydantic.BaseModel`` subclasses (expose ``model_json_schema``) AND
    ``pydantic.dataclasses``-decorated classes (expose ``__pydantic_fields__`` but
    NOT ``model_json_schema`` - e.g. vLLM's ``@config`` sub-configs). Both carry the
    validation metadata (``Literal`` enums, ``Field(ge/le/...)`` bounds) that the
    rich :func:`_fold_model_defs` path captures from the pydantic JSON schema; a
    PLAIN stdlib dataclass has neither and routes to :func:`_resolve_dataclass_type`,
    whose bare-fields walk cannot see those bounds. Nested generics
    (``list[SubConfig]``) have no single class to ``$ref`` and yield None.
    """
    candidate = _single_candidate(annotation)
    if isinstance(candidate, type) and (
        hasattr(candidate, "model_json_schema") or hasattr(candidate, "__pydantic_fields__")
    ):
        return candidate
    return None


def _resolve_dataclass_type(annotation: Any) -> type | None:
    """Return the non-Pydantic ``@dataclass`` in ``annotation`` (unwrapping ``X | None``), else None.

    The Pydantic path (:func:`_resolve_pydantic_type`) takes precedence; this
    catches the stdlib-dataclass sub-configs that engine-args classes nest
    WITHOUT Pydantic (e.g. vLLM ``CompilationConfig`` / ``AttentionConfig``,
    which expose no ``model_json_schema``), so they recurse into ``$defs``
    instead of flattening to a bare type-name string.
    """
    candidate = _single_candidate(annotation)
    if (
        isinstance(candidate, type)
        and dataclasses.is_dataclass(candidate)
        and not hasattr(candidate, "model_json_schema")
        and not hasattr(candidate, "__pydantic_fields__")
    ):
        return candidate
    return None


def _literal_enum_spec(annotation: Any) -> dict[str, Any] | None:
    """If *annotation* is ``Literal[...]`` (optionally ``X | None``), return ``{type, enum}``.

    A flat ``Literal``-annotated field (e.g. vLLM ``EngineArgs.dtype``) otherwise
    renders as the opaque ``"Literal[...]"`` type STRING via
    :func:`annotation_to_type_str`, which the codegen's scalar-only translator
    cannot parse and collapses to ``Any | None`` - so the membership set never
    reaches the generated config. Capturing it as a structured JSON Schema ``enum``
    (plus the members' base scalar type) lets the codegen project a real
    ``Literal[...]`` field. Mirrors the enum capture the Pydantic ``$ref`` path
    already gets from :func:`_fold_model_defs`. Returns ``None`` for non-Literal
    annotations. Membership is rendered in declared order (deterministic).
    """
    candidate = annotation
    origin = get_origin(candidate)
    if origin is Union or origin is types.UnionType:
        non_none = [a for a in get_args(candidate) if a is not type(None)]
        if len(non_none) != 1:
            return None
        candidate = non_none[0]
        origin = get_origin(candidate)
    if origin is None or "Literal" not in str(origin):
        return None
    members = list(get_args(candidate))
    spec: dict[str, Any] = {"enum": [jsonable(m) for m in members]}
    member_types = {type(m) for m in members}
    if member_types == {str}:
        spec["type"] = "str"
    elif member_types == {bool}:
        spec["type"] = "bool"
    elif member_types == {int}:
        spec["type"] = "int"
    elif member_types <= {int, float}:
        spec["type"] = "float"
    # Mixed / non-scalar members: emit the enum alone (codegen builds the Literal
    # from membership without a base scalar type).
    return spec


def dataclass_fields_to_specs(
    cls: type,
    *,
    skip_private: bool = False,
    defs: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract ``{name: {type, default}}`` specs from a dataclass.

    Resolves ``default_factory`` by calling it (swallowing errors to ``None``)
    so downstream JSON stays concrete. Types are rendered via
    ``annotation_to_type_str``.

    When a field's resolved type is a sub-config class and a ``defs`` accumulator
    is supplied, the field is emitted as a JSON Schema ``$ref`` and the class is
    folded into ``defs``: a Pydantic model via its ``model_json_schema()``
    (:func:`_fold_model_defs`), a stdlib ``@dataclass`` via a recursive walk of
    its own fields (:func:`_fold_dataclass_defs`). This surfaces the sub-configs
    nested inside stdlib-dataclass engine-args (e.g. vllm ``EngineArgs`` ->
    ``CompilationConfig`` / ``AttentionConfig``) that would otherwise flatten to
    a bare type-name string. Pass the same ``defs`` dict to :func:`make_envelope`
    to ship it.
    """
    specs: dict[str, dict[str, Any]] = {}
    # Only resolve string annotations to real types when recursion is requested -
    # the legacy (defs=None) path keeps rendering ``fld.type`` exactly as before
    # so existing committed schema type strings are unchanged.
    hints = _safe_type_hints(cls) if defs is not None else {}
    for fld in dataclasses.fields(cls):
        if skip_private and fld.name.startswith("_"):
            continue
        default: Any = None
        if fld.default is not dataclasses.MISSING:
            default = fld.default
        elif fld.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            try:
                default = fld.default_factory()
            except Exception:
                default = None
        if defs is not None:
            annotation = hints.get(fld.name, fld.type)
            nested_model = _resolve_pydantic_type(annotation)
            if nested_model is not None:
                _fold_model_defs(nested_model, defs)
                specs[fld.name] = {
                    "$ref": f"#/$defs/{nested_model.__name__}",
                    "default": exposable_default(default),
                }
                continue
            nested_dc = _resolve_dataclass_type(annotation)
            if nested_dc is not None:
                _fold_dataclass_defs(nested_dc, defs)
                specs[fld.name] = {
                    "$ref": f"#/$defs/{nested_dc.__name__}",
                    "default": exposable_default(default),
                }
                continue
            enum_spec = _literal_enum_spec(annotation)
            if enum_spec is not None:
                specs[fld.name] = {**enum_spec, "default": exposable_default(default)}
                continue
        specs[fld.name] = {
            "type": annotation_to_type_str(fld.type),
            "default": exposable_default(default),
        }
    return specs


def _safe_type_hints(cls: type) -> dict[str, Any]:
    """Resolve string annotations to real types, falling back to raw ``field.type``.

    ``from __future__ import annotations`` makes ``field.type`` a string, which
    cannot be introspected for nested Pydantic models. ``get_type_hints`` evaluates
    them; resolution can fail on TYPE_CHECKING-only forward refs, so fall back to
    the raw annotations rather than raising during discovery.
    """
    try:
        return typing.get_type_hints(cls)
    except Exception:
        return {f.name: f.type for f in dataclasses.fields(cls)}


def _fold_model_defs(model: type, defs: dict[str, Any]) -> None:
    """Fold ``model``'s root definition and its own ``$defs`` into ``defs``.

    The pydantic JSON schema (``{$defs: {...}, ...root...}`` under the standard
    ``#/$defs/<Name>`` namespace) carries the validation metadata (``enum``,
    ``minimum``/``maximum``/``exclusive*``) the bare-dataclass walk loses. A
    ``BaseModel`` exposes ``model_json_schema``; a ``pydantic.dataclasses`` class
    does not, so its schema is built via ``pydantic.TypeAdapter`` - the field
    bounds/enums land identically either way. We lift the root body under the
    model name and merge transitively-referenced sub-models so every ``$ref``
    resolves.
    """
    try:
        if hasattr(model, "model_json_schema"):
            schema = model.model_json_schema(ref_template="#/$defs/{model}")
        else:
            import pydantic  # local: keep the host-side dispatch import surface minimal

            schema = pydantic.TypeAdapter(model).json_schema(ref_template="#/$defs/{model}")
    except Exception:
        return
    for name, body in schema.pop("$defs", {}).items():
        defs.setdefault(name, body)
    # The root schema body (title/properties/...) becomes this model's own def.
    defs.setdefault(model.__name__, schema)


def _fold_dataclass_defs(cls: type, defs: dict[str, Any]) -> None:
    """Fold a non-Pydantic ``@dataclass`` and its nested dataclasses into ``defs``.

    The dataclass analogue of :func:`_fold_model_defs` (stdlib dataclasses carry
    no ``model_json_schema``). Recurses through :func:`dataclass_fields_to_specs`,
    so a dataclass-typed field (``CompilationConfig.pass_config``) folds its own
    ``$def`` and ``$ref`` too. A placeholder entry is written BEFORE recursing so
    a self- or mutually-referential dataclass terminates - the standard ``$defs``
    cycle break (a re-entry sees ``cls.__name__`` already present and returns,
    leaving the dangling ``$ref`` to be filled once the outer call completes).
    """
    if cls.__name__ in defs:
        return
    defs[cls.__name__] = {}  # cycle-break placeholder; overwritten below
    defs[cls.__name__] = {
        "title": cls.__name__,
        "type": "object",
        "properties": dataclass_fields_to_specs(cls, defs=defs),
    }


def fold_dict_typed_subconfig(
    specs: dict[str, dict[str, Any]],
    field: str,
    cls: type,
    defs: dict[str, Any],
) -> bool:
    """Rewrite a dict-typed sub-config field to a ``$ref`` of its real class.

    Some engine-args fields are annotated as a bare dict / ``object`` (e.g. vLLM
    ``EngineArgs.speculative_config: dict[str, Any] | None``, tensorrt
    ``TrtLlmArgs.build_config: Optional[object]``) yet are coerced to a concrete
    config class at construction. The annotation alone cannot be recursed, so the
    introspector carries a field->class hint and calls this: ``cls`` is folded
    into ``defs`` (via the Pydantic or dataclass path) and the field's spec
    becomes a ``$ref`` so its leaves are discoverable. The original ``default`` is
    preserved.

    Returns ``True`` when the field was folded, ``False`` (no mutation) when the
    field is absent or ``cls`` is neither a Pydantic model nor a dataclass (a
    plain class the lift cannot introspect) - so the caller can record an honest
    discovery limitation instead of leaving a silent gap.
    """
    spec = specs.get(field)
    if spec is None:
        return False
    if hasattr(cls, "model_json_schema"):
        _fold_model_defs(cls, defs)
    elif dataclasses.is_dataclass(cls):
        _fold_dataclass_defs(cls, defs)
    else:
        return False
    specs[field] = {"$ref": f"#/$defs/{cls.__name__}", "default": spec.get("default")}
    return True


def merge_source_constraints(
    schema_fields: dict[str, dict[str, Any]],
    source_paths: list[Path],
    *,
    suffixes: tuple[str, ...] = ("Config", "Params", "Args"),
) -> int:
    """Fold source-text declarative constraints onto discovered ``schema_fields``.

    Runtime introspection captures type + default but not the ``Field(ge/le/...)``
    numeric bounds or ``Literal[...]`` membership sets that live in the class
    source. This walks each path in ``source_paths`` with
    :func:`scripts.engine_producers._source_walker.walk_declarative_constraints`
    and overlays the per-field bounds/enum keys onto the matching discovered
    field (by bare field name across all walked classes). Type/default already
    present are never overwritten; only the constraint keys in
    :data:`_CONSTRAINT_KEYS` are added, and only when the field is already in
    ``schema_fields`` (discovery is the source of truth for which fields exist).

    Mutates ``schema_fields`` in place and returns the number of fields that
    gained at least one constraint key (a surface-trend metric - expected near
    zero on the current pins, where the entry classes carry few bounds the
    walker reaches).
    """
    overlay: dict[str, dict[str, Any]] = {}
    for path in source_paths:
        if not path.is_file():
            continue
        module = ast.parse(path.read_text())
        for fields in walk_declarative_constraints(module, suffixes=suffixes).values():
            for field_name, fragment in fields.items():
                constraints = {k: v for k, v in fragment.items() if k in _CONSTRAINT_KEYS}
                if constraints:
                    overlay.setdefault(field_name, {}).update(constraints)

    touched = 0
    for field_name, constraints in overlay.items():
        spec = schema_fields.get(field_name)
        if spec is None:
            continue
        added = False
        for key, value in constraints.items():
            if key not in spec:
                spec[key] = value
                added = True
        if added:
            touched += 1
    return touched


def recover_field_types(target_type: type) -> dict[str, str]:
    """Recover ``{field: type_str}`` from a ``msgspec.Struct`` via type inspection.

    ``msgspec.json.schema`` drops to an untyped ``anyOf`` (no top-level ``type``
    key) for union / enum / nested-struct fields, so the schema introspector
    renders them ``"unknown"``. ``msgspec.inspect.type_info`` still resolves the
    concrete field type for those same fields, so this recovers them for the
    schema product.

    Returns ``{}`` for non-Struct types. Fields whose type is genuinely opaque
    (a bare ``Any``) are omitted so the caller keeps its existing label rather
    than overwriting ``unknown`` with another non-informative token.
    """
    import msgspec  # local: keep the host-side schema-substrate import surface minimal

    if not (isinstance(target_type, type) and issubclass(target_type, msgspec.Struct)):
        return {}
    info = msgspec.inspect.type_info(target_type)
    if not isinstance(info, msgspec.inspect.StructType):
        return {}
    recovered: dict[str, str] = {}
    for field in info.fields:
        type_str = _render_msgspec_type(field.type)
        if type_str is not None:
            recovered[field.name] = type_str
    return recovered


def _render_msgspec_type(node: Any) -> str | None:
    """Render a ``msgspec.inspect`` type node to a compact type string, or None.

    Returns ``None`` only for a bare ``AnyType`` (no information to add). Unions
    render their members joined by `` | `` with ``None`` last, mirroring the
    introspector's existing ``X | None`` convention; a union that is *only*
    ``Any | None`` collapses to ``None`` (still no type information beyond
    nullability) so the caller does not overwrite ``unknown`` with ``Any | None``.
    """
    import msgspec

    inspect_mod = msgspec.inspect
    if isinstance(node, inspect_mod.AnyType):
        return None
    if isinstance(node, inspect_mod.UnionType):
        members = [m for m in node.types if not isinstance(m, inspect_mod.NoneType)]
        has_none = len(members) < len(node.types)
        rendered = [_render_msgspec_type(m) for m in members]
        parts = [r for r in rendered if r is not None]
        if not parts:
            return None
        if has_none:
            parts.append("None")
        return " | ".join(parts)
    return _render_msgspec_scalar(node)


def _render_msgspec_scalar(node: Any) -> str:
    """Render a non-union ``msgspec.inspect`` node to a type token."""
    import msgspec

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
        return f"list[{_render_msgspec_scalar_or_any(node.item_type)}]"
    if isinstance(node, inspect_mod.DictType):
        key = _render_msgspec_scalar_or_any(node.key_type)
        val = _render_msgspec_scalar_or_any(node.value_type)
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


def _render_msgspec_scalar_or_any(node: Any) -> str:
    """Like :func:`_render_msgspec_scalar` but renders a bare Any item as ``Any``."""
    import msgspec

    if isinstance(node, msgspec.inspect.AnyType):
        return "Any"
    rendered = _render_msgspec_type(node)
    return rendered if rendered is not None else "Any"


def make_envelope(
    *,
    engine: str,
    engine_version: str,
    engine_commit_sha: str | None,
    image_ref: str | None,
    base_image_ref: str | None,
    discovery_method: str,
    discovery_limitations: list[dict[str, Any]],
    engine_params: dict[str, Any],
    sampling_params: dict[str, Any],
    defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Honour LLENERGY_DISCOVERY_FROZEN_AT when the caller (CI) wants the
    # envelope pinned to a stable anchor - typically the author date of the
    # most recent commit touching any input path. Without this override every
    # CI run produces a fresh wallclock timestamp, which the workflow's
    # commit-back picks up as a 2-line diff, re-firing the path filter and
    # creating a synchronize loop.
    discovered_at = (
        os.environ.get("LLENERGY_DISCOVERY_FROZEN_AT") or datetime.now(timezone.utc).isoformat()
    )
    envelope: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_version": engine_version,
        "engine_commit_sha": engine_commit_sha,
        "image_ref": image_ref,
        "base_image_ref": base_image_ref,
        "discovered_at": discovered_at,
        "discovery_method": discovery_method,
        "discovery_limitations": discovery_limitations,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    # Nested-class definitions are canonical JSON Schema 2020-12 ``$defs`` -
    # exactly the shape ``model_json_schema()`` / ``msgspec.json.schema()``
    # already emit and that ``$ref`` entries in engine_params/sampling_params
    # point at. Preserve them rather than dropping at envelope assembly;
    # ``jsonable`` keeps the block free of object-repr noise without
    # re-flattening the canonical structure. Only emitted when non-empty so
    # additive-schema loaders stay simple.
    if defs:
        envelope["$defs"] = jsonable(defs)
    return envelope
