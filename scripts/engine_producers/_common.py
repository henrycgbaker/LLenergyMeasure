"""Shared helpers for per-engine introspectors.

Introspectors run inside an environment where the target engine package is
installed (typically a Docker container). For each engine, they introspect
the native Python API surface and write a JSON schema file with a common
envelope. The envelope is versioned separately from the engines (see
:data:`SCHEMA_VERSION`); major bumps are breaking and SchemaLoader rejects
them. Minor bumps add envelope keys; downstream loaders are expected to be
forward-compatible.

"Introspector" is the runtime-introspection counterpart to "miner"
(AST/static-source extraction). Per-engine modules under
``scripts.engine_producers`` (schema introspector modules) mirror the per-engine miner structure
under ``scripts.engine_producers``.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import inspect
import os
import re
import types
import typing
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Union, get_args, get_origin

# Schema version 2.0.0 (bump from 1.0.0):
#
# Per-field specs are now JSON-Schema-2020-12-conformant: ``type`` holds a
# canonical primitive name (``"string"``, ``"integer"``, ``"number"``,
# ``"boolean"``, ``"array"``, ``"object"``, ``"null"``) or an array of
# those names (e.g. ``["string", "null"]`` for ``X | None``); complex
# unions surface as ``anyOf``; class-typed fields surface as ``"object"``
# with the class name preserved on ``description``; literals / enums
# surface as ``{"type": <primitive>, "enum": [...]}``. The free-form
# ``discovery_method`` envelope key has been dropped (engine + version +
# producer file path together carry equivalent information). The major
# bump signals to SchemaLoader that the per-field type representation is
# no longer the legacy Python-string compact form.
SCHEMA_VERSION = "2.0.0"

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


def jsonable(value: Any) -> Any:
    """Coerce a value into something json.dumps can handle without default=str.

    Handles primitives, lists, tuples, dicts, sets, enums, and falls back to
    str(value) for anything else. This keeps the output deterministic and
    free of object repr noise.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return sorted(jsonable(v) for v in value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, type):
        return value.__name__
    return str(value)


# Maps Python builtin type names to canonical JSON Schema 2020-12 primitive names.
# Class names not in this map are treated as ``"object"`` with the class name
# surfaced on ``description`` for human reference.
_PYTHON_TO_JSONSCHEMA_PRIMITIVE: dict[str, str] = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "bytes": "string",
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "frozenset": "array",
    "NoneType": "null",
    "None": "null",
}

# Canonical JSON Schema 2020-12 primitive type names. Anything else in a
# ``type`` slot is a class name (rendered ``"object"``) or a union.
_JSONSCHEMA_PRIMITIVES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null"}
)


def _is_pydantic_model(cls: Any) -> bool:
    """Detect a Pydantic v2 ``BaseModel`` subclass without forcing pydantic import.

    Returns False for strings, generics, non-class objects, ImportError on
    pydantic, or anything that isn't a BaseModel subclass.
    """
    try:
        import pydantic  # type: ignore[import-not-found]
    except ImportError:
        return False
    return isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)


def _emit_pydantic_ref(cls: type, defs_acc: dict[str, Any]) -> dict[str, Any]:
    """Add ``cls.model_json_schema()`` to ``defs_acc`` and return a ``$ref`` to it.

    Idempotent on ``cls.__name__``: a second call with the same class is a
    no-op on ``defs_acc`` (we don't re-emit schemas that already landed).
    Pydantic's transitive ``$defs`` (sub-classes referenced from ``cls``'s
    fields) are flattened into ``defs_acc`` as siblings.

    On failure (some Pydantic classes raise during schema generation, e.g.
    classes with un-resolvable forward refs), falls back to an opaque
    ``{"type": "object", "description": "<class-name>"}`` so discovery
    never aborts.

    The emitted entries are NOT canonicalized here; canonicalize at envelope
    assembly via :func:`canonicalize_defs` so the cleanup pass is uniform
    across all defs sources.
    """
    name = cls.__name__
    if name in defs_acc:
        return {"$ref": f"#/$defs/{name}"}
    try:
        schema = cls.model_json_schema(ref_template="#/$defs/{model}")
    except Exception:
        return {"type": "object", "description": name}
    sub_defs = schema.pop("$defs", None) or {}
    defs_acc[name] = schema
    for sub_name, sub_schema in sub_defs.items():
        if sub_name not in defs_acc:
            defs_acc[sub_name] = sub_schema
    return {"$ref": f"#/$defs/{name}"}


def _is_stdlib_dataclass(cls: Any) -> bool:
    """Detect a stdlib ``@dataclass``-decorated class (not a Pydantic model).

    Pydantic v2 ``BaseModel`` subclasses can also satisfy ``is_dataclass`` in
    some configurations; check Pydantic first to keep the two emission paths
    disjoint.
    """
    if not isinstance(cls, type):
        return False
    if _is_pydantic_model(cls):
        return False
    return dataclasses.is_dataclass(cls)


def _emit_dataclass_ref(cls: type, defs_acc: dict[str, Any]) -> dict[str, Any]:
    """Add a stdlib-dataclass schema to ``defs_acc`` and return a ``$ref`` to it.

    Mirrors :func:`_emit_pydantic_ref` for HF / stdlib ``@dataclass`` classes
    (e.g. ``transformers.generation.configuration_utils.CompileConfig``).
    Idempotent on ``cls.__name__``. Walks the dataclass fields via
    :func:`dataclass_fields_to_specs` (which itself recurses for nested
    dataclass / Pydantic fields, passing the same ``defs_acc``).

    The emitted def shape is canonical JSON Schema 2020-12:
    ``{"type": "object", "properties": {<field>: <spec>}, "default": null}``.
    """
    name = cls.__name__
    if name in defs_acc:
        return {"$ref": f"#/$defs/{name}"}
    # Reserve the slot first to prevent infinite recursion on self-referential dataclasses.
    defs_acc[name] = {"type": "object"}
    try:
        properties = dataclass_fields_to_specs(cls, defs_acc=defs_acc)
    except Exception:
        defs_acc.pop(name, None)
        return {"type": "object", "description": name}
    defs_acc[name] = {"type": "object", "properties": properties}
    return {"$ref": f"#/$defs/{name}"}


def annotation_to_json_schema(
    annotation: Any,
    *,
    defs_acc: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Render a Python type annotation as a canonical JSON Schema 2020-12 dict.

    Mapping:
    - ``int / float / str / bool / bytes / dict / list / tuple / set / None``
      -> ``{"type": "<canonical-primitive>"}``.
    - ``Literal[...]`` / :class:`enum.Enum` subclass ->
      ``{"type": <inferred-primitive>, "enum": [<values>]}``.
    - ``X | None`` -> ``{"type": ["<canonical-X>", "null"]}``.
    - ``X | Y`` (non-None) ->
      ``{"anyOf": [{"type": "<canonical-X>"}, {"type": "<canonical-Y>"}]}``.
    - Generic containers (``list[str]``, ``dict[str, int]``) ->
      ``{"type": "array", "items": {...}}`` / ``{"type": "object"}``.
    - Pydantic ``BaseModel`` subclasses (only when ``defs_acc`` is provided)
      -> ``{"$ref": "#/$defs/<ClassName>"}``; the class's
      ``model_json_schema()`` is added to ``defs_acc`` so consumers can
      resolve the reference. This is how vllm ``EngineArgs`` Pydantic-typed
      sub-config fields (``KVTransferConfig``, ``CompilationConfig`` etc.)
      surface as nested classes rather than opaque ``{"type": "object"}``.
    - Other class names (e.g. ``PretrainedConfig``, ``BitsAndBytesConfig``
      when ``defs_acc`` is None or class is not Pydantic) ->
      ``{"type": "object", "description": "<class-name>"}``.
    - ``inspect.Parameter.empty`` -> ``{"description": "no annotation"}`` (any type).
    - String annotations (from ``from __future__ import annotations`` or
      forward refs) are best-effort parsed via
      :func:`_coerce_jsonable_type_string`; complex syntax like
      ``Literal[...]`` or generic subscripts that the legacy compact-string
      parser can't handle become ``{"type": "object", "description": "<str>"}``.

    Never raises: any unrecognised annotation falls back to
    ``{"type": "object", "description": "<repr>"}`` so discovery never
    aborts on an exotic upstream type.

    ``defs_acc`` is mutated in place when provided. Callers that don't care
    about Pydantic-class recursion pass ``None`` (default) and get the
    pre-``$defs`` behaviour byte-identical to the historical helper.
    """
    if annotation is type(None):
        return {"type": "null"}
    if annotation is inspect.Parameter.empty or annotation is inspect.Signature.empty:
        # Discovery sentinel: parameter / field had no annotation. Emit an
        # empty schema (matches anything) with a description so the
        # original opacity is preserved.
        return {"description": "no annotation (discovery saw inspect.Parameter.empty)"}

    # String annotations: arise from ``from __future__ import annotations``
    # or explicit forward refs. The legacy compact-string parser handles
    # the simple union / primitive cases; complex syntax (Literal['...'],
    # generics with brackets, ...) collapses to a fallback opaque-object
    # schema with the source string preserved on ``description``.
    if isinstance(annotation, str):
        return _string_annotation_to_json_schema(annotation)

    enum_shape = _literal_or_enum_to_json_schema(annotation)
    if enum_shape is not None:
        return enum_shape

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is None:
        # A bare type (int, str, MyClass, ...). Map builtins to primitives;
        # Pydantic BaseModel subclasses get a $ref (when an accumulator is
        # available); stdlib @dataclass classes get a $ref the same way;
        # everything else is an opaque object with the class name as a hint.
        name = getattr(annotation, "__name__", str(annotation))
        primitive = _PYTHON_TO_JSONSCHEMA_PRIMITIVE.get(name)
        if primitive is not None:
            return {"type": primitive}
        if defs_acc is not None and _is_pydantic_model(annotation):
            return _emit_pydantic_ref(annotation, defs_acc)
        if defs_acc is not None and _is_stdlib_dataclass(annotation):
            return _emit_dataclass_ref(annotation, defs_acc)
        return {"type": "object", "description": name}

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        sub_schemas = [annotation_to_json_schema(a, defs_acc=defs_acc) for a in non_none]
        return _union_schema(sub_schemas, nullable=has_none)

    origin_str = str(origin)
    if "Literal" in origin_str:
        # Edge case: empty Literal[] returns {} as a defensive any-shape.
        return _enum_shape([a for a in args]) or {}

    # Generic container types: list[X], dict[K, V], tuple[...], set[X], etc.
    origin_name = getattr(origin, "__name__", origin_str)
    primitive = _PYTHON_TO_JSONSCHEMA_PRIMITIVE.get(origin_name)
    if primitive == "array":
        # list[X] / tuple[X, ...] / set[X] -> {"type": "array", "items": <X>}
        if args:
            return {"type": "array", "items": annotation_to_json_schema(args[0], defs_acc=defs_acc)}
        return {"type": "array"}
    if primitive == "object":
        # dict[K, V] -> {"type": "object", "additionalProperties": <V>}
        if len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": annotation_to_json_schema(args[1], defs_acc=defs_acc),
            }
        return {"type": "object"}

    # Unknown generic origin (callable, awaitable, etc.) - keep the
    # repr as a description for human debugging.
    arg_strs = ", ".join(annotation_to_type_str(a) for a in args)
    description = f"{origin_name}[{arg_strs}]" if args else origin_name
    return {"type": "object", "description": description}


_LITERAL_STR_RE = re.compile(r"^Literal\[(.+)\]$")
_GENERIC_STR_RE = re.compile(r"^(list|tuple|set|frozenset|dict)\[(.+)\]$")


def _string_annotation_to_json_schema(annotation: str) -> dict[str, Any]:
    """Parse a string annotation (PEP 563 / forward ref) into canonical JSON Schema.

    Handles the common shapes the legacy compact parser couldn't:
    - ``"Literal['a', 'b']"`` -> ``{"type": "string", "enum": ["a", "b"]}``.
    - ``"list[str]"`` -> ``{"type": "array", "items": {"type": "string"}}``.
    - ``"dict[str, int]"`` ->
      ``{"type": "object", "additionalProperties": {"type": "integer"}}``.
    - Simple unions / primitives delegate to
      :func:`_coerce_jsonable_type_string`.
    """
    annotation = annotation.strip()
    if not annotation:
        return {"description": "no annotation"}

    # Literal['a', 'b', ...] / Literal[1, 2, ...]
    m = _LITERAL_STR_RE.match(annotation)
    if m:
        inner = m.group(1)
        # Parse a comma-separated list of repr'd literal values.
        parts: list[Any] = []
        for raw in _split_top_level_commas(inner):
            raw = raw.strip()
            if (raw.startswith("'") and raw.endswith("'")) or (
                raw.startswith('"') and raw.endswith('"')
            ):
                parts.append(raw[1:-1])
            else:
                # Try int, then float, then fall back to string.
                try:
                    parts.append(int(raw))
                    continue
                except ValueError:
                    pass
                try:
                    parts.append(float(raw))
                    continue
                except ValueError:
                    pass
                parts.append(raw)
        shape = _enum_shape(parts)
        return shape if shape is not None else {"description": annotation}

    # list[X] / dict[K, V] / tuple[...] / set[X]
    m = _GENERIC_STR_RE.match(annotation)
    if m:
        outer = m.group(1)
        inner = m.group(2)
        if outer in ("list", "tuple", "set", "frozenset"):
            # tuple[X, ...] / tuple[X, Y, ...] - take the first element type.
            first = _split_top_level_commas(inner)[0].strip()
            return {
                "type": "array",
                "items": _string_annotation_to_json_schema(first),
            }
        if outer == "dict":
            sub_parts = _split_top_level_commas(inner)
            if len(sub_parts) == 2:
                return {
                    "type": "object",
                    "additionalProperties": _string_annotation_to_json_schema(sub_parts[1].strip()),
                }
            return {"type": "object"}

    return _coerce_jsonable_type_string(annotation)


def _split_top_level_commas(s: str) -> list[str]:
    """Split a string on commas, ignoring those inside ``[]`` or quotes."""
    parts: list[str] = []
    depth = 0
    quote: str | None = None
    current: list[str] = []
    for ch in s:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            current.append(ch)
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _literal_or_enum_to_json_schema(annotation: Any) -> dict[str, Any] | None:
    """Return ``{"type": <inferred>, "enum": [...]}`` for Literal / Enum; else None."""
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _enum_shape([member.value for member in annotation])
    origin = get_origin(annotation)
    if origin is None or "Literal" not in str(origin):
        return None
    values = list(get_args(annotation))
    if not values:
        return None
    return _enum_shape(values)


def _enum_shape(values: list[Any]) -> dict[str, Any] | None:
    """Infer a JSON Schema ``{"type", "enum"}`` for a non-empty list of literal values.

    Returns ``None`` when ``values`` is empty so callers can fall through to
    a defensive any-shape.
    """
    if not values:
        return None
    # Order matters: ``bool`` is a subclass of ``int`` in Python, so test it first.
    if all(isinstance(v, bool) for v in values):
        kind = "boolean"
    elif all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        kind = "integer"
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        kind = "number"
    elif all(isinstance(v, str) for v in values):
        kind = "string"
    else:
        # Mixed-type enums are rare upstream; default to ``string`` and let
        # the value list speak for itself.
        kind = "string"
    return {"type": kind, "enum": [jsonable(v) for v in values]}


def _union_schema(sub_schemas: list[dict[str, Any]], *, nullable: bool) -> dict[str, Any]:
    """Collapse a list of sub-schemas (plus optional null) into the simplest canonical form.

    - Single sub-schema with primitive ``type``, plus null -> ``{"type": ["X", "null"]}``.
    - Single sub-schema without null -> the sub-schema as-is.
    - Multiple sub-schemas -> ``{"anyOf": [...]}`` (with a ``{"type": "null"}``
      branch appended when nullable). Duplicates with identical ``type`` /
      ``description`` are collapsed.
    """
    if not sub_schemas:
        # All branches were None (e.g. ``None | None`` defensively) -> just null.
        return {"type": "null"} if nullable else {}
    if len(sub_schemas) == 1:
        only = sub_schemas[0]
        if not nullable:
            return only
        # Single non-None branch + null: prefer the canonical type-array form
        # when the branch is a plain primitive (no enum, no items, etc.).
        plain_type = only.get("type")
        if (
            isinstance(plain_type, str)
            and plain_type in _JSONSCHEMA_PRIMITIVES
            and set(only.keys()) == {"type"}
        ):
            return {"type": [plain_type, "null"]}
        # Otherwise keep ``anyOf`` so the non-primitive structure (enum,
        # items, $ref, ...) survives intact.
        return {"anyOf": [only, {"type": "null"}]}
    # Multi-branch union: dedupe by JSON shape.
    branches: list[dict[str, Any]] = []
    seen: set[str] = set()
    import json as _json

    for sub in sub_schemas:
        key = _json.dumps(sub, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        branches.append(sub)
    if nullable:
        null_key = _json.dumps({"type": "null"}, sort_keys=True)
        if null_key not in seen:
            branches.append({"type": "null"})
    return {"anyOf": branches}


def _coerce_jsonable_type_string(type_str: str) -> dict[str, Any]:
    """Render a legacy compact type string (``"X | None"``) as canonical JSON Schema.

    Used by producers that already extracted a type string from an upstream
    schema (e.g. msgspec / Pydantic JSON Schema output) and need to round-trip
    it into the canonical envelope shape. Splits on ``" | "`` (at the top
    level only, so ``list[str | int]`` is not naively split), recurses each
    part through :func:`_string_annotation_to_json_schema` (so generics and
    Literals inside a union still canonicalise), and applies the same
    union collapsing as :func:`annotation_to_json_schema`.

    Unknown atoms (class names like ``"PretrainedConfig"``) become
    ``{"type": "object", "description": "<name>"}``. The sentinel
    ``"unknown"`` becomes the empty schema with a description.
    """
    type_str = type_str.strip()
    if not type_str or type_str == "unknown":
        return {"description": "discovery emitted 'unknown' (untyped upstream)"}

    parts = [p.strip() for p in _split_top_level_pipes(type_str)]
    non_none_parts = [p for p in parts if p not in ("None", "null", "NoneType")]
    has_none = len(non_none_parts) < len(parts)

    sub_schemas: list[dict[str, Any]] = []
    for part in non_none_parts:
        if part in _JSONSCHEMA_PRIMITIVES:
            sub_schemas.append({"type": part})
            continue
        primitive = _PYTHON_TO_JSONSCHEMA_PRIMITIVE.get(part)
        if primitive is not None:
            sub_schemas.append({"type": primitive})
            continue
        # Try the richer annotation parser for generics (``list[str]``) and
        # literals (``Literal['a', 'b']``). Only delegate when the part
        # actually matches one of those regexes; otherwise the fallback
        # would route right back through here and recurse forever.
        if _LITERAL_STR_RE.match(part) or _GENERIC_STR_RE.match(part):
            sub_schemas.append(_string_annotation_to_json_schema(part))
        else:
            sub_schemas.append({"type": "object", "description": part})
    return _union_schema(sub_schemas, nullable=has_none)


def _split_top_level_pipes(s: str) -> list[str]:
    """Split a type string on ``|`` at bracket depth zero (preserves ``list[X | Y]``)."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def runtime_value_to_spec(value: Any) -> dict[str, Any]:
    """Build a per-field canonical spec from a runtime value (no annotation).

    Used by producers that walk an object's runtime attributes (e.g.
    ``GenerationConfig().to_dict()``) where the only signal available is
    the value's type. ``None`` values surface as an empty schema with a
    description so consumers know the upstream had no annotation; primitive
    values map to their canonical JSON Schema type; container values
    surface ``"array"`` / ``"object"`` without item types (the runtime
    value's element types are not reliable across instantiations).
    """
    if value is None:
        return {
            "description": "runtime default was None; upstream has no type annotation",
            "default": None,
        }
    name = type(value).__name__
    primitive = _PYTHON_TO_JSONSCHEMA_PRIMITIVE.get(name)
    if primitive is not None:
        return {"type": primitive, "default": jsonable(value)}
    return {"type": "object", "description": name, "default": jsonable(value)}


def signature_param_to_spec(param: inspect.Parameter) -> dict[str, Any]:
    """Build a per-field canonical-JSON-Schema spec from a single ``inspect.Parameter``.

    Combines :func:`annotation_to_json_schema` for the ``type`` shape with a
    JSON-safe ``default`` derived from ``param.default``. ``inspect.Parameter.empty``
    defaults are rendered as ``null``.
    """
    spec = dict(annotation_to_json_schema(param.annotation))
    default: Any = None if param.default is inspect.Parameter.empty else param.default
    spec["default"] = jsonable(default)
    return spec


def jsonschema_property_to_canonical(spec: dict[str, Any]) -> dict[str, Any]:
    """Re-shape a raw JSON Schema property dict into the canonical 2.0.0 form.

    Used by producers that consume an upstream tool's JSON Schema output
    (e.g. ``msgspec.json.schema(...)`` or ``model_json_schema()``). The
    upstream shape is mostly already canonical (``type``, ``anyOf``,
    ``$ref``, ``items``, ``enum``, ``description``, ``deprecated``,
    ``default``); this helper:

    - Drops the ``"title"`` auto-key (Pydantic / msgspec restate the field
      name; it just adds noise to the per-field row).
    - Coerces ``type: "null"`` (the JSON Schema null primitive) to be
      consistent with our envelope (it's already canonical, but some
      upstream tools emit ``type: null`` as the literal Python ``None``;
      we normalise to the string).
    - Collapses ``anyOf: [X, {"type":"null"}]`` to ``{"type": ["X", "null"]}``
      when ``X`` is a single-primitive branch, matching the shape produced
      by :func:`annotation_to_json_schema`.
    - Leaves ``anyOf`` with multiple branches, ``$ref``, ``enum``, and
      bounds keywords untouched.

    Pre-existing extension keys (``x-source``, ``x-source-ref``) pass
    through unchanged so PR-0.5's enum-lifting output rides naturally
    in the canonical envelope.
    """
    out: dict[str, Any] = {}
    for k, v in spec.items():
        if k == "title":
            continue
        out[k] = v
    # Normalise anyOf [X, null] -> type: [X, "null"] when collapsing safely
    any_of = out.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2 and all(isinstance(b, dict) for b in any_of):
        null_branches = [b for b in any_of if b.get("type") == "null"]
        non_null = [b for b in any_of if b.get("type") != "null"]
        if (
            len(null_branches) == 1
            and len(non_null) == 1
            and isinstance(non_null[0].get("type"), str)
            and non_null[0]["type"] in _JSONSCHEMA_PRIMITIVES
            and set(non_null[0].keys()) == {"type"}
        ):
            out.pop("anyOf")
            out["type"] = [non_null[0]["type"], "null"]
    # Coerce a Python None smuggled into ``type`` slot to the string form.
    if out.get("type") is None and "type" in out and "anyOf" not in out and "$ref" not in out:
        out["type"] = "null"
    return out


def canonicalize_defs(
    defs: dict[str, Any] | None,
    *,
    exclude: Iterable[str] = (),
) -> dict[str, Any]:
    """Canonicalize a JSON Schema ``$defs`` block for the envelope.

    Each value in ``$defs`` is an object schema (the definition of a reusable
    nested config class - ``KvCacheConfig``, ``CompileConfig`` etc.). This
    helper drops ``title`` on each def (Pydantic restates the class name; it
    just adds noise) and canonicalizes the inner ``properties`` via
    :func:`jsonschema_property_to_canonical` so leaf-level shape is consistent
    with the top-level ``engine_params`` / ``sampling_params`` entries.

    ``exclude`` drops named root entries that have already been projected into
    the envelope's ``engine_params`` / ``sampling_params`` sections (e.g.
    msgspec's ``msgspec.json.schema(SamplingParams)`` returns a
    ``$ref: "#/$defs/SamplingParams"`` envelope with ``SamplingParams`` as a
    ``$def``; producers unpack its ``properties`` into ``sampling_params``
    directly, so the root ``SamplingParams`` entry would be a redundant
    duplicate). Pass ``exclude=["SamplingParams"]`` to drop it.

    Idempotent. ``None`` or empty input returns ``{}``.

    Out of scope: recursive flattening / inlining of ``$ref`` chains. The
    envelope keeps the original reference structure; downstream consumers
    (``datamodel-code-generator``) resolve refs natively.
    """
    if not defs:
        return {}
    excluded = set(exclude)
    out: dict[str, Any] = {}
    for name, spec in defs.items():
        if name in excluded:
            continue
        if not isinstance(spec, dict):
            continue
        canon = {k: v for k, v in spec.items() if k != "title"}
        props = canon.get("properties")
        if isinstance(props, dict):
            canon["properties"] = {
                pname: jsonschema_property_to_canonical(pspec)
                for pname, pspec in props.items()
                if isinstance(pspec, dict)
            }
        out[name] = canon
    return out


def dataclass_fields_to_specs(
    cls: type,
    *,
    skip_private: bool = False,
    defs_acc: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract ``{name: <canonical-JSON-Schema-spec>}`` mapping from a dataclass.

    Each leaf spec is canonical JSON Schema 2020-12: ``type`` is a primitive
    name (or array including ``"null"``), ``default`` is JSON-safe via
    :func:`jsonable`. ``default_factory`` is evaluated (errors swallowed to
    ``None``) so output stays concrete.

    String annotations (from ``from __future__ import annotations`` or
    explicit forward refs) are resolved at the class level via
    :func:`typing.get_type_hints` before being passed to
    :func:`annotation_to_json_schema`. This is what makes Pydantic-class
    recognition work for libraries that opt into PEP 563 deferred
    evaluation: without it, ``fld.type`` is the literal string
    ``"KVTransferConfig | None"`` and the ``$ref`` emission path can't
    fire. Falls back to ``fld.type`` when hint resolution itself fails
    (some classes have un-importable forward refs).

    When ``defs_acc`` is provided, Pydantic-typed fields surface as ``$ref``
    entries and the referenced class's ``model_json_schema()`` is added to
    ``defs_acc`` (recursively flattening any transitive ``$defs``). When
    ``defs_acc`` is None, Pydantic fields fall through to opaque
    ``{"type": "object", "description": "<class-name>"}`` (legacy
    pre-``$defs`` behaviour).
    """
    try:
        resolved_hints = typing.get_type_hints(cls)
    except Exception:
        resolved_hints = {}
    specs: dict[str, dict[str, Any]] = {}
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
        annotation = resolved_hints.get(fld.name, fld.type)
        spec = dict(annotation_to_json_schema(annotation, defs_acc=defs_acc))
        spec["default"] = jsonable(default)
        specs[fld.name] = spec
    return specs


def make_envelope(
    *,
    engine: str,
    engine_version: str,
    engine_commit_sha: str | None,
    image_ref: str | None,
    base_image_ref: str | None,
    discovery_limitations: list[dict[str, Any]],
    engine_params: dict[str, Any],
    sampling_params: dict[str, Any],
    defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a discovered-schema envelope at the current :data:`SCHEMA_VERSION`.

    The free-form ``discovery_method`` key (a per-engine string describing the
    introspection technique) was dropped in 2.0.0: engine + version + the
    producer file path together carry equivalent information.

    The optional ``defs`` parameter carries JSON Schema ``$defs`` reusable
    sub-schemas (nested config classes). When the upstream discovery surface
    is Pydantic (``model_json_schema()``) or msgspec (``msgspec.json.schema()``)
    those tools emit ``$defs`` natively for nested classes; producers pass them
    through here so the structural shape survives envelope assembly. Without
    this, ``kv_cache_config: {"$ref": "#/$defs/KvCacheConfig"}`` references go
    nowhere when consumers (e.g. ``datamodel-code-generator``) try to resolve
    them. ``defs`` is omitted from the emitted envelope when None or empty so
    pre-existing envelopes round-trip identically.
    """
    # Honour LLENERGY_DISCOVERY_FROZEN_AT when the caller (CI) wants the
    # envelope pinned to a stable anchor - typically the author date of the
    # most recent commit touching any input path. Without this override every
    # CI run produces a fresh wallclock timestamp, which the workflow's
    # commit-back picks up as a 2-line diff, re-firing the path filter and
    # creating a synchronize loop. Mirrors LLENERGY_VALIDATION_FROZEN_AT in
    # scripts/validate_invariants.py.
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
        "discovery_limitations": discovery_limitations,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if defs:
        envelope["$defs"] = defs
    return envelope


# ---------------------------------------------------------------------------
# Validation-collection extractor
# ---------------------------------------------------------------------------
#
# Many engine modules guard a field's accepted values against a module-level
# constant set/tuple/dict. ``GenerationConfig.validate`` for example raises
# when ``self.cache_implementation not in ALL_CACHE_IMPLEMENTATIONS``; vLLM's
# ``ModelConfig`` validates ``dtype`` against keys of
# ``_STR_DTYPE_TO_TORCH_DTYPE``. Lifting these constants into schema ``enum``
# annotations turns implicit runtime gates into machine-readable contracts.
#
# ``discover_validation_collections`` walks a module's AST in two passes:
# pass 1 collects candidate constants (literal set/frozenset/tuple/list/dict
# at module scope, plus simple ``A + B`` concatenations of prior candidates);
# pass 2 walks validator-method bodies for ``if v [not] in <Name>:`` and
# returns ``{field_name: {enum: [...], x-source: ..., x-source-ref: ...}}``
# entries only for constants that pass both passes. Pure naming heuristics
# (e.g. an unused ``_VALID_X`` constant) are NOT lifted -- the ``in``
# reference is the load-bearing filter against false positives.


# Decorator names that mark a function as a validator. Pydantic v2 spellings
# plus the historical ``@validator``; plus the dataclass-style
# ``__post_init__``/``_verify_*``/``validate_*`` patterns recognised by name.
_VALIDATOR_DECORATORS: frozenset[str] = frozenset(
    {"field_validator", "model_validator", "validator", "root_validator"}
)


def _is_validator_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if ``node`` looks like a validator method.

    Recognised: pydantic decorators (``@field_validator``, ``@model_validator``,
    ``@validator``, ``@root_validator``); dataclass-style ``__post_init__``;
    naming conventions ``_verify_*`` and ``validate_*`` (used by vLLM,
    TensorRT-LLM, transformers' ``GenerationConfig.validate``).
    """
    name = node.name
    if name == "__post_init__":
        return True
    if name.startswith("_verify_") or name.startswith("validate_") or name == "validate":
        return True
    for dec in node.decorator_list:
        # ``@field_validator(...)`` or ``@field_validator.something(...)``
        func = dec.func if isinstance(dec, ast.Call) else dec
        # Bare name: ``@validator`` -> Name; dotted: ``@pydantic.field_validator``
        # -> Attribute, take the last attr.
        dec_name: str | None = None
        if isinstance(func, ast.Name):
            dec_name = func.id
        elif isinstance(func, ast.Attribute):
            dec_name = func.attr
        if dec_name is not None and dec_name in _VALIDATOR_DECORATORS:
            return True
    return False


def _extract_field_validator_targets(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Return the field names listed in ``@field_validator("a", "b")`` decorators.

    Returns an empty list when the function carries no ``@field_validator``
    decorator or when the decorator's positional args aren't string literals.
    Multiple ``@field_validator`` decorators are merged; duplicates are
    preserved in source order (callers dedupe if needed).
    """
    targets: list[str] = []
    for dec in node.decorator_list:
        if not isinstance(dec, ast.Call):
            continue
        func = dec.func
        dec_name: str | None = None
        if isinstance(func, ast.Name):
            dec_name = func.id
        elif isinstance(func, ast.Attribute):
            dec_name = func.attr
        if dec_name not in {"field_validator", "validator"}:
            continue
        for arg in dec.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                targets.append(arg.value)
    return targets


def _ast_literal_to_python(node: ast.expr) -> Any:
    """Best-effort conversion of an AST literal node to a Python value.

    Wraps :func:`ast.literal_eval`; returns ``None`` on failure (any name
    reference, unresolved attribute, or non-literal value yields ``None``).
    The caller treats ``None`` as "not a literal" and discards the candidate.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


def _collect_module_constants(module_ast: ast.Module) -> dict[str, list[Any]]:
    """Pass 1: collect module-scope assignments of set/frozenset/tuple/list/dict literals.

    Returns ``{name: [values...]}`` where the value list is sorted-and-deduped
    for set/frozenset (deterministic enum output) and preserves insertion
    order for tuple/list/dict keys. For ``dict`` literals the KEYS are kept
    as the enum (the design's example: ``_STR_DTYPE_TO_TORCH_DTYPE``'s keys
    are the valid dtype strings).

    Also handles two derived cases:
      - ``X = frozenset({...})`` / ``X = set(...)`` / ``X = tuple(...)`` / ``X = list(...)``
        / ``X = dict(...)`` constructor calls with literal arguments.
      - ``X = A + B`` where both ``A`` and ``B`` are previously collected
        candidates (concatenation chain). Recursively flattens.

    Pure dataclass / function / class decorators / imports are ignored.
    """
    constants: dict[str, list[Any]] = {}

    def _value_from_call(call: ast.Call) -> list[Any] | None:
        # ``frozenset({...})``, ``set([...])``, ``tuple((...))``, ``list((...))``,
        # ``dict({...})`` -- caller must supply exactly one literal argument.
        if not isinstance(call.func, ast.Name):
            return None
        ctor = call.func.id
        if ctor not in {"frozenset", "set", "tuple", "list", "dict"}:
            return None
        if len(call.args) != 1:
            return None
        py = _ast_literal_to_python(call.args[0])
        if py is None:
            return None
        if isinstance(py, dict):
            # ``dict({...})`` -> keys (mirrors literal-dict handling)
            return list(py)
        if isinstance(py, (set, frozenset)):
            # Deterministic ordering for set-like
            try:
                return sorted(py)
            except TypeError:
                return list(py)
        if isinstance(py, (list, tuple)):
            return list(py)
        return None

    def _resolve(name: str, value_expr: ast.expr) -> list[Any] | None:
        # Direct literal: set / frozenset / tuple / list / dict
        if isinstance(value_expr, (ast.Set, ast.Tuple, ast.List, ast.Dict)):
            py = _ast_literal_to_python(value_expr)
            if py is None:
                return None
            if isinstance(py, dict):
                return list(py.keys())
            if isinstance(py, (set, frozenset)):
                try:
                    return sorted(py)
                except TypeError:
                    return list(py)
            return list(py)
        # Constructor call
        if isinstance(value_expr, ast.Call):
            return _value_from_call(value_expr)
        # Concatenation: ``A + B`` of two previously collected candidates
        if isinstance(value_expr, ast.BinOp) and isinstance(value_expr.op, ast.Add):
            left = _resolve_operand(value_expr.left)
            right = _resolve_operand(value_expr.right)
            if left is None or right is None:
                return None
            merged: list[Any] = []
            for v in [*left, *right]:
                if v not in merged:
                    merged.append(v)
            return merged
        return None

    def _resolve_operand(expr: ast.expr) -> list[Any] | None:
        # Operands may be a Name referencing a previously collected constant,
        # or another literal/call expression we can resolve inline.
        if isinstance(expr, ast.Name) and expr.id in constants:
            return list(constants[expr.id])
        return _resolve("<inline>", expr)

    for node in module_ast.body:
        # Plain ``X = <expr>``
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            name = node.targets[0].id
            values = _resolve(name, node.value)
            if values is not None:
                constants[name] = values
            continue
        # Annotated ``X: T = <expr>``
        if isinstance(node, ast.AnnAssign):
            if not isinstance(node.target, ast.Name) or node.value is None:
                continue
            name = node.target.id
            values = _resolve(name, node.value)
            if values is not None:
                constants[name] = values

    return constants


def _names_referenced_in_membership(node: ast.AST, *, candidates: set[str]) -> set[str]:
    """Walk a function body and return the subset of ``candidates`` used in ``in`` tests.

    Matches both ``X in NAME`` and ``X not in NAME`` shapes. The LHS is
    intentionally unconstrained -- the design's "v" stands in for any value
    being validated (``self.<field>``, a parameter, a temporary). Constants
    on the RHS that aren't in ``candidates`` are ignored (a same-module
    constant the introspector already lifted is the only safe ref to expand).
    """
    hits: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Compare):
            continue
        for op, comparator in zip(sub.ops, sub.comparators, strict=False):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            if isinstance(comparator, ast.Name) and comparator.id in candidates:
                hits.add(comparator.id)
    return hits


def _field_names_from_assign_chain(
    func: ast.FunctionDef | ast.AsyncFunctionDef, local_name: str
) -> list[str]:
    """Find ``self.<field>`` sources for a local name through simple aliases.

    Handles direct ``local_name = self.<field>`` assignment chains. Returns
    every field that flows into ``local_name`` (the validator might be
    parameterised over several inputs). Anything more elaborate (subscript,
    function call, conditional) yields no field name and the caller falls
    back to whatever signal it can get from the decorator / condition shape.
    """
    out: list[str] = []
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            continue
        tgt = stmt.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == local_name):
            continue
        rhs = stmt.value
        if (
            isinstance(rhs, ast.Attribute)
            and isinstance(rhs.value, ast.Name)
            and rhs.value.id == "self"
        ):
            out.append(rhs.attr)
    return out


def _fields_for_membership_check(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    constant_name: str,
) -> list[str]:
    """Best-effort: which field(s) does an ``in <constant_name>`` reference guard?

    Examines every ``X [not] in <constant_name>`` Compare node in ``func`` and
    extracts the field name from ``X`` where possible:

    - ``self.<field> [not] in CONST``                       -> ``<field>``
    - ``<local> [not] in CONST``  with ``<local> = self.<field>`` -> ``<field>``
    - ``@field_validator("<field>", ...)`` decorator on func     -> ``<field>``

    Returns a deduplicated list in first-seen order. Empty when none of the
    above patterns match -- caller drops the entry rather than guess.
    """
    decorator_fields = _extract_field_validator_targets(func)
    fields: list[str] = []

    def _add(name: str) -> None:
        if name not in fields:
            fields.append(name)

    for sub in ast.walk(func):
        if not isinstance(sub, ast.Compare):
            continue
        for op, comparator in zip(sub.ops, sub.comparators, strict=False):
            if not isinstance(op, (ast.In, ast.NotIn)):
                continue
            if not (isinstance(comparator, ast.Name) and comparator.id == constant_name):
                continue
            lhs = sub.left
            # ``self.<field>`` directly
            if (
                isinstance(lhs, ast.Attribute)
                and isinstance(lhs.value, ast.Name)
                and lhs.value.id == "self"
            ):
                _add(lhs.attr)
                continue
            # Local name aliased from ``self.<field>``
            if isinstance(lhs, ast.Name):
                aliased = _field_names_from_assign_chain(func, lhs.id)
                for fld in aliased:
                    _add(fld)
                # Fall through: even if the alias chain didn't resolve, the
                # decorator-named field is still a valid association when this
                # is a ``@field_validator(...)``.
                continue
            # Other LHS shapes (subscript, call, tuple) intentionally unhandled;
            # the decorator path below may still recover a field name.

    for name in decorator_fields:
        _add(name)
    return fields


def discover_validation_collections(
    module: ModuleType,
) -> dict[str, dict[str, Any]]:
    """Lift module-scope value-set constants into per-field ``enum`` entries.

    Two-pass AST walk on ``module``'s source:

    1. Collect every module-scope assignment of a ``set`` / ``frozenset`` /
       ``tuple`` / ``list`` / ``dict`` literal (plus ``X = A + B``
       concatenations and ``X = ctor(...)`` constructor calls).
    2. Walk every validator-method body (``@field_validator``,
       ``@model_validator``, ``@validator``, ``@root_validator``,
       ``__post_init__``, ``_verify_*``, ``validate_*``, ``validate``) for
       ``if v [not] in <Name>:`` -- where ``<Name>`` matches a pass-1
       candidate. The membership-check requirement is the load-bearing
       filter: a pure naming heuristic (``_VALID_X`` constant referenced
       nowhere) is NOT lifted, avoiding false positives.

    Returns ``{field_name: spec}`` where each spec has shape::

        {
            "enum": [...],
            "x-source": "module_validation_collection",
            "x-source-ref": "<module>.<constant_name>",
        }

    Field-name attribution rules (best-effort, in priority order):

    - ``self.<field> [not] in CONST`` -> ``<field>``
    - local ``<var>`` aliased from ``self.<field>``, then
      ``<var> [not] in CONST`` -> ``<field>``
    - ``@field_validator("<field>", ...)`` decorator -> ``<field>``

    A constant referenced in a membership check whose field cannot be
    attributed via any of these rules is skipped. A single constant guarding
    multiple fields produces one entry per field, all pointing to the same
    ``x-source-ref``.
    """
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    constants = _collect_module_constants(tree)
    if not constants:
        return {}

    candidate_set = set(constants)
    referenced: dict[str, list[str]] = {}  # constant_name -> [fields...]
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not _is_validator_function(node):
            continue
        hits = _names_referenced_in_membership(node, candidates=candidate_set)
        for const_name in hits:
            fields = _fields_for_membership_check(node, const_name)
            if not fields:
                continue
            bucket = referenced.setdefault(const_name, [])
            for f in fields:
                if f not in bucket:
                    bucket.append(f)

    if not referenced:
        return {}

    module_name = getattr(module, "__name__", "<unknown>")
    out: dict[str, dict[str, Any]] = {}
    for const_name, fields in referenced.items():
        values = constants[const_name]
        # ``jsonable`` is applied so callers don't have to worry about exotic
        # ast.literal_eval byproducts (frozenset surfaces as set; nested
        # tuples become lists). Deterministic ordering for set-derived
        # constants is already handled in pass 1.
        enum_values = jsonable(values)
        spec = {
            "enum": enum_values,
            "x-source": "module_validation_collection",
            "x-source-ref": f"{module_name}.{const_name}",
        }
        for field_name in fields:
            # First constant wins per field: a field referenced against two
            # constants in the same module is rare, and the caller can post-
            # process if a richer policy is needed.
            out.setdefault(field_name, spec)
    return out


def merge_validation_collections(
    field_specs: dict[str, dict[str, Any]],
    collections: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Merge ``collections`` (from :func:`discover_validation_collections`) into ``field_specs``.

    Returns the same ``field_specs`` dict mutated in place. For each field in
    ``collections``: if it exists in ``field_specs`` the ``enum`` /
    ``x-source`` / ``x-source-ref`` keys are added without disturbing the
    existing ``type`` / ``default`` / ``description`` / etc. If the field is
    not present in ``field_specs`` (e.g. the introspector skipped it) the
    enum spec is recorded under its own key so the information isn't lost.
    """
    for field_name, enum_spec in collections.items():
        if field_name in field_specs:
            field_specs[field_name].update(enum_spec)
        else:
            field_specs[field_name] = dict(enum_spec)
    return field_specs


# ---------------------------------------------------------------------------
# Move 1 walker: Sphinx-style kwargs-docstring extractor.
#
# Many engine APIs document their **kwargs in the class / method docstring
# rather than the signature (HuggingFace transformers in particular -
# `AutoModelForCausalLM.from_pretrained(*model_args, **kwargs)` documents
# attention impl, dtype, device_map, tp_plan etc. only in the docstring).
# `inspect.signature` misses them; this walker recovers them.
#
# Matches the standard HuggingFace / pytorch Sphinx pattern:
#
#     name (`type-expr`, *optional*[, defaults to `default-expr`]):
#         <multi-line description>
#
# Where `type-expr` is a backtick-quoted Python type expression (possibly
# multi-typed via " or "), optional is literal, and defaults-clause is
# optional.
# ---------------------------------------------------------------------------


_SPHINX_PARAM_RE = re.compile(
    r"""
    ^[ \t]+                              # leading indent (matters under MULTILINE)
    (?P<name>[a-z_][a-z0-9_]*)           # parameter name (lowercase ident)
    [ \t]*\(                              # opening paren
    [ \t]*`(?P<type>[^`]+)`               # primary backticked type
    (?P<types_rest>                       # zero+ additional or-joined types,
        (?:                               # each "or <alt>" where <alt> is
            [ \t]*or[ \t]*                #   either backticked OR a bare
            (?:                           #   identifier (HF docstrings mix
                `[^`]+`                   #   both forms - e.g. `torch.dtype`
                |                         #   or str).
                [A-Za-z_][A-Za-z0-9_.\[\],\ ]*
            )
        )*
    )
    [ \t]*,
    [ \t]*\*optional\*                    # *optional* marker (literal)
    (?:[ \t]*,[ \t]*defaults?[ \t]+to[ \t]+(?P<default>[^)]+?))?  # optional default
    [ \t]*\)\s*:\s*$                     # closing paren + colon
    """,
    re.MULTILINE | re.VERBOSE | re.IGNORECASE,
)

# Canonical JSON Schema type for simple Python type expressions.
_PYTYPE_TO_JSONSCHEMA: dict[str, str] = {
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "str": "string",
    "string": "string",
    "list": "array",
    "tuple": "array",
    "set": "array",
    "dict": "object",
}


def _parse_sphinx_default(raw: str) -> Any:
    """Parse a Sphinx ``defaults to <expr>`` value to a Python literal.

    Returns ``ast.literal_eval(stripped)`` when the expression parses;
    otherwise returns the stripped string itself (so non-literal defaults
    like ``the engine's own default`` round-trip without crashing).
    """
    s = raw.strip()
    # Strip backticks around a literal-ish expression.
    if s.startswith("`") and s.endswith("`"):
        s = s[1:-1]
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def _sphinx_type_to_jsonschema(type_expr: str) -> str | None:
    """Map a Sphinx type expression to a JSON Schema type keyword.

    Returns one of ``"string"`` / ``"integer"`` / ``"number"`` /
    ``"boolean"`` / ``"array"`` / ``"object"`` for a recognisable Python
    type. Returns ``None`` for complex / union / parameterised types
    (e.g. ``"Union[A, B]"``, ``"dict[str, ...]"``, ``"torch.dtype"``);
    the caller should emit no ``type`` key in that case rather than guess.
    """
    s = type_expr.strip().lower()
    # Strip generic parameters: dict[str, int] -> dict
    if "[" in s:
        s = s.split("[", 1)[0].strip()
    # Strip qualifiers: typing.optional[str] won't appear here but be safe
    if "." in s:
        s = s.split(".")[-1]
    return _PYTYPE_TO_JSONSCHEMA.get(s)


def parse_sphinx_kwargs(
    docstring: str,
    *,
    skip_names: set[str] | None = None,
    source_ref: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract `name (`type`, *optional*[, defaults to <default>]):` blocks.

    Args:
        docstring: The text to scan; pass ``cls.method.__doc__`` directly.
        skip_names: Names already present in the API signature - skip
            these so callers can use the walker for kwargs-only coverage.
            ``None`` returns every match.
        source_ref: Dotted source path written to ``x-source-ref`` on
            every emitted field (e.g.
            ``"transformers.PreTrainedModel.from_pretrained.__doc__"``).
            ``None`` omits the key.

    Returns:
        ``{field_name: {"type": <canonical>, "default": <parsed>,
        "description": <body>, "x-source": "kwargs_docstring",
        "x-source-ref": <source_ref>}}``. Fields whose Sphinx type
        doesn't map cleanly to JSON Schema omit the ``type`` key.

    The function is pure (input docstring + skip set -> output dict);
    no I/O, no inspect calls. Safe to test against synthetic docstrings.
    """
    skip = skip_names or set()
    out: dict[str, dict[str, Any]] = {}
    for match in _SPHINX_PARAM_RE.finditer(docstring):
        name = match.group("name")
        if name in skip or name in out:
            continue
        spec: dict[str, Any] = {}
        type_expr = match.group("type")
        json_type = _sphinx_type_to_jsonschema(type_expr)
        if json_type is not None:
            spec["type"] = json_type
        else:
            # Untyped (complex / union / parameterised / dotted) - preserve
            # the upstream type expression as a description so the schema
            # entry carries SOME informational content (the schema-shape
            # contract: every field has type, anyOf, OR description). Also
            # surfaces in the generated config's docstring per
            # --use-attribute-docstrings.
            spec["description"] = f"Upstream type: {type_expr}"
        default_raw = match.group("default")
        if default_raw is not None:
            spec["default"] = _parse_sphinx_default(default_raw)
        spec["x-source"] = "kwargs_docstring"
        if source_ref is not None:
            spec["x-source-ref"] = source_ref
        out[name] = spec
    return out
