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

import dataclasses
import enum
import inspect
import os
import re
import types
from datetime import datetime, timezone
from pathlib import Path
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


def annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
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
    - Class names (e.g. ``PretrainedConfig``, ``BitsAndBytesConfig``) ->
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
        # everything else is an opaque object with the class name as a hint.
        name = getattr(annotation, "__name__", str(annotation))
        primitive = _PYTHON_TO_JSONSCHEMA_PRIMITIVE.get(name)
        if primitive is not None:
            return {"type": primitive}
        return {"type": "object", "description": name}

    if origin is Union or origin is types.UnionType:
        non_none = [a for a in args if a is not type(None)]
        has_none = len(non_none) < len(args)
        sub_schemas = [annotation_to_json_schema(a) for a in non_none]
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
            return {"type": "array", "items": annotation_to_json_schema(args[0])}
        return {"type": "array"}
    if primitive == "object":
        # dict[K, V] -> {"type": "object", "additionalProperties": <V>}
        if len(args) == 2:
            return {
                "type": "object",
                "additionalProperties": annotation_to_json_schema(args[1]),
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


def dataclass_fields_to_specs(
    cls: type, *, skip_private: bool = False
) -> dict[str, dict[str, Any]]:
    """Extract ``{name: <canonical-JSON-Schema-spec>}`` mapping from a dataclass.

    Each leaf spec is canonical JSON Schema 2020-12: ``type`` is a primitive
    name (or array including ``"null"``), ``default`` is JSON-safe via
    :func:`jsonable`. ``default_factory`` is evaluated (errors swallowed to
    ``None``) so output stays concrete.
    """
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
        spec = dict(annotation_to_json_schema(fld.type))
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
) -> dict[str, Any]:
    """Build a discovered-schema envelope at the current :data:`SCHEMA_VERSION`.

    The free-form ``discovery_method`` key (a per-engine string describing the
    introspection technique) was dropped in 2.0.0: engine + version + the
    producer file path together carry equivalent information.
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
    return {
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
