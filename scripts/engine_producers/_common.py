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
# The per-field spec dicts are now JSON-Schema-conformant. Each leaf may
# additionally carry standard JSON Schema keys (``enum``, ``minimum``,
# ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``, ``pattern``,
# ``multipleOf``, ``deprecated``, ``description``) plus nested ``$ref``
# / ``$defs`` for Pydantic sub-models. The legacy ``type`` (compact
# stringified form) and ``default`` keys are preserved, so the change is
# additive at the per-field level. The major bump signals to downstream
# consumers that they may now safely depend on the new keys; older
# consumers that only read ``type`` / ``default`` are unaffected by
# anything except the version string itself.
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


def literal_to_json_schema(annotation: Any) -> dict[str, Any] | None:
    """Return a JSON-Schema-shaped ``{type, enum}`` dict for Literal / Enum.

    For ``Literal[a, b, c]`` returns ``{"type": <inferred>, "enum": [a, b, c]}``
    where the inferred ``type`` is ``"string"`` if every member is ``str``,
    ``"integer"`` if every member is ``int`` (booleans excluded), ``"number"``
    if every member is ``int`` or ``float``, and ``"boolean"`` if every member
    is ``bool``. For an :class:`enum.Enum` subclass returns the same shape with
    the members' ``.value``\\ s. Returns ``None`` for anything that is neither a
    Literal nor an Enum subclass so callers can fall through to other branches.
    """
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values: list[Any] = [member.value for member in annotation]
        return _enum_shape(values)

    origin = get_origin(annotation)
    if origin is None:
        return None
    # ``typing.Literal`` from the stdlib renders to ``typing.Literal`` in 3.10+
    # and ``Literal`` under PEP-604. Both expose the same get_args tuple.
    if "Literal" not in str(origin):
        return None
    values = list(get_args(annotation))
    if not values:
        return None
    return _enum_shape(values)


def _enum_shape(values: list[Any]) -> dict[str, Any]:
    """Infer a JSON Schema ``type`` for a non-empty list of enum / Literal members."""
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
        # Mixed-type enums are rare upstream; fall back to ``string`` and let
        # the value list speak for itself. Downstream consumers can re-infer
        # per-member if they really care.
        kind = "string"
    return {"type": kind, "enum": [jsonable(v) for v in values]}


def pydantic_to_json_schema(cls: type) -> dict[str, Any]:
    """Return Pydantic v2's native ``model_json_schema`` as-is, with ``$defs``.

    Preserves every standard JSON Schema key Pydantic emits per-field
    (``type``, ``default``, ``enum``, ``minimum``, ``maximum``,
    ``exclusiveMinimum``, ``exclusiveMaximum``, ``pattern``, ``multipleOf``,
    ``description``, ``deprecated``, ``required``, ``additionalProperties``)
    and the nested ``$defs`` / ``$ref`` structure for sub-models. Adds
    ``x-source: "pydantic_field"`` to each top-level property so downstream
    consumers can trace provenance.

    Raises ``TypeError`` if ``cls`` is not a Pydantic v2 ``BaseModel``
    subclass; callers should fall back to ``dataclass_fields_to_specs``
    or a manual extractor in that case.
    """
    if not hasattr(cls, "model_json_schema"):
        raise TypeError(f"{cls!r} is not a Pydantic v2 BaseModel (no model_json_schema method)")
    schema: dict[str, Any] = cls.model_json_schema(
        mode="serialization", ref_template="#/$defs/{model}"
    )
    props = schema.get("properties")
    if isinstance(props, dict):
        for spec in props.values():
            if isinstance(spec, dict):
                spec.setdefault("x-source", "pydantic_field")
    return schema


def msgspec_properties_to_specs(
    cls: type, *, skip_private: bool = True
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return ``(properties, defs)`` from a msgspec type's JSON Schema.

    msgspec wraps the target type in a ``$ref`` envelope: the actual
    property dict lives under ``$defs[<name>].properties``. This helper
    flattens that envelope and applies the same per-field shape contract
    as :func:`pydantic_properties_to_specs` -- compact ``type`` string
    preserved on each leaf for backward compatibility, all standard JSON
    Schema keys (``enum``, ``minimum``, ``maximum``, ``exclusiveMinimum``,
    ``exclusiveMaximum``, ``pattern``, ``multipleOf``, ``description``,
    ``deprecated``, ``$ref``, ``anyOf``) carried alongside, ``default``
    normalised through :func:`jsonable`. Each leaf gets ``x-source:
    "msgspec_field"``.

    ``skip_private`` filters property names starting with ``_``.
    """
    raw = msgspec_to_json_schema(cls)
    defs = raw.get("$defs") or raw.get("definitions") or {}
    props: dict[str, Any] = raw.get("properties") or {}
    if not props and defs:
        target_def: Any = defs.get(cls.__name__) or next(iter(defs.values()), {})
        if isinstance(target_def, dict):
            props = target_def.get("properties", {}) or {}
    specs: dict[str, dict[str, Any]] = {}
    for name, spec in props.items():
        if skip_private and name.startswith("_"):
            continue
        if not isinstance(spec, dict):
            continue
        enriched = dict(spec)
        enriched.pop("title", None)
        original_type = spec.get("type")
        compact = compact_type_from_jsonschema(spec)
        enriched["type"] = compact
        if (
            isinstance(original_type, str)
            and original_type in _JSONSCHEMA_PRIMITIVES
            and original_type != compact
        ):
            enriched["json_schema_type"] = original_type
        if "default" in enriched:
            enriched["default"] = jsonable(enriched["default"])
        enriched.setdefault("x-source", "msgspec_field")
        specs[name] = enriched
    return specs, dict(defs) if isinstance(defs, dict) else {}


def msgspec_to_json_schema(cls: type) -> dict[str, Any]:
    """Return msgspec's native ``msgspec.json.schema`` output as-is.

    The shape mirrors :func:`pydantic_to_json_schema`: a top-level dict with
    ``properties`` / ``$defs`` (or ``definitions``) that downstream introspectors
    consume directly. ``x-source: "msgspec_field"`` is added to each top-level
    property leaf. msgspec returns its own ``$ref`` envelope around the target
    type; callers extract ``properties`` themselves so we don't duplicate the
    walk here.

    Raises ``ImportError`` if msgspec isn't installed (vLLM ships it; other
    environments should not call this).
    """
    import msgspec  # type: ignore[import-not-found]

    schema: dict[str, Any] = msgspec.json.schema(cls)
    props = schema.get("properties")
    if isinstance(props, dict):
        for spec in props.values():
            if isinstance(spec, dict):
                spec.setdefault("x-source", "msgspec_field")
    return schema


def dataclass_fields_to_specs(
    cls: type, *, skip_private: bool = False
) -> dict[str, dict[str, Any]]:
    """Extract ``{name: {type, default, ...}}`` specs from a dataclass.

    Resolves ``default_factory`` by calling it (swallowing errors to ``None``)
    so downstream JSON stays concrete. The compact ``type`` string from
    :func:`annotation_to_type_str` is always preserved (v1 shape parity).
    When the annotation is a ``Literal[...]`` or :class:`enum.Enum`
    subclass, the spec is enriched with ``enum`` (additive) and the
    JSON-Schema-conformant primitive type (``"string"`` / ``"integer"`` /
    etc.) is surfaced under ``json_schema_type`` so structured consumers
    can still reach it without clobbering the legacy ``type`` string.
    ``description`` is surfaced when ``dataclasses.field(metadata=
    {"description": ...})`` is set. Each leaf carries ``x-source:
    "dataclass_field"`` for provenance.
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
        type_str = annotation_to_type_str(fld.type)
        spec: dict[str, Any] = {
            "type": type_str,
            "default": jsonable(default),
            "x-source": "dataclass_field",
        }
        # Literal / Enum yields a structured enum: keep the v1 ``type``
        # string in place for byte-stable backward compatibility, layer
        # the JSON Schema primitive on ``json_schema_type``, and surface
        # the value list on ``enum``.
        enum_shape = literal_to_json_schema(fld.type)
        if enum_shape is not None:
            spec["json_schema_type"] = enum_shape["type"]
            spec["enum"] = enum_shape["enum"]
        # Optional per-field description via dataclasses field metadata.
        meta = getattr(fld, "metadata", None)
        if meta:
            description = meta.get("description")
            if description:
                spec["description"] = str(description)
        specs[fld.name] = spec
    return specs


def signature_param_to_spec(param: inspect.Parameter) -> dict[str, Any]:
    """Build a per-field spec dict from a single ``inspect.Parameter``.

    Mirrors :func:`dataclass_fields_to_specs`: the compact ``type`` string
    from :func:`annotation_to_type_str` always lands on ``type`` for v1
    backward compatibility. When the annotation is a ``Literal[...]`` or
    :class:`enum.Enum` subclass, ``enum`` and ``json_schema_type``
    (``"string"`` / ``"integer"`` / etc.) are added as siblings. Each
    leaf carries ``x-source: "signature_param"`` for provenance.
    """
    default: Any = None if param.default is inspect.Parameter.empty else param.default
    type_str = annotation_to_type_str(param.annotation)
    spec: dict[str, Any] = {
        "type": type_str,
        "default": jsonable(default),
        "x-source": "signature_param",
    }
    enum_shape = literal_to_json_schema(param.annotation)
    if enum_shape is not None:
        spec["json_schema_type"] = enum_shape["type"]
        spec["enum"] = enum_shape["enum"]
    return spec


def pydantic_properties_to_specs(
    cls: type, *, skip_private: bool = True
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Return ``(properties, defs)`` from a Pydantic v2 model's JSON Schema.

    ``properties`` is the per-field dict (top-level only); ``defs`` is the
    nested ``$defs`` block (empty dict if the model has no sub-models).

    Per-field shape is the v2 union: the legacy ``type`` carries the compact
    Python-ish stringified form (e.g. ``"KvCacheConfig | None"`` for an
    ``anyOf`` of ``$ref`` + ``"null"``) so backward-compatible consumers
    (including ``diff_discovered_schemas``) keep working, while all
    JSON-Schema-conformant keys Pydantic emits (``enum``, ``minimum``,
    ``maximum``, ``exclusiveMinimum``, ``exclusiveMaximum``, ``pattern``,
    ``multipleOf``, ``description``, ``deprecated``, ``$ref``, ``anyOf``)
    ride alongside. ``default`` is normalised through :func:`jsonable` to
    survive ``json.dumps``. Each leaf carries ``x-source:
    "pydantic_field"`` for provenance.

    ``skip_private`` filters property names starting with ``_``.
    """
    raw = pydantic_to_json_schema(cls)
    raw_props = raw.get("properties") or {}
    defs = raw.get("$defs") or raw.get("definitions") or {}
    specs: dict[str, dict[str, Any]] = {}
    for name, spec in raw_props.items():
        if skip_private and name.startswith("_"):
            continue
        if not isinstance(spec, dict):
            continue
        enriched = dict(spec)
        # Drop the Pydantic ``title`` (auto-generated from the field name)
        # so it doesn't add noise to every field. Real per-field titles
        # would survive on a deeper extraction; the top-level title is
        # just ``"Kv Cache Config"`` etc. which restates the field name.
        enriched.pop("title", None)
        # Legacy compact type string takes the ``type`` key for backward
        # compatibility with the v1 shape and downstream diff classifiers.
        # ``compact_type_from_jsonschema`` honours an explicit Pydantic-
        # stuffed Python-ish ``type`` repr (e.g. ``"Optional[X]"``) before
        # falling into anyOf/ref unwinding, so the v1 string survives where
        # Pydantic emits one. When Pydantic emits a primitive (``"string"``,
        # ``"integer"``, etc.) the compact form equals it, so no extra key
        # is needed; only surface ``json_schema_type`` when the unwinding
        # produced a different compact string than Pydantic's primitive.
        original_type = spec.get("type")
        compact = compact_type_from_jsonschema(spec)
        enriched["type"] = compact
        if (
            isinstance(original_type, str)
            and original_type in _JSONSCHEMA_PRIMITIVES
            and original_type != compact
        ):
            enriched["json_schema_type"] = original_type
        # Normalise default through jsonable so callers can json.dumps the
        # spec dict directly without a custom encoder. Pydantic omits
        # ``default`` for required fields and for fields whose default is
        # a model instance the schema can't serialise; backfill ``null``
        # so the v2 shape's per-field contract stays ``{type, default}``
        # for every leaf and the v1 backward-compat invariant holds.
        if "default" in enriched:
            enriched["default"] = jsonable(enriched["default"])
        else:
            enriched["default"] = None
        # Pydantic only emits ``deprecated`` when ``True``; the v1 shape
        # always carried a literal ``false`` for non-deprecated fields,
        # so backfill to match.
        enriched.setdefault("deprecated", False)
        specs[name] = enriched
    return specs, dict(defs) if isinstance(defs, dict) else {}


def compact_type_from_jsonschema(spec: dict[str, Any]) -> str:
    """Render a JSON Schema property's compact Python-ish type string.

    Mirrors the v1 collapsing logic so the v2 envelope's ``type`` field
    stays byte-stable across the bump. Pydantic v2 sometimes stuffs a
    Python-ish repr like ``"Optional[my.module.X]"`` directly into the
    ``type`` key for opaque non-Pydantic types; when present alongside an
    ``anyOf`` with a ``{}`` opaque branch, that string takes precedence
    over the anyOf rendering (which would collapse to ``"None"`` and lose
    the upstream type-name signal).

    Otherwise: ``$ref`` becomes the target model name; ``anyOf`` becomes a
    ``" | "``-joined union; primitive types pass through with
    ``"null" -> "None"``.
    """
    type_repr: Any = spec.get("type")
    # Honour an explicit non-primitive ``type`` string before falling into
    # anyOf / ref unwinding. Pydantic emits ``type: "Optional[X]"`` for
    # opaque types whose anyOf branch is ``{}``; keeping that string
    # preserves the v1 shape byte-for-byte.
    if isinstance(type_repr, str) and type_repr not in _JSONSCHEMA_PRIMITIVES:
        return "None" if type_repr == "null" else type_repr
    if "$ref" in spec:
        return str(spec["$ref"]).rsplit("/", 1)[-1]
    if "anyOf" in spec:
        parts: list[str] = []
        for sub in spec["anyOf"]:
            if not isinstance(sub, dict):
                continue
            if "type" in sub:
                kind = sub["type"]
                part = "None" if kind == "null" else str(kind)
            elif "$ref" in sub:
                part = str(sub["$ref"]).rsplit("/", 1)[-1]
            else:
                continue
            if part not in parts:
                parts.append(part)
        return " | ".join(parts) if parts else "unknown"
    if isinstance(type_repr, list):
        return " | ".join("None" if t == "null" else str(t) for t in type_repr)
    if type_repr == "null":
        return "None"
    return str(type_repr) if type_repr is not None else "unknown"


# JSON Schema primitive type names recognised by ``compact_type_from_jsonschema``;
# anything else in a ``type`` string is treated as a Pydantic-stuffed Python-ish
# repr and kept verbatim.
_JSONSCHEMA_PRIMITIVES: frozenset[str] = frozenset(
    {"string", "integer", "number", "boolean", "array", "object", "null"}
)


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
    engine_params_defs: dict[str, Any] | None = None,
    sampling_params_defs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a discovered-schema envelope.

    ``schema_version`` is set from the module-level :data:`SCHEMA_VERSION`
    constant. v2.0.0 carries JSON-Schema-conformant per-field specs (enum,
    minimum, maximum, $defs, etc.) in addition to the legacy ``type`` /
    ``default`` keys; see the module docstring for the bump rationale.

    The optional ``engine_params_defs`` / ``sampling_params_defs`` dicts
    capture nested ``$defs`` blocks emitted by Pydantic / msgspec when a
    parameter's value is itself a structured model. They are emitted as
    top-level envelope keys only when non-empty so older single-class
    engines stay byte-identical apart from the bumped version.
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
        "discovery_method": discovery_method,
        "discovery_limitations": discovery_limitations,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if engine_params_defs:
        envelope["engine_params_defs"] = engine_params_defs
    if sampling_params_defs:
        envelope["sampling_params_defs"] = sampling_params_defs
    return envelope
