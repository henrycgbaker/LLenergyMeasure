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


def _resolve_pydantic_type(annotation: Any) -> type | None:
    """Return the Pydantic model in ``annotation`` (unwrapping ``X | None``), else None.

    Handles the common ``SubConfig`` and ``SubConfig | None`` / ``Optional[SubConfig]``
    shapes the dataclass walker meets on engine-args classes; nested generics
    (``list[SubConfig]``) are left as ``type: object`` since there is no single
    sub-config to ``$ref``. Recognises both ``pydantic.BaseModel`` subclasses and
    ``pydantic.dataclasses``-decorated classes (both expose ``model_json_schema``).
    """
    candidates: tuple[Any, ...]
    origin = get_origin(annotation)
    if origin is Union or origin is types.UnionType:
        candidates = tuple(a for a in get_args(annotation) if a is not type(None))
    elif origin is None:
        candidates = (annotation,)
    else:
        return None
    if len(candidates) != 1:
        return None
    candidate = candidates[0]
    if isinstance(candidate, type) and hasattr(candidate, "model_json_schema"):
        return candidate
    return None


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

    When a field's resolved type is a Pydantic model and a ``defs`` accumulator
    is supplied, the field is emitted as a JSON Schema ``$ref`` and the model's
    own ``model_json_schema()`` (its ``$defs`` plus its root definition) is
    folded into ``defs``. This surfaces the Pydantic sub-configs nested inside
    stdlib-dataclass engine-args (e.g. vllm ``EngineArgs``) that would otherwise
    flatten to ``type: object`` - the 2026-05-24 ``$defs`` resolution. Pass the
    same ``defs`` dict to :func:`make_envelope` to ship it.
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
            nested = _resolve_pydantic_type(hints.get(fld.name, fld.type))
            if nested is not None:
                _fold_model_defs(nested, defs)
                specs[fld.name] = {
                    "$ref": f"#/$defs/{nested.__name__}",
                    "default": jsonable(default),
                }
                continue
        specs[fld.name] = {
            "type": annotation_to_type_str(fld.type),
            "default": jsonable(default),
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

    ``model_json_schema(ref_template=...)`` returns ``{$defs: {...}, ...root...}``
    keyed under the standard ``#/$defs/<Name>`` namespace. We lift the root
    body under the model name and merge any transitively-referenced sub-models,
    so the envelope's ``$defs`` is self-contained (every ``$ref`` resolves).
    """
    try:
        schema = model.model_json_schema(ref_template="#/$defs/{model}")
    except Exception:
        return
    for name, body in schema.pop("$defs", {}).items():
        defs.setdefault(name, body)
    # The root schema body (title/properties/...) becomes this model's own def.
    defs.setdefault(model.__name__, schema)


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
    gained at least one constraint key (the D3 surface-trend metric - expected
    near zero on the current pins, where the entry classes carry few bounds the
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
    # creating a synchronize loop. Mirrors LLENERGY_VALIDATION_FROZEN_AT in
    # scripts/validate_rules.py.
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
    # point at. Preserve them rather than dropping at envelope assembly (the
    # 2026-05-24 ``$defs`` resolution); ``jsonable`` keeps the block free of
    # object-repr noise without re-flattening the canonical structure. Only
    # emitted when non-empty so additive-schema loaders stay simple.
    if defs:
        envelope["$defs"] = jsonable(defs)
    return envelope
