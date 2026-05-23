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
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Union, get_args, get_origin

SCHEMA_VERSION = "1.0.0"

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


def dataclass_fields_to_specs(
    cls: type, *, skip_private: bool = False
) -> dict[str, dict[str, Any]]:
    """Extract ``{name: {type, default}}`` specs from a dataclass.

    Resolves ``default_factory`` by calling it (swallowing errors to ``None``)
    so downstream JSON stays concrete. Types are rendered via
    ``annotation_to_type_str``.
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
        specs[fld.name] = {
            "type": annotation_to_type_str(fld.type),
            "default": jsonable(default),
        }
    return specs


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
) -> dict[str, Any]:
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
        "discovery_method": discovery_method,
        "discovery_limitations": discovery_limitations,
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }


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
