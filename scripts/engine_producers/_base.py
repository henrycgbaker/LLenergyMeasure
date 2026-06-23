"""Shared infrastructure for per-engine validation-invariant miners.

Miner depth is fixed at 1 (same module, no helper-call tracing). This
module ships the shared LEAF primitives the per-engine miners compose to
extract validation invariants from pinned library source: the AST text
helpers, the candidate output type, and the structured error types. Each
engine owns its own walker and detector set - the invariant shapes genuinely
diverge per engine - so this module holds no pattern-detector framework, only
the mechanical primitives every engine reuses.

- :class:`InvariantCandidate` - the miner output type, serialised to the YAML
  corpus entry shape in :mod:`llenergymeasure.config.engine_rules.loader`.
- :class:`MinerLandmarkMissingError` - fail-loud exception CI treats as fatal
  when the producer's AST landmarks no longer resolve under the live library.
- AST helpers (:func:`extract_condition_fields`, :func:`resolve_local_assign`,
  etc.) - deterministic, stateless primitives for AST-based miners.

Tests cover each primitive on synthetic AST fixtures; the per-engine miners
run against pinned real libraries.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Literal, get_args, get_origin

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

Severity = Literal["error", "warn", "dormant"]
EmissionChannel = Literal[
    "warnings_warn",
    "logger_warning",
    "logger_warning_once",
    "minor_issues_dict",
    "runtime_exception",
    "none",
]


@dataclass
class MinerSource:
    """Provenance for a miner-extracted invariant candidate."""

    path: str
    method: str
    line_at_scan: int


@dataclass
class InvariantCandidate:
    """One extracted invariant candidate.

    Serialised verbatim into ``src/llenergymeasure/engines/{engine}/rules.proposed.yaml``
    after human review. Field names match the corpus schema so no
    translation step is needed between miner output and corpus entry.
    """

    id: str
    engine: str
    library: str
    invariant_under_test: str
    severity: Severity
    native_type: str
    miner_source: MinerSource
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    expected_outcome: dict[str, Any]
    message_template: str | None
    references: list[str] = field(default_factory=list)
    added_by: str = "static_miner"
    added_at: str = ""


# ---------------------------------------------------------------------------
# Error types (all inherit from a common base so per-engine miners can
# raise-or-collect uniformly at CI time)
# ---------------------------------------------------------------------------


class MinerError(Exception):
    """Base class for structured miner failures."""


class MinerLandmarkMissingError(MinerError):
    """Raised when an expected source landmark (class/method/file) is missing.

    Library refactors (class renamed, method split, file relocated) trip this
    error and the miner refuses to emit partial output. This is load-bearing
    for the "silent coverage loss becomes a visible CI failure" contract.
    """

    def __init__(self, landmark: str, detail: str = "") -> None:
        msg = f"Miner landmark missing: {landmark}"
        if detail:
            msg = f"{msg} ({detail})"
        super().__init__(msg)
        self.landmark = landmark
        self.detail = detail


# ---------------------------------------------------------------------------
# AST primitives
# ---------------------------------------------------------------------------


def call_func_path(call: ast.Call) -> list[str] | None:
    """Return dotted path for a ``Call`` node's func, or ``None`` if opaque.

    ``logger.warning(...)`` → ``["logger", "warning"]``.
    ``foo()()`` → ``None`` (not a pure attribute/name chain).
    """
    parts: list[str] = []
    node: ast.expr = call.func
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def first_string_arg(call: ast.Call) -> str | None:
    """First string-like positional argument of a Call, or ``None``.

    Returns a substitution-template string for the corpus's
    ``message_template`` field. Recognised input shapes:

    - ``ast.Constant(str)`` - returned as-is.
    - ``ast.JoinedStr`` (f-string) - interpolations rendered to ``{name}``
      placeholders matching the runtime substitution vocabulary; ``self.X``
      collapses to ``{X}``.
    - ``"literal {x}".format(...)`` - the LHS literal returned as the template.
      Variable-template forms (``template_var.format(...)``) need scope
      resolution unavailable at this layer; they fall through to ``None``.

    All three avoid returning literal Python source - validation-CI substring
    matching against the live library's rendered string fails on source.
    """
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
        if isinstance(arg, ast.JoinedStr):
            return render_joinedstr_template(arg)
        template = format_call_template(arg)
        if template is not None:
            return template
    return None


def format_call_template(node: ast.expr) -> str | None:
    """If ``node`` is ``"literal".format(...)``, return the LHS template literal.

    Returns ``None`` for any other expression shape, including variable-template
    forms (``template_var.format(...)``) that would need scope resolution to
    render. Shared by :func:`first_string_arg` and the minor-issues-dict-assign
    detectors so the "no source-leak" policy stays in one place.
    """
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
        and isinstance(node.func.value, ast.Constant)
        and isinstance(node.func.value.value, str)
    ):
        return node.func.value.value
    return None


def render_joinedstr_template(node: ast.JoinedStr) -> str:
    """Render an f-string AST to a substitution-template string.

    ``FormattedValue`` nodes become ``{name}`` placeholders; ``self.X``
    collapses to ``{X}``, other expressions use the unparsed source as the
    placeholder name. ``Constant`` parts kept verbatim.
    """
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            inner = value.value
            if (
                isinstance(inner, ast.Attribute)
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
            ):
                parts.append(f"{{{inner.attr}}}")
            else:
                parts.append(f"{{{ast.unparse(inner)}}}")
    return "".join(parts)


def render_binop_concat_template(node: ast.expr) -> str | None:
    """Render a ``+``-concatenation AST to a substitution-template string.

    Walks an ``ast.BinOp`` Add chain (or any leaf operand): ``Constant(str)``
    parts kept verbatim; ``self.X`` and ``str(self.X)`` / ``repr(self.X)``
    render as ``{X}`` placeholders. Returns ``None`` if any operand can't be
    rendered cleanly - preferable to leaking literal Python source via
    ``ast.unparse``, which breaks validation-CI substring matching.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return f"{{{node.attr}}}"
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("str", "repr")
        and len(node.args) == 1
    ):
        return render_binop_concat_template(node.args[0])
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = render_binop_concat_template(node.left)
        right = render_binop_concat_template(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def extract_condition_fields(condition: ast.expr) -> set[str]:
    """Return the set of ``self.<field>`` attribute names referenced in ``condition``.

    Used by the ``condition_references_self`` filter (invariant must reference at
    least one public field of the native type).
    """
    fields: set[str] = set()
    for node in ast.walk(condition):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            fields.add(node.attr)
    return fields


def extract_assign_target(stmt: ast.Assign) -> str | None:
    """Return ``"self.<attr>"`` → ``<attr>`` for a single-target self-assignment.

    ``None`` for anything else (tuple unpacking, subscripts, non-self targets).
    """
    if len(stmt.targets) != 1:
        return None
    target = stmt.targets[0]
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def resolve_local_assign(func: ast.FunctionDef, name: str) -> str | None:
    """Find the first ``name = <string-literal>`` inside ``func`` and return the literal.

    Used for HF's ``greedy_wrong_parameter_msg`` pattern - the message template
    is a local variable defined earlier in the same function body.

    **Scope limitation:** only scans top-level statements in ``func.body``.
    Assignments nested in ``if`` / ``try`` / ``with`` / ``for`` blocks are
    not followed. Returns the *first* matching assignment, so a function
    that rebinds the name later will still surface the earliest value -
    fine for message templates that are constant per function call,
    brittle for names the function reassigns. Suits current HF validate()
    shape; if a future library uses branch-local message templates, the
    miner calling site needs a richer resolver.
    """
    for node in func.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if (
            isinstance(tgt, ast.Name)
            and tgt.id == name
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return node.value.value
    return None


def extract_loop_literal_iterable(loop: ast.For) -> list[Any] | None:
    """Return the literal list/tuple a ``for`` loop iterates over, or ``None``.

    ``for arg in [a, b]:`` → ``[a, b]``; ``for arg in self.x:`` → ``None``.
    Enables one parameterised invariant per loop when the iterable is AST-static.
    """
    iter_node = loop.iter
    if not isinstance(iter_node, (ast.List, ast.Tuple)):
        return None
    values: list[Any] = []
    for elt in iter_node.elts:
        if isinstance(elt, ast.Constant):
            values.append(elt.value)
        else:
            return None
    return values


# ---------------------------------------------------------------------------
# Class helpers
# ---------------------------------------------------------------------------


def find_class(module: ast.Module, class_name: str) -> ast.ClassDef | None:
    """Return the first ``ClassDef`` named ``class_name`` in ``module``."""
    for node in ast.iter_child_nodes(module):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    return None


def find_method(cls: ast.ClassDef, method_name: str) -> ast.FunctionDef | None:
    """Return the first ``FunctionDef`` named ``method_name`` on ``cls``."""
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name == method_name:
            return item
    return None


def violates_numeric(op_key: str, threshold: Any) -> Any:
    """Return a value that violates the numeric predicate ``op_key threshold``."""
    if op_key == ">":
        return threshold
    if op_key == ">=":
        return threshold - 1 if isinstance(threshold, int) else threshold - 1.0
    if op_key == "<":
        return threshold
    if op_key == "<=":
        return threshold + 1 if isinstance(threshold, int) else threshold + 1.0
    if op_key == "multiple_of":
        return threshold + 1 if isinstance(threshold, int) else threshold + 0.5
    raise ValueError(f"Unknown numeric op {op_key!r}")


def satisfies_numeric(op_key: str, threshold: Any) -> Any:
    """Return a value that satisfies the numeric predicate ``op_key threshold``."""
    if op_key == ">":
        return threshold + 1 if isinstance(threshold, int) else threshold + 1.0
    if op_key == ">=":
        return threshold
    if op_key == "<":
        return threshold - 1 if isinstance(threshold, int) else threshold - 1.0
    if op_key == "<=":
        return threshold
    if op_key == "multiple_of":
        return threshold * 2 if isinstance(threshold, (int, float)) else threshold
    raise ValueError(f"Unknown numeric op {op_key!r}")


def slug(value: Any) -> str:
    """Stable, filesystem-safe slug for embedding values in invariant ids."""
    return str(value).replace("-", "neg").replace(".", "p").replace(" ", "_").replace("/", "_")


def extract_literal_values(annotation: Any) -> tuple[Any, ...] | None:
    """Return Literal values if ``annotation`` is a (possibly nested) Literal.

    Handles bare ``Literal[...]``, ``Optional[Literal[...]]``, and
    ``Annotated[Literal[...], ...]`` shapes. Returns ``None`` when no Literal
    is present.
    """
    origin = get_origin(annotation)
    if origin is Literal:
        return get_args(annotation)
    if origin is not None:
        for arg in get_args(annotation):
            inner = extract_literal_values(arg)
            if inner is not None:
                return inner
    return None
