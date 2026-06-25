"""Shared substrate for the per-engine drift-completeness probers.

The landmark-resolution drift check (``scripts._drift``) catches a covered
landmark that VANISHED upstream (a stale path that no longer resolves). This
module is the complementary half: it surfaces live raise-bearing config
validators that the miner's covered set (the ``ast_targets`` registry) does NOT
yet cover - the "stale-ABSENT" relocation gap, where upstream relocates a check
into a new validator method the landmark set never named.

The detector + the uncovered-computation here are engine-agnostic; each engine
ships a thin prober (``probe_uncovered_validators``) that imports its live
config tree and feeds the seeds / covered-set / flat-hoist field set into
:func:`compute_uncovered_validators`.

Exclusions (a flagged validator must clear ALL of them to count as uncovered):

- Blob-only: a validator on a class that is NOT genuinely flat-hoisted is
  excluded. A class is treated as flat when it is in the seed set OR it shares
  at least three settable fields with the flat-hoist field set (the engine-args
  surface). Validators on deeply-nested blob sub-configs the user never reaches
  flat are noise.
- No settable non-private self-field: a validator whose only ``self.X``
  references are private/derived, or that reads a field-validator argument
  rather than a stored field, has no user-facing knob to flag and is excluded
  (covers all-private, derived, and field-validator-arg shapes).

False-negative guard: a validator-named method that is NOT defined in the
class's own source (inherited or wrapped, hence not statically analyzable) is
surfaced to the ``unanalyzable`` list rather than silently dropped, so the
census never claims coverage it cannot prove.

Torch-free at import: only stdlib (``ast``, ``inspect``, ``textwrap``,
``dataclasses``) plus :mod:`scripts.engine_producers._common`.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap

from scripts.engine_producers import _common


def _settable_fields(cls: type) -> set[str]:
    """Return the user-settable field names of ``cls``.

    Robust across pydantic (``__pydantic_fields__``), msgspec ``Struct``
    (``__struct_fields__`` - SamplingParams is a msgspec Struct), and stdlib
    dataclasses (``dataclasses.fields``). Returns an empty set for any other
    class shape.
    """
    pyd_fields = getattr(cls, "__pydantic_fields__", None)
    if pyd_fields:
        return set(pyd_fields.keys())
    struct_fields = getattr(cls, "__struct_fields__", None)
    if struct_fields:
        return set(struct_fields)
    if dataclasses.is_dataclass(cls):
        return {f.name for f in dataclasses.fields(cls)}
    return set()


def _self_fields(fn: ast.AST) -> list[str]:
    """Return the sorted set of ``X`` for every ``self.X`` attribute in ``fn``."""
    names: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            names.add(node.attr)
    return sorted(names)


def _has_raise(fn: ast.AST) -> bool:
    """Return True if ``fn`` contains any ``raise`` statement."""
    return any(isinstance(node, ast.Raise) for node in ast.walk(fn))


def _self_callees(fn: ast.AST) -> set[str]:
    """Return the method names ``X`` for every ``self.X(...)`` call in ``fn``.

    Used to model the miner's same-class helper-follow: a raise relocated into a
    helper that a covered method calls is still mined, so the completeness check
    must treat it as covered too (else it false-flags it as a gap).
    """
    out: set[str] = set()
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            out.add(node.func.attr)
    return out


def _own_methods(cls: type) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None:
    """Parse ``cls``'s own source and return ``{name: FunctionDef}`` for its methods.

    Returns ``None`` when the source is unreadable or the top node is not a
    ``ClassDef`` (callers surface that as an unanalyzable class).
    """
    try:
        node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    if not isinstance(node, ast.ClassDef):
        return None
    return {n.name: n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _effective_covered(
    cn: str, own: dict[str, ast.FunctionDef | ast.AsyncFunctionDef], covered: set[tuple[str, str]]
) -> set[str]:
    """Expand the directly-covered methods of class ``cn`` with the transitive
    closure of same-class ``self.method()`` callees.

    Mirrors the static miner's helper-follow (``descend`` over ``self.helper()``
    calls, each helper once): a covered method's body that calls a same-class
    helper makes that helper's raises mined too, so the completeness check must
    not flag the helper as an uncovered gap.
    """
    effective = {m for (c, m) in covered if c == cn}
    stack = list(effective)
    while stack:
        fn = own.get(stack.pop())
        if fn is None:
            continue
        for callee in _self_callees(fn):
            if callee in own and callee not in effective:
                effective.add(callee)
                stack.append(callee)
    return effective


def find_raise_validators(cls: type) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Find the raise-bearing validators defined on ``cls``'s own source.

    Returns ``(analyzable_raisers, unanalyzable)``:

    - ``analyzable_raisers``: ``(method_name, self_fields)`` for each candidate
      validator that is defined in this class's own source AND contains a
      ``raise``. ``self_fields`` are the sorted ``self.X`` names it references.
    - ``unanalyzable``: validator-named methods that are NOT defined in this
      class's own source (inherited or wrapped, so not statically analyzable)
      - surfaced per the false-negative guard, never dropped. When the whole
      class has no readable source, this is ``["<no-source>"]``.

    Candidate validator names = pydantic decorator names (model + field
    validators, guarded via ``getattr`` since not all classes carry
    ``__pydantic_decorators__``) UNION method names matching the imperative
    naming conventions: ``__post_init__`` / ``post_init`` or any of the
    ``verify`` / ``_verify`` / ``validate`` / ``_validate`` / ``check`` /
    ``_check`` prefixes. The prefix set is deliberately wide (both underscore
    and bare forms) because upstream uses both - e.g. vLLM's
    ``SchedulerConfig.verify_max_model_len`` has no leading underscore; missing
    it would blind the tripwire to exactly the relocation class it exists to
    catch.
    """
    candidates: set[str] = set()
    decorators = getattr(cls, "__pydantic_decorators__", None)
    if decorators is not None:
        candidates.update(getattr(decorators, "model_validators", {}).keys())
        candidates.update(getattr(decorators, "field_validators", {}).keys())
    for name in dir(cls):
        if name in {"__post_init__", "post_init"} or name.startswith(
            ("_verify", "verify", "validate", "_validate", "_check", "check")
        ):
            candidates.add(name)

    own_methods = _own_methods(cls)
    if own_methods is None:
        return [], [f"{cls.__name__}:<no-source>"]

    analyzable_raisers: list[tuple[str, list[str]]] = []
    unanalyzable: list[str] = []
    for name in sorted(candidates):
        fn = own_methods.get(name)
        if fn is not None:
            if _has_raise(fn):
                analyzable_raisers.append((name, _self_fields(fn)))
        else:
            # Validator-named but not defined in this class's own source:
            # inherited / wrapped, so it cannot be statically analyzed.
            unanalyzable.append(name)
    return analyzable_raisers, unanalyzable


def compute_uncovered_validators(
    *,
    roots: list[type],
    covered: set[tuple[str, str]],
    hoist_fields: set[str],
    seed_names: set[str],
) -> tuple[list[str], list[str]]:
    """Compute the uncovered + unanalyzable raise-validator surface.

    Walks every config class reachable from ``roots`` (via
    :func:`scripts.engine_producers._common.enumerate_config_classes`), finds
    each class's raise-bearing validators, and flags those NOT in ``covered``
    (the ``(class_short_name, method)`` pairs the miner already walks) once they
    clear the blob-only and no-settable-non-private-self-field exclusions.

    ``hoist_fields`` is the engine-args flat-hoist field set; ``seed_names`` is
    the set of class names treated as genuinely flat by definition. Returns
    ``(validators_uncovered, validators_unanalyzable)``, both sorted + deduped.
    """
    reachable: dict[str, type] = {}
    for r in roots:
        for c in _common.enumerate_config_classes(r):
            reachable[f"{c.__module__}.{c.__qualname__}"] = c

    uncovered: list[str] = []
    unanalyzable: list[str] = []
    for key in sorted(reachable):
        cls = reachable[key]
        cn = cls.__name__
        raisers, unanalyz = find_raise_validators(cls)
        for u in unanalyz:
            unanalyzable.append(f"{cn}.{u}")
        # Effective coverage = the covered methods of this class PLUS the
        # transitive closure of their same-class self.helper() callees, so a
        # raise the miner reaches via helper-follow is not false-flagged.
        own = _own_methods(cls)
        effective = (
            _effective_covered(cn, own, covered)
            if own is not None
            else {m for (c, m) in covered if c == cn}
        )
        settable = _settable_fields(cls)
        # Blob-only exclusion: a class is genuinely flat when it is a seed or
        # shares >=3 settable fields with the flat-hoist (engine-args) surface.
        # The threshold demotes blob sub-configs that share an incidental field
        # name or two with the engine-args surface (e.g. CompilationConfig)
        # while keeping the genuinely-hoisted scalar configs.
        genuinely_flat = cn in seed_names or len(settable & hoist_fields) >= 3
        for method, self_fields in raisers:
            if method in effective:
                continue
            real = [f for f in self_fields if f in settable and not f.startswith("_")]
            if not genuinely_flat:
                continue
            if not real:
                # No settable non-private self.X (all-private / derived /
                # field-validator-arg) - nothing user-facing to flag.
                continue
            uncovered.append(f"{cn}.{method}")
    return sorted(set(uncovered)), sorted(set(unanalyzable))
