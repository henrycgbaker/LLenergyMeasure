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
from scripts.engine_producers._base import extract_condition_fields

# Imperative validator naming conventions. Deliberately wide (both underscore
# and bare forms) because upstream uses both - e.g. vLLM's
# ``SchedulerConfig.verify_max_model_len`` has no leading underscore; missing it
# would blind the tripwire to exactly the relocation class it exists to catch.
_VALIDATOR_NAME_PREFIXES = ("_verify", "verify", "validate", "_validate", "_check", "check")

# Pydantic validator decorator names (as they appear in source / on the class).
_VALIDATOR_DECORATORS = frozenset(
    {"field_validator", "model_validator", "validator", "root_validator"}
)

# Base classes whose own bodies carry no domain raise-validators, so a class
# inheriting ONLY from these needs no "<base not in source>" unanalyzable marker
# in the AST-source path. Framework/stdlib roots only; engine-specific safe
# bases are passed in by the per-engine prober.
_SAFE_BASE_NAMES = frozenset(
    {
        "object",
        "BaseModel",
        "StrictBaseModel",
        "Enum",
        "IntEnum",
        "StrEnum",
        "str",
        "int",
        "float",
        "ABC",
        "Generic",
        "Protocol",
        "NamedTuple",
        "TypedDict",
        "Struct",
    }
)


def _is_validator_named(name: str) -> bool:
    """True if ``name`` matches an imperative validator naming convention.

    ``__post_init__`` / ``post_init`` or any of the :data:`_VALIDATOR_NAME_PREFIXES`.
    Shared by the live-class and AST-source detectors so both stay in step.
    """
    return name in {"__post_init__", "post_init"} or name.startswith(_VALIDATOR_NAME_PREFIXES)


def _init_param_names(cls: type) -> set[str]:
    """Return the explicit ``__init__`` parameter names of ``cls`` (constructor knobs).

    Excludes ``self`` / ``*args`` / ``**kwargs``; empty when ``__init__`` is not
    introspectable (C-extension / builtin). Some configs declare a single (or no)
    field yet accept their real user knobs as explicit ``__init__`` params set as
    instance attributes - e.g. transformers' ``BitsAndBytesConfig`` (one dataclass
    field ``quant_method`` but ten ``__init__`` knobs like ``load_in_4bit``). A
    no-op for dataclass / msgspec / auto-``__init__`` classes (their ``__init__``
    mirrors the fields) and for pydantic (whose ``__init__`` is ``(self, **data)``).
    """
    try:
        params = inspect.signature(cls.__init__).parameters
    except (ValueError, TypeError):
        return set()
    return {
        n
        for n, p in params.items()
        if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _settable_fields(cls: type) -> set[str]:
    """Return the user-settable field names of ``cls``.

    Robust across pydantic (``__pydantic_fields__``), msgspec ``Struct``
    (``__struct_fields__`` - SamplingParams is a msgspec Struct), and stdlib
    dataclasses (``dataclasses.fields``), UNIONED with the explicit ``__init__``
    parameter knobs (:func:`_init_param_names`) so a config whose knobs are
    constructor params rather than declared fields (BitsAndBytesConfig) is not
    silently under-reported. Returns an empty set for an opaque class with no
    declared fields and a ``**kwargs``-only constructor (e.g. GenerationConfig),
    which the empty-settable plain-class fallback then handles.
    """
    fields: set[str] = set()
    pyd_fields = getattr(cls, "__pydantic_fields__", None)
    if pyd_fields:
        fields |= set(pyd_fields.keys())
    struct_fields = getattr(cls, "__struct_fields__", None)
    if struct_fields:
        fields |= set(struct_fields)
    if dataclasses.is_dataclass(cls):
        fields |= {f.name for f in dataclasses.fields(cls)}
    fields |= _init_param_names(cls)
    return fields


def _self_fields(fn: ast.AST) -> list[str]:
    """Return the sorted set of ``X`` for every ``self.X`` attribute in ``fn``.

    Delegates to the canonical AST primitive ``_base.extract_condition_fields``
    (a generic ``self.X`` attribute walk); the only delta is the sorted() wrap.
    """
    return sorted(extract_condition_fields(fn))


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


def find_raise_validators(
    cls: type,
    own_methods: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] | None = None,
) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Find the raise-bearing validators defined on ``cls``'s own source.

    ``own_methods`` may be passed by a caller that already parsed the class
    (e.g. :func:`compute_uncovered_validators`, which also needs them for the
    helper-follow), to avoid re-parsing the source; omit it for standalone use.

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
    naming conventions (:func:`_is_validator_named`).
    """
    candidates: set[str] = set()
    decorators = getattr(cls, "__pydantic_decorators__", None)
    if decorators is not None:
        candidates.update(getattr(decorators, "model_validators", {}).keys())
        candidates.update(getattr(decorators, "field_validators", {}).keys())
    for name in dir(cls):
        if _is_validator_named(name):
            candidates.add(name)

    if own_methods is None:
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
        # Parse the class source ONCE and share it between the raiser-find and
        # the helper-follow coverage expansion.
        own = _own_methods(cls)
        raisers, unanalyz = find_raise_validators(cls, own_methods=own)
        for u in unanalyz:
            unanalyzable.append(f"{cn}.{u}")
        settable = _settable_fields(cls)
        # Blob-only exclusion (loop-invariant): a class is genuinely flat when it
        # is a seed or shares >=3 settable fields with the flat-hoist (engine-args)
        # surface. The threshold demotes blob sub-configs that share an incidental
        # field name or two (e.g. CompilationConfig) while keeping the genuinely-
        # hoisted scalar configs. Skip the whole method scan for non-flat classes.
        genuinely_flat = cn in seed_names or len(settable & hoist_fields) >= 3
        if not genuinely_flat:
            continue
        # Effective coverage = the covered methods of this class PLUS the
        # transitive closure of their same-class self.helper() callees, so a
        # raise the miner reaches via helper-follow is not false-flagged.
        effective = (
            _effective_covered(cn, own, covered)
            if own is not None
            else {m for (c, m) in covered if c == cn}
        )
        for method, self_fields in raisers:
            if method in effective:
                continue
            real = _real_self_fields(self_fields, settable)
            if not real:
                # No settable non-private self.X (all-private / derived /
                # field-validator-arg) - nothing user-facing to flag.
                continue
            uncovered.append(f"{cn}.{method}")
    return sorted(set(uncovered)), sorted(set(unanalyzable))


def _real_self_fields(self_fields: list[str], settable: set[str]) -> list[str]:
    """Return the user-facing knobs among ``self_fields`` (the flag predicate).

    Normally a knob is a non-private field present in ``settable``. When
    ``settable`` is EMPTY the class is a plain Python class with no introspectable
    field set (e.g. transformers' ``GenerationConfig``, which assigns its knobs in
    ``__init__`` rather than as declared fields); a seed plain class still has
    user-facing public attributes, so fall back to every non-private ``self.X``
    rather than zeroing the flag (which would silently blind the tripwire). Only
    reached for genuinely-flat classes (the blob-only exclusion already ran).
    """
    if not settable:
        return [f for f in self_fields if not f.startswith("_")]
    return [f for f in self_fields if f in settable and not f.startswith("_")]


# ---------------------------------------------------------------------------
# AST-source twins (for engines that cannot import their library on host)
# ---------------------------------------------------------------------------
#
# tensorrt has no host CUDA, so its miner walks the EXTRACTED SOURCE via AST and
# never imports tensorrt_llm. The live-class detector above cannot run there
# (no live types, no inspect.getsource, no __pydantic_decorators__). These twins
# operate purely on parsed ``ast.ClassDef`` nodes and reuse the same generic
# FunctionDef helpers (``_has_raise`` / ``_self_fields`` / ``_self_callees`` /
# ``_effective_covered``) so the two paths stay in step.


def _own_methods_ast(
    classdef: ast.ClassDef,
) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return ``{name: FunctionDef}`` for ``classdef``'s own (body-defined) methods."""
    return {
        n.name: n for n in classdef.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _has_validator_decorator(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """True if ``fn`` carries a pydantic validator decorator (in :data:`_VALIDATOR_DECORATORS`).

    Matches both ``@field_validator(...)`` (an ``ast.Call``) and bare ``@validator``,
    via attribute (``pydantic.field_validator``) or plain name.
    """
    for dec in fn.decorator_list:
        node = dec.func if isinstance(dec, ast.Call) else dec
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
        if name in _VALIDATOR_DECORATORS:
            return True
    return False


def _settable_fields_ast(classdef: ast.ClassDef) -> set[str]:
    """Return the declared field names of ``classdef`` from its source body.

    Captures annotated class-body assignments (``name: type [= ...]`` - the
    pydantic / dataclass field shape) plus plain ``name = ...`` class vars.
    Private (leading-underscore) names are kept here (the flag predicate filters
    them) but dunders are skipped. The AST twin of :func:`_settable_fields`.
    """
    fields: set[str] = set()
    for stmt in classdef.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields.add(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    fields.add(tgt.id)
    return {f for f in fields if not f.startswith("__")}


def _settable_fields_ast_inherited(
    cls_name: str, classdefs: dict[str, ast.ClassDef], _seen: set[str] | None = None
) -> set[str]:
    """Union of ``cls_name``'s own declared fields and those of its PARSED ancestors.

    The AST analogue of pydantic's merged ``__pydantic_fields__`` (which already
    includes inherited fields - the live and AST paths must agree). Climbs base
    classes present in ``classdefs``; out-of-source bases contribute nothing
    (their fields are invisible to the AST - the base-class false-negative guard
    surfaces them separately). Cycle-guarded. Without this the engine-args
    surface of an inheritance-based config (tensorrt: ``TrtLlmArgs`` inherits
    ~48 fields from ``BaseLlmArgs``) collapses to the few own-body fields, which
    kills the ``>=3`` blob-escape and blinds the tripwire to relocations into
    not-yet-named config classes.
    """
    if _seen is None:
        _seen = set()
    if cls_name in _seen or cls_name not in classdefs:
        return set()
    _seen.add(cls_name)
    node = classdefs[cls_name]
    fields = _settable_fields_ast(node)
    for base in _base_names(node):
        fields |= _settable_fields_ast_inherited(base, classdefs, _seen)
    return fields


def find_raise_validators_ast(
    classdef: ast.ClassDef,
) -> list[tuple[str, list[str]]]:
    """AST-source twin of :func:`find_raise_validators`.

    Candidate validator names = body methods carrying a pydantic validator
    decorator UNION the imperative naming conventions (:func:`_is_validator_named`),
    same as the live detector. Returns ``(method_name, self_fields)`` for each
    candidate whose body contains a ``raise``. (Inherited/wrapped validators are
    handled at the enumeration level via the base-class guard, since the AST path
    cannot see them on a single ClassDef the way ``dir(cls)`` can.)
    """
    own = _own_methods_ast(classdef)
    raisers: list[tuple[str, list[str]]] = []
    for name in sorted(own):
        fn = own[name]
        if (_is_validator_named(name) or _has_validator_decorator(fn)) and _has_raise(fn):
            raisers.append((name, _self_fields(fn)))
    return raisers


def _base_names(classdef: ast.ClassDef) -> list[str]:
    """Return the simple names of ``classdef``'s base classes (best-effort)."""
    names: list[str] = []
    for base in classdef.bases:
        if isinstance(base, ast.Name):
            names.append(base.id)
        elif isinstance(base, ast.Attribute):
            names.append(base.attr)
    return names


def _missing_bases_ast(
    cls_name: str,
    classdefs: dict[str, ast.ClassDef],
    safe_bases: frozenset[str],
    _seen: set[str] | None = None,
) -> list[str]:
    """Transitive-closure base names reachable from ``cls_name`` that are neither
    parsed (in ``classdefs``) nor ``safe`` - i.e. their inherited validators are
    invisible to the AST.

    Walks THROUGH parsed-but-non-flat intermediates (which the per-class scan
    skips) so a flat class re-parented through an unparsed grandparent is still
    surfaced, matching the live path's full-MRO ``dir(cls)`` visibility and the
    design's "never silently dropped" contract.
    """
    if _seen is None:
        _seen = set()
    missing: list[str] = []
    node = classdefs.get(cls_name)
    if node is None:
        return missing
    for base in _base_names(node):
        if base in _seen:
            continue
        _seen.add(base)
        if base in classdefs:
            missing.extend(_missing_bases_ast(base, classdefs, safe_bases, _seen))
        elif base not in safe_bases:
            missing.append(base)
    return missing


def compute_uncovered_validators_ast(
    *,
    classdefs: dict[str, ast.ClassDef],
    covered: set[tuple[str, str]],
    hoist_fields: set[str],
    seed_names: set[str],
    safe_bases: frozenset[str] = _SAFE_BASE_NAMES,
) -> tuple[list[str], list[str]]:
    """AST-source twin of :func:`compute_uncovered_validators`.

    ``classdefs`` maps class short-name -> its parsed ``ast.ClassDef`` (every
    class the prober extracted from the source tree). Flags raise-bearing
    validators NOT in ``covered`` once they clear the same blob-only +
    no-settable-non-private-self-field exclusions.

    False-negative guard (the AST analogue of the inherited/wrapped surfacing):
    a class with a base that is NOT among the parsed ``classdefs`` and NOT in
    ``safe_bases`` may inherit validators the AST cannot see -> that base is
    surfaced to ``validators_unanalyzable`` (``"Cls:<base Base not in source>"``),
    never silently assumed clean. Returns ``(uncovered, unanalyzable)`` sorted+deduped.
    """
    uncovered: list[str] = []
    unanalyzable: list[str] = []
    for cn in sorted(classdefs):
        node = classdefs[cn]
        # Inheritance-aware settable (matches the live path's merged fields), so a
        # class whose engine-arg surface lives on a parsed base still clears the
        # blob-escape and has its validators' self.X correctly recognised as knobs.
        settable = _settable_fields_ast_inherited(cn, classdefs)
        genuinely_flat = cn in seed_names or len(settable & hoist_fields) >= 3
        if not genuinely_flat:
            # Out-of-scope blob (e.g. a stdlib JSONEncoder / metaclass): not a
            # config knob surface, so its out-of-source bases are noise, not a gap.
            continue
        # False-negative guard - only meaningful for an in-scope flat class: any
        # base in the TRANSITIVE chain NOT in the parsed source may hide inherited
        # validators the AST cannot see, so surface it (never silently drop).
        for base in _missing_bases_ast(cn, classdefs, safe_bases):
            unanalyzable.append(f"{cn}:<base {base} not in source>")
        own = _own_methods_ast(node)
        effective = _effective_covered(cn, own, covered)
        for method, self_fields in find_raise_validators_ast(node):
            if method in effective:
                continue
            if not _real_self_fields(self_fields, settable):
                continue
            uncovered.append(f"{cn}.{method}")
    return sorted(set(uncovered)), sorted(set(unanalyzable))
