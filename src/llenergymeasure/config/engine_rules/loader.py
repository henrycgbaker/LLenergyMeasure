"""Load, match, and render engine validation rules from the shipped corpus.

Each engine ships exactly one rules file at
``src/llenergymeasure/engines/{engine}/rules.yaml``. That file is the sole
runtime source of engine-correctness knowledge: the loader reads nothing else
(no proposed/validated split, no overlay merge). Every rule is parsed into a
typed :class:`Rule` entry carrying a match predicate (operators defined in
:func:`evaluate_predicate`), an optional message template, and a
:class:`Provenance` block.

Severity is a CLOSED two-value enum defined by the autonomous runtime action a
match triggers:

- ``error``   - reject the config at expansion (the engine would raise).
- ``dormant`` - canonicalise / dedup (the engine silently ignores or aliases a
  field). Dormant rules may carry ``normalised_fields`` naming the paths the
  engine drives back to their default.

There is no ``warn`` severity: the study workflow records effective parameters,
which surfaces dormancy organically, and invalid configs never run.

Provenance is metadata only. It records where a rule came from and how it was
verified, for reviewer legibility and the absorb/verification workflow - it
never influences runtime behaviour (no severity coupling).

Design mirror: this module parallels :mod:`llenergymeasure.config.schema_loader`
from parameter-discovery - same envelope validation
(:class:`UnsupportedSchemaVersionError` on major-version mismatch), same
per-instance caching for test isolation, same lazy load pattern.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

SUPPORTED_MAJOR_VERSION = 1
"""Major version the loader knows how to parse.

Raised via :class:`UnsupportedSchemaVersionError` on mismatch; the loader
refuses partial reads to avoid silently accepting a future schema shape.
"""


Severity = Literal["error", "dormant"]
"""Autonomous runtime action a rule match produces. Closed two-value enum.

- ``error``   - reject the config at expansion (the engine raises on it).
- ``dormant`` - canonicalise + dedup: the engine silently ignores or coerces a
  field, so distinct declared configs resolve to the same effective config.
"""

Source = Literal["deterministic_miner", "analyst", "manual", "migrated"]
"""Where a rule was first proposed. Provenance only; never gates runtime.

- ``deterministic_miner`` - derived from engine source (AST scan, pydantic /
  msgspec / dataclass field introspection). Re-derivable on a library bump.
- ``analyst``             - proposed by a frontier-LLM cold read with a cited
  ``file:line``, then confirmed by the verification ladder.
- ``manual``              - hand-encoded by a maintainer for cases the
  automated proposers cannot reach.
- ``migrated``            - carried in from an earlier corpus topology.
"""

Verified = Literal["construction", "runtime", "human"]
"""How a rule's claim was confirmed. Provenance only; never gates runtime.

- ``construction`` - building the offending config in the engine raises (the
  constraint is enforced at object construction / field validation).
- ``runtime``      - confirmed by exercising the engine's validation entry
  point (e.g. transformers ``GenerationConfig.validate``).
- ``human``        - signed off by a maintainer where no probe is available.
"""

VALID_SEVERITY: frozenset[str] = frozenset(get_args(Severity))
VALID_SOURCE: frozenset[str] = frozenset(get_args(Source))
VALID_VERIFIED: frozenset[str] = frozenset(get_args(Verified))


class RuleCorpusError(ValueError):
    """A rules corpus file is malformed or violates the closed schema."""


class UnsupportedSchemaVersionError(RuleCorpusError):
    """Rules corpus has a ``schema_version`` major the loader cannot parse."""


class UnknownEnumValueError(RuleCorpusError):
    """A rule carries a closed-enum value outside the permitted set.

    Covers ``severity`` and the two provenance enums (``source`` /
    ``verified``). Subclassed per field for callers that want to distinguish.
    """


class UnknownSeverityError(UnknownEnumValueError):
    """A rule has a ``severity`` value outside :data:`Severity`."""


class UnknownSourceError(UnknownEnumValueError):
    """A rule has a ``provenance.source`` value outside :data:`Source`."""


class UnknownVerifiedError(UnknownEnumValueError):
    """A rule has a ``provenance.verified`` value outside :data:`Verified`."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Provenance:
    """Where a rule came from and how it was verified. Metadata only.

    Nothing in the runtime match path reads this: it exists for reviewer
    legibility, the citation checker, and the absorb/verification workflow.
    Coupling severity to provenance is explicitly disallowed (an LLM-proposed
    rule and a deterministically-mined rule with the same predicate must
    behave identically at runtime).
    """

    source: str
    verified: str
    engine_version: str
    citation: str | None = None
    date: str = ""


@dataclass(frozen=True)
class RuleMatch:
    """Result of a rule matching a concrete config.

    ``declared_value`` is the user-set value for the subject field (corpus
    convention puts precondition fields first and the subject field last).

    ``effective_value`` is reserved for the value-aliasing dormancy case (the
    engine remaps a declared value to a different effective one). It is always
    ``None`` today: ``try_match`` does not populate it, because dormant rules
    express the strip case via ``normalised_fields`` rather than an inline
    remap.
    """

    rule: Rule
    declared_value: Any
    effective_value: Any | None = None
    matched_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Rule:
    """One engine validation rule parsed from ``rules.yaml``.

    Construction goes through :func:`_parse_rule`; tests may instantiate
    directly. ``match_fields`` maps dotted canonical field paths to predicate
    specs (see :func:`evaluate_predicate`).
    """

    id: str
    engine: str
    severity: str
    match_fields: dict[str, Any]
    provenance: Provenance
    message_template: str | None = None
    normalised_fields: tuple[str, ...] = ()
    """Canonical paths a ``dormant`` rule drives back to their default.

    Consumed by the sweep library-resolution dedup to canonicalise
    field-inert configs. Empty for ``error`` rules.
    """

    def try_match(self, config: Any) -> RuleMatch | None:
        """Return an :class:`RuleMatch` if every predicate in ``match_fields`` holds.

        Field paths are dotted (``"transformers.sampling_params.temperature"``)
        and resolve against ``config`` attribute-by-attribute, tolerating
        Pydantic models, dataclasses, and plain dicts.

        Predicate specs may carry ``@field_path`` references on the right-hand
        side of any operator. References resolve against the same ``config``
        before predicate evaluation. Bare references (``@num_beams``) resolve
        as siblings of the predicate's field; dotted references
        (``@transformers.sampling_params.num_return_sequences``) resolve from
        the config root.

        ``declared_value`` on the returned match is the last field's value -
        corpus convention puts precondition fields first and the subject field
        last.
        """
        matched: dict[str, Any] = {}
        last_value: Any = None
        for path, spec in self.match_fields.items():
            actual = resolve_field_path(config, path)
            resolved_spec = _resolve_field_refs_in_spec(spec, config, path)
            if not evaluate_predicate(actual, resolved_spec):
                return None
            matched[path] = actual
            last_value = actual
        return RuleMatch(rule=self, declared_value=last_value, matched_fields=matched)

    def render_message(self, match: RuleMatch) -> str:
        """Substitute ``{declared_value}`` / ``{effective_value}`` / ``{invariant_id}`` in the template.

        ``matched_fields`` is keyed by full dotted paths
        (``vllm.sampling_params.min_tokens``), which are not valid
        ``str.format`` placeholder names, so each key is also exposed under its
        bare leaf name (``{min_tokens}``) - the corpus authoring convention.

        Uses ``str.format`` with permissive defaults - templates that reference
        a still-missing key fall back to the raw template rather than raising at
        user-facing time. The fallback does NOT prefix the rule id: the sole
        caller (:meth:`ExperimentConfig._apply_rules`) already annotates every
        rendered message with ``[rule.id]``.
        """
        if self.message_template is None:
            return "<no message template>"
        # Seed the full dotted paths first, then add leaf names via setdefault:
        # a dot-free path is never re-added under its own key (which would make
        # ``.format`` raise "multiple values"), and on a leaf collision between
        # two distinct paths the first-seen path wins (deterministic rendering).
        fmt_kwargs: dict[str, Any] = dict(match.matched_fields)
        for path, value in match.matched_fields.items():
            fmt_kwargs.setdefault(path.rsplit(".", 1)[-1], value)
        try:
            return self.message_template.format(
                declared_value=match.declared_value,
                effective_value=match.effective_value,
                invariant_id=self.id,
                **fmt_kwargs,
            )
        except (KeyError, IndexError):
            return self.message_template


@dataclass(frozen=True)
class EngineRules:
    """Parsed rules corpus for one engine."""

    engine: str
    schema_version: str
    engine_version: str
    rules: tuple[Rule, ...]


# ---------------------------------------------------------------------------
# Predicate engine
# ---------------------------------------------------------------------------


_FIELD_REF_PREFIX = "@"


def spec_has_field_ref(spec: Any) -> bool:
    """Return True if ``spec`` contains any ``@field_path`` string anywhere.

    Public predicate for spotting cross-field specs. Its first use is to
    short-circuit :func:`_resolve_field_refs_in_spec` on the hot path: most
    predicates are literal (no cross-field refs), so paying a cheap pre-scan
    to skip the substitution-recursion's allocations is a win - every
    ``Rule.try_match`` runs this on every match_fields spec on every config
    construction. :mod:`llenergymeasure.config.series` reuses it to skip
    cross-field rules it cannot evaluate against a bare candidate value.
    """
    if isinstance(spec, str):
        return spec.startswith(_FIELD_REF_PREFIX)
    if isinstance(spec, dict):
        return any(spec_has_field_ref(v) for v in spec.values())
    if isinstance(spec, (list, tuple)):
        return any(spec_has_field_ref(v) for v in spec)
    return False


def _resolve_field_refs_in_spec(spec: Any, config: Any, predicate_field_path: str) -> Any:
    """Substitute ``@field`` references in a predicate spec with config values.

    Walks the spec dict (or bare value) and replaces any string starting
    with ``@`` with the corresponding field's value resolved against
    ``config``. Bare references (``@num_beams``) resolve as siblings of
    ``predicate_field_path``; dotted references (``@a.b.c``) resolve from
    the config root.

    Short-circuits via :func:`spec_has_field_ref` when the spec contains
    no references - returns the original spec unchanged in that case,
    avoiding the dict/list rebuild overhead on the common literal-spec
    path.

    Returns a new spec with substitutions applied; the input is not
    mutated. Non-ref strings pass through unchanged.
    """
    if not spec_has_field_ref(spec):
        return spec
    if isinstance(spec, str):
        return _resolve_one_ref(spec, config, predicate_field_path)
    if isinstance(spec, dict):
        return {
            op: _resolve_field_refs_in_spec(v, config, predicate_field_path)
            for op, v in spec.items()
        }
    if isinstance(spec, (list, tuple)):
        return type(spec)(
            _resolve_field_refs_in_spec(v, config, predicate_field_path) for v in spec
        )
    return spec


def _resolve_one_ref(ref: str, config: Any, predicate_field_path: str) -> Any:
    target = ref[len(_FIELD_REF_PREFIX) :]
    if "." in target:
        return resolve_field_path(config, target)
    parent_parts = predicate_field_path.split(".")[:-1]
    full_path = ".".join([*parent_parts, target]) if parent_parts else target
    return resolve_field_path(config, full_path)


_OPERATOR_HANDLERS: dict[str, Any] = {
    # Comparison operators: bilaterally None-safe on the *asymmetric* ones.
    # ``a`` may be None when the predicate's field is unset; ``b`` may be
    # None when a ``@field_ref`` resolves against a missing target. Both
    # cases must yield False (rule does not fire) rather than raise.
    # ``==`` and ``equals`` stay as plain equality - `None == x` evaluates
    # to `False` for any non-None `x`, so they naturally don't fire on None.
    # ``equals`` / ``not_equal`` are word-form aliases of ``==`` / ``!=``
    # and MUST match their symbol forms exactly - corpus authors swap them.
    # The four ordering operators route through :func:`_ordered`, which also
    # treats a type-incomparable pair (e.g. a mined numeric bound landing on a
    # dict-valued field) as no-match instead of letting the raw ``<`` raise a
    # TypeError out of config construction.
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a is not None and b is not None and a != b,
    "<": lambda a, b: _ordered(a, b, operator.lt),
    "<=": lambda a, b: _ordered(a, b, operator.le),
    ">": lambda a, b: _ordered(a, b, operator.gt),
    ">=": lambda a, b: _ordered(a, b, operator.ge),
    "equals": lambda a, b: a == b,
    "not_equal": lambda a, b: a is not None and b is not None and a != b,
    # Membership operators: None-safe on the asymmetric one (``not_in``)
    # so unset fields don't trip the rule. ``in`` against a missing field
    # naturally yields False without an explicit guard. Reject non-iterable
    # specs (a string spec would otherwise fall through to substring match -
    # "ab" in "abc" is True, which surprises corpus authors writing
    # {"in": "abc"} thinking "exactly one of these three chars").
    "in": lambda a, b: _require_iterable(b, "in") and a in b,
    "not_in": lambda a, b: a is not None and _require_iterable(b, "not_in") and a not in b,
    # Presence operators: no None-guard needed (they're the test for None).
    "present": lambda a, _: a is not None,
    "absent": lambda a, _: a is None,
    # Type predicates match by the concrete type's __name__. None-safe (a
    # missing field doesn't trip ``type_is_not``). Spec takes a bare string
    # or a list of strings (any-of); predicate holds if the field's concrete
    # type name matches (resp. does not match) any. See :func:`_type_name`
    # for the type-name format and its known ambiguities.
    "type_is": lambda a, b: a is not None and _type_name(a) in _as_name_set(b),
    "type_is_not": lambda a, b: a is not None and _type_name(a) not in _as_name_set(b),
    # Cross-field divisibility checks. Naming follows the existing
    # positive/negative convention (``equals`` / ``not_equal``,
    # ``in`` / ``not_in``). Both operands must be non-bool integers
    # (``True``/``False`` would silently pass via ``bool`` < ``int``).
    # ``not_divisible_by`` fires when ``a % b != 0`` - corpus authors
    # write ``num_beams: {not_divisible_by: '@num_beam_groups'}`` to
    # express "rule fires when num_beams isn't a multiple of
    # num_beam_groups". A zero divisor yields False (no rule fires).
    "divisible_by": lambda a, b: _is_int_pair(a, b) and b != 0 and a % b == 0,
    "not_divisible_by": lambda a, b: _is_int_pair(a, b) and b != 0 and a % b != 0,
}


def _ordered(a: Any, b: Any, op: Callable[[Any, Any], bool]) -> bool:
    """Apply an ordering comparison, treating None and incomparable types as no-match.

    ``a`` may be None when the predicate's field is unset; ``b`` may be None
    when a ``@field_ref`` resolves against a missing target - both yield False.

    ``bool`` operands are treated as non-comparable and never fire, mirroring
    the ``_is_int_pair`` exclusion the divisibility ops apply. Python evaluates
    ``True > 0`` cleanly (``bool`` subclasses ``int``), so an ordering bound
    mined against a scalar numeric field would otherwise silently reject a
    boolean-valued field (e.g. a ``{'>': 0}`` bound firing on
    ``early_stopping=True``). The generated pydantic model remains the authority
    on a boolean field's type validity, so the ordering rule stays inert.

    A mined numeric bound can also land on a field that naturally holds a
    non-numeric value (e.g. transformers ``compile_config``, a dict /
    CompileConfig shape). Comparing such a pair with ``<`` / ``<=`` / ``>`` /
    ``>=`` raises TypeError; that would escape config construction as an
    uncaught crash. Since the bound is a corpus artifact rather than a user
    error, the rule simply does not fire (False) on a type-incomparable pair.
    """
    if a is None or b is None:
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    try:
        return op(a, b)
    except TypeError:
        return False


def _is_int_pair(a: Any, b: Any) -> bool:
    """Return True iff both operands are non-bool integers."""
    return (
        isinstance(a, int)
        and isinstance(b, int)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    )


def _type_name(value: Any) -> str:
    """Return the concrete class name of ``value`` - ``type(value).__name__``.

    **Collision limitation:** this is the bare class name without the module
    qualifier, so unrelated libraries that happen to use the same class name
    (e.g. ``torch.dtype`` and ``numpy.dtype``) can't be distinguished with
    ``type_is: "dtype"`` alone. Disambiguate with a companion predicate
    (``present`` + path specificity) or use a bare-Python type (``bool``,
    ``int``, ``str``, ``list``, ``dict``) where collisions don't arise.
    """
    return type(value).__name__


def _as_name_set(spec: Any) -> frozenset[str]:
    """Accept a single type name or an iterable of names; return a frozenset."""
    if isinstance(spec, str):
        return frozenset({spec})
    return frozenset(str(x) for x in spec)


def _require_iterable(b: Any, op_name: str) -> bool:
    """Reject non-iterable specs for ``in`` / ``not_in`` at evaluation time.

    Naked string specs would silently do substring matching, which is not
    what corpus authors mean when they write ``{"in": "abc"}``. Force a
    list/tuple/set by raising on anything else.
    """
    if isinstance(b, (list, tuple, set, frozenset)):
        return True
    raise TypeError(
        f"Operator {op_name!r} requires list/tuple/set spec; got {type(b).__name__}: {b!r}"
    )


def evaluate_predicate(actual: Any, spec: Any) -> bool:
    """Evaluate ``actual`` against the corpus predicate ``spec``.

    ``spec`` shapes:

    - Bare value -> equality (``spec == actual``).
    - One-key dict -> operator predicate (``{"<": 1}``, ``{"in": ["a", "b"]}``).
    - Multi-key dict -> every operator must hold (all predicates AND-combined).

    The last form covers corpus entries like
    ``{present: true, not_equal: 1.0}`` - field must be set AND not default.
    """
    if isinstance(spec, dict):
        if not spec:
            raise ValueError("Empty match predicate dict")
        for op, value in spec.items():
            handler = _OPERATOR_HANDLERS.get(op)
            if handler is None:
                raise ValueError(f"Unknown match operator: {op!r}")
            if not handler(actual, value):
                return False
        return True
    return bool(actual == spec)


# ---------------------------------------------------------------------------
# Field-path resolver
# ---------------------------------------------------------------------------


def resolve_field_path(config: Any, path: str) -> Any:
    """Walk dotted attribute / key path against ``config``.

    Missing attributes return ``None`` rather than raising - the predicate
    engine treats ``None`` as an absent field. Supports nested Pydantic models,
    dataclasses, and plain dicts mixed in any combination.

    **Method collision guard:** ``getattr(pydantic_model, "items")`` returns
    the bound ``.items()`` method, not a field named ``items``. Pydantic ships
    several attribute names (``copy``, ``dict``, ``json``, ``model_copy``,
    ``model_dump``, ``model_fields``, ``items``, ``keys``, ``values``) that
    would collide with field lookups. We check `__dict__` / `model_fields`
    first and only fall back to `getattr` when the key isn't a known field,
    ensuring that a corpus predicate on a real field wins over an accidental
    method match.
    """
    current: Any = config
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
            continue
        # Pydantic models - use model_fields for the authoritative field set.
        model_fields = getattr(type(current), "model_fields", None)
        if isinstance(model_fields, dict) and part in model_fields:
            current = getattr(current, part, None)
            continue
        # Dataclasses - use __dataclass_fields__ for the authoritative field set.
        dc_fields = getattr(type(current), "__dataclass_fields__", None)
        if isinstance(dc_fields, dict) and part in dc_fields:
            current = getattr(current, part, None)
            continue
        # Fallback: plain objects. Reject callables - they're methods or
        # descriptors, never config field values.
        candidate = getattr(current, part, None)
        current = None if callable(candidate) else candidate
    return current


# ---------------------------------------------------------------------------
# Corpus parsing
# ---------------------------------------------------------------------------


def _major(version: str) -> int:
    try:
        return int(version.split(".", 1)[0])
    except (ValueError, AttributeError) as exc:
        raise UnsupportedSchemaVersionError(
            f"Unparseable schema_version {version!r}; expected semver '1.0.0' form."
        ) from exc


def _parse_provenance(rule_id: str, raw: Any) -> Provenance:
    if not isinstance(raw, dict):
        raise RuleCorpusError(
            f"Rule {rule_id!r} is missing a provenance mapping (got {type(raw).__name__})."
        )
    source = str(raw.get("source", ""))
    if source not in VALID_SOURCE:
        raise UnknownSourceError(
            f"Rule {rule_id!r} has provenance.source={source!r}; "
            f"must be one of: {sorted(VALID_SOURCE)}"
        )
    verified = str(raw.get("verified", ""))
    if verified not in VALID_VERIFIED:
        raise UnknownVerifiedError(
            f"Rule {rule_id!r} has provenance.verified={verified!r}; "
            f"must be one of: {sorted(VALID_VERIFIED)}"
        )
    citation = raw.get("citation")
    return Provenance(
        source=source,
        verified=verified,
        engine_version=str(raw.get("engine_version", "")),
        citation=str(citation) if citation is not None else None,
        date=str(raw.get("date", "")),
    )


def _parse_rule(raw: dict[str, Any]) -> Rule:
    for key in ("id", "engine", "severity", "match", "provenance"):
        if key not in raw:
            raise RuleCorpusError(f"Rule {raw.get('id', '<unknown>')} missing field: {key}")
    rule_id = str(raw["id"])
    match = raw["match"]
    if not isinstance(match, dict) or "fields" not in match:
        raise RuleCorpusError(f"Rule {rule_id!r} has malformed match (missing `fields`): {match!r}")
    severity = str(raw["severity"])
    if severity not in VALID_SEVERITY:
        raise UnknownSeverityError(
            f"Rule {rule_id!r} has severity={severity!r}; must be one of: {sorted(VALID_SEVERITY)}"
        )
    normalised = raw.get("normalised_fields") or ()
    if isinstance(normalised, str):
        normalised = (normalised,)
    if normalised and severity != "dormant":
        raise RuleCorpusError(
            f"Rule {rule_id!r} has severity={severity!r} but declares normalised_fields; "
            "normalised_fields drives dedup canonicalisation and is only meaningful on "
            "'dormant' rules (it is dead data on 'error' rules)."
        )
    return Rule(
        id=rule_id,
        engine=str(raw["engine"]),
        severity=severity,
        match_fields=dict(match["fields"]),
        provenance=_parse_provenance(rule_id, raw["provenance"]),
        message_template=raw.get("message_template"),
        normalised_fields=tuple(str(p) for p in normalised),
    )


def _parse_envelope(engine: str, raw_text: str) -> EngineRules:
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise RuleCorpusError(
            f"Engine rules for {engine!r} must be a YAML mapping; got {type(data).__name__}"
        )
    schema_version = str(data.get("schema_version", ""))
    if not schema_version:
        raise UnsupportedSchemaVersionError(f"Engine rules for {engine!r} missing schema_version.")
    if _major(schema_version) != SUPPORTED_MAJOR_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Engine rules for {engine!r} has schema_version={schema_version!r}; "
            f"this loader only supports major {SUPPORTED_MAJOR_VERSION}. "
            f"Regenerate the corpus or upgrade the loader."
        )
    raw_rules = data.get("rules") or []
    rules = tuple(_parse_rule(r) for r in raw_rules)
    return EngineRules(
        engine=engine,
        schema_version=schema_version,
        engine_version=str(data.get("engine_version", "")),
        rules=rules,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "engines"

RULES_FILENAME = "rules.yaml"


class EngineRulesLoader:
    """Load, cache, and serve :class:`EngineRules` per engine.

    Binds to exactly one file per engine:
    ``src/llenergymeasure/engines/{engine}/rules.yaml``. There is no
    proposed/validated split and no overlay merge - the shipped file is the
    verification ladder's committed output and the sole runtime source.

    Per-instance cache (rather than module-level LRU) - tests can instantiate
    a loader and point ``corpus_root`` at a fixture without polluting other
    tests.
    """

    def __init__(self, corpus_root: Path | None = None) -> None:
        self.corpus_root: Path = corpus_root or _DEFAULT_CORPUS_ROOT
        self._cache: dict[str, EngineRules] = {}

    def load_rules(self, engine: str) -> EngineRules:
        """Return the parsed rules corpus for ``engine``, parsing once per engine."""
        cached = self._cache.get(engine)
        if cached is not None:
            return cached

        yaml_path = self.corpus_root / engine / RULES_FILENAME
        try:
            yaml_text = yaml_path.read_text()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Engine rules for {engine!r} not found at {yaml_path}."
            ) from exc

        parsed = _parse_envelope(engine, yaml_text)
        self._cache[engine] = parsed
        return parsed

    def invalidate(self, engine: str | None = None) -> None:
        """Clear cached rules (all or for one engine)."""
        if engine is None:
            self._cache.clear()
        else:
            self._cache.pop(engine, None)
