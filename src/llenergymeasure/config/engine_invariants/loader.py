"""Load, match, and render validation rules from the YAML corpus.

The corpus at ``src/llenergymeasure/engines/{engine}/invariants.proposed.yaml``
is parsed here into typed :class:`Invariant` entries. Each invariant carries a match
predicate (operators defined in :func:`evaluate_predicate`) and a message
template. The generic ``@model_validator`` in ``config/models.py`` calls
:meth:`Invariant.try_match` on every rule for a given engine and emits
error/warn/dormant annotations based on the rule's severity.

Design mirror: this module parallels :mod:`llenergymeasure.config.schema_loader`
from parameter-discovery — same envelope validation
(:class:`UnsupportedSchemaVersionError` on major-version mismatch), same
per-instance caching for test isolation, same lazy load pattern.

Lifecycle pair (per the engine-coupling architecture, 2026-04-28):
  The proposed YAML carries each rule's declared ``expected_outcome``. The
  ``engine-invariants`` (validation gate) CI pipeline (see
  ``scripts/validate_invariants.py``) runs every rule through the real library
  and emits ``src/llenergymeasure/engines/{engine}/invariants.validated.yaml``
  — this YAML captures observed outcomes. When present, the loader overlays
  the validated observations onto the corpus so downstream consumers see
  CI-validated truth; absent, the loader falls back to the proposed YAML so
  local development without a validation run still works.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, get_args

import yaml

SUPPORTED_MAJOR_VERSION = 1
"""Major version the loader knows how to parse.

Raised via :class:`UnsupportedSchemaVersionError` on mismatch; the loader
refuses partial reads to avoid silently accepting a future schema shape.
"""


Severity = Literal["dormant", "warn", "error"]
"""Severity tier a rule match produces at validation time.

- ``dormant`` — the config still runs, but a field is silently ignored or
  coerced by the engine. Observable-but-user-invisible; the rule surfaces
  this.
- ``warn`` — the engine announces a suboptimal setting at construct or
  runtime but still proceeds.
- ``error`` — the engine raises and the config cannot run.
"""

Outcome = Literal[
    "dormant_silent",
    "dormant_announced",
    "warn",
    "error",
    "pass",
]
"""What the engine does when the rule's predicate holds.

- ``dormant_silent`` — the engine silently normalises or ignores
  (observable only via ``extract_observed_params`` post-construction).
- ``dormant_announced`` — the engine writes to a ``minor_issues`` dict /
  logger, but the config still runs.
- ``warn`` — the engine calls ``warnings.warn(...)`` or equivalent.
- ``error`` — the engine raises at construct / validate time.
- ``pass`` — the predicate matched but the engine handles it cleanly;
  used for positive-reference rules.
"""

EmissionChannel = Literal[
    "warnings_warn",
    "logger_warning",
    "logger_warning_once",
    "minor_issues_dict",
    "none",
    "runtime_exception",
]
"""How the engine user-visibly signals the issue.

- ``warnings_warn`` — Python ``warnings.warn(...)``.
- ``logger_warning`` / ``logger_warning_once`` — stdlib logger.
- ``minor_issues_dict`` — an internal dict (e.g. HF's ``minor_issues``)
  whose presence is user-observable via strict-mode raise OR log.
- ``none`` — no user-visible emission (silent coercion or raise).
- ``runtime_exception`` — exception raised at engine construct / runtime.

Canonical rule: ``minor_issues_dict`` alone is an internal signal; if HF
composes the dict then emits via ``logger.warning_once``, the
user-visible channel is ``logger_warning_once``. Corpus authors should
record what users see, not the internal staging buffer.
"""

AddedBy = Literal[
    "static_miner",
    "dynamic_miner",
    "pydantic_lift",
    "msgspec_lift",
    "dataclass_lift",
    "manual_seed",
    "runtime_warning",
    "observed_collision",
]
"""Provenance of a rule in the corpus.

Eight discovery paths with distinct trust/verifiability profiles:

- ``static_miner`` — rule extracted by parsing Python source AST
  (used by vLLM / TRT-LLM miners; CI can re-derive on library bump).
- ``dynamic_miner`` — rule extracted via library-API introspection
  (transformers' ``GenerationConfig.validate(strict=True)`` returning
  structured ``minor_issues`` dict; CI can re-derive on library bump).
- ``pydantic_lift`` — rule extracted from a ``pydantic.BaseModel`` (or
  ``pydantic.dataclasses.dataclass``) via ``model_json_schema()`` plus
  ``FieldInfo.metadata`` (annotated-types constraints + Literal allowlists).
- ``msgspec_lift`` — rule extracted from a ``msgspec.Struct`` via
  ``msgspec.inspect.type_info`` (``Meta(ge=, le=, ...)`` constraints).
- ``dataclass_lift`` — rule extracted from a ``@dataclasses.dataclass``
  via ``dataclasses.fields()`` plus ``Literal[...]`` annotation parsing.
- ``manual_seed`` — hand-written by a maintainer for cases the miners
  can't reach (e.g. BNB type rules; not auto-regenerable).
- ``runtime_warning`` — proposed by the feedback loop from captured
  ``logger.warning_once`` emissions (needs human generalisation before
  landing).
- ``observed_collision`` — proposed by the feedback loop from observed-config-hash
  collision detection (needs human generalisation before landing).
"""

VALID_SEVERITY: frozenset[str] = frozenset(get_args(Severity))
VALID_OUTCOME: frozenset[str] = frozenset(get_args(Outcome))
VALID_EMISSION_CHANNEL: frozenset[str] = frozenset(get_args(EmissionChannel))
VALID_ADDED_BY: frozenset[str] = frozenset(get_args(AddedBy))


class UnsupportedSchemaVersionError(ValueError):
    """Validated invariants corpus has a schema_version major the loader can't parse."""


class UnknownEnumValueError(ValueError):
    """Invariant entry has a closed-enum field value outside the permitted set.

    Covers ``added_by``, ``severity``, ``expected_outcome.outcome``, and
    ``expected_outcome.emission_channel``. Subclassed per field for callers
    that want to distinguish.
    """


class UnknownAddedByError(UnknownEnumValueError):
    """Invariant entry has an ``added_by`` value outside :data:`AddedBy`."""


class UnknownSeverityError(UnknownEnumValueError):
    """Invariant entry has a ``severity`` value outside :data:`Severity`."""


class UnknownOutcomeError(UnknownEnumValueError):
    """Invariant entry has an ``expected_outcome.outcome`` value outside :data:`Outcome`."""


class UnknownEmissionChannelError(UnknownEnumValueError):
    """Invariant entry has an ``expected_outcome.emission_channel`` value outside :data:`EmissionChannel`."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantMatch:
    """Result of a rule matching a concrete config.

    ``declared_value`` is the user-set value for the *trigger* field (the first
    non-trivially-predicated field in the match spec). ``effective_value``
    populates only when the rule's ``expected_outcome`` lists the rule as
    ``dormant_silent`` with a ``normalised_fields`` mapping.
    """

    rule: Invariant
    declared_value: Any
    effective_value: Any | None = None
    matched_fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Invariant:
    """One validation rule parsed from the corpus.

    Field names mirror the YAML schema documented in
    ``src/llenergymeasure/engines/INVARIANTS_README.md``. Construction goes through
    :func:`_parse_invariant`; tests can instantiate directly for unit coverage.
    """

    id: str
    engine: str
    library: str
    rule_under_test: str
    severity: str
    native_type: str
    match_engine: str
    match_fields: dict[str, Any]
    kwargs_positive: dict[str, Any]
    kwargs_negative: dict[str, Any]
    expected_outcome: dict[str, Any]
    message_template: str | None
    miner_source: dict[str, Any]
    references: tuple[str, ...]
    added_by: str
    added_at: str
    cross_validated_by: tuple[str, ...] = ()
    """Other miner sources that produced the same fingerprint as this rule.

    Empty for single-source rules. Populated by the corpus merger
    (``scripts/engine_miners/build_corpus.py``) when two or more miners
    independently emitted a rule with the same ``(engine, severity,
    match.fields)`` fingerprint. ``added_by`` remains the *primary*
    source (used for downstream filtering); ``cross_validated_by`` is
    additional provenance for the reviewer.

    Schema choice rationale: keeping ``added_by`` as a single string
    preserves the existing ``AddedBy`` Literal and the corpus-invariants
    test that pins it; ``cross_validated_by`` is a strictly additive field
    with a sane default for older rules.
    """

    def try_match(self, config: Any) -> InvariantMatch | None:
        """Return a :class:`InvariantMatch` if every predicate in ``match_fields`` holds.

        Field paths are dotted (``"transformers.sampling.temperature"``) and
        resolve against ``config`` attribute-by-attribute, tolerating Pydantic
        models, dataclasses, and plain dicts.

        Predicate specs may carry ``@field_path`` references on the
        right-hand side of any operator. References are resolved against
        the same ``config`` before predicate evaluation. Bare references
        (``@num_beams``) resolve as siblings of the predicate's field;
        dotted references (``@transformers.sampling.num_beams``) resolve
        from the config root.

        ``declared_value`` on the returned match is the last field's value —
        corpus convention puts the precondition fields first and the
        *subject* field last (the field the rule is actually about). Users
        see the subject value substituted into message templates.
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
        return InvariantMatch(rule=self, declared_value=last_value, matched_fields=matched)

    def render_message(self, match: InvariantMatch) -> str:
        """Substitute ``{declared_value}`` / ``{effective_value}`` / ``{invariant_id}`` in the template.

        Uses ``str.format`` with permissive defaults — templates that reference
        missing keys fall back to the rule id + raw template rather than
        raising at user-facing time.
        """
        if self.message_template is None:
            return f"[{self.id}] <no message template>"
        try:
            return self.message_template.format(
                declared_value=match.declared_value,
                effective_value=match.effective_value,
                invariant_id=self.id,
                **match.matched_fields,
            )
        except (KeyError, IndexError):
            return f"[{self.id}] {self.message_template}"


@dataclass(frozen=True)
class EngineInvariants:
    """Parsed corpus for one engine."""

    engine: str
    schema_version: str
    engine_version: str
    invariants: tuple[Invariant, ...]


# ---------------------------------------------------------------------------
# Predicate engine
# ---------------------------------------------------------------------------


_FIELD_REF_PREFIX = "@"


def _spec_has_field_ref(spec: Any) -> bool:
    """Return True if ``spec`` contains any ``@field_path`` string anywhere.

    Used to short-circuit :func:`_resolve_field_refs_in_spec` on the hot
    path: most predicates are literal (no cross-field refs), so paying a
    cheap pre-scan to skip the substitution-recursion's allocations is
    a win — every ``Invariant.try_match`` runs this on every match_fields
    spec on every config construction.
    """
    if isinstance(spec, str):
        return spec.startswith(_FIELD_REF_PREFIX)
    if isinstance(spec, dict):
        return any(_spec_has_field_ref(v) for v in spec.values())
    if isinstance(spec, (list, tuple)):
        return any(_spec_has_field_ref(v) for v in spec)
    return False


def _resolve_field_refs_in_spec(spec: Any, config: Any, predicate_field_path: str) -> Any:
    """Substitute ``@field`` references in a predicate spec with config values.

    Walks the spec dict (or bare value) and replaces any string starting
    with ``@`` with the corresponding field's value resolved against
    ``config``. Bare references (``@num_beams``) resolve as siblings of
    ``predicate_field_path``; dotted references (``@a.b.c``) resolve from
    the config root.

    Short-circuits via :func:`_spec_has_field_ref` when the spec contains
    no references — returns the original spec unchanged in that case,
    avoiding the dict/list rebuild overhead on the common literal-spec
    path.

    Returns a new spec with substitutions applied; the input is not
    mutated. Non-ref strings pass through unchanged.
    """
    if not _spec_has_field_ref(spec):
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
    # ``==`` and ``equals`` stay as plain equality — `None == x` evaluates
    # to `False` for any non-None `x`, so they naturally don't fire on None.
    # ``equals`` / ``not_equal`` are word-form aliases of ``==`` / ``!=``
    # and MUST match their symbol forms exactly — corpus authors swap them.
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a is not None and b is not None and a != b,
    "<": lambda a, b: a is not None and b is not None and a < b,
    "<=": lambda a, b: a is not None and b is not None and a <= b,
    ">": lambda a, b: a is not None and b is not None and a > b,
    ">=": lambda a, b: a is not None and b is not None and a >= b,
    "equals": lambda a, b: a == b,
    "not_equal": lambda a, b: a is not None and b is not None and a != b,
    # Membership operators: None-safe on the asymmetric one (``not_in``)
    # so unset fields don't trip the rule. ``in`` against a missing field
    # naturally yields False without an explicit guard. Reject non-iterable
    # specs (a string spec would otherwise fall through to substring match —
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
    # ``not_divisible_by`` fires when ``a % b != 0`` — corpus authors
    # write ``num_beams: {not_divisible_by: '@num_beam_groups'}`` to
    # express "rule fires when num_beams isn't a multiple of
    # num_beam_groups". A zero divisor yields False (no rule fires).
    "divisible_by": lambda a, b: _is_int_pair(a, b) and b != 0 and a % b == 0,
    "not_divisible_by": lambda a, b: _is_int_pair(a, b) and b != 0 and a % b != 0,
}


def _is_int_pair(a: Any, b: Any) -> bool:
    """Return True iff both operands are non-bool integers."""
    return (
        isinstance(a, int)
        and isinstance(b, int)
        and not isinstance(a, bool)
        and not isinstance(b, bool)
    )


def _type_name(value: Any) -> str:
    """Return the concrete class name of ``value`` — ``type(value).__name__``.

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

    - Bare value → equality (``spec == actual``).
    - One-key dict → operator predicate (``{"<": 1}``, ``{"in": ["a", "b"]}``).
    - Multi-key dict → every operator must hold (all predicates AND-combined).

    The last form covers corpus entries like
    ``{present: true, not_equal: 1.0}`` — field must be set AND not default.
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

    Missing attributes return ``None`` rather than raising — the predicate
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
        # Pydantic models — use model_fields for the authoritative field set.
        model_fields = getattr(type(current), "model_fields", None)
        if isinstance(model_fields, dict) and part in model_fields:
            current = getattr(current, part, None)
            continue
        # Dataclasses — use __dataclass_fields__ for the authoritative field set.
        dc_fields = getattr(type(current), "__dataclass_fields__", None)
        if isinstance(dc_fields, dict) and part in dc_fields:
            current = getattr(current, part, None)
            continue
        # Fallback: plain objects. Reject callables — they're methods or
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


def _parse_invariant(raw: dict[str, Any]) -> Invariant:
    required = (
        "id",
        "engine",
        "severity",
        "native_type",
        "match",
        "kwargs_positive",
        "kwargs_negative",
        "expected_outcome",
    )
    for key in required:
        if key not in raw:
            raise ValueError(f"Invariant {raw.get('id', '<unknown>')} missing field: {key}")
    match = raw["match"]
    if not isinstance(match, dict) or "fields" not in match:
        raise ValueError(f"Invariant {raw['id']} has malformed match (missing `fields`): {match!r}")
    invariant_id = raw["id"]
    severity = str(raw["severity"])
    if severity not in VALID_SEVERITY:
        raise UnknownSeverityError(
            f"Invariant {invariant_id!r} has severity={severity!r}; must be one of: {sorted(VALID_SEVERITY)}"
        )
    expected_outcome = dict(raw["expected_outcome"])
    outcome = str(expected_outcome.get("outcome", ""))
    if outcome not in VALID_OUTCOME:
        raise UnknownOutcomeError(
            f"Invariant {invariant_id!r} has expected_outcome.outcome={outcome!r}; "
            f"must be one of: {sorted(VALID_OUTCOME)}"
        )
    emission_channel = str(expected_outcome.get("emission_channel", ""))
    if emission_channel not in VALID_EMISSION_CHANNEL:
        raise UnknownEmissionChannelError(
            f"Invariant {invariant_id!r} has expected_outcome.emission_channel={emission_channel!r}; "
            f"must be one of: {sorted(VALID_EMISSION_CHANNEL)}"
        )
    added_by = str(raw.get("added_by", "manual_seed"))
    if added_by not in VALID_ADDED_BY:
        raise UnknownAddedByError(
            f"Invariant {invariant_id!r} has added_by={added_by!r}; must be one of: {sorted(VALID_ADDED_BY)}"
        )
    raw_cross = raw.get("cross_validated_by") or ()
    if isinstance(raw_cross, str):
        # A single string is a corpus-authoring slip; coerce to a one-tuple.
        raw_cross = (raw_cross,)
    cross_validated_by: tuple[str, ...] = tuple(str(s) for s in raw_cross)
    for source in cross_validated_by:
        if source not in VALID_ADDED_BY:
            raise UnknownAddedByError(
                f"Invariant {invariant_id!r} has cross_validated_by entry={source!r}; "
                f"must be one of: {sorted(VALID_ADDED_BY)}"
            )
    return Invariant(
        id=str(invariant_id),
        engine=str(raw["engine"]),
        library=str(raw.get("library", raw["engine"])),
        rule_under_test=str(raw.get("rule_under_test", "")),
        severity=severity,
        native_type=str(raw["native_type"]),
        match_engine=str(match.get("engine", raw["engine"])),
        match_fields=dict(match["fields"]),
        kwargs_positive=dict(raw["kwargs_positive"]),
        kwargs_negative=dict(raw["kwargs_negative"]),
        expected_outcome=expected_outcome,
        message_template=raw.get("message_template"),
        miner_source=dict(raw.get("miner_source") or raw.get("walker_source") or {}),
        references=tuple(raw.get("references") or ()),
        added_by=added_by,
        added_at=str(raw.get("added_at", "")),
        cross_validated_by=cross_validated_by,
    )


def _parse_envelope(engine: str, raw_text: str) -> EngineInvariants:
    data = yaml.safe_load(raw_text)
    if not isinstance(data, dict):
        raise ValueError(
            f"Engine invariants for {engine!r} must be a YAML mapping; got {type(data).__name__}"
        )
    schema_version = str(data.get("schema_version", ""))
    if not schema_version:
        raise UnsupportedSchemaVersionError(
            f"Engine invariants for {engine!r} missing schema_version."
        )
    if _major(schema_version) != SUPPORTED_MAJOR_VERSION:
        raise UnsupportedSchemaVersionError(
            f"Engine invariants for {engine!r} has schema_version={schema_version!r}; "
            f"this loader only supports major {SUPPORTED_MAJOR_VERSION}. "
            f"Regenerate the corpus or upgrade the loader."
        )
    raw_invariants = data.get("invariants") or []
    invariants = tuple(_parse_invariant(r) for r in raw_invariants)
    return EngineInvariants(
        engine=engine,
        schema_version=schema_version,
        engine_version=str(data.get("engine_version", "")),
        invariants=invariants,
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_DEFAULT_CORPUS_ROOT = Path(__file__).resolve().parents[2] / "engines"


class EngineInvariantsLoader:
    """Load, cache, and serve :class:`EngineInvariants` per engine.

    Per-instance cache (rather than module-level LRU) — tests can instantiate
    a loader and monkeypatch ``corpus_root`` without polluting other tests.

    Load order (picked up automatically; per-engine sub-package layout):
      1. **Proposed YAML** under ``src/llenergymeasure/engines/{engine}/invariants.proposed.yaml`` —
         the maintainer-seeded source of truth; always present in-repo.
      2. **Validated YAML** under ``src/llenergymeasure/engines/{engine}/invariants.validated.yaml`` —
         CI-validated observed behaviour, overlaid onto the corpus's invariants
         when present. Written by ``scripts/validate_invariants.py`` under the
         engine-invariants CI.
    """

    def __init__(self, corpus_root: Path | None = None) -> None:
        self.corpus_root: Path = corpus_root or _DEFAULT_CORPUS_ROOT
        self._cache: dict[str, EngineInvariants] = {}

    def load_invariants(self, engine: str) -> EngineInvariants:
        """Return the parsed corpus for ``engine``, parsing once per engine.

        When a CI-validated validated YAML envelope exists at the configured
        ``corpus_root``, the loader overlays its observed outcomes onto the
        corpus's rules — downstream consumers see empirically-confirmed
        behaviour rather than the corpus's declared shape alone.
        """
        cached = self._cache.get(engine)
        if cached is not None:
            return cached

        yaml_path = self.corpus_root / engine / "invariants.proposed.yaml"
        try:
            yaml_text = yaml_path.read_text()
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Engine invariants for {engine!r} not found at {yaml_path}. "
                f"Run `python -m scripts.engine_miners.{engine}_miner "
                f"--out {yaml_path}` to generate."
            ) from exc

        parsed = _parse_envelope(engine, yaml_text)

        validated = _try_load_validated_yaml(self.corpus_root, engine)
        if validated is not None:
            parsed = _overlay_validated_observations(parsed, validated)

        self._cache[engine] = parsed
        return parsed

    def invalidate(self, engine: str | None = None) -> None:
        """Clear cached rules (all or for one engine)."""
        if engine is None:
            self._cache.clear()
        else:
            self._cache.pop(engine, None)


# ---------------------------------------------------------------------------
# Validated YAML overlay (engine-invariants CI)
# ---------------------------------------------------------------------------


def _try_load_validated_yaml(corpus_root: Path, engine: str) -> dict[str, Any] | None:
    """Return parsed validated-rules YAML for ``engine`` or ``None`` if absent.

    Reads from ``{corpus_root}/{engine}/invariants.validated.yaml``. Swallows YAML parse
    errors and rejects unsupported envelope versions to avoid breaking
    startup on a corrupt or future-schema commit-back — the validation CI job
    will resurface the issue.
    """
    path = corpus_root / engine / "invariants.validated.yaml"
    try:
        raw = path.read_text()
    except FileNotFoundError:
        return None
    try:
        parsed = yaml.safe_load(raw)
    except yaml.YAMLError:
        return None
    if not isinstance(parsed, dict):
        return None
    envelope_version = str(parsed.get("schema_version", ""))
    if envelope_version:
        try:
            major = _major(envelope_version)
        except UnsupportedSchemaVersionError:
            return None
        if major != SUPPORTED_MAJOR_VERSION:
            return None
    return parsed


_OBSERVED_KEY_MAP = {
    "outcome": "observed_outcome",
    "emission_channel": "observed_emission_channel",
    "observed_messages": "observed_messages",
}


def _overlay_validated_observations(
    parsed: EngineInvariants, validated: dict[str, Any]
) -> EngineInvariants:
    """Overlay observed outcomes from a validated YAML envelope onto the corpus rules.

    The corpus carries the declared shape; the validated YAML carries what
    CI observed. When the validated YAML is present, the loader writes
    observed-* keys alongside the corpus's declared ``outcome`` /
    ``emission_channel`` so consumers (the generic ``@model_validator``)
    can act on CI-validated truth. The declared fields are left untouched
    — strict validation in :func:`_parse_invariant` is not re-exercised against
    the observed vocabulary (which is deliberately wider; see
    ``scripts/_invariant_validation_common.py``).
    """
    cases = {c["id"]: c for c in validated.get("cases", []) if isinstance(c, dict) and "id" in c}
    overlaid = tuple(_overlay_invariant(rule, cases.get(rule.id)) for rule in parsed.invariants)
    return replace(parsed, invariants=overlaid)


def _overlay_invariant(rule: Invariant, observed: dict[str, Any] | None) -> Invariant:
    """Merge observed-* fields from a validated case into an invariant's expected_outcome.

    Observed keys are written directly (not via ``setdefault``) so a
    re-applied overlay with updated CI observations replaces a prior
    overlay's values. The corpus declares no ``observed_*`` keys itself,
    so this never clobbers corpus-authored data.
    """
    if observed is None:
        return rule
    merged = dict(rule.expected_outcome)
    for validated_key, expected_key in _OBSERVED_KEY_MAP.items():
        value = observed.get(validated_key)
        if value not in (None, [], {}):
            merged[expected_key] = list(value) if isinstance(value, list) else value
    return replace(rule, expected_outcome=merged)
