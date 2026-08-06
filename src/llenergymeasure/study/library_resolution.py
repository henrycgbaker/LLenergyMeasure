"""Sweep library-resolution mechanism - apply validated dormant rules to fixpoint, dedup by resolved_config_hash.

The library-resolution mechanism is the host-side, pre-dispatch layer that normalises every
field the engine-rules corpus marks as ``dormant``. Each rule's fired-state
projection drives its subject field (the one marked ``!=`` / ``present``, or
named in ``normalised_fields``) back to *absent* via :data:`_STRIP`, so a
config that set the dormant field collapses with one that never did.
:func:`_apply_rules_fixpoint` iterates that projection to a stable fixpoint;
its idempotence and shuffle-stability are covered by the unit tests in
``tests/unit/study/test_library_resolution.py``.

Rules chain (vLLM epsilon-clamp → greedy-normalise); iteration is capped at
:data:`_MAX_ITER` to surface cycles via :class:`LibraryResolutionCycleError`.

vLLM/TRT-LLM corpora don't exist yet, so today this exercises mostly the
transformers rules; the library-resolution mechanism itself is engine-generic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from llenergymeasure.config.engine_rules.loader import (
    EngineRulesLoader,
    Rule,
    canonical_operator,
    resolve_field_path,
)
from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.config.ssot import engine_str
from llenergymeasure.study.hashing import build_resolved_view, hash_config

logger = logging.getLogger(__name__)

_MAX_ITER = 10
"""Maximum fixpoint passes before declaring non-convergence.

Every seeded-corpus case converged within 2 passes; 10 is generous headroom
that still surfaces a rule cycle quickly.
"""

_STRIP = object()
"""Marker: drive a dormant field back to *absent* (its library default).

Distinct from ``None``. The resolved-config hash deliberately keeps ``None``
and missing keys distinguishable (see :mod:`llenergymeasure.domain.hashing`),
so a field the engine silently ignores (e.g. vLLM ``seed=-1``) must be driven
to absent - not ``None`` - for it to collapse with a config that never set it.
The vLLM dormant fields are all pydantic extras, so "absent" means removing the
extra key.
"""


class LibraryResolutionCycleError(RuntimeError):
    """The library-resolution mechanism did not reach a fixpoint within :data:`_MAX_ITER` passes.

    Indicates a cycle in the engine-rules corpus (rule A produces state
    matching rule B which produces state matching rule A). The corpus
    validation step's shuffle-application test is supposed to catch this at CI time,
    but this guard prevents runtime hangs if a bad corpus ships anyway.
    """

    def __init__(self, final_config: ExperimentConfig, iterations: int) -> None:
        super().__init__(
            f"Library resolution did not reach fixpoint within {iterations} iterations. "
            f"Likely a cycle in the engine-rules corpus. "
            f"Final engine={final_config.engine}."
        )
        self.final_config = final_config
        self.iterations = iterations


# ---------------------------------------------------------------------------
# Core _apply_rules_fixpoint() - one config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DormantObservation:
    """One dormant normalisation the library-resolution mechanism applied.

    Records that a dormant rule fired on a config and drove a field to its
    canonical form, so ``llem study plan`` / preflight can surface the
    (otherwise silent) mutation of the executed config.
    """

    engine: str
    rule_id: str
    field_path: str
    normalisation: str
    """Human-readable effect: ``"stripped"`` for a strip-to-absent, else
    ``"-> <value>"`` for a concrete rewrite."""


def _apply_rules_fixpoint(
    config: ExperimentConfig,
    rules: list[Rule] | tuple[Rule, ...],
    observations: list[DormantObservation] | None = None,
) -> ExperimentConfig:
    """Apply every ``dormant``-severity rule to ``config`` repeatedly until stable.

    Returns a deep-copy of ``config`` with each dormant rule's normalisations
    projected onto the fired fields. The input is not mutated.

    Args:
        config: A validated ``ExperimentConfig``.
        rules: The rule list for the config's engine (typically from
            ``EngineRulesLoader.load_rules(engine).rules``).
        observations: Optional sink; each applied normalisation is appended as a
            :class:`DormantObservation` for downstream (plan / preflight)
            display. ``None`` skips the bookkeeping entirely.

    Raises:
        LibraryResolutionCycleError: If the fixpoint loop exceeds
            :data:`_MAX_ITER` passes - the validated corpus has a rule cycle.
    """
    dormant_rules = [r for r in rules if r.severity == "dormant"]
    if not dormant_rules:
        return config.model_copy(deep=True)

    # A rule's normalisations depend only on the frozen Rule, not on the config
    # under resolution, so precompute once per rule rather than recomputing on
    # every fixpoint iteration for every config. Rule carries a dict field
    # (match_fields), so it is not hashable - key the precompute by object
    # identity rather than by the Rule itself.
    normalisations_by_rule = {id(r): _rule_normalisations(r) for r in dormant_rules}

    current = config.model_copy(deep=True)
    for _iteration in range(_MAX_ITER):
        fired = False
        for rule in dormant_rules:
            match = rule.try_match(current)
            if match is None:
                continue
            updates = normalisations_by_rule[id(rule)]
            if not updates:
                continue
            for field_path, target_value in updates.items():
                current_value = resolve_field_path(current, field_path)
                if target_value is _STRIP:
                    # Canonical form is *absent*; only fire while a value lingers.
                    already_canonical = current_value is None
                else:
                    already_canonical = current_value == target_value
                if not already_canonical:
                    _assign_field_path(current, field_path, target_value)
                    fired = True
                    if observations is not None:
                        norm = "stripped" if target_value is _STRIP else f"-> {target_value!r}"
                        observations.append(
                            DormantObservation(
                                engine=rule.engine,
                                rule_id=rule.id,
                                field_path=field_path,
                                normalisation=norm,
                            )
                        )
        if not fired:
            return current
    raise LibraryResolutionCycleError(current, _MAX_ITER)


def _resolve_normalised_field(name: str, match_fields: dict[str, Any]) -> str:
    """Resolve a ``normalised_fields`` entry to a dotted config path.

    A dotted entry passes through unchanged. A bare leaf name (the vLLM corpus
    convention - it stores ``seed``, ``all2all_backend``, ... not the fully
    dotted paths) is anchored as a *sibling* of the rule's subject match field.
    Corpus convention orders ``match_fields`` preconditions-first, subject-last,
    so the anchor is the parent path of the LAST match-field key. This mirrors
    the bare-``@field_ref`` sibling semantics in the loader's ``_resolve_one_ref``.

    A bare name with no match fields to anchor against passes through unchanged
    rather than guessing - assignment against config root then no-ops silently,
    preserving today's behaviour for that (corpus-absent) shape.
    """
    if "." in name or not match_fields:
        return name
    subject_path = next(reversed(match_fields))
    parent_parts = subject_path.split(".")[:-1]
    if not parent_parts:
        return name
    return ".".join([*parent_parts, name])


def _rule_normalisations(rule: Rule) -> dict[str, Any]:
    """Return ``{field_path: canonical_value}`` the rule normalises to.

    Strategy (mirrors the fixpoint test's projection):

    1. If ``normalised_fields`` lists explicit paths, each resolves to a dotted
       config path (see :func:`_resolve_normalised_field`) and collapses to
       :data:`_STRIP` - the field is driven back to *absent* (its library
       default), so a config that set it equals one that never did.
    2. Otherwise, fall back to the rule's *match* predicate: any field marked
       ``!=`` / ``not_equal`` / ``present`` is a *subject* field and is driven
       to :data:`_STRIP` (absent). Stripping is the only projection that
       collapses correctly for both declared fields (reset to their ``None``
       default) and pydantic extras (remove the key) - emitting ``None`` or a
       concrete sentinel leaves an extra key present, so the resolved-config
       hash distinguishes it from a config that never set the field and dedup
       silently no-ops.

    Operator keys are canonicalised (``!=`` and ``not_equal`` are the same
    operator) so a rule written with a word alias projects identically to one
    written with the symbol.

    Rules that match only on equality (e.g. ``do_sample: false``) do not
    normalise those fields - equality predicates are *triggers*, not
    *subjects*. Subject fields are the ones marked ``present`` / ``!=``.
    """
    out: dict[str, Any] = {}

    explicit = rule.normalised_fields or ()
    for raw_name in explicit:
        path = _resolve_normalised_field(str(raw_name), rule.match_fields)
        out[path] = _STRIP

    if out:
        return out

    for path, spec in rule.match_fields.items():
        if not isinstance(spec, dict):
            continue
        canon = {canonical_operator(op): value for op, value in spec.items()}
        is_not_equal = "!=" in canon
        is_present = bool(canon.get("present")) and "in" not in canon
        if is_not_equal or is_present:
            # Subject field - drive it back to *absent* so a config that set it
            # collapses with one that never did.
            out[path] = _STRIP
    return out


def _assign_field_path(config: ExperimentConfig, path: str, value: Any) -> None:
    """Set ``value`` at dotted ``path`` on ``config`` in place.

    Walks nested Pydantic models / dicts, tolerant of ``None`` intermediate
    attributes (silently returns if the path doesn't resolve to an assignable
    location - mirrors :func:`resolve_field_path`'s permissive traversal).
    """
    parts = path.split(".")
    parent: Any = config
    for part in parts[:-1]:
        if parent is None:
            return
        parent = parent.get(part) if isinstance(parent, dict) else getattr(parent, part, None)
    if parent is None:
        return
    leaf = parts[-1]
    if value is _STRIP:
        # Drive the field back to *absent* so it collapses with a config that
        # never set it. vLLM's dormant fields are pydantic extras; removing the
        # extra makes the resolved-config view identical to the unset case.
        if isinstance(parent, dict):
            parent.pop(leaf, None)
            return
        extra = getattr(parent, "__pydantic_extra__", None)
        if isinstance(extra, dict) and leaf in extra:
            del extra[leaf]
            return
        # Declared field: it cannot be removed, so reset to None as a
        # best-effort strip - enough to reach a fixpoint (it will not collapse
        # with a truly-absent field, but no shipped rule hits this branch).
        value = None
    try:
        if isinstance(parent, dict):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)
    except (ValueError, TypeError) as exc:
        # Pydantic model_config={"frozen": True} or field constraints can
        # reject the assignment. Log at debug - the library-resolution mechanism is best-
        # effort; a rejected field just stays at its declared value.
        logger.debug("Library resolution could not assign %s=%r: %s", path, value, exc)


# ---------------------------------------------------------------------------
# Sweep dedup
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquivalenceGroup:
    """Pre-run group of declared configs that collapse to the same resolved canonical form."""

    resolved_config_hash: str
    canonical_excerpt: dict[str, Any]
    member_indices: tuple[int, ...]
    representative_index: int

    @property
    def member_count(self) -> int:
        return len(self.member_indices)


@dataclass
class DedupResult:
    """Return bundle from :func:`resolve_library_effective`.

    Attributes:
        canonical_configs: The canonicalised configs after dedup (or all
            canonicalised configs when ``deduplicate=False``, with duplicates
            kept). This is what the runner iterates over.
        groups: One :class:`EquivalenceGroup` per unique resolved_config_hash, recording
            which indices of the input sweep collapsed together.
        declared_resolved_hashes: ``declared_index → resolved_config_hash`` - lets the runner tag
            the manifest entry for each run with its equivalence group.
        would_dedup: ``True`` iff any group has > 1 member (dedup would save
            runs even when ``deduplicate=False``).
        deduplicated: ``True`` iff dedup was actually applied to the
            ``canonical_configs`` return slot.
        dormant_observations: Distinct dormant normalisations applied across the
            sweep (dedup'd by engine/rule/field/effect, ordered), for plan /
            preflight display. Empty when no dormant rule fired.
    """

    canonical_configs: list[ExperimentConfig]
    groups: list[EquivalenceGroup] = field(default_factory=list)
    declared_resolved_hashes: list[str] = field(default_factory=list)
    would_dedup: bool = False
    deduplicated: bool = False
    dormant_observations: list[DormantObservation] = field(default_factory=list)


def resolve_library_effective(
    configs: list[ExperimentConfig],
    *,
    rules: list[Rule] | tuple[Rule, ...] | None = None,
    loader: EngineRulesLoader | None = None,
    deduplicate: bool = True,
) -> DedupResult:
    """Canonicalise then (optionally) resolved-config-hash dedup ``configs``.

    Rules are resolved lazily: if ``rules`` is None the loader is consulted
    per-engine for each config (cached by the loader instance). Callers
    running homogeneous sweeps may pass ``rules`` directly to skip the
    loader hop.

    Args:
        configs: Sweep-expanded declared configs.
        rules: Optional explicit rule list. Overrides the loader when the
            sweep is single-engine and the caller has a rules handle.
        loader: Optional ``EngineRulesLoader``. Defaults to a fresh one
            (per-process cache is internal to each instance).
        deduplicate: When ``False``, every declared config still runs -
            groups are computed for the equivalence-groups sidecar but the
            returned ``canonical_configs`` list has one entry per input.

    Returns:
        :class:`DedupResult` - see fields above.
    """
    if not configs:
        return DedupResult(canonical_configs=[])

    resolved_loader = loader or EngineRulesLoader()
    explicit_rules = tuple(rules) if rules is not None else None

    def _rules_for(cfg: ExperimentConfig) -> tuple[Rule, ...]:
        if explicit_rules is not None:
            return explicit_rules
        engine = engine_str(cfg.engine)
        try:
            return resolved_loader.load_rules(engine).rules
        except FileNotFoundError:
            return ()

    canonicalised: list[ExperimentConfig] = []
    hashes: list[str] = []
    observations: list[DormantObservation] = []
    for cfg in configs:
        canon = _apply_rules_fixpoint(cfg, _rules_for(cfg), observations=observations)
        canonicalised.append(canon)
        hashes.append(hash_config(build_resolved_view(canon)))

    # Distinct observations, first-seen order - the same rule fires on many
    # sweep points but the display wants one line per (engine, rule, field).
    # DormantObservation is frozen/hashable and dict preserves insertion order,
    # so dict.fromkeys gives first-seen dedup in one pass.
    distinct_observations = list(dict.fromkeys(observations))

    groups_by_hash: dict[str, list[int]] = {}
    for idx, h in enumerate(hashes):
        groups_by_hash.setdefault(h, []).append(idx)

    groups: list[EquivalenceGroup] = []
    for h1, indices in groups_by_hash.items():
        representative = indices[0]
        rep = canonicalised[representative]
        excerpt = _canonical_excerpt(rep)
        groups.append(
            EquivalenceGroup(
                resolved_config_hash=h1,
                canonical_excerpt=excerpt,
                member_indices=tuple(indices),
                representative_index=representative,
            )
        )

    would_dedup = any(g.member_count > 1 for g in groups)

    if deduplicate:
        selected = [canonicalised[g.representative_index] for g in groups]
    else:
        selected = list(canonicalised)

    return DedupResult(
        canonical_configs=selected,
        groups=groups,
        declared_resolved_hashes=hashes,
        would_dedup=would_dedup,
        deduplicated=deduplicate and would_dedup,
        dormant_observations=distinct_observations,
    )


def _canonical_excerpt(config: ExperimentConfig) -> dict[str, Any]:
    """Small human-readable excerpt of the canonical form for display/logs."""
    engine = engine_str(config.engine)
    excerpt: dict[str, Any] = {
        "engine": engine,
        "task.model": config.task.model,
    }
    sampling = config.active_sampling_params()
    if sampling is not None:
        dumped = sampling.model_dump(mode="python", exclude_none=True)
        for key, value in dumped.items():
            excerpt[f"{engine}.sampling_params.{key}"] = value
    return excerpt
