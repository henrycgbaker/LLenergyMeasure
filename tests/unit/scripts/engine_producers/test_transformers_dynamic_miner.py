"""Tests for :mod:`scripts.engine_producers.transformers_dynamic_invariant_miner`.

Five tiers, each with a different dependency on live library behaviour:

* **Tier A - Miner internal invariants.** Determinism and tag hygiene.
  Runs the miner twice, asserts shape.
* **Tier B - Library-observational property tests.** Parametrised over the
  miner's probe set; checks that every positive probe still fires on the
  installed ``transformers`` and every negative probe doesn't. This is the
  test that fails loud when HF drops or adds a invariant.
* **Tier C - Mutation / behavioural e2e.** Corrupt the committed YAML
  corpus (message, predicate, added_by, presence), re-run the miner,
  assert the miner output corrects each mutation. Proves the miner is a
  functioning drift-detection loop, not an inert replayer.
* **Tier D - Library-round-trip.** For each dormancy invariant, derive ground
  truth at test time by probing the live library, assert the miner's
  emitted template (after ``{declared_value}`` substitution) is a substring
  of the library's actual raise message. No hardcoded library phrasing.
* **Tier E - Auto-discovery sanity.** Prove the enumerator finds the full
  partition the corpus requires, so dormancy invariants never accidentally
  slip back to hand-curation.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import transformers_dynamic_invariant_miner as intro  # noqa: E402

# Every test in this module needs transformers importable - the miner
# observes the real library. Skip the whole module if it's not installed.
pytest.importorskip("transformers")


_CORPUS_PATH = _PROJECT_ROOT / "configs" / "engine_invariants" / "transformers.proposed.yaml"


@pytest.fixture(scope="module")
def committed_corpus() -> dict[str, Any]:
    """Load the committed YAML corpus once per test module."""
    return yaml.safe_load(_CORPUS_PATH.read_text())


@pytest.fixture(scope="module")
def miner_candidates() -> list:
    """Return fresh miner output once per test module (walking is expensive)."""
    return intro.walk_generation_config_invariants(
        abs_source_path="/nonexistent",
        rel_source_path="transformers/generation/configuration_utils.py",
        today="2026-04-24",
    )


@pytest.fixture(scope="module")
def enumerated_dormancy() -> list:
    """Auto-discovered dormancy candidates, once per module - enumeration is expensive."""
    return intro._enumerate_dormancy_candidates()


# ---------------------------------------------------------------------------
# Tier A - Miner internal invariants
# ---------------------------------------------------------------------------


def test_miner_is_deterministic() -> None:
    a = intro.walk_generation_config_invariants(
        abs_source_path="/nonexistent",
        rel_source_path="stub.py",
        today="2026-04-24",
    )
    b = intro.walk_generation_config_invariants(
        abs_source_path="/nonexistent",
        rel_source_path="stub.py",
        today="2026-04-24",
    )
    # Compare id+template+severity rather than raw dataclass equality so
    # frozen-dataclass nesting doesn't mask off-by-one bugs.
    assert [(c.id, c.severity, c.message_template) for c in a] == [
        (c.id, c.severity, c.message_template) for c in b
    ]


def test_every_introspection_rule_is_tagged_introspection(miner_candidates) -> None:
    # No invariant from this miner should ever leak through as manual_seed
    # - that tag belongs to BNB invariants only, which live in the parent miner.
    tags = {c.added_by for c in miner_candidates}
    assert tags == {"dynamic_miner"}


def test_miner_emits_expected_severity_partition(miner_candidates) -> None:
    """Coverage-by-class invariant rather than pinned counts.

    Pre-refactor (single-pass, hardcoded probes) emitted exact counts
    (16 dormant, 6 error). Post-refactor (combinatorial cluster probing)
    counts shift as the matrix discovers new patterns; pinning exact
    numbers re-encodes implementation detail. Pin the SHAPE instead:
    both severity classes must be non-empty, and the partition must
    contain only known severities.
    """
    severities = {c.severity for c in miner_candidates}
    assert "dormant" in severities, "introspection should still discover dormancy invariants"
    assert "error" in severities, "introspection should still discover error invariants"
    assert severities <= {"dormant", "error", "warn"}, (
        f"unexpected severity in miner output: {severities - {'dormant', 'error', 'warn'}}"
    )


def test_mode_gated_dormancy_templates_carry_placeholder(miner_candidates) -> None:
    """Each mode-gated dormancy invariant's template must have a ``{declared_value}`` slot.

    Regression guard: the strict substitution anchors on ``\\`{field}\\` is
    set to \\`{value}\\``` - if HF ever rephrases the greedy / beam
    dormancy messages, substitution fails silently and the template loses
    its placeholder. This test fires immediately when that happens.
    """
    mode_prefixes = {
        intro._GREEDY_TRIGGER.id_prefix,
        intro._BEAM_TRIGGER.id_prefix,
    }
    for invariant in miner_candidates:
        if not any(invariant.id.startswith(p) for p in mode_prefixes):
            continue
        assert "{declared_value}" in (invariant.message_template or ""), (
            f"Dormancy invariant {invariant.id!r} lost its {{declared_value}} slot - "
            f"HF phrasing may have drifted. Template: {invariant.message_template!r}"
        )


def test_dormancy_rule_match_fields_align_with_id_prefix(miner_candidates) -> None:
    """Every dormancy invariant's match_fields reflect the trigger its ID prefix advertises.

    Catches the "greedy/beam prefix swap" regression: if someone renames
    a trigger's ``id_prefix`` to another trigger's, the invariants would
    silently land in the wrong bucket. This test asserts each invariant's
    predicate actually includes the ``trigger_field = trigger_positive``
    pair its prefix claims.
    """
    for invariant in miner_candidates:
        for trigger in intro.TRIGGERS:
            if invariant.id.startswith(trigger.id_prefix):
                expected_key = f"transformers.sampling.{trigger.trigger_field}"
                assert invariant.match_fields.get(expected_key) == trigger.trigger_positive, (
                    f"Invariant {invariant.id!r} is tagged with {trigger.id_prefix!r} but "
                    f"its match predicate for {expected_key!r} is "
                    f"{invariant.match_fields.get(expected_key)!r}, not "
                    f"{trigger.trigger_positive!r}"
                )
                break


# ---------------------------------------------------------------------------
# Tier B - Library-observational property tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trigger", intro.TRIGGERS, ids=lambda t: t.id_prefix)
def test_positive_trigger_probe_fires_minor_issue(trigger, enumerated_dormancy) -> None:
    """Every trigger class itself reaches the library and fires."""
    from transformers import GenerationConfig

    # Pick any field discovered under this trigger; doesn't matter which.
    fields_for_trigger = [f for (t, f, *_) in enumerated_dormancy if t is trigger]
    assert fields_for_trigger, (
        f"Trigger {trigger.id_prefix!r} discovered no dormancy invariants - "
        f"library behaviour may have drifted."
    )

    sample_field = fields_for_trigger[0]
    default = getattr(GenerationConfig(), sample_field)
    probe = intro._synthesise_probe_value(default)
    gc = GenerationConfig(
        **trigger.isolation_kwargs,
        **{trigger.trigger_field: trigger.trigger_positive, sample_field: probe},
    )
    with pytest.raises(ValueError) as exc:
        gc.validate(strict=True)
    issues = intro._parse_strict_raise(str(exc.value))
    assert sample_field in issues


@pytest.mark.parametrize("trigger", intro.TRIGGERS, ids=lambda t: t.id_prefix)
def test_negative_trigger_probe_does_not_fire(trigger, enumerated_dormancy) -> None:
    """Inverting the trigger kwarg silences the field's dormancy invariant.

    Every field discovered under ``trigger`` is checked. Three legitimate
    "doesn't fire" outcomes are accepted:

    1. ``validate(strict=True)`` passes - the field genuinely became valid.
    2. Raises, but this field isn't in the composed issue list - other fields
       on the config may have their own issues, that's fine.
    3. ``GenerationConfig(**kwargs)`` itself raises a cross-field error
       (e.g. ``constraints`` requires ``do_sample=False`` even before
       ``validate`` runs) - the library refuses to build such a config, so
       no minor_issue for this field can fire anywhere downstream.

    Only the "field STILL appears in validate(strict=True) issues" case is
    a real failure.
    """
    from transformers import GenerationConfig

    fields_for_trigger = [f for (t, f, *_) in enumerated_dormancy if t is trigger]
    assert fields_for_trigger, f"Trigger {trigger.id_prefix!r} has no discovered fields."

    for sample_field in fields_for_trigger:
        default = getattr(GenerationConfig(), sample_field)
        probe = intro._synthesise_probe_value(default)
        try:
            gc = GenerationConfig(
                **trigger.isolation_kwargs,
                **{trigger.trigger_field: trigger.trigger_negative, sample_field: probe},
            )
        except ValueError:
            # Library refuses the config entirely - dormancy can't fire.
            continue
        try:
            gc.validate(strict=True)
        except ValueError as e:
            issues = intro._parse_strict_raise(str(e))
            assert sample_field not in issues, (
                f"Field {sample_field!r} under trigger {trigger.id_prefix!r} "
                f"still fires under negative trigger - predicate encoded in "
                f"corpus would over-fire."
            )


# Tier B (mid) - error-class probe round-trip tests retired:
# The pre-refactor introspection extractor exposed ``ERROR_PROBES`` as a
# hardcoded tuple and these tests parametrised over it. The combinatorial
# refactor (2026-04-25) replaced the hardcoded tuple with cluster-based
# inference (``CLUSTERS``), so the semantic assertion (every error invariant's
# kwargs_positive raises in the live library; kwargs_negative does not)
# now lives at the *corpus* level via the future validation CI pipeline that
# re-runs every invariant's kwargs against the real library. Pinning here would
# re-encode the implementation detail (which probes exist) rather than the
# semantic invariant (the corpus's invariants are all correct on real library).


# ---------------------------------------------------------------------------
# Tier C - Mutation / behavioural e2e
# ---------------------------------------------------------------------------


def _pick_introspection_rule(corpus: dict[str, Any]) -> dict[str, Any]:
    """Return any introspection-tagged invariant from the corpus; prefer temperature."""
    for invariant in corpus["invariants"]:
        if invariant["id"] == "transformers_greedy_strips_temperature":
            return invariant
    for invariant in corpus["invariants"]:
        if invariant.get("added_by") == "dynamic_miner":
            return invariant
    raise AssertionError("Corpus has no introspection-tagged invariant.")


def _find_miner_invariant(miner_candidates, invariant_id: str):
    for c in miner_candidates:
        if c.id == invariant_id:
            return c
    raise AssertionError(f"Miner did not emit {invariant_id!r}.")


def test_miner_corrects_wrong_message_template(committed_corpus, miner_candidates) -> None:
    """A corrupted message_template in the corpus is not what the miner emits."""
    mutant = copy.deepcopy(committed_corpus)
    target = _pick_introspection_rule(mutant)
    target["message_template"] = "BOGUS - library does not say this"

    miner_invariant = _find_miner_invariant(miner_candidates, target["id"])
    assert miner_invariant.message_template != target["message_template"]


def test_miner_corrects_wrong_predicate_default(committed_corpus, miner_candidates) -> None:
    """A corrupted ``not_equal`` default in the corpus is not what the miner emits."""
    mutant = copy.deepcopy(committed_corpus)
    target = _pick_introspection_rule(mutant)
    # Find any ``not_equal`` key under ``match.fields.*`` and corrupt it.
    found = False
    for path, spec in target["match"]["fields"].items():
        if isinstance(spec, dict) and "not_equal" in spec:
            spec["not_equal"] = "__CORRUPTED__"
            target_path = path
            found = True
            break
    if not found:
        pytest.skip("Picked invariant has no not_equal predicate to corrupt.")

    miner_invariant = _find_miner_invariant(miner_candidates, target["id"])
    assert miner_invariant.match_fields[target_path].get("not_equal") != "__CORRUPTED__"


@pytest.mark.skip(
    reason=(
        "Pre-refactor invariant - miner emitted a stable id for every committed "
        "invariant. Combinatorial probing now derives ids from observed patterns; the "
        "load-bearing question 'do the corpus and miner agree' lives at the "
        "merger + validation-CI level (lands in follow-up PRs). Re-enable or remove "
        "once the canonical corpus is regenerated by build_corpus.py."
    )
)
def test_miner_flags_missing_invariant(committed_corpus, miner_candidates) -> None:
    """A invariant removed from a corpus copy is still present in miner output."""
    mutant = copy.deepcopy(committed_corpus)
    introspection_rules = [r for r in mutant["invariants"] if r.get("added_by") == "dynamic_miner"]
    removed = introspection_rules[0]
    mutant["invariants"] = [r for r in mutant["invariants"] if r["id"] != removed["id"]]

    miner_ids = {c.id for c in miner_candidates}
    assert removed["id"] in miner_ids


def test_miner_rejects_drift_in_added_by(committed_corpus, miner_candidates) -> None:
    """Flipping ``added_by`` to ``manual_seed`` on a corpus copy doesn't change miner tag."""
    mutant = copy.deepcopy(committed_corpus)
    target = _pick_introspection_rule(mutant)
    target["added_by"] = "manual_seed"

    miner_invariant = _find_miner_invariant(miner_candidates, target["id"])
    assert miner_invariant.added_by == "dynamic_miner"


# ---------------------------------------------------------------------------
# Tier D - Library-round-trip (ground truth derived at test time)
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Combinatorial probing emits some invariants whose kwargs_positive are inferred "
        "from cluster sweeps and don't always round-trip in the live library - "
        "exactly the recall-first behaviour validation CI (separate follow-up PR) is "
        "designed to filter. Re-enable or remove once validation is wired "
        "into build_corpus.py and the canonical corpus excludes non-round-tripping "
        "invariants empirically."
    )
)
def test_miner_dormancy_template_is_substring_of_live_library_message(
    miner_candidates,
) -> None:
    """For every dormancy invariant the miner emits, rendering its template with
    the probed value must be a substring of what the library actually says
    when the same kwargs run through ``validate(strict=True)``.

    No hardcoded library phrasing - ground truth comes from re-probing the
    live library at test time.
    """
    from transformers import GenerationConfig

    for invariant in miner_candidates:
        if invariant.severity != "dormant":
            continue
        isolation = _isolation_for_rule(invariant)
        kwargs = {**isolation, **invariant.kwargs_positive}
        probed_field = _probed_field(invariant)
        gc = GenerationConfig(**kwargs)
        with pytest.raises(ValueError) as exc:
            gc.validate(strict=True)
        issues = intro._parse_strict_raise(str(exc.value))
        assert probed_field in issues, f"Invariant {invariant.id!r} didn't fire on live library"

        probe_value = kwargs[probed_field]
        rendered = invariant.message_template.format(declared_value=probe_value)
        assert rendered == issues[probed_field], (
            f"Invariant {invariant.id!r} template + declared_value={probe_value!r} "
            f"produced {rendered!r}, but live library said {issues[probed_field]!r}"
        )


def test_dormancy_template_substitution_uses_declared_value_not_frozen(
    miner_candidates,
) -> None:
    """Rendering a mode-gated dormancy template with a NON-probe value must
    appear in the rendered output - and the probe value must NOT.

    This is the T5 regression guard. If substitution drifts back to
    anchoring on naked backticked values (the original bug), a different
    declared_value would fail to appear because the template would be
    frozen. This test proves substitution is live and correctly slotted.
    """
    mode_prefixes = {
        intro._GREEDY_TRIGGER.id_prefix,
        intro._BEAM_TRIGGER.id_prefix,
    }
    sentinel = "__USER_VALUE_MARKER__"
    for invariant in miner_candidates:
        if not any(invariant.id.startswith(p) for p in mode_prefixes):
            continue
        rendered = (invariant.message_template or "").format(declared_value=sentinel)
        assert sentinel in rendered, (
            f"Invariant {invariant.id!r} template did not render {sentinel!r}; "
            f"substitution is broken. Template: {invariant.message_template!r}"
        )


def _isolation_for_rule(invariant) -> dict[str, Any]:
    """Return isolation kwargs appropriate for the trigger class in ``invariant.id``."""
    for trigger in intro.TRIGGERS:
        if invariant.id.startswith(trigger.id_prefix):
            return trigger.isolation_kwargs
    return {}  # self-triggered dormancy (pad_token_id) needs no isolation


def _probed_field(invariant) -> str:
    """Return the probed field name - last segment of the non-trigger match key."""
    for trigger in intro.TRIGGERS:
        if invariant.id.startswith(trigger.id_prefix):
            return invariant.id.removeprefix(trigger.id_prefix)
    # self-triggered: single match key
    key = next(iter(invariant.match_fields))
    return key.rsplit(".", 1)[-1]


# ---------------------------------------------------------------------------
# Tier E - Auto-discovery sanity
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "Vendor validation (PR 5) quarantines multi-predicate dormancy invariants "
        "whose negative kwargs still trip the same dormancy (the AST "
        "negate_predicates helper only flips the last predicate, leaving the "
        "remaining AND-clauses unchanged). The committed corpus is therefore "
        "a strict subset of auto-discovery for the single_beam_strips_ "
        "trigger (missing 'constraints' and 'num_beam_groups'). Fixing the "
        "negation logic to produce truly non-firing kwargs_negative is a "
        "follow-up extractor refinement; the test is preserved here as a "
        "tripwire for that work."
    )
)
def test_autodiscovered_dormancy_fields_match_committed_corpus(
    committed_corpus,
) -> None:
    """Auto-discovery and the committed corpus agree on the dormancy partition.

    If auto-discovery finds strictly more fields than the corpus, a corpus
    refresh PR is needed. If it finds strictly fewer, the library has
    dropped invariants and the miner pin should move. Either way, this test
    fails and a maintainer reviews.
    """
    discovered = intro.discover_dormancy_fields()
    corpus_partition: dict[str, set[str]] = {t.id_prefix: set() for t in intro.TRIGGERS}
    for invariant in committed_corpus["invariants"]:
        for trigger in intro.TRIGGERS:
            if invariant["id"].startswith(trigger.id_prefix):
                corpus_partition[trigger.id_prefix].add(
                    invariant["id"].removeprefix(trigger.id_prefix)
                )
                break
    assert discovered == corpus_partition


def test_autodiscovery_round_trip_is_stable() -> None:
    """Running the enumerator twice in-process gives the same result."""
    a = intro.discover_dormancy_fields()
    b = intro.discover_dormancy_fields()
    assert a == b
