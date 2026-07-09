"""Schema-level checks on the shipped rules corpus (all engines).

The verification ladder (construction / runtime probes) is what earns a rule
its place in ``rules.yaml``; these tests are the offline backstop that the
committed files stay inside the closed schema and keep their semantic
coverage. They read the real shipped corpus, not fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llenergymeasure.config.engine_rules import (
    VALID_SEVERITY,
    VALID_SOURCE,
    VALID_VERIFIED,
    EngineRulesLoader,
)

ENGINES = ("transformers", "vllm", "tensorrt")


@pytest.fixture(scope="module")
def corpora():
    loader = EngineRulesLoader()
    return {engine: loader.load_rules(engine) for engine in ENGINES}


@pytest.fixture(scope="module")
def transformers_corpus(corpora):
    return corpora["transformers"]


def test_corpus_covers_required_invariants(transformers_corpus) -> None:
    """Coverage-by-rule: every required field surface has at least one rule.

    Pins SEMANTIC coverage, not specific rule IDs. Renames don't break this
    test; regressions that drop coverage of a real constraint do. If a path
    drops out, investigate WHY (real extraction regression, real library
    change, or verification gap) before weakening this list.
    """
    rules = transformers_corpus.rules

    def covers_field(field_path: str) -> bool:
        return any(field_path in rule.match_fields for rule in rules)

    required_fields = (
        # Greedy dormancy: do_sample=False / num_beams=1 strip these.
        "transformers.sampling_params.temperature",
        "transformers.sampling_params.top_p",
        "transformers.sampling_params.top_k",
        "transformers.sampling_params.min_p",
        "transformers.sampling_params.typical_p",
        "transformers.sampling_params.epsilon_cutoff",
        "transformers.sampling_params.eta_cutoff",
        # Single-beam dormancy (num_beams/early_stopping/length_penalty are
        # engine-construction knobs under the nested shape).
        "transformers.engine_params.early_stopping",
        "transformers.engine_params.length_penalty",
        # Note: num_beam_groups + diversity_penalty validations were softened
        # in transformers 4.57.x (error -> announced or no-op). Coverage loss
        # tracked separately; do NOT re-add without first confirming the
        # library re-introduced enforcement.
        # No-return-dict dormancy.
        "transformers.sampling_params.output_scores",
        "transformers.sampling_params.output_attentions",
        "transformers.sampling_params.output_hidden_states",
        # GenerationConfig.validate() error rules.
        "transformers.sampling_params.max_new_tokens",
        "transformers.engine_params.cache_implementation",
        "transformers.sampling_params.num_return_sequences",
        "transformers.sampling_params.pad_token_id",
        "transformers.sampling_params.compile_config",
        # Cross-field beam-search gates.
        "transformers.engine_params.num_beams",
        # Watermarking + BNB type-check paths.
        "transformers.sampling_params.watermarking_config",
        "transformers.engine_params.load_in_4bit",
        "transformers.engine_params.load_in_8bit",
        # Note: the four llm_int8_* BitsAndBytesConfig type-check rules shipped
        # for transformers 4.57.3 dropped out at the 5.7.0 pin - those fields are
        # no longer on the engine config surface (the generated Config exposes
        # load_in_8bit + bnb_4bit_* but not llm_int8_*), so a match-field rule on
        # them is unreachable. Do NOT re-add without confirming the library
        # re-exposes them.
        "transformers.engine_params.bnb_4bit_quant_type",
        "transformers.engine_params.bnb_4bit_use_double_quant",
    )
    missing = [path for path in required_fields if not covers_field(path)]
    assert not missing, (
        f"corpus is missing rules for {len(missing)} required constraints: {missing}"
    )

    # Cross-field rules - at least one rule must AND-combine the listed
    # fields. Catches regressions that lose the cross-field predicate.
    cross_field_pairs = (
        (
            "transformers.engine_params.num_beams",
            "transformers.sampling_params.num_return_sequences",
        ),
    )
    missing_pairs = [
        pair
        for pair in cross_field_pairs
        if not any(all(p in rule.match_fields for p in pair) for rule in rules)
    ]
    assert not missing_pairs, (
        f"corpus missing cross-field rules for {len(missing_pairs)} constraints: {missing_pairs}"
    )


def test_cross_section_field_refs_fire(corpora) -> None:
    """Cross-section @field_ref rules resolve and fire (firing regression guard).

    Bare @refs resolve as siblings within the anchor field's parent section
    only. When the nested re-home split num_beams (engine_params) from
    num_return_sequences (sampling_params), sibling-form refs silently
    resolved to None and the rules went dead without any test noticing -
    the presence check above cannot catch a rule that exists but never
    fires. Refs that cross sections must use the root-dotted form.
    """
    rules = corpora["transformers"].rules

    violating = {
        "transformers": {
            "engine_params": {"num_beams": 2},
            "sampling_params": {"num_return_sequences": 4},
        }
    }
    fired = {inv.id for inv in rules if inv.try_match(violating)}
    assert "transformers_num_return_vs_beams_num_beams_lt_num_return_sequences" in fired

    return_gt_beams = {
        "transformers": {
            "engine_params": {"num_beams": 4},
            "sampling_params": {"num_return_sequences": 5},
        }
    }
    fired = {inv.id for inv in rules if inv.try_match(return_gt_beams)}
    assert "transformers_num_return_vs_beams_num_return_sequences_gt_num_beams" in fired

    # num_return_sequences <= num_beams but not a divisor: valid upstream, must
    # not fire (regression guard for the not_divisible_by -> <= re-encode).
    non_divisor_ok = {
        "transformers": {
            "engine_params": {"num_beams": 4},
            "sampling_params": {"num_return_sequences": 3},
        }
    }
    fired = {inv.id for inv in rules if inv.try_match(non_divisor_ok)}
    assert not fired & {
        "transformers_num_return_vs_beams_num_beams_lt_num_return_sequences",
        "transformers_num_return_vs_beams_num_return_sequences_gt_num_beams",
    }


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_schema_version_is_current(corpora, engine: str) -> None:
    assert corpora[engine].schema_version.startswith("1.")


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_engine_version_present(corpora, engine: str) -> None:
    assert corpora[engine].engine_version, f"{engine}: envelope missing engine_version"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_ids_unique(corpora, engine: str) -> None:
    ids = [rule.id for rule in corpora[engine].rules]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_match_fields_non_empty(corpora, engine: str) -> None:
    for rule in corpora[engine].rules:
        assert rule.match_fields, f"Rule {rule.id} has empty match.fields"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_severity_values_are_valid(corpora, engine: str) -> None:
    # Redundant with the loader's UnknownSeverityError (defence-in-depth):
    # the shipped files carry only the closed {error, dormant} set.
    for rule in corpora[engine].rules:
        assert rule.severity in VALID_SEVERITY, rule.id


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_provenance_complete(corpora, engine: str) -> None:
    for rule in corpora[engine].rules:
        prov = rule.provenance
        assert prov.source in VALID_SOURCE, f"{rule.id}: source={prov.source!r}"
        assert prov.verified in VALID_VERIFIED, f"{rule.id}: verified={prov.verified!r}"
        assert prov.engine_version, f"{rule.id}: provenance missing engine_version"
        assert prov.date, f"{rule.id}: provenance missing date"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_provenance_pinned_to_envelope_version(corpora, engine: str) -> None:
    # Rule-level verdicts are version-scoped: a shipped rule verified against a
    # different engine version than the envelope's would be stale knowledge.
    envelope_version = corpora[engine].engine_version
    for rule in corpora[engine].rules:
        assert rule.provenance.engine_version == envelope_version, (
            f"{rule.id}: verified at {rule.provenance.engine_version!r} "
            f"but envelope pins {envelope_version!r}"
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_dormant_rules_carry_normalised_fields(corpora, engine: str) -> None:
    # A dormant rule's runtime action is canonicalise-for-dedup; without
    # normalised_fields the dedup consumer falls back to predicate projection.
    # Error rules must never carry normalisation targets.
    for rule in corpora[engine].rules:
        if rule.severity == "error":
            assert not rule.normalised_fields, f"{rule.id}: error rule carries normalised_fields"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_file_is_valid_yaml(engine: str) -> None:
    # Sanity: the on-disk file is parseable YAML (redundant with loader, but
    # guards against accidental corruption by direct edits).
    path = (
        Path(__file__).resolve().parents[4]
        / "src"
        / "llenergymeasure"
        / "engines"
        / engine
        / "rules.yaml"
    )
    assert path.exists()
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict)
    assert "rules" in doc
