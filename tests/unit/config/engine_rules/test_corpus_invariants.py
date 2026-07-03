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
    EngineInvariantsLoader,
)

ENGINES = ("transformers", "vllm", "tensorrt")


@pytest.fixture(scope="module")
def corpora():
    loader = EngineInvariantsLoader()
    return {engine: loader.load_invariants(engine) for engine in ENGINES}


@pytest.fixture(scope="module")
def transformers_corpus(corpora):
    return corpora["transformers"]


def test_corpus_covers_required_invariants(transformers_corpus) -> None:
    """Coverage-by-invariant: every required field surface has at least one rule.

    Pins SEMANTIC coverage, not specific rule IDs. Renames don't break this
    test; regressions that drop coverage of a real constraint do. If a path
    drops out, investigate WHY (real extraction regression, real library
    change, or verification gap) before weakening this list.
    """
    invariants = transformers_corpus.invariants

    def covers_field(field_path: str) -> bool:
        return any(field_path in invariant.match_fields for invariant in invariants)

    required_fields = (
        # Greedy dormancy: do_sample=False / num_beams=1 strip these.
        "transformers.sampling.temperature",
        "transformers.sampling.top_p",
        "transformers.sampling.top_k",
        "transformers.sampling.min_p",
        "transformers.sampling.typical_p",
        "transformers.sampling.epsilon_cutoff",
        "transformers.sampling.eta_cutoff",
        # Single-beam dormancy.
        "transformers.sampling.early_stopping",
        "transformers.sampling.length_penalty",
        # Note: num_beam_groups + diversity_penalty validations were softened
        # in transformers 4.57.x (error -> announced or no-op). Coverage loss
        # tracked separately; do NOT re-add without first confirming the
        # library re-introduced enforcement.
        # No-return-dict dormancy.
        "transformers.sampling.output_scores",
        "transformers.sampling.output_attentions",
        "transformers.sampling.output_hidden_states",
        # GenerationConfig.validate() error rules.
        "transformers.sampling.max_new_tokens",
        "transformers.sampling.cache_implementation",
        "transformers.sampling.num_return_sequences",
        "transformers.sampling.pad_token_id",
        "transformers.sampling.compile_config",
        # Cross-field beam-search gates.
        "transformers.sampling.num_beams",
        # Watermarking + BNB type-check paths.
        "transformers.sampling.watermarking_config",
        "transformers.load_in_4bit",
        "transformers.load_in_8bit",
        "transformers.llm_int8_threshold",
        "transformers.llm_int8_skip_modules",
        "transformers.llm_int8_enable_fp32_cpu_offload",
        "transformers.llm_int8_has_fp16_weight",
        "transformers.bnb_4bit_quant_type",
        "transformers.bnb_4bit_use_double_quant",
    )
    missing = [path for path in required_fields if not covers_field(path)]
    assert not missing, (
        f"corpus is missing rules for {len(missing)} required constraints: {missing}"
    )

    # Cross-field rules - at least one rule must AND-combine the listed
    # fields. Catches regressions that lose the cross-field predicate.
    cross_field_pairs = (
        ("transformers.sampling.num_beams", "transformers.sampling.num_return_sequences"),
    )
    missing_pairs = [
        pair
        for pair in cross_field_pairs
        if not any(all(p in invariant.match_fields for p in pair) for invariant in invariants)
    ]
    assert not missing_pairs, (
        f"corpus missing cross-field rules for {len(missing_pairs)} constraints: {missing_pairs}"
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_schema_version_is_current(corpora, engine: str) -> None:
    assert corpora[engine].schema_version.startswith("1.")


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_engine_version_present(corpora, engine: str) -> None:
    assert corpora[engine].engine_version, f"{engine}: envelope missing engine_version"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_ids_unique(corpora, engine: str) -> None:
    ids = [invariant.id for invariant in corpora[engine].invariants]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_match_fields_non_empty(corpora, engine: str) -> None:
    for invariant in corpora[engine].invariants:
        assert invariant.match_fields, f"Rule {invariant.id} has empty match.fields"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_severity_values_are_valid(corpora, engine: str) -> None:
    # Redundant with the loader's UnknownSeverityError (defence-in-depth):
    # the shipped files carry only the closed {error, dormant} set.
    for invariant in corpora[engine].invariants:
        assert invariant.severity in VALID_SEVERITY, invariant.id


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_provenance_complete(corpora, engine: str) -> None:
    for invariant in corpora[engine].invariants:
        prov = invariant.provenance
        assert prov.source in VALID_SOURCE, f"{invariant.id}: source={prov.source!r}"
        assert prov.verified in VALID_VERIFIED, f"{invariant.id}: verified={prov.verified!r}"
        assert prov.engine_version, f"{invariant.id}: provenance missing engine_version"
        assert prov.date, f"{invariant.id}: provenance missing date"


@pytest.mark.parametrize("engine", ENGINES)
def test_corpus_provenance_pinned_to_envelope_version(corpora, engine: str) -> None:
    # Rule-level verdicts are version-scoped: a shipped rule verified against a
    # different engine version than the envelope's would be stale knowledge.
    envelope_version = corpora[engine].engine_version
    for invariant in corpora[engine].invariants:
        assert invariant.provenance.engine_version == envelope_version, (
            f"{invariant.id}: verified at {invariant.provenance.engine_version!r} "
            f"but envelope pins {envelope_version!r}"
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_dormant_rules_carry_normalised_fields(corpora, engine: str) -> None:
    # A dormant rule's runtime action is canonicalise-for-dedup; without
    # normalised_fields the dedup consumer falls back to predicate projection.
    # Error rules must never carry normalisation targets.
    for invariant in corpora[engine].invariants:
        if invariant.severity == "error":
            assert not invariant.normalised_fields, (
                f"{invariant.id}: error rule carries normalised_fields"
            )


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
