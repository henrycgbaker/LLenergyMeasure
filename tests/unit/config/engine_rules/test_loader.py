"""Tests for :class:`EngineRulesLoader` and corpus envelope parsing."""

from __future__ import annotations

from pathlib import Path

import pytest

from llenergymeasure.config.engine_rules import (
    VALID_ADDED_BY,
    VALID_EMISSION_CHANNEL,
    VALID_OUTCOME,
    VALID_SEVERITY,
    EngineInvariants,
    EngineRulesLoader,
    LLMProposedSeverityError,
    UnknownAddedByError,
    UnknownEmissionChannelError,
    UnknownEnumValueError,
    UnknownOutcomeError,
    UnknownSeverityError,
    UnsupportedSchemaVersionError,
)

_CORPUS_MINIMAL = """\
schema_version: "1.0.0"
engine: transformers
engine_version: "4.56.0"
invariants:
  - id: transformers_test_rule
    engine: transformers
    library: transformers
    invariant_under_test: "Test invariant"
    severity: dormant
    native_type: transformers.GenerationConfig
    miner_source:
      path: transformers/generation/configuration_utils.py
      method: validate
      line_at_scan: 42
    match:
      engine: transformers
      fields:
        transformers.sampling.temperature: {present: true}
    kwargs_positive:
      temperature: 0.5
    kwargs_negative:
      temperature: null
    expected_outcome:
      outcome: dormant_announced
      emission_channel: minor_issues_dict
      normalised_fields: []
    message_template: "Dormant {declared_value}"
    references:
      - "ref"
    added_by: static_miner
    added_at: "2026-04-23"
"""


def _write_corpus(root: Path, engine: str, text: str) -> None:
    engine_dir = root / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "rules.proposed.yaml").write_text(text)


def test_load_rules_returns_parsed_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    corpus = loader.load_rules("transformers")
    assert isinstance(corpus, EngineInvariants)
    assert corpus.engine == "transformers"
    assert corpus.schema_version == "1.0.0"
    assert corpus.engine_version == "4.56.0"
    assert len(corpus.invariants) == 1
    assert corpus.invariants[0].id == "transformers_test_rule"


def test_load_rules_per_instance_cache(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    corpus1 = loader.load_rules("transformers")
    corpus2 = loader.load_rules("transformers")
    # Same identity: pulled from cache on second call.
    assert corpus1 is corpus2


def test_invalidate_clears_cache(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    first = loader.load_rules("transformers")
    loader.invalidate("transformers")
    second = loader.load_rules("transformers")
    # Different instances: cache was cleared.
    assert first is not second


def test_invalidate_all_clears_all(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    loader.load_rules("transformers")
    assert loader._cache
    loader.invalidate()
    assert not loader._cache


def test_missing_corpus_raises_file_not_found(tmp_path: Path) -> None:
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        loader.load_rules("transformers")


def test_unsupported_major_version_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace('"1.0.0"', '"2.0.0"', 1)
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnsupportedSchemaVersionError):
        loader.load_rules("transformers")


def test_missing_schema_version_raises(tmp_path: Path) -> None:
    corpus = """\
engine: transformers
engine_version: "4.56.0"
invariants: []
"""
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnsupportedSchemaVersionError):
        loader.load_rules("transformers")


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", "- just a list")
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        loader.load_rules("transformers")


def test_empty_rules_list_is_valid(tmp_path: Path) -> None:
    corpus = """\
schema_version: "1.0.0"
engine: transformers
engine_version: "4.56.0"
invariants: []
"""
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    result = loader.load_rules("transformers")
    assert result.invariants == ()


_CORPUS_BARE_NORMALISED = """\
schema_version: "1.0.0"
engine: vllm
engine_version: "0.7.3"
invariants:
  - id: vllm_seed_dormant
    engine: vllm
    library: vllm
    invariant_under_test: "seed dormant at -1"
    severity: dormant
    native_type: vllm.SamplingParams
    miner_source:
      path: sampling_params.py
      method: __post_init__
      line_at_scan: 311
    match:
      engine: vllm
      fields:
        vllm.sampling_params.seed: -1
    kwargs_positive:
      seed: -1
    kwargs_negative:
      seed: 0
    expected_outcome:
      outcome: dormant_silent
      emission_channel: none
      normalised_fields:
        - seed
    message_template: null
    references:
      - "sampling_params.py:311"
    added_by: static_miner
    added_at: "2026-04-27"
"""


def test_bare_normalised_field_normalised_to_dotted_path_at_load(tmp_path: Path) -> None:
    """A bare ``normalised_fields`` name is rewritten to the dotted config path.

    The corpus records bare engine field names; the dedup canonicaliser's
    fixpoint assigns them via ``resolve_field_path`` which needs the full dotted
    path. The loader derives the section prefix from the rule's own match-field
    key (``vllm.sampling_params.seed``) and rewrites ``seed`` -> that path, so the
    fixpoint can strip it. One load-time fix, no edit to the committed corpus.
    """
    _write_corpus(tmp_path, "vllm", _CORPUS_BARE_NORMALISED)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    (rule,) = loader.load_rules("vllm").invariants
    assert rule.expected_outcome["normalised_fields"] == ["vllm.sampling_params.seed"]


def test_dotted_normalised_field_passes_through_unchanged(tmp_path: Path) -> None:
    """An already-dotted ``normalised_fields`` entry is left untouched."""
    text = _CORPUS_BARE_NORMALISED.replace("        - seed", "        - vllm.sampling_params.seed")
    _write_corpus(tmp_path, "vllm", text)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    (rule,) = loader.load_rules("vllm").invariants
    assert rule.expected_outcome["normalised_fields"] == ["vllm.sampling_params.seed"]


def test_default_corpus_root_resolves_to_engines(tmp_path: Path) -> None:
    # Constructing without corpus_root uses the repo's src/llenergymeasure/engines/.
    loader = EngineRulesLoader()
    assert loader.corpus_root.name == "engines"
    assert loader.corpus_root.parent.name == "llenergymeasure"


# ---------------------------------------------------------------------------
# AddedBy provenance enum
# ---------------------------------------------------------------------------


def test_valid_added_by_set_has_all_provenance_classes() -> None:
    assert (
        frozenset(
            {
                "static_miner",
                "dynamic_miner",
                "pydantic_lift",
                "msgspec_lift",
                "dataclass_lift",
                "manual_seed",
                "runtime_warning",
                "observed_collision",
                "reclassified_decayed_announcement",
                "llm_diagnose",
            }
        )
        == VALID_ADDED_BY
    )


@pytest.mark.parametrize(
    "provenance",
    [
        "static_miner",
        "dynamic_miner",
        "pydantic_lift",
        "msgspec_lift",
        "dataclass_lift",
        "manual_seed",
        "runtime_warning",
        "observed_collision",
        "reclassified_decayed_announcement",
        "llm_diagnose",
    ],
)
def test_all_added_by_values_round_trip(tmp_path: Path, provenance: str) -> None:
    corpus = _CORPUS_MINIMAL.replace("added_by: static_miner", f"added_by: {provenance}")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    invariants = loader.load_rules("transformers").invariants
    assert invariants[0].added_by == provenance


def test_unknown_added_by_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace(
        "added_by: static_miner", "added_by: chatgpt_hallucination"
    )
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownAddedByError, match="chatgpt_hallucination"):
        loader.load_rules("transformers")


def test_missing_added_by_defaults_to_manual_seed(tmp_path: Path) -> None:
    # Omitting added_by is not a corpus authoring error - unknown provenance
    # falls back to manual_seed (the conservative default).
    corpus = _CORPUS_MINIMAL.replace("    added_by: static_miner\n", "")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    invariants = loader.load_rules("transformers").invariants
    assert invariants[0].added_by == "manual_seed"


# ---------------------------------------------------------------------------
# Closed-enum validation - severity / outcome / emission_channel
# ---------------------------------------------------------------------------


def test_valid_severity_set_matches_design_spec() -> None:
    assert frozenset({"dormant", "warn", "error"}) == VALID_SEVERITY


def test_valid_outcome_set_matches_design_spec() -> None:
    assert (
        frozenset({"dormant_silent", "dormant_announced", "warn", "error", "pass"}) == VALID_OUTCOME
    )


def test_valid_emission_channel_set_matches_design_spec() -> None:
    assert (
        frozenset(
            {
                "warnings_warn",
                "logger_warning",
                "logger_warning_once",
                "minor_issues_dict",
                "none",
                "runtime_exception",
            }
        )
        == VALID_EMISSION_CHANNEL
    )


def test_unknown_severity_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace("severity: dormant", "severity: kritical")
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownSeverityError, match="kritical"):
        loader.load_rules("transformers")


def test_unknown_outcome_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace("outcome: dormant_announced", "outcome: totally_made_up")
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownOutcomeError, match="totally_made_up"):
        loader.load_rules("transformers")


def test_unknown_emission_channel_value_raises(tmp_path: Path) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace(
        "emission_channel: minor_issues_dict", "emission_channel: smoke_signals"
    )
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownEmissionChannelError, match="smoke_signals"):
        loader.load_rules("transformers")


def test_enum_value_errors_share_common_base_class() -> None:
    # Callers that don't care which enum is wrong can catch UnknownEnumValueError.
    assert issubclass(UnknownAddedByError, UnknownEnumValueError)
    assert issubclass(UnknownSeverityError, UnknownEnumValueError)
    assert issubclass(UnknownOutcomeError, UnknownEnumValueError)


# ---------------------------------------------------------------------------
# Universal safety gate - an llm_proposed invariant may only ever warn
# ---------------------------------------------------------------------------


def test_llm_proposed_with_non_warn_severity_raises(tmp_path: Path) -> None:
    # The safety gate: an LLM is a recall localiser, never an adjudicator, so an
    # llm_proposed invariant carrying a hard severity is rejected at load.
    bad_corpus = _CORPUS_MINIMAL.replace(
        "    severity: dormant\n", "    severity: error\n    llm_proposed: true\n"
    )
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(LLMProposedSeverityError, match="severity='error'"):
        loader.load_rules("transformers")


def test_llm_proposed_with_warn_severity_loads(tmp_path: Path) -> None:
    corpus = _CORPUS_MINIMAL.replace(
        "    severity: dormant\n", "    severity: warn\n    llm_proposed: true\n"
    )
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    inv = loader.load_rules("transformers").invariants[0]
    assert inv.llm_proposed is True and inv.severity == "warn"


def test_llm_fields_default_false_and_localised_allows_error(tmp_path: Path) -> None:
    # Absent flags default False (older corpora are unaffected); llm_localised is
    # NOT gated on severity (it may carry error once a positive-violation probe
    # confirmed the deterministically re-derived bound).
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    base = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers").invariants[0]
    assert base.llm_proposed is False and base.llm_localised is False

    localised = _CORPUS_MINIMAL.replace(
        "    severity: dormant\n", "    severity: error\n    llm_localised: true\n"
    )
    _write_corpus(tmp_path, "transformers", localised)
    inv = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers").invariants[0]
    assert inv.llm_localised is True and inv.severity == "error"
    assert issubclass(UnknownEmissionChannelError, UnknownEnumValueError)


# ---------------------------------------------------------------------------
# Validated YAML overlay (invariant-miner CI)
# ---------------------------------------------------------------------------


def test_overlay_applied_when_validated_yaml_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llenergymeasure.config.engine_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)

    validated_payload = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": "4.56.0",
        "cases": [
            {
                "id": "transformers_test_rule",
                "outcome": "dormant_announced",
                "emission_channel": "logger_warning",
                "observed_messages": ["library emitted this"],
            }
        ],
        "divergences": [],
    }

    monkeypatch.setattr(
        loader_mod,
        "_try_load_validated_yaml",
        lambda _root, _engine: validated_payload,
    )

    result = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    assert len(result.invariants) == 1
    expected = result.invariants[0].expected_outcome
    assert expected["observed_outcome"] == "dormant_announced"
    assert expected["observed_emission_channel"] == "logger_warning"
    assert expected["observed_messages"] == ["library emitted this"]


def test_no_overlay_when_validated_yaml_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llenergymeasure.config.engine_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    monkeypatch.setattr(loader_mod, "_try_load_validated_yaml", lambda _root, _engine: None)

    result = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    assert len(result.invariants) == 1
    expected = result.invariants[0].expected_outcome
    # The corpus's declared fields are preserved; no observed_* keys appear.
    assert "observed_outcome" not in expected
    assert "observed_emission_channel" not in expected


def test_overlay_skips_invariants_without_matching_validation_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llenergymeasure.config.engine_rules import loader as loader_mod

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    monkeypatch.setattr(
        loader_mod,
        "_try_load_validated_yaml",
        lambda _root, _engine: {"cases": [{"id": "some_other_rule", "outcome": "error"}]},
    )
    result = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    # No matching case -> invariant is returned unchanged.
    assert "observed_outcome" not in result.invariants[0].expected_outcome


def test_try_load_validated_yaml_rejects_non_numeric_schema_version(
    tmp_path: Path,
) -> None:
    # A corrupt commit-back could write a non-numeric schema_version
    # (e.g. "dev"). The loader must return None rather than propagating
    # UnsupportedSchemaVersionError from _major() - the validation CI job
    # resurfaces the issue separately.
    import yaml

    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    (tmp_path / "transformers.validated.yaml").write_text(
        yaml.safe_dump({"schema_version": "dev", "cases": []})
    )

    # Should not raise; should fall back to YAML-only (no observed_* keys).
    result = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers")
    assert "observed_outcome" not in result.invariants[0].expected_outcome
