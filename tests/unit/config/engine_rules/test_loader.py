"""Tests for :class:`EngineRulesLoader` and rules corpus envelope parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llenergymeasure.config.engine_rules import (
    VALID_SEVERITY,
    VALID_SOURCE,
    VALID_VERIFIED,
    EngineRules,
    EngineRulesLoader,
    RuleCorpusError,
    UnknownEnumValueError,
    UnknownSeverityError,
    UnknownSourceError,
    UnknownVerifiedError,
    UnsupportedSchemaVersionError,
)

_CORPUS_MINIMAL = """\
schema_version: "1.0.0"
engine: transformers
engine_version: "4.57.3"
rules:
  - id: transformers_test_rule
    engine: transformers
    severity: dormant
    match:
      fields:
        transformers.sampling.temperature: {present: true}
    normalised_fields:
      - transformers.sampling.temperature
    message_template: "Dormant {declared_value}"
    provenance:
      source: deterministic_miner
      verified: runtime
      engine_version: "4.57.3"
      citation: "transformers/generation/configuration_utils.py:42"
      date: "2026-04-23"
"""


def _write_corpus(root: Path, engine: str, text: str) -> None:
    engine_dir = root / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / "rules.yaml").write_text(text)


def test_load_rules_returns_parsed_corpus(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    corpus = loader.load_rules("transformers")
    assert isinstance(corpus, EngineRules)
    assert corpus.engine == "transformers"
    assert corpus.schema_version == "1.0.0"
    assert corpus.engine_version == "4.57.3"
    assert len(corpus.rules) == 1
    rule = corpus.rules[0]
    assert rule.id == "transformers_test_rule"
    assert rule.severity == "dormant"
    assert rule.normalised_fields == ("transformers.sampling.temperature",)
    assert rule.provenance.source == "deterministic_miner"
    assert rule.provenance.verified == "runtime"
    assert rule.provenance.citation == "transformers/generation/configuration_utils.py:42"
    assert rule.provenance.engine_version == "4.57.3"
    assert rule.provenance.date == "2026-04-23"


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


def test_loader_reads_only_rules_yaml(tmp_path: Path) -> None:
    # The runtime binds to exactly one file. Legacy proposed/validated files
    # sitting next to it are ignored entirely - a regression here would
    # resurrect the two-file topology this loader exists to remove.
    engine_dir = tmp_path / "transformers"
    engine_dir.mkdir(parents=True)
    (engine_dir / "rules.proposed.yaml").write_text("schema_version: '1.0.0'\nrules: []\n")
    (engine_dir / "rules.validated.yaml").write_text("schema_version: '1.0.0'\ncases: []\n")
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(FileNotFoundError, match=r"rules\.yaml"):
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
engine_version: "4.57.3"
rules: []
"""
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnsupportedSchemaVersionError):
        loader.load_rules("transformers")


def test_non_mapping_root_raises(tmp_path: Path) -> None:
    _write_corpus(tmp_path, "transformers", "- just a list")
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(RuleCorpusError, match="must be a YAML mapping"):
        loader.load_rules("transformers")


def test_empty_rules_list_is_valid(tmp_path: Path) -> None:
    corpus = """\
schema_version: "1.0.0"
engine: transformers
engine_version: "4.57.3"
rules: []
"""
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    result = loader.load_rules("transformers")
    assert result.rules == ()


def test_default_corpus_root_resolves_to_engines(tmp_path: Path) -> None:
    # Constructing without corpus_root uses the repo's src/llenergymeasure/engines/.
    loader = EngineRulesLoader()
    assert loader.corpus_root.name == "engines"
    assert loader.corpus_root.parent.name == "llenergymeasure"


# ---------------------------------------------------------------------------
# Closed severity enum
# ---------------------------------------------------------------------------


def test_valid_severity_set_is_closed_two_value() -> None:
    assert frozenset({"error", "dormant"}) == VALID_SEVERITY


@pytest.mark.parametrize("severity", ["error", "dormant"])
def test_both_severities_round_trip(tmp_path: Path, severity: str) -> None:
    corpus = _CORPUS_MINIMAL.replace("severity: dormant", f"severity: {severity}")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    assert loader.load_rules("transformers").rules[0].severity == severity


@pytest.mark.parametrize(
    "severity",
    [
        "warn",  # killed: the study workflow surfaces dormancy organically
        "dormant_silent",  # killed: outcome vocabulary, never a severity
        "llm_proposed",  # killed: provenance must never set severity
        "llm_advisory",
        "kritical",  # plain typo
    ],
)
def test_non_closed_severity_raises_at_load(tmp_path: Path, severity: str) -> None:
    bad_corpus = _CORPUS_MINIMAL.replace("severity: dormant", f"severity: {severity}")
    _write_corpus(tmp_path, "transformers", bad_corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownSeverityError, match=severity):
        loader.load_rules("transformers")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_valid_source_set_matches_design_spec() -> None:
    assert frozenset({"deterministic_miner", "analyst", "manual", "migrated"}) == VALID_SOURCE


def test_valid_verified_set_matches_design_spec() -> None:
    assert frozenset({"construction", "runtime", "human"}) == VALID_VERIFIED


@pytest.mark.parametrize("source", sorted(VALID_SOURCE))
def test_all_sources_round_trip(tmp_path: Path, source: str) -> None:
    corpus = _CORPUS_MINIMAL.replace("source: deterministic_miner", f"source: {source}")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    assert loader.load_rules("transformers").rules[0].provenance.source == source


@pytest.mark.parametrize("verified", sorted(VALID_VERIFIED))
def test_all_verified_values_round_trip(tmp_path: Path, verified: str) -> None:
    corpus = _CORPUS_MINIMAL.replace("verified: runtime", f"verified: {verified}")
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    assert loader.load_rules("transformers").rules[0].provenance.verified == verified


def test_unknown_source_raises(tmp_path: Path) -> None:
    # Old added_by vocabulary must not silently pass as a source.
    bad = _CORPUS_MINIMAL.replace("source: deterministic_miner", "source: static_miner")
    _write_corpus(tmp_path, "transformers", bad)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownSourceError, match="static_miner"):
        loader.load_rules("transformers")


def test_unknown_verified_raises(tmp_path: Path) -> None:
    bad = _CORPUS_MINIMAL.replace("verified: runtime", "verified: vibes")
    _write_corpus(tmp_path, "transformers", bad)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(UnknownVerifiedError, match="vibes"):
        loader.load_rules("transformers")


def test_provenance_must_be_mapping(tmp_path: Path) -> None:
    data = yaml.safe_load(_CORPUS_MINIMAL)
    data["rules"][0]["provenance"] = "static_miner"
    _write_corpus(tmp_path, "transformers", yaml.safe_dump(data))
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(RuleCorpusError, match="provenance"):
        loader.load_rules("transformers")


def test_citation_is_optional(tmp_path: Path) -> None:
    # Not every migrated rule has a source citation; the field is nullable.
    corpus = "".join(
        line for line in _CORPUS_MINIMAL.splitlines(keepends=True) if "citation:" not in line
    )
    _write_corpus(tmp_path, "transformers", corpus)
    loader = EngineRulesLoader(corpus_root=tmp_path)
    assert loader.load_rules("transformers").rules[0].provenance.citation is None


def test_enum_value_errors_share_common_base_class() -> None:
    # Callers that don't care which enum is wrong can catch UnknownEnumValueError,
    # and everything corpus-shaped is a RuleCorpusError (a ValueError).
    assert issubclass(UnknownSeverityError, UnknownEnumValueError)
    assert issubclass(UnknownSourceError, UnknownEnumValueError)
    assert issubclass(UnknownVerifiedError, UnknownEnumValueError)
    assert issubclass(UnknownEnumValueError, RuleCorpusError)
    assert issubclass(UnsupportedSchemaVersionError, RuleCorpusError)
    assert issubclass(RuleCorpusError, ValueError)


# ---------------------------------------------------------------------------
# Rule-shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", ["id", "engine", "severity", "match", "provenance"])
def test_missing_required_rule_field_raises(tmp_path: Path, missing: str) -> None:
    data = yaml.safe_load(_CORPUS_MINIMAL)
    del data["rules"][0][missing]
    _write_corpus(tmp_path, "transformers", yaml.safe_dump(data))
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(RuleCorpusError, match=f"missing field: {missing}"):
        loader.load_rules("transformers")


def test_match_without_fields_raises(tmp_path: Path) -> None:
    data = yaml.safe_load(_CORPUS_MINIMAL)
    data["rules"][0]["match"] = {"engine": "transformers"}
    _write_corpus(tmp_path, "transformers", yaml.safe_dump(data))
    loader = EngineRulesLoader(corpus_root=tmp_path)
    with pytest.raises(RuleCorpusError, match="malformed match"):
        loader.load_rules("transformers")


def test_normalised_fields_string_coerces_to_tuple(tmp_path: Path) -> None:
    data = yaml.safe_load(_CORPUS_MINIMAL)
    data["rules"][0]["normalised_fields"] = "transformers.sampling.temperature"
    _write_corpus(tmp_path, "transformers", yaml.safe_dump(data))
    loader = EngineRulesLoader(corpus_root=tmp_path)
    rule = loader.load_rules("transformers").rules[0]
    assert rule.normalised_fields == ("transformers.sampling.temperature",)


def test_expected_outcome_shim_exposes_normalised_fields(tmp_path: Path) -> None:
    # Deprecated compatibility surface: consumers not yet reworked (B2/B4)
    # read normalised_fields via the old expected_outcome mapping shape.
    _write_corpus(tmp_path, "transformers", _CORPUS_MINIMAL)
    rule = EngineRulesLoader(corpus_root=tmp_path).load_rules("transformers").rules[0]
    assert rule.expected_outcome == {"normalised_fields": ["transformers.sampling.temperature"]}
