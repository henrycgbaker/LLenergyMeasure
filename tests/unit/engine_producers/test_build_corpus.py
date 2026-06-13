"""Unit tests for the canonical corpus merger (``scripts/engine_producers/build_corpus.py``).

The merger orchestrates the per-engine staging extractors, dedups by
fingerprint with cross-validation provenance, and emits the canonical
:file:`src/llenergymeasure/engines/{engine}/rules.proposed.yaml`. These tests exercise each
contract behaviour in isolation against synthetic staging files - no live
extractors, no real library dependencies.

Coverage:

- Fingerprint stability across float-precision jitter and dict-key ordering.
- Cross-validation: shared fingerprint -> single invariant with both sources cited.
- Different fingerprints kept as separate invariants.
- Per-field precedence: AST-miner wins predicates / kwargs; introspection
  wins message_template.
- Stability: byte-identical YAML on re-runs.
- ``--check`` mode: drift surfaces as exit 1 with a diff.
- Empty staging: error gracefully (no canonical write).
- ``cross_validated_by`` parses round-trip via the loader.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

# Make scripts/ importable for direct module access in tests.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llenergymeasure.config.engine_rules import EngineRulesLoader  # noqa: E402
from scripts.engine_producers import build_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures: minimal staging-file invariant shapes
# ---------------------------------------------------------------------------


def _ast_rule(
    *,
    invariant_id: str = "transformers_negative_max_new_tokens",
    severity: str = "error",
    fields: dict[str, Any] | None = None,
    message: str = "max_new_tokens must be > 0 (AST-derived).",
    line: int = 352,
    kwargs_positive: dict[str, Any] | None = None,
    kwargs_negative: dict[str, Any] | None = None,
    references: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": invariant_id,
        "engine": "transformers",
        "library": "transformers",
        "invariant_under_test": "max_new_tokens > 0",
        "severity": severity,
        "native_type": "transformers.GenerationConfig",
        "miner_source": {
            "path": "transformers/generation/configuration_utils.py",
            "method": "validate",
            "line_at_scan": line,
        },
        "match": {
            "engine": "transformers",
            "fields": fields or {"transformers.sampling.max_new_tokens": {"<=": 0}},
        },
        "kwargs_positive": kwargs_positive or {"max_new_tokens": -1},
        "kwargs_negative": kwargs_negative or {"max_new_tokens": 16},
        "expected_outcome": {
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
        },
        "message_template": message,
        "references": references or ["AST-miner reference"],
        "added_by": "static_miner",
        "added_at": "2026-04-25",
    }


def _introspection_rule(
    *,
    invariant_id: str = "transformers_negative_max_new_tokens",
    severity: str = "error",
    fields: dict[str, Any] | None = None,
    message: str = "`max_new_tokens` must be greater than 0, but is -1.",
    observed_messages: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "id": invariant_id,
        "engine": "transformers",
        "library": "transformers",
        "invariant_under_test": "max_new_tokens > 0",
        "severity": severity,
        "native_type": "transformers.GenerationConfig",
        "miner_source": {
            "path": "transformers/generation/configuration_utils.py",
            "method": "validate",
            "line_at_scan": 0,
        },
        "match": {
            "engine": "transformers",
            "fields": fields or {"transformers.sampling.max_new_tokens": {"<=": 0}},
        },
        "kwargs_positive": {"max_new_tokens": -1},
        "kwargs_negative": {"max_new_tokens": 16},
        "expected_outcome": {
            "outcome": "error",
            "emission_channel": "none",
            "normalised_fields": [],
            **({"observed_messages": observed_messages} if observed_messages else {}),
        },
        "message_template": message,
        "references": ["transformers.GenerationConfig - observed via construction-time ValueError"],
        "added_by": "dynamic_miner",
        "added_at": "2026-04-25",
    }


def _envelope(invariants: list[dict[str, Any]], engine_version: str = "4.56.0") -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": engine_version,
        "mined_at": "2026-04-25T00:00:00Z",
        "invariants": invariants,
    }


def _write_staging(staging_dir: Path, basename: str, envelope: dict[str, Any]) -> Path:
    staging_dir.mkdir(parents=True, exist_ok=True)
    path = staging_dir / basename
    path.write_text(yaml.safe_dump(envelope, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_identical_rules_have_identical_fingerprints(self) -> None:
        invariant_a = _ast_rule()
        invariant_b = _ast_rule()
        assert build_corpus.fingerprint_invariant(
            invariant_a
        ) == build_corpus.fingerprint_invariant(invariant_b)

    def test_fingerprint_ignores_dict_key_order(self) -> None:
        # canonical_serialise sorts keys; the merger inherits that.
        invariant_a = _ast_rule(
            fields={
                "transformers.sampling.do_sample": False,
                "transformers.sampling.num_beams": 1,
            },
        )
        invariant_b = _ast_rule(
            fields={
                "transformers.sampling.num_beams": 1,
                "transformers.sampling.do_sample": False,
            },
        )
        assert build_corpus.fingerprint_invariant(
            invariant_a
        ) == build_corpus.fingerprint_invariant(invariant_b)

    def test_fingerprint_stable_across_float_jitter(self) -> None:
        # Floats round to 12 sig digits via canonical_serialise; bit-level
        # jitter in the last place must not change the fingerprint.
        invariant_a = _ast_rule(fields={"transformers.sampling.temperature": 0.7})
        invariant_b = _ast_rule(fields={"transformers.sampling.temperature": 0.7000000000001})
        assert build_corpus.fingerprint_invariant(
            invariant_a
        ) == build_corpus.fingerprint_invariant(invariant_b)

    def test_fingerprint_excludes_id_and_message(self) -> None:
        # Two invariants with the same constraint but different ids / messages
        # still bucket together - the corpus is about the constraint.
        invariant_a = _ast_rule(invariant_id="foo", message="msg A")
        invariant_b = _ast_rule(invariant_id="bar", message="msg B")
        assert build_corpus.fingerprint_invariant(
            invariant_a
        ) == build_corpus.fingerprint_invariant(invariant_b)

    def test_fingerprint_distinguishes_severity(self) -> None:
        invariant_a = _ast_rule(severity="error")
        invariant_b = _ast_rule(severity="warn")
        assert build_corpus.fingerprint_invariant(
            invariant_a
        ) != build_corpus.fingerprint_invariant(invariant_b)

    def test_fingerprint_collapses_int_and_float_thresholds(self) -> None:
        # Static miners read literals from source (`0.0` -> float); dynamic
        # miners emit Python int probes (`0` -> int). Same constraint, two
        # numeric types. Without canonicalisation the cross-validation safety
        # net would split a single library invariant into two corpus invariants.
        rule_int = _ast_rule(fields={"vllm.sampling.repetition_penalty": {"<=": 0}})
        rule_float = _ast_rule(fields={"vllm.sampling.repetition_penalty": {"<=": 0.0}})
        assert build_corpus.fingerprint_invariant(rule_int) == build_corpus.fingerprint_invariant(
            rule_float
        )

    def test_fingerprint_preserves_bool_distinct_from_int(self) -> None:
        # Bool must NOT collapse into int despite ``True == 1``: a invariant that
        # fires on ``do_sample is True`` is semantically different from one
        # that fires on ``num_beams == 1``.
        rule_bool = _ast_rule(fields={"transformers.sampling.do_sample": True})
        rule_int = _ast_rule(fields={"transformers.sampling.do_sample": 1})
        assert build_corpus.fingerprint_invariant(rule_bool) != build_corpus.fingerprint_invariant(
            rule_int
        )


# ---------------------------------------------------------------------------
# Merge: cross-validation
# ---------------------------------------------------------------------------


class TestCrossValidation:
    def test_two_sources_one_fingerprint_merged_to_one_rule(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging, "transformers_dynamic_miner.yaml", _envelope([_introspection_rule()])
        )

        invariants, _envelope_out = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(invariants) == 1
        merged = invariants[0]
        assert merged["added_by"] == "static_miner"
        assert merged["cross_validated_by"] == ["dynamic_miner"]

    def test_introspection_message_overrides_ast_message(self, tmp_path: Path) -> None:
        # Per the precedence table, introspection's message_template wins
        # because it's the real library text.
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(message="AST-derived placeholder.")]),
        )
        _write_staging(
            staging,
            "transformers_dynamic_miner.yaml",
            _envelope([_introspection_rule(message="`max_new_tokens` must be > 0, but is -1.")]),
        )

        invariants, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(invariants) == 1
        assert invariants[0]["message_template"] == "`max_new_tokens` must be > 0, but is -1."
        # Conflict surfaced for review (since the messages differed).
        assert "conflict_note" in invariants[0]
        assert "message_template" in invariants[0]["conflict_note"]

    def test_ast_kwargs_positive_overrides_introspection(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope(
                [
                    _ast_rule(
                        kwargs_positive={"max_new_tokens": -42},
                        kwargs_negative={"max_new_tokens": 99},
                    )
                ]
            ),
        )
        intro = _introspection_rule()
        intro["kwargs_positive"] = {"max_new_tokens": -1}
        intro["kwargs_negative"] = {"max_new_tokens": 16}
        _write_staging(staging, "transformers_dynamic_miner.yaml", _envelope([intro]))

        invariants, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(invariants) == 1
        assert invariants[0]["kwargs_positive"] == {"max_new_tokens": -42}
        assert invariants[0]["kwargs_negative"] == {"max_new_tokens": 99}

    def test_observed_messages_carry_over_from_introspection(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging,
            "transformers_dynamic_miner.yaml",
            _envelope(
                [
                    _introspection_rule(
                        observed_messages=["`max_new_tokens` must be greater than 0, but is -1."]
                    )
                ]
            ),
        )

        invariants, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(invariants) == 1
        observed = invariants[0]["expected_outcome"].get("observed_messages")
        assert observed == ["`max_new_tokens` must be greater than 0, but is -1."]

    def test_references_unioned_across_sources(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(references=["AST ref 1"])]),
        )
        intro = _introspection_rule()
        intro["references"] = ["Introspection ref 2"]
        _write_staging(staging, "transformers_dynamic_miner.yaml", _envelope([intro]))

        invariants, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        refs = invariants[0]["references"]
        assert "AST ref 1" in refs
        assert "Introspection ref 2" in refs


# ---------------------------------------------------------------------------
# Merge: distinct fingerprints stay separate
# ---------------------------------------------------------------------------


class TestDistinctFingerprints:
    def test_different_match_fields_kept_as_two_rules(self, tmp_path: Path) -> None:
        # Same id, different match.fields -> two separate invariants. Vendor CI
        # will prove which fires correctly on the live library.
        staging = tmp_path / "transformers" / "_staging"
        ast = _ast_rule(
            fields={"transformers.sampling.max_new_tokens": {"<=": 0}},
        )
        intro = _introspection_rule(
            fields={"transformers.sampling.max_new_tokens": {"<": 0}},
        )
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([ast]))
        _write_staging(staging, "transformers_dynamic_miner.yaml", _envelope([intro]))

        invariants, _ = build_corpus.merge_staging(
            [build_corpus._load_staging(p) for p in sorted(staging.glob("transformers_*.yaml"))]
        )
        assert len(invariants) == 2
        # Both keep their original added_by; neither has cross_validated_by.
        sources = {r["added_by"] for r in invariants}
        assert sources == {"static_miner", "dynamic_miner"}
        for r in invariants:
            assert "cross_validated_by" not in r


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------


class TestStability:
    def test_repeated_runs_produce_identical_yaml(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope(
                [
                    _ast_rule(
                        invariant_id="invariant_b", fields={"transformers.sampling.top_k": 51}
                    ),
                    _ast_rule(invariant_id="invariant_a"),
                ]
            ),
        )
        _write_staging(
            staging, "transformers_dynamic_miner.yaml", _envelope([_introspection_rule()])
        )

        first = build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)
        second = build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)
        assert first == second

    def test_frozen_at_env_overrides_mined_at(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Contract relied on by .github/workflows/engine-invariants.yml: when
        # LLENERGY_MINER_FROZEN_AT is set, the merged envelope's ``mined_at``
        # MUST equal that exact value, regardless of staging timestamps. This
        # is the anchor that keeps re-runs on unchanged source byte-identical
        # - without it the workflow's commit-back synchronize-loops.
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule()], engine_version="4.56.0"),
        )
        _write_staging(
            staging, "transformers_dynamic_miner.yaml", _envelope([_introspection_rule()])
        )

        frozen = "2026-04-23T12:34:56+00:00"
        monkeypatch.setenv("LLENERGY_MINER_FROZEN_AT", frozen)
        text = build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)
        doc = yaml.safe_load(text)
        assert doc["mined_at"] == frozen

        # And: two runs at the same anchor produce byte-identical YAML even when
        # the staging envelopes' own mined_at fields differ (the workflow's
        # anchor step computes once per run).
        second = build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)
        assert text == second

    def test_rules_sorted_alphabetically_by_id(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope(
                [
                    _ast_rule(invariant_id="zzz_late", fields={"f1": 1}),
                    _ast_rule(invariant_id="aaa_early", fields={"f2": 2}),
                ]
            ),
        )
        text = build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)
        doc = yaml.safe_load(text)
        ids = [r["id"] for r in doc["invariants"]]
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


class TestCheckMode:
    def test_check_passes_when_corpus_matches_staging(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging, "transformers_dynamic_miner.yaml", _envelope([_introspection_rule()])
        )

        # Build then immediately check -> should pass.
        build_corpus.write_corpus("transformers", tmp_path, skip_validation=True)
        code, _ = build_corpus.check_drift("transformers", tmp_path, skip_validation=True)
        assert code == 0

    def test_check_fails_with_diff_on_drift(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))

        build_corpus.write_corpus("transformers", tmp_path, skip_validation=True)

        # Mutate staging to introduce drift.
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(message="Different message - drift!")]),
        )
        code, diff = build_corpus.check_drift("transformers", tmp_path, skip_validation=True)
        assert code == 1
        assert "Different message" in diff

    def test_check_returns_2_when_canonical_corpus_missing(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))
        # No write_corpus call - canonical YAML missing.
        code, msg = build_corpus.check_drift("transformers", tmp_path, skip_validation=True)
        assert code == 2
        assert "not found" in msg


# ---------------------------------------------------------------------------
# Empty staging
# ---------------------------------------------------------------------------


class TestEmptyStaging:
    def test_no_staging_files_raises_filenotfounderror(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No staging files"):
            build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)

    def test_no_staging_does_not_touch_existing_corpus(self, tmp_path: Path) -> None:
        # A pre-existing corpus must NOT be wiped if the merger fails to
        # find staging - the canonical file stays untouched.
        canonical = tmp_path / "transformers" / "rules.proposed.yaml"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("schema_version: 1.0.0\nengine: transformers\nrules: []\n")

        with pytest.raises(FileNotFoundError):
            build_corpus.build_corpus_text("transformers", tmp_path, skip_validation=True)

        assert canonical.read_text() == ("schema_version: 1.0.0\nengine: transformers\nrules: []\n")


# ---------------------------------------------------------------------------
# canonical_out override (SSOT redirection): the canonical corpus is read +
# written at an explicit path while staging stays under corpus_root.
# ---------------------------------------------------------------------------


class TestCanonicalOutOverride:
    def test_write_targets_canonical_out_not_corpus_root(self, tmp_path: Path) -> None:
        """``canonical_out`` redirects the written corpus; staging stays under
        ``corpus_root``. This is the SSOT redirection the CI cell uses."""
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))

        ssot = tmp_path / "ssot_outputs" / "rules.proposed.yaml"
        build_corpus.write_corpus(
            "transformers", tmp_path, skip_validation=True, canonical_out=ssot
        )

        # Written to the override, NOT the corpus_root default.
        assert ssot.exists()
        assert not (tmp_path / "transformers" / "rules.proposed.yaml").exists()
        # Staging still lives under corpus_root.
        assert (staging / "transformers_static_miner.yaml").exists()

    def test_check_compares_against_canonical_out(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))
        ssot = tmp_path / "ssot_outputs" / "rules.proposed.yaml"

        build_corpus.write_corpus(
            "transformers", tmp_path, skip_validation=True, canonical_out=ssot
        )
        # check against the same override path passes; against the (absent)
        # default path it reports the corpus missing.
        code_ok, _ = build_corpus.check_drift(
            "transformers", tmp_path, skip_validation=True, canonical_out=ssot
        )
        assert code_ok == 0
        code_missing, msg = build_corpus.check_drift("transformers", tmp_path, skip_validation=True)
        assert code_missing == 2
        assert "not found" in msg

    def test_prior_carry_reads_from_canonical_out(self, tmp_path: Path) -> None:
        """added_at preservation reads the prior from ``canonical_out``, not the
        corpus_root default - so a re-mine to the SSOT keeps discovery dates."""
        staging = tmp_path / "transformers" / "_staging"
        ssot = tmp_path / "ssot_outputs" / "rules.proposed.yaml"

        first = _ast_rule()
        first["added_at"] = "2026-04-01"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([first]))
        build_corpus.write_corpus(
            "transformers", tmp_path, skip_validation=True, canonical_out=ssot
        )

        second = _ast_rule()
        second["added_at"] = "2026-05-09"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([second]))
        build_corpus.write_corpus(
            "transformers", tmp_path, skip_validation=True, canonical_out=ssot
        )

        rebuilt = yaml.safe_load(ssot.read_text())
        assert rebuilt["invariants"][0]["added_at"] == "2026-04-01"


# ---------------------------------------------------------------------------
# Loader round-trip - cross_validated_by parses correctly
# ---------------------------------------------------------------------------


class TestLoaderRoundTrip:
    def test_merger_output_loads_via_engine_rules_loader(self, tmp_path: Path) -> None:
        staging = tmp_path / "transformers" / "_staging"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))
        _write_staging(
            staging, "transformers_dynamic_miner.yaml", _envelope([_introspection_rule()])
        )

        build_corpus.write_corpus("transformers", tmp_path, skip_validation=True)

        loader = EngineRulesLoader(corpus_root=tmp_path)
        parsed = loader.load_rules("transformers")
        assert len(parsed.invariants) == 1
        invariant = parsed.invariants[0]
        assert invariant.added_by == "static_miner"
        assert invariant.cross_validated_by == ("dynamic_miner",)

    def test_loader_rejects_unknown_cross_validated_by_value(self, tmp_path: Path) -> None:
        # Bypass the merger's single-source normalisation by writing a
        # corpus YAML directly with a bad cross_validated_by entry - the
        # loader must reject it, since the closed-enum guard is the
        # whole point of validating cross-validation provenance.
        from llenergymeasure.config.engine_rules import UnknownAddedByError

        invariant = _ast_rule()
        invariant["cross_validated_by"] = ["NOT_A_REAL_PROVENANCE"]
        canonical = tmp_path / "transformers" / "rules.proposed.yaml"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text(yaml.safe_dump(_envelope([invariant]), sort_keys=False))

        loader = EngineRulesLoader(corpus_root=tmp_path)
        with pytest.raises(UnknownAddedByError):
            loader.load_rules("transformers")


# ---------------------------------------------------------------------------
# added_at preservation across re-mines
# ---------------------------------------------------------------------------


class TestAddedAtPreservation:
    def test_load_prior_added_at_map_missing_corpus(self, tmp_path: Path) -> None:
        out = build_corpus._load_prior_added_at_map(tmp_path / "missing.yaml")
        assert out == {}

    def test_load_prior_added_at_map_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("not: valid: yaml: [")
        out = build_corpus._load_prior_added_at_map(path)
        assert out == {}

    def test_load_prior_added_at_map_extracts_fingerprint_to_date(self, tmp_path: Path) -> None:
        invariant = _ast_rule()
        invariant["added_at"] = "2026-04-01"
        path = tmp_path / "prior.yaml"
        path.write_text(yaml.safe_dump(_envelope([invariant]), sort_keys=False))
        out = build_corpus._load_prior_added_at_map(path)
        fp = build_corpus.fingerprint_invariant(invariant)
        assert out == {fp: "2026-04-01"}

    def test_preserve_added_at_restores_matching_fingerprint(self) -> None:
        invariant = _ast_rule()
        invariant["added_at"] = "2026-04-30"
        prior = {build_corpus.fingerprint_invariant(invariant): "2026-04-01"}
        build_corpus._preserve_added_at([invariant], prior)
        assert invariant["added_at"] == "2026-04-01"

    def test_preserve_added_at_keeps_today_when_no_match(self) -> None:
        invariant = _ast_rule()
        invariant["added_at"] = "2026-04-30"
        # Different fingerprint in prior - no match expected.
        other = _ast_rule(fields={"transformers.sampling.top_p": {"<": 0.0}})
        prior = {build_corpus.fingerprint_invariant(other): "2026-04-01"}
        build_corpus._preserve_added_at([invariant], prior)
        assert invariant["added_at"] == "2026-04-30"

    def test_preserve_added_at_no_op_when_prior_empty(self) -> None:
        invariant = _ast_rule()
        invariant["added_at"] = "2026-04-30"
        build_corpus._preserve_added_at([invariant], {})
        assert invariant["added_at"] == "2026-04-30"

    def test_e2e_added_at_preserved_across_remine(self, tmp_path: Path) -> None:
        """Re-running the merger after a prior canonical exists keeps each
        invariant's original ``added_at`` instead of stamping today's date.

        Stops Renovate-driven rebuilds from producing noise diffs that
        flip ``added_at`` on every invariant even when content is unchanged.
        """
        staging = tmp_path / "transformers" / "_staging"

        # First run: produce canonical with added_at "2026-04-01".
        first_rule = _ast_rule()
        first_rule["added_at"] = "2026-04-01"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([first_rule]))
        build_corpus.write_corpus("transformers", tmp_path, skip_validation=True)

        prior_path = tmp_path / "transformers" / "rules.proposed.yaml"
        prior = yaml.safe_load(prior_path.read_text())
        assert prior["invariants"][0]["added_at"] == "2026-04-01"

        # Second run: re-stage with same fingerprint but today's date.
        second_rule = _ast_rule()
        second_rule["added_at"] = "2026-05-03"
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([second_rule]))
        build_corpus.write_corpus("transformers", tmp_path, skip_validation=True)

        rebuilt = yaml.safe_load(prior_path.read_text())
        assert rebuilt["invariants"][0]["added_at"] == "2026-04-01"


# ---------------------------------------------------------------------------
# Vendor-validation gate
# ---------------------------------------------------------------------------


def _stub_validate_engine(
    *, divergent_rule_ids: tuple[str, ...] = (), divergence_field: str = "outcome"
):
    """Return a callable mirroring :func:`scripts.validate_rules.validate_engine`.

    The stub doesn't run the real library - it returns synthetic divergences
    keyed off invariant ids. Tests monkeypatch ``scripts.validate_rules.validate_engine``
    onto this stub so the merger's validation wiring runs without needing the
    transformers package available in the test environment.
    """
    from scripts._rules_validation_common import Divergence

    def _stub(*, engine: str, corpus_path: Path, out_path: Path, **kwargs: Any):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n")
        divergences = [
            Divergence(
                invariant_id=rid,
                field=divergence_field,
                expected="expected_value",
                observed="observed_value",
            )
            for rid in divergent_rule_ids
        ]
        envelope = {
            "schema_version": "1.0.0",
            "engine": engine,
            "engine_version": "stub",
            "cases": [],
            "divergences": [d.as_dict() for d in divergences],
        }
        return envelope, divergences

    return _stub


class TestVendorValidationGate:
    """Integration tests for the validation step in the merger."""

    def test_validated_kept_invariants_land_in_canonical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invariants with no divergence are kept in the canonical YAML."""
        import scripts.validate_rules as vr

        monkeypatch.setattr(vr, "validate_engine", _stub_validate_engine())

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(invariant_id="rule_kept")]),
        )

        result = build_corpus.write_corpus("transformers", tmp_path)
        assert result.invariants_in_canonical == 1
        assert result.invariants_quarantined == 0
        assert result.quarantined_ids == ()

        canonical = (tmp_path / "transformers" / "rules.proposed.yaml").read_text()
        assert "rule_kept" in canonical

    def test_divergent_invariant_is_quarantined(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An invariant whose validation outcome diverges is dropped from canonical."""
        import scripts.validate_rules as vr

        monkeypatch.setattr(
            vr,
            "validate_engine",
            _stub_validate_engine(divergent_rule_ids=("rule_bad",)),
        )

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope(
                [
                    _ast_rule(invariant_id="rule_bad", fields={"f1": 1}),
                    _ast_rule(invariant_id="rule_kept", fields={"f2": 2}),
                ]
            ),
        )

        result = build_corpus.write_corpus("transformers", tmp_path)
        assert result.invariants_in_canonical == 1
        assert result.invariants_quarantined == 1
        assert "rule_bad" in result.quarantined_ids

        canonical = (tmp_path / "transformers" / "rules.proposed.yaml").read_text()
        assert "rule_kept" in canonical
        assert "rule_bad" not in canonical

    def test_skip_validation_keeps_all_candidates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--skip-validation`` short-circuits the gate; validate_engine never runs."""
        import scripts.validate_rules as vr

        # If validate_engine were called, this stub would mark ALL invariants as
        # divergent - but skip_validation should prevent the call entirely.
        monkeypatch.setattr(
            vr,
            "validate_engine",
            _stub_validate_engine(divergent_rule_ids=("invariant_a", "invariant_b")),
        )

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope(
                [
                    _ast_rule(invariant_id="invariant_a", fields={"f1": 1}),
                    _ast_rule(invariant_id="invariant_b", fields={"f2": 2}),
                ]
            ),
        )

        result = build_corpus.write_corpus("transformers", tmp_path, skip_validation=True)
        assert result.validation_skipped is True
        assert result.invariants_in_canonical == 2
        assert result.invariants_quarantined == 0

        canonical = (tmp_path / "transformers" / "rules.proposed.yaml").read_text()
        assert "invariant_a" in canonical
        assert "invariant_b" in canonical
        # No quarantine file when validation is skipped.
        assert not (
            tmp_path / "transformers" / "_staging" / "_failed_validation_transformers.yaml"
        ).exists()

    def test_quarantine_yaml_has_documented_schema(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The quarantine file matches the documented {schema_version, engine, engine_version, generated_at, quarantined_rules} shape."""
        import scripts.validate_rules as vr

        monkeypatch.setattr(
            vr,
            "validate_engine",
            _stub_validate_engine(
                divergent_rule_ids=("rule_bad",),
                divergence_field="outcome",
            ),
        )

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(invariant_id="rule_bad")]),
        )

        build_corpus.write_corpus("transformers", tmp_path)

        quarantine_path = (
            tmp_path / "transformers" / "_staging" / "_failed_validation_transformers.yaml"
        )
        assert quarantine_path.exists()
        doc = yaml.safe_load(quarantine_path.read_text())
        assert set(doc) >= {
            "schema_version",
            "engine",
            "engine_version",
            "generated_at",
            "quarantined_rules",
        }
        assert doc["engine"] == "transformers"
        assert isinstance(doc["quarantined_rules"], list)
        assert len(doc["quarantined_rules"]) == 1

        entry = doc["quarantined_rules"][0]
        assert entry["invariant"]["id"] == "rule_bad"
        assert entry["divergences"][0]["field"] == "outcome"
        assert entry["divergences"][0]["expected"] == "expected_value"
        assert entry["divergences"][0]["observed"] == "observed_value"

    def test_quarantine_yaml_removed_when_no_divergences(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing quarantine file is cleared when the new run has no divergences."""
        import scripts.validate_rules as vr

        # Plant a stale quarantine file from an earlier (hypothetical) run.
        staging = tmp_path / "transformers" / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
        stale = staging / "_failed_validation_transformers.yaml"
        stale.write_text("schema_version: 1.0.0\nengine: transformers\nquarantined_rules: []\n")

        monkeypatch.setattr(vr, "validate_engine", _stub_validate_engine())
        _write_staging(staging, "transformers_static_miner.yaml", _envelope([_ast_rule()]))

        build_corpus.write_corpus("transformers", tmp_path)
        assert not stale.exists(), (
            "stale quarantine file must be removed when the latest run has no divergences"
        )

    def test_check_mode_runs_validation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``--check`` re-runs validation so drift detection compares apples-to-apples."""
        import scripts.validate_rules as vr

        monkeypatch.setattr(
            vr,
            "validate_engine",
            _stub_validate_engine(divergent_rule_ids=("rule_bad",)),
        )

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope(
                [
                    _ast_rule(invariant_id="rule_bad", fields={"f1": 1}),
                    _ast_rule(invariant_id="rule_kept", fields={"f2": 2}),
                ]
            ),
        )

        # Build with validation: rule_bad gets quarantined and only rule_kept
        # lands in canonical.
        build_corpus.write_corpus("transformers", tmp_path)
        canonical_path = tmp_path / "transformers" / "rules.proposed.yaml"
        assert "rule_bad" not in canonical_path.read_text()

        # --check should now agree (re-runs validation, observes the same
        # quarantine, produces matching canonical YAML).
        code, _diff = build_corpus.check_drift("transformers", tmp_path)
        assert code == 0

    def test_merged_candidates_yaml_excluded_from_self_globbing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The merger's own previous output must not feed back into itself.

        Regression guard: ``discover_staging_files`` previously matched
        ``transformers_*.yaml`` indiscriminately, including the merger's own
        ``transformers_merged_candidates.yaml`` from the prior run. That
        caused stale kwargs to dominate the re-merge under fingerprint
        dedup and silently masked extractor-side fixes.
        """
        import scripts.validate_rules as vr

        monkeypatch.setattr(vr, "validate_engine", _stub_validate_engine())

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(invariant_id="rule_real")]),
        )

        build_corpus.write_corpus("transformers", tmp_path)
        # The merger writes its candidates file; the next run must skip it.
        merged_candidates = staging / "transformers_merged_candidates.yaml"
        assert merged_candidates.exists()

        discovered = build_corpus.discover_staging_files("transformers", tmp_path)
        assert merged_candidates not in discovered
        assert (staging / "transformers_static_miner.yaml") in discovered


# ---------------------------------------------------------------------------
# Manual-seed carry across re-mines
# ---------------------------------------------------------------------------


def _manual_seed_rule(
    *,
    invariant_id: str = "transformers_bnb_load_in_4bit_xor_load_in_8bit",
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A hand-shaped manual_seed invariant the miners cannot reach."""
    rule = _ast_rule(invariant_id=invariant_id, fields=fields or {"f_seed": True})
    rule["added_by"] = "manual_seed"
    return rule


def _reclassified_rule(
    *,
    invariant_id: str = "transformers_dormant_decayed",
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A dormant rule reclassified to silent dormancy after its announcement decayed.

    Shaped exactly as the decay-alarm reconciliation persists it: severity
    ``dormant``, outcome ``dormant_silent``, provenance
    ``reclassified_decayed_announcement``. The miners never re-emit it, so it
    must be carried forward across re-mines or the equivalence rots out.
    """
    rule = _ast_rule(
        invariant_id=invariant_id, severity="dormant", fields=fields or {"f_dec": True}
    )
    rule["added_by"] = "reclassified_decayed_announcement"
    rule["expected_outcome"] = {
        "outcome": "dormant_silent",
        "emission_channel": "none",
        "normalised_fields": [],
    }
    return rule


def _diagnose_rule(
    *,
    invariant_id: str = "transformers_llm_gap",
    severity: str = "error",
    fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A gate-confirmed Stage-1 diagnose proposal (P2b), shaped as `llm_diagnose`.

    Mirrors what `llenergymeasure.api.diagnose.render_proposed_yaml` emits in its
    `invariants` list: a normal proposed-corpus entry with provenance
    ``llm_diagnose``. Only gate-confirmed entries ever reach the fold.
    """
    rule = _ast_rule(invariant_id=invariant_id, severity=severity, fields=fields or {"f_gap": True})
    rule["added_by"] = "llm_diagnose"
    return rule


class TestSeededCarry:
    """The merger carries miner-unreachable invariants forward from the prior
    committed corpus so a re-mine does not clobber them: hand-shaped
    ``manual_seed`` rules and ``reclassified_decayed_announcement`` rules the
    decay-alarm persisted (the silent-knowledge-loss bug, CR1)."""

    def test_load_carried_seeded_filters_to_carried_provenances(self, tmp_path: Path) -> None:
        path = tmp_path / "prior.yaml"
        path.write_text(
            yaml.safe_dump(
                _envelope(
                    [
                        _ast_rule(invariant_id="mined"),
                        _manual_seed_rule(),
                        _reclassified_rule(),
                    ]
                ),
                sort_keys=False,
            )
        )
        carried = build_corpus._load_carried_seeded(path)
        assert {c["id"] for c in carried} == {
            "transformers_bnb_load_in_4bit_xor_load_in_8bit",
            "transformers_dormant_decayed",
        }

    def test_load_carried_seeded_carries_reclassified_dormant(self, tmp_path: Path) -> None:
        # CR1: a reclassified_decayed_announcement entry must be picked up by the
        # carry so it survives the next re-mine.
        path = tmp_path / "prior.yaml"
        path.write_text(yaml.safe_dump(_envelope([_reclassified_rule()]), sort_keys=False))
        carried = build_corpus._load_carried_seeded(path)
        assert [c["id"] for c in carried] == ["transformers_dormant_decayed"]
        assert carried[0]["added_by"] == "reclassified_decayed_announcement"

    def test_load_carried_seeded_missing_corpus(self, tmp_path: Path) -> None:
        assert build_corpus._load_carried_seeded(tmp_path / "absent.yaml") == []

    def test_load_carried_seeded_invalid_yaml(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.yaml"
        path.write_text("not: valid: yaml: [")
        assert build_corpus._load_carried_seeded(path) == []

    def test_carried_seed_preserved_when_registry_omits_it(self) -> None:
        registry = [_ast_rule(invariant_id="mined", fields={"f1": 1})]
        carried = [_manual_seed_rule()]
        merged = build_corpus._merge_carried_seeded(registry, carried)
        ids = {r["id"] for r in merged}
        assert "transformers_bnb_load_in_4bit_xor_load_in_8bit" in ids
        assert "mined" in ids

    def test_registry_wins_on_id_collision(self) -> None:
        # The miners learned to extract the seed's id: the machine-extracted
        # version must win and the carried copy must NOT be appended.
        registry = [_ast_rule(invariant_id="collide", message="machine-extracted")]
        carried = [_manual_seed_rule(invariant_id="collide")]
        merged = build_corpus._merge_carried_seeded(registry, carried)
        collide = [r for r in merged if r["id"] == "collide"]
        assert len(collide) == 1
        assert collide[0]["added_by"] == "static_miner"
        assert collide[0]["message_template"] == "machine-extracted"

    def test_no_carried_seeds_is_identity(self) -> None:
        registry = [_ast_rule(invariant_id="mined")]
        assert build_corpus._merge_carried_seeded(registry, []) is registry

    def test_carried_seed_passes_through_validation_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A carried seed that validates is kept; a carried seed the gate rejects
        is dropped - the gate adjudicates carried entries exactly as registry ones."""
        import scripts.validate_rules as vr

        # Stub the gate to reject one carried seed and accept the other.
        monkeypatch.setattr(
            vr,
            "validate_engine",
            _stub_validate_engine(divergent_rule_ids=("seed_stale",)),
        )

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(invariant_id="mined", fields={"f_mined": 1})]),
        )
        # Seed the prior committed corpus with two manual seeds.
        prior = tmp_path / "transformers" / "rules.proposed.yaml"
        prior.write_text(
            yaml.safe_dump(
                _envelope(
                    [
                        _manual_seed_rule(invariant_id="seed_good", fields={"f_good": True}),
                        _manual_seed_rule(invariant_id="seed_stale", fields={"f_stale": True}),
                    ]
                ),
                sort_keys=False,
            )
        )

        result = build_corpus.write_corpus("transformers", tmp_path)
        canonical = prior.read_text()
        # Good seed carried + gate-confirmed; stale seed dropped by the gate.
        assert "seed_good" in canonical
        assert "seed_stale" not in canonical
        assert "mined" in canonical
        assert "seed_stale" in result.quarantined_ids

    def test_reclassified_dormant_survives_a_re_mine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CR1: a ``reclassified_decayed_announcement`` entry persisted into a
        pin's rules.proposed.yaml is carried forward by a subsequent re-mine.

        Without the carry the miners never re-emit it and it silently rots out -
        the cardinal silent-knowledge-loss bug. The gate is stubbed to confirm
        it (the entry is dormant_silent, so production lands it in proposed)."""
        import scripts.validate_rules as vr

        monkeypatch.setattr(vr, "validate_engine", _stub_validate_engine())

        staging = tmp_path / "transformers" / "_staging"
        _write_staging(
            staging,
            "transformers_static_miner.yaml",
            _envelope([_ast_rule(invariant_id="mined", fields={"f_mined": 1})]),
        )
        # The prior pin's committed corpus holds a reclassified-dormant rule the
        # miners never re-emit (it is not in the fresh staging above).
        prior = tmp_path / "transformers" / "rules.proposed.yaml"
        prior.write_text(yaml.safe_dump(_envelope([_reclassified_rule()]), sort_keys=False))

        build_corpus.write_corpus("transformers", tmp_path)

        rebuilt = yaml.safe_load(prior.read_text())
        ids = {inv["id"]: inv for inv in rebuilt["invariants"]}
        assert "transformers_dormant_decayed" in ids, "reclassified rule rotted out across re-mine"
        assert (
            ids["transformers_dormant_decayed"]["added_by"] == "reclassified_decayed_announcement"
        )
        assert "mined" in ids


class TestFoldReclassifiedIntoProposed:
    """CR1 persist: the decay alarm's reclassified-dormant payloads are folded
    into the new pin's rules.proposed.yaml so they do not vanish after being
    computed."""

    def test_folds_reclassified_payload_by_id(self, tmp_path: Path) -> None:
        proposed = tmp_path / "rules.proposed.yaml"
        proposed.write_text(
            yaml.safe_dump(_envelope([_ast_rule(invariant_id="mined")]), sort_keys=False)
        )
        report = {
            "reconciliation": {
                "reclassified": [
                    {"id": "transformers_dormant_decayed", "invariant": _reclassified_rule()}
                ]
            }
        }
        folded = build_corpus.fold_reclassified_into_proposed(report, proposed)
        assert folded == 1
        doc = yaml.safe_load(proposed.read_text())
        ids = {inv["id"]: inv for inv in doc["invariants"]}
        assert "transformers_dormant_decayed" in ids
        assert (
            ids["transformers_dormant_decayed"]["added_by"] == "reclassified_decayed_announcement"
        )
        assert "mined" in ids  # untouched
        # Envelope preserved.
        assert doc["engine"] == "transformers"
        assert doc["schema_version"] == "1.0.0"

    def test_existing_id_wins_no_clobber(self, tmp_path: Path) -> None:
        # The new mine already re-emitted the id: the machine-extracted entry is
        # authoritative and the reclassified payload must NOT overwrite it.
        proposed = tmp_path / "rules.proposed.yaml"
        proposed.write_text(
            yaml.safe_dump(
                _envelope(
                    [_ast_rule(invariant_id="transformers_dormant_decayed", message="from-mine")]
                ),
                sort_keys=False,
            )
        )
        report = {
            "reconciliation": {
                "reclassified": [
                    {"id": "transformers_dormant_decayed", "invariant": _reclassified_rule()}
                ]
            }
        }
        folded = build_corpus.fold_reclassified_into_proposed(report, proposed)
        assert folded == 0
        doc = yaml.safe_load(proposed.read_text())
        rows = [inv for inv in doc["invariants"] if inv["id"] == "transformers_dormant_decayed"]
        assert len(rows) == 1
        assert rows[0]["message_template"] == "from-mine"
        assert rows[0]["added_by"] == "static_miner"

    def test_no_reconciliation_is_noop(self, tmp_path: Path) -> None:
        proposed = tmp_path / "rules.proposed.yaml"
        original = yaml.safe_dump(_envelope([_ast_rule(invariant_id="mined")]), sort_keys=False)
        proposed.write_text(original)
        assert build_corpus.fold_reclassified_into_proposed({}, proposed) == 0
        assert proposed.read_text() == original


class TestFoldDiagnoseIntoProposed:
    """P2b persist: gate-confirmed Stage-1 diagnose proposals are folded into the
    new pin's rules.proposed.yaml (the SSOT) so they ride the bump-PR data diff
    for review - nothing auto-merges, the maintainer reviews + merges."""

    def test_folds_diagnose_proposal_by_id(self, tmp_path: Path) -> None:
        proposed = tmp_path / "rules.proposed.yaml"
        proposed.write_text(
            yaml.safe_dump(_envelope([_ast_rule(invariant_id="mined")]), sort_keys=False)
        )
        # The fragment shape `llem diagnose-bump --out` writes.
        fragment = {
            "schema_version": "1.0.0",
            "engine": "transformers",
            "engine_version": "5.7.0",
            "invariants": [_diagnose_rule(invariant_id="transformers_llm_gap")],
        }
        folded = build_corpus.fold_diagnose_into_proposed(fragment, proposed)
        assert folded == 1
        doc = yaml.safe_load(proposed.read_text())
        ids = {inv["id"]: inv for inv in doc["invariants"]}
        assert "transformers_llm_gap" in ids
        assert ids["transformers_llm_gap"]["added_by"] == "llm_diagnose"
        assert "mined" in ids  # untouched
        # Envelope preserved (the prior pin's, not the fragment's).
        assert doc["engine"] == "transformers"
        assert doc["schema_version"] == "1.0.0"

    def test_existing_id_wins_no_clobber(self, tmp_path: Path) -> None:
        # A deterministic miner already re-emitted the id: it is authoritative
        # and the diagnose proposal must NOT overwrite it.
        proposed = tmp_path / "rules.proposed.yaml"
        proposed.write_text(
            yaml.safe_dump(
                _envelope([_ast_rule(invariant_id="transformers_llm_gap", message="from-mine")]),
                sort_keys=False,
            )
        )
        fragment = {
            "schema_version": "1.0.0",
            "engine": "transformers",
            "engine_version": "5.7.0",
            "invariants": [_diagnose_rule(invariant_id="transformers_llm_gap")],
        }
        folded = build_corpus.fold_diagnose_into_proposed(fragment, proposed)
        assert folded == 0
        doc = yaml.safe_load(proposed.read_text())
        rows = [inv for inv in doc["invariants"] if inv["id"] == "transformers_llm_gap"]
        assert len(rows) == 1
        assert rows[0]["message_template"] == "from-mine"
        assert rows[0]["added_by"] == "static_miner"

    def test_no_invariants_is_noop(self, tmp_path: Path) -> None:
        # A diagnose run with zero gate-confirmed entries writes no fragment in
        # production; a fragment with an empty/absent invariants list is a no-op.
        proposed = tmp_path / "rules.proposed.yaml"
        original = yaml.safe_dump(_envelope([_ast_rule(invariant_id="mined")]), sort_keys=False)
        proposed.write_text(original)
        assert build_corpus.fold_diagnose_into_proposed({"invariants": []}, proposed) == 0
        assert build_corpus.fold_diagnose_into_proposed({}, proposed) == 0
        assert proposed.read_text() == original

    def test_folds_multiple_id_sorted(self, tmp_path: Path) -> None:
        proposed = tmp_path / "rules.proposed.yaml"
        proposed.write_text(
            yaml.safe_dump(_envelope([_ast_rule(invariant_id="m_mined")]), sort_keys=False)
        )
        fragment = {
            "invariants": [
                _diagnose_rule(invariant_id="z_gap", fields={"f_z": 1}),
                _diagnose_rule(invariant_id="a_gap", fields={"f_a": 1}),
            ],
        }
        folded = build_corpus.fold_diagnose_into_proposed(fragment, proposed)
        assert folded == 2
        doc = yaml.safe_load(proposed.read_text())
        assert [inv["id"] for inv in doc["invariants"]] == ["a_gap", "m_mined", "z_gap"]
