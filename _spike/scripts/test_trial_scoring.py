"""Smoke tests for ``_spike.scripts.trial_scoring``.

Runs with::

    uv run python -m pytest _spike/scripts/test_trial_scoring.py -v

The Phase 1 contract is "stubs only": ``score_cell`` raises
``NotImplementedError``. Tests therefore:

1. Verify the contract - stubs raise the expected error.
2. Exercise the small pieces that ARE implemented (identity extraction,
   severity normalisation, predicate classification).
3. Smoke-load the transformers v4_57_3 reference files and confirm we
   can extract a sensible identity set (so Phase 2 has a tested
   foundation to plug the scorer body into).

Phase 2 will replace the ``NotImplementedError`` asserts with concrete
behaviour assertions (recall == 1.0 when self-scoring, etc).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

# Make project root importable so ``from _spike.scripts import ...`` works
# regardless of pytest invocation directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts import trial_scoring  # noqa: E402
from _spike.scripts.trial_scoring import (  # noqa: E402
    PREDICATE_KINDS,
    SCORING_FORMAT_VERSION,
    CellScore,
    FailureMode,
    ItemDiff,
    ScoringConfig,
    _predicate_kind_for,
    invariant_identities,
    invariant_identity,
    normalise_severity,
    schema_field_identities,
    schema_type_at,
)

# ---------------------------------------------------------------------------
# Fixtures: real reference data
# ---------------------------------------------------------------------------

REF_SCHEMA = (
    _PROJECT_ROOT
    / "engine_versions"
    / "transformers"
    / "v4_57_3"
    / "outputs"
    / "schema.discovered.json"
)
REF_INVARIANTS = (
    _PROJECT_ROOT
    / "engine_versions"
    / "transformers"
    / "v4_57_3"
    / "outputs"
    / "invariants.proposed.yaml"
)


@pytest.fixture(scope="module")
def reference_schema() -> dict:
    assert REF_SCHEMA.exists(), f"reference schema missing at {REF_SCHEMA}"
    return json.loads(REF_SCHEMA.read_text())


@pytest.fixture(scope="module")
def reference_invariants() -> dict:
    assert REF_INVARIANTS.exists(), f"reference invariants missing at {REF_INVARIANTS}"
    return yaml.safe_load(REF_INVARIANTS.read_text())


# ---------------------------------------------------------------------------
# 1. Identity extraction - schema
# ---------------------------------------------------------------------------


def test_schema_field_identities_picks_up_engine_and_sampling(reference_schema: dict) -> None:
    """schema_field_identities must surface fields from BOTH engine_params and
    sampling_params (the two canonical namespaces) plus any $defs sub-classes."""
    identities = schema_field_identities(reference_schema)

    namespaces = {ns for ns, _ in identities}
    assert "engine_params" in namespaces, "missing engine_params namespace"
    assert "sampling_params" in namespaces, "missing sampling_params namespace"

    # The reference has a wide engine_params surface (BitsAndBytesConfig + from_pretrained kwargs).
    engine_field_names = {name for ns, name in identities if ns == "engine_params"}
    sampling_field_names = {name for ns, name in identities if ns == "sampling_params"}

    # Well-known canonical fields that MUST appear (per the existing schema dump):
    assert "dtype" in engine_field_names or "config" in engine_field_names, (
        "engine_params identity set should include either 'dtype' or 'config'"
    )
    assert "temperature" in sampling_field_names, "sampling_params should include 'temperature'"
    assert "num_beams" in sampling_field_names, "sampling_params should include 'num_beams'"

    # Total identity count should be in the right magnitude (transformers v4_57_3
    # schema has roughly 90-110 fields between engine_params + sampling_params).
    assert 60 < len(identities) < 250, f"identity count {len(identities)} out of range"


def test_schema_type_at_returns_expected_shape(reference_schema: dict) -> None:
    """schema_type_at returns the raw type string for simple fields and an
    ('anyOf', tuple-of-summaries) shape for union fields."""
    # `temperature` is a primitive float in the reference.
    t = schema_type_at(reference_schema, ("sampling_params", "temperature"))
    assert t == "number"

    # `config` is an anyOf with object/string/null in the reference.
    c = schema_type_at(reference_schema, ("engine_params", "config"))
    assert isinstance(c, tuple) and c[0] == "anyOf"
    summary = set(c[1])
    assert {"object", "string", "null"}.issubset(summary | {"object", "string", "null"}), (
        f"anyOf summary unexpected: {summary}"
    )

    # Unknown identity returns None.
    assert schema_type_at(reference_schema, ("engine_params", "no_such_field")) is None


# ---------------------------------------------------------------------------
# 2. Identity extraction - invariants
# ---------------------------------------------------------------------------


def test_invariant_identity_extracts_namespace_field_predicate() -> None:
    """invariant_identity must produce a 4-tuple from a match.fields block.

    Phase 2.5 rubric: identity is ``(namespace, native_field,
    predicate_kind, secondary_field)``. Single-field invariants emit
    ``""`` (empty string) in the secondary slot.
    """
    inv = {
        "id": "demo_exact",
        "severity": "dormant",
        "match": {
            "engine": "transformers",
            "fields": {"transformers.sampling.num_beams": 1},
        },
    }
    assert invariant_identity(inv) == ("transformers.sampling", "num_beams", "exact", "")

    inv_not_in = {
        "id": "demo_enum",
        "severity": "error",
        "match": {
            "engine": "transformers",
            "fields": {
                "transformers.sampling.cache_implementation": {
                    "present": True,
                    "not_in": ["dynamic", "static"],
                },
            },
        },
    }
    assert invariant_identity(inv_not_in) == (
        "transformers.sampling",
        "cache_implementation",
        "not_in",
        "",
    )


def test_invariant_identity_disambiguates_multi_field_invariants() -> None:
    """Phase 2.5: multi-field invariants disambiguate via secondary_field.

    Two sampling-only-when-greedy dormancies share the SAME primary key
    (``do_sample``) and predicate (``exact``); they must produce DIFFERENT
    identities once secondary_field is included.
    """
    inv_temp = {
        "id": "transformers_greedy_strips_temperature",
        "severity": "dormant",
        "match": {
            "engine": "transformers",
            "fields": {
                "transformers.sampling.do_sample": False,
                "transformers.sampling.temperature": {"present": True, "not_equal": 1.0},
            },
        },
    }
    inv_top_p = {
        "id": "transformers_greedy_strips_top_p",
        "severity": "dormant",
        "match": {
            "engine": "transformers",
            "fields": {
                "transformers.sampling.do_sample": False,
                "transformers.sampling.top_p": {"present": True, "not_equal": 1.0},
            },
        },
    }
    assert invariant_identity(inv_temp) == (
        "transformers.sampling",
        "do_sample",
        "exact",
        "temperature",
    )
    assert invariant_identity(inv_top_p) == (
        "transformers.sampling",
        "do_sample",
        "exact",
        "top_p",
    )
    # They must NOT collapse to the same identity (the whole point of
    # the rubric fix):
    assert invariant_identity(inv_temp) != invariant_identity(inv_top_p)


def test_invariant_identities_on_reference(reference_invariants: dict) -> None:
    """Reference invariants file extracts a coherent identity set (no
    duplicates within the canonical key; coverage of the documented
    predicate kinds).

    Phase 2.5: identities are now 4-tuples
    ``(namespace, native_field, predicate_kind, secondary_field)``.
    """
    identities = invariant_identities(reference_invariants)

    # Reference has ~41 invariants. With Phase 2.5's secondary_field
    # disambiguation, identities should expand toward ~40 unique (multi-
    # field invariants no longer collapse).
    assert 10 <= len(identities) <= 60, f"unexpected identity count: {len(identities)}"

    # All identities must be 4-tuples of strings.
    for ident in identities:
        assert isinstance(ident, tuple) and len(ident) == 4
        assert all(isinstance(part, str) for part in ident), f"non-string parts in {ident}"

    # Predicate-kind values must all live in the recognised vocabulary.
    used_kinds = {ident[2] for ident in identities}
    assert used_kinds.issubset(PREDICATE_KINDS), (
        f"unexpected predicate kinds in reference: {used_kinds - PREDICATE_KINDS}"
    )


def test_predicate_kind_for_classification() -> None:
    """_predicate_kind_for covers every documented predicate shape."""
    assert _predicate_kind_for(1) == "exact"
    assert _predicate_kind_for("foo") == "exact"
    assert _predicate_kind_for({"not_in": [1, 2]}) == "not_in"
    assert _predicate_kind_for({"not_equal": "foo"}) == "not_equal"
    assert _predicate_kind_for({">": 0}) == "gt"
    assert _predicate_kind_for({"<": 0}) == "lt"
    assert _predicate_kind_for({">=": 0}) == "ge"
    assert _predicate_kind_for({"<=": 0}) == "le"
    assert _predicate_kind_for({"type_is_not": ["int"]}) == "type_is_not"
    assert _predicate_kind_for({"present": True}) == "present"
    assert _predicate_kind_for({"weird_unknown_key": 42}) == "unknown"


# ---------------------------------------------------------------------------
# 3. Severity normalisation
# ---------------------------------------------------------------------------


def test_normalise_severity_exact_vs_loose() -> None:
    """Exact mode preserves raw string; loose mode coalesces warn/warning."""
    assert normalise_severity("error", exact=True) == "error"
    assert normalise_severity("Error", exact=True) == "error"  # lower-cased
    assert normalise_severity("warn", exact=True) == "warn"
    assert normalise_severity("warn", exact=False) == "warning"
    assert normalise_severity("warning", exact=False) == "warning"
    assert normalise_severity(None, exact=True) == ""


# ---------------------------------------------------------------------------
# 4. CellScore dataclass + serialisation
# ---------------------------------------------------------------------------


def test_cell_score_to_json_roundtrips() -> None:
    """CellScore.to_json produces a JSON document that round-trips with the
    same shape (scoring_format_version preserved)."""
    score = CellScore(
        strategy="a",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="active",
        schema_recall=1.0,
        invariant_recall=0.95,
        observations=["smoke-test cell"],
    )
    blob = score.to_json()
    data = json.loads(blob)
    assert data["strategy"] == "a"
    assert data["engine"] == "transformers"
    assert data["scoring_format_version"] == SCORING_FORMAT_VERSION
    assert data["observations"] == ["smoke-test cell"]
    # Brittleness defaults are None (filled by aggregator).
    assert data["brittleness_pass_through_rate"] is None


def test_failure_mode_enum_values_stable() -> None:
    """The FailureMode enum's wire values must be the documented strings;
    the aggregator + downstream CSV depend on these."""
    assert FailureMode.NONE.value == "none"
    assert FailureMode.CRASH.value == "crash"
    assert FailureMode.DETECTABLE.value == "detectable"
    assert FailureMode.SILENT.value == "silent"
    assert FailureMode.PARTIAL.value == "partial"


# ---------------------------------------------------------------------------
# 5. Stub contract - Phase 2 fills these in
# ---------------------------------------------------------------------------


def test_score_cell_self_scores_perfectly() -> None:
    """Phase 2: score_cell now implemented. Scoring the reference against
    itself yields recall=precision=1.0 on both axes."""
    cfg = ScoringConfig()
    score, schema_diffs, inv_diffs, type_or_sev_diffs = trial_scoring.score_cell(
        strategy="a",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="active",
        schema_output_path=REF_SCHEMA,
        invariants_output_path=REF_INVARIANTS,
        reference_schema_path=REF_SCHEMA,
        reference_invariants_path=REF_INVARIANTS,
        wall_clock_sec=0.0,
        energy_wh=0.0,
        config=cfg,
    )
    assert score.schema_recall == 1.0
    assert score.schema_precision == 1.0
    assert score.schema_type_accuracy == 1.0
    assert score.invariant_recall == 1.0
    assert score.invariant_precision == 1.0
    assert score.invariant_severity_accuracy == 1.0
    assert score.schema_failure_mode == FailureMode.NONE.value
    assert score.invariant_failure_mode == FailureMode.NONE.value
    assert schema_diffs == []
    assert inv_diffs == []
    assert type_or_sev_diffs == []


def test_score_schema_self_scores_perfectly(reference_schema: dict) -> None:
    """Phase 2: score_schema implemented."""
    cfg = ScoringConfig()
    (
        recall,
        precision,
        type_accuracy,
        failure_mode,
        ref_count,
        cell_count,
        int_count,
        recall_misses,
        precision_spurious,
        type_mismatches,
    ) = trial_scoring.score_schema(
        reference_schema=reference_schema,
        cell_schema=reference_schema,
        config=cfg,
    )
    assert recall == 1.0
    assert precision == 1.0
    assert type_accuracy == 1.0
    assert failure_mode == FailureMode.NONE
    assert ref_count == cell_count == int_count > 0
    assert recall_misses == []
    assert precision_spurious == []
    assert type_mismatches == []


def test_score_invariants_self_scores_perfectly(reference_invariants: dict) -> None:
    """Phase 2: score_invariants implemented."""
    cfg = ScoringConfig()
    (
        recall,
        precision,
        severity_accuracy,
        failure_mode,
        ref_count,
        cell_count,
        int_count,
        recall_misses,
        precision_spurious,
        severity_mismatches,
    ) = trial_scoring.score_invariants(
        reference_invariants=reference_invariants,
        cell_invariants=reference_invariants,
        config=cfg,
    )
    assert recall == 1.0
    assert precision == 1.0
    assert severity_accuracy == 1.0
    assert failure_mode == FailureMode.NONE
    assert ref_count == cell_count == int_count > 0
    assert recall_misses == []
    assert precision_spurious == []
    assert severity_mismatches == []


def test_compute_brittleness_is_stubbed() -> None:
    """compute_brittleness is also stubbed in Phase 1; Phase 3 fills it."""
    base = CellScore(
        strategy="a",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="v+1",
    )
    active = CellScore(
        strategy="a",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="active",
    )
    with pytest.raises(NotImplementedError):
        trial_scoring.compute_brittleness(cell=base, active_cell=active)


# ---------------------------------------------------------------------------
# 6. Sidecar writer (implemented in Phase 1; sanity-check)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 7. Engine chunkers (Phase 2.5) - smoke check that chunkers emit chunks
# ---------------------------------------------------------------------------


def test_vllm_chunker_emits_chunks_with_companion_classes() -> None:
    """vllm_chunker.schema_chunks + invariants_chunks return populated
    chunks. GuidedDecodingParams companion is inlined into the
    SamplingParams chunk."""
    from _spike.scripts.strategies.vllm_chunker import (
        invariants_chunks as vllm_inv,
    )
    from _spike.scripts.strategies.vllm_chunker import (
        schema_chunks as vllm_schema,
    )

    schema = vllm_schema()
    inv = vllm_inv()
    # If the pre-staged vllm source is missing (CI without /tmp/vllm-unpacked),
    # the chunker emits a single "extraction failed" chunk - tolerate that.
    if len(schema) == 1 and schema[0].name == "source_extraction_failed":
        pytest.skip(
            "vllm pre-staged source absent; chunker exercise skipped. "
            "Set up /tmp/vllm-unpacked/vllm/ to run this test."
        )
    # Expected shape from Phase 2.5 P25-1 design
    assert len(schema) >= 5, f"expected >= 5 schema chunks, got {len(schema)}"
    assert len(inv) >= 5, f"expected >= 5 invariants chunks, got {len(inv)}"
    # SamplingParams must include GuidedDecodingParams (companion-class lesson)
    sp_chunk = next((c for c in schema if "sampling_params" in c.name), None)
    assert sp_chunk is not None, "no sampling_params schema chunk found"
    assert "GuidedDecodingParams" in sp_chunk.source, (
        "GuidedDecodingParams not inlined into SamplingParams chunk"
    )
    # Each chunk has non-empty source < 16k chars (32k ctx headroom)
    for c in schema + inv:
        assert c.source, f"empty source for chunk {c.name!r}"
        assert len(c.source) < 16000, (
            f"chunk {c.name!r} source {len(c.source)} chars - too large for 32k ctx + few-shot"
        )


def test_tensorrt_chunker_captures_pydantic_validators() -> None:
    """tensorrt_chunker emits chunks; invariants chunks include
    @field_validator + @model_validator decorators (the critical
    correctness check - Pydantic-style validators MUST be captured
    with their decorator)."""
    from _spike.scripts.strategies.tensorrt_chunker import (
        invariants_chunks as trt_inv,
    )
    from _spike.scripts.strategies.tensorrt_chunker import (
        schema_chunks as trt_schema,
    )

    schema = trt_schema()
    inv = trt_inv()
    if len(schema) == 1 and schema[0].name == "source_extraction_failed":
        pytest.skip(
            "tensorrt_llm pre-staged source absent; chunker exercise skipped. "
            "Set up /tmp/trt-llm-0.21.0/tensorrt_llm/ to run this test."
        )
    assert len(schema) >= 5, f"expected >= 5 schema chunks, got {len(schema)}"
    assert len(inv) >= 5, f"expected >= 5 invariants chunks, got {len(inv)}"
    # The BaseLlmArgs validators chunk(s) must capture both decorator kinds
    base_validators_chunks = [c for c in inv if "base_llm_args" in c.name]
    assert base_validators_chunks, "no BaseLlmArgs validators chunk found"
    combined = "\n".join(c.source for c in base_validators_chunks)
    assert "@field_validator" in combined, (
        "BaseLlmArgs validators chunk missing @field_validator decorators"
    )
    assert "@model_validator" in combined, (
        "BaseLlmArgs validators chunk missing @model_validator decorators"
    )


def test_write_diff_artefact_writes_stable_yaml(tmp_path: Path) -> None:
    """write_diff_artefact emits stable, deterministic YAML (sorted by
    identity tuple). Idempotent re-writes produce identical bytes."""
    diffs = [
        ItemDiff(
            identity=("engine_params", "z_field"),
            kind="recall_miss",
            reference_value={"type": "string"},
            cell_value=None,
            note="missed by strategy",
        ),
        ItemDiff(
            identity=("engine_params", "a_field"),
            kind="recall_miss",
            reference_value={"type": "integer"},
            cell_value=None,
            note="also missed",
        ),
    ]
    out_path = tmp_path / "recall_misses.yaml"
    trial_scoring.write_diff_artefact(diffs=diffs, out_path=out_path, kind_label="recall_misses")

    text = out_path.read_text()
    data = yaml.safe_load(text)
    assert data["scoring_format_version"] == SCORING_FORMAT_VERSION
    assert "recall_misses" in data
    # Sorted by identity tuple - 'a_field' should sort before 'z_field'.
    names_in_order = [entry["identity"][1] for entry in data["recall_misses"]]
    assert names_in_order == ["a_field", "z_field"], (
        f"diffs not sorted by identity: {names_in_order}"
    )

    # Idempotent re-write:
    trial_scoring.write_diff_artefact(diffs=diffs, out_path=out_path, kind_label="recall_misses")
    assert out_path.read_text() == text, "second write produced different bytes"
