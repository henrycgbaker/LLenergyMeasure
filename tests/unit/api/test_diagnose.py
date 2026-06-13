"""Unit tests for api/diagnose.py - the Stage-1 LLM bump-diagnose proposer.

No live model and no GPU: the model call is injected as a stub returning
RECORDED fixtures (captured from the trial's raw outputs under
``tests/unit/api/fixtures/diagnose/``), and the gate is injected as a stubbed
verdict list. The two live paths (ollama + the docker gate) are not exercised.

Coverage:
  - prompt construction from a synthetic residual + diff-scoped source (1a) and
    from a mined envelope + config surface (1b);
  - structured-output parsing, including malformed-kwargs flagging (the trial's
    12b type-malformed case) - the parser must flag, never crash;
  - the inject-stubbed-model end-to-end path -> entries (recorded fixture);
  - gate wiring: a stubbed verdict -> only gate-confirmed entries are written;
  - llm_diagnose provenance round-trips through the rules loader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llenergymeasure.api.diagnose import (
    CLASSIFICATIONS,
    DiagnoseError,
    Diagnosis,
    build_carried_triage_prompt,
    build_gap_diagnose_prompt,
    diagnose_carried_failures,
    diagnose_gaps,
    diff_scoped_source,
    parse_diagnoses,
    render_proposed_yaml,
)

FIXTURES = Path(__file__).parent / "fixtures" / "diagnose"
CLEAN_RAW = FIXTURES / "qwen2_5-coder_32b.clean.json"
MALFORMED_RAW = FIXTURES / "gemma3_12b.malformed-kwargs.json"

# The 7 residual ids the recorded fixtures diagnose (bump-1 transformers).
_RESIDUAL_IDS = [
    "transformers_cache_choice_cache_implementation_not_in_allowlist",
    "transformers_num_return_vs_beams_do_sample_eq_false_and_num_beams_eq_1",
    "transformers_raises_num_beams_eq_1",
    "transformers_watermarking_type_watermarking_config_type_not_in_WatermarkingConfig",
    "transformers_bnb_4bit_quant_type_dormant_without_load_in_4bit",
    "transformers_bnb_4bit_compute_dtype_dormant_without_load_in_4bit",
    "transformers_bnb_4bit_use_double_quant_dormant_without_load_in_4bit",
]


class _StubModel:
    """Inject a recorded fixture as the model response; record the prompt."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompt: str | None = None

    def complete(self, prompt: str) -> str:
        self.prompt = prompt
        return self.response


def _carried_entry(rule_id: str, severity: str = "error") -> dict[str, Any]:
    return {
        "id": rule_id,
        "native_type": "transformers.GenerationConfig",
        "severity": severity,
        "invariant_under_test": f"{rule_id} fires",
        "message_template": "old message",
        "match": {"fields": {f"transformers.engine_params.{rule_id}": {"present": True}}},
        "expected_outcome": {"outcome": "error"},
        "kwargs_positive": {"x": 1},
        "kwargs_negative": {"x": 0},
    }


def _carried_corpus() -> dict[str, Any]:
    invs = []
    for rid in _RESIDUAL_IDS:
        sev = "dormant" if "bnb_4bit" in rid else "error"
        invs.append(_carried_entry(rid, severity=sev))
    return {"schema_version": "1.0.0", "engine": "transformers", "invariants": invs}


# ---------------------------------------------------------------------------
# Diff-scoping
# ---------------------------------------------------------------------------


def test_diff_scoped_source_numbers_lines_and_labels_files(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("first\nsecond\n")
    (tmp_path / "b.py").write_text("only\n")
    src = diff_scoped_source(source_root=tmp_path, files=["a.py", "b.py"])
    assert "### FILE: a.py" in src
    assert "### FILE: b.py" in src
    assert "    1: first" in src
    assert "    2: second" in src


def test_diff_scoped_source_skips_missing_and_empty(tmp_path: Path) -> None:
    (tmp_path / "real.py").write_text("content\n")
    (tmp_path / "blank.py").write_text("   \n")
    src = diff_scoped_source(source_root=tmp_path, files=["real.py", "blank.py", "absent.py"])
    assert "### FILE: real.py" in src
    assert "blank.py" not in src
    assert "absent.py" not in src


def test_diff_scoped_source_empty_raises(tmp_path: Path) -> None:
    with pytest.raises(DiagnoseError, match="diff-scoped source is empty"):
        diff_scoped_source(source_root=tmp_path, files=["nope.py"])


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_carried_triage_prompt_embeds_residual_and_source() -> None:
    carried = [_carried_entry("rule_a"), _carried_entry("rule_b")]
    prompt = build_carried_triage_prompt(
        engine="transformers",
        old_version="4.57.3",
        new_version="5.7.0",
        carried=carried,
        source="### FILE: x.py\n    1: code",
    )
    assert "transformers 4.57.3 -> 5.7.0" in prompt
    assert "rule_a" in prompt and "rule_b" in prompt
    assert "exactly 2 objects" in prompt
    assert "### FILE: x.py" in prompt
    # The output contract demands native JSON types (the kwargs lever).
    assert "NATIVE JSON types" in prompt


def test_carried_triage_prompt_requires_residual_and_source() -> None:
    with pytest.raises(DiagnoseError, match="at least one residual"):
        build_carried_triage_prompt(
            engine="transformers",
            old_version="4.57.3",
            new_version="5.7.0",
            carried=[],
            source="x",
        )
    with pytest.raises(DiagnoseError, match="non-empty diff-scoped source"):
        build_carried_triage_prompt(
            engine="transformers",
            old_version="4.57.3",
            new_version="5.7.0",
            carried=[_carried_entry("r")],
            source="   ",
        )


def test_gap_diagnose_prompt_summarises_envelope() -> None:
    schema = {
        "engine": "transformers",
        "engine_version": "5.7.0",
        "engine_params": {"config": {}, "cache_dir": {}},
        "sampling_params": {"max_length": {}, "temperature": {}},
    }
    prompt = build_gap_diagnose_prompt(
        engine="transformers",
        new_version="5.7.0",
        schema=schema,
        source="### FILE: c.py\n    1: x",
    )
    assert "BLINDNESS check" in prompt
    assert "engine_params (2 fields)" in prompt
    assert "cache_dir" in prompt
    assert "### FILE: c.py" in prompt


# ---------------------------------------------------------------------------
# Structured-output parsing
# ---------------------------------------------------------------------------


def test_parse_clean_fixture_yields_seven_well_formed() -> None:
    diags = parse_diagnoses(CLEAN_RAW.read_text())
    assert len(diags) == 7
    assert all(isinstance(d, Diagnosis) for d in diags)
    assert all(d.classification in CLASSIFICATIONS for d in diags)
    # The clean (qwen2.5-coder:32b) fixture has native-typed kwargs -> none flagged.
    assert all(not d.is_malformed for d in diags)


def test_parse_malformed_fixture_flags_stringly_typed_kwargs() -> None:
    diags = parse_diagnoses(MALFORMED_RAW.read_text())
    assert len(diags) == 7
    malformed = [d for d in diags if d.is_malformed]
    # The 12b fixture emits string "True"/"False"/"None" probes -> flagged, not crash.
    assert malformed, "expected at least one malformed diagnosis from the 12b fixture"
    flagged_fields = {f for d in malformed for keys in d.kwargs_malformed.values() for f in keys}
    assert "do_sample" in flagged_fields or "use_cache" in flagged_fields


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ("not json at all", "not valid JSON"),
        ("[1, 2, 3]", "root is not an object"),
        ('{"other": []}', "no 'diagnoses' array"),
        ('{"diagnoses": [{"reason": "no id"}]}', "no usable entries"),
    ],
)
def test_parse_explicit_failures(raw: str, match: str) -> None:
    with pytest.raises(DiagnoseError, match=match):
        parse_diagnoses(raw)


def test_parse_normalises_unknown_classification() -> None:
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": "r",
                    "classification": "not_a_real_class",
                    "reason": "x",
                    "citation": "f:1",
                    "kwargs_positive": {},
                    "kwargs_negative": {},
                }
            ]
        }
    )
    diags = parse_diagnoses(raw)
    assert diags[0].classification == "unknown"


# ---------------------------------------------------------------------------
# End-to-end: stubbed model + stubbed gate -> only confirmed entries
# ---------------------------------------------------------------------------


def _gate_stub(verdicts_by_id: dict[str, str]):
    def runner(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for p in proposals:
            out.append({"rule_id": p["rule_id"], "verdict": verdicts_by_id.get(p["rule_id"])})
        return out

    return runner


def test_carried_failures_end_to_end_emits_only_confirmed(tmp_path: Path) -> None:
    (tmp_path / "configuration_utils.py").write_text("source line\n" * 10)
    model = _StubModel(CLEAN_RAW.read_text())

    # Confirm 3 of the 4 error rules; leave one unconfirmed; bnb trio not
    # construction-confirmable (silent dormancy).
    verdicts = {
        "transformers_cache_choice_cache_implementation_not_in_allowlist": "confirmed",
        "transformers_num_return_vs_beams_do_sample_eq_false_and_num_beams_eq_1": "confirmed",
        "transformers_raises_num_beams_eq_1": "confirmed",
        "transformers_watermarking_type_watermarking_config_type_not_in_WatermarkingConfig": "not_confirmed",
        "transformers_bnb_4bit_quant_type_dormant_without_load_in_4bit": "not_construction_confirmable",
        "transformers_bnb_4bit_compute_dtype_dormant_without_load_in_4bit": "not_construction_confirmable",
        "transformers_bnb_4bit_use_double_quant_dormant_without_load_in_4bit": "not_construction_confirmable",
    }

    result = diagnose_carried_failures(
        engine="transformers",
        old_version="4.57.3",
        new_version="5.7.0",
        residual_ids=_RESIDUAL_IDS,
        carried_corpus=_carried_corpus(),
        source_root=tmp_path,
        cited_files=["configuration_utils.py"],
        model=model,
        gate_runner=_gate_stub(verdicts),
    )

    confirmed_ids = {e["id"] for e in result.confirmed}
    assert confirmed_ids == {
        "transformers_cache_choice_cache_implementation_not_in_allowlist",
        "transformers_num_return_vs_beams_do_sample_eq_false_and_num_beams_eq_1",
        "transformers_raises_num_beams_eq_1",
    }
    # Only confirmed entries carry the provenance and are emitted.
    assert all(e["added_by"] == "llm_diagnose" for e in result.confirmed)
    assert len(result.unconfirmed) == 1
    assert len(result.not_construction_confirmable) == 3
    # The YAML payload contains the confirmed entries only.
    assert result.proposed_yaml is not None
    assert "not_confirmed" not in (result.proposed_yaml or "")
    assert result.proposed_yaml.count("added_by: llm_diagnose") == 3
    # The prompt was diff-scoped and saw the source.
    assert model.prompt is not None and "### FILE: configuration_utils.py" in model.prompt

    # The emitted YAML (the REAL emission path through _validated_entry) loads
    # through the production rules loader with the llm_diagnose provenance.
    from llenergymeasure.config.engine_rules import EngineRulesLoader

    engine_dir = tmp_path / "engines" / "transformers"
    engine_dir.mkdir(parents=True)
    (engine_dir / "rules.proposed.yaml").write_text(result.proposed_yaml)
    loaded = EngineRulesLoader(corpus_root=tmp_path / "engines").load_rules("transformers")
    assert len(loaded.invariants) == 3
    assert all(inv.added_by == "llm_diagnose" for inv in loaded.invariants)


def test_malformed_kwargs_dropped_before_gating(tmp_path: Path) -> None:
    (tmp_path / "configuration_utils.py").write_text("src\n" * 5)
    model = _StubModel(MALFORMED_RAW.read_text())

    gated_ids: list[str] = []

    def recording_gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        gated_ids.extend(p["rule_id"] for p in proposals)
        return [{"rule_id": p["rule_id"], "verdict": "confirmed"} for p in proposals]

    result = diagnose_carried_failures(
        engine="transformers",
        old_version="4.57.3",
        new_version="5.7.0",
        residual_ids=_RESIDUAL_IDS,
        carried_corpus=_carried_corpus(),
        source_root=tmp_path,
        cited_files=["configuration_utils.py"],
        model=model,
        gate_runner=recording_gate,
    )
    # The 12b fixture's malformed probes are dropped BEFORE the gate sees them.
    assert result.dropped_malformed, "expected malformed diagnoses to be dropped"
    dropped_ids = {r["rule_id"] for r in result.dropped_malformed}
    assert dropped_ids.isdisjoint(set(gated_ids)), "malformed proposals must not reach the gate"


def test_carried_failures_no_confirmed_writes_nothing(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("src\n")
    model = _StubModel(CLEAN_RAW.read_text())
    result = diagnose_carried_failures(
        engine="transformers",
        old_version="4.57.3",
        new_version="5.7.0",
        residual_ids=_RESIDUAL_IDS,
        carried_corpus=_carried_corpus(),
        source_root=tmp_path,
        cited_files=["x.py"],
        model=model,
        gate_runner=_gate_stub(dict.fromkeys(_RESIDUAL_IDS, "not_confirmed")),
    )
    assert result.confirmed == []
    assert result.proposed_yaml is None


def test_carried_failures_rejects_unknown_residual_id(tmp_path: Path) -> None:
    (tmp_path / "x.py").write_text("src\n")
    with pytest.raises(DiagnoseError, match="absent from carried corpus"):
        diagnose_carried_failures(
            engine="transformers",
            old_version="4.57.3",
            new_version="5.7.0",
            residual_ids=["not_a_real_rule"],
            carried_corpus=_carried_corpus(),
            source_root=tmp_path,
            cited_files=["x.py"],
            model=_StubModel(CLEAN_RAW.read_text()),
            gate_runner=_gate_stub({}),
        )


def test_gap_diagnose_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text("class Config:\n    pass\n" * 5)
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": "transformers_gap_docstring_bound",
                    "classification": "rule_morphed",
                    "reason": "constraint only in docstring Args",
                    "citation": "config.py:3",
                    "kwargs_positive": {"n": -1},
                    "kwargs_negative": {"n": 1},
                }
            ]
        }
    )
    result = diagnose_gaps(
        engine="transformers",
        new_version="5.7.0",
        schema={"engine": "transformers", "engine_params": {"n": {}}},
        source_root=tmp_path,
        config_surface_files=["config.py"],
        model=_StubModel(raw),
        gate_runner=_gate_stub({"transformers_gap_docstring_bound": "confirmed"}),
    )
    assert len(result.confirmed) == 1
    assert result.confirmed[0]["id"] == "transformers_gap_docstring_bound"
    assert result.confirmed[0]["added_by"] == "llm_diagnose"
    assert result.mode == "gap_diagnose"


# ---------------------------------------------------------------------------
# llm_diagnose provenance round-trips through the loader
# ---------------------------------------------------------------------------


def test_emitted_yaml_loads_with_llm_diagnose_provenance(tmp_path: Path) -> None:
    """A gate-confirmed emitted entry loads through the real rules loader with
    added_by == llm_diagnose - proving the SSOT extension round-trips."""
    from llenergymeasure.config.engine_rules import EngineRulesLoader

    entry = {
        "id": "transformers_diagnosed_rule",
        "engine": "transformers",
        "library": "transformers",
        "invariant_under_test": "x fires",
        "severity": "error",
        "native_type": "transformers.GenerationConfig",
        "match": {"fields": {"transformers.engine_params.x": {"present": True}}},
        "kwargs_positive": {"x": 1},
        "kwargs_negative": {"x": 0},
        "expected_outcome": {"outcome": "error", "emission_channel": "none"},
        "references": ["llm_diagnose: reworded"],
        "added_by": "llm_diagnose",
    }
    yaml_doc = render_proposed_yaml(engine="transformers", engine_version="5.7.0", entries=[entry])
    corpus_root = tmp_path / "engines"
    engine_dir = corpus_root / "transformers"
    engine_dir.mkdir(parents=True)
    (engine_dir / "rules.proposed.yaml").write_text(yaml_doc)

    loader = EngineRulesLoader(corpus_root=corpus_root)
    invariants = loader.load_rules("transformers").invariants
    assert len(invariants) == 1
    assert invariants[0].added_by == "llm_diagnose"
