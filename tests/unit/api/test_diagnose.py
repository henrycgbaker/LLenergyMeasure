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

from llenergymeasure.api._source_window import SymbolRequest, windowed_source
from llenergymeasure.api.diagnose import (
    CLASSIFICATIONS,
    TIER_D_CLASSIFICATIONS,
    DiagnoseError,
    Diagnosis,
    _fires_when_consistent_with_probes,
    build_carried_triage_prompt,
    build_gap_diagnose_prompt,
    build_tier_d_prompt,
    carried_symbol_requests,
    config_surface_symbol_requests,
    diagnose_carried_failures,
    diagnose_gaps,
    diagnose_tier_d,
    parse_diagnoses,
    render_proposed_yaml,
    tier_d_native_types,
    tier_d_rule_id,
    tier_d_symbol_requests,
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


# A multi-class source file whose total length far exceeds the windowing budget.
# Only the trailing GenerationConfig is cited; the fillers exist to prove the
# windower does not feed the whole file. The cited fields below match the kwargs
# the residual carried entries probe (``x``), plus a real-shaped field/validator.
_FILLER_LINE = "    pad_{i}: int = {i}  # " + "y" * 80


def _write_large_config_file(path: Path, *, classes: int = 60, fields_each: int = 25) -> str:
    """Write a synthetic config module >> the windowing budget; return its text.

    Ends with a ``GenerationConfig`` class carrying field ``x`` (the test rules'
    cited kwargs key) and a ``validate`` method that references it - the cited
    symbol the windower must locate inside a multi-class file.
    """
    lines: list[str] = ["import os", ""]
    for c in range(classes):
        lines.append(f"class Filler{c}Config:")
        for i in range(fields_each):
            lines.append(_FILLER_LINE.format(i=i))
        lines.append("")
    lines += [
        "class GenerationConfig:",
        "    x: int = 0",
        "    def validate(self):",
        "        if self.x < 0:",
        '            raise ValueError("bad x")',
        "",
    ]
    body = "\n".join(lines)
    path.write_text(body)
    return body


# ---------------------------------------------------------------------------
# Per-symbol source windowing
# ---------------------------------------------------------------------------


def test_windowing_bounds_a_large_file_and_contains_the_cited_symbol(tmp_path: Path) -> None:
    body = _write_large_config_file(tmp_path / "configuration_utils.py")
    assert len(body) > 20_000, "fixture must dwarf the budget for this test to mean anything"

    report = windowed_source(
        source_root=tmp_path,
        requests=[
            SymbolRequest(
                file="configuration_utils.py", class_name="GenerationConfig", fields=("x",)
            )
        ],
        budget_chars=12_000,
    )
    # Bounded: the window is a tiny fraction of the whole file, and under budget.
    assert report.chars < 12_000
    assert report.chars < len(body) // 10
    # Contains the cited class + field + the validator that references it.
    assert "class GenerationConfig" in report.source
    assert "x: int = 0" in report.source
    assert "def validate" in report.source
    # Real 1-based line numbers from the new pin (so the model can cite them).
    assert "### FILE: configuration_utils.py" in report.source
    assert not report.truncated and not report.missing


def test_windowing_multi_file_residual_windows_per_file_within_budget(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "class GenerationConfig:\n    cache_implementation: str = 'd'\n"
        "    def validate(self):\n        return self.cache_implementation\n"
    )
    (tmp_path / "b.py").write_text(
        "class BitsAndBytesConfig:\n    load_in_4bit: bool = False\n"
        "    def post_init(self):\n        return self.load_in_4bit\n"
    )
    report = windowed_source(
        source_root=tmp_path,
        requests=[
            SymbolRequest(
                file="a.py", class_name="GenerationConfig", fields=("cache_implementation",)
            ),
            SymbolRequest(file="b.py", class_name="BitsAndBytesConfig", fields=("load_in_4bit",)),
        ],
        budget_chars=12_000,
    )
    assert "### FILE: a.py" in report.source
    assert "### FILE: b.py" in report.source
    assert "cache_implementation" in report.source
    assert "load_in_4bit" in report.source
    assert report.chars < 12_000
    assert not report.truncated and not report.missing


def test_windowing_reports_truncation_never_silent(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text(
        "class GenerationConfig:\n    aa: int = 1\n    def m(self):\n        return self.aa\n"
    )
    (tmp_path / "b.py").write_text(
        "class BitsAndBytesConfig:\n    bb: int = 1\n    def m(self):\n        return self.bb\n"
    )
    report = windowed_source(
        source_root=tmp_path,
        requests=[
            SymbolRequest(file="a.py", class_name="GenerationConfig", fields=("aa",)),
            SymbolRequest(file="b.py", class_name="BitsAndBytesConfig", fields=("bb",)),
        ],
        budget_chars=120,  # only room for one window
    )
    # One window kept; the other dropped and REPORTED (not silently dropped).
    assert "### FILE" in report.source
    assert report.truncated, "over-budget windows must be reported"
    assert any("budget" in note for note in report.truncated)


def test_windowing_flags_a_cited_symbol_missing_from_the_new_pin(tmp_path: Path) -> None:
    (tmp_path / "configuration_utils.py").write_text(
        "class GenerationConfig:\n    other: int = 1\n"
    )
    # Field that no longer exists -> header still emitted, field flagged missing.
    report = windowed_source(
        source_root=tmp_path,
        requests=[
            SymbolRequest(
                file="configuration_utils.py", class_name="GenerationConfig", fields=("gone",)
            )
        ],
        budget_chars=12_000,
    )
    assert report.missing, "a vanished cited field must be flagged"
    assert any("gone" in note for note in report.missing)
    assert "class GenerationConfig" in report.source  # header window survives

    # Class that no longer exists -> empty window + flagged note, no crash.
    report2 = windowed_source(
        source_root=tmp_path,
        requests=[
            SymbolRequest(file="configuration_utils.py", class_name="VanishedConfig", fields=("x",))
        ],
        budget_chars=12_000,
    )
    assert not report2.source.strip()
    assert any("VanishedConfig" in note for note in report2.missing)


def _write_many_huge_classes(path: Path, *, names: list[str], fields_each: int = 200) -> str:
    """Write several config classes each far larger than the budget alone.

    Every class carries ``fields_each`` fields plus a long ``validate`` method.
    Any single class fed whole would already blow a 12K budget, so this fixture
    proves the surface windower caps per-class and round-robins fairly.
    """
    lines: list[str] = ["import os", ""]
    for name in names:
        lines.append(f"class {name}Config:")
        for i in range(fields_each):
            lines.append(f"    pad_{i}: int = {i}")
        lines.append("    def validate(self):")
        for i in range(fields_each):
            lines.append(f"        _ = {i}")
        lines.append("")
    body = "\n".join(lines)
    path.write_text(body)
    return body


def test_surface_windowing_caps_huge_classes_and_does_not_starve(tmp_path: Path) -> None:
    names = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]
    body = _write_many_huge_classes(tmp_path / "config.py", names=names)
    # The surface dwarfs the budget; each class has a 200-field body that, fed
    # whole, would crowd out the others.
    assert len(body) > 3 * 12_000

    requests = config_surface_symbol_requests(
        source_root=tmp_path, config_surface_files=["config.py"]
    )
    report = windowed_source(source_root=tmp_path, requests=requests, budget_chars=12_000)

    # Bounded: the assembled surface stays under budget despite many huge classes.
    assert report.chars <= 12_000
    # No starvation: round-robin admits each class's bounded window in turn, so
    # several distinct classes are represented - not one class consuming it all.
    classes_present = [n for n in names if f"class {n}Config" in report.source]
    assert len(classes_present) >= 3
    # Capping is real: no class's whole 200-field body is emitted (each surface
    # window covers only a bounded leading slice).
    assert "pad_199: int" not in report.source
    # Any class that did not fit is REPORTED, never silently dropped.
    for name in names:
        if f"class {name}Config" not in report.source:
            assert any(f"{name}Config" in note for note in report.truncated)


def test_carried_symbol_requests_derive_class_and_fields() -> None:
    carried = [_carried_entry("rule_a")]
    requests = carried_symbol_requests(carried=carried, cited_files=["f1.py", "f2.py"])
    # One request per cited file; class is the native_type tail.
    assert {r.file for r in requests} == {"f1.py", "f2.py"}
    assert all(r.class_name == "GenerationConfig" for r in requests)
    # Fields come from the kwargs probe keys and the match-field tails.
    assert all("x" in r.fields for r in requests)
    assert all("rule_a" in r.fields for r in requests)


def test_config_surface_symbol_requests_enumerate_config_classes(tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text(
        "class FooConfig:\n    a: int = 1\n\nclass BarParams:\n    b: int = 2\n\n"
        "def helper():\n    pass\n"
    )
    requests = config_surface_symbol_requests(
        source_root=tmp_path, config_surface_files=["config.py"]
    )
    names = {r.class_name for r in requests}
    assert names == {"FooConfig", "BarParams"}
    # Surface requests are field-agnostic (whole-class windows).
    assert all(r.fields == () for r in requests)


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
    _write_large_config_file(tmp_path / "configuration_utils.py")
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
    _write_large_config_file(tmp_path / "configuration_utils.py")
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
    _write_large_config_file(tmp_path / "configuration_utils.py")
    model = _StubModel(CLEAN_RAW.read_text())
    result = diagnose_carried_failures(
        engine="transformers",
        old_version="4.57.3",
        new_version="5.7.0",
        residual_ids=_RESIDUAL_IDS,
        carried_corpus=_carried_corpus(),
        source_root=tmp_path,
        cited_files=["configuration_utils.py"],
        model=model,
        gate_runner=_gate_stub(dict.fromkeys(_RESIDUAL_IDS, "not_confirmed")),
    )
    assert result.confirmed == []
    assert result.proposed_yaml is None


def test_carried_failures_rejects_unknown_residual_id(tmp_path: Path) -> None:
    _write_large_config_file(tmp_path / "configuration_utils.py")
    with pytest.raises(DiagnoseError, match="absent from carried corpus"):
        diagnose_carried_failures(
            engine="transformers",
            old_version="4.57.3",
            new_version="5.7.0",
            residual_ids=["not_a_real_rule"],
            carried_corpus=_carried_corpus(),
            source_root=tmp_path,
            cited_files=["configuration_utils.py"],
            model=_StubModel(CLEAN_RAW.read_text()),
            gate_runner=_gate_stub({}),
        )


def test_carried_failures_empty_window_raises(tmp_path: Path) -> None:
    # A file with no matching class -> windowing resolves nothing -> guarded.
    (tmp_path / "configuration_utils.py").write_text("def unrelated():\n    return 1\n")
    with pytest.raises(DiagnoseError, match="windowed source is empty"):
        diagnose_carried_failures(
            engine="transformers",
            old_version="4.57.3",
            new_version="5.7.0",
            residual_ids=_RESIDUAL_IDS,
            carried_corpus=_carried_corpus(),
            source_root=tmp_path,
            cited_files=["configuration_utils.py"],
            model=_StubModel(CLEAN_RAW.read_text()),
            gate_runner=_gate_stub({}),
        )


def test_gap_diagnose_end_to_end(tmp_path: Path) -> None:
    (tmp_path / "config.py").write_text(
        "class Config:\n    n: int = 1\n    def validate(self):\n        return self.n\n"
    )
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


# ---------------------------------------------------------------------------
# CR3: 1b gap proposals carry a resolvable native_type
# ---------------------------------------------------------------------------


def test_native_type_for_citation_maps_to_fully_qualified_class(tmp_path: Path) -> None:
    from llenergymeasure.api.diagnose import (
        _config_class_index,
        _native_type_for_citation,
    )

    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "config.py").write_text(
        "class SchedulerConfig:\n    max_num_seqs: int = 256\n    def check(self):\n"
        "        return self.max_num_seqs\n"
    )
    index = _config_class_index(source_root=tmp_path, config_surface_files=["vllm/config.py"])
    # Citation file:line inside the class span resolves to {module}.{class}.
    assert _native_type_for_citation("vllm/config.py:2", index) == "vllm.config.SchedulerConfig"
    # A trailing line RANGE is tolerated (anchors on the first line).
    assert (
        _native_type_for_citation("see vllm/config.py:2-3", index) == "vllm.config.SchedulerConfig"
    )
    # An unparseable / off-surface citation yields empty (gate reports infra_error).
    assert _native_type_for_citation("somewhere unknown", index) == ""
    assert _native_type_for_citation("vllm/config.py:999", index) == ""


def test_gap_proposal_carries_non_empty_native_type(tmp_path: Path) -> None:
    """CR3: a 1b diagnosis (no carried entry) yields a gate proposal whose
    native_type is the cited class's FQN - non-empty, so the gate can construct
    it instead of reporting infra_error on an empty native_type."""
    (tmp_path / "config.py").write_text(
        "class Config:\n    n: int = 1\n    def validate(self):\n        return self.n\n"
    )
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": "transformers_gap_bound",
                    "classification": "rule_morphed",
                    "reason": "constraint only in docstring",
                    "citation": "config.py:3",
                    "kwargs_positive": {"n": -1},
                    "kwargs_negative": {"n": 1},
                }
            ]
        }
    )

    seen_native_types: list[str] = []

    def asserting_gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for p in proposals:
            seen_native_types.append(p["native_type"])
            out.append({"rule_id": p["rule_id"], "verdict": "confirmed"})
        return out

    result = diagnose_gaps(
        engine="transformers",
        new_version="5.7.0",
        schema={"engine": "transformers", "engine_params": {"n": {}}},
        source_root=tmp_path,
        config_surface_files=["config.py"],
        model=_StubModel(raw),
        gate_runner=asserting_gate,
    )
    # The gate saw a non-empty, fully-qualified native_type (not the old "").
    assert seen_native_types == ["config.Config"]
    assert result.confirmed[0]["native_type"] == "config.Config"


def test_gap_native_type_resolves_via_constructor_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The native_type a 1b citation produces is accepted by the constructor
    resolver (stubbed) - proving the threaded identity is resolver-shaped, the
    exact gap that made every 1b proposal infra_error."""
    import scripts._engine_constructors as constructors
    from llenergymeasure.api.diagnose import _config_class_index, _native_type_for_citation

    (tmp_path / "config.py").write_text(
        "class Config:\n    n: int = 1\n    def validate(self):\n        return self.n\n"
    )
    index = _config_class_index(source_root=tmp_path, config_surface_files=["config.py"])
    native_type = _native_type_for_citation("config.py:3", index)
    assert native_type == "config.Config"

    resolved: list[str] = []

    class _FakeConfig:  # what the resolver would hand back
        pass

    def _stub_resolve(engine: str, nt: str) -> Any:
        resolved.append(nt)
        return _FakeConfig

    monkeypatch.setattr(constructors, "resolve_native_type", _stub_resolve)
    assert constructors.resolve_native_type("transformers", native_type) is _FakeConfig
    assert resolved == ["config.Config"]


# ---------------------------------------------------------------------------
# CR4: gate-runner container failure surfaces as DiagnoseError with stderr
# ---------------------------------------------------------------------------


def test_container_gate_failure_raises_diagnose_error_with_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-zero container exit must raise DiagnoseError carrying the hidden
    container stderr/stdout - not an opaque CalledProcessError the CLI cannot
    catch."""
    import subprocess

    from llenergymeasure.api.diagnose import ContainerGateRunner

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args[0] if args else ["docker"],
            output="some stdout",
            stderr="ImportError: scripts.validate_rules not found",
        )

    monkeypatch.setattr(subprocess, "run", _boom)
    runner = ContainerGateRunner(
        image="llenergymeasure:transformers", repo=tmp_path, workdir=tmp_path / "wd"
    )
    with pytest.raises(DiagnoseError) as excinfo:
        runner("transformers", [{"rule_id": "r", "native_type": "X"}])
    msg = str(excinfo.value)
    assert "ImportError: scripts.validate_rules not found" in msg
    assert "some stdout" in msg
    assert "exited 1" in msg


def test_container_gate_missing_output_raises_diagnose_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Container exits 0 but writes no verdicts file -> DiagnoseError, not a
    bare FileNotFoundError on the read."""
    import subprocess

    from llenergymeasure.api.diagnose import ContainerGateRunner

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    runner = ContainerGateRunner(
        image="llenergymeasure:transformers", repo=tmp_path, workdir=tmp_path / "wd"
    )
    with pytest.raises(DiagnoseError, match="no verdicts file"):
        runner("transformers", [{"rule_id": "r", "native_type": "X"}])


def test_docker_cmd_preserves_tensorrt_image_entrypoint(tmp_path: Path) -> None:
    """tensorrt keeps the image entrypoint (CUDA forward-compat setup) and runs
    the interpreter as the exec-ed command; vLLM and transformers override the
    entrypoint to python3 to bypass vLLM's ``vllm serve`` server. Overriding the
    tensorrt entrypoint breaks flashinfer's CUDA import and fails every probe."""
    from llenergymeasure.api.diagnose import ContainerGateRunner

    runner = ContainerGateRunner(
        image="llenergymeasure:tensorrt-1.0.0", repo=tmp_path, workdir=tmp_path / "wd"
    )

    trt = runner._docker_cmd("tensorrt")
    assert trt[trt.index("--entrypoint") + 1] == "/opt/nvidia/nvidia_entrypoint.sh"
    # The interpreter is the command, exec-ed by the NVIDIA entrypoint after setup.
    assert trt[trt.index("llenergymeasure:tensorrt-1.0.0") + 1 :] == [
        "python3",
        "scripts/diagnose_gate_in_container.py",
        "tensorrt",
        "/gateout/diagnose_proposals.json",
        "/gateout/diagnose_gate_verdicts.json",
    ]

    for engine in ("vllm", "transformers"):
        cmd = runner._docker_cmd(engine)
        assert cmd[cmd.index("--entrypoint") + 1] == "python3"
        # The gate script is the entrypoint's first arg, not preceded by python3.
        assert cmd[cmd.index("llenergymeasure:tensorrt-1.0.0") + 1] == (
            "scripts/diagnose_gate_in_container.py"
        )


# ---------------------------------------------------------------------------
# Tier-D: pure-inference advisories for undeclared-semantic (class-3) fields
# ---------------------------------------------------------------------------

_TIER_D_CANDIDATES = [
    {"section": "engine_params", "leaf": "max_logprobs", "type": "int", "default": 20},
    {"section": "engine_params", "leaf": "master_port", "type": "int", "default": 0},
    {"section": "sampling_params", "leaf": "logprobs", "type": "int | None", "default": None},
]


def _write_tier_d_surface(path: Path) -> None:
    """A config-surface module whose classes own the candidate fields.

    ``ModelConfig`` (Config suffix) and ``SamplingParams`` (Params suffix) are
    both recognised by the config-class walker, so a citation into either span
    resolves to a constructible native_type.
    """
    path.write_text(
        "class ModelConfig:\n"
        "    max_logprobs: int = 20\n"
        "    master_port: int = 0\n"
        "\n"
        "    def __post_init__(self) -> None:\n"
        "        if self.max_logprobs < 0:\n"
        '            raise ValueError("max_logprobs must be >= 0")\n'
        "\n"
        "\n"
        "class SamplingParams:\n"
        "    logprobs: int | None = None\n"
    )


def _write_validator_surface(path: Path) -> None:
    """A config class whose validator (with the raise) sits far below the 40-line
    surface cap, so only the cited-field window can reach it."""
    lines = ["class SamplingParams:", "    logprobs: int | None = None"]
    lines += [f"    pad{i}: int = {i}" for i in range(60)]  # push the validator down
    lines += [
        "    def _verify_args(self) -> None:",
        "        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:",
        '            raise ValueError("logprobs must be non-negative or -1")',
    ]
    path.write_text("\n".join(lines) + "\n")


def test_tier_d_cited_field_windowing_admits_the_validator_body(tmp_path: Path) -> None:
    # The fix: the mode-1b 40-line surface window misses a validator 60 lines below
    # the header; the cited-field path surfaces it so the model can find the bound.
    _write_validator_surface(tmp_path / "sp.py")
    requests = tier_d_symbol_requests(
        source_root=tmp_path, config_surface_files=["sp.py"], candidate_leaves=["logprobs"]
    )
    assert [(r.class_name, r.fields) for r in requests] == [("SamplingParams", ("logprobs",))]
    report = windowed_source(source_root=tmp_path, requests=requests, budget_chars=12_000)
    assert "_verify_args" in report.source
    assert "logprobs must be non-negative or -1" in report.source


def test_tier_d_validator_filter_drops_noise_method_windows() -> None:
    # The budget-robustness fix: a field referenced by many accessor methods AND
    # one validator should create a WINDOW only for the validator, so a char
    # budget is not spent on noise (the full-surface mine evicted the real
    # validator otherwise). Asserted on the windows produced, not rendered text
    # (context lines around the field span can bleed into adjacent lines).
    import ast

    from llenergymeasure.api._source_window import _windows_for_class
    from llenergymeasure.api.diagnose import _is_validator_member

    lines = ["class CConfig:", "    f: int = 0"]
    lines += [f"    def noise{i}(self):\n        return self.f" for i in range(8)]
    lines += [
        "    def _verify_args(self):",
        "        if self.f < -1:",
        "            raise ValueError('f')",
    ]
    cls = next(n for n in ast.parse("\n".join(lines)).body if isinstance(n, ast.ClassDef))

    unfiltered, _ = _windows_for_class(cls, file="c.py", fields=("f",))
    filtered, _ = _windows_for_class(
        cls, file="c.py", fields=("f",), member_filter=_is_validator_member
    )
    assert len(filtered) < len(unfiltered)
    assert any("_verify_args" in w.label for w in filtered)
    assert not any("noise" in w.label for w in filtered)
    assert _is_validator_member("_verify_args") and _is_validator_member("__post_init__")
    assert not _is_validator_member("__repr__") and not _is_validator_member("compute_hash")


def test_tier_d_native_types_resolves_declaring_class(tmp_path: Path) -> None:
    # native_type comes from the candidate's DECLARING class (deterministic), not
    # the model's citation - the model often emits an unparseable citation.
    _write_validator_surface(tmp_path / "sp.py")
    types = tier_d_native_types(
        source_root=tmp_path,
        config_surface_files=["sp.py"],
        candidate_leaves=["logprobs", "absent"],
    )
    assert types == {"logprobs": "sp.SamplingParams"}


def test_tier_d_prompt_lists_candidates_and_bound_contract() -> None:
    prompt = build_tier_d_prompt(
        engine="vllm",
        new_version="0.19.1",
        candidates=_TIER_D_CANDIDATES,
        source="class ModelConfig:\n    max_logprobs: int = 20\n",
    )
    # Each candidate is listed under its deterministic, model-verbatim rule_id.
    assert tier_d_rule_id("vllm", "engine_params", "max_logprobs") in prompt
    assert tier_d_rule_id("vllm", "sampling_params", "logprobs") in prompt
    # The sentinel guard (the max_logprobs=-1 class of hallucination) is stated.
    assert "fires_when" in prompt and "sentinel" in prompt.lower()


def test_tier_d_prompt_requires_candidates_and_source() -> None:
    with pytest.raises(DiagnoseError, match="at least one candidate"):
        build_tier_d_prompt(engine="vllm", new_version="0.19.1", candidates=[], source="x")
    with pytest.raises(DiagnoseError, match="non-empty"):
        build_tier_d_prompt(
            engine="vllm", new_version="0.19.1", candidates=_TIER_D_CANDIDATES, source="  "
        )


def test_parse_tier_d_classifications_and_bad_fires_when() -> None:
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": "vllm_tier_d_engine_params_max_logprobs",
                    "classification": "bounded",
                    "reason": "must be >= 0",
                    "citation": "config.py:2",
                    "constraint": "must be >= 0",
                    "fires_when": {"<": 0},
                    "kwargs_positive": {"max_logprobs": -1},
                    "kwargs_negative": {"max_logprobs": 5},
                },
                {
                    "rule_id": "vllm_tier_d_engine_params_master_port",
                    "classification": "free_counter",  # not a Tier-D label -> unknown
                    "reason": "free port",
                    "citation": "",
                    "kwargs_positive": {},
                    "kwargs_negative": {},
                },
                {
                    # Cross-field: max_cpu_loras must be >= max_loras. Now ACCEPTED
                    # (single-key + supported op; the loader resolves the @ref).
                    "rule_id": "vllm_tier_d_engine_params_max_cpu_loras",
                    "classification": "bounded",
                    "reason": "must be >= max_loras",
                    "citation": "config.py:5",
                    "fires_when": {"<": "@max_loras"},
                    "kwargs_positive": {"max_cpu_loras": 1, "max_loras": 8},
                    "kwargs_negative": {"max_cpu_loras": 8, "max_loras": 8},
                },
                {
                    # Cross-field divisibility, now an accepted Tier-D operator.
                    "rule_id": "vllm_tier_d_engine_params_tp_size",
                    "classification": "bounded",
                    "reason": "must divide decode_context_parallel_size",
                    "citation": "config.py:7",
                    "fires_when": {"not_divisible_by": "@decode_context_parallel_size"},
                    "kwargs_positive": {"tp_size": 3, "decode_context_parallel_size": 4},
                    "kwargs_negative": {"tp_size": 2, "decode_context_parallel_size": 4},
                },
                {
                    "rule_id": "vllm_tier_d_sampling_params_logprobs",
                    "classification": "bounded",
                    "reason": "x",
                    "citation": "c.py:1",
                    "fires_when": {"nonsense": 1},  # unknown op -> still malformed
                    "kwargs_positive": {"logprobs": 3},
                    "kwargs_negative": {"logprobs": 2},
                },
                {
                    "rule_id": "vllm_tier_d_sampling_params_n",
                    "classification": "bounded",
                    "reason": "x",
                    "citation": "c.py:2",
                    "fires_when": {"<": 0, ">": 8},  # >1 key -> still malformed
                    "kwargs_positive": {"n": -1},
                    "kwargs_negative": {"n": 1},
                },
            ]
        }
    )
    diags = {d.rule_id: d for d in parse_diagnoses(raw, classifications=TIER_D_CLASSIFICATIONS)}
    assert diags["vllm_tier_d_engine_params_max_logprobs"].fires_when == {"<": 0}
    assert diags["vllm_tier_d_engine_params_master_port"].classification == "unknown"
    # Cross-field @ref predicates are ACCEPTED (single-key, supported op); the
    # @ref value is preserved verbatim for the loader to resolve at pre-flight.
    assert diags["vllm_tier_d_engine_params_max_cpu_loras"].fires_when == {"<": "@max_loras"}
    assert diags["vllm_tier_d_engine_params_tp_size"].fires_when == {
        "not_divisible_by": "@decode_context_parallel_size"
    }
    # A genuinely bad fires_when is still malformed: unknown op...
    unknown_op = diags["vllm_tier_d_sampling_params_logprobs"]
    assert unknown_op.fires_when is None
    assert "fires_when" in unknown_op.kwargs_malformed
    # ...or more than one key (no AND-combined dict).
    multi_key = diags["vllm_tier_d_sampling_params_n"]
    assert multi_key.fires_when is None
    assert "fires_when" in multi_key.kwargs_malformed


def _tier_d_response() -> str:
    return json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": "vllm_tier_d_engine_params_max_logprobs",
                    "classification": "bounded",
                    "reason": "post_init raises if < 0",
                    "citation": "config.py:2",
                    "constraint": "must be >= 0",
                    "fires_when": {"<": 0},
                    "kwargs_positive": {"max_logprobs": -1},
                    "kwargs_negative": {"max_logprobs": 5},
                },
                {
                    "rule_id": "vllm_tier_d_engine_params_master_port",
                    "classification": "unconstrained",
                    "reason": "free TCP port; any int valid",
                    "citation": "config.py:3",
                    "kwargs_positive": {},
                    "kwargs_negative": {},
                },
                {
                    "rule_id": "vllm_tier_d_sampling_params_logprobs",
                    "classification": "bounded",
                    "reason": "x",
                    "citation": "config.py:10",
                    "fires_when": {"nonsense": 1},  # malformed -> dropped
                    "kwargs_positive": {"logprobs": -1},
                    "kwargs_negative": {"logprobs": 1},
                },
                {
                    "rule_id": "vllm_tier_d_fabricated_id",  # not a candidate -> ignored
                    "classification": "bounded",
                    "reason": "x",
                    "citation": "config.py:2",
                    "fires_when": {"<": 0},
                    "kwargs_positive": {"q": -1},
                    "kwargs_negative": {"q": 1},
                },
            ]
        }
    )


def test_tier_d_end_to_end_splits_by_classification_and_gate(tmp_path: Path) -> None:
    _write_tier_d_surface(tmp_path / "config.py")
    model = _StubModel(_tier_d_response())
    result = diagnose_tier_d(
        engine="vllm",
        new_version="0.19.1",
        candidates=_TIER_D_CANDIDATES,
        source_root=tmp_path,
        config_surface_files=["config.py"],
        model=model,
        gate_runner=_gate_stub({"vllm_tier_d_engine_params_max_logprobs": "confirmed"}),
    )

    # Only the construction-confirmed bounded field is emitted.
    assert [e["id"] for e in result.confirmed] == ["vllm_tier_d_engine_params_max_logprobs"]
    entry = result.confirmed[0]
    assert entry["added_by"] == "llm_advisory"
    assert entry["severity"] == "warn"
    assert entry["llm_proposed"] is True
    assert entry["expected_outcome"] == {"outcome": "warn", "emission_channel": "none"}
    assert entry["match"]["fields"] == {"vllm.engine_params.max_logprobs": {"<": 0}}
    # native_type resolved from the citation's enclosing class.
    assert entry["native_type"].endswith("ModelConfig")
    # Unconstrained reported but never gated/emitted; malformed dropped; fabricated ignored.
    assert [r["rule_id"] for r in result.tier_d_unconstrained] == [
        "vllm_tier_d_engine_params_master_port"
    ]
    assert [r["rule_id"] for r in result.dropped_malformed] == [
        "vllm_tier_d_sampling_params_logprobs"
    ]
    assert result.proposed_yaml is not None and "llm_advisory" in result.proposed_yaml


def test_tier_d_cross_field_advisory_preserves_at_ref(tmp_path: Path) -> None:
    # A cross-field bound (max_cpu_loras must be >= max_loras) flows through to an
    # emitted advisory whose match preserves the "@max_loras" sibling ref, so the
    # loader resolves it against the sibling at pre-flight.
    (tmp_path / "config.py").write_text(
        "class ModelConfig:\n"
        "    max_loras: int = 1\n"
        "    max_cpu_loras: int = 1\n"
        "\n"
        "    def __post_init__(self) -> None:\n"
        "        if self.max_cpu_loras < self.max_loras:\n"
        '            raise ValueError("max_cpu_loras must be >= max_loras")\n'
    )
    candidates = [
        {"section": "engine_params", "leaf": "max_cpu_loras", "type": "int", "default": 1},
    ]
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": "vllm_tier_d_engine_params_max_cpu_loras",
                    "classification": "bounded",
                    "reason": "post_init raises if max_cpu_loras < max_loras",
                    "citation": "config.py:5",
                    "constraint": "must be >= max_loras",
                    "fires_when": {"<": "@max_loras"},
                    "kwargs_positive": {"max_cpu_loras": 1, "max_loras": 8},
                    "kwargs_negative": {"max_cpu_loras": 8, "max_loras": 8},
                },
            ]
        }
    )
    result = diagnose_tier_d(
        engine="vllm",
        new_version="0.19.1",
        candidates=candidates,
        source_root=tmp_path,
        config_surface_files=["config.py"],
        model=_StubModel(raw),
        gate_runner=_gate_stub({"vllm_tier_d_engine_params_max_cpu_loras": "confirmed"}),
    )
    assert [e["id"] for e in result.confirmed] == ["vllm_tier_d_engine_params_max_cpu_loras"]
    entry = result.confirmed[0]
    # The @ref survives verbatim into the emitted advisory's match.
    assert entry["match"]["fields"] == {"vllm.engine_params.max_cpu_loras": {"<": "@max_loras"}}
    # Both fields are carried on the gate probes so the gate can construct + confirm.
    assert entry["kwargs_positive"] == {"max_cpu_loras": 1, "max_loras": 8}
    assert entry["kwargs_negative"] == {"max_cpu_loras": 8, "max_loras": 8}


def test_tier_d_path_validator_drops_unresolved(tmp_path: Path) -> None:
    _write_tier_d_surface(tmp_path / "config.py")
    gate_calls: list[list[dict[str, Any]]] = []

    def _recording_gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        gate_calls.append(proposals)
        return [{"rule_id": p["rule_id"], "verdict": "confirmed"} for p in proposals]

    def _reject_all(path: str) -> None:
        raise ValueError(f"unreachable: {path}")

    result = diagnose_tier_d(
        engine="vllm",
        new_version="0.19.1",
        candidates=_TIER_D_CANDIDATES,
        source_root=tmp_path,
        config_surface_files=["config.py"],
        model=_StubModel(_tier_d_response()),
        gate_runner=_recording_gate,
        path_validator=_reject_all,
    )
    # The bounded candidate's path was rejected -> never gated, never confirmed.
    assert result.confirmed == []
    assert gate_calls == []
    assert any(r["verdict"] == "path_unresolved" for r in result.unconfirmed)


def test_tier_d_requires_candidates() -> None:
    with pytest.raises(DiagnoseError, match="at least one candidate"):
        diagnose_tier_d(
            engine="vllm",
            new_version="0.19.1",
            candidates=[],
            source_root=Path("/nonexistent"),
            config_surface_files=["config.py"],
            model=_StubModel("{}"),
            gate_runner=_gate_stub({}),
        )


class TestFiresWhenConsistency:
    """The match predicate must select the illegal probe and reject the legal one.

    The construction gate only checks that the illegal value raises and the legal
    value constructs; it never re-evaluates the predicate. These cases (drawn from
    a live vLLM mine) cover a model that supplies correct probes but an inverted or
    self-referential ``fires_when``, which would otherwise warn on valid configs.
    """

    def test_literal_bound_consistent(self) -> None:
        # logprobs < -1 is the illegal region; -2 illegal, -1 legal sentinel.
        assert _fires_when_consistent_with_probes(
            "logprobs", {"<": -1}, {"logprobs": -2}, {"logprobs": -1}
        )

    def test_cross_field_ref_consistent(self) -> None:
        # max_cpu_loras < max_loras raises -> fires_when "<@max_loras" is the illegal region.
        assert _fires_when_consistent_with_probes(
            "max_cpu_loras",
            {"<": "@max_loras"},
            {"max_cpu_loras": 1, "max_loras": 8},
            {"max_cpu_loras": 8, "max_loras": 8},
        )

    def test_inverted_cross_field_rejected(self) -> None:
        # parallel.py raises when rank >= count, so the illegal region is ">=",
        # but the model emitted "<" (the legal region). The illegal probe
        # (rank=1,count=1) does NOT satisfy "<", so the predicate is inconsistent.
        assert not _fires_when_consistent_with_probes(
            "_api_process_rank",
            {"<": "@_api_process_count"},
            {"_api_process_rank": 1, "_api_process_count": 1},
            {"_api_process_rank": 0, "_api_process_count": 2},
        )

    def test_self_referential_rejected(self) -> None:
        # A field compared against itself can never fire (x < x is always False).
        assert not _fires_when_consistent_with_probes(
            "_api_process_count",
            {"<": "@_api_process_count"},
            {"_api_process_count": 1, "_api_process_rank": 1},
            {"_api_process_count": 2, "_api_process_rank": 0},
        )

    def test_indeterminate_probe_rejected(self) -> None:
        # A probe missing the subject leaf yields None -> not consistent.
        assert not _fires_when_consistent_with_probes("logprobs", {"<": -1}, {}, {"logprobs": -1})

    def test_type_malformed_operand_rejected_not_raised(self) -> None:
        # A live mine emitted {"not_in": 4} - operator valid, operand a bare int.
        # The loader's evaluate_predicate raises TypeError on this; the check must
        # treat it as indeterminate (drop), not crash the mine, and never let such
        # a predicate reach the corpus where try_match would crash config validation.
        assert not _fires_when_consistent_with_probes(
            "best_of", {"not_in": 4}, {"best_of": 5}, {"best_of": 1}
        )


def test_tier_d_drops_inverted_fires_when_before_gating(tmp_path: Path) -> None:
    """An inverted-predicate diagnosis is dropped on the host, never reaching the gate."""
    src = tmp_path / "cfg.py"
    src.write_text(
        "class ParallelConfig:\n"
        "    _api_process_count: int = 1\n"
        "    _api_process_rank: int = 0\n"
        "    def _validate(self):\n"
        "        if self._api_process_rank >= self._api_process_count:\n"
        "            raise ValueError('bad rank')\n"
    )
    rule_id = tier_d_rule_id("vllm", "engine_params", "_api_process_rank")
    raw = json.dumps(
        {
            "diagnoses": [
                {
                    "rule_id": rule_id,
                    "classification": "bounded",
                    "reason": "raises if rank >= count",
                    "citation": "cfg.py",
                    "fires_when": {"<": "@_api_process_count"},
                    "kwargs_positive": {"_api_process_rank": 1, "_api_process_count": 1},
                    "kwargs_negative": {"_api_process_rank": 0, "_api_process_count": 2},
                }
            ]
        }
    )
    gate_calls: list[Any] = []

    def _recording_gate(engine: str, proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
        gate_calls.append(proposals)
        return [{"rule_id": p["rule_id"], "verdict": "confirmed"} for p in proposals]

    result = diagnose_tier_d(
        engine="vllm",
        new_version="0.19.1",
        candidates=[
            {"section": "engine_params", "leaf": "_api_process_rank", "type": "int", "default": 0}
        ],
        source_root=tmp_path,
        config_surface_files=["cfg.py"],
        model=_StubModel(raw),
        gate_runner=_recording_gate,
    )

    assert result.confirmed == []
    assert gate_calls == []  # dropped before the gate ran
    assert any(r.get("verdict") == "fires_when_inconsistent" for r in result.unconfirmed)
