"""Host-side tests for scripts/probe_candidates.py (verification-ladder tiers 2-3).

These run on plain CPU with no engine installed: the engine boundary
(``_engine_constructors``) is faked, so every test exercises the kernel's own
logic - deterministic probe-value derivation, kwargs assembly, verdict
classification, schema parsing, and writeback - never a real container.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

import _engine_constructors as ec
import probe_candidates as pc

_NO_REFS: set[str] = set()


def _REF_OF(leaf: str) -> int:
    """A fixed @field_ref resolver for unit derivation (every ref -> 4)."""
    return 4


def _fire(spec: Any, *, leaf: str = "f", referenced: set[str] | None = None) -> Any:
    return pc._fire_value(spec, leaf, _REF_OF, referenced or _NO_REFS)


def _nofire(spec: Any, *, leaf: str = "f", referenced: set[str] | None = None) -> Any:
    return pc._nofire_value(spec, leaf, _REF_OF, referenced or _NO_REFS)


# ---------------------------------------------------------------------------
# Fire / nofire derivation, per operator in the shipped corpus vocabulary
# ---------------------------------------------------------------------------


def test_threshold_less_than() -> None:
    assert _fire({"<": 1}) == 0
    assert _nofire({"<": 1}) == 1


def test_threshold_less_equal_float() -> None:
    assert _fire({"<=": 0.0}) == 0.0
    assert _nofire({"<=": 0.0}) == 0.5


def test_threshold_greater_than_float() -> None:
    assert _fire({">": 2.0}) == 2.5
    assert _nofire({">": 2.0}) == 2.0


def test_threshold_greater_equal_int() -> None:
    assert _fire({">=": 4}) == 4
    assert _nofire({">=": 4}) == 3


def test_bare_value_equality() -> None:
    assert _fire(1) == 1
    assert _nofire(1) == 2
    assert _fire(True) is True
    assert _nofire(True) is False


def test_not_equal_with_present() -> None:
    assert _fire({"!=": 0.0, "present": True}) == 1.0
    assert _nofire({"!=": 0.0, "present": True}) == 0.0


def test_in_membership() -> None:
    assert _fire({"in": ["pplx", "naive"]}) == "pplx"
    assert _nofire({"in": ["pplx", "naive"]}) == "__llem_probe_nonmember__"


def test_not_in_numeric_members() -> None:
    assert _fire({"present": True, "not_in": [1, 16]}) == 17
    assert _nofire({"present": True, "not_in": [1, 16]}) == 1


def test_not_in_mixed_members_uses_first_non_null() -> None:
    spec = {"not_in": [None, True, False, "never"]}
    assert _fire(spec) == "__llem_probe_nonmember__"
    assert _nofire(spec) is True  # first non-null member satisfies the claim


def test_type_is_not_scalar() -> None:
    assert _fire({"type_is_not": "int"}) == "x"
    assert _nofire({"type_is_not": "int"}) == 1


def test_type_is_not_list_picks_value_outside_all() -> None:
    spec = {"present": True, "type_is_not": ["bool", "int", "str"]}
    assert _fire(spec) == 1.0  # float is outside {bool, int, str}
    assert _nofire(spec) is True  # bool is inside the allowlist


def test_type_is_not_config_class_nofire_is_unprobeable() -> None:
    spec = {"type_is_not": "CompileConfig", "present": True}
    assert _fire(spec) == "x"  # any non-CompileConfig fires
    with pytest.raises(pc.Unprobeable):
        _nofire(spec)  # cannot synthesise a CompileConfig satisfying value


def test_present_only_flag() -> None:
    assert _fire({"present": True}) is True
    assert _nofire({"present": True}) is pc._OMIT


def test_present_only_when_referenced_uses_ref_base() -> None:
    assert _fire({"present": True}, leaf="max_tokens", referenced={"max_tokens"}) == pc.REF_BASE


def test_absent_flag() -> None:
    assert _fire({"absent": True}) is pc._OMIT
    assert _nofire({"absent": True}) is True


def test_not_divisible_by_ref() -> None:
    assert _fire({"not_divisible_by": "@x"}) == 5  # REF_BASE + 1, not a multiple of 4
    assert _nofire({"not_divisible_by": "@x"}) == 8  # 2 * REF_BASE, a clean multiple


def test_threshold_against_field_ref() -> None:
    assert _fire({">": "@max_tokens"}) == 5  # REF_BASE + step
    assert _nofire({">": "@max_tokens"}) == 4  # the referenced base itself


def test_equality_against_field_ref_resolves_not_literal() -> None:
    # The fire value must be the REFERENCED FIELD'S value, never the "@..." string.
    assert _fire({"==": "@max_tokens"}) == 4
    assert _nofire({"==": "@max_tokens"}) == 5  # contrast of the resolved value
    assert _fire({"!=": "@max_tokens"}) == 5
    assert _nofire({"!=": "@max_tokens"}) == 4


# ---------------------------------------------------------------------------
# Construction-probe kwargs assembly
# ---------------------------------------------------------------------------


def test_single_field_construction_kwargs() -> None:
    violating, satisfying, subject = pc.build_construction_kwargs(
        {"vllm.sampling_params.n": {"<": 1}}
    )
    assert violating == {"n": 0}
    assert satisfying == {"n": 1}
    assert subject == "n"


def test_cross_field_ref_scaffolds_unnamed_operand() -> None:
    fields = {"vllm.engine_params.data_parallel_size_local": {">": "@data_parallel_size"}}
    violating, satisfying, subject = pc.build_construction_kwargs(fields)
    assert violating == {"data_parallel_size_local": 5, "data_parallel_size": 4}
    assert satisfying == {"data_parallel_size_local": 4, "data_parallel_size": 4}
    assert subject == "data_parallel_size_local"


def test_cross_field_named_present_operand() -> None:
    fields = {
        "vllm.sampling_params.max_tokens": {"present": True},
        "vllm.sampling_params.min_tokens": {">": "@max_tokens"},
    }
    violating, satisfying, _ = pc.build_construction_kwargs(fields)
    assert violating == {"max_tokens": 4, "min_tokens": 5}
    assert satisfying == {"max_tokens": 4, "min_tokens": 4}


def test_cross_field_equality_kwargs_use_resolved_operand() -> None:
    fields = {
        "vllm.engine_params.data_parallel_size": {">": 1},
        "vllm.engine_params.data_parallel_size_local": {"==": "@data_parallel_size"},
    }
    violating, satisfying, subject = pc.build_construction_kwargs(fields)
    assert violating == {"data_parallel_size": 2, "data_parallel_size_local": 2}
    assert satisfying == {"data_parallel_size": 2, "data_parallel_size_local": 3}
    assert subject == "data_parallel_size_local"


def test_absent_precondition_flips_subject_to_set() -> None:
    fields = {
        "vllm.engine_params.max_num_batched_tokens": {"<": "@max_model_len"},
        "vllm.engine_params.enable_chunked_prefill": {"absent": True},
    }
    violating, satisfying, subject = pc.build_construction_kwargs(fields)
    assert violating == {"max_num_batched_tokens": 3, "max_model_len": 4}
    assert satisfying == {
        "max_num_batched_tokens": 3,
        "max_model_len": 4,
        "enable_chunked_prefill": True,
    }
    assert subject == "enable_chunked_prefill"


# ---------------------------------------------------------------------------
# Identity-probe value derivation
# ---------------------------------------------------------------------------


def _dormant(
    fields: dict[str, Any], normalised: list[str] | None = None, **prov: Any
) -> pc.Candidate:
    return pc.Candidate(
        id="d",
        engine="vllm",
        severity="dormant",
        fields=fields,
        normalised=normalised or [],
        source=prov.get("source"),
        declared_values=prov.get("declared_values"),
        raw={},
    )


def test_identity_bare_value_contrasts_with_default() -> None:
    cand = _dormant({"vllm.sampling_params.seed": -1})
    values = pc.identity_values(cand, "vllm.sampling_params.seed")
    assert values == [-1, pc._DEFAULT]


def test_identity_membership_values() -> None:
    cand = _dormant({"vllm.engine_params.all2all_backend": {"in": ["pplx", "naive"]}})
    assert pc.identity_values(cand, "vllm.engine_params.all2all_backend") == ["pplx", "naive"]


def test_identity_not_in_numeric_two_nonmembers() -> None:
    cand = _dormant({"f": {"present": True, "not_in": [0, 50256]}})
    assert pc.identity_values(cand, "f") == [50257, 50258]


def test_identity_threshold_two_firing_values() -> None:
    cand = _dormant({"f": {"<": 0}})
    assert pc.identity_values(cand, "f") == [-1, -2]


def test_identity_not_equal_two_firing_values() -> None:
    cand = _dormant({"f": {"!=": 0.0, "present": True}})
    assert pc.identity_values(cand, "f") == [0.5, 1.0]


def test_identity_present_only_is_unprobeable() -> None:
    cand = _dormant(
        {
            "transformers.sampling_params.do_sample": False,
            "transformers.sampling_params.temperature": {"present": True},
        }
    )
    with pytest.raises(pc.Unprobeable):
        pc.identity_values(cand, "transformers.sampling_params.temperature")


def test_identity_observed_collision_uses_declared_values() -> None:
    cand = _dormant(
        {"vllm.engine_params.enforce_eager": {"present": True}},
        source="observed_collision",
        declared_values=[True, False],
    )
    assert pc.identity_values(cand, "vllm.engine_params.enforce_eager") == [True, False]


def test_identity_observed_collision_absent_sentinel_becomes_default() -> None:
    cand = _dormant(
        {"f": {"present": True}},
        source="observed_collision",
        declared_values=["<absent>", 7],
    )
    assert pc.identity_values(cand, "f") == [pc._DEFAULT, 7]


# ---------------------------------------------------------------------------
# Verdict classification (engine boundary faked)
# ---------------------------------------------------------------------------


class _FakeCls:
    pass


@pytest.fixture
def fake_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec, "engine_importable", lambda engine: True)
    monkeypatch.setattr(ec, "candidate_classes", lambda engine, path, leaves: [_FakeCls])
    monkeypatch.setattr(ec, "accepts", lambda cls, leaf: True)


def _error_cand(fields: dict[str, Any]) -> pc.Candidate:
    return pc.Candidate(
        id="e",
        engine="vllm",
        severity="error",
        fields=fields,
        normalised=[],
        source=None,
        declared_values=None,
        raw={},
    )


def test_construction_confirmed(monkeypatch: pytest.MonkeyPatch, fake_engine: None) -> None:
    def construct(engine: str, cls: type, kwargs: dict[str, Any], *, validate: bool) -> Any:
        if kwargs.get("n", 99) < 1:  # the violating leg raises, the satisfying leg builds
            raise ValueError("n must be at least 1")
        return object()

    monkeypatch.setattr(ec, "construct", construct)
    verdict = pc.probe_candidate(_error_cand({"vllm.sampling_params.n": {"<": 1}}), "vllm", True)
    assert verdict.status == "confirmed"
    assert verdict.tier == "construction"


def test_construction_unconfirmed_when_violating_builds(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    # Nothing raises at construction: that does NOT contradict the claim (the
    # engine may validate at init or generation), so the verdict is unconfirmed.
    monkeypatch.setattr(ec, "construct", lambda *a, **k: object())
    verdict = pc.probe_candidate(_error_cand({"vllm.sampling_params.n": {"<": 1}}), "vllm", True)
    assert verdict.status == "unconfirmed"
    assert "construction grain" in verdict.evidence


def test_construction_infra_error_when_satisfying_also_raises(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    def construct(*a: Any, **k: Any) -> Any:
        raise RuntimeError("unrelated boom")

    monkeypatch.setattr(ec, "construct", construct)
    verdict = pc.probe_candidate(_error_cand({"vllm.sampling_params.n": {"<": 1}}), "vllm", True)
    assert verdict.status == "infra_error"


def test_construction_unprobeable_when_no_satisfying_value(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    # type_is_not a config class: no synthesisable satisfying value -> unprobeable,
    # decided before any construct call.
    monkeypatch.setattr(ec, "construct", lambda *a, **k: pytest.fail("construct should not run"))
    fields = {
        "transformers.sampling_params.compile_config": {
            "type_is_not": "CompileConfig",
            "present": True,
        }
    }
    verdict = pc.probe_candidate(_error_cand(fields), "transformers", True)
    assert verdict.status == "unprobeable"


def test_identity_confirmed_when_resolved_state_identical(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    monkeypatch.setattr(ec, "construct", lambda *a, **k: object())
    monkeypatch.setattr(ec, "resolved_value", lambda obj, leaf: None)  # every value aliases to None
    cand = _dormant({"vllm.sampling_params.seed": -1}, source=None)
    verdict = pc.probe_candidate(cand, "vllm", True)
    assert verdict.status == "confirmed"
    assert verdict.tier == "identity"


def test_identity_unconfirmed_when_resolved_state_differs(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    class Obj:
        def __init__(self, kwargs: dict[str, Any]) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(ec, "construct", lambda engine, cls, kwargs, *, validate: Obj(kwargs))
    monkeypatch.setattr(ec, "resolved_value", lambda obj, leaf: obj.kwargs.get(leaf, "<unset>"))
    cand = _dormant({"vllm.sampling_params.seed": -1})
    verdict = pc.probe_candidate(cand, "vllm", True)
    # Differing constructor-grain state does not disprove effective-config
    # identity at the grain the collision was observed - unconfirmed, not refuted.
    assert verdict.status == "unconfirmed"
    assert "differ" in verdict.evidence


def test_identity_infra_error_when_a_leg_raises(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    def construct(engine: str, cls: type, kwargs: dict[str, Any], *, validate: bool) -> Any:
        if "seed" not in kwargs:  # the default leg blows up
            raise ValueError("seed required")
        return object()

    monkeypatch.setattr(ec, "construct", construct)
    monkeypatch.setattr(ec, "resolved_value", lambda obj, leaf: 0)
    cand = _dormant({"vllm.sampling_params.seed": -1})
    verdict = pc.probe_candidate(cand, "vllm", True)
    assert verdict.status == "infra_error"


def test_inert_subjects_are_normalised_fields_outside_match() -> None:
    inert = _dormant({"transformers.engine_params.num_beams": 1}, normalised=["early_stopping"])
    assert pc._inert_subjects(inert) == ["early_stopping"]
    alias = _dormant({"vllm.sampling_params.seed": -1}, normalised=["seed"])
    assert pc._inert_subjects(alias) == []  # normalises the match field: value-alias


class _Kwargs:
    def __init__(self, kwargs: dict[str, Any]) -> None:
        self.kwargs = kwargs


def _inert_cand(normalised: list[str]) -> pc.Candidate:
    return _dormant({"transformers.engine_params.num_beams": 1}, normalised=normalised)


def test_condition_inert_confirmed_when_engine_rewrites_to_default(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    monkeypatch.setattr(ec, "construct", lambda engine, cls, kwargs, *, validate: _Kwargs(kwargs))
    # The engine normalises early_stopping to False whatever was passed.
    monkeypatch.setattr(
        ec, "resolved_value", lambda obj, leaf: False if leaf == "early_stopping" else 1
    )
    verdict = pc.probe_candidate(_inert_cand(["early_stopping"]), "transformers", True)
    assert verdict.status == "confirmed"
    assert "collapsed" in verdict.evidence


def test_condition_inert_unconfirmed_when_constructor_keeps_value(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    monkeypatch.setattr(ec, "construct", lambda engine, cls, kwargs, *, validate: _Kwargs(kwargs))
    # The constructor stores whatever was passed (default False when omitted).
    monkeypatch.setattr(ec, "resolved_value", lambda obj, leaf: obj.kwargs.get(leaf, False))
    verdict = pc.probe_candidate(_inert_cand(["early_stopping"]), "transformers", True)
    assert verdict.status == "unconfirmed"
    assert "kept True" in verdict.evidence


def test_condition_inert_multiple_subjects_all_must_collapse(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    monkeypatch.setattr(ec, "construct", lambda engine, cls, kwargs, *, validate: _Kwargs(kwargs))

    def resolved(obj: _Kwargs, leaf: str) -> Any:
        if leaf == "early_stopping":  # rewritten to its default
            return False
        return obj.kwargs.get(leaf, 1.0)  # length_penalty stored verbatim

    monkeypatch.setattr(ec, "resolved_value", resolved)
    verdict = pc.probe_candidate(
        _inert_cand(["early_stopping", "length_penalty"]), "transformers", True
    )
    assert verdict.status == "unconfirmed"
    assert "early_stopping" in verdict.evidence  # per-subject evidence, both named
    assert "length_penalty" in verdict.evidence


def test_condition_inert_unprobeable_when_default_has_no_contrast(
    monkeypatch: pytest.MonkeyPatch, fake_engine: None
) -> None:
    monkeypatch.setattr(ec, "construct", lambda engine, cls, kwargs, *, validate: _Kwargs(kwargs))
    monkeypatch.setattr(ec, "resolved_value", lambda obj, leaf: None)  # no contrast for None
    verdict = pc.probe_candidate(_inert_cand(["early_stopping"]), "transformers", True)
    assert verdict.status == "unprobeable"


def test_engine_not_importable_is_infra_error() -> None:
    cand = _error_cand({"vllm.sampling_params.n": {"<": 1}})
    verdict = pc.probe_candidate(cand, "vllm", engine_ready=False)
    assert verdict.status == "infra_error"


def test_unknown_severity_is_unprobeable() -> None:
    cand = pc.Candidate(
        id="x",
        engine="vllm",
        severity="warn",
        fields={"a": 1},
        normalised=[],
        source=None,
        declared_values=None,
        raw={},
    )
    verdict = pc.probe_candidate(cand, "vllm", True)
    assert verdict.status == "unprobeable"


# ---------------------------------------------------------------------------
# Schema parsing + verdict writeback
# ---------------------------------------------------------------------------


def test_parse_candidate_reads_unified_and_shipped_shapes() -> None:
    raw = {
        "id": "r1",
        "engine": "vllm",
        "severity": "error",
        "match": {"fields": {"vllm.sampling_params.n": {"<": 1}}},
        "provenance": {"source": "manual"},
    }
    cand = pc.parse_candidate(raw, 0)
    assert cand.id == "r1"
    assert cand.severity == "error"
    assert list(cand.fields) == ["vllm.sampling_params.n"]
    assert cand.source == "manual"


def test_load_document_accepts_rules_or_candidates_key(tmp_path: Path) -> None:
    rules_file = tmp_path / "rules.yaml"
    rules_file.write_text(
        "engine: vllm\nrules:\n- id: a\n  severity: error\n  match:\n    fields:\n      vllm.sampling_params.n: {'<': 1}\n"
    )
    document, entries = pc.load_document(rules_file)
    assert document["engine"] == "vllm"
    assert len(entries) == 1


def test_load_document_rejects_missing_list(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("engine: vllm\n")
    with pytest.raises(ValueError, match="candidates' or 'rules'"):
        pc.load_document(bad)


def test_write_verdicts_roundtrip(tmp_path: Path) -> None:
    document = {
        "engine": "vllm",
        "candidates": [{"id": "a", "severity": "error", "match": {"fields": {}}}],
    }
    entries = document["candidates"]
    verdicts = [pc.Verdict("confirmed", "construction", "violating raised; satisfying built")]
    out = tmp_path / "verdicts.yaml"
    pc.write_verdicts(document, entries, verdicts, "2026-07-06", out)

    reloaded = yaml.safe_load(out.read_text())
    verdict = reloaded["candidates"][0]["verdict"]
    assert verdict == {
        "status": "confirmed",
        "tier": "construction",
        "date": "2026-07-06",
        "evidence": "violating raised; satisfying built",
    }


def test_main_writes_to_out_and_returns_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(ec, "engine_importable", lambda engine: True)
    monkeypatch.setattr(ec, "candidate_classes", lambda engine, path, leaves: [_FakeCls])
    monkeypatch.setattr(ec, "accepts", lambda cls, leaf: True)

    def construct(engine: str, cls: type, kwargs: dict[str, Any], *, validate: bool) -> Any:
        if kwargs.get("n", 99) < 1:
            raise ValueError("n must be at least 1")
        return object()

    monkeypatch.setattr(ec, "construct", construct)

    candidates = tmp_path / "cands.yaml"
    candidates.write_text(
        "engine: vllm\ncandidates:\n- id: n_rule\n  engine: vllm\n  severity: error\n"
        "  match:\n    fields:\n      vllm.sampling_params.n: {'<': 1}\n"
    )
    out = tmp_path / "out.yaml"
    code = pc.main(
        [
            "--engine",
            "vllm",
            "--candidates",
            str(candidates),
            "--out",
            str(out),
            "--date",
            "2026-07-06",
        ]
    )
    assert code == 0
    written = yaml.safe_load(out.read_text())
    assert written["candidates"][0]["verdict"]["status"] == "confirmed"


def test_main_unparseable_entry_appears_in_summary_table(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(ec, "engine_importable", lambda engine: False)
    candidates = tmp_path / "cands.yaml"
    candidates.write_text(
        "engine: vllm\ncandidates:\n- not_a_mapping\n"
        "- id: ok\n  severity: error\n  match:\n    fields:\n      vllm.sampling_params.n: 1\n"
    )
    code = pc.main(["--engine", "vllm", "--candidates", str(candidates), "--date", "2026-07-06"])
    assert code == 0
    out = capsys.readouterr().out
    assert "#0" in out  # the unparseable entry is a visible table row, not silence
    assert "ok" in out
    written = yaml.safe_load(candidates.read_text())
    assert written["candidates"][0] == "not_a_mapping"  # preserved, no verdict to attach
    assert written["candidates"][1]["verdict"]["status"] == "infra_error"


def test_main_hard_error_on_unreadable_returns_two(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    code = pc.main(["--engine", "vllm", "--candidates", str(missing)])
    assert code == 2
