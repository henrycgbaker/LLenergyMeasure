"""Host-side tests for scripts/absorb.py (the absorb conductor).

No model, no network, no container: the analyst/interrogation boundary is a stub
Generator and the probe is monkeypatched, so every test exercises absorb's own
logic - pool union dedup and verdict memory, canonical field translation, the
recall-interrogation prompt shape and ABSENT handling, the promotion matrix,
byte-stable corpus regeneration, the infra_error ladder gate, and that --dry-run
and the skip flags write nothing they should not.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

import absorb as ab

# --- Tiny builders ---


class _Reply:
    def __init__(self, text: str) -> None:
        self.text = text


def _verdict(status: str) -> dict[str, Any]:
    return {"status": status, "tier": None, "date": "2026-07-07", "evidence": None}


def _rule(rid: str, severity: str, fields: dict[str, Any], **extra: Any) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "id": rid,
        "engine": "vllm",
        "severity": severity,
        "match": {"fields": fields},
    }
    rule.update(extra)
    rule.setdefault(
        "provenance", {"source": "manual", "verified": "human", "engine_version": "0.19.1"}
    )
    return rule


def _cand(
    cid: str, severity: str, fields: dict[str, Any], status: str = "unverified", **extra: Any
) -> dict[str, Any]:
    cand: dict[str, Any] = {
        "id": cid,
        "engine": "vllm",
        "engine_version": "0.19.1",
        "severity": severity,
        "match": {"fields": fields},
        "provenance": {"source": "analyst_cold_read"},
        "verdict": _verdict(status),
    }
    cand.update(extra)
    return cand


# --- Stage 3: union dedup + verdict memory ---


def test_union_dedup_prefers_verdict_carrying_entry() -> None:
    fresh = _cand("vllm_a", "error", {"temperature": {"<": 0}}, status="unverified")
    remembered = _cand("vllm_a", "error", {"temperature": {"<": 0}}, status="confirmed")
    other = _cand("vllm_b", "error", {"top_p": {">": 1}}, status="unverified")

    union = ab.union_candidates([[fresh, other], [remembered]])

    assert [c["id"] for c in union] == ["vllm_a", "vllm_b"]  # deduped on id, sorted
    kept = next(c for c in union if c["id"] == "vllm_a")
    assert ab._status(kept) == "confirmed"  # the verdict-carrying entry won


def test_union_first_seen_wins_when_neither_has_verdict() -> None:
    first = _cand("vllm_a", "error", {"temperature": {"<": 0}}, status="unverified")
    second = _cand("vllm_a", "dormant", {"temperature": {"<": 0}}, status="unverified")
    union = ab.union_candidates([[first], [second]])
    assert union[0]["severity"] == "error"


# --- Stage 3: match-spec dedup (two proposers, one claim) ---


def test_dedup_by_match_collapses_identical_claim_keeps_lexicographic_winner() -> None:
    # Same canonical fields/operators/values, different id hash and wording:
    # one shipped rule, the lexicographically smaller id.
    hi = _rule(
        "vllm_zzz", "error", {"vllm.engine_params.load_format": {"not_in": ["auto", "dummy"]}}
    )
    lo = _rule(
        "vllm_aaa", "error", {"vllm.engine_params.load_format": {"not_in": ["auto", "dummy"]}}
    )
    kept = ab.dedup_by_match([hi, lo])
    assert [r["id"] for r in kept] == ["vllm_aaa"]


def test_dedup_by_match_prefers_verdict_carrying_over_unverified() -> None:
    fresh = _cand("vllm_aaa", "error", {"vllm.sampling_params.n": {">": 1}}, status="unverified")
    remembered = _cand(
        "vllm_zzz", "error", {"vllm.sampling_params.n": {">": 1}}, status="confirmed"
    )
    kept = ab.dedup_by_match([fresh, remembered])
    # The confirmed entry wins even though its id sorts later - verdict
    # precedence must beat the lexicographic tiebreak.
    assert [c["id"] for c in kept] == ["vllm_zzz"]


def test_dedup_by_match_keeps_distinct_claims() -> None:
    a = _rule("r_a", "error", {"vllm.sampling_params.n": {">": 1}})
    b = _rule("r_b", "error", {"vllm.sampling_params.n": {">": 2}})  # different value
    c = _rule("r_c", "dormant", {"vllm.sampling_params.n": {">": 1}})  # different severity
    kept = ab.dedup_by_match([a, b, c])
    assert {r["id"] for r in kept} == {"r_a", "r_b", "r_c"}


def test_dedup_by_match_treats_scalar_and_eq_spec_as_equal() -> None:
    scalar = _rule("r_a", "error", {"vllm.engine_params.x": "auto"})
    eq = _rule("r_b", "error", {"vllm.engine_params.x": {"==": "auto"}})
    kept = ab.dedup_by_match([scalar, eq])
    assert [r["id"] for r in kept] == ["r_a"]


def test_build_union_reads_extractor_source(tmp_path: Path) -> None:
    pool = tmp_path
    for name, cid in (
        ("analyst_cold_read.yaml", "vllm_analyst_a"),
        ("cross_field_extractor.yaml", "vllm_extractor_b"),
        ("manual_seeds.yaml", "vllm_manual_c"),
    ):
        (pool / name).write_text(
            yaml.safe_dump({"candidates": [_cand(cid, "error", {"temperature": {"<": 0}})]})
        )
    # Each proposer claims temperature<0; canonical match-spec dedup collapses to one.
    union = ab.build_union(pool, "vllm")
    assert len(union) == 1
    assert union[0]["id"] == "vllm_analyst_a"  # lexicographically smallest id wins


# --- Canonical field addressing ---


def test_canonicalise_maps_bare_leaves_and_keeps_unknown() -> None:
    mapping = {"temperature": "vllm.sampling_params.temperature"}
    cand = _cand("vllm_a", "error", {"temperature": {"<": 0}, "mystery": {">": 1}})
    ab._canonicalise(cand, mapping)
    assert cand["match"]["fields"] == {
        "vllm.sampling_params.temperature": {"<": 0},  # mapped
        "mystery": {">": 1},  # unknown leaf left bare
    }


def test_canonical_map_reads_real_vllm_schema() -> None:
    mapping = ab.canonical_map("vllm")
    assert mapping["temperature"] == "vllm.sampling_params.temperature"
    assert mapping["max_num_batched_tokens"] == "vllm.engine_params.max_num_batched_tokens"
    assert mapping["seed"] == "vllm.sampling_params.seed"  # sampling_params wins the collision


# --- Stage 4: recall interrogation ---


def test_interrogation_prompt_carries_predicate_fields_severity_and_absent_instruction() -> None:
    rule = _rule(
        "r",
        "error",
        {"vllm.sampling_params.temperature": {"<": 0}},
        message_template="temperature must be >= 0",
    )
    prompt = ab.interrogation_prompt("vllm", "0.19.1", rule, source="SRC", label="f.py")
    assert "temperature" in prompt
    assert "temperature must be >= 0" in prompt  # predicate injected, not the old citation
    assert "severity: error" in prompt
    assert "ABSENT" in prompt
    assert "SRC" in prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"present": true, "citation": "f.py:5"}', ("present", "f.py:5")),
        ('{"present": false}', ("absent", None)),
        ('{"present": true}', ("present", None)),
        # Unparseable / non-object / missing-key replies are UNKNOWN, never ABSENT:
        # a parse failure must not be able to retire a real shipped rule.
        ("not json at all", ("unknown", None)),
        ("[1, 2, 3]", ("unknown", None)),  # valid JSON but not an object
        ('{"citation": "f.py:5"}', ("unknown", None)),  # object missing "present"
    ],
)
def test_parse_interrogation(text: str, expected: tuple[str, str | None]) -> None:
    assert ab.parse_interrogation(text) == expected


def _cited_rule(tmp_path: Path, rid: str, fields: dict[str, Any]) -> dict[str, Any]:
    # A rule whose citation resolves to a real file, so _rule_source locates it.
    (tmp_path / "s.py").write_text("guard source\n")
    return _rule(rid, "error", fields, provenance={"source": "manual", "citation": "s.py:1"})


def test_interrogate_rule_present_mints_candidate_with_recall_provenance(tmp_path: Path) -> None:
    rule = _cited_rule(tmp_path, "r", {"vllm.sampling_params.temperature": {"<": 0}})
    outcome, cand = ab.interrogate_rule(
        "vllm",
        "0.19.1",
        rule,
        tmp_path,
        lambda p, t: _Reply('{"present": true, "citation": "s.py:9"}'),
        "2026-07-07",
    )
    assert outcome == "present" and cand is not None
    assert cand["provenance"]["source"] == "recall_interrogation"
    assert cand["provenance"]["from_rule"] == "r"
    assert cand["provenance"]["interrogation_citation"] == "s.py:9"
    assert cand["match"]["fields"] == rule["match"]["fields"]  # carries the precise old predicate


def test_interrogate_rule_absent_yields_no_candidate(tmp_path: Path) -> None:
    rule = _cited_rule(tmp_path, "r", {"vllm.sampling_params.temperature": {"<": 0}})
    outcome, cand = ab.interrogate_rule(
        "vllm", "0.19.1", rule, tmp_path, lambda p, t: _Reply('{"present": false}'), "2026-07-07"
    )
    assert outcome == "absent" and cand is None


def test_interrogate_rule_skips_when_source_not_located(tmp_path: Path) -> None:
    """A citation-less rule (no retrievable source) is never sent to the model:
    the generate callable is not invoked, nothing is minted, and no retirement
    proposal or sign-off annotation is raised. A zero-evidence interrogation
    would otherwise draw a guaranteed false ABSENT every run."""
    rule = _rule("gone", "error", {"vllm.engine_params.foo": {"<": 1}})  # no citation

    def gen(prompt: str, temp: float) -> _Reply:
        raise AssertionError("generate must not be called for an un-locatable rule")

    minted, absent = ab.recall_interrogation(
        "vllm", "0.19.1", [rule], set(), tmp_path, gen, "2026-07-07"
    )
    assert minted == [] and absent == []  # no candidate, no retirement proposal, no annotation


def test_recall_interrogation_only_probes_unrediscovered_rules(tmp_path: Path) -> None:
    rediscovered = _rule("kept", "error", {"vllm.sampling_params.temperature": {"<": 0}})
    missing = _cited_rule(tmp_path, "gone", {"vllm.engine_params.foo": {"<": 1}})
    pool_leaf_keys = {ab._leaf_key(rediscovered)}  # temperature present in fresh pool

    asked: list[str] = []

    def gen(prompt: str, temp: float) -> _Reply:
        asked.append("q")
        return _Reply('{"present": false}')  # ABSENT

    minted, absent = ab.recall_interrogation(
        "vllm", "0.19.1", [rediscovered, missing], pool_leaf_keys, tmp_path, gen, "2026-07-07"
    )
    assert asked == ["q"]  # only the un-rediscovered rule was interrogated
    assert minted == []
    assert absent == ["gone"]


def test_unknown_interrogation_neither_mints_nor_retires_and_rule_ships(tmp_path: Path) -> None:
    """A garbage LLM reply is UNKNOWN, not ABSENT (must-fix 1): the rule is
    neither minted nor flagged for retirement, and still ships on an unverified
    probe verdict."""
    missing = _cited_rule(tmp_path, "gone", {"vllm.engine_params.foo": {"<": 1}})
    minted, absent = ab.recall_interrogation(
        "vllm", "0.19.1", [missing], set(), tmp_path, lambda p, t: _Reply("garbage"), "2026-07-07"
    )
    assert minted == []  # UNKNOWN mints nothing ...
    assert absent == []  # ... and raises no proposal (no parse-failure conflation)

    new_rules, delta = ab.promote("vllm", "0.19.1", "2026-07-07", [missing], {}, [], [], absent)
    assert "gone" in {r["id"] for r in new_rules}  # unverified verdict -> kept untouched
    assert "gone" not in delta.proposed_retirements


def test_rule_source_prefers_full_path_suffix_over_basename(tmp_path: Path) -> None:
    """Two files share a basename; the path-qualified citation must retrieve the
    right one, not sorted(rglob)[0] (should-fix 4)."""
    root = tmp_path / "src"
    (root / "core").mkdir(parents=True)
    (root / "distributed").mkdir(parents=True)
    (root / "core" / "scheduler.py").write_text("core scheduler\n")
    (root / "distributed" / "scheduler.py").write_text("distributed scheduler\n")
    rule = _rule(
        "r",
        "error",
        {"x": {"<": 1}},
        provenance={"source": "manual", "citation": "distributed/scheduler.py:1"},
    )
    numbered, label = ab._rule_source(root, rule)
    assert label == "distributed/scheduler.py"  # not core/scheduler.py (sorted-first)
    assert "distributed scheduler" in numbered


# --- Stage 6: promotion matrix ---


def _promote_matrix(
    signoff: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], ab.Delta]:
    old_rules = [
        _rule("old_direct", "error", {"vllm.sampling_params.temperature": {"<": 0}}),
        _rule("old_family", "error", {"vllm.sampling_params.top_p": {">": 1}}),
        _rule("old_absent", "error", {"vllm.engine_params.foo": {"<": 1}}),
        _rule("old_unconfirmed", "error", {"vllm.engine_params.baz": {"<": 1}}),
        _rule(
            "old_residue", "dormant", {"vllm.sampling_params.seed": -1}, normalised_fields=["seed"]
        ),
    ]
    verdicts = {
        "old_direct": _verdict("confirmed"),
        "old_family": _verdict("unconfirmed"),
        "old_absent": _verdict("unconfirmed"),
        "old_unconfirmed": _verdict("unconfirmed"),
        "old_residue": _verdict("unprobeable"),
    }
    union = [
        _cand("vllm_top_p", "error", {"vllm.sampling_params.top_p": {">": 1}}, status="confirmed"),
        _cand(
            "vllm_new",
            "error",
            {"vllm.sampling_params.min_p": {">": 1}},
            status="confirmed",
            provenance={"source": "analyst_cold_read", "rationale": "min_p must be > 0"},
        ),
        _cand("vllm_pooled", "error", {"vllm.sampling_params.bar": {">": 1}}, status="unconfirmed"),
    ]
    return ab.promote(
        "vllm",
        "0.19.1",
        "2026-07-07",
        old_rules,
        verdicts,
        union,
        signoff or [],
        # ABSENT on an unconfirmed rule AND on a probe-confirmed rule: annotation
        # only, so the confirmed one must still ship.
        ["old_absent", "old_direct"],
    )


def test_promotion_matrix_dispositions() -> None:
    new_rules, delta = _promote_matrix()
    shipped = {r["id"] for r in new_rules}

    assert "old_direct" in shipped  # confirmed directly -> survives verbatim
    assert "old_direct" in delta.proposed_retirements  # ABSENT never overrides the probe
    assert "old_family" in shipped  # a confirmed pool candidate on its family kept it alive
    assert "vllm_new" in shipped  # brand-new confirmed family -> added with pool id
    # ABSENT is an annotation, never a disposition: the unconfirmed rule lands in
    # residue (withheld pending sign-off) and is listed as a retirement proposal.
    assert "old_absent" not in shipped and "old_absent" in delta.residue
    assert delta.proposed_retirements == ["old_absent", "old_direct"]
    # unconfirmed and NOT declared absent -> residue too, but with no proposal
    assert "old_unconfirmed" not in shipped and "old_unconfirmed" in delta.residue
    assert "old_unconfirmed" not in delta.proposed_retirements
    assert "old_residue" not in shipped and "old_residue" in delta.residue
    assert "vllm_top_p" not in shipped  # rediscovery, not an add (family already shipped)
    assert "vllm_pooled" not in shipped  # unconfirmed candidate stays pooled

    assert delta.added == ["vllm_new"]
    assert "old_family" in delta.changed  # survived via pool candidate, flagged for drift review


def test_promotion_survivor_reconfirmed_is_restamped() -> None:
    """A directly re-confirmed survivor keeps its claim verbatim but its
    provenance is re-pinned to the confirming version (the corpus invariant
    pins provenance.engine_version to the envelope version)."""
    new_rules, _ = _promote_matrix()
    survivor = next(r for r in new_rules if r["id"] == "old_direct")
    expected = _rule("old_direct", "error", {"vllm.sampling_params.temperature": {"<": 0}})
    assert {k: v for k, v in survivor.items() if k != "provenance"} == {
        k: v for k, v in expected.items() if k != "provenance"
    }
    assert survivor["provenance"] == {
        "source": "manual",
        "verified": "construction",
        "engine_version": "0.19.1",
        "date": "2026-07-07",
    }


def test_promotion_residue_ships_only_when_signed_off() -> None:
    new_rules, delta = _promote_matrix(signoff=[{"id": "old_residue", "human_confirmed": True}])
    shipped = {r["id"]: r for r in new_rules}
    assert "old_residue" in shipped
    assert shipped["old_residue"]["provenance"]["verified"] == "human"
    assert "old_residue" not in delta.residue


def test_signoff_entry_carries_absent_annotation(tmp_path: Path) -> None:
    """An interrogation-ABSENT residue rule is annotated as a retirement proposal
    in the sign-off file; the maintainer decides, the tool never drops it."""
    old = {
        "r_absent": _rule("r_absent", "error", {"vllm.engine_params.foo": {"<": 1}}),
        "r_plain": _rule("r_plain", "error", {"vllm.engine_params.baz": {"<": 1}}),
    }
    delta = ab.Delta()
    delta.residue = ["r_absent", "r_plain"]
    ab.write_signoff(tmp_path, delta, old, {"r_absent"})
    rows = yaml.safe_load((tmp_path / "absorb_signoff.yaml").read_text())["residue"]
    by_id = {r["id"]: r for r in rows}
    assert by_id["r_absent"]["interrogation"] == "absent"
    assert by_id["r_absent"]["human_confirmed"] is False
    assert "interrogation" not in by_id["r_plain"]


def test_candidate_to_rule_shape() -> None:
    cand = _cand(
        "vllm_new",
        "error",
        {"vllm.sampling_params.min_p": {">": 1}},
        provenance={"source": "analyst_cold_read", "rationale": "min_p must be > 0"},
    )
    rule = ab.candidate_to_rule(cand, "vllm", "0.19.1", "2026-07-07")
    assert rule["id"] == "vllm_new"
    assert rule["message_template"] == "min_p must be > 0"
    assert rule["provenance"] == {
        "source": "analyst",  # analyst_cold_read maps to the loader's closed enum
        "verified": "construction",
        "engine_version": "0.19.1",
        "date": "2026-07-07",
    }


def test_promoted_corpus_parses_through_real_shipped_loader(tmp_path: Path) -> None:
    """A promote() corpus with an added analyst candidate and a human-signed
    residue rule must parse through the runtime loader's closed-enum gate - the
    thing that catches "absorb writes a corpus the runtime rejects" (should-fix 5)."""
    from llenergymeasure.config.engine_rules.loader import EngineRulesLoader

    old_rules = [
        _rule("old_signed", "error", {"vllm.engine_params.tensor_parallel_size": {"<": 1}})
    ]
    verdicts = {"old_signed": _verdict("unconfirmed")}  # -> sign-off path (verified: human)
    union = [
        _cand(
            "vllm_added",
            "error",
            {"vllm.sampling_params.min_p": {">": 1}},
            status="confirmed",
            provenance={"source": "analyst_cold_read", "rationale": "min_p must be > 0"},
        )
    ]
    new_rules, _ = ab.promote(
        "vllm",
        "0.19.1",
        "2026-07-07",
        old_rules,
        verdicts,
        union,
        [{"id": "old_signed", "human_confirmed": True}],
        [],
    )

    corpus_dir = tmp_path / "engines" / "vllm"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "rules.yaml").write_text(ab.serialize_corpus("vllm", "0.19.1", new_rules))

    parsed = EngineRulesLoader(corpus_root=tmp_path / "engines").load_rules("vllm")
    by_id = {r.id: r for r in parsed.rules}
    assert by_id["vllm_added"].provenance.source == "analyst"  # closed Source enum
    assert by_id["vllm_added"].provenance.verified == "construction"  # closed Verified enum
    assert by_id["old_signed"].provenance.verified == "human"  # signed residue


# --- Byte-stable corpus regeneration ---


def test_serialize_corpus_is_deterministic() -> None:
    rules = [
        _rule("b", "error", {"vllm.sampling_params.top_p": {">": 1}}),
        _rule("a", "error", {"vllm.sampling_params.temperature": {"<": 0}}),
    ]
    once = ab.serialize_corpus("vllm", "0.19.1", sorted(rules, key=lambda r: r["id"]))
    twice = ab.serialize_corpus("vllm", "0.19.1", sorted(rules, key=lambda r: r["id"]))
    assert once == twice
    assert once.startswith(ab._CORPUS_HEADER)


def test_shipped_vllm_corpus_round_trips_byte_identical() -> None:
    path, rules = ab.load_shipped_rules("vllm")
    rebuilt = ab.serialize_corpus("vllm", "0.19.1", sorted(rules, key=lambda r: str(r["id"])))
    assert rebuilt == path.read_text()


# --- Stage 5: the infra_error ladder gate ---


def _fake_probe(status: str) -> Any:
    def probe(engine: str, probe_input: Path, run_date: str) -> None:
        doc = yaml.safe_load(probe_input.read_text())
        for entry in doc["candidates"]:
            entry["verdict"] = _verdict(status)
        probe_input.write_text(yaml.safe_dump(doc, sort_keys=False))

    return probe


def test_ladder_fails_loudly_when_all_verdicts_infra_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ab, "run_probe", _fake_probe("infra_error"))
    entries = [_cand("vllm_a", "error", {"vllm.sampling_params.temperature": {"<": 0}})]
    with pytest.raises(SystemExit, match="infra_error"):
        ab.probe_ladder("vllm", entries, tmp_path, "2026-07-07", clean_room=False, skip_probe=False)


def test_ladder_records_confirmed_verdicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "run_probe", _fake_probe("confirmed"))
    entries = [_cand("vllm_a", "error", {"vllm.sampling_params.temperature": {"<": 0}})]
    verdicts = ab.probe_ladder(
        "vllm", entries, tmp_path, "2026-07-07", clean_room=False, skip_probe=False
    )
    assert verdicts["vllm_a"]["status"] == "confirmed"


def test_ladder_memory_skips_already_verdicted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ab, "run_probe", lambda *a: calls.append("probe"))
    entries = [
        _cand("vllm_a", "error", {"vllm.sampling_params.temperature": {"<": 0}}, status="confirmed")
    ]
    ab.probe_ladder("vllm", entries, tmp_path, "2026-07-07", clean_room=False, skip_probe=False)
    assert calls == []  # already carries a verdict; not re-probed


def test_citation_pass_rate_cleans_up_temp_on_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The citation gate writes a throwaway input file; an exception mid-check must
    not leave it behind in the pool dir (try/finally cleanup)."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("citation check exploded")

    monkeypatch.setattr(ab, "load_candidates", _boom)
    union = [_cand("vllm_a", "error", {"vllm.sampling_params.temperature": {"<": 0}})]
    with pytest.raises(RuntimeError, match="exploded"):
        ab.citation_pass_rate(union, tmp_path, tmp_path)
    assert not (tmp_path / "_citation_input.yaml").exists()


def test_run_probe_hard_errors_on_nonzero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe wrapper that exits non-zero before writing verdicts must be a hard
    error, not a silent no-op absorb (the operator would otherwise read the empty
    result as a clean 'nothing to probe')."""
    probe_input = tmp_path / "_probe_input.yaml"
    probe_input.write_text("candidates: []\n")

    def _crash(*args: Any, **kwargs: Any) -> Any:
        return subprocess.CompletedProcess(args=args, returncode=1)

    monkeypatch.setattr(ab.subprocess, "run", _crash)
    monkeypatch.setattr(ab, "_require_under_repo", lambda p: Path(p.name))
    with pytest.raises(SystemExit, match=re.escape("probe_candidates.sh")):
        ab.run_probe("vllm", probe_input, "2026-07-07")


# --- Orchestration: --dry-run + skip flags ---


@pytest.fixture
def engine_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A throwaway engine corpus + discovered schema + seeded analyst pool."""
    corpus_root = tmp_path / "engines"
    engine_dir = corpus_root / "vllm"
    engine_dir.mkdir(parents=True)
    rules = [_rule("vllm_keep", "error", {"vllm.sampling_params.temperature": {"<": 0}})]
    (engine_dir / "rules.yaml").write_text(ab.serialize_corpus("vllm", "0.19.1", rules))
    (engine_dir / "schema.discovered.json").write_text(
        '{"sampling_params": {"temperature": {}}, "engine_params": {}}'
    )
    monkeypatch.setattr(ab, "CORPUS_ROOT", corpus_root)
    monkeypatch.setattr(ab, "resolve_version", lambda engine: "0.19.1")

    pool_root = tmp_path / "pool"
    seed = ab.pool_path(pool_root, "vllm", "0.19.1", "analyst_cold_read.yaml")
    seed.parent.mkdir(parents=True)
    ab.write_pool(
        seed,
        source="analyst_cold_read",
        engine="vllm",
        version="0.19.1",
        generated_at="2026-07-07",
        candidates=[_cand("vllm_temp", "error", {"temperature": {"<": 0}})],
        header="# seed\n",
    )
    return corpus_root, pool_root


def _args(pool_root: Path, **overrides: Any) -> Any:
    base = dict(
        engine="vllm",
        source_root=None,
        pool_root=pool_root,
        date="2026-07-07",
        clean_room=False,
        dry_run=False,
        skip_cold_read=True,
        skip_extractor=True,
        skip_interrogation=True,
        skip_probe=True,
        ollama_host="x",
        model="m",
    )
    base.update(overrides)
    return ab.argparse.Namespace(**base)


def test_dry_run_leaves_corpus_untouched(engine_tree: tuple[Path, Path]) -> None:
    corpus_root, pool_root = engine_tree
    corpus = corpus_root / "vllm" / "rules.yaml"
    before = corpus.read_text()
    ab.absorb(_args(pool_root, dry_run=True))
    assert corpus.read_text() == before  # nothing written to the shipped corpus
    assert (
        pool_root / "vllm" / "v0_19_1" / "candidates" / "union.yaml"
    ).is_file()  # workspace written


def test_live_run_regenerates_corpus_byte_stable(engine_tree: tuple[Path, Path]) -> None:
    corpus_root, pool_root = engine_tree
    corpus = corpus_root / "vllm" / "rules.yaml"
    before = corpus.read_text()
    ab.absorb(_args(pool_root))  # --skip-probe: untested rule kept, not dropped
    once = corpus.read_text()
    ab.absorb(_args(pool_root))
    assert once == before  # conservative keep leaves an untested corpus byte-identical
    assert corpus.read_text() == once  # and the regeneration is idempotent


def test_signoff_mark_survives_across_reruns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A signed-off unconfirmed rule ships on every run and its mark is never
    erased: the corpus is byte-identical across two live runs and the rule ships
    both times (must-fix 2 - the self-erasing sign-off oscillation)."""
    corpus_root = tmp_path / "engines"
    engine_dir = corpus_root / "vllm"
    engine_dir.mkdir(parents=True)
    rule = _rule("vllm_signed", "error", {"vllm.engine_params.tensor_parallel_size": {"<": 1}})
    (engine_dir / "rules.yaml").write_text(ab.serialize_corpus("vllm", "0.19.1", [rule]))
    (engine_dir / "schema.discovered.json").write_text(
        '{"sampling_params": {}, "engine_params": {"tensor_parallel_size": {}}}'
    )
    monkeypatch.setattr(ab, "CORPUS_ROOT", corpus_root)
    monkeypatch.setattr(ab, "resolve_version", lambda engine: "0.19.1")
    monkeypatch.setattr(ab, "run_probe", _fake_probe("unconfirmed"))  # never construction-confirmed

    pool_root = tmp_path / "pool"
    pool_dir = ab.pool_path(pool_root, "vllm", "0.19.1", "x").parent
    pool_dir.mkdir(parents=True)
    # The sign-off is the durable approval record: it lives in the versioned,
    # git-tracked outputs/ snapshot dir (sibling of the gitignored candidates/
    # pool), so it survives a fresh clone.
    signoff_dir = pool_dir.parent / "outputs"
    signoff_dir.mkdir(parents=True)
    # Pre-seed the human sign-off mark. No analyst pool file: the rule is never
    # rediscovered, and with --skip-interrogation it is never proposed for retirement.
    (signoff_dir / ab._SIGNOFF_FILE).write_text(
        yaml.safe_dump({"residue": [{"id": "vllm_signed", "human_confirmed": True}]})
    )

    corpus = engine_dir / "rules.yaml"
    args = _args(pool_root, skip_cold_read=True, skip_interrogation=True, skip_probe=False)
    ab.absorb(args)
    first = corpus.read_text()
    assert "vllm_signed" in first  # ships via the sign-off path ...
    assert {e["id"] for e in ab.read_signoff(signoff_dir) if e["human_confirmed"]} == {
        "vllm_signed"
    }  # mark retained
    # The gitignored candidates/ pool must NOT hold the durable approval record.
    assert not (pool_dir / ab._SIGNOFF_FILE).exists()

    ab.absorb(_args(pool_root, skip_cold_read=True, skip_interrogation=True, skip_probe=False))
    second = corpus.read_text()
    assert second == first  # byte-identical: no oscillation across re-runs
    assert "vllm_signed" in second
    assert {e["id"] for e in ab.read_signoff(signoff_dir) if e["human_confirmed"]} == {
        "vllm_signed"
    }  # mark alive after run two


def test_signoff_record_resources_withheld_rule_across_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Full sign-off lifecycle across runs. Run 1 withholds two unconfirmed
    rules and records their complete bodies in the sign-off file (rules.yaml no
    longer carries them, so the record is the only surviving source). The
    maintainer marks one; run 2 re-ships it from the record body with human
    provenance while the declined rule stays withheld WITH its body. Run 3
    proves double-run byte-stability of both the corpus and the record."""
    corpus_root = tmp_path / "engines"
    engine_dir = corpus_root / "vllm"
    engine_dir.mkdir(parents=True)
    rule_a = _rule("vllm_keep", "error", {"vllm.engine_params.tensor_parallel_size": {"<": 1}})
    rule_b = _rule("vllm_lapse", "error", {"vllm.engine_params.tensor_parallel_size": {"<": 0}})
    (engine_dir / "rules.yaml").write_text(ab.serialize_corpus("vllm", "0.19.1", [rule_a, rule_b]))
    (engine_dir / "schema.discovered.json").write_text(
        '{"sampling_params": {}, "engine_params": {"tensor_parallel_size": {}}}'
    )
    monkeypatch.setattr(ab, "CORPUS_ROOT", corpus_root)
    monkeypatch.setattr(ab, "resolve_version", lambda engine: "0.19.1")
    monkeypatch.setattr(ab, "run_probe", _fake_probe("unconfirmed"))

    pool_root = tmp_path / "pool"
    pool_dir = ab.pool_path(pool_root, "vllm", "0.19.1", "x").parent
    pool_dir.mkdir(parents=True)
    signoff_dir = pool_dir.parent / "outputs"
    corpus = engine_dir / "rules.yaml"
    signoff_path = signoff_dir / ab._SIGNOFF_FILE
    args = _args(pool_root, skip_cold_read=True, skip_interrogation=True, skip_probe=False)

    # Run 1: both rules withheld; the record carries their full bodies.
    ab.absorb(args)
    assert "vllm_keep" not in corpus.read_text() and "vllm_lapse" not in corpus.read_text()
    rows = {e["id"]: e for e in ab.read_signoff(signoff_dir)}
    assert rows["vllm_keep"]["human_confirmed"] is False
    assert rows["vllm_keep"]["rule"]["match"]["fields"] == rule_a["match"]["fields"]
    assert rows["vllm_lapse"]["rule"]["match"]["fields"] == rule_b["match"]["fields"]

    # Maintainer signs off one rule; the other stays declined.
    doc = yaml.safe_load(signoff_path.read_text())
    for row in doc["residue"]:
        if row["id"] == "vllm_keep":
            row["human_confirmed"] = True
    signoff_path.write_text(yaml.safe_dump(doc, sort_keys=False))

    # Run 2: the marked rule is re-sourced from the record body (it is in no
    # live input any more) and ships with human provenance at the consuming
    # run's pin; the declined rule stays withheld and keeps its body.
    ab.absorb(args)
    second = corpus.read_text()
    assert "vllm_keep" in second and "vllm_lapse" not in second
    shipped = yaml.safe_load(second)["rules"]
    by_id = {r["id"]: r for r in shipped}
    assert by_id["vllm_keep"]["provenance"]["verified"] == "human"
    assert by_id["vllm_keep"]["provenance"]["engine_version"] == "0.19.1"
    rows = {e["id"]: e for e in ab.read_signoff(signoff_dir)}
    assert rows["vllm_keep"]["human_confirmed"] is True
    assert rows["vllm_lapse"]["human_confirmed"] is False
    assert isinstance(rows["vllm_lapse"]["rule"], dict)  # body persists for a later decision

    # Run 3: byte-stability of both artifacts.
    signoff_after_two = signoff_path.read_text()
    ab.absorb(args)
    assert corpus.read_text() == second
    assert signoff_path.read_text() == signoff_after_two


def test_signed_entry_without_body_fails_loudly() -> None:
    """A human_confirmed entry whose rule is in no live input and whose record
    carries no body must abort with a pointer at git history, never silently
    skip (the mark exists precisely because the rule text was withheld)."""
    with pytest.raises(SystemExit, match="git history"):
        ab.promote(
            "vllm",
            "0.19.1",
            "2026-07-07",
            [],
            {},
            [],
            [{"id": "vllm_ghost", "human_confirmed": True}],
            [],
        )


def test_skip_flags_bypass_cold_read_and_probe(
    engine_tree: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, pool_root = engine_tree

    def explode(*a: Any, **k: Any) -> None:
        raise AssertionError("skipped stage was invoked")

    monkeypatch.setattr(ab, "run_cold_read", explode)
    monkeypatch.setattr(ab, "run_probe", explode)
    monkeypatch.setattr(ab, "ollama_generator", explode)
    ab.absorb(_args(pool_root, dry_run=True))  # must not raise
