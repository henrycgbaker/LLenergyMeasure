"""Host-side tests for scripts/analyst_cold_read.py (proposer S2).

No model and no network: the Ollama boundary is a stub Generator, so every test
exercises the harness's own logic - chunking and line fidelity, the salvage
parser, predicate translation, citation resolution, cap-hit split-and-rerun,
union dedup, and the emission that must round-trip through the ladder's tier-1
(check_citations) and tier-2/3 (probe_candidates) consumers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

import analyst_cold_read as acr
import check_citations as cc
import probe_candidates as pc


def _gen(text: str, done: str = "stop", eval_count: int = 100) -> Any:
    return acr.GenResult(text=text, done_reason=done, eval_count=eval_count)


def _rules_json(*rules: dict[str, Any]) -> str:
    return json.dumps({"rules": list(rules)})


# A rule the LLM might emit for the fixture source below (citation points at line 3):
#   1 class SamplingParams:
#   2     def _verify_args(self):
#   3         if self.n < 1:
#   4             raise ValueError("n must be >= 1")
#   5         if self.temperature < 0.0:
#   6             raise ValueError("temperature must be >= 0")
_N_RULE = {
    "kind": "single_field_bound",
    "fields": ["self.n"],
    "predicate": "n >= 1",
    "condition": "SamplingParams._verify_args",
    "citation": "sampling_params.py:3",
    "severity_suggestion": "error",
    "rationale": "n must be a positive integer",
}
_FIXTURE_SRC = (
    "class SamplingParams:\n"
    "    def _verify_args(self):\n"
    "        if self.n < 1:\n"
    '            raise ValueError("n must be >= 1")\n'
    "        if self.temperature < 0.0:\n"
    '            raise ValueError("temperature must be >= 0")\n'
)


@pytest.fixture
def fake_tree(tmp_path: Path) -> tuple[Path, dict[str, list[str]], list[Any]]:
    """A one-file source root plus its loaded sources and built chunks."""
    (tmp_path / "sampling_params.py").write_text(_FIXTURE_SRC)
    cluster_files, sources = acr.load_sources(tmp_path, {"sampling": ["sampling_params.py"]})
    return tmp_path, sources, acr.build_chunks(cluster_files, sources)


def _build(chunk: Any, sources: dict[str, list[str]], rule: dict[str, Any]) -> dict[str, Any] | str:
    return acr.build_candidate(
        chunk, sources, rule, engine="vllm", version="0.19.1", model="m", samples=3, run_date="d"
    )


# Chunking + line fidelity.
def test_window_preserves_real_line_numbers() -> None:
    seg = acr.Segment("f.py", 10, [f"x{i} = {i}" for i in range(1, 6)])
    numbered = seg.numbered().splitlines()
    assert numbered[0].strip().startswith("10 ")
    assert numbered[0].endswith("x1 = 1")
    assert seg.end_line == 14


def test_window_overlap_and_progress() -> None:
    lines = [f"line-{i}" for i in range(1, 5001)]
    segs = acr._window_file("big.py", lines)
    assert len(segs) >= 2  # a 5000-line file exceeds one char budget
    assert segs[1].start_line == segs[0].end_line + 1 - acr.OVERLAP_LINES  # overlap
    assert all(segs[i + 1].start_line > segs[i].start_line for i in range(len(segs) - 1))
    assert segs[0].start_line == 1 and segs[-1].end_line == 5000  # full coverage, none dropped


def test_build_chunks_is_deterministic(fake_tree: Any) -> None:
    _, _, chunks = fake_tree
    assert [c.chunk_id for c in chunks] == ["sampling.0"]
    assert chunks[0].class_names() == ["SamplingParams"]


def test_prompt_enumerates_class_checklist(fake_tree: Any) -> None:
    _, _, chunks = fake_tree
    prompt = acr.render_prompt(chunks[0], "vllm", "0.19.1")
    assert "CLASS COVERAGE CHECKLIST" in prompt
    assert "- SamplingParams" in prompt
    assert "vllm 0.19.1 source" in prompt


# Salvage parser.
def test_salvage_clean_json() -> None:
    text = _rules_json({"kind": "x", "fields": ["a"]}, {"kind": "y", "fields": ["b"]})
    assert len(acr.salvage_rules(text)) == 2


def test_salvage_recovers_complete_objects_from_truncation() -> None:
    # Wrapper never closes; one complete rule, one cut off -> recover the complete one.
    text = (
        '{"rules": [{"kind":"single_field_bound","fields":["n"],'
        '"predicate":"n >= 1","citation":"p.py:3"}, {"kind":"cross_fie'
    )
    rules = acr.salvage_rules(text)
    assert len(rules) == 1 and rules[0]["fields"] == ["n"]


@pytest.mark.parametrize("text", ["not json at all", ""])
def test_salvage_garbage_is_empty(text: str) -> None:
    assert acr.salvage_rules(text) == []


# Predicate translation: free-text predicate -> match.fields spec.
@pytest.mark.parametrize(
    ("kind", "fields", "predicate", "severity", "match", "normalised"),
    [
        ("single_field_bound", ["self.n"], "n >= 1", "error", {"n": {"<": 1}}, []),
        (
            "single_field_enum",
            ["kv_role"],
            "kv_role in {producer, consumer}",
            "error",
            {"kv_role": {"not_in": ["producer", "consumer"]}},
            [],
        ),
        (
            "cross_field",
            ["a", "b"],
            "a <= b",
            "error",
            {"a": {"present": True}, "b": {"present": True}},
            [],
        ),
        (
            "single_field_bound",
            ["seed"],
            "seed >= 0 or seed == -1",
            "error",
            {"seed": {"present": True}},
            [],
        ),
        (
            "dormancy",
            ["seed"],
            "seed == -1 -> None",
            "dormant",
            {"seed": {"present": True}},
            ["seed"],
        ),
    ],
)
def test_predicate_translation(
    kind: str,
    fields: list[str],
    predicate: str,
    severity: str,
    match: dict[str, Any],
    normalised: list[str],
) -> None:
    assert acr.predicate_to_match(kind, fields, predicate, severity) == (match, normalised)


# Citation resolution.
def test_resolve_citation_extracts_verbatim_window(fake_tree: Any) -> None:
    _, sources, chunks = fake_tree
    citation = acr.resolve_citation(chunks[0], sources, "sampling_params.py:3")
    assert citation is not None
    assert citation["file"] == "sampling_params.py"
    assert citation["lines"] == [1, 6]  # line 3 +/- QUOTE_RADIUS, clamped to the 6-line file
    assert "if self.n < 1:" in citation["quote"]


@pytest.mark.parametrize("raw", ["imaginary.py:3", "sampling_params.py:999"])
def test_resolve_citation_rejects_bad_reference(fake_tree: Any, raw: str) -> None:
    _, sources, chunks = fake_tree
    assert acr.resolve_citation(chunks[0], sources, raw) is None


# Candidate assembly + emission round-trip through the ladder consumers.
def test_candidate_has_unified_schema(fake_tree: Any) -> None:
    _, sources, chunks = fake_tree
    cand = _build(chunks[0], sources, _N_RULE)
    assert isinstance(cand, dict)
    assert cand["engine"] == "vllm" and cand["engine_version"] == "0.19.1"
    assert cand["severity"] == "error"
    assert cand["match"]["fields"] == {"n": {"<": 1}}
    assert set(cand["citation"]) == {"file", "lines", "quote"}
    assert cand["provenance"]["source"] == "analyst_cold_read"
    assert cand["verdict"] == {"status": "unverified", "tier": None, "date": None, "evidence": None}


@pytest.mark.parametrize(
    ("rule", "reason"),
    [
        ({"citation": "sampling_params.py:3"}, "no_field"),
        (dict(_N_RULE, citation="ghost.py:1"), "unresolved_citation"),
    ],
)
def test_candidate_dropped_when_unusable(fake_tree: Any, rule: dict[str, Any], reason: str) -> None:
    _, sources, chunks = fake_tree
    assert _build(chunks[0], sources, rule) == reason  # the drop reason run_analyst counts


@pytest.fixture
def emitted(fake_tree: Any, tmp_path: Path) -> tuple[Path, Path]:
    """Build the _N_RULE candidate and write the pool file; return (src_root, pool_path)."""
    src_root, sources, chunks = fake_tree
    cand = _build(chunks[0], sources, _N_RULE)
    assert isinstance(cand, dict)
    out = acr.write_candidates(
        [cand],
        tmp_path / "pool",
        engine="vllm",
        version="0.19.1",
        model="m",
        samples=3,
        run_date="d",
    )
    assert out == tmp_path / "pool" / "vllm" / "v0_19_1" / "candidates" / "analyst_cold_read.yaml"
    return src_root, out


def test_emission_round_trips_through_check_citations(emitted: tuple[Path, Path]) -> None:
    src_root, out = emitted
    verdict = cc.check_candidate(cc.load_candidates(out)[0], src_root)
    assert verdict.ok, verdict.reason


def test_emission_round_trips_through_probe_candidates(emitted: tuple[Path, Path]) -> None:
    _, out = emitted
    _document, entries = pc.load_document(out)
    parsed = pc.parse_candidate(entries[0], 0)
    assert parsed.severity == "error"
    assert parsed.fields == {"n": {"<": 1}}
    assert parsed.engine == "vllm"


# Sampling orchestration: union dedup + cap-hit split-and-rerun.
def test_union_dedups_identical_rules_across_samples(fake_tree: Any) -> None:
    _, sources, chunks = fake_tree
    other = dict(
        _N_RULE,
        fields=["temperature"],
        predicate="temperature >= 0.0",
        citation="sampling_params.py:5",
    )

    def stub(prompt: str, temperature: float) -> Any:
        return _gen(_rules_json(_N_RULE, other))  # every sample proposes the same two rules

    candidates, drops = acr.run_analyst(
        chunks, sources, stub, engine="vllm", version="0.19.1", model="m", samples=3, run_date="d"
    )
    assert len(candidates) == 2  # union collapses the duplicates
    assert {next(iter(c["match"]["fields"])) for c in candidates} == {"n", "temperature"}
    assert drops == {"dedup_collapsed": 4}  # 3 samples x 2 rules -> 2 kept, 4 collapsed


def test_same_claim_at_shifted_window_collapses_to_one_id(fake_tree: Any) -> None:
    # The id digests the claim (fields, file, predicate), not the line window:
    # one claim re-proposed at a shifted cite is one candidate, first cite kept.
    _, sources, chunks = fake_tree
    shifted = dict(_N_RULE, citation="sampling_params.py:6")

    def stub(prompt: str, temperature: float) -> Any:
        return _gen(_rules_json(_N_RULE, shifted))

    candidates, drops = acr.run_analyst(
        chunks, sources, stub, engine="vllm", version="0.19.1", model="m", samples=1, run_date="d"
    )
    assert len(candidates) == 1
    assert drops == {"dedup_collapsed": 1}
    assert candidates[0]["citation"]["lines"] == [1, 6]  # window of the first cite, line 3


def test_cap_hit_splits_and_reruns() -> None:
    # One 400-line file: the temp-0 pass on the whole chunk hits the cap, so the
    # chunk is halved and each half re-run instead of accepting truncation.
    sources = {"big.py": "\n".join(f"x{i} = {i}" for i in range(1, 401)).splitlines()}
    chunks = acr.build_chunks({"c": ["big.py"]}, sources)
    assert len(chunks) == 1
    seen_temps: list[float] = []

    def stub(prompt: str, temperature: float) -> Any:
        seen_temps.append(temperature)
        if temperature == 0.0 and "x400" in prompt and "x1 = 1" in prompt:  # only the full chunk
            return _gen('{"rules": []}', done="length", eval_count=acr.NUM_PREDICT)
        return _gen('{"rules": []}')

    acr.sample_chunk(chunks[0], stub, 1, engine="vllm", version="0.19.1")
    assert seen_temps.count(0.0) >= 3  # full chunk + the two halves each read at temp-0


def test_no_split_when_under_cap(fake_tree: Any) -> None:
    _, _sources, chunks = fake_tree
    calls: list[float] = []

    def stub(prompt: str, temperature: float) -> Any:
        calls.append(temperature)
        return _gen(_rules_json(_N_RULE))

    acr.sample_chunk(chunks[0], stub, 3, engine="vllm", version="0.19.1")
    assert calls == [0.0, 0.6, 0.6]  # temp-0 then samples-1 temp-0.6, no split


# Manifest loading.
def test_missing_manifest_fails_clearly() -> None:
    with pytest.raises(FileNotFoundError) as exc:
        acr.load_manifest("transformers")
    assert "no analyst cluster manifest" in str(exc.value)
    assert "transformers" in str(exc.value)


def test_vllm_manifest_loads() -> None:
    manifest = acr.load_manifest("vllm")
    assert "sampling" in manifest and "platforms" in manifest
    assert "engine/arg_utils.py" not in [p for ps in manifest.values() for p in ps]
