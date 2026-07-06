"""Tests for scripts/check_citations.py (verification-ladder tier 1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parents[3] / "scripts"))

from check_citations import (
    Candidate,
    CandidateSchemaError,
    Citation,
    check_candidate,
    load_candidates,
    main,
)

# A tiny fixture source file. Line numbers are 1-based:
#   1 class EngineArgs:
#   2     def __post_init__(self):
#   3         if self.max_num_seqs < 1:
#   4             raise ValueError(
#   5                 "max_num_seqs must be at least 1")
#   6         self.max_num_seqs = self.max_num_seqs
#   7         self.done = True
SRC = (
    "class EngineArgs:\n"
    "    def __post_init__(self):\n"
    "        if self.max_num_seqs < 1:\n"
    "            raise ValueError(\n"
    '                "max_num_seqs must be at least 1")\n'
    "        self.max_num_seqs = self.max_num_seqs\n"
    "        self.done = True\n"
)


def _src_root(tmp_path: Path) -> Path:
    (tmp_path / "engine.py").write_text(SRC)
    return tmp_path


def _quote(start: int, end: int) -> str:
    return "\n".join(SRC.splitlines()[start - 1 : end])


def _cand(
    fields: dict[str, Any],
    lines: tuple[int, int],
    *,
    file: str = "engine.py",
    quote: str | None = None,
    cid: str = "c1",
) -> Candidate:
    q = quote if quote is not None else _quote(*lines)
    return Candidate(id=cid, fields=fields, citation=Citation(file, lines[0], lines[1], q))


def test_clean_pass(tmp_path: Path) -> None:
    cand = _cand({"vllm.engine_params.max_num_seqs": {"<": 1}}, (3, 5))
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert verdict.ok
    assert verdict.reason is None


def test_quoted_span_mismatch(tmp_path: Path) -> None:
    cand = _cand(
        {"vllm.engine_params.max_num_seqs": {"<": 1}},
        (3, 5),
        quote="        if self.max_num_seqs < 2:",
    )
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert not verdict.ok
    assert "does not match file content at lines [3, 5]" in verdict.reason


def test_line_range_drifted(tmp_path: Path) -> None:
    # Cite lines 1-3 but quote the text that actually lives at lines 3-5.
    cand = _cand({"vllm.engine_params.max_num_seqs": {"<": 1}}, (1, 3), quote=_quote(3, 5))
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert not verdict.ok
    assert "citation drifted" in verdict.reason
    assert "[3, 5]" in verdict.reason


def test_cited_file_missing(tmp_path: Path) -> None:
    cand = _cand({"vllm.engine_params.max_num_seqs": {"<": 1}}, (3, 5), file="does_not_exist.py")
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert not verdict.ok
    assert "not found under source root" in verdict.reason


def test_cited_file_escapes_root(tmp_path: Path) -> None:
    root = tmp_path / "src"
    root.mkdir()
    (tmp_path / "engine.py").write_text(SRC)
    cand = _cand({"vllm.engine_params.max_num_seqs": {"<": 1}}, (3, 5), file="../engine.py")
    verdict = check_candidate(cand, root)
    assert not verdict.ok
    assert "escapes the source root" in verdict.reason


def test_constrained_field_absent_from_span(tmp_path: Path) -> None:
    # Field leaf max_num_seqs does not appear at line 7.
    cand = _cand({"vllm.engine_params.max_num_seqs": {"present": True}}, (7, 7))
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert not verdict.ok
    assert verdict.reason == "constrained field 'max_num_seqs' not present in cited span"


def test_constrained_value_absent_from_span(tmp_path: Path) -> None:
    # Field leaf present at line 6, but the claimed bound 999 is not.
    cand = _cand({"vllm.engine_params.max_num_seqs": {"<": 999}}, (6, 6))
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert not verdict.ok
    assert verdict.reason == "constrained value '999' not present in cited span"


def test_line_range_out_of_bounds(tmp_path: Path) -> None:
    cand = _cand({"vllm.engine_params.max_num_seqs": {"<": 1}}, (1, 99))
    verdict = check_candidate(cand, _src_root(tmp_path))
    assert not verdict.ok
    assert "out of bounds (file has 7 lines)" in verdict.reason


def test_cross_field_ref_token_entailed(tmp_path: Path) -> None:
    # A cross-field @ref contributes the referenced field's leaf as a token.
    (tmp_path / "engine.py").write_text(
        "if num_beams % num_beam_groups != 0:\n    raise ValueError\n"
    )
    cand = _cand(
        {"transformers.generation.num_beams": {"not_divisible_by": "@num_beam_groups"}},
        (1, 2),
        quote="if num_beams % num_beam_groups != 0:\n    raise ValueError",
    )
    verdict = check_candidate(cand, tmp_path)
    assert verdict.ok


def test_word_boundary_kills_substring_false_positive(tmp_path: Path) -> None:
    # The field leaf 'max_tokens' must NOT be counted present just because a
    # different identifier 'max_tokens_per_batch' appears in the span.
    (tmp_path / "engine.py").write_text("if max_tokens_per_batch < 1:\n    raise ValueError\n")
    cand = _cand(
        {"vllm.engine_params.max_tokens": {"<": 1}},
        (1, 2),
        quote="if max_tokens_per_batch < 1:\n    raise ValueError",
    )
    verdict = check_candidate(cand, tmp_path)
    assert not verdict.ok
    assert verdict.reason == "constrained field 'max_tokens' not present in cited span"


def test_word_boundary_accepts_standalone_identifier(tmp_path: Path) -> None:
    (tmp_path / "engine.py").write_text("if max_tokens < 1:\n    raise ValueError\n")
    cand = _cand(
        {"vllm.engine_params.max_tokens": {"<": 1}},
        (1, 2),
        quote="if max_tokens < 1:\n    raise ValueError",
    )
    assert check_candidate(cand, tmp_path).ok


def test_malformed_citation_raises_loudly(tmp_path: Path) -> None:
    # A citation block that is PRESENT but missing its file is a schema error.
    path = tmp_path / "candidates.yaml"
    path.write_text(
        "schema_version: 1.0.0\n"
        "candidates:\n"
        "- id: broken\n"
        "  match:\n"
        "    fields:\n"
        "      vllm.x.y: {'<': 1}\n"
        "  citation:\n"
        "    lines: [1, 2]\n"
        "    quote: something\n"
    )
    with pytest.raises(CandidateSchemaError, match=r"'broken'.*file"):
        load_candidates(path)


def test_uncited_candidate_is_skipped_not_failed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # An observed-collision proposal has no citation key: skipped, counted, exit 0.
    root = _src_root(tmp_path)
    path = tmp_path / "candidates.yaml"
    path.write_text(
        "schema_version: 1.0.0\n"
        "candidates:\n"
        "- id: collision1\n"
        "  match:\n"
        "    fields:\n"
        "      vllm.engine_params.enforce_eager: {present: true}\n"
    )
    parsed = load_candidates(path)
    assert parsed[0].citation is None

    code = main([str(path), "--source-root", str(root)])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checked"] == 0
    assert report["skipped_uncited"] == 1
    assert report["skipped_ids"] == ["collision1"]


def test_missing_candidates_list_raises(tmp_path: Path) -> None:
    path = tmp_path / "candidates.yaml"
    path.write_text("schema_version: 1.0.0\n")
    with pytest.raises(CandidateSchemaError, match="'candidates' list"):
        load_candidates(path)


def _candidates_file(tmp_path: Path, lines: tuple[int, int]) -> Path:
    path = tmp_path / "candidates.yaml"
    path.write_text(
        "schema_version: 1.0.0\n"
        "candidates:\n"
        "- id: c1\n"
        "  match:\n"
        "    fields:\n"
        "      vllm.engine_params.max_num_seqs: {'<': 1}\n"
        "  citation:\n"
        "    file: engine.py\n"
        f"    lines: [{lines[0]}, {lines[1]}]\n"
        "    quote: |\n" + "".join(f"      {ln}\n" for ln in _quote(3, 5).splitlines())
    )
    return path


def test_main_all_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = _src_root(tmp_path)
    candidates = _candidates_file(tmp_path, (3, 5))
    code = main([str(candidates), "--source-root", str(root)])
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["checked"] == 1
    assert report["failed"] == 0
    assert report["verdicts"][0] == {"id": "c1", "ok": True, "reason": None}


def test_main_reports_failure_and_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _src_root(tmp_path)
    candidates = _candidates_file(tmp_path, (1, 3))  # wrong lines -> drift
    code = main([str(candidates), "--source-root", str(root)])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["failed"] == 1
    assert report["verdicts"][0]["ok"] is False
