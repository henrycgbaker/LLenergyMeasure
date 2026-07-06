#!/usr/bin/env python3
"""Citation check: tier 1 of the engine-rule verification ladder.

Proposers emit candidate rules into a per-version workspace, each carrying a
mandatory citation into the pinned engine source tree (a file, an inclusive
line range, and the verbatim quoted span). This script verifies each citation
deterministically and offline, before the construction/identity probes that
actually run the engine. Against a ``--source-root`` engine source tree it
confirms, per candidate, that:

- the cited file exists under the source root (and does not escape it);
- the line range is in bounds;
- the quoted span matches the file content there (a shifted quote is reported
  as a drift, naming the line range where the text now lives);
- the constrained field and value tokens appear in the span, so the span
  entails the claimed constraint.

Output is machine readable: a JSON report on stdout with a per-candidate
verdict (``ok`` plus a precise ``reason`` on failure); exit 0 iff all checked
candidates pass. A candidate with NO ``citation`` key (an observed-collision
proposal cites runtime evidence, not source) is SKIPPED and counted, not
failed; a candidate whose citation is present but malformed still raises
:class:`CandidateSchemaError` loudly.

The candidate shape is a superset of the shipped ``rules.yaml`` rule (``id``,
``match.fields``, ...) plus an optional ``citation`` mapping {``file``,
``lines``: [start, end], ``quote``}; the absorb step drops ``citation`` once the
ladder passes and ships the rest.

Run: python scripts/check_citations.py CANDIDATES.yaml --source-root SRC/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class CandidateSchemaError(ValueError):
    """A candidate entry is missing required keys or has the wrong shape."""


@dataclass(frozen=True)
class Citation:
    """A verbatim citation into the pinned engine source tree."""

    file: str
    line_start: int
    line_end: int
    quote: str


@dataclass(frozen=True)
class Candidate:
    """A proposed engine rule awaiting verification.

    ``citation`` is ``None`` for proposals that cite runtime evidence rather than
    source (observed collisions); such candidates are skipped by the checker.
    """

    id: str
    fields: dict[str, Any]
    citation: Citation | None


@dataclass(frozen=True)
class Verdict:
    """The citation-check outcome for one candidate."""

    candidate_id: str
    ok: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.candidate_id, "ok": self.ok, "reason": self.reason}


# --- Candidate loading (malformed entries raise loudly) ---


def _require(cond: bool, where: str, msg: str) -> None:
    if not cond:
        raise CandidateSchemaError(f"candidate {where}: {msg}")


def _parse_candidate(index: int, raw: Any) -> Candidate:
    where = f"#{index}"
    _require(isinstance(raw, dict), where, f"must be a mapping, got {type(raw).__name__}")
    cid = raw.get("id")
    _require(isinstance(cid, str) and bool(cid), where, "missing string 'id'")
    where = f"{cid!r}"

    match = raw.get("match")
    _require(isinstance(match, dict), where, "missing 'match' mapping")
    fields = match.get("fields")
    _require(isinstance(fields, dict) and bool(fields), where, "missing non-empty 'match.fields'")

    if "citation" not in raw:
        return Candidate(id=cid, fields=dict(fields), citation=None)

    cite = raw.get("citation")
    _require(isinstance(cite, dict), where, "missing 'citation' mapping")
    file = cite.get("file")
    _require(isinstance(file, str) and bool(file), where, "citation missing string 'file'")
    lines = cite.get("lines")
    _require(
        isinstance(lines, (list, tuple))
        and len(lines) == 2
        and all(isinstance(n, int) and not isinstance(n, bool) for n in lines),
        where,
        "citation 'lines' must be a two-integer list [start, end]",
    )
    quote = cite.get("quote")
    _require(isinstance(quote, str) and bool(quote.strip()), where, "citation missing 'quote' text")

    return Candidate(
        id=cid,
        fields=dict(fields),
        citation=Citation(file=file, line_start=lines[0], line_end=lines[1], quote=quote),
    )


def load_candidates(path: Path) -> list[Candidate]:
    """Parse and shape-validate a candidates file; raise on any malformed entry."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise CandidateSchemaError("candidates file must be a YAML mapping")
    raw = data.get("candidates")
    if not isinstance(raw, list):
        raise CandidateSchemaError("candidates file must have a 'candidates' list")
    return [_parse_candidate(i, entry) for i, entry in enumerate(raw)]


# --- Claim tokens: what the cited span must entail ---


def _leaf(path: str) -> str:
    """Canonical field path -> the source-level identifier (its last segment)."""
    return path.rsplit(".", 1)[-1]


def _value_tokens(spec: Any) -> list[str]:
    """Literal value tokens a predicate spec claims must appear in the source.

    Operator keys carry no token; booleans (``present`` / ``absent`` flags) have
    no source literal; an ``@field_ref`` contributes the referenced field's leaf.
    """
    if isinstance(spec, dict):
        return [t for v in spec.values() for t in _value_tokens(v)]
    if isinstance(spec, (list, tuple)):
        return [t for v in spec for t in _value_tokens(v)]
    if isinstance(spec, bool):
        return []
    if isinstance(spec, str):
        return [_leaf(spec[1:])] if spec.startswith("@") else [spec]
    if isinstance(spec, (int, float)):
        return [str(spec)]
    return []


def claim_tokens(candidate: Candidate) -> tuple[list[str], list[str]]:
    """Return ``(field_tokens, value_tokens)`` the cited span must contain."""
    field_tokens = [_leaf(p) for p in candidate.fields]
    value_tokens = [t for spec in candidate.fields.values() for t in _value_tokens(spec)]
    return field_tokens, value_tokens


# --- Span matching ---


def _norm_block(text: str) -> list[str]:
    """Content lines of a span, trailing-stripped and indentation-insensitive.

    The dedent is required, not cosmetic: a YAML block scalar cannot preserve a
    span's absolute leading indentation (it strips the block's own common indent
    on parse), so the quoted span is only ever faithful modulo a uniform left
    shift. Comparing after dedent still catches real content and line drift.
    """
    dedented = textwrap.dedent("\n".join(ln.rstrip() for ln in text.splitlines()))
    lines = dedented.splitlines()
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _find_block(file_lines: list[str], quoted: list[str]) -> int | None:
    """1-based line where the normalised ``quoted`` block first occurs in the file."""
    if not quoted:
        return None
    for i in range(len(file_lines) - len(quoted) + 1):
        if _norm_block("\n".join(file_lines[i : i + len(quoted)])) == quoted:
            return i + 1
    return None


def _entailed(token: str, quote: str) -> bool:
    """True iff ``token`` occurs in ``quote`` at a word boundary.

    Boundary, not substring: the field token ``max_tokens`` must NOT be counted
    as present just because ``max_tokens_per_batch`` appears. The lookarounds
    reject an occurrence flanked by a word character, so a token is only entailed
    when it stands as its own identifier/literal.
    """
    return re.search(rf"(?<!\w){re.escape(token)}(?!\w)", quote) is not None


# --- Per-candidate check ---


def check_candidate(candidate: Candidate, source_root: Path) -> Verdict:
    """Verify one candidate's citation against the pinned source tree."""
    cite = candidate.citation
    assert cite is not None, "check_candidate is only called for cited candidates"

    def fail(reason: str) -> Verdict:
        return Verdict(candidate.id, ok=False, reason=reason)

    root = source_root.resolve()
    target = (source_root / cite.file).resolve()
    if not target.is_relative_to(root):
        return fail(f"cited file escapes the source root: {cite.file}")
    if not target.is_file():
        return fail(f"cited file not found under source root: {cite.file}")

    file_lines = target.read_text().splitlines()
    n = len(file_lines)
    if not 1 <= cite.line_start <= cite.line_end <= n:
        return fail(
            f"line range [{cite.line_start}, {cite.line_end}] out of bounds (file has {n} lines)"
        )

    quoted = _norm_block(cite.quote)
    if _norm_block("\n".join(file_lines[cite.line_start - 1 : cite.line_end])) != quoted:
        drift = _find_block(file_lines, quoted)
        if drift is not None:
            return fail(
                f"quoted span does not match lines [{cite.line_start}, {cite.line_end}]; "
                f"verbatim text is at lines [{drift}, {drift + len(quoted) - 1}] (citation drifted)"
            )
        return fail(
            f"quoted span does not match file content at lines [{cite.line_start}, {cite.line_end}]"
        )

    field_tokens, value_tokens = claim_tokens(candidate)
    for tok in field_tokens:
        if not _entailed(tok, cite.quote):
            return fail(f"constrained field {tok!r} not present in cited span")
    for tok in value_tokens:
        if not _entailed(tok, cite.quote):
            return fail(f"constrained value {tok!r} not present in cited span")

    return Verdict(candidate.id, ok=True)


# --- CLI ---


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify engine-rule candidate citations (ladder tier 1)."
    )
    parser.add_argument("candidates", type=Path, help="Candidates YAML file to check.")
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Root of the pinned engine source tree that citations resolve against.",
    )
    args = parser.parse_args(argv)

    if not args.source_root.is_dir():
        parser.error(f"--source-root is not a directory: {args.source_root}")

    candidates = load_candidates(args.candidates)
    cited = [c for c in candidates if c.citation is not None]
    skipped = [c for c in candidates if c.citation is None]
    verdicts = [check_candidate(c, args.source_root) for c in cited]
    failed = [v for v in verdicts if not v.ok]
    report = {
        "verdicts": [v.to_dict() for v in verdicts],
        "checked": len(verdicts),
        "failed": len(failed),
        "skipped_uncited": len(skipped),
        "skipped_ids": [c.id for c in skipped],
    }
    print(json.dumps(report, indent=2))
    if skipped:
        print(
            f"skipped {len(skipped)} uncited candidate(s) (runtime-evidence proposals)",
            file=sys.stderr,
        )
    if failed:
        print(f"\n{len(failed)}/{len(verdicts)} citation checks failed:", file=sys.stderr)
        for v in failed:
            print(f"  - {v.candidate_id}: {v.reason}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
