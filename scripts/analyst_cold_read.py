#!/usr/bin/env python3
"""Analyst cold read (proposer S2 of the engine-rule verification ladder).

An LLM analyst reads the pinned engine source cold - no prior corpus, no
seeding - and proposes every config-validation constraint it can see, each a
candidate carrying a mandatory source citation. This is the recall-maximising
proposer; the deterministic ladder (citation check -> construction probe ->
identity probe) earns a candidate its place in the shipped corpus. At this
precision (~77-80% content-correct on the parity experiment) the analyst never
self-certifies. A pure completion loop over line-numbered source chunks: one
exhaustive-extraction prompt, self-consistency union across samples, a salvage
parser for truncated output. No agentic access, no repair loops, no retries.
Design ratified by the 2026-07-02 OSS analyst parity experiment
(qwen2.5-coder:32b measured against the frontier bump-analyst reference).

--source-root is the engine's top-level package directory at the pinned version
(for vllm, the ``vllm/`` folder) - an explicit maintainer input, as it is for
scripts/check_citations.py, with no downloader built in. Obtain it the cheapest
defensible way, e.g. ``pip download --no-binary :all: --no-deps vllm==0.19.1``
then unpack the sdist, or copy the package out of the pinned engine image.

Output goes to ``<pool-root>/<engine>/v<version>/candidates/analyst_cold_read.yaml``
in the unified candidate schema, consumable UNCHANGED by check_citations.py
(tier 1) and probe_candidates.py (tiers 2-3). This NEVER writes the shipped
``src/llenergymeasure/engines/<engine>/rules.yaml`` corpora.

Run: python scripts/analyst_cold_read.py --engine vllm --source-root SRC/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("analyst_cold_read")

SCHEMA_VERSION = "1.0.0"
SOURCE = "analyst_cold_read"

DEFAULT_MODEL = "qwen2.5-coder:32b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_SAMPLES = 3

CHUNK_CHAR_BUDGET = 60000  # ~18-20k tokens of source; leaves output room in 32k ctx
OVERLAP_LINES = 40
NUM_PREDICT = 8192
NUM_CTX = 32768
QUOTE_RADIUS = 3  # lines of context each side of a cited line in the emitted quote
MAX_SPLIT_DEPTH = 3  # cap-hit chunks are halved and re-run, up to this recursion depth

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_VERSIONS = REPO_ROOT / "engine_versions"
MANIFEST_NAME = "analyst_clusters.yaml"


@dataclass(frozen=True)
class GenResult:
    """One model completion: the raw text plus the stop signals we act on."""

    text: str
    done_reason: str | None
    eval_count: int | None


# A generator maps (prompt, temperature) -> GenResult. The real one calls
# Ollama; tests inject a deterministic stub. This IS the only client boundary -
# there is no frontier-model abstraction layer; the candidate file is the
# interface a different proposer would produce.
Generator = Callable[[str, float], GenResult]


# Source chunking: line numbers preserved so citations resolve.


@dataclass(frozen=True)
class Segment:
    """A contiguous window of one source file, tagged with its real start line."""

    filename: str  # source-root-relative path, e.g. "config/parallel.py"
    start_line: int  # 1-based line number of ``lines[0]`` in the real file
    lines: list[str]

    @property
    def end_line(self) -> int:
        return self.start_line + len(self.lines) - 1

    def numbered(self) -> str:
        return "\n".join(f"{self.start_line + i:6d}  {line}" for i, line in enumerate(self.lines))


@dataclass(frozen=True)
class Chunk:
    """One prompt's worth of source: one or more segments under a cluster."""

    cluster: str
    chunk_id: str
    segments: list[Segment]

    def numbered_source(self) -> str:
        return "\n\n".join(seg.numbered() for seg in self.segments)

    def label(self) -> str:
        return ", ".join(
            f"{seg.filename} (lines {seg.start_line}-{seg.end_line})" for seg in self.segments
        )

    def class_names(self) -> list[str]:
        """Top-level class names declared anywhere in this chunk, first-seen order."""
        names: list[str] = []
        for seg in self.segments:
            for line in seg.lines:
                match = re.match(r"class\s+(\w+)", line)
                if match and match.group(1) not in names:
                    names.append(match.group(1))
        return names


def _window_file(filename: str, lines: list[str]) -> list[Segment]:
    """Split one file into char-budget-limited overlapping segments."""
    segments: list[Segment] = []
    n = len(lines)
    i = 0
    while i < n:
        j, chars = i, 0
        while j < n and chars < CHUNK_CHAR_BUDGET:
            chars += len(lines[j]) + 8
            j += 1
        segments.append(Segment(filename, i + 1, lines[i:j]))
        if j >= n:
            break
        step = j - OVERLAP_LINES
        i = step if step > i else j  # overlap, but always make forward progress
    return segments


def _pack_cluster(cluster: str, segments: list[Segment]) -> list[Chunk]:
    """Pack one cluster's windows into char-budget chunks in order."""
    chunks: list[Chunk] = []
    buf: list[Segment] = []
    buf_chars = 0
    for seg in segments:
        seg_chars = len(seg.numbered())
        if buf and buf_chars + seg_chars > CHUNK_CHAR_BUDGET:
            chunks.append(Chunk(cluster, f"{cluster}.{len(chunks)}", buf))
            buf, buf_chars = [], 0
        buf.append(seg)
        buf_chars += seg_chars
    if buf:
        chunks.append(Chunk(cluster, f"{cluster}.{len(chunks)}", buf))
    return chunks


def build_chunks(cluster_files: dict[str, list[str]], sources: dict[str, list[str]]) -> list[Chunk]:
    """Pack per-file windows into char-budget chunks, one chunk list per cluster."""
    chunks: list[Chunk] = []
    for cluster, files in cluster_files.items():
        segments: list[Segment] = []
        for filename in files:
            file_lines = sources.get(filename)
            if file_lines is not None:
                segments += _window_file(filename, file_lines)
        chunks += _pack_cluster(cluster, segments)
    return chunks


def split_chunk(chunk: Chunk) -> list[Chunk] | None:
    """Halve a chunk that hit the output cap; real line numbers are preserved.

    Multi-segment chunks split down the segment list; a single-segment chunk
    splits that segment's lines in half with the standard overlap. Returns None
    when the chunk is too small to split further (accept truncation then).
    """
    if len(chunk.segments) > 1:
        mid = len(chunk.segments) // 2
        halves = [chunk.segments[:mid], chunk.segments[mid:]]
    else:
        seg = chunk.segments[0]
        if len(seg.lines) <= 2 * OVERLAP_LINES:
            return None
        mid = len(seg.lines) // 2
        left = Segment(seg.filename, seg.start_line, seg.lines[:mid])
        right_start = seg.start_line + mid - OVERLAP_LINES
        right = Segment(seg.filename, right_start, seg.lines[mid - OVERLAP_LINES :])
        halves = [[left], [right]]
    return [
        Chunk(chunk.cluster, f"{chunk.chunk_id}.{tag}", segs)
        for tag, segs in zip(("a", "b"), halves, strict=True)
    ]


# Prompt: single exhaustive-extraction instruction with a class-coverage checklist.
PROMPT_TEMPLATE = """You are a meticulous configuration-constraint analyst reading {engine} {version} source.

Your job: from the source file(s) below, extract EVERY validation constraint that {engine}
enforces on user-supplied configuration fields. Be exhaustive - every guard, bound, enum check,
cross-field relation, and silent-normalisation counts. Do not skip any.

Constraint kinds (use exactly these strings for "kind"):
  single_field_bound - one field, numeric/range/type predicate (e.g. n >= 1, temperature >= 0.0)
  single_field_enum  - one field must be in a set of allowed values (e.g. kv_role in {{producer,consumer}})
  cross_field        - a relation between two or more fields (e.g. min_tokens <= max_tokens; A requires B)
  dormancy           - a field is silently normalised/reset/clamped without raising (e.g. seed==-1 -> None)
  other              - env-var-gated, platform-gated, or requires a live model/hf-config to evaluate

For EACH constraint emit an object with these keys:
  "kind":                one of the five strings above
  "fields":              list of the config field name(s) involved (bare names, strip any "self." prefix)
  "predicate":           a normalised boolean expression for the VALID/legal condition
                         (e.g. "n >= 1", "-2.0 <= presence_penalty <= 2.0", "min_tokens <= max_tokens").
                         For dormancy, describe the normalisation (e.g. "seed == -1 -> None").
  "condition":           the class / method context in which it applies (e.g. "SamplingParams._verify_args")
  "citation":            "FILENAME.py:LINE" using the exact 1-based line numbers shown in the left margin
                         below. Cite the line where the guard/raise/assignment actually appears.
  "severity_suggestion": "error" (raises), "warning" (logs+continues), or "dormant" (silent normalise)
  "rationale":           one short sentence: what is enforced and what happens on violation.

Rules:
- Cover ALL classes in the file, not just the first. Include pydantic Field(ge=/gt=/le=/lt=/Literal[...])
  bounds as single_field_bound / single_field_enum with the field's declared line as citation.
- Cross-field constraints are the highest priority; do not down-cast a two-field relation into a
  single-field bound.
- Include silent normalisations (dormancy) - assignments in __post_init__ that overwrite a user value
  without raising.
- Do NOT invent constraints not present in the source. Only emit what the code actually enforces.
- Output STRICT JSON only, an object of the form {{"rules": [ ... ]}} with no prose before or after.

CLASS COVERAGE CHECKLIST - the following classes are declared in this source:
{class_checklist}
Scan EVERY listed class before you finish. For each class emit at least one constraint, or, if a class
genuinely enforces nothing on user config, emit one entry with "kind": "other", the class name in
"fields", and rationale "no user-config constraint in <ClassName>". Do not silently skip a class.

SOURCE ({label}):
{source}
"""


def render_prompt(chunk: Chunk, engine: str, version: str) -> str:
    classes = chunk.class_names()
    checklist = (
        "\n".join(f"  - {name}" for name in classes) if classes else "  (no top-level classes)"
    )
    return PROMPT_TEMPLATE.format(
        engine=engine,
        version=version,
        class_checklist=checklist,
        label=chunk.label(),
        source=chunk.numbered_source(),
    )


# Ollama client: the single proposer boundary.


def ollama_generator(host: str, model: str) -> Generator:
    """A Generator bound to a live Ollama endpoint (format=json completion)."""

    def generate(prompt: str, temperature: float) -> GenResult:
        body = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "num_ctx": NUM_CTX,
                "temperature": temperature,
                "num_predict": NUM_PREDICT,
                "top_p": 0.9,
            },
        }
        request = urllib.request.Request(
            f"{host.rstrip('/')}/api/generate",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = json.load(response)
        return GenResult(
            text=payload.get("response", ""),
            done_reason=payload.get("done_reason"),
            eval_count=payload.get("eval_count"),
        )

    return generate


def hit_cap(result: GenResult) -> bool:
    """True when the completion stopped by exhausting the num_predict budget."""
    if result.done_reason == "length":
        return True
    return result.eval_count is not None and result.eval_count >= NUM_PREDICT


# Salvage parser: truncated JSON still yields its complete rule objects.


def salvage_rules(text: str) -> list[dict[str, Any]]:
    """Parse ``{"rules": [...]}``; on truncation, recover the complete objects.

    A clean parse returns the ``rules`` list. When output is cut at the token
    cap json.loads fails, so we scan for balanced ``{...}`` objects and keep the
    rule-shaped ones that parse.
    """
    try:
        obj = json.loads(text)
        rules = obj.get("rules", obj) if isinstance(obj, dict) else obj
        return [r for r in rules if isinstance(r, dict)] if isinstance(rules, list) else []
    except (json.JSONDecodeError, AttributeError):
        pass

    # Truncated output: the wrapper object never closes, but the complete rule
    # objects inside it do. Track every balanced {...} with a start-index stack
    # (respecting string escapes) and keep the rule-shaped ones.
    rules: list[dict[str, Any]] = []
    starts: list[int] = []
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char == "{":
            starts.append(index)
        elif char == "}" and starts:
            start = starts.pop()
            try:
                obj = json.loads(text[start : index + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "rules" not in obj and ("fields" in obj or "kind" in obj):
                rules.append(obj)
    return rules


# LLM rule -> unified candidate.


def _leaf(field: str) -> str:
    return re.sub(r"^self\.", "", field).rsplit(".", 1)[-1]


def _coerce_scalar(token: str) -> Any:
    """Parse a source-verbatim literal token into int/float/bool/None/str."""
    token = token.strip().strip("'\"")
    low = token.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


# valid-predicate operator -> the operator whose match is the VIOLATION we probe.
_NEGATED_OP = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


def predicate_to_match(
    kind: str, fields: list[str], predicate: str, severity: str
) -> tuple[dict[str, Any], list[str]]:
    """Translate the analyst's free-text predicate into a match.fields spec.

    match.fields encodes the config the rule ACTS on, in the corpus polarity:
    for an error the VIOLATING side (the valid predicate negated), for a dormant
    candidate the normalised field(s). Only clean single-field comparison/enum
    predicates are structured; anything compound, cross-field, or arithmetic
    falls back to a ``present`` spec (the field still grounds the citation, the
    raw predicate is kept in provenance). Returns (match_fields, normalised).
    """
    normalised = [_leaf(f) for f in fields] if severity == "dormant" else []
    leaves = [_leaf(f) for f in fields]

    if (
        severity == "error"
        and len(leaves) == 1
        and " or " not in predicate
        and " and " not in predicate
    ):
        leaf = leaves[0]
        cmp = re.fullmatch(
            rf"\s*{re.escape(leaf)}\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*", predicate
        )
        if cmp:
            return {leaf: {_NEGATED_OP[cmp.group(1)]: _coerce_scalar(cmp.group(2))}}, normalised
        enum = re.fullmatch(rf"\s*{re.escape(leaf)}\s+in\s+[\[{{(]([^\]}})]+)[\]}})]\s*", predicate)
        if enum:
            members = [_coerce_scalar(tok) for tok in enum.group(1).split(",") if tok.strip()]
            if members:
                return {leaf: {"not_in": members}}, normalised

    return {leaf: {"present": True} for leaf in leaves}, normalised


def resolve_citation(
    chunk: Chunk, sources: dict[str, list[str]], raw: str
) -> dict[str, Any] | None:
    """Resolve ``FILE.py:LINE`` to a {file, lines, quote} citation, or None.

    Basename is matched against the chunk's own segments first, then the whole
    pool. The quote is a verbatim window from the real file, so span-match
    passes by construction and tier 1 adjudicates only token entailment.
    """
    match = re.search(r"([\w./-]+\.py)\s*:\s*(\d+)", raw)
    if not match:
        return None
    basename = Path(match.group(1)).name
    line = int(match.group(2))

    relpath: str | None = None
    for seg in chunk.segments:
        if Path(seg.filename).name == basename:
            relpath = seg.filename
            break
    if relpath is None:
        hits = [path for path in sources if Path(path).name == basename]
        if len(hits) == 1:
            relpath = hits[0]
    if relpath is None:
        return None

    file_lines = sources.get(relpath)
    if file_lines is None or not 1 <= line <= len(file_lines):
        return None
    start = max(1, line - QUOTE_RADIUS)
    end = min(len(file_lines), line + QUOTE_RADIUS)
    return {"file": relpath, "lines": [start, end], "quote": "\n".join(file_lines[start - 1 : end])}


def _slug(text: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", text)).strip("_").lower()


def build_candidate(
    chunk: Chunk,
    sources: dict[str, list[str]],
    rule: dict[str, Any],
    *,
    engine: str,
    version: str,
    model: str,
    samples: int,
    run_date: str,
) -> dict[str, Any] | str:
    """Assemble one unified candidate, or the drop reason when unusable.

    Returns "no_field" when the rule names no usable field, "unresolved_citation"
    when its citation points at no real file:line (a hallucination); run_analyst
    counts these as the per-bump analyst-health signal. All the analyst said is
    kept under provenance. A "warning" severity_suggestion (logs+continues)
    deliberately still mints an error claim: recall-preserving, and contained by
    the ladder - a warn-only guard never raises at construction, so the
    candidate stays unconfirmed and is never promoted. Do not "fix" it here.
    """
    fields = [f for f in (rule.get("fields") or []) if isinstance(f, str) and _leaf(f)]
    if not fields:
        return "no_field"
    kind = str(rule.get("kind") or "other")
    predicate = str(rule.get("predicate") or "")
    suggestion = str(rule.get("severity_suggestion") or "")
    severity = "dormant" if kind == "dormancy" or suggestion == "dormant" else "error"

    citation = resolve_citation(chunk, sources, str(rule.get("citation") or ""))
    if citation is None:
        return "unresolved_citation"

    match_fields, normalised = predicate_to_match(kind, fields, predicate, severity)
    digest = hashlib.sha1(
        f"{severity}|{sorted(match_fields)}|{citation['file']}|{predicate}".encode()
    ).hexdigest()[:8]
    candidate: dict[str, Any] = {
        "id": f"{engine}_analyst_{severity}_{_slug('_'.join(_leaf(f) for f in fields))}_{digest}",
        "engine": engine,
        "engine_version": version,
        "severity": severity,
        "match": {"fields": match_fields},
    }
    if normalised:
        candidate["normalised_fields"] = normalised
    candidate["citation"] = citation
    candidate["provenance"] = {
        "source": SOURCE,
        "model": model,
        "samples": samples,
        "date": run_date,
        "kind": kind,
        "predicate": predicate,
        "condition": str(rule.get("condition") or ""),
        "rationale": str(rule.get("rationale") or ""),
        "severity_suggestion": suggestion,
    }
    candidate["verdict"] = {"status": "unverified", "tier": None, "date": None, "evidence": None}
    return candidate


def dedup_key(candidate: dict[str, Any]) -> str:
    """Union key: two samples proposing the same constraint at the same cite collapse."""
    citation = candidate.get("citation") or {}
    predicate = (candidate.get("provenance") or {}).get("predicate", "")
    return "|".join(
        [
            candidate["severity"],
            ",".join(sorted(candidate["match"]["fields"])),
            str(citation.get("file")),
            str(citation.get("lines")),
            re.sub(r"\s+", "", predicate.lower()),
        ]
    )


# Sampling orchestration.


def sample_chunk(
    chunk: Chunk,
    generate: Generator,
    samples: int,
    *,
    engine: str,
    version: str,
    _depth: int = 0,
) -> list[tuple[Chunk, dict[str, Any]]]:
    """One temp-0 pass plus (samples-1) temp-0.6 passes; split on a cap hit.

    Returns (chunk, raw-llm-rule) pairs. When the temp-0 pass hits the output
    cap the chunk is halved and each half re-run in full, not read truncated.
    """
    prompt = render_prompt(chunk, engine, version)
    first = generate(prompt, 0.0)
    if hit_cap(first) and _depth < MAX_SPLIT_DEPTH:
        parts = split_chunk(chunk)
        if parts:
            collected: list[tuple[Chunk, dict[str, Any]]] = []
            for part in parts:
                collected += sample_chunk(
                    part, generate, samples, engine=engine, version=version, _depth=_depth + 1
                )
            return collected

    pairs = [(chunk, rule) for rule in salvage_rules(first.text)]
    for _ in range(max(0, samples - 1)):
        extra = generate(prompt, 0.6)
        pairs += [(chunk, rule) for rule in salvage_rules(extra.text)]
    return pairs


def run_analyst(
    chunks: list[Chunk],
    sources: dict[str, list[str]],
    generate: Generator,
    *,
    engine: str,
    version: str,
    model: str,
    samples: int,
    run_date: str,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Cold-read every chunk, build candidates, and union-dedup the pool.

    Also returns the drop counters (no_field / unresolved_citation /
    dedup_collapsed): the per-bump analyst-health signal - citation-resolution
    rate is the parity experiment's headline quality metric - which the absorb
    workflow reads to judge whether a cold read degraded.
    """
    seen: set[str] = set()
    drops: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    for position, chunk in enumerate(chunks, start=1):
        logger.info("chunk %d/%d %s", position, len(chunks), chunk.chunk_id)
        for source_chunk, rule in sample_chunk(
            chunk, generate, samples, engine=engine, version=version
        ):
            candidate = build_candidate(
                source_chunk,
                sources,
                rule,
                engine=engine,
                version=version,
                model=model,
                samples=samples,
                run_date=run_date,
            )
            if isinstance(candidate, str):
                drops[candidate] += 1
                continue
            key = dedup_key(candidate)
            if key in seen:
                drops["dedup_collapsed"] += 1
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates, drops


# Manifest, version, source loading, emission.


def load_manifest(engine: str) -> dict[str, list[str]]:
    """Read the per-engine cluster manifest; fail clearly when it is absent."""
    path = ENGINE_VERSIONS / engine / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"no analyst cluster manifest for engine {engine!r} at {path}. "
            f"Only vllm ships one so far; author {path} before cold-reading {engine}."
        )
    document = yaml.safe_load(path.read_text())
    clusters = document.get("clusters") if isinstance(document, dict) else None
    if not isinstance(clusters, dict) or not clusters:
        raise ValueError(f"manifest {path} has no non-empty 'clusters' mapping")
    return {str(name): [str(p) for p in patterns] for name, patterns in clusters.items()}


def resolve_version(engine: str) -> str:
    """Resolve the pinned engine version from engine_versions/<engine>/current.yaml."""
    path = ENGINE_VERSIONS / engine / "current.yaml"
    document = yaml.safe_load(path.read_text())
    version = (document.get("library") or {}).get("current_version")
    if not version:
        raise ValueError(f"{path} has no library.current_version")
    return str(version)


def load_sources(
    source_root: Path, manifest: dict[str, list[str]]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Expand manifest patterns and read every file into line lists.

    Returns ``(cluster_files, sources)``: cluster -> ordered relative paths, and
    relative path -> line list. Missing files are warned and skipped, not fatal.
    """
    cluster_files: dict[str, list[str]] = {}
    sources: dict[str, list[str]] = {}
    for cluster, patterns in manifest.items():
        resolved: list[str] = []
        for pattern in patterns:
            matches = (
                sorted(source_root.glob(pattern)) if "*" in pattern else [source_root / pattern]
            )
            for path in matches:
                if not path.is_file():
                    logger.warning("skipping missing source file: %s", pattern)
                    continue
                rel = path.relative_to(source_root).as_posix()
                if rel not in sources:
                    sources[rel] = path.read_text().splitlines()
                resolved.append(rel)
        if resolved:
            cluster_files[cluster] = resolved
    return cluster_files, sources


_POOL_HEADER = (
    "# Analyst cold-read candidate rules (proposer S2).\n"
    "# Emitted by scripts/analyst_cold_read.py: an LLM analyst's exhaustive cold\n"
    "# read of the pinned engine source, one cited candidate per constraint.\n"
    "# UNVERIFIED candidates for the verification ladder. Do NOT hand-copy into\n"
    "# the shipped src/llenergymeasure/engines/<engine>/rules.yaml corpora.\n"
)


def _pool_path(pool_root: Path, engine: str, version: str) -> Path:
    version_dir = "v" + re.sub(r"[^0-9a-zA-Z]+", "_", version).strip("_")
    return pool_root / engine / version_dir / "candidates" / "analyst_cold_read.yaml"


def write_candidates(
    candidates: list[dict[str, Any]],
    pool_root: Path,
    *,
    engine: str,
    version: str,
    model: str,
    samples: int,
    run_date: str,
) -> Path:
    """Write the version-scoped candidate pool file and return its path."""
    document = {
        "schema_version": SCHEMA_VERSION,
        "source": SOURCE,
        "engine": engine,
        "engine_version": version,
        "model": model,
        "samples": samples,
        "generated_at": run_date,
        "candidates": candidates,
    }
    path = _pool_path(pool_root, engine, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(document, sort_keys=False, default_flow_style=False)
    path.write_text(_POOL_HEADER + body)
    return path


# CLI.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True, choices=("vllm", "transformers", "tensorrt"))
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Engine package directory at the pinned version (citations resolve against it)",
    )
    parser.add_argument(
        "--pool-root",
        type=Path,
        default=ENGINE_VERSIONS,
        help="Root of the version-scoped candidate pool (default: engine_versions)",
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
    )
    parser.add_argument(
        "--date", default=None, help="Provenance date (default: today; set for reproducible output)"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not args.source_root.is_dir():
        print(f"error: --source-root is not a directory: {args.source_root}", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.engine)
        version = resolve_version(args.engine)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    run_date = args.date or date.today().isoformat()
    cluster_files, sources = load_sources(args.source_root, manifest)
    chunks = build_chunks(cluster_files, sources)
    logger.info(
        "engine=%s version=%s chunks=%d samples=%d", args.engine, version, len(chunks), args.samples
    )

    started = time.time()
    generate = ollama_generator(args.ollama_host, args.model)
    candidates, drops = run_analyst(
        chunks,
        sources,
        generate,
        engine=args.engine,
        version=version,
        model=args.model,
        samples=args.samples,
        run_date=run_date,
    )
    path = write_candidates(
        candidates,
        args.pool_root,
        engine=args.engine,
        version=version,
        model=args.model,
        samples=args.samples,
        run_date=run_date,
    )
    by_kind = Counter(c["provenance"]["kind"] for c in candidates)
    drop_note = " ".join(f"{reason}={count}" for reason, count in sorted(drops.items())) or "none"
    print(
        f"analyst cold read: {len(candidates)} candidates "
        f"({dict(sorted(by_kind.items()))}), drops: {drop_note}, "
        f"in {time.time() - started:.0f}s\nwrote {path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
