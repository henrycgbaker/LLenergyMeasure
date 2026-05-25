"""Hybrid strategy (d-ab / d-ac) "deterministic-first → LLM extends" scaffolding.

Per Decision 8 default + Phase 2 scope:
- (a) runs first (deterministic miner output already in
  ``engine_versions/<engine>/v<v>/outputs/``).
- LLM reads (a)'s output + the same engine source as (b).
- LLM produces an EXTENSION yaml (added_by: llm_verifier,
  flagged_for_review: true) + a RECONCILIATION yaml (x-conflict per
  disputed entry).

Phase 2 SCAFFOLDING ONLY:
- Wires the contract; tests the prompt against transformers v4_57_3.
- Full hybrid exploration happens in Phase 3b. The variants (LLM-as-
  orchestrator, LLM-as-curator, etc. per trial_epistemic_framing.md)
  are deferred.

CONTRACT:
- ``run_d_ab_on_transformers_active(out_dir, ...)`` -> emits an
  ``invariants.proposed.yaml`` (the EXTENSION; not a full replacement)
  + a ``reconciliation.yaml`` showing what (a) had, what LLM proposes
  to add, what LLM flagged as spurious.
- The trial_scoring layer scores this output against the reference
  using the same invariant_identity logic; the cell's recall measures
  whether the LLM's EXTENSION items hit reference items that (a) missed.
"""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts.strategies.llm_b_oss import _write_transcript  # noqa: E402
from _spike.scripts.strategies.llm_extractor import (  # noqa: E402
    AnthropicBackend,
    LLMBackend,
    OllamaBackend,
    extract_with_retry,
)
from _spike.scripts.strategies.prompts import HYBRID_EXTENSION_PROMPT_TEMPLATE  # noqa: E402
from _spike.scripts.strategies.transformers_chunker import (  # noqa: E402
    get_active_sources,
    get_sources_from_path,
)


@dataclass
class HybridOutputs:
    """Bundled outputs from a hybrid (d-ab / d-ac) run.

    The schema file is just a copy of (a)'s output for this version
    (hybrid (d) is invariants-only in Phase 2 scope; (a)'s schema
    discovery is already mature for transformers).
    """

    schema_path: Path
    """Copy of (a)'s schema.discovered.json (no LLM contribution to schema in d-ab Phase 2)."""

    invariants_path: Path
    """invariants.proposed.yaml emitted by hybrid: (a)'s entries + LLM extension entries."""

    reconciliation_path: Path
    """reconciliation.yaml: per-entry flags (kept / extended / flagged-spurious)."""

    observations: list[str]
    wall_sec: float


def run_d_ab_on_transformers_active(
    *,
    out_dir: Path,
    engine_version: str = "4.57.3",
    backend: LLMBackend | None = None,
    max_retries: int = 2,
    source_root: Path | None = None,
) -> HybridOutputs:
    """Run hybrid (d-ab) on transformers (active OR bumped version).

    Reads (a)'s output from
    ``engine_versions/transformers/v4_57_3/outputs/`` (the canonical
    deterministic-miner output for the ACTIVE version - always used as
    the deterministic seed) AND the transformers source via inspect
    (active) or AST file-parse (bumped). Prompts the LLM with both and
    asks for:
    1. Extension invariants (a) missed.
    2. Flagged-spurious entries in (a)'s output.
    3. Brief diagnoses.

    Bumped-version semantics (Phase 3a.2): (a)'s seed is the active
    reference (we don't have bumped static-miner output). The source
    block is read from the bumped wheel. The intended signal: how much
    extension the LLM proposes when shown bumped source - this captures
    whether the bumped engine's validate() body differs enough from
    active to surface new invariants OR loses existing ones.

    NOTE: Phase 2 sends the FULL (a) output + the FULL source-summary
    in one prompt. Phase 3 may revisit chunked hybrid prompts if the
    full prompt exceeds context.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if backend is None:
        backend = OllamaBackend()  # default: container Ollama llama3.1:70b

    # 1. Load (a)'s output - always active reference (Phase 3a.2 doesn't
    # have bumped (a) outputs without per-version miner runs).
    a_invariants_path = (
        _PROJECT_ROOT / "engine_versions/transformers/v4_57_3/outputs/invariants.proposed.yaml"
    )
    a_schema_path = (
        _PROJECT_ROOT / "engine_versions/transformers/v4_57_3/outputs/schema.discovered.json"
    )

    if not a_invariants_path.exists():
        raise FileNotFoundError(
            f"hybrid (d-ab) requires (a)'s output; missing: {a_invariants_path}"
        )

    a_invariants_text = a_invariants_path.read_text()
    a_invariants_obj = yaml.safe_load(a_invariants_text)

    # 2. Pull a compact source summary - validate() is the highest-yield
    # surface for missed invariants per Bake-off D (8 silent misses
    # there). Send validate() in full + GenerationConfig.__init__ in
    # full + a one-line note on the other classes.
    sources = (
        get_active_sources() if source_root is None else get_sources_from_path(source_root)
    )
    source_summary = (
        "=== SOURCE: GenerationConfig.validate() ===\n"
        + sources.get("GenerationConfig.validate", "")
        + "\n\n=== SOURCE: GenerationConfig.__init__ (type checks at construction) ===\n"
        + sources.get("GenerationConfig.__init__", "")
        + "\n\n=== COMPANION CLASS NOTES ===\n"
        + "CompileConfig, BitsAndBytesConfig, WatermarkingConfig all "
        + "have validators referenced from validate(). The deterministic "
        + "miner already scanned them; only flag novel patterns.\n\n"
    )

    # 3. Truncate (a)'s output to keep total prompt under context budget.
    # Phase 2: send full (a) output (transformers reference is ~1300
    # lines / ~40k chars - within 32k tokens at default tokeniser).
    # If we exceed, fall back to summary (id + severity only).
    if len(a_invariants_text) > 35000:
        # Compress: just list id + severity + match-field
        compressed = []
        for inv in a_invariants_obj.get("invariants", []):
            ident = inv.get("id", "<no-id>")
            sev = inv.get("severity", "<no-severity>")
            match_fields = (inv.get("match") or {}).get("fields") or {}
            field_repr = next(iter(match_fields.keys()), "<no-field>")
            compressed.append(f"- {ident}: severity={sev}, field={field_repr}")
        a_text_for_prompt = "\n".join(compressed)
    else:
        a_text_for_prompt = a_invariants_text

    prompt = HYBRID_EXTENSION_PROMPT_TEMPLATE.format(
        engine="transformers",
        engine_version=engine_version,
        deterministic_output=a_text_for_prompt,
        source=source_summary,
    )

    # 4. Send to LLM
    extraction = extract_with_retry(
        backend=backend,
        prompt=prompt,
        chunk_name="hybrid_d_ab_extension",
        output_format="yaml",
        json_schema=None,
        max_retries=max_retries,
    )

    # Save transcript (single chunk = the whole hybrid prompt)
    from _spike.scripts.strategies.transformers_chunker import ChunkSpec

    transcript_chunk = ChunkSpec(
        name="hybrid_d_ab_extension",
        description="Hybrid d-ab: deterministic-output + source -> extension",
        source=source_summary[:500] + "...",
        expected_namespaces=["transformers.sampling"],
        expected_kind="invariants",
    )
    _write_transcript(transcripts_dir, "hybrid", transcript_chunk, extraction)

    observations: list[str] = []
    extension_invariants: list[dict] = []
    flagged_spurious: list[dict] = []
    diagnoses: list[dict] = []

    if extraction.parsed is None:
        observations.append(f"hybrid (d-ab) extraction failed: modes={extraction.failure_modes}")
    else:
        parsed = extraction.parsed
        if isinstance(parsed, dict):
            ext = parsed.get("added_by_llm_verifier", []) or []
            spurious = parsed.get("flagged_spurious_in_deterministic", []) or []
            diag = parsed.get("missed_diagnosis", []) or []
            if isinstance(ext, list):
                extension_invariants = [e for e in ext if isinstance(e, dict)]
            if isinstance(spurious, list):
                flagged_spurious = [s for s in spurious if isinstance(s, dict)]
            if isinstance(diag, list):
                diagnoses = [d for d in diag if isinstance(d, dict)]
        else:
            observations.append("hybrid (d-ab): parsed but not a dict; ignored")

    # 5. Build the proposed.yaml: (a)'s entries + LLM extension entries
    merged_invariants = list(a_invariants_obj.get("invariants", []))
    flagged_ids = {f.get("id") for f in flagged_spurious if isinstance(f.get("id"), str)}
    # Annotate flagged entries
    for inv in merged_invariants:
        if inv.get("id") in flagged_ids:
            inv["x-conflict"] = "llm_flagged_spurious"
    # Append extension entries with required metadata
    for ext_inv in extension_invariants:
        ext_inv.setdefault("added_by", "llm_verifier")
        ext_inv["flagged_for_review"] = True
        merged_invariants.append(ext_inv)

    # 6. Write outputs
    inv_envelope = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": "strategy_d_ab_hybrid",
        "invariants": merged_invariants,
    }
    invariants_path = out_dir / "invariants.proposed.yaml"
    invariants_path.write_text(
        yaml.safe_dump(inv_envelope, sort_keys=False, default_flow_style=False)
    )

    reconciliation_envelope = {
        "schema_version": "1.0.0",
        "engine": "transformers",
        "engine_version": engine_version,
        "reconciled_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "added_by_llm_verifier": extension_invariants,
        "flagged_spurious_in_deterministic": flagged_spurious,
        "missed_diagnoses": diagnoses,
    }
    reconciliation_path = out_dir / "reconciliation.yaml"
    reconciliation_path.write_text(
        yaml.safe_dump(reconciliation_envelope, sort_keys=False, default_flow_style=False)
    )

    # 7. Schema: copy (a)'s output (hybrid d-ab Phase 2 doesn't touch schema)
    schema_path = out_dir / "schema.json"
    schema_path.write_text(a_schema_path.read_text())

    observations.append(
        f"strategy_d_ab: extension={len(extension_invariants)}, "
        f"flagged_spurious={len(flagged_spurious)}, "
        f"merged_total={len(merged_invariants)}, "
        f"elapsed={extraction.elapsed_sec:.1f}s"
    )

    return HybridOutputs(
        schema_path=schema_path,
        invariants_path=invariants_path,
        reconciliation_path=reconciliation_path,
        observations=observations,
        wall_sec=extraction.elapsed_sec,
    )


def run_d_ac_on_transformers_active(
    *,
    out_dir: Path,
    engine_version: str = "4.57.3",
    model: str = "claude-sonnet-4-5",
    max_retries: int = 2,
) -> HybridOutputs:
    """Run hybrid (d-ac): same as (d-ab) but with Claude as the LLM.

    Raises ``KeyAbsentError`` if ANTHROPIC_API_KEY is absent.
    """
    backend = AnthropicBackend(model=model)
    return run_d_ab_on_transformers_active(
        out_dir=out_dir,
        engine_version=engine_version,
        backend=backend,
        max_retries=max_retries,
    )


# ---------------------------------------------------------------------------
# Phase 3 Prep-3: vllm + tensorrt hybrid dispatchers
# ---------------------------------------------------------------------------
#
# Generic shape mirroring the transformers version:
# 1. Read (a)'s output from engine_versions/<engine>/<version_slug>/outputs/.
# 2. Pull a compact source summary using the engine's chunker (use the
#    invariants chunks - that's the validate/post_init surface where the
#    LLM should look for what (a) missed).
# 3. Send hybrid extension prompt; parse the 3-section YAML.
# 4. Merge (a)'s entries + LLM extension; flag spurious; emit reconciliation.


def _build_source_summary_from_chunks(chunks: list, max_total_chars: int = 25000) -> str:
    """Concatenate chunk sources into a compact summary for the hybrid prompt.

    Caps at ``max_total_chars`` to keep the total prompt within the 32k
    context window when the (a) output is also inlined.

    Each chunk contributes its name + truncated source (up to a per-chunk
    budget derived from ``max_total_chars / len(chunks)``).
    """
    if not chunks:
        return ""
    per_chunk = max(1500, max_total_chars // max(len(chunks), 1))
    parts: list[str] = []
    for c in chunks:
        # Each chunk's `source` already has the engine context inlined.
        body = (c.source or "")[:per_chunk]
        parts.append(f"=== CHUNK: {c.name} ===\n{body}\n\n")
        if sum(len(p) for p in parts) >= max_total_chars:
            break
    return "".join(parts)


def _run_d_ab_generic(
    *,
    engine: str,
    engine_version: str,
    version_slug: str,
    invariants_chunks_fn: Any,
    out_dir: Path,
    backend: LLMBackend | None = None,
    max_retries: int = 2,
) -> HybridOutputs:
    """Engine-agnostic hybrid (d-ab) driver.

    Reads (a)'s output for the cell + a compact source summary from the
    engine's chunker; sends one hybrid extension prompt; merges.

    For Phase 3a.1 scope: active version per engine; version_slug encodes
    the engine_versions/<engine>/<version_slug>/outputs/ location.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if backend is None:
        backend = OllamaBackend()

    # 1. Load (a)'s output for this cell
    a_invariants_path = (
        _PROJECT_ROOT / f"engine_versions/{engine}/{version_slug}/outputs/invariants.proposed.yaml"
    )
    a_schema_path = (
        _PROJECT_ROOT / f"engine_versions/{engine}/{version_slug}/outputs/schema.discovered.json"
    )

    if not a_invariants_path.exists():
        raise FileNotFoundError(
            f"hybrid (d-ab) requires (a)'s output for {engine}/{version_slug}; "
            f"missing: {a_invariants_path}"
        )

    a_invariants_text = a_invariants_path.read_text()
    a_invariants_obj = yaml.safe_load(a_invariants_text)

    # 2. Pull compact source summary from the engine's chunker (use the
    #    invariants chunks - that's the validate/post_init surface).
    inv_chunks = invariants_chunks_fn()
    source_summary = _build_source_summary_from_chunks(inv_chunks, max_total_chars=22000)

    # 3. Compress (a)'s output if it's huge (vllm has 10 invariants, fits
    #    easily; tensorrt has 3; transformers has 41 -> already-tested branch).
    if len(a_invariants_text) > 35000:
        compressed = []
        for inv in a_invariants_obj.get("invariants", []) or []:
            ident = inv.get("id", "<no-id>")
            sev = inv.get("severity", "<no-severity>")
            match_fields = (inv.get("match") or {}).get("fields") or {}
            field_repr = next(iter(match_fields.keys()), "<no-field>")
            compressed.append(f"- {ident}: severity={sev}, field={field_repr}")
        a_text_for_prompt = "\n".join(compressed)
    else:
        a_text_for_prompt = a_invariants_text

    prompt = HYBRID_EXTENSION_PROMPT_TEMPLATE.format(
        engine=engine,
        engine_version=engine_version,
        deterministic_output=a_text_for_prompt,
        source=source_summary,
    )

    # 4. Send to LLM
    extraction = extract_with_retry(
        backend=backend,
        prompt=prompt,
        chunk_name=f"hybrid_d_ab_{engine}_extension",
        output_format="yaml",
        json_schema=None,
        max_retries=max_retries,
    )

    # Save transcript (single chunk = the whole hybrid prompt)
    from _spike.scripts.strategies.transformers_chunker import ChunkSpec as _TxChunkSpec

    transcript_chunk = _TxChunkSpec(
        name=f"hybrid_d_ab_{engine}_extension",
        description=f"Hybrid d-ab on {engine}: (a) output + source -> extension",
        source=source_summary[:500] + "...",
        expected_namespaces=[f"{engine}.*"],
        expected_kind="invariants",
    )
    _write_transcript(transcripts_dir, "hybrid", transcript_chunk, extraction)

    observations: list[str] = []
    extension_invariants: list[dict] = []
    flagged_spurious: list[dict] = []
    diagnoses: list[dict] = []

    if extraction.parsed is None:
        observations.append(
            f"hybrid (d-ab) for {engine}: extraction failed; modes={extraction.failure_modes}"
        )
    else:
        parsed = extraction.parsed
        if isinstance(parsed, dict):
            ext = parsed.get("added_by_llm_verifier", []) or []
            spurious = parsed.get("flagged_spurious_in_deterministic", []) or []
            diag = parsed.get("missed_diagnosis", []) or []
            if isinstance(ext, list):
                extension_invariants = [e for e in ext if isinstance(e, dict)]
            if isinstance(spurious, list):
                flagged_spurious = [s for s in spurious if isinstance(s, dict)]
            if isinstance(diag, list):
                diagnoses = [d for d in diag if isinstance(d, dict)]
        else:
            observations.append(f"hybrid (d-ab) for {engine}: parsed but not a dict; ignored")

    # 5. Build the proposed.yaml: (a)'s entries + LLM extension entries
    merged_invariants = list((a_invariants_obj or {}).get("invariants", []) or [])
    flagged_ids = {f.get("id") for f in flagged_spurious if isinstance(f.get("id"), str)}
    for inv in merged_invariants:
        if inv.get("id") in flagged_ids:
            inv["x-conflict"] = "llm_flagged_spurious"
    for ext_inv in extension_invariants:
        ext_inv.setdefault("added_by", "llm_verifier")
        ext_inv["flagged_for_review"] = True
        merged_invariants.append(ext_inv)

    # 6. Write outputs
    inv_envelope = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": f"strategy_d_ab_hybrid_{engine}",
        "invariants": merged_invariants,
    }
    invariants_path = out_dir / "invariants.proposed.yaml"
    invariants_path.write_text(
        yaml.safe_dump(inv_envelope, sort_keys=False, default_flow_style=False)
    )

    reconciliation_envelope = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": engine_version,
        "reconciled_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "added_by_llm_verifier": extension_invariants,
        "flagged_spurious_in_deterministic": flagged_spurious,
        "missed_diagnoses": diagnoses,
    }
    reconciliation_path = out_dir / "reconciliation.yaml"
    reconciliation_path.write_text(
        yaml.safe_dump(reconciliation_envelope, sort_keys=False, default_flow_style=False)
    )

    # 7. Schema: copy (a)'s output (hybrid d-ab is invariants-only).
    schema_path = out_dir / "schema.json"
    if a_schema_path.exists():
        schema_path.write_text(a_schema_path.read_text())
    else:
        schema_path.write_text("{}")
        observations.append(
            f"hybrid (d-ab) for {engine}: (a) schema missing at {a_schema_path}; "
            "wrote empty schema.json"
        )

    observations.append(
        f"strategy_d_ab on {engine}: extension={len(extension_invariants)}, "
        f"flagged_spurious={len(flagged_spurious)}, "
        f"merged_total={len(merged_invariants)}, "
        f"elapsed={extraction.elapsed_sec:.1f}s"
    )

    return HybridOutputs(
        schema_path=schema_path,
        invariants_path=invariants_path,
        reconciliation_path=reconciliation_path,
        observations=observations,
        wall_sec=extraction.elapsed_sec,
    )


def run_d_ab_on_vllm_active(
    *,
    out_dir: Path,
    engine_version: str = "0.7.3",
    backend: LLMBackend | None = None,
    max_retries: int = 2,
) -> HybridOutputs:
    """Run hybrid (d-ab) on vllm active version (0.7.3)."""
    from _spike.scripts.strategies.vllm_chunker import invariants_chunks as vllm_inv_chunks

    return _run_d_ab_generic(
        engine="vllm",
        engine_version=engine_version,
        version_slug="v0_7_3",
        invariants_chunks_fn=vllm_inv_chunks,
        out_dir=out_dir,
        backend=backend,
        max_retries=max_retries,
    )


def run_d_ab_on_tensorrt_active(
    *,
    out_dir: Path,
    engine_version: str = "0.21.0",
    backend: LLMBackend | None = None,
    max_retries: int = 2,
) -> HybridOutputs:
    """Run hybrid (d-ab) on tensorrt_llm active version (0.21.0)."""
    from _spike.scripts.strategies.tensorrt_chunker import (
        invariants_chunks as trt_inv_chunks,
    )

    return _run_d_ab_generic(
        engine="tensorrt",
        engine_version=engine_version,
        version_slug="v0_21_0",
        invariants_chunks_fn=trt_inv_chunks,
        out_dir=out_dir,
        backend=backend,
        max_retries=max_retries,
    )
