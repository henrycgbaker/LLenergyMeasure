"""Strategy (b) "pure OSS LLM" executor.

Per Phase 2 design:
- Uses container Ollama on port 11435 with llama3.1:70b at 32k context.
- Chunks source via ``transformers_chunker.schema_chunks()`` and
  ``invariants_chunks()``.
- Sends each chunk through ``llm_extractor.extract_with_retry()`` with
  JSON-mode for schema, plain-text for invariants (YAML).
- Merges per-chunk outputs, deduplicates, applies the internal-plumbing
  post-filter.
- Emits ``schema.json`` + ``invariants.proposed.yaml`` matching the
  canonical envelope shapes used by trial_scoring.

CONTRACT: callable as ``run_b_on_transformers_active(out_dir, transcripts_dir, backend=...)``.
The function emits the two output files and returns
``(StrategyOutputPaths, observations)``.

For (c) and (d) reuse: the schema + invariant extraction loops are
backend-agnostic - swap ``backend=OllamaBackend()`` for an
``AnthropicBackend()`` to get (c) behaviour with the same prompts.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root sits three parents up from _spike/scripts/strategies/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts.strategies.llm_extractor import (  # noqa: E402
    ChunkExtraction,
    LLMBackend,
    OllamaBackend,
    extract_with_retry,
    filter_internal_plumbing,
    is_internal_plumbing_field,
)
from _spike.scripts.strategies.prompts import (  # noqa: E402
    INVARIANTS_EXTEND_PROMPT_TEMPLATE,
    INVARIANTS_PROMPT_TEMPLATE,
    INVARIANTS_VERIFY_PROMPT_TEMPLATE,
    SCHEMA_JSON_SCHEMA,
    SCHEMA_PROMPT_TEMPLATE,
)
from _spike.scripts.strategies.transformers_chunker import (  # noqa: E402
    ChunkSpec,
    invariants_chunks,
    schema_chunks,
)


@dataclass
class StrategyBOutputs:
    """Bundled outputs from a (b) run.

    The two file paths feed into trial_scoring; the observations list
    captures adjacent observations + failure modes for the cell record.
    """

    schema_path: Path
    invariants_path: Path
    observations: list[str]
    schema_wall_sec: float
    invariants_wall_sec: float


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def extract_schema(
    *,
    backend: LLMBackend,
    chunks: list[ChunkSpec],
    engine: str,
    engine_version: str,
    transcripts_dir: Path,
    max_retries: int = 2,
) -> tuple[dict[str, Any], float, list[str]]:
    """Run each schema chunk through the backend; merge into a single
    schema envelope matching the canonical schema.discovered.json shape.

    Returns ``(schema_dict, total_wall_sec, observations)``.

    Merging rules:
    - For each chunk's ``chunk_fields``, group by ``namespace``.
    - Within a namespace, last-write-wins on field name (chunks rarely
      overlap so this is mostly defensive).
    - Drop fields matching ``is_internal_plumbing_field``.
    """
    engine_params: dict[str, Any] = {}
    sampling_params: dict[str, Any] = {}
    defs: dict[str, dict[str, Any]] = {}

    observations: list[str] = []
    total_wall = 0.0
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        prompt = SCHEMA_PROMPT_TEMPLATE.format(
            engine=engine,
            engine_version=engine_version,
            chunk_name=chunk.name,
            namespaces=", ".join(chunk.expected_namespaces),
            source=chunk.source,
        )

        extraction = extract_with_retry(
            backend=backend,
            prompt=prompt,
            chunk_name=chunk.name,
            output_format="json",
            json_schema=SCHEMA_JSON_SCHEMA,
            max_retries=max_retries,
        )
        total_wall += extraction.elapsed_sec

        # Save raw transcript for audit (1 file per chunk; latest
        # response, latest prompt - we keep retry trail in observations).
        _write_transcript(transcripts_dir, "schema", chunk, extraction)

        if extraction.parsed is None:
            observations.append(
                f"schema chunk {chunk.name!r}: extraction failed; modes={extraction.failure_modes}"
            )
            continue

        parsed = extraction.parsed
        chunk_fields = parsed.get("chunk_fields") if isinstance(parsed, dict) else None
        if not isinstance(chunk_fields, dict):
            observations.append(
                f"schema chunk {chunk.name!r}: parsed but chunk_fields not a dict (got "
                f"{type(chunk_fields).__name__}); skipping"
            )
            continue

        # Merge by namespace
        for field_name, field_def in chunk_fields.items():
            if not isinstance(field_def, dict):
                continue
            if is_internal_plumbing_field(field_name):
                continue
            namespace = field_def.get("namespace")
            if not isinstance(namespace, str):
                continue
            # Strip the namespace key from the field def - canonical
            # shape has it implicit by section.
            field_def_clean = {k: v for k, v in field_def.items() if k != "namespace"}
            # Defensive: drop empty values
            field_def_clean = {
                k: v for k, v in field_def_clean.items() if v is not None or k == "default"
            }
            if namespace == "engine_params":
                engine_params[field_name] = field_def_clean
            elif namespace == "sampling_params":
                sampling_params[field_name] = field_def_clean
            elif namespace.startswith("$defs."):
                class_name = namespace.removeprefix("$defs.")
                defs.setdefault(class_name, {"type": "object", "properties": {}})
                defs[class_name]["properties"][field_name] = field_def_clean
            else:
                # Unrecognised namespace - log
                observations.append(
                    f"schema chunk {chunk.name!r}: unrecognised namespace "
                    f"{namespace!r} for field {field_name!r}"
                )

    # Apply final pass of internal-plumbing filter (belt-and-braces)
    engine_params = filter_internal_plumbing(engine_params)
    sampling_params = filter_internal_plumbing(sampling_params)

    schema_envelope: dict[str, Any] = {
        "schema_version": "2.0.0",
        "engine": engine,
        "engine_version": engine_version,
        "discovered_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "discovery_method": "strategy_b_oss_llm",
        "engine_params": engine_params,
        "sampling_params": sampling_params,
    }
    if defs:
        schema_envelope["$defs"] = defs

    return schema_envelope, total_wall, observations


# ---------------------------------------------------------------------------
# Invariants extraction
# ---------------------------------------------------------------------------


def extract_invariants(
    *,
    backend: LLMBackend,
    chunks: list[ChunkSpec],
    engine: str,
    engine_version: str,
    transcripts_dir: Path,
    max_retries: int = 2,
) -> tuple[dict[str, Any], float, list[str]]:
    """Run each invariant chunk; merge + deduplicate the YAML output.

    Returns ``(invariants_envelope, total_wall_sec, observations)``.

    Merging rules:
    - Concatenate `invariants:` arrays across chunks.
    - Deduplicate by id (later wins) AND by (namespace, native_field,
      predicate_kind) identity tuple.
    - Skip entries that fail basic shape validation (missing match,
      missing severity, etc.).
    """
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str, str]] = set()
    observations: list[str] = []
    total_wall = 0.0
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        # Determine the canonical match-field namespace for this chunk.
        # Most validate() sections use `transformers.sampling`; the
        # BitsAndBytesConfig chunk uses bare `transformers` (engine_params).
        ns_list = chunk.expected_namespaces or ["transformers.sampling"]
        # Use the first expected_namespace as the canonical example
        field_namespace = ns_list[0]
        prompt = INVARIANTS_PROMPT_TEMPLATE.format(
            engine=engine,
            engine_version=engine_version,
            field_namespace=field_namespace,
            source=chunk.source,
        )

        extraction = extract_with_retry(
            backend=backend,
            prompt=prompt,
            chunk_name=chunk.name,
            output_format="yaml",
            json_schema=None,  # YAML doesn't get JSON Schema validation
            max_retries=max_retries,
        )
        total_wall += extraction.elapsed_sec

        _write_transcript(transcripts_dir, "invariants", chunk, extraction)

        if extraction.parsed is None:
            observations.append(
                f"invariants chunk {chunk.name!r}: extraction failed; "
                f"modes={extraction.failure_modes}"
            )
            continue

        parsed = extraction.parsed
        # Several legal shapes:
        # - {"invariants": [...]}
        # - [...]  (just the list at the top level - some LLMs do this)
        # - {"<arbitrary-key>": [...]}  (when LLM gets confused)
        inv_list: list[Any] | None = None
        if isinstance(parsed, dict):
            for k in ("invariants", "results", "extracted"):
                if k in parsed and isinstance(parsed[k], list):
                    inv_list = parsed[k]
                    break
            if inv_list is None:
                # Take any top-level list value if there's exactly one
                lists = [v for v in parsed.values() if isinstance(v, list)]
                if len(lists) == 1:
                    inv_list = lists[0]
        elif isinstance(parsed, list):
            inv_list = parsed

        if inv_list is None:
            observations.append(
                f"invariants chunk {chunk.name!r}: parsed but no invariants list found"
            )
            continue

        chunk_count = 0
        for inv in inv_list:
            if not isinstance(inv, dict):
                continue
            inv_id = inv.get("id")
            if not isinstance(inv_id, str) or not inv_id:
                continue
            # Identity check (sliced)
            ident = _invariant_identity(inv)
            if ident in seen_identities:
                continue
            if inv_id in seen_ids:
                continue
            seen_ids.add(inv_id)
            seen_identities.add(ident)
            # Ensure required envelope keys
            inv.setdefault("engine", engine)
            inv.setdefault("library", engine)
            inv.setdefault("added_by", "llm_miner")
            inv.setdefault("added_at", _dt.date.today().isoformat())
            merged.append(inv)
            chunk_count += 1

        if chunk_count == 0:
            observations.append(
                f"invariants chunk {chunk.name!r}: parsed but yielded 0 unique invariants"
            )

    envelope: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": "strategy_b_oss_llm",
        "invariants": merged,
    }
    return envelope, total_wall, observations


# ---------------------------------------------------------------------------
# Multi-pass invariants refinement (Phase 2.5)
# ---------------------------------------------------------------------------


def _serialise_inv_list_for_prompt(inv_list: list[dict[str, Any]], max_chars: int = 8000) -> str:
    """Serialise an invariants list as YAML for embedding in a verify/extend
    prompt. Capped at ``max_chars`` so the prompt fits the 32k context
    window even with the source chunk inlined.

    Strips kwargs_positive/kwargs_negative and message_template (which
    can be large) to focus the LLM on identity + severity + predicate.
    """
    minimal: list[dict[str, Any]] = []
    for inv in inv_list:
        if not isinstance(inv, dict):
            continue
        minimal.append(
            {
                "id": inv.get("id"),
                "severity": inv.get("severity"),
                "match": inv.get("match"),
                "invariant_under_test": inv.get("invariant_under_test"),
            }
        )
    text = yaml.safe_dump({"invariants": minimal}, sort_keys=False, default_flow_style=False)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n# ...<truncated>...\n"
    return text


def _parse_verify_output(parsed: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Parse a verify-pass YAML output into (confirmed_ids, flagged_entries).

    Tolerant of three shapes:
    - {"confirmed": [...], "flagged": [...]}
    - {"confirmed": [...]}    (no flags emitted)
    - {"flagged": [...]}      (everything implicitly confirmed)
    """
    confirmed: list[str] = []
    flagged: list[dict[str, Any]] = []
    if not isinstance(parsed, dict):
        return confirmed, flagged
    c = parsed.get("confirmed")
    if isinstance(c, list):
        for item in c:
            if isinstance(item, str):
                confirmed.append(item)
            elif isinstance(item, dict) and isinstance(item.get("id"), str):
                confirmed.append(item["id"])
    f = parsed.get("flagged")
    if isinstance(f, list):
        for item in f:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                flagged.append(item)
    return confirmed, flagged


def extract_invariants_multipass(
    *,
    backend: LLMBackend,
    chunks: list[ChunkSpec],
    engine: str,
    engine_version: str,
    transcripts_dir: Path,
    max_retries: int = 2,
    enable_verify: bool = True,
    enable_extend: bool = True,
) -> tuple[dict[str, Any], float, list[str]]:
    """Multi-pass refinement of invariants extraction (Phase 2.5).

    Pipeline (per chunk):
    1. **Extract pass** - existing single-shot extraction (round-3
       locked prompts). Identical to ``extract_invariants``.
    2. **Verify pass** (optional; default on): re-prompt the LLM with
       its own pass-1 output + the source, asking it to flag any
       invariants that look wrong. Flagged entries with ``fix: drop``
       are removed from the merged result.
    3. **Extend pass** (optional; default on): re-prompt with the
       confirmed pass-1 output + the source + the pass-2 flags, asking
       what was missed. New invariants get ``added_by: llm_miner_pass3``.

    Reconciliation:
    - Confirmed pass-1 entries (not flagged with ``fix: drop``) keep
      their original entry. ``fix: correct_*`` flags are recorded as
      adjacent observations but do not auto-modify (conservative;
      pass-1 prompt is locked).
    - Pass-3 entries that don't collide with pass-1 (by identity OR ID)
      are appended.

    Returns ``(envelope, total_wall_sec, observations)`` matching the
    shape of ``extract_invariants``.

    NOTE: The pass-1 chunk is unmodified from the locked Phase 2 prompt.
    The verify+extend passes are NEW prompts (Phase 2.5).
    """
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_identities: set[tuple[str, str, str, str]] = set()
    observations: list[str] = []
    total_wall = 0.0
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    pass2_skipped_invariants: list[str] = []
    pass3_added_count = 0

    for chunk in chunks:
        ns_list = chunk.expected_namespaces or ["transformers.sampling"]
        field_namespace = ns_list[0]

        # -----------------------------------------------------------------
        # PASS 1: extract (locked Phase 2 prompt)
        # -----------------------------------------------------------------
        pass1_prompt = INVARIANTS_PROMPT_TEMPLATE.format(
            engine=engine,
            engine_version=engine_version,
            field_namespace=field_namespace,
            source=chunk.source,
        )
        pass1_extraction = extract_with_retry(
            backend=backend,
            prompt=pass1_prompt,
            chunk_name=chunk.name + "__pass1",
            output_format="yaml",
            json_schema=None,
            max_retries=max_retries,
        )
        total_wall += pass1_extraction.elapsed_sec
        _write_transcript(transcripts_dir, "invariants_pass1", chunk, pass1_extraction)

        if pass1_extraction.parsed is None:
            observations.append(
                f"invariants chunk {chunk.name!r} pass1: extraction failed; "
                f"modes={pass1_extraction.failure_modes}"
            )
            continue

        pass1_inv_list = _pull_invariants_list(pass1_extraction.parsed)
        if pass1_inv_list is None or len(pass1_inv_list) == 0:
            observations.append(
                f"invariants chunk {chunk.name!r} pass1: yielded 0 invariants; "
                f"skipping verify/extend passes for this chunk"
            )
            continue

        # -----------------------------------------------------------------
        # PASS 2: verify (optional)
        # -----------------------------------------------------------------
        confirmed_ids: set[str] = {
            str(i.get("id")) for i in pass1_inv_list if isinstance(i.get("id"), str)
        }
        flagged_entries: list[dict[str, Any]] = []
        if enable_verify:
            pass2_prompt = INVARIANTS_VERIFY_PROMPT_TEMPLATE.format(
                engine=engine,
                engine_version=engine_version,
                pass1_invariants=_serialise_inv_list_for_prompt(pass1_inv_list, max_chars=6000),
                source=chunk.source,
            )
            pass2_extraction = extract_with_retry(
                backend=backend,
                prompt=pass2_prompt,
                chunk_name=chunk.name + "__pass2_verify",
                output_format="yaml",
                json_schema=None,
                max_retries=max_retries,
            )
            total_wall += pass2_extraction.elapsed_sec
            _write_transcript(transcripts_dir, "invariants_pass2_verify", chunk, pass2_extraction)
            if pass2_extraction.parsed is not None:
                c_ids, flags = _parse_verify_output(pass2_extraction.parsed)
                if c_ids:
                    confirmed_ids = set(c_ids)
                flagged_entries = flags
                # Apply drop fix
                drop_ids: set[str] = set()
                for f in flagged_entries:
                    fix = (f.get("fix") or "").strip().lower()
                    if fix == "drop":
                        drop_ids.add(str(f.get("id", "")))
                        pass2_skipped_invariants.append(str(f.get("id", "")))
                if drop_ids:
                    pass1_inv_list = [
                        i for i in pass1_inv_list if str(i.get("id", "")) not in drop_ids
                    ]
                # Record other (non-drop) corrections as observations
                for f in flagged_entries:
                    fix = (f.get("fix") or "").strip().lower()
                    if fix and fix != "drop":
                        observations.append(
                            f"chunk {chunk.name!r} pass2 flag (non-applied): id={f.get('id')!r} "
                            f"reason={f.get('reason')!r} fix={f.get('fix')!r}"
                        )
            else:
                observations.append(
                    f"invariants chunk {chunk.name!r} pass2_verify: extraction failed; "
                    f"modes={pass2_extraction.failure_modes}; pass1 unchanged"
                )

        # -----------------------------------------------------------------
        # PASS 3: extend (optional)
        # -----------------------------------------------------------------
        pass3_inv_list: list[Any] = []
        if enable_extend:
            pass3_prompt = INVARIANTS_EXTEND_PROMPT_TEMPLATE.format(
                engine=engine,
                engine_version=engine_version,
                field_namespace=field_namespace,
                pass1_invariants=_serialise_inv_list_for_prompt(pass1_inv_list, max_chars=5000),
                pass2_flags=yaml.safe_dump(
                    {"flagged": flagged_entries}, sort_keys=False, default_flow_style=False
                )[:2000]
                if flagged_entries
                else "(no flags)",
                source=chunk.source,
            )
            pass3_extraction = extract_with_retry(
                backend=backend,
                prompt=pass3_prompt,
                chunk_name=chunk.name + "__pass3_extend",
                output_format="yaml",
                json_schema=None,
                max_retries=max_retries,
            )
            total_wall += pass3_extraction.elapsed_sec
            _write_transcript(transcripts_dir, "invariants_pass3_extend", chunk, pass3_extraction)
            if pass3_extraction.parsed is not None:
                p3 = _pull_invariants_list(pass3_extraction.parsed)
                if isinstance(p3, list):
                    pass3_inv_list = p3
            else:
                observations.append(
                    f"invariants chunk {chunk.name!r} pass3_extend: extraction failed; "
                    f"modes={pass3_extraction.failure_modes}"
                )

        # -----------------------------------------------------------------
        # RECONCILE: merge pass1 (confirmed only) + pass3 into the global set
        # -----------------------------------------------------------------
        for inv in pass1_inv_list:
            if not _add_invariant_if_unique(
                inv,
                merged=merged,
                seen_ids=seen_ids,
                seen_identities=seen_identities,
                engine=engine,
                added_by_default="llm_miner",
            ):
                continue
        for inv in pass3_inv_list:
            if not isinstance(inv, dict):
                continue
            # Ensure pass-3 tagging is preserved (or set if LLM forgot).
            inv.setdefault("added_by", "llm_miner_pass3")
            if _add_invariant_if_unique(
                inv,
                merged=merged,
                seen_ids=seen_ids,
                seen_identities=seen_identities,
                engine=engine,
                added_by_default="llm_miner_pass3",
            ):
                pass3_added_count += 1

    observations.append(
        f"multipass summary: pass2_dropped={len(pass2_skipped_invariants)}, "
        f"pass3_added={pass3_added_count}, total_invariants={len(merged)}"
    )

    envelope: dict[str, Any] = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": engine_version,
        "mined_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "miner": "strategy_b_oss_llm_multipass",
        "invariants": merged,
    }
    return envelope, total_wall, observations


def _pull_invariants_list(parsed: Any) -> list[Any] | None:
    """Pull the canonical invariants list out of a parsed YAML/JSON payload.

    Tolerates the same shape variants as ``extract_invariants``.
    """
    if isinstance(parsed, dict):
        for k in ("invariants", "results", "extracted"):
            if k in parsed and isinstance(parsed[k], list):
                return parsed[k]
        lists = [v for v in parsed.values() if isinstance(v, list)]
        if len(lists) == 1:
            return lists[0]
    elif isinstance(parsed, list):
        return parsed
    return None


def _add_invariant_if_unique(
    inv: dict[str, Any],
    *,
    merged: list[dict[str, Any]],
    seen_ids: set[str],
    seen_identities: set[tuple[str, str, str, str]],
    engine: str,
    added_by_default: str,
) -> bool:
    """Append ``inv`` to ``merged`` iff its id+identity are unseen.

    Sets default envelope keys (engine, library, added_by, added_at)
    if missing. Returns True if appended.
    """
    if not isinstance(inv, dict):
        return False
    inv_id = inv.get("id")
    if not isinstance(inv_id, str) or not inv_id:
        return False
    ident = _invariant_identity(inv)
    if ident in seen_identities or inv_id in seen_ids:
        return False
    seen_ids.add(inv_id)
    seen_identities.add(ident)
    inv.setdefault("engine", engine)
    inv.setdefault("library", engine)
    inv.setdefault("added_by", added_by_default)
    inv.setdefault("added_at", _dt.date.today().isoformat())
    merged.append(inv)
    return True


def _invariant_identity(inv: dict[str, Any]) -> tuple[str, str, str, str]:
    """Inline copy of trial_scoring's invariant_identity logic - avoids
    a heavy circular import (strategies/ shouldn't depend on the scoring
    harness internals). Matches the canonical 4-tuple form
    ``(namespace, native_field, predicate_kind, secondary_field)`` as
    defined by Phase 2.5 rubric fix.

    For single-field invariants the secondary_field slot is the empty
    string. For multi-field invariants it is the leaf name of the second
    match-field key.
    """
    match_fields = ((inv.get("match") or {}).get("fields")) or {}
    if not isinstance(match_fields, dict) or not match_fields:
        return ("", str(inv.get("id", "")) or "<unknown>", "unknown", "")
    keys = list(match_fields.keys())
    primary_key = keys[0]
    primary_value = match_fields[primary_key]
    namespace, _, native_field = primary_key.rpartition(".")
    if not isinstance(primary_value, dict):
        kind = "exact"
    else:
        value_keys = set(primary_value.keys())
        if "not_in" in value_keys:
            kind = "not_in"
        elif "not_equal" in value_keys:
            kind = "not_equal"
        elif ">" in value_keys:
            kind = "gt"
        elif "<" in value_keys:
            kind = "lt"
        elif ">=" in value_keys:
            kind = "ge"
        elif "<=" in value_keys:
            kind = "le"
        elif "type_is_not" in value_keys:
            kind = "type_is_not"
        elif "present" in value_keys:
            kind = "present"
        else:
            kind = "unknown"
    if len(keys) >= 2:
        secondary_key = keys[1]
        _, _, secondary_field = secondary_key.rpartition(".")
    else:
        secondary_field = ""
    return (namespace, native_field, kind, secondary_field)


# ---------------------------------------------------------------------------
# Transcript writing
# ---------------------------------------------------------------------------


def _write_transcript(
    transcripts_dir: Path, task: str, chunk: ChunkSpec, extraction: ChunkExtraction
) -> None:
    """Save the raw prompts + responses for a chunk's extraction.

    One file per (task, chunk) pair. Format: a single markdown file
    with sections for each retry attempt.
    """
    safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in chunk.name)
    out_path = transcripts_dir / f"{task}__{safe_name}.md"
    lines: list[str] = []
    lines.append(f"# {task} extraction transcript: {chunk.name}")
    lines.append("")
    lines.append(f"- chunk_description: {chunk.description}")
    lines.append(f"- expected_namespaces: {chunk.expected_namespaces}")
    lines.append(f"- attempts: {len(extraction.prompts)}")
    lines.append(f"- elapsed_sec: {extraction.elapsed_sec:.2f}")
    lines.append(f"- failure_modes: {extraction.failure_modes}")
    lines.append(f"- schema_errors: {extraction.schema_errors}")
    lines.append(f"- parsed: {'yes' if extraction.parsed is not None else 'no'}")
    lines.append("")
    for i, (p, r) in enumerate(zip(extraction.prompts, extraction.raw_responses)):
        lines.append(f"## Attempt {i + 1}")
        lines.append("")
        lines.append("### Prompt")
        lines.append("")
        lines.append("```")
        lines.append(p[:8000] + ("\n...<truncated>..." if len(p) > 8000 else ""))
        lines.append("```")
        lines.append("")
        lines.append("### Response")
        lines.append("")
        lines.append("```")
        lines.append(r[:8000] + ("\n...<truncated>..." if len(r) > 8000 else ""))
        lines.append("```")
        lines.append("")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Top-level: run (b) for one cell
# ---------------------------------------------------------------------------


def run_b_on_transformers_active(
    *,
    out_dir: Path,
    engine_version: str = "4.57.3",
    backend: LLMBackend | None = None,
    max_retries: int = 2,
    multipass: bool = False,
    source_root: Path | None = None,
) -> StrategyBOutputs:
    """Run strategy (b) for transformers (active OR bumped version).

    Writes:
    - ``out_dir/schema.json``
    - ``out_dir/invariants.proposed.yaml``
    - ``out_dir/raw_llm_transcripts/{schema,invariants}__<chunk>.md``

    Parameters
    ----------
    multipass : bool, default False
        If True, runs invariants extraction through the Phase 2.5 three-
        pass pipeline (extract -> verify -> extend). Schema extraction is
        unchanged (single-shot per chunk). The legacy single-pass remains
        the default for backward-compat with existing trial runs.
    source_root : Path | None
        If None, sources via inspect on the project venv (active cell).
        If provided, AST-parse from the path (bumped-version Phase 3a.2).

    Returns ``StrategyBOutputs`` with the paths + observations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if backend is None:
        backend = OllamaBackend()  # default: container Ollama 11435 + llama3.1:70b

    s_chunks = schema_chunks(source_root=source_root)
    i_chunks = invariants_chunks(source_root=source_root)

    schema_envelope, schema_wall, schema_obs = extract_schema(
        backend=backend,
        chunks=s_chunks,
        engine="transformers",
        engine_version=engine_version,
        transcripts_dir=transcripts_dir,
        max_retries=max_retries,
    )

    if multipass:
        invariants_envelope, inv_wall, inv_obs = extract_invariants_multipass(
            backend=backend,
            chunks=i_chunks,
            engine="transformers",
            engine_version=engine_version,
            transcripts_dir=transcripts_dir,
            max_retries=max_retries,
        )
    else:
        invariants_envelope, inv_wall, inv_obs = extract_invariants(
            backend=backend,
            chunks=i_chunks,
            engine="transformers",
            engine_version=engine_version,
            transcripts_dir=transcripts_dir,
            max_retries=max_retries,
        )

    schema_path = out_dir / "schema.json"
    inv_path = out_dir / "invariants.proposed.yaml"
    schema_path.write_text(json.dumps(schema_envelope, indent=2, sort_keys=True))
    inv_path.write_text(
        yaml.safe_dump(invariants_envelope, sort_keys=False, default_flow_style=False)
    )

    observations = schema_obs + inv_obs
    observations.append(
        f"strategy_b: schema_chunks={len(s_chunks)}, invariants_chunks={len(i_chunks)}, "
        f"schema_wall={schema_wall:.1f}s, invariants_wall={inv_wall:.1f}s"
    )

    return StrategyBOutputs(
        schema_path=schema_path,
        invariants_path=inv_path,
        observations=observations,
        schema_wall_sec=schema_wall,
        invariants_wall_sec=inv_wall,
    )


# ---------------------------------------------------------------------------
# Phase 3 Prep-2: vllm + tensorrt dispatchers
# ---------------------------------------------------------------------------
#
# Mirror the transformers shape: schema_chunks() + invariants_chunks() pulled
# from the engine-specific chunker, fed through the same backend-agnostic
# extract_schema / extract_invariants / extract_invariants_multipass machinery.
#
# Active-version cells use the pre-staged source trees at
# /tmp/{vllm-unpacked,trt-llm-0.21.0}/. The chunkers hard-wire those paths
# for the active cell; Phase 3a.2 (bumped versions) wires the chunker source
# root via the lazy-build venv_setup output.


def _run_b_generic(
    *,
    engine: str,
    engine_version: str,
    schema_chunks_fn: Any,
    invariants_chunks_fn: Any,
    out_dir: Path,
    backend: LLMBackend | None = None,
    max_retries: int = 2,
    multipass: bool = False,
) -> StrategyBOutputs:
    """Engine-agnostic strategy (b) driver.

    Used by ``run_b_on_{vllm,tensorrt}_active``. The two callables
    (``schema_chunks_fn``, ``invariants_chunks_fn``) return the engine's
    chunked source; everything downstream (prompts, backend, parsing,
    merging) is shared with the transformers path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    if backend is None:
        backend = OllamaBackend()  # default: container Ollama 11435 + llama3.1:70b

    s_chunks = schema_chunks_fn()
    i_chunks = invariants_chunks_fn()

    schema_envelope, schema_wall, schema_obs = extract_schema(
        backend=backend,
        chunks=s_chunks,
        engine=engine,
        engine_version=engine_version,
        transcripts_dir=transcripts_dir,
        max_retries=max_retries,
    )

    if multipass:
        invariants_envelope, inv_wall, inv_obs = extract_invariants_multipass(
            backend=backend,
            chunks=i_chunks,
            engine=engine,
            engine_version=engine_version,
            transcripts_dir=transcripts_dir,
            max_retries=max_retries,
        )
    else:
        invariants_envelope, inv_wall, inv_obs = extract_invariants(
            backend=backend,
            chunks=i_chunks,
            engine=engine,
            engine_version=engine_version,
            transcripts_dir=transcripts_dir,
            max_retries=max_retries,
        )

    schema_path = out_dir / "schema.json"
    inv_path = out_dir / "invariants.proposed.yaml"
    schema_path.write_text(json.dumps(schema_envelope, indent=2, sort_keys=True))
    inv_path.write_text(
        yaml.safe_dump(invariants_envelope, sort_keys=False, default_flow_style=False)
    )

    observations = schema_obs + inv_obs
    observations.append(
        f"strategy_b: engine={engine!r}, schema_chunks={len(s_chunks)}, "
        f"invariants_chunks={len(i_chunks)}, schema_wall={schema_wall:.1f}s, "
        f"invariants_wall={inv_wall:.1f}s, multipass={multipass}"
    )

    return StrategyBOutputs(
        schema_path=schema_path,
        invariants_path=inv_path,
        observations=observations,
        schema_wall_sec=schema_wall,
        invariants_wall_sec=inv_wall,
    )


def run_b_on_vllm_active(
    *,
    out_dir: Path,
    engine_version: str = "0.7.3",
    backend: LLMBackend | None = None,
    max_retries: int = 2,
    multipass: bool = False,
) -> StrategyBOutputs:
    """Run strategy (b) for vllm active version (0.7.3).

    Source is read from the pre-staged tree at /tmp/vllm-unpacked/vllm/
    via ``vllm_chunker.{schema_chunks, invariants_chunks}``.

    Output layout matches ``run_b_on_transformers_active``:
    - ``out_dir/schema.json``
    - ``out_dir/invariants.proposed.yaml``
    - ``out_dir/raw_llm_transcripts/{schema,invariants}__<chunk>.md``
    """
    from _spike.scripts.strategies.vllm_chunker import (
        invariants_chunks as vllm_inv_chunks,
    )
    from _spike.scripts.strategies.vllm_chunker import (
        schema_chunks as vllm_schema_chunks,
    )

    return _run_b_generic(
        engine="vllm",
        engine_version=engine_version,
        schema_chunks_fn=vllm_schema_chunks,
        invariants_chunks_fn=vllm_inv_chunks,
        out_dir=out_dir,
        backend=backend,
        max_retries=max_retries,
        multipass=multipass,
    )


def run_b_on_tensorrt_active(
    *,
    out_dir: Path,
    engine_version: str = "0.21.0",
    backend: LLMBackend | None = None,
    max_retries: int = 2,
    multipass: bool = False,
) -> StrategyBOutputs:
    """Run strategy (b) for tensorrt_llm active version (0.21.0).

    Source is read from the pre-staged tree at
    /tmp/trt-llm-0.21.0/tensorrt_llm/ via
    ``tensorrt_chunker.{schema_chunks, invariants_chunks}``.

    Output layout matches ``run_b_on_transformers_active``.
    """
    from _spike.scripts.strategies.tensorrt_chunker import (
        invariants_chunks as trt_inv_chunks,
    )
    from _spike.scripts.strategies.tensorrt_chunker import (
        schema_chunks as trt_schema_chunks,
    )

    return _run_b_generic(
        engine="tensorrt",
        engine_version=engine_version,
        schema_chunks_fn=trt_schema_chunks,
        invariants_chunks_fn=trt_inv_chunks,
        out_dir=out_dir,
        backend=backend,
        max_retries=max_retries,
        multipass=multipass,
    )
