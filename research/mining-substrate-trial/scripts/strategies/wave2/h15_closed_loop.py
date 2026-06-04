"""Wave 2 hybrid H15: closed-loop det-extract + runtime-gate + LLM re-emit.

Pattern (Tier B+ cheap-hybrid, WAVE2_PROTOCOL section 3):

    Round 0: load the deterministic (a) baseline from
        engine_versions/<engine>/<version>/outputs/invariants.proposed.yaml.
        h15 does NOT re-run the miner; the canonical artefact IS the input.
    Rounds 1..max_rounds:
        a. Runtime-validate via trial_scoring.runtime_validate_invariants_dispatch.
        b. Compute failure list (entries that did not validate).
        c. If empty OR identical to previous round: break.
        d. Build failure-focused prompt (locked template).
        e. Dispatch LLM (deferred boundary; raises until Ollama wired).
        f. Merge re-emitted entries by id-match, replacing failures.

Hypothesis: det-feedback unlocks LLM correction in a way single-shot (b)
cannot, while keeping LLM cost sub-(b) (small failure list, not the
whole surface).

Scope: transformers v4.57.3 only. vllm/tensorrt raise NotImplementedError
(protocol step 11 wires h15 for transformers only in Wave 2.1; vllm is
gated by Tier-A evidence per rule H).
"""

from __future__ import annotations

import copy
import datetime as _dt
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Project root sits six parents above this module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_TRIAL_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
if str(_TRIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(_TRIAL_SCRIPTS_DIR))

from strategies.wave2._base import CellOutput, standard_out_dir  # noqa: E402

SUPPORTED_ENGINE = "transformers"
SUPPORTED_VERSION = "4.57.3"
SUPPORTED_VERSION_SLUG = "v4_57_3"

# Caps to keep the prompt inside llama3.1:70b's 32k context.
MAX_SNIPPETS_PER_PROMPT = 12
MAX_SNIPPET_LINES = 40


# Locked H15 prompt (Wave 2; do not iterate inside the loop). Modelled on
# Wave 1's INVARIANTS_VERIFY_PROMPT_TEMPLATE in shape, but the task is
# different: the LLM receives a runtime-gate-rejected subset and must
# either correct each entry or mark it truly spurious.
FAILURE_REEMIT_PROMPT_TEMPLATE = """You are a code analyser doing CLOSED-LOOP CORRECTION on a
deterministic invariant miner's output for {engine} v{engine_version}.

A runtime validation gate has just replayed each invariant's
``kwargs_positive`` / ``kwargs_negative`` cases against the live library.
The entries below FAILED the gate. For each failure, you must either:

  (A) emit a CORRECTED version of the invariant that would pass the gate
      (typically: fix ``kwargs_positive`` so it triggers the predicate,
      fix ``kwargs_negative`` so it does not trigger, or correct
      ``native_type`` / ``severity`` / ``match.fields``), OR

  (B) mark the entry as ``truly_spurious: true`` if the source code shows
      no real invariant at the cited location (deterministic miner false
      positive; nothing for the LLM to salvage).

INPUT 1 - failed invariants and their gate-failure reasons (round
{round_idx} of {max_rounds}; {n_failures} entries):

{failure_block}

INPUT 2 - relevant validator-method source snippets (referenced from
each entry's ``miner_source``; truncated to {max_snippet_lines} lines
per snippet, {n_snippets} of {n_failures} entries shown):

{source_snippets}

OUTPUT FORMAT: a YAML document with ONE top-level ``invariants`` list.
Emit ONE entry per failed input id. Use this shape exactly:

invariants:
- id: <same id as the failing input>
  truly_spurious: false
  engine: {engine}
  library: {engine}
  invariant_under_test: <one-line: what the library checks here>
  severity: <error|dormant|warning>
  native_type: <transformers.GenerationConfig|transformers.BitsAndBytesConfig>
  miner_source:
    path: <same path as the failing input>
    method: <validate|__init__>
    line_at_scan: <approximate line number if visible>
  match:
    engine: {engine}
    fields:
      <namespace>.<field>: <value or predicate>
  kwargs_positive:
    <field>: <value that TRIGGERS the invariant under the source>
  kwargs_negative:
    <field>: <value that does NOT trigger>
  expected_outcome:
    outcome: <error|dormant_announced|warning>
    emission_channel: <none|logger_warning_once|logger_warning>
    normalised_fields: []
  message_template: '<exact f-string literal from raise/minor_issues, preserving {{}}>'
  added_by: llm_miner_h15_reemit
  added_at: '{today}'

For entries you mark spurious, emit only:

- id: <same id as the failing input>
  truly_spurious: true
  reason: <one-line: why the source does not support this invariant>

CRITICAL RULES:

1. Return ONLY the YAML document. NO markdown code fences. NO commentary.
   First character must be ``i`` (from ``invariants:``).
2. The ``id`` field MUST match the failing input id exactly so the
   closed-loop runner can merge by identity. Do NOT invent new ids.
3. ``truly_spurious: true`` is the ONLY way to remove an entry. Do not
   silently drop ids.
4. For corrections: re-derive ``kwargs_positive`` and ``kwargs_negative``
   from the source snippet so the runtime gate would now accept them.
   The gate criterion is: positive case triggers ANY of (exception,
   warning, logger message); negative case triggers NONE of those.
5. Do not invent invariants that are not present in the source snippet.
   If the snippet is truncated and you cannot see the predicate, prefer
   ``truly_spurious: true`` with reason "source-snippet insufficient" over
   guessing.

Emit the YAML now:
"""


@dataclass
class _FailureRecord:
    """One entry that failed the runtime gate this round."""

    invariant_id: str
    failure_reason: str
    invariant_body: dict[str, Any]


def _load_invariants_envelope(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"invariants": []}
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        return {"invariants": []}
    if not isinstance(data.get("invariants"), list):
        data["invariants"] = []
    return data


def _invariant_identity_key(inv: dict[str, Any]) -> str:
    """Merge-key for h15: the id field directly (the LLM is instructed to echo it)."""
    return str(inv.get("id", ""))


def _classify_runtime_result(rv: Any) -> tuple[bool, str]:
    """Return (validated_ok, failure_reason).

    validated_ok iff the positive case triggered and there's no infra
    error (the H3 gate criterion). Everything else is a failure with a
    textual reason for prompt feedback.
    """
    err = getattr(rv, "error", None)
    pos_confirmed = bool(getattr(rv, "positive_confirmed", False))
    if pos_confirmed and not err:
        return True, ""
    if err:
        return False, f"runtime gate error: {str(err)[:200]}"
    outcome = getattr(rv, "observed_outcome", "") or "no_op"
    return False, f"positive case did not trigger (observed {outcome!r})"


def _extract_failures(
    *,
    invariants: list[dict[str, Any]],
    runtime_results: list[Any],
) -> list[_FailureRecord]:
    by_id: dict[str, dict[str, Any]] = {}
    for inv in invariants:
        if isinstance(inv, dict):
            iid = _invariant_identity_key(inv)
            if iid:
                by_id[iid] = inv
    failures: list[_FailureRecord] = []
    for rv in runtime_results:
        iid = str(getattr(rv, "invariant_id", "") or "")
        if not iid or iid not in by_id:
            continue
        ok, reason = _classify_runtime_result(rv)
        if ok:
            continue
        failures.append(
            _FailureRecord(invariant_id=iid, failure_reason=reason, invariant_body=by_id[iid])
        )
    return failures


def _resolve_source_snippet(inv: dict[str, Any], *, engine_source_root: Path | None) -> str | None:
    """Slice the validator method around miner_source.line_at_scan."""
    if engine_source_root is None:
        return None
    miner_source = inv.get("miner_source")
    if not isinstance(miner_source, dict):
        return None
    rel_path = miner_source.get("path")
    if not isinstance(rel_path, str) or not rel_path:
        return None
    source_path = engine_source_root / rel_path
    if not source_path.exists() or not source_path.is_file():
        return None
    try:
        all_lines = source_path.read_text().splitlines()
    except OSError:
        return None
    line_at_scan = miner_source.get("line_at_scan")
    if isinstance(line_at_scan, int) and line_at_scan > 0:
        half = MAX_SNIPPET_LINES // 2
        start = max(0, line_at_scan - half - 1)
        end = min(len(all_lines), start + MAX_SNIPPET_LINES)
        snippet_lines = all_lines[start:end]
        prefix = f"# {rel_path}:{start + 1}-{end}\n"
    else:
        snippet_lines = all_lines[:MAX_SNIPPET_LINES]
        prefix = f"# {rel_path}:1-{len(snippet_lines)}\n"
    return prefix + "\n".join(snippet_lines)


def _detect_engine_source_root() -> Path | None:
    """Locate canonical transformers source.

    Preferred: the pre-staged trial venv used by Wave 1 H3. Falls back
    to scanning local venvs. None when nothing resolves (prompt then
    omits source snippets and the LLM is asked to defer to spurious).
    """
    for cand in (
        Path("/tmp/trial_transformers_v4_57_6_venv/src/transformers"),
        Path("/tmp/trial_transformers_v4_57_3_venv/src/transformers"),
    ):
        if cand.exists() and cand.is_dir():
            return cand
    for parent in (_PROJECT_ROOT / ".venv", Path.home() / ".venv"):
        if not parent.exists():
            continue
        for lib_dir in parent.glob("lib/python*/site-packages/transformers"):
            if lib_dir.is_dir():
                return lib_dir
    return None


def _build_failure_block(failures: list[_FailureRecord]) -> str:
    minimal = [
        {
            "id": f.invariant_id,
            "failure_reason": f.failure_reason,
            "current_severity": f.invariant_body.get("severity"),
            "current_native_type": f.invariant_body.get("native_type"),
            "current_match": f.invariant_body.get("match"),
            "current_kwargs_positive": f.invariant_body.get("kwargs_positive"),
            "current_kwargs_negative": f.invariant_body.get("kwargs_negative"),
            "miner_source": f.invariant_body.get("miner_source"),
        }
        for f in failures
    ]
    return yaml.safe_dump({"failed_invariants": minimal}, sort_keys=False, default_flow_style=False)


def _build_source_snippets(
    failures: list[_FailureRecord], *, engine_source_root: Path | None
) -> tuple[str, int]:
    pieces: list[str] = []
    for f in failures[:MAX_SNIPPETS_PER_PROMPT]:
        snippet = _resolve_source_snippet(f.invariant_body, engine_source_root=engine_source_root)
        if snippet is None:
            pieces.append(f"# id={f.invariant_id} -- source snippet unavailable")
        else:
            pieces.append(f"# id={f.invariant_id}\n{snippet}")
    if not pieces:
        return ("(no source snippets resolvable)", 0)
    return ("\n\n".join(pieces), min(len(failures), MAX_SNIPPETS_PER_PROMPT))


def _build_failure_prompt(
    failures: list[_FailureRecord],
    *,
    round_idx: int,
    max_rounds: int,
    engine_source_root: Path | None,
) -> str:
    failure_block = _build_failure_block(failures)
    source_snippets, n_snippets = _build_source_snippets(
        failures, engine_source_root=engine_source_root
    )
    return FAILURE_REEMIT_PROMPT_TEMPLATE.format(
        engine=SUPPORTED_ENGINE,
        engine_version=SUPPORTED_VERSION,
        round_idx=round_idx,
        max_rounds=max_rounds,
        n_failures=len(failures),
        n_snippets=n_snippets,
        max_snippet_lines=MAX_SNIPPET_LINES,
        failure_block=failure_block,
        source_snippets=source_snippets,
        today=_dt.date.today().isoformat(),
    )


def _invoke_llm_on_failure_list(
    prompt: str,
    *,
    ollama_host: str,
    model: str,
    round_idx: int,
) -> list[dict[str, Any]]:
    """Single exit point to Ollama. Deferred until Wave 2.1 step 11 wires
    a live container Ollama; the failure-detection + prompt-construction
    path upstream stays testable by stubbing this function.
    """
    raise NotImplementedError(
        "wave2 h15: Ollama dispatch wiring deferred; smoke-test requires "
        "running Ollama at port 11435"
    )


def _merge_reemitted_into_envelope(
    *,
    envelope: dict[str, Any],
    reemitted: list[dict[str, Any]],
    failures: list[_FailureRecord],
) -> tuple[dict[str, Any], int, int]:
    """Replace failing entries by id-match.

    truly_spurious: true -> drop. Entries not echoed back stay in place
    (they'll resurface on the next round's failure list).
    Returns (updated_envelope, n_corrected, n_dropped_spurious).
    """
    failure_ids = {f.invariant_id for f in failures}
    reemitted_by_id: dict[str, dict[str, Any]] = {}
    for entry in reemitted:
        if not isinstance(entry, dict):
            continue
        eid = str(entry.get("id", ""))
        if eid and eid in failure_ids:
            reemitted_by_id[eid] = entry

    n_corrected = 0
    n_dropped_spurious = 0
    new_invariants: list[dict[str, Any]] = []
    for inv in envelope.get("invariants", []):
        if not isinstance(inv, dict):
            continue
        iid = _invariant_identity_key(inv)
        replacement = reemitted_by_id.get(iid)
        if replacement is None:
            new_invariants.append(inv)
            continue
        if bool(replacement.get("truly_spurious", False)):
            n_dropped_spurious += 1
            continue
        cleaned = {k: v for k, v in replacement.items() if k != "truly_spurious"}
        cleaned.setdefault("engine", SUPPORTED_ENGINE)
        cleaned.setdefault("library", SUPPORTED_ENGINE)
        cleaned.setdefault("added_by", "llm_miner_h15_reemit")
        cleaned.setdefault("added_at", _dt.date.today().isoformat())
        new_invariants.append(cleaned)
        n_corrected += 1

    updated = dict(envelope)
    updated["invariants"] = new_invariants
    return updated, n_corrected, n_dropped_spurious


def _runtime_validate(invariants_yaml: Path, *, engine: str, version_slug: str) -> list[Any]:
    """Wrap trial_scoring's dispatcher; matches the H3 call shape."""
    from trial_scoring import runtime_validate_invariants_dispatch  # type: ignore[import]

    return runtime_validate_invariants_dispatch(
        invariants_yaml, engine=engine, version_slug=version_slug
    )


def run_cell(
    *,
    engine: str,
    version: str,
    ollama_host: str = "http://localhost:11435",
    model: str = "llama3.1:70b",
    max_rounds: int = 2,
) -> CellOutput:
    """Execute one Wave 2 H15 closed-loop cell.

    Transformers v4.57.3 only; other (engine, version) pairs raise
    NotImplementedError.
    """
    strategy_id = "w2-h15-closed-loop"

    if engine != SUPPORTED_ENGINE or version != SUPPORTED_VERSION:
        raise NotImplementedError(
            f"wave2 h15: only ({SUPPORTED_ENGINE!r}, {SUPPORTED_VERSION!r}) supported; "
            f"got ({engine!r}, {version!r})"
        )

    version_slug = SUPPORTED_VERSION_SLUG
    out_dir = standard_out_dir(strategy_id, engine, version_slug)

    # ---- Round 0: load deterministic baseline ---------------------------
    t_round0 = time.perf_counter()
    baseline_inv_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / engine
        / version_slug
        / "outputs"
        / "invariants.proposed.yaml"
    )
    baseline_schema_path = (
        _PROJECT_ROOT
        / "engine_versions"
        / engine
        / version_slug
        / "outputs"
        / "schema.discovered.json"
    )
    if not baseline_inv_path.exists():
        raise FileNotFoundError(
            f"wave2 h15: canonical (a) invariants envelope missing at {baseline_inv_path}"
        )

    envelope = copy.deepcopy(_load_invariants_envelope(baseline_inv_path))
    envelope.setdefault("schema_version", "1.0.0")
    envelope.setdefault("engine", engine)
    envelope.setdefault("engine_version", version)
    envelope["mined_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    envelope["miner"] = "wave2_h15_closed_loop"

    round0_inv_count = len(envelope.get("invariants", []))
    wall_sec: dict[str, float] = {"round_0_load": time.perf_counter() - t_round0}
    observations: list[str] = [
        f"round_0: loaded {round0_inv_count} invariants from {baseline_inv_path.name}",
    ]

    engine_source_root = _detect_engine_source_root()
    if engine_source_root is None:
        observations.append(
            "round_0: no canonical transformers source tree found; "
            "failure prompts will omit source snippets"
        )
    else:
        observations.append(f"round_0: engine source root = {engine_source_root}")

    # ---- Rounds 1..max_rounds -------------------------------------------
    rounds_dir = out_dir / "rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    current_inv_path = rounds_dir / "round_0__baseline.yaml"
    current_inv_path.write_text(yaml.safe_dump(envelope, sort_keys=False))

    previous_failure_ids: set[str] = set()
    rounds_run = 0

    for round_idx in range(1, max_rounds + 1):
        t_round = time.perf_counter()

        # (a) runtime-validate
        runtime_results = _runtime_validate(
            current_inv_path, engine=engine, version_slug=version_slug
        )

        # (b) compute failures
        failures = _extract_failures(
            invariants=envelope.get("invariants", []),
            runtime_results=runtime_results,
        )
        current_failure_ids = {f.invariant_id for f in failures}
        observations.append(
            f"round_{round_idx}: gate produced {len(runtime_results)} results, "
            f"{len(failures)} failures"
        )

        # (c) termination
        if not failures:
            observations.append(f"round_{round_idx}: no failures; closing loop")
            wall_sec[f"round_{round_idx}_total"] = time.perf_counter() - t_round
            rounds_run = round_idx
            break
        if current_failure_ids == previous_failure_ids:
            observations.append(
                f"round_{round_idx}: failure set unchanged from previous round; "
                f"closing loop without further re-emission"
            )
            wall_sec[f"round_{round_idx}_total"] = time.perf_counter() - t_round
            rounds_run = round_idx
            break

        # (d) build prompt
        prompt = _build_failure_prompt(
            failures,
            round_idx=round_idx,
            max_rounds=max_rounds,
            engine_source_root=engine_source_root,
        )
        prompt_path = rounds_dir / f"round_{round_idx}__prompt.txt"
        prompt_path.write_text(prompt)
        observations.append(
            f"round_{round_idx}: prompt written ({len(prompt)} chars) -> {prompt_path.name}"
        )

        # (e) LLM dispatch (deferred boundary)
        reemitted = _invoke_llm_on_failure_list(
            prompt, ollama_host=ollama_host, model=model, round_idx=round_idx
        )

        # (f) merge by id
        envelope, n_corrected, n_spurious = _merge_reemitted_into_envelope(
            envelope=envelope, reemitted=reemitted, failures=failures
        )
        observations.append(
            f"round_{round_idx}: merged re-emission ({n_corrected} corrected, "
            f"{n_spurious} dropped-spurious)"
        )

        current_inv_path = rounds_dir / f"round_{round_idx}__post_reemit.yaml"
        current_inv_path.write_text(yaml.safe_dump(envelope, sort_keys=False))
        previous_failure_ids = current_failure_ids
        wall_sec[f"round_{round_idx}_total"] = time.perf_counter() - t_round
        rounds_run = round_idx

    # ---- Finalise -------------------------------------------------------
    final_inv_path = out_dir / "invariants.proposed.yaml"
    final_inv_path.write_text(yaml.safe_dump(envelope, sort_keys=False))

    final_schema_path: Path | None = None
    if baseline_schema_path.exists():
        final_schema_path = out_dir / "schema.discovered.json"
        shutil.copyfile(baseline_schema_path, final_schema_path)
        observations.append("schema: copied from (a) baseline (h15 does not touch schema)")
    else:
        observations.append(
            f"schema: (a) baseline schema missing at {baseline_schema_path}; emitted none"
        )

    wall_sec["total"] = sum(v for v in wall_sec.values())
    observations.append(
        f"summary: rounds_run={rounds_run}/{max_rounds}, "
        f"final_invariant_count={len(envelope.get('invariants', []))}"
    )

    return CellOutput(
        strategy_id=strategy_id,
        engine=engine,
        engine_version=version,
        schema_path=final_schema_path,
        invariants_path=final_inv_path,
        observations=observations,
        wall_sec=wall_sec,
        extra={
            "rounds_run": rounds_run,
            "max_rounds": max_rounds,
            "round_0_invariant_count": round0_inv_count,
            "final_invariant_count": len(envelope.get("invariants", [])),
        },
    )
