"""Phase 1, Wave 4 - pure-LLM AGENTIC (LangGraph) extraction. Still llm-only (no
deterministic floor), but the model drives an agent with tools to explore the
engine source and build the invariant catalogue.

  --mode extract_only  (4b): tools = list/grep/read source + emit_invariant.
                              The gate scores the catalogue AFTER the run.
  --mode gate_tool     (4c): adds gate_probe(...) so the agent tests each probe
                              against the REAL gate and self-corrects before
                              emitting. (Expensive: one container gate call each.)

Both score RECALL vs the runtime-gated GT, like wave4_pure. Tool-calling model
served by the containerized ollama (WAVE_OLLAMA, default :11435).

Usage: wave4_agentic.py --engine E --vslug V --version X --model M
         --mode {extract_only,gate_tool} [--image IMG] [--max-steps 60]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))

import wave2_llm_source as W  # noqa: E402
import yaml  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402

# reuse wave4_pure's gate + scoring (same package dir)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wave4_pure as PURE  # noqa: E402

OLLAMA = os.environ.get("WAVE_OLLAMA", "http://localhost:11435")


def build_tools(engine: str, vslug: str, image: str | None, mode: str, catalogue: list):
    """Tool set closed over the cell context + the accumulating catalogue."""
    files = W.source_files_for(engine, vslug)
    file_map = {str(f): f for f in files if f.exists()}

    @tool
    def list_source_files() -> str:
        """List the engine validator source files available to read (path + line count)."""
        out = []
        for s, f in file_map.items():
            n = len(f.read_text().splitlines())
            out.append(f"{s}  ({n} lines)")
        return "\n".join(out)

    @tool
    def read_source(path: str, start_line: int = 1, end_line: int = 200) -> str:
        """Read lines [start_line, end_line] of a source file (max 200 lines/call). Use a path from list_source_files."""
        f = file_map.get(path) or next((v for k, v in file_map.items() if path in k), None)
        if not f:
            return f"NO SUCH FILE: {path}. Use list_source_files first."
        lines = f.read_text().splitlines()
        end_line = min(end_line, start_line + 199, len(lines))
        chunk = lines[start_line - 1 : end_line]
        return "\n".join(f"{i + start_line}: {ln}" for i, ln in enumerate(chunk))

    @tool
    def grep_source(pattern: str, file: str = "", context_lines: int = 0) -> str:
        """Regex-search validator source. pattern=regex (required). file=optional filename substring to restrict the search to one file. context_lines=optional N lines of context around each hit. Returns up to 50 hits."""
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"BAD REGEX: {e}"
        targets = list(file_map.items())
        if file:
            targets = [(s, fp) for s, fp in file_map.items() if file in s]
            if not targets:
                return f"no file matching '{file}'. Call list_source_files for valid paths."
        hits: list[str] = []
        for s, fp in targets:
            lines = fp.read_text().splitlines()
            for i, ln in enumerate(lines, 1):
                if rx.search(ln):
                    if context_lines and context_lines > 0:
                        lo, hi = max(0, i - 1 - context_lines), min(len(lines), i + context_lines)
                        ctx = "\n".join(f"  {j + 1}: {lines[j]}" for j in range(lo, hi))
                        hits.append(f"{Path(s).name}:{i}:\n{ctx}")
                    else:
                        hits.append(f"{Path(s).name}:{i}: {ln.strip()[:160]}")
                    if len(hits) >= 50:
                        return "\n".join(hits) + "\n...(truncated; refine pattern or pass file=)"
        return "\n".join(hits) if hits else "no matches"

    @tool
    def emit_invariant(
        id: str,
        invariant_under_test: str,
        native_type: str,
        match_fields_json: str,
        kwargs_positive_json: str,
        kwargs_negative_json: str,
        severity: str = "error",
    ) -> str:
        """Record ONE extracted invariant. match_fields_json = JSON of {namespaced_field: predicate}. kwargs_*_json = JSON dict of constructor kwargs (positive triggers the rule, negative passes). Use {} or null if a probe is unconstructible."""
        try:
            match_fields = json.loads(match_fields_json) if match_fields_json else {}
            kp = (
                json.loads(kwargs_positive_json)
                if kwargs_positive_json not in ("", "null", None)
                else None
            )
            kn = (
                json.loads(kwargs_negative_json)
                if kwargs_negative_json not in ("", "null", None)
                else None
            )
        except json.JSONDecodeError as e:
            return f"INVALID JSON: {e}. Re-emit with valid JSON for match/kwargs."
        inv = {
            "id": id,
            "engine": engine,
            "invariant_under_test": invariant_under_test,
            "severity": severity,
            "native_type": native_type,
            "match": {"engine": engine, "fields": match_fields},
            "kwargs_positive": kp,
            "kwargs_negative": kn,
            "added_by": f"llm_agentic_{mode}",
        }
        catalogue.append(inv)
        return f"recorded invariant {id} (total so far: {len(catalogue)})"

    tools = [list_source_files, read_source, grep_source, emit_invariant]

    if mode == "gate_tool":

        @tool
        def gate_probe(
            native_type: str,
            match_fields_json: str,
            kwargs_positive_json: str,
            kwargs_negative_json: str,
        ) -> str:
            """Test ONE candidate invariant against the REAL runtime gate BEFORE emitting. Returns the verdict (confirmed/failed/infra_error/skipped). Use this to check your kwargs actually fire the rule, then fix and re-test if needed."""
            try:
                mf = json.loads(match_fields_json) if match_fields_json else {}
                kp = (
                    json.loads(kwargs_positive_json)
                    if kwargs_positive_json not in ("", "null", None)
                    else None
                )
                kn = (
                    json.loads(kwargs_negative_json)
                    if kwargs_negative_json not in ("", "null", None)
                    else None
                )
            except json.JSONDecodeError as e:
                return f"INVALID JSON: {e}"
            probe = [
                {
                    "id": "probe",
                    "engine": engine,
                    "invariant_under_test": "probe",
                    "severity": "error",
                    "native_type": native_type,
                    "match": {"engine": engine, "fields": mf},
                    "kwargs_positive": kp,
                    "kwargs_negative": kn,
                }
            ]
            try:
                cands = PURE.gate(engine, vslug, image, probe)
            except Exception as e:
                return f"gate error: {e}"
            if not cands:
                return "ungateable (no native_type / kwargs)"
            c = cands[0]
            return f"verdict={c.verdict}; error={(c.error or '')[:200]}"

        tools.append(gate_probe)

    return tools


SYSTEM_4B = """You are extracting the COMPLETE validation-invariant catalogue for {engine} v{version} from its source code, using tools.

An invariant is a rule the library enforces at config/params construction: `if <predicate>: raise ValueError/TypeError(...)`, `warnings.warn(...)`, or a silent normalisation.

CRITICAL BEHAVIOUR: ACT ONLY BY CALLING TOOLS. NEVER write prose describing what you will do next - if you intend to use a tool, CALL IT in the same turn. Every turn you take MUST be a tool call until you are completely finished. Do NOT stop after listing files or after a few invariants - keep calling grep_source / read_source / emit_invariant until you have worked through ALL the source files.

PROCESS:
1. Call list_source_files.
2. Call grep_source for 'raise ', 'warnings.warn', '__post_init__', 'def _verify', 'def validate' and read_source around each hit to read the validators in EVERY file.
3. For EACH invariant call emit_invariant with a constructible kwargs_positive (constructor kwargs that TRIGGER the rule) and kwargs_negative (that PASS). Include every REQUIRED constructor field. For cross-field rules set all fields in the relation. Use null kwargs only if the class truly cannot be built in isolation.
4. Cover single-field bounds/types/enums AND cross-field relations, conditional/feature-gate guards, presence checks.
5. ONLY after you have emit_invariant'd every invariant across ALL files, reply with the single word DONE.

Field namespace: engine params -> `{field_namespace}`, sampling -> `{sampling_namespace}`."""

SYSTEM_4C = (
    SYSTEM_4B
    + "\n\nBEFORE emitting each invariant, call gate_probe to verify your kwargs_positive actually fires the rule (verdict=confirmed). If it is failed/infra_error, FIX the kwargs (wrong direction, missing required field, wrong class) and re-probe. Only emit invariants you have confirmed, or genuinely-unconstructible ones with null kwargs."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", required=True, choices=["extract_only", "gate_tool"])
    ap.add_argument("--max-steps", type=int, default=80)
    a = ap.parse_args()

    eng_ns, samp_ns = PURE.L.NS[a.engine]
    catalogue: list[dict] = []
    tools = build_tools(a.engine, a.vslug, a.image, a.mode, catalogue)
    llm = ChatOllama(model=a.model, base_url=OLLAMA, temperature=0)

    sys_tmpl = SYSTEM_4B if a.mode == "extract_only" else SYSTEM_4C
    system = sys_tmpl.format(
        engine=a.engine, version=a.version, field_namespace=eng_ns, sampling_namespace=samp_ns
    )
    # system prompt MUST go via prompt=, not as an input message
    agent = create_react_agent(llm, tools, prompt=system)

    t0 = time.time()
    err = ""
    try:
        agent.invoke(
            {
                "messages": [
                    (
                        "user",
                        "Begin. Call list_source_files now, then work through EVERY file with "
                        "grep_source/read_source and call emit_invariant for each invariant. "
                        "Respond ONLY with tool calls until done.",
                    )
                ]
            },
            config={"recursion_limit": a.max_steps},
        )
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    wall = time.time() - t0

    deduped = PURE.dedup_internal(catalogue)
    cands = PURE.gate(a.engine, a.vslug, a.image, deduped)
    tol, ck = PURE.gt_keys(a.engine, a.vslug)
    confirmed = [c for c in cands if c.verdict == "confirmed"]
    verdicts = {
        v: sum(1 for c in cands if c.verdict == v)
        for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
    }
    conf_tol = {c.tkey for c in confirmed}
    conf_ck = {c.ckey for c in confirmed}
    gateable = [c for c in cands if c.gateable]

    result = {
        "engine": a.engine,
        "vslug": a.vslug,
        "shape": f"pure_agentic_{a.mode}",
        "model": a.model,
        "n_emitted": len(catalogue),
        "n_deduped_internal": len(deduped),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "gate_confirmed_precision": round(len(confirmed) / len(gateable), 3) if gateable else 0.0,
        "gt_confirmed_total": len(tol),
        "recall_vs_gt_tol": len(conf_tol & tol),
        "recall_frac": round(len(conf_tol & tol) / len(tol), 3) if tol else 0.0,
        "gt_growth_ckeys": sorted("|".join(map(str, k)) for k in (conf_ck - ck)),
        "gt_growth_count": len(conf_ck - ck),
        "wall_sec": round(wall, 1),
        "agent_error": err,
    }
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    Path(f"/tmp/phase1_w4_{a.engine}_{a.vslug}_{a.mode}_{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    Path(f"/tmp/phase1_w4_{a.engine}_{a.vslug}_{a.mode}_{tag}_catalogue.yaml").write_text(
        yaml.safe_dump({"invariants": catalogue}, sort_keys=False)
    )
    print("PHASE1_W4_AGENTIC_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
