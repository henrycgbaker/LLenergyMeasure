"""Phase 1, Wave 4 - pure-LLM (llm-only assembly, NO deterministic floor) extract
with kwargs-emission. Call-shape 4a = prompt (chunked). The LLM extracts the FULL
catalogue from source unaided; the full catalogue is deduped INTERNALLY (not vs
floor), gated through the real production gate, and scored for RECALL vs the
runtime-gated GT. Reuses the wave-1/2 gate path. Per PHASE1_WAVE4_PREREG.md.

Usage: wave4_pure.py --engine E --vslug V --version X --model M --prompt-file P
       [--image IMG]   (WAVE_OLLAMA env points at the ollama server)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))

import study_gt_pilot as P  # noqa: E402
import wave2_llm_cells as L  # noqa: E402
import wave2_llm_source as W  # noqa: E402
import yaml  # noqa: E402

L.OLLAMA = os.environ.get("WAVE_OLLAMA", "http://localhost:11434")


def gen_pure(
    engine: str, vslug: str, version_str: str, model: str, body: str
) -> tuple[list[dict], float, list[str]]:
    """Chunked LLM-only extraction - NO floor injected into the prompt."""
    eng_ns, samp_ns = L.NS[engine]
    chunks = W.chunk_validator_source(W.source_files_for(engine, vslug))
    invs: list[dict] = []
    raw: list[str] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        prompt = L.render_prompt(
            body,
            engine=engine,
            engine_version=version_str,
            field_namespace=eng_ns,
            sampling_namespace=samp_ns,
            source=chunk,
        )
        try:
            resp = L.ollama_generate(model, prompt)
        except Exception as ex:
            raw.append(f"[CHUNK {ci} ERROR: {ex}]")
            continue
        raw.append(resp)
        invs.extend(L.parse_invariants(resp))
    return invs, time.time() - t0, raw


def dedup_internal(invs: list[dict]) -> list[dict]:
    """Dedup by tolerant key ONLY (no floor subtraction - this is llm-only)."""
    seen: set = set()
    out: list[dict] = []
    for inv in invs:
        k = L.tolerant_key(inv)
        if k is None or k in seen:
            continue
        seen.add(k)
        inv.setdefault("added_by", "llm_pure_kwargs")
        out.append(inv)
    return out


def gate(engine: str, vslug: str, image: str | None, invs: list[dict]) -> list:
    tmp = Path(f"/tmp/phase1_w4_{engine}_{vslug}_llmcorpus.yaml")
    tmp.write_text(yaml.safe_dump({"invariants": invs}, sort_keys=False))
    P.configure(engine, vslug, image)
    P.SOURCES = {"llm": (tmp, False)}
    cands = P.load_candidates()
    P.gate(cands)
    return cands


def gt_keys(engine: str, vslug: str) -> tuple[set, set]:
    p = P._FINDINGS / "study" / "ground_truth" / engine / vslug / "invariants" / "PILOT_GT.yaml"
    g = yaml.safe_load(p.read_text()) or {}
    conf = g.get("confirmed") or []
    tol = {tuple(e["tolerant_key"]) for e in conf if e.get("tolerant_key")}
    ck = {tuple(e["constraint_key"]) for e in conf if e.get("constraint_key")}
    return tol, ck


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    a = ap.parse_args()

    body = L.load_prompt_body(Path(a.prompt_file))
    invs, wall, raw = gen_pure(a.engine, a.vslug, a.version, a.model, body)
    Path(f"/tmp/phase1_w4_{a.engine}_{a.vslug}_raw.txt").write_text("\n\n===CHUNK===\n\n".join(raw))

    n_raw = len(invs)
    deduped = dedup_internal(invs)
    cands = gate(a.engine, a.vslug, a.image, deduped)

    tol, ck = gt_keys(a.engine, a.vslug)
    confirmed = [c for c in cands if c.verdict == "confirmed"]
    verdicts = {
        v: sum(1 for c in cands if c.verdict == v)
        for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
    }
    conf_tol = {c.tkey for c in confirmed}
    conf_ck = {c.ckey for c in confirmed}
    gateable = [c for c in cands if c.gateable]
    recall_tol = conf_tol & tol
    growth_ck = conf_ck - ck
    precision = len(confirmed) / len(gateable) if gateable else 0.0

    result = {
        "engine": a.engine,
        "vslug": a.vslug,
        "shape": "pure_llm_prompt",
        "model": a.model,
        "n_llm_proposed_raw": n_raw,
        "n_deduped_internal": len(deduped),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "gate_confirmed_precision": round(precision, 3),
        "gt_confirmed_total": len(tol),
        "recall_vs_gt_tol": len(recall_tol),
        "recall_frac": round(len(recall_tol) / len(tol), 3) if tol else 0.0,
        "gt_growth_ckeys": sorted("|".join(map(str, k)) for k in growth_ck),
        "gt_growth_count": len(growth_ck),
        "wall_sec": round(wall, 1),
    }
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    out = Path(f"/tmp/phase1_w4_{a.engine}_{a.vslug}_{tag}.json")
    out.write_text(json.dumps(result, indent=2))
    print("PHASE1_W4_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
