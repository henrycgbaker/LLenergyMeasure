"""Phase 1, Wave 1 - minimal LLM-extend integration probe.

Per PHASE1_WAVE1_PREREG.md. Re-points the PoC LLM-extend harness onto the new
runtime-gated GT + the REAL production gate (study_gt_pilot load+gate, i.e.
validate_invariants.py in-container) instead of the PoC hallucination proxy.

Role=extract, assembly=det-then-llm-extend, call-shape=single. OSS rung
(gemma3:12b) is chunked via the locked wg_extend prompt; the Opus rung is run
separately via the Agent tool over the whole validator source (its context
allows one call/cell) and its YAML is gated+scored here with --proposed.

Usage:
  wave1.py --engine E --vslug V --version X --image IMG --rung oss
  wave1.py --engine E --vslug V --version X --image IMG --rung opus --proposed FILE.yaml
Writes /tmp/phase1_w1_<engine>_<vslug>_<rung>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))

import study_gt_pilot as P  # noqa: E402
import wave2_llm_cells as L  # noqa: E402
import wave2_llm_source as W  # noqa: E402
import yaml  # noqa: E402

L.OLLAMA = "http://localhost:11434"  # live Ollama port (harness default was 11435)
# baseline floor = improved-det-v2 (the locked deterministic baseline)
L.FLOOR_ROOT = L.FINDINGS / "trial_runs" / "wave2" / "w2-a-improved-det-v2"


def gen_oss(
    engine: str, vslug: str, version_str: str, model: str, body: str | None = None
) -> tuple[list[dict], float, list[str]]:
    eng_ns, samp_ns = L.NS[engine]
    body = body or L.WG_BODY
    chunks = W.chunk_validator_source(W.source_files_for(engine, vslug))
    floor_blob = L.floor_summary_for_prompt(L.floor_invariants(engine, vslug))
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
            floor_invariants=floor_blob,
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


def dedup_vs_floor(engine: str, vslug: str, llm_invs: list[dict]) -> list[dict]:
    floor_keys = {k for inv in L.floor_invariants(engine, vslug) if (k := L.tolerant_key(inv))}
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for inv in llm_invs:
        k = L.tolerant_key(inv)
        if k is None or k in seen or k in floor_keys:
            continue
        seen.add(k)
        inv.setdefault("added_by", "llm_wg_extend")
        out.append(inv)
    return out


def gate(engine: str, vslug: str, image: str | None, llm_invs: list[dict]) -> list:
    """Gate LLM invariants through the REAL gate (validate_invariants in
    container) by routing them as a single 'llm' source through study_gt_pilot."""
    tmp = Path(f"/tmp/phase1_w1_{engine}_{vslug}_llmcorpus.yaml")
    tmp.write_text(yaml.safe_dump({"invariants": llm_invs}, sort_keys=False))
    P.configure(engine, vslug, image)
    P.SOURCES = {"llm": (tmp, False)}  # proposed schema, not GT schema
    cands = P.load_candidates()
    P.gate(cands)
    return cands


def gt_keys(engine: str, vslug: str) -> tuple[set, set, set]:
    p = P._FINDINGS / "study" / "ground_truth" / engine / vslug / "invariants" / "PILOT_GT.yaml"
    g = yaml.safe_load(p.read_text()) or {}
    conf = g.get("confirmed") or []
    tol = {tuple(e["tolerant_key"]) for e in conf if e.get("tolerant_key")}
    ck = {tuple(e["constraint_key"]) for e in conf if e.get("constraint_key")}
    return tol, ck, {t[0] for t in tol}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--rung", required=True, choices=["oss", "opus"])
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--proposed", default=None, help="opus rung: YAML of proposed invariants")
    ap.add_argument("--prompt-file", default=None, help="oss rung: override the prompt body")
    a = ap.parse_args()

    if a.rung == "oss":
        body = L.load_prompt_body(Path(a.prompt_file)) if a.prompt_file else None
        llm_invs, wall, raw = gen_oss(a.engine, a.vslug, a.version, a.model, body)
        Path(f"/tmp/phase1_w1_{a.engine}_{a.vslug}_oss_raw.txt").write_text(
            "\n\n===CHUNK===\n\n".join(raw)
        )
        model_tag = a.model
    else:
        doc = yaml.safe_load(Path(a.proposed).read_text()) or {}
        llm_invs = doc.get("invariants") or []
        wall = 0.0
        model_tag = "opus-4-8(agent-tool)"

    n_raw = len(llm_invs)
    deduped = dedup_vs_floor(a.engine, a.vslug, llm_invs)
    cands = gate(a.engine, a.vslug, a.image, deduped)

    tol, ck, gt_leaves = gt_keys(a.engine, a.vslug)
    floor_tol = {k for inv in L.floor_invariants(a.engine, a.vslug) if (k := L.tolerant_key(inv))}

    confirmed = [c for c in cands if c.verdict == "confirmed"]
    verdicts = {
        v: sum(1 for c in cands if c.verdict == v)
        for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
    }
    conf_tol = {c.tkey for c in confirmed}
    conf_ck = {c.ckey for c in confirmed}

    floor_recovered = floor_tol & tol
    llm_new_recovered = (conf_tol & tol) - floor_recovered  # GT keys LLM adds over the floor
    growth_ck = conf_ck - ck  # gate-confirmed, not in GT (GT-growth candidates)
    gateable = [c for c in cands if c.gateable]
    precision = len(confirmed) / len(gateable) if gateable else 0.0

    result = {
        "engine": a.engine,
        "vslug": a.vslug,
        "rung": a.rung,
        "model": model_tag,
        "n_llm_proposed_raw": n_raw,
        "n_deduped_vs_floor": len(deduped),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "gate_confirmed_precision": round(precision, 3),
        "gt_confirmed_total": len(tol),
        "floor_recall_tol": len(floor_recovered),
        "llm_new_gt_keys_recovered": sorted("|".join(k) for k in llm_new_recovered),
        "llm_lift_over_floor": len(llm_new_recovered),
        "gt_growth_ckeys": sorted("|".join(map(str, k)) for k in growth_ck),
        "gt_growth_count": len(growth_ck),
        "wall_sec": round(wall, 1),
    }
    out = Path(f"/tmp/phase1_w1_{a.engine}_{a.vslug}_{a.rung}.json")
    out.write_text(json.dumps(result, indent=2))
    print("PHASE1_W1_RESULT " + json.dumps(result), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
