"""Phase 1, Wave 4 - HYBRID langchain chain (consume + extend the deterministic floor).

A 2-stage langchain CHAIN that builds on deterministic findings (one cell among many):
  STAGE 1 (CONSUME + EXTEND): given the deterministic FLOOR (what the det miner
                  already found) + the source, propose ADDITIONAL invariants the
                  miner MISSED (no kwargs). The LLM consumes + extends det output.
  STAGE 2 (CONSTRUCT): given the extension candidates + AST class signatures, emit
                  constructible probe kwargs.
This is W-G (det-then-llm-extend) as a proper langchain chain with construction-
grounding. Scores LIFT over the floor + the combined hybrid recall vs GT.

Usage: wave4_hybrid_chain.py --engine E --vslug V --version X --model M [--image IMG]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wave2_llm_source as W  # noqa: E402
import wave4_construct as CONSTRUCT  # noqa: E402
import wave4_multistage as MS  # noqa: E402
import wave4_pure as PURE  # noqa: E402
import yaml  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

# the LOCKED deterministic floor (improved-det-v2) - the det findings to consume
PURE.L.FLOOR_ROOT = PURE.L.FINDINGS / "trial_runs" / "wave2" / "w2-a-improved-det-v2"

STAGE1 = """You are EXTENDING a deterministically-mined invariant catalogue for {engine} v{version}. A deterministic miner already found the invariants in INPUT 1 - do NOT re-emit those. Read the SOURCE in INPUT 2 and list ADDITIONAL invariants the miner MISSED: cross-field relations, conditional/feature-gate guards, presence checks, enum/Literal constraints. For THIS stage, no kwargs - just identify the rules.

INPUT 1 - FLOOR (already mined deterministically; do NOT duplicate):
{floor}

INPUT 2 - SOURCE:
{source}

Output ONLY a YAML document of EXTENSION invariants (what the floor missed), first two chars `in`:
invariants:
- id: <snake_case_with_{engine}_prefix>
  invariant_under_test: <one line>
  native_type: <ClassName>
  match:
    engine: {engine}
    fields:
      {field_namespace}.<field>: <predicate, e.g. {{'<': 0}} or {{present: true, not_in: [...]}}>

Emit the YAML now:"""


def gen_hybrid(engine, vslug, version_str, model, sigs):
    eng_ns, samp_ns = PURE.L.NS[engine]
    chunks = W.chunk_validator_source(W.source_files_for(engine, vslug))
    floor_blob = PURE.L.floor_summary_for_prompt(PURE.L.floor_invariants(engine, vslug))
    llm = ChatOllama(
        model=model, base_url=PURE.L.OLLAMA, temperature=0, num_ctx=16384, num_predict=4096
    )
    invs: list[dict] = []
    raw: list[str] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        s1 = MS._chat(
            llm,
            STAGE1.format(
                engine=engine,
                version=version_str,
                field_namespace=eng_ns,
                floor=floor_blob,
                source=chunk,
            ),
        )
        cand = PURE.L.parse_invariants(s1)
        if not cand:
            raw.append(f"[CHUNK {ci} STAGE1 no extension]")
            continue
        present = [cn for cn in sigs if re.search(r"\b" + re.escape(cn) + r"\b", chunk)]
        sig_block = CONSTRUCT.format_sig_block(sigs, present)
        cand_yaml = yaml.safe_dump({"invariants": cand}, sort_keys=False)
        s2 = MS._chat(
            llm,
            MS.STAGE2.format(
                engine=engine,
                version=version_str,
                class_signatures=sig_block,
                candidates=cand_yaml,
                source=chunk,
            ),
        )
        raw.append(f"=STAGE1=\n{s1}\n=STAGE2=\n{s2}")
        invs.extend(PURE.L.parse_invariants(s2))
    return invs, time.time() - t0, raw


def dedup_vs_floor(engine, vslug, invs):
    floor_keys = {
        k for inv in PURE.L.floor_invariants(engine, vslug) if (k := PURE.L.tolerant_key(inv))
    }
    seen: set = set()
    out: list[dict] = []
    for inv in invs:
        k = PURE.L.tolerant_key(inv)
        if k is None or k in seen or k in floor_keys:
            continue
        seen.add(k)
        inv.setdefault("added_by", "llm_hybrid_chain")
        out.append(inv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    sigs = CONSTRUCT.extract_signatures(W.source_files_for(a.engine, a.vslug))
    invs, wall, raw = gen_hybrid(a.engine, a.vslug, a.version, a.model, sigs)
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    Path(f"/tmp/phase1_w4hc_{a.engine}_{a.vslug}_{tag}_raw.txt").write_text(
        "\n\n===CHUNK===\n\n".join(raw)
    )

    extension = dedup_vs_floor(a.engine, a.vslug, invs)  # what the LLM adds OVER the floor
    cands = PURE.gate(a.engine, a.vslug, a.image, extension)
    tol, ck = PURE.gt_keys(a.engine, a.vslug)
    confirmed = [c for c in cands if c.verdict == "confirmed"]
    verdicts = {
        v: sum(1 for c in cands if c.verdict == v)
        for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
    }
    conf_tol = {c.tkey for c in confirmed}
    gateable = [c for c in cands if c.gateable]

    floor_keys = {
        k for inv in PURE.L.floor_invariants(a.engine, a.vslug) if (k := PURE.L.tolerant_key(inv))
    }
    floor_rec = floor_keys & tol
    llm_rec = conf_tol & tol
    hybrid_rec = floor_rec | llm_rec

    result = {
        "engine": a.engine,
        "vslug": a.vslug,
        "shape": "hybrid_chain",
        "model": a.model,
        "n_extension_raw": len(invs),
        "n_extension_deduped": len(extension),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "gate_confirmed_precision": round(len(confirmed) / len(gateable), 3) if gateable else 0.0,
        "gt_confirmed_total": len(tol),
        "floor_recall": len(floor_rec),
        "llm_lift_over_floor": len(llm_rec - floor_rec),
        "hybrid_recall": len(hybrid_rec),
        "hybrid_recall_frac": round(len(hybrid_rec) / len(tol), 3) if tol else 0.0,
        "wall_sec": round(wall, 1),
    }
    Path(f"/tmp/phase1_w4hc_{a.engine}_{a.vslug}_{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    Path(f"/tmp/phase1_w4hc_{a.engine}_{a.vslug}_{tag}_corpus.yaml").write_text(
        yaml.safe_dump({"invariants": extension}, sort_keys=False)
    )
    print("PHASE1_W4HC_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
