"""Phase 1, Wave 4 - SELF-CONSISTENCY (k-vote) on construction-grounding.

Runs the construct-grounded prompt k times per chunk at temperature>0 and UNIONS
the invariants (recall-oriented: catch what any single sample found). Targets the
genuine cross-field tail that single-shot construction-grounding misses. Then the
standard internal-dedup + gate + recall-vs-GT scoring. Per PHASE1_WAVE4_PREREG.md.

Usage: wave4_selfconsistency.py --engine E --vslug V --version X --model M
         --prompt-file P [--k 3] [--temp 0.7] [--image IMG]
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
import wave4_pure as PURE  # noqa: E402
import yaml  # noqa: E402


def gen_selfconsistency(engine, vslug, version_str, model, body, sigs, k, temp):
    eng_ns, samp_ns = PURE.L.NS[engine]
    chunks = W.chunk_validator_source(W.source_files_for(engine, vslug))
    invs: list[dict] = []
    raw: list[str] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        present = [cn for cn in sigs if re.search(r"\b" + re.escape(cn) + r"\b", chunk)]
        sig_block = CONSTRUCT.format_sig_block(sigs, present)
        prompt = PURE.L.render_prompt(
            body,
            engine=engine,
            engine_version=version_str,
            field_namespace=eng_ns,
            sampling_namespace=samp_ns,
            class_signatures=sig_block,
            source=chunk,
        )
        for ki in range(k):
            try:
                resp = PURE.L.ollama_generate(model, prompt, temperature=temp)
            except Exception as ex:
                raw.append(f"[CHUNK {ci} SAMPLE {ki} ERROR: {ex}]")
                continue
            raw.append(resp)
            invs.extend(PURE.L.parse_invariants(resp))  # UNION across samples
    return invs, time.time() - t0, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()

    body = PURE.L.load_prompt_body(Path(a.prompt_file))
    sigs = CONSTRUCT.extract_signatures(W.source_files_for(a.engine, a.vslug))
    invs, wall, raw = gen_selfconsistency(
        a.engine, a.vslug, a.version, a.model, body, sigs, a.k, a.temp
    )
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    Path(f"/tmp/phase1_w4sc_{a.engine}_{a.vslug}_{tag}_raw.txt").write_text(
        "\n\n===CHUNK===\n\n".join(raw)
    )

    n_raw = len(invs)
    deduped = PURE.dedup_internal(invs)
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
        "shape": f"selfconsistency_k{a.k}_t{a.temp}",
        "model": a.model,
        "k": a.k,
        "temp": a.temp,
        "n_llm_proposed_raw": n_raw,
        "n_deduped_internal": len(deduped),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "gate_confirmed_precision": round(len(confirmed) / len(gateable), 3) if gateable else 0.0,
        "gt_confirmed_total": len(tol),
        "recall_vs_gt_tol": len(conf_tol & tol),
        "recall_frac": round(len(conf_tol & tol) / len(tol), 3) if tol else 0.0,
        "gt_growth_count": len(conf_ck - ck),
        "wall_sec": round(wall, 1),
    }
    Path(f"/tmp/phase1_w4sc_{a.engine}_{a.vslug}_{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    Path(f"/tmp/phase1_w4sc_{a.engine}_{a.vslug}_{tag}_corpus.yaml").write_text(
        yaml.safe_dump({"invariants": deduped}, sort_keys=False)
    )
    print("PHASE1_W4SC_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
