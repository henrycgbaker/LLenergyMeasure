"""Phase 1, Wave 4 - MULTI-STAGE reasoning chain (langchain, no agent).

Decomposes the conflated single-shot task into a fixed langchain CHAIN (NOT a
ReAct agent - so no model-driven tool-calling, which is what OSS flakes at):
  STAGE 1 (FIND):      read source chunk -> list candidate invariants (rule +
                       native_type + match), NO kwargs. Pure rule discovery.
  STAGE 2 (CONSTRUCT): given stage-1 candidates + AST class signatures -> emit
                       constructible kwargs_positive/negative for each. Pure
                       probe construction.
Targets the hypothesis that a larger model (70B) + decomposition exceeds 32B-code
single-shot. Then standard internal-dedup + gate + recall-vs-GT scoring.

Usage: wave4_multistage.py --engine E --vslug V --version X --model M [--image IMG]
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
from langchain_core.messages import HumanMessage  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

STAGE1 = """You are a code analyser. From the {engine} v{version} validator SOURCE below, list EVERY validation invariant you can find (a rule enforced at config/params construction: `if <pred>: raise/warn`, a Literal/enum field constraint, a cross-field relation). For THIS stage, do NOT write probe kwargs - just identify the rules.

Output ONLY a YAML document, first two chars `in`:
invariants:
- id: <snake_case_with_{engine}_prefix>
  invariant_under_test: <one line>
  native_type: <ClassName>
  match:
    engine: {engine}
    fields:
      {field_namespace}.<field>: <predicate, e.g. {{'<': 0}} or {{present: true, not_in: [...]}}>

SOURCE:
{source}

Emit the YAML now:"""

STAGE2 = """You are constructing runnable gate probes for invariants already found in {engine} v{version}. For EACH invariant below, add a kwargs_positive (constructor kwargs that TRIGGER the rule) and kwargs_negative (that PASS), using the CONSTRUCTOR SIGNATURES to make them constructible.

CONSTRUCTOR SIGNATURES (include every REQUIRED field; use exact field names):
{class_signatures}

INVARIANTS TO PROBE:
{candidates}

Output ONLY a YAML document, first two chars `in`, EACH item = the original invariant PLUS kwargs:
invariants:
- id: <same id>
  invariant_under_test: <same>
  native_type: <same ClassName>
  match: <same match block>
  kwargs_positive:
    <ctor_arg>: <value that fires the rule>
  kwargs_negative:
    <ctor_arg>: <value that passes>
  added_by: llm_multistage

Rules: use exact ctor field names; include every REQUIRED field; for cross-field rules set all fields in the relation; null kwargs only if genuinely unconstructible. Emit the YAML now:"""


def _chat(llm, text: str) -> str:
    return llm.invoke([HumanMessage(content=text)]).content or ""


def gen_multistage(engine, vslug, version_str, model, sigs):
    eng_ns, samp_ns = PURE.L.NS[engine]
    chunks = W.chunk_validator_source(W.source_files_for(engine, vslug))
    llm = ChatOllama(
        model=model, base_url=PURE.L.OLLAMA, temperature=0, num_ctx=16384, num_predict=4096
    )
    invs: list[dict] = []
    raw: list[str] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        s1 = _chat(
            llm,
            STAGE1.format(engine=engine, version=version_str, field_namespace=eng_ns, source=chunk),
        )
        cand = PURE.L.parse_invariants(s1)
        if not cand:
            raw.append(f"[CHUNK {ci} STAGE1 no candidates]")
            continue
        present = [cn for cn in sigs if re.search(r"\b" + re.escape(cn) + r"\b", chunk)]
        sig_block = CONSTRUCT.format_sig_block(sigs, present)
        cand_yaml = yaml.safe_dump({"invariants": cand}, sort_keys=False)
        s2 = _chat(
            llm,
            STAGE2.format(
                engine=engine, version=version_str, class_signatures=sig_block, candidates=cand_yaml
            ),
        )
        raw.append(f"=STAGE1=\n{s1}\n=STAGE2=\n{s2}")
        invs.extend(PURE.L.parse_invariants(s2))
    return invs, time.time() - t0, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    sigs = CONSTRUCT.extract_signatures(W.source_files_for(a.engine, a.vslug))
    invs, wall, raw = gen_multistage(a.engine, a.vslug, a.version, a.model, sigs)
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    Path(f"/tmp/phase1_w4ms_{a.engine}_{a.vslug}_{tag}_raw.txt").write_text(
        "\n\n===CHUNK===\n\n".join(raw)
    )

    deduped = PURE.dedup_internal(invs)
    cands = PURE.gate(a.engine, a.vslug, a.image, deduped)
    tol, ck = PURE.gt_keys(a.engine, a.vslug)
    confirmed = [c for c in cands if c.verdict == "confirmed"]
    verdicts = {
        v: sum(1 for c in cands if c.verdict == v)
        for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
    }
    conf_tol = {c.tkey for c in confirmed}
    gateable = [c for c in cands if c.gateable]
    result = {
        "engine": a.engine,
        "vslug": a.vslug,
        "shape": "multistage_chain",
        "model": a.model,
        "n_llm_proposed_raw": len(invs),
        "n_deduped_internal": len(deduped),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "gate_confirmed_precision": round(len(confirmed) / len(gateable), 3) if gateable else 0.0,
        "gt_confirmed_total": len(tol),
        "recall_vs_gt_tol": len(conf_tol & tol),
        "recall_frac": round(len(conf_tol & tol) / len(tol), 3) if tol else 0.0,
        "wall_sec": round(wall, 1),
    }
    Path(f"/tmp/phase1_w4ms_{a.engine}_{a.vslug}_{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    Path(f"/tmp/phase1_w4ms_{a.engine}_{a.vslug}_{tag}_corpus.yaml").write_text(
        yaml.safe_dump({"invariants": deduped}, sort_keys=False)
    )
    print("PHASE1_W4MS_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
