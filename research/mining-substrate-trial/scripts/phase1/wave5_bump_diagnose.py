"""Phase 1, Wave 5 (CROSS-BUMP) - LLM bump-diagnose (the W-F diff-reviewer).

The defensible LLM role per the strategic review + wave2_bump_survivability: when a
bump degrades carried knowledge, an LLM diff-reviewer compares the OLD catalogue
against the NEW source and flags what BROKE + what is NEW - the external signal the
silent deterministic substrate cannot raise about itself.

Per new-source chunk: give the LLM the OLD catalogue entries for the classes in the
chunk + the NEW source, ask for a STRUCTURED diagnosis (broke / new). Aggregate and
score against the actual GT diff (old GT keys vs new GT keys) by tolerant leaf.

Usage: wave5_bump_diagnose.py --engine E --old-vslug V_OLD --new-vslug V_NEW
         --old-version X_OLD --new-version X_NEW --model M
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
import wave4_multistage as MS  # noqa: E402
import wave4_pure as PURE  # noqa: E402
import yaml  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

DIAGNOSE = """You are reviewing an inference engine config across an UPSTREAM VERSION BUMP ({engine} {old_version} -> {new_version}). We previously mined the OLD catalogue below; the engine has since bumped. Compare the OLD catalogue against the NEW source and flag what changed - this is a degradation review, not exhaustive mining.

OLD CATALOGUE (what we knew for {old_version}, for the classes in this source):
{old_catalogue}

NEW SOURCE ({new_version}, this chunk):
{source}

Output ONLY a YAML document, first two chars `br`, with two lists:
broke:    # OLD catalogue entries whose validator is REMOVED or CHANGED in the new source (stale knowledge)
- field: <namespaced field from the old entry>
  native_type: <class>
  reason: <one line: removed / moved to declarative Field / bound changed / now feature-gated>
new:      # invariants VISIBLE in the new source that are NOT in the old catalogue (the new surface)
- field: <namespaced field>
  native_type: <class>
  invariant_under_test: <one line>

If nothing changed for this chunk, emit `broke: []` and `new: []`. Emit the YAML now:"""


def _gt_catalogue(engine, vslug):
    p = (
        PURE.P._FINDINGS
        / "study"
        / "ground_truth"
        / engine
        / vslug
        / "invariants"
        / "PILOT_GT.yaml"
    )
    gt = yaml.safe_load(p.read_text()) or {}
    return gt.get("confirmed") or []


def _leaf(e):
    tk = e.get("tolerant_key")
    if tk:
        return str(tk[0])
    return str(e.get("native_field", "")).split(".")[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--old-vslug", required=True)
    ap.add_argument("--new-vslug", required=True)
    ap.add_argument("--old-version", required=True)
    ap.add_argument("--new-version", required=True)
    ap.add_argument("--model", required=True)
    a = ap.parse_args()

    old_gt = _gt_catalogue(a.engine, a.old_vslug)
    new_gt = _gt_catalogue(a.engine, a.new_vslug)
    old_leaves = {_leaf(e) for e in old_gt}
    new_leaves = {_leaf(e) for e in new_gt}
    actually_broke = old_leaves - new_leaves  # old fields gone in new GT
    actually_new = new_leaves - old_leaves  # new fields absent in old GT

    # old catalogue compacted by class, for prompt injection
    by_class: dict[str, list[str]] = {}
    for e in old_gt:
        cls = (e.get("native_type") or "?").split(".")[-1]
        by_class.setdefault(cls, []).append(
            f"{_leaf(e)}: {e.get('invariant_under_test') or e.get('predicate_kind')}"
        )

    chunks = W.chunk_validator_source(W.source_files_for(a.engine, a.new_vslug))
    llm = ChatOllama(
        model=a.model, base_url=PURE.L.OLLAMA, temperature=0, num_ctx=16384, num_predict=4096
    )

    diag_broke: set[str] = set()
    diag_new: set[str] = set()
    raw: list[str] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        present = [c for c in by_class if re.search(r"\b" + re.escape(c) + r"\b", chunk)]
        if not present:
            continue
        old_cat = "\n".join(f"class {c}:\n  " + "\n  ".join(by_class[c]) for c in present)
        resp = MS._chat(
            llm,
            DIAGNOSE.format(
                engine=a.engine,
                old_version=a.old_version,
                new_version=a.new_version,
                old_catalogue=old_cat,
                source=chunk,
            ),
        )
        raw.append(resp)
        try:
            d = yaml.safe_load(PURE.L._strip_fences(resp))
        except yaml.YAMLError:
            continue
        if isinstance(d, dict):
            for item in d.get("broke") or []:
                if isinstance(item, dict) and item.get("field"):
                    diag_broke.add(str(item["field"]).split(".")[-1])
            for item in d.get("new") or []:
                if isinstance(item, dict) and item.get("field"):
                    diag_new.add(str(item["field"]).split(".")[-1])

    def prf(pred, truth):
        tp = len(pred & truth)
        p = tp / len(pred) if pred else 0.0
        r = tp / len(truth) if truth else 0.0
        return round(p, 3), round(r, 3), tp

    bp, br, btp = prf(diag_broke, actually_broke)
    np_, nr, ntp = prf(diag_new, actually_new)
    result = {
        "engine": a.engine,
        "bump": f"{a.old_vslug}->{a.new_vslug}",
        "model": a.model,
        "actually_broke_n": len(actually_broke),
        "actually_new_n": len(actually_new),
        "diag_broke_n": len(diag_broke),
        "diag_new_n": len(diag_new),
        "broke_precision": bp,
        "broke_recall": br,
        "broke_tp": btp,
        "new_precision": np_,
        "new_recall": nr,
        "new_tp": ntp,
        "alarm_raised": len(diag_broke) > 0 or len(diag_new) > 0,
        "wall_sec": round(time.time() - t0, 1),
    }
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    Path(f"/tmp/wave5_diagnose_{a.engine}_{a.old_vslug}_to_{a.new_vslug}_{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    Path(f"/tmp/wave5_diagnose_{a.engine}_{a.old_vslug}_to_{a.new_vslug}_{tag}_raw.txt").write_text(
        "\n\n===CHUNK===\n\n".join(raw)
    )
    print("WAVE5_DIAGNOSE_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
