"""Phase 1, Wave 4 - CONSTRUCTION-GROUNDED extraction strategy.

Attacks the dominant infra_error wall from 4a: the LLM emits kwargs that cannot
construct the config (missing required args). Here a deterministic AST pass
extracts each class's constructor signature (REQUIRED / OPTIONAL fields + types)
and injects it into the prompt, so the LLM emits CONSTRUCTIBLE kwargs. Still
llm-only for the invariant content (no floor of invariants). A det+LLM hybrid:
det does construction-context discovery, LLM does synthesis.

Same gate + recall-vs-GT scoring as wave4_pure. Per PHASE1_WAVE4_PREREG.md.

Usage: wave4_construct.py --engine E --vslug V --version X --model M
         --prompt-file P [--image IMG]   (WAVE_OLLAMA env -> ollama server)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import wave2_llm_source as W  # noqa: E402
import wave4_pure as PURE  # noqa: E402
import yaml  # noqa: E402


def _class_signature(node: ast.ClassDef) -> tuple[list[str], list[str]]:
    required: list[str] = []
    optional: list[str] = []
    for item in node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            name = item.target.id
            if name.startswith("__"):
                continue
            try:
                typ = ast.unparse(item.annotation)
            except Exception:
                typ = "?"
            if item.value is None:
                required.append(f"{name}: {typ}")
            else:
                try:
                    dflt = ast.unparse(item.value)
                except Exception:
                    dflt = "..."
                optional.append(f"{name}={dflt[:28]}")
    return required, optional


def extract_signatures(files: list[Path]) -> dict[str, tuple[list[str], list[str]]]:
    sigs: dict[str, tuple[list[str], list[str]]] = {}
    for f in files:
        if not f.exists():
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                req, opt = _class_signature(node)
                if req or opt:
                    sigs[node.name] = (req, opt)
    return sigs


def format_sig_block(sigs: dict, class_names: list[str]) -> str:
    parts: list[str] = []
    for cn in class_names:
        if cn not in sigs:
            continue
        req, opt = sigs[cn]
        req_s = ", ".join(req) if req else "(none)"
        opt_s = ", ".join(opt[:25]) + (" ..." if len(opt) > 25 else "")
        parts.append(f"class {cn}:\n  REQUIRED: {req_s}\n  OPTIONAL: {opt_s}")
    return "\n".join(parts) if parts else "(no constructor signatures matched this chunk)"


def gen_construct(
    engine: str, vslug: str, version_str: str, model: str, body: str, sigs: dict
) -> tuple[list[dict], float, list[str]]:
    eng_ns, samp_ns = PURE.L.NS[engine]
    chunks = W.chunk_validator_source(W.source_files_for(engine, vslug))
    invs: list[dict] = []
    raw: list[str] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        present = [cn for cn in sigs if re.search(r"\b" + re.escape(cn) + r"\b", chunk)]
        sig_block = format_sig_block(sigs, present)
        prompt = PURE.L.render_prompt(
            body,
            engine=engine,
            engine_version=version_str,
            field_namespace=eng_ns,
            sampling_namespace=samp_ns,
            class_signatures=sig_block,
            source=chunk,
        )
        try:
            resp = PURE.L.ollama_generate(model, prompt)
        except Exception as ex:
            raw.append(f"[CHUNK {ci} ERROR: {ex}]")
            continue
        raw.append(resp)
        invs.extend(PURE.L.parse_invariants(resp))
    return invs, time.time() - t0, raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--image", default=None)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    a = ap.parse_args()

    body = PURE.L.load_prompt_body(Path(a.prompt_file))
    sigs = extract_signatures(W.source_files_for(a.engine, a.vslug))
    invs, wall, raw = gen_construct(a.engine, a.vslug, a.version, a.model, body, sigs)
    Path(f"/tmp/phase1_w4c_{a.engine}_{a.vslug}_raw.txt").write_text(
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
        "shape": "construct_grounded",
        "model": a.model,
        "n_classes_extracted": len(sigs),
        "n_llm_proposed_raw": n_raw,
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
    }
    tag = a.model.replace(":", "_").replace("/", "_").replace(".", "_")
    Path(f"/tmp/phase1_w4c_{a.engine}_{a.vslug}_{tag}.json").write_text(
        json.dumps(result, indent=2)
    )
    # archive corpus for verification
    Path(f"/tmp/phase1_w4c_{a.engine}_{a.vslug}_{tag}_corpus.yaml").write_text(
        yaml.safe_dump({"invariants": deduped}, sort_keys=False)
    )
    print("PHASE1_W4C_RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
