"""Wave 2 LLM-involving cells: W-G hybrid (det floor + small-LLM extend),
pure-b (LLM-only extract), and a model-scale sweep (7b/8b/14b).

Minimal DIRECT Ollama dispatch (the registered w2-h* strategies are stubs;
we do NOT touch them). Runs cells SEQUENTIALLY - models are single-tenant on
one A100. Scores every produced catalogue against GT via gt_scoring.

Per cell records: tolerant + strict inv recall/precision vs GT (floor-only,
+LLM, delta), wall-sec, model tag + digest, count of LLM-proposed entries,
and a hallucination PROXY = (LLM entries whose (leaf,bucket) not in GT) /
(total LLM entries). The proxy is NOT a true runtime-validate gate.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gt_scoring as G
import trial_scoring as S
import wave2_llm_source as W
import yaml

OLLAMA = "http://localhost:11435"
SCRIPTS = Path(__file__).resolve().parent
FINDINGS = SCRIPTS.parent / "findings"
FLOOR_ROOT = FINDINGS / "trial_runs" / "wave2" / "w2-a-improved-det"
RUN_ROOT = FINDINGS / "trial_runs" / "wave2"
PROMPTS = FINDINGS / "wave2_locked_prompts"

MODEL_DIGEST: dict[str, str] = {}

# namespace conventions, matching the improved-det floor per engine
NS = {
    "transformers": ("transformers", "transformers.sampling"),
    "vllm": ("vllm.engine", "vllm.sampling"),
    "tensorrt": ("tensorrt", "tensorrt.sampling"),
}

ENGINE_VERSION = {
    "transformers": {"v4_57_3": "4.57.3", "v5_6_2": "5.6.2"},
    "vllm": {"v0_7_3": "0.7.3", "v0_19_1": "0.19.1"},
    "tensorrt": {"v0_21_0": "0.21.0"},
}


def _digest(model: str) -> str:
    if not MODEL_DIGEST:
        with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=30) as r:
            for m in json.load(r)["models"]:
                MODEL_DIGEST[m["name"]] = m["digest"][:16]
    return MODEL_DIGEST.get(model, "?")


def ollama_generate(
    model: str,
    prompt: str,
    *,
    num_ctx: int = 16384,
    num_predict: int = 4096,
    temperature: float = 0,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
                "num_predict": num_predict,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["response"]


def _strip_fences(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\n", "", t)
    t = re.sub(r"\n```\s*$", "", t)
    # find first 'invariants:' if model added preamble
    idx = t.find("invariants:")
    if idx > 0:
        t = t[idx:]
    return t


# scalar keys whose values often contain unquoted ':' (breaks YAML). We
# wrap their values in single quotes if not already quoted.
_RISKY_KEYS = ("invariant_under_test", "path", "message_template", "native_type")
_KEY_RE = re.compile(r"^(\s*)(" + "|".join(_RISKY_KEYS) + r"):\s*(.*)$")


def _sanitize_yaml(t: str) -> str:
    out = []
    for line in t.splitlines():
        m = _KEY_RE.match(line)
        if m:
            indent, key, val = m.group(1), m.group(2), m.group(3).rstrip()
            if val and val[0] not in "'\"[{" and val not in ("|", ">"):
                val = "'" + val.replace("'", "''") + "'"
            out.append(f"{indent}{key}: {val}")
        else:
            out.append(line)
    return "\n".join(out)


def _parse_per_entry(t: str) -> list[dict]:
    """Tolerant fallback: split the invariants list on top-level ``- id:``
    boundaries and parse each entry independently (sanitised), discarding any
    that fail (e.g. the final truncated entry when generation hit num_predict).
    """
    lines = t.splitlines()
    # find the indentation of list items under invariants:
    body = []
    started = False
    for ln in lines:
        if ln.strip().startswith("invariants:"):
            started = True
            continue
        if started:
            body.append(ln)
    # split into entries on lines beginning '- ' (at the list indent)
    entries: list[list[str]] = []
    cur: list[str] = []
    for ln in body:
        if re.match(r"^\s*-\s+\S", ln):
            if cur:
                entries.append(cur)
            cur = [ln]
        elif cur:
            cur.append(ln)
    if cur:
        entries.append(cur)

    out: list[dict] = []
    for ent in entries:
        # parse each entry as a single-item list, preserving the '- ' so YAML
        # indentation stays consistent. Sanitise risky scalar values first.
        block = _sanitize_yaml("\n".join(ent))
        try:
            d = yaml.safe_load(block)
        except yaml.YAMLError:
            continue
        if isinstance(d, list) and d and isinstance(d[0], dict):
            item = d[0]
            m = item.get("match")
            if isinstance(m, dict) and m.get("fields"):
                out.append(item)
    return out


def parse_invariants(text: str) -> list[dict]:
    t = _strip_fences(text)
    for attempt in (t, _sanitize_yaml(t)):
        try:
            d = yaml.safe_load(attempt)
        except yaml.YAMLError:
            continue
        if not isinstance(d, dict):
            continue
        invs = d.get("invariants")
        if not isinstance(invs, list):
            continue
        good = [
            i
            for i in invs
            if isinstance(i, dict) and isinstance(i.get("match"), dict) and i["match"].get("fields")
        ]
        if good:
            return good
    # both whole-document attempts failed (often truncation) -> per-entry
    return _parse_per_entry(t)


def floor_invariants(engine: str, version: str) -> list[dict]:
    p = FLOOR_ROOT / engine / version / "invariants.proposed.yaml"
    d = yaml.safe_load(p.read_text())
    return d.get("invariants", [])


def floor_schema_path(engine: str, version: str) -> Path | None:
    p = FLOOR_ROOT / engine / version / "schema.json"
    return p if p.exists() else None


def tolerant_key(inv: dict) -> tuple[str, str] | None:
    """(leaf_field, coarse_bucket) for an invariant - same identity the
    tolerant scorer uses."""
    try:
        identity = S.invariant_identity(inv)
        _, leaf, pk, _sec = identity
        return (str(leaf), G._coarse_from_scorer_pk(pk))
    except Exception:
        return None


def write_envelope(invs: list[dict], engine: str, version: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    env = {
        "schema_version": "1.0.0",
        "engine": engine,
        "engine_version": ENGINE_VERSION[engine][version],
        "invariants": invs,
    }
    out.write_text(yaml.safe_dump(env, sort_keys=False))


def gt_tolerant_keys(engine: str, version: str) -> set[tuple[str, str]]:
    cs, ci = G.canonicalise_gt_dir(engine, version)
    canon_inv = yaml.safe_load(ci.read_text()) or {}
    return G.tolerant_invariant_keys(canon_inv)


def render_prompt(template: str, **kw) -> str:
    # the locked prompt bodies use {placeholder}; we substitute the few we set
    # and leave YAML braces alone by only replacing known keys.
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


def load_prompt_body(md_path: Path) -> str:
    text = md_path.read_text()
    # the prompt body lives in the first ```text ... ``` fenced block
    m = re.search(r"```text\n(.*?)\n```", text, re.DOTALL)
    if not m:
        raise ValueError(f"no prompt body in {md_path}")
    return m.group(1)


WG_BODY = load_prompt_body(PROMPTS / "wg_extend_prompt.md")
PUREB_BODY = load_prompt_body(PROMPTS / "pure_b_prompt.md")


def floor_summary_for_prompt(floor: list[dict]) -> str:
    """Compact rendering of the floor catalogue for INPUT 1 (id + field +
    predicate only - keeps token cost down vs full YAML)."""
    lines = []
    for inv in floor:
        m = inv.get("match")
        fields = m.get("fields", {}) if isinstance(m, dict) else {}
        for fk, fv in fields.items():
            lines.append(f"- {inv.get('id', '?')}: {fk} -> {fv}")
    return "\n".join(lines)


def run_extend_cell(
    engine: str,
    version: str,
    model: str,
    *,
    pure_b: bool,
    run_label: str,
) -> dict:
    version_str = ENGINE_VERSION[engine][version]
    eng_ns, samp_ns = NS[engine]
    files = W.source_files_for(engine, version)
    chunks = W.chunk_validator_source(files)
    floor = floor_invariants(engine, version)
    floor_blob = floor_summary_for_prompt(floor)

    raw_responses = []
    llm_invs: list[dict] = []
    t0 = time.time()
    for ci, chunk in enumerate(chunks):
        if pure_b:
            prompt = render_prompt(
                PUREB_BODY,
                engine=engine,
                engine_version=version_str,
                field_namespace=eng_ns,
                sampling_namespace=samp_ns,
                source=chunk,
            )
        else:
            prompt = render_prompt(
                WG_BODY,
                engine=engine,
                engine_version=version_str,
                field_namespace=eng_ns,
                sampling_namespace=samp_ns,
                floor_invariants=floor_blob,
                source=chunk,
            )
        try:
            resp = ollama_generate(model, prompt)
        except Exception as ex:
            raw_responses.append(f"[CHUNK {ci} ERROR: {ex}]")
            continue
        raw_responses.append(resp)
        llm_invs.extend(parse_invariants(resp))
    wall = time.time() - t0

    # dedup LLM invs by tolerant key
    seen: set[tuple[str, str]] = set()
    floor_keys = {k for inv in floor if (k := tolerant_key(inv))}
    deduped_llm: list[dict] = []
    for inv in llm_invs:
        k = tolerant_key(inv)
        if k is None:
            continue
        if k in seen:
            continue
        seen.add(k)
        if not pure_b and k in floor_keys:
            continue  # drop floor-duplicate for W-G
        inv.setdefault("added_by", "llm_pure_b" if pure_b else "llm_wg_extend")
        deduped_llm.append(inv)

    if pure_b:
        merged = deduped_llm
    else:
        merged = list(floor) + deduped_llm

    out_dir = RUN_ROOT / run_label / engine / version
    inv_out = out_dir / "invariants.proposed.yaml"
    write_envelope(merged, engine, version, inv_out)
    # copy floor schema so schema scoring is meaningful (LLM does invariants)
    sch_path = floor_schema_path(engine, version)
    sch_out = None
    if sch_path is not None:
        sch_out = out_dir / "schema.json"
        sch_out.write_text(sch_path.read_text())
    (out_dir / "raw_llm_responses.txt").write_text("\n\n===CHUNK===\n\n".join(raw_responses))

    # score merged vs GT
    res = G.score_cell_vs_gt(
        strategy=run_label,
        engine=engine,
        version_slug=version,
        bump_distance="active",
        schema_output_path=sch_out,
        invariants_output_path=inv_out,
        wall_clock_sec=wall,
    )

    # hallucination proxy: LLM-proposed keys not in GT / total LLM-proposed
    gt_keys = gt_tolerant_keys(engine, version)
    llm_keys = [tolerant_key(i) for i in deduped_llm]
    llm_keys = [k for k in llm_keys if k]
    halluc = sum(1 for k in llm_keys if k not in gt_keys)
    halluc_rate = halluc / len(llm_keys) if llm_keys else 0.0

    return {
        "engine": engine,
        "version_slug": version,
        "run_label": run_label,
        "model": model,
        "model_digest": _digest(model),
        "mode": "pure_b" if pure_b else "wg_extend",
        "n_chunks": len(chunks),
        "n_llm_proposed_raw": len(llm_invs),
        "n_llm_proposed_deduped": len(deduped_llm),
        "n_merged": len(merged),
        "wall_sec": round(wall, 1),
        "tolerant": {
            "inv_recall": round(res.tolerant.invariant_recall, 4),
            "inv_precision": round(res.tolerant.invariant_precision, 4),
            "inv_ref_n": res.tolerant.invariant_ref_n,
            "inv_cell_n": res.tolerant.invariant_cell_n,
            "inv_int_n": res.tolerant.invariant_int_n,
            "schema_recall": round(res.tolerant.schema_recall, 4),
            "schema_precision": round(res.tolerant.schema_precision, 4),
        },
        "strict": {
            "inv_recall": round(res.strict.invariant_recall, 4),
            "inv_precision": round(res.strict.invariant_precision, 4),
        },
        "hallucination_proxy": {
            "n_llm_keys": len(llm_keys),
            "n_not_in_gt": halluc,
            "rate": round(halluc_rate, 4),
            "note": "PROXY: (leaf,bucket) not in GT / total LLM keys. NOT a runtime-validate gate.",
        },
        "out_inv_path": str(inv_out),
    }


def floor_only_record(engine: str, version: str) -> dict:
    sch = floor_schema_path(engine, version)
    inv = FLOOR_ROOT / engine / version / "invariants.proposed.yaml"
    res = G.score_cell_vs_gt(
        strategy="w2-a-improved-det",
        engine=engine,
        version_slug=version,
        bump_distance="active",
        schema_output_path=sch,
        invariants_output_path=inv,
    )
    return {
        "engine": engine,
        "version_slug": version,
        "run_label": "floor_only",
        "tolerant": {
            "inv_recall": round(res.tolerant.invariant_recall, 4),
            "inv_precision": round(res.tolerant.invariant_precision, 4),
            "inv_ref_n": res.tolerant.invariant_ref_n,
            "inv_cell_n": res.tolerant.invariant_cell_n,
            "inv_int_n": res.tolerant.invariant_int_n,
        },
        "strict": {
            "inv_recall": round(res.strict.invariant_recall, 4),
            "inv_precision": round(res.strict.invariant_precision, 4),
        },
    }
