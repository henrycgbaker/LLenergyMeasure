"""Drive the Wave 2 LLM cells in priority order, saving incrementally to
findings/wave2_llm_cells.json. Resumable-ish: each completed record is
appended and the file rewritten after every cell.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import wave2_llm_cells as M

OUT = M.FINDINGS / "wave2_llm_cells.json"
QWEN = "qwen2.5-coder:7b-instruct-fp16"
LLAMA = "llama3.1:8b-instruct-fp16"
PHI = "phi4:14b-fp16"

# priority cells (engine, version)
CELLS = [
    ("vllm", "v0_7_3"),
    ("transformers", "v4_57_3"),
    ("vllm", "v0_19_1"),
    ("transformers", "v5_6_2"),
    ("tensorrt", "v0_21_0"),
]


def load() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"floor_only": [], "wg_extend": [], "pure_b": [], "model_scale": []}


def save(state: dict) -> None:
    OUT.write_text(json.dumps(state, indent=2))


def have(state: dict, bucket: str, engine: str, version: str, model: str | None = None) -> bool:
    for r in state[bucket]:
        if r.get("engine") == engine and r.get("version_slug") == version:
            if model is None or r.get("model") == model:
                return True
    return False


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "wg"
    state = load()

    # floor-only baselines (cheap, no LLM) - always refresh once
    if not state["floor_only"]:
        for e, v in CELLS:
            if (M.FLOOR_ROOT / e / v / "invariants.proposed.yaml").exists():
                state["floor_only"].append(M.floor_only_record(e, v))
        save(state)
        print(f"[floor] {len(state['floor_only'])} baselines recorded")

    if mode in ("wg", "all"):
        for e, v in CELLS:
            if have(state, "wg_extend", e, v, QWEN):
                print(f"[wg] skip {e}/{v} (done)")
                continue
            print(f"[wg] {e}/{v} {QWEN} ...", flush=True)
            t0 = time.time()
            try:
                rec = M.run_extend_cell(e, v, QWEN, pure_b=False, run_label="w2-wg-qwen7b")
                state["wg_extend"].append(rec)
                save(state)
                print(
                    f"[wg] {e}/{v} done in {rec['wall_sec']}s "
                    f"tol_inv_recall={rec['tolerant']['inv_recall']} "
                    f"(+{rec['n_llm_proposed_deduped']} llm)",
                    flush=True,
                )
            except Exception as ex:
                state["wg_extend"].append(
                    {
                        "engine": e,
                        "version_slug": v,
                        "model": QWEN,
                        "error": str(ex),
                        "wall_sec": round(time.time() - t0, 1),
                    }
                )
                save(state)
                print(f"[wg] {e}/{v} ERROR: {ex}", flush=True)

    if mode in ("pureb", "all"):
        for e, v in CELLS:
            if have(state, "pure_b", e, v, QWEN):
                print(f"[pureb] skip {e}/{v} (done)")
                continue
            print(f"[pureb] {e}/{v} {QWEN} ...", flush=True)
            t0 = time.time()
            try:
                rec = M.run_extend_cell(e, v, QWEN, pure_b=True, run_label="w2-pureb-qwen7b")
                state["pure_b"].append(rec)
                save(state)
                print(
                    f"[pureb] {e}/{v} done in {rec['wall_sec']}s "
                    f"tol_inv_recall={rec['tolerant']['inv_recall']} "
                    f"({rec['n_llm_proposed_deduped']} llm)",
                    flush=True,
                )
            except Exception as ex:
                state["pure_b"].append(
                    {
                        "engine": e,
                        "version_slug": v,
                        "model": QWEN,
                        "error": str(ex),
                        "wall_sec": round(time.time() - t0, 1),
                    }
                )
                save(state)
                print(f"[pureb] {e}/{v} ERROR: {ex}", flush=True)

    if mode in ("scale", "all"):
        # model-scale sweep: W-G extend on ONE cell with llama3.1-8b + phi4-14b
        e, v = "vllm", "v0_7_3"
        for model, label in ((LLAMA, "w2-wg-llama8b"), (PHI, "w2-wg-phi14b")):
            if have(state, "model_scale", e, v, model):
                print(f"[scale] skip {e}/{v} {model} (done)")
                continue
            print(f"[scale] {e}/{v} {model} ...", flush=True)
            t0 = time.time()
            try:
                rec = M.run_extend_cell(e, v, model, pure_b=False, run_label=label)
                state["model_scale"].append(rec)
                save(state)
                print(
                    f"[scale] {e}/{v} {model} done in {rec['wall_sec']}s "
                    f"tol_inv_recall={rec['tolerant']['inv_recall']} "
                    f"(+{rec['n_llm_proposed_deduped']} llm)",
                    flush=True,
                )
            except Exception as ex:
                state["model_scale"].append(
                    {
                        "engine": e,
                        "version_slug": v,
                        "model": model,
                        "error": str(ex),
                        "wall_sec": round(time.time() - t0, 1),
                    }
                )
                save(state)
                print(f"[scale] {e}/{v} {model} ERROR: {ex}", flush=True)

    print("DONE", mode)


if __name__ == "__main__":
    main()
