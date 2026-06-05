"""Wave 2.6 substrate analysis: complementarity + frontier inputs.

For each (engine, version) cell, compute which GT invariant keys (tolerant
(leaf_field, coarse_bucket) identities) each static substrate catches, then the
intersection / union / disjoint breakdown across substrates. Answers the
"do substrates catch disjoint entries (union >> max)?" question for the
complementarity deliverable, and emits per-substrate recall/cost rows for the
frontier deliverable.

Pure-static substrates only (tree-sitter, improved-det); GPU-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gt_scoring as G  # noqa: E402
import yaml  # noqa: E402

_FINDINGS = _SCRIPTS.parent / "findings"
_RUNS = _FINDINGS / "trial_runs" / "wave2"

CELLS = [
    ("transformers", "v4_57_3"),
    ("transformers", "v5_6_2"),
    ("vllm", "v0_7_3"),
    ("vllm", "v0_19_1"),
    ("tensorrt", "v0_21_0"),
    ("tensorrt", "v1_2_1"),
]
SUBSTRATES = ["w2-a-treesitter", "w2-a-improved-det"]


def _cell_inv_keys(strategy: str, engine: str, slug: str) -> set[tuple[str, str]] | None:
    p = _RUNS / strategy / engine / slug / "invariants.proposed.yaml"
    if not p.exists():
        return None
    data = yaml.safe_load(p.read_text()) or {}
    return G.tolerant_invariant_keys(data)


def main() -> int:
    out: list[dict] = []
    for engine, slug in CELLS:
        _cs, ci = G.canonicalise_gt_dir(engine, slug)
        gt_keys = G.tolerant_invariant_keys(yaml.safe_load(ci.read_text()) or {})
        caught: dict[str, set] = {}
        for strat in SUBSTRATES:
            keys = _cell_inv_keys(strat, engine, slug)
            if keys is None:
                continue
            caught[strat] = keys & gt_keys
        if not caught:
            continue
        row: dict = {"engine": engine, "version_slug": slug, "gt_inv_keys": len(gt_keys)}
        for strat, kset in caught.items():
            row[f"{strat}__caught"] = len(kset)
            row[f"{strat}__recall"] = round(len(kset) / len(gt_keys), 4) if gt_keys else 0.0
        if len(caught) == 2:
            a, b = SUBSTRATES
            ka, kb = caught.get(a, set()), caught.get(b, set())
            union = ka | kb
            inter = ka & kb
            row["union_caught"] = len(union)
            row["union_recall"] = round(len(union) / len(gt_keys), 4) if gt_keys else 0.0
            row["max_single_recall"] = (
                round(max(len(ka), len(kb)) / len(gt_keys), 4) if gt_keys else 0.0
            )
            row["complementarity_gain"] = (
                round((len(union) - max(len(ka), len(kb))) / len(gt_keys), 4) if gt_keys else 0.0
            )
            row["treesitter_only"] = len(ka - kb)
            row["improveddet_only"] = len(kb - ka)
            row["both"] = len(inter)
        out.append(row)
        print(json.dumps(row))

    dst = _FINDINGS / "wave2_substrate_analysis.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
