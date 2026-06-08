"""Round 0b NON-DESTRUCTIVE gate.

Gate the new improved-det-v2 mech candidates for one cell and compare the
gate-confirmed constraints vs the committed Round-0 PILOT_GT. Does NOT overwrite
the committed GT (unlike study_gt_pilot). Writes /tmp/round0b_gate_<e>_<v>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "research/mining-substrate-trial/scripts")
import study_gt_pilot as P
import yaml
from strategies.wave2 import a_improved_det_v2 as v2


def run(engine: str, vslug: str, version: str, src: str | None, image: str | None) -> None:
    # 1. Refresh the mech (improved-det-v2) output with the Round 0b primitives.
    root = Path(src) if src else None
    v2.run_cell(engine=engine, version=version, engine_source_root=root, task="invariants")

    # 2. Configure the gate for this cell; restrict to the mech source only.
    P.configure(engine, vslug, image)
    if "mech" not in P.SOURCES:
        print(f"[{engine}/{vslug}] NO mech source discovered; aborting", flush=True)
        return
    P.SOURCES = {"mech": P.SOURCES["mech"]}

    cands = P.load_candidates()
    print(f"[{engine}/{vslug}] gating {len(cands)} mech candidates in-container...", flush=True)
    P.gate(cands)
    groups = P.build_groups(cands)
    confirmed_ckeys = {k for k, g in groups.items() if g.status == "confirmed"}

    # 3. Compare vs committed Round-0 confirmed constraints (NON-destructive).
    gtp = P._OUT_DIR / "PILOT_GT.yaml"
    r0 = yaml.safe_load(gtp.read_text()) if gtp.exists() else {}
    r0_confirmed = {tuple(e["constraint_key"]) for e in (r0.get("confirmed") or [])}
    new_conf = confirmed_ckeys - r0_confirmed

    def _is_plugin(c):
        return "plugin/plugin.py" in (c.inv.get("miner_source") or {}).get("path", "")

    plugin = [c for c in cands if _is_plugin(c)]
    plat = [c for c in cands if c.inv.get("fired_by") == "p10"]
    result = {
        "engine": engine,
        "vslug": vslug,
        "mech_candidates": len(cands),
        "mech_confirmed_ckeys": len(confirmed_ckeys),
        "r0_confirmed_ckeys": len(r0_confirmed),
        "new_confirmed_vs_r0_count": len(new_conf),
        "new_confirmed_vs_r0": sorted("|".join(map(str, k)) for k in new_conf),
        "plugin_total": len(plugin),
        "plugin_confirmed": sum(1 for c in plugin if c.verdict == "confirmed"),
        "plugin_verdicts": {
            v: sum(1 for c in plugin if c.verdict == v)
            for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
        },
        "platform_total": len(plat),
        "platform_confirmed": sum(1 for c in plat if c.verdict == "confirmed"),
        "platform_verdicts": {
            v: sum(1 for c in plat if c.verdict == v)
            for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
        },
        "all_verdicts": {
            v: sum(1 for c in cands if c.verdict == v)
            for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
        },
    }
    outp = Path(f"/tmp/round0b_gate_{engine}_{vslug}.json")
    outp.write_text(json.dumps(result, indent=2))
    print("ROUND0B_GATE_RESULT " + json.dumps(result), flush=True)
    print(f"[{engine}/{vslug}] wrote {outp}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--src", default=None)
    ap.add_argument("--image", default=None)
    a = ap.parse_args()
    run(a.engine, a.vslug, a.version, a.src, a.image)
