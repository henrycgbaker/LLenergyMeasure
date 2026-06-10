"""Phase 1, Wave 5 (CROSS-BUMP) - gate-acceptance-rate degradation signal.

The prior bump-survivability finding (wave2_bump_survivability.md): when a bump
degrades carried knowledge, the deterministic output gives NO signal - it silently
emits fewer invariants. The named-but-never-built external signal: the runtime
gate's ACCEPTANCE RATE. This runner builds it.

Carry the OLD version's confirmed-GT catalogue (each entry already has a
kwargs_positive/negative probe) and re-gate it against the NEW version's container.
The acceptance-rate DROP = the degradation alarm (old invariants whose validator
moved/changed/was-removed now fail or infra_error against the new engine).

Usage: wave5_gate_acceptance.py --engine E --old-vslug V_OLD --new-vslug V_NEW
         --new-image IMG
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))

import study_gt_pilot as P  # noqa: E402
import yaml  # noqa: E402


def _old_catalogue(engine: str, old_vslug: str) -> list[dict]:
    """The OLD version's confirmed GT invariants as a gateable corpus (they carry
    the kwargs probes from when they were originally confirmed)."""
    p = P._FINDINGS / "study" / "ground_truth" / engine / old_vslug / "invariants" / "PILOT_GT.yaml"
    gt = yaml.safe_load(p.read_text()) or {}
    out: list[dict] = []
    for e in gt.get("confirmed") or []:
        inv = {
            "id": e.get("id"),
            "engine": engine,
            "invariant_under_test": e.get("invariant_under_test") or e.get("native_field") or "",
            "severity": e.get("severity", "error"),
            "native_type": e.get("native_type"),
            "match": e.get("match") or {"engine": engine, "fields": {}},
        }
        if e.get("kwargs_positive") is not None:
            inv["kwargs_positive"] = e["kwargs_positive"]
        if e.get("kwargs_negative") is not None:
            inv["kwargs_negative"] = e["kwargs_negative"]
        out.append(inv)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--old-vslug", required=True)
    ap.add_argument("--new-vslug", required=True)
    ap.add_argument("--new-image", default=None)
    a = ap.parse_args()

    old = _old_catalogue(a.engine, a.old_vslug)
    tmp = Path(f"/tmp/wave5_oldcat_{a.engine}_{a.old_vslug}.yaml")
    tmp.write_text(yaml.safe_dump({"invariants": old}, sort_keys=False))

    # gate the OLD catalogue against the NEW version's container
    P.configure(a.engine, a.new_vslug, a.new_image)
    P.SOURCES = {"old": (tmp, False)}
    cands = P.load_candidates()
    P.gate(cands)

    gateable = [c for c in cands if c.gateable]
    verdicts = {
        v: sum(1 for c in cands if c.verdict == v)
        for v in ("confirmed", "failed", "skipped", "infra_error", "ungated")
    }
    n_conf = verdicts["confirmed"]
    acceptance = n_conf / len(gateable) if gateable else 0.0
    # the "broke" set: old invariants that no longer hold against the new version
    broke = [
        {"id": c.orig_id, "verdict": c.verdict, "native_type": c.inv.get("native_type")}
        for c in cands
        if c.gateable and c.verdict in ("failed", "infra_error")
    ]

    result = {
        "engine": a.engine,
        "old_vslug": a.old_vslug,
        "new_vslug": a.new_vslug,
        "n_old_catalogue": len(old),
        "n_gateable": len(gateable),
        "verdicts": verdicts,
        "acceptance_rate_vs_new": round(acceptance, 3),
        "n_broke": len(broke),
        "broke_sample": broke[:25],
    }
    out = Path(f"/tmp/wave5_gateacc_{a.engine}_{a.old_vslug}_to_{a.new_vslug}.json")
    out.write_text(json.dumps(result, indent=2))
    print(
        "WAVE5_GATEACC_RESULT "
        + json.dumps({k: v for k, v in result.items() if k != "broke_sample"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
