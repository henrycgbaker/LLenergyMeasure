"""Phase 1, Wave 3 - re-gate an archived proposal corpus and dump the CONFIRMED
candidates with full detail (match, kwargs, observed outcome, raised error,
full invariant) for the MANDATORY adversarial source-verification.

The wave-3 runner persists only verdict COUNTS + ckeys, not per-candidate errors.
The gate is deterministic (same kwargs -> same construction -> same raise), so we
re-gate the archived corpus to recover each confirmed candidate's raised error -
the exact thing the adversarial reviewer must match to the labelled rule.

A cross_field heuristic is emitted, but the FULL inv (match + predicate) is dumped
so the reviewer classifies cross-field itself rather than trusting the heuristic
(a relation can hide in a single-key match whose predicate references another
field, e.g. max_cpu_loras vs max_loras).

Usage: wave3_dump_confirmed.py --engine E --vslug V --corpus C.yaml --out O.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))

import study_gt_pilot as P  # noqa: E402
import yaml  # noqa: E402


def _looks_cross_field(inv: dict) -> bool:
    """Heuristic: multi-key match, OR a predicate value that references another
    field path (contains a '.' namespace or names a second leaf)."""
    match = inv.get("match") or {}
    if len(match) > 1:
        return True
    blob = (
        str(inv.get("ckey", ""))
        + str(inv.get("match", ""))
        + str(inv.get("invariant_under_test", ""))
    )
    return "cross" in blob.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    P.configure(a.engine, a.vslug, None)
    P.SOURCES = {"llm": (Path(a.corpus), False)}
    cands = P.load_candidates()
    P.gate(cands)

    confirmed = [c for c in cands if c.verdict == "confirmed"]
    out = []
    for c in confirmed:
        inv = c.inv
        out.append(
            {
                "id": c.orig_id,
                "ckey": list(c.ckey),
                "cross_field_heuristic": _looks_cross_field(inv),
                "n_match_fields": len(inv.get("match") or {}),
                "invariant_under_test": inv.get("invariant_under_test") or inv.get("description"),
                "native_type": inv.get("native_type"),
                "match": inv.get("match"),
                "kwargs_positive": inv.get("kwargs_positive"),
                "kwargs_negative": inv.get("kwargs_negative"),
                "observed_outcome": c.observed_outcome,
                "error": c.error,
            }
        )
    cf = [e for e in out if e["cross_field_heuristic"]]
    Path(a.out).write_text(yaml.safe_dump({"confirmed": out}, sort_keys=False))
    print(
        f"DUMP {a.engine}/{a.vslug}: {len(confirmed)} confirmed, "
        f"{len(cf)} cross-field(heuristic) -> {a.out}",
        flush=True,
    )
    for e in cf:
        print(f"  CF[{e['n_match_fields']}f] {e['id']} :: {e['invariant_under_test']}", flush=True)


if __name__ == "__main__":
    main()
