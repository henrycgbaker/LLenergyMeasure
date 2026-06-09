"""Phase 1, Wave 3 - lenient re-parse for a tier whose raw output used a
non-standard root key. qwen2.5-coder:7b emitted its invariant list under `i:`
(not `invariants:`), so the strict parser (`parse_invariants`, which keys on
`invariants:`) dropped all 208 entries -> 0 proposals. gemma and the 32B used the
correct key, so this lenient pass only recovers content the strict parser WOULD
have kept had the key been right - it levels the field, it does not advantage the
7B. The format-following failure (wrong root key) is itself reported as a finding;
this pass measures the separate axis of extraction CAPABILITY.

Extracts the invariant list under ANY root key, filters to valid entries
(match.fields present), dedups vs the deterministic floor (same as wave1.py), and
writes a corpus YAML to feed `wave3_dump_confirmed.py` (gate + confirmed dump).

Usage: wave3_reparse_lenient.py --engine E --vslug V --raw RAW.txt --corpus-out C.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))

import wave2_llm_cells as L  # noqa: E402
import yaml  # noqa: E402


def _valid(inv: object) -> bool:
    return (
        isinstance(inv, dict)
        and isinstance(inv.get("match"), dict)
        and bool(inv["match"].get("fields"))
    )


def lenient_parse_chunks(raw: str) -> list[dict]:
    invs: list[dict] = []
    for chunk in raw.split("===CHUNK==="):
        t = L._strip_fences(chunk.strip())
        for attempt in (t, L._sanitize_yaml(t)):
            try:
                d = yaml.safe_load(attempt)
            except yaml.YAMLError:
                continue
            if not isinstance(d, dict):
                continue
            lst = d.get("invariants")
            if not isinstance(lst, list):
                # accept the invariant list under ANY root key (e.g. `i:`)
                for v in d.values():
                    if isinstance(v, list) and any(_valid(i) for i in v):
                        lst = v
                        break
            if isinstance(lst, list):
                good = [i for i in lst if _valid(i)]
                if good:
                    invs.extend(good)
                    break
    return invs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--vslug", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--corpus-out", required=True)
    a = ap.parse_args()

    invs = lenient_parse_chunks(Path(a.raw).read_text())
    print(f"lenient-parsed {len(invs)} valid invariants from {a.raw}", flush=True)

    floor_keys = {k for inv in L.floor_invariants(a.engine, a.vslug) if (k := L.tolerant_key(inv))}
    seen: set = set()
    out: list[dict] = []
    for inv in invs:
        k = L.tolerant_key(inv)
        if k is None or k in seen or k in floor_keys:
            continue
        seen.add(k)
        inv.setdefault("added_by", "llm_lenient_reparse")
        out.append(inv)
    print(f"deduped vs floor: {len(out)}", flush=True)

    Path(a.corpus_out).write_text(yaml.safe_dump({"invariants": out}, sort_keys=False))
    print(f"wrote corpus -> {a.corpus_out}", flush=True)


if __name__ == "__main__":
    main()
