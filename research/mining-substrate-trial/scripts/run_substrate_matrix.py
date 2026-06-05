"""Wave 2.1 substrate matrix driver.

Runs each deterministic substrate across every (engine, version) cell that has
ground truth, generates the cell's mined catalogue via ``wave2_runner.run_one``,
then scores it against the Opus GT via ``gt_scoring.score_cell_vs_gt`` (strict +
tolerant). Aggregates to ``findings/wave2_substrate_matrix.json`` and prints a
table.

GPU-free substrates only (tree-sitter, improved-det are pure-static; pydantic-
native is attempted but imports the engine at runtime so it reflects the
project-venv version or fails without a per-version container - recorded as
such).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import gt_scoring as G  # noqa: E402
import wave2_runner as R  # noqa: E402

_FINDINGS = _SCRIPTS.parent / "findings"
_RUNS = _FINDINGS / "trial_runs" / "wave2"

# (engine, cli_version, version_slug, bump_distance)
CELLS = [
    ("transformers", "v4.57.3", "v4_57_3", "active"),
    ("transformers", "v5.6.2", "v5_6_2", "v+major"),
    ("vllm", "v0.7.3", "v0_7_3", "active"),
    ("vllm", "v0.19.1", "v0_19_1", "v+major"),
    ("tensorrt", "v0.21.0", "v0_21_0", "active"),
    ("tensorrt", "v1.2.1", "v1_2_1", "v+major"),
]

SUBSTRATES = ["w2-a-treesitter", "w2-a-improved-det", "w2-a-pydantic-native"]


def _cell_output_paths(strategy: str, engine: str, version_slug: str) -> tuple[Path, Path]:
    base = _RUNS / strategy / engine / version_slug
    return base / "schema.json", base / "invariants.proposed.yaml"


def main() -> int:
    results: list[dict] = []
    for strategy in SUBSTRATES:
        entry = R.WAVE2_REGISTRY.get(strategy)
        for engine, cli_v, slug, bump in CELLS:
            if entry is not None and engine not in entry.engines:
                results.append(
                    {
                        "strategy": strategy,
                        "engine": engine,
                        "version_slug": slug,
                        "status": "unsupported_engine",
                    }
                )
                continue
            row: dict = {
                "strategy": strategy,
                "engine": engine,
                "version_slug": slug,
                "bump_distance": bump,
            }
            try:
                run = R.run_one(strategy_id=strategy, engine=engine, version=cli_v)
                row["gen_status"] = run.get("status")
            except Exception as exc:
                row["gen_status"] = "driver_exception"
                row["error"] = f"{type(exc).__name__}: {exc}"
                results.append(row)
                print(f"  {strategy:22} {engine:12} {slug:9} GEN-FAIL {row['error'][:60]}")
                continue

            schema_p, inv_p = _cell_output_paths(strategy, engine, slug)
            if not schema_p.exists() and not inv_p.exists():
                row["status"] = "no_output"
                results.append(row)
                print(f"  {strategy:22} {engine:12} {slug:9} NO-OUTPUT ({row['gen_status']})")
                continue
            try:
                res = G.score_cell_vs_gt(
                    strategy=strategy,
                    engine=engine,
                    version_slug=slug,
                    bump_distance=bump,
                    schema_output_path=schema_p if schema_p.exists() else None,
                    invariants_output_path=inv_p if inv_p.exists() else None,
                )
            except Exception as exc:
                row["status"] = "score_exception"
                row["error"] = f"{type(exc).__name__}: {exc}"
                results.append(row)
                print(f"  {strategy:22} {engine:12} {slug:9} SCORE-FAIL {row['error'][:60]}")
                continue

            s, t = res.strict, res.tolerant
            row.update(
                {
                    "status": "scored",
                    "strict_schema_recall": round(s.schema_recall, 4),
                    "strict_schema_precision": round(s.schema_precision, 4),
                    "strict_inv_recall": round(s.invariant_recall, 4),
                    "strict_inv_precision": round(s.invariant_precision, 4),
                    "tol_schema_recall": round(t.schema_recall, 4),
                    "tol_schema_precision": round(t.schema_precision, 4),
                    "tol_inv_recall": round(t.invariant_recall, 4),
                    "tol_inv_precision": round(t.invariant_precision, 4),
                    "tol_schema_ref_n": t.schema_ref_n,
                    "tol_inv_ref_n": t.invariant_ref_n,
                    "tol_schema_cell_n": t.schema_cell_n,
                    "tol_inv_cell_n": t.invariant_cell_n,
                }
            )
            results.append(row)
            print(
                f"  {strategy:22} {engine:12} {slug:9} "
                f"sch r/p(tol)={t.schema_recall:.3f}/{t.schema_precision:.3f} "
                f"inv r/p(tol)={t.invariant_recall:.3f}/{t.invariant_precision:.3f} "
                f"[strict inv r={s.invariant_recall:.3f}]"
            )

    out = _FINDINGS / "wave2_substrate_matrix.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out} ({len(results)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
