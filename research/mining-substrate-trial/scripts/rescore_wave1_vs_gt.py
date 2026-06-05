"""Driver: re-score experiment-queue step 0.4 - Wave 1 cells vs canonical GT.

For every Wave 1 cell whose ``(engine, version_slug)`` has an Opus-established
ground truth, locate the cell's RAW mined outputs (schema + invariants
artefacts, NOT the score JSONs) and re-score them against the canonical GT
using ``gt_scoring.score_cell_vs_gt`` (strict + tolerant).

Output: ``findings/wave1_rescored_against_gt.md`` - per cell, the original
validated-union recall/precision (from ``findings/trial_scores/*__vu.json``)
SIDE BY SIDE with the new-vs-GT recall/precision (strict and tolerant).
Cells whose raw output is absent are listed as
"not re-scoreable (raw output absent)" rather than dropped.

Raw-output locations
--------------------
* strategy ``a`` (pure mining): canonical pipeline outputs at
  ``engine_versions/<e>/<v>/outputs/{schema.discovered.json,
  invariants.proposed.yaml}``.
* strategies ``b`` / ``b_8b`` / ``d-ab`` (LLM / hybrid): parallel artefacts
  at ``findings/trial_runs/<strategy>/<e>/<v>/{schema.json,
  invariants.proposed.yaml}``.
"""

from __future__ import annotations

import json
from pathlib import Path

import gt_scoring as G

_SCRIPTS_DIR = Path(__file__).resolve().parent
_TRIAL_ROOT = _SCRIPTS_DIR.parent
_FINDINGS = _TRIAL_ROOT / "findings"
_REPO_ROOT = _TRIAL_ROOT.parent.parent
_TRIAL_SCORES = _FINDINGS / "trial_scores"
_TRIAL_RUNS = _FINDINGS / "trial_runs"
_ENGINE_VERSIONS = _REPO_ROOT / "engine_versions"

# Engines x versions for which a GT exists.
_GT_PAIRS = {
    "transformers": {"v4_57_3", "v5_6_2"},
    "vllm": {"v0_7_3", "v0_19_1"},
    "tensorrt": {"v0_21_0", "v1_2_1"},
}


def _raw_output_paths(strategy: str, engine: str, version: str) -> tuple[Path | None, Path | None]:
    """Return ``(schema_path, invariants_path)`` for a cell's raw outputs,
    or ``(None, None)`` slots that don't exist on disk."""
    if strategy == "a":
        sd = _ENGINE_VERSIONS / engine / version / "outputs" / "schema.discovered.json"
        ip = _ENGINE_VERSIONS / engine / version / "outputs" / "invariants.proposed.yaml"
    else:
        cell = _TRIAL_RUNS / strategy / engine / version
        sd = cell / "schema.json"
        ip = cell / "invariants.proposed.yaml"
    return (sd if sd.exists() else None, ip if ip.exists() else None)


def _original_vu(strategy: str, engine: str, version: str) -> dict | None:
    """Load the existing validated-union score JSON for a cell, if present."""
    p = _TRIAL_SCORES / f"{strategy}__{engine}__{version}__vu.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _enumerate_wave1_cells() -> list[tuple[str, str, str]]:
    """Enumerate Wave 1 ``(strategy, engine, version)`` cells from the
    existing ``*__vu.json`` files, restricted to engines x versions that
    have a GT. Wave 2 substrate cells (``wave2/...``) are not in this set."""
    cells: list[tuple[str, str, str]] = []
    for p in sorted(_TRIAL_SCORES.glob("*__vu.json")):
        stem = p.name[: -len("__vu.json")]
        parts = stem.split("__")
        if len(parts) != 3:
            continue
        strategy, engine, version = parts
        if engine in _GT_PAIRS and version in _GT_PAIRS[engine]:
            cells.append((strategy, engine, version))
    return cells


def main() -> None:
    cells = _enumerate_wave1_cells()
    rows: list[dict] = []
    rescored = 0
    absent = 0

    for strategy, engine, version in cells:
        orig = _original_vu(strategy, engine, version)
        sd, ip = _raw_output_paths(strategy, engine, version)
        row: dict = {
            "strategy": strategy,
            "engine": engine,
            "version": version,
            "orig_schema_r": orig.get("schema_recall") if orig else None,
            "orig_schema_p": orig.get("schema_precision") if orig else None,
            "orig_inv_r": orig.get("invariant_recall") if orig else None,
            "orig_inv_p": orig.get("invariant_precision") if orig else None,
        }
        if sd is None and ip is None:
            row["status"] = "not re-scoreable (raw output absent)"
            absent += 1
            rows.append(row)
            continue

        res = G.score_cell_vs_gt(
            strategy=strategy,
            engine=engine,
            version_slug=version,
            bump_distance="active",
            schema_output_path=sd,
            invariants_output_path=ip,
        )
        s = res.strict
        t = res.tolerant
        row.update(
            {
                "status": "rescored",
                "strict_schema_r": s.schema_recall,
                "strict_schema_p": s.schema_precision,
                "strict_inv_r": s.invariant_recall,
                "strict_inv_p": s.invariant_precision,
                "tol_schema_r": t.schema_recall,
                "tol_schema_p": t.schema_precision,
                "tol_inv_r": t.invariant_recall,
                "tol_inv_p": t.invariant_precision,
                "partial_schema": sd is None,
                "partial_inv": ip is None,
            }
        )
        rescored += 1
        rows.append(row)

    _write_report(rows, rescored, absent)
    print(f"rescored={rescored} absent={absent} total_cells={len(cells)}")


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "-"


def _write_report(rows: list[dict], rescored: int, absent: int) -> None:
    out = _FINDINGS / "wave1_rescored_against_gt.md"
    lines: list[str] = []
    lines.append("# Wave 1 cells re-scored against ground truth (experiment-queue step 0.4)")
    lines.append("")
    lines.append(
        "Each Wave 1 cell whose (engine, version) has an Opus-established ground "
        "truth, re-scored against the canonical GT. Columns: original "
        "validated-union (VU) recall/precision (from "
        "`findings/trial_scores/*__vu.json`) side by side with the new vs-GT "
        "numbers, both STRICT (locked-scorer identity) and TOLERANT "
        "(convention-insensitive: invariants on (field, coarse predicate "
        "bucket); schema on field name, namespace dropped)."
    )
    lines.append("")
    lines.append(
        "TOLERANT is the defensible headline given the namespace + "
        "predicate-kind convention drift between GT and the mined catalogues; "
        "STRICT bounds quality from below. See `wave2_deviations.md` for the "
        "matching-method record."
    )
    lines.append("")
    lines.append(
        f"Summary: {rescored} cells re-scored, {absent} not re-scoreable (raw output absent)."
    )
    lines.append("")

    # Re-scored table
    lines.append("## Re-scored cells")
    lines.append("")
    header = (
        "| cell | VU sch r/p | VU inv r/p | "
        "strict sch r/p | strict inv r/p | tol sch r/p | tol inv r/p | note |"
    )
    sep = "|" + "---|" * 8
    lines.append(header)
    lines.append(sep)
    for r in rows:
        if r["status"] != "rescored":
            continue
        cell = f"{r['strategy']}/{r['engine']}/{r['version']}"
        note_bits = []
        if r.get("partial_schema"):
            note_bits.append("schema raw absent")
        if r.get("partial_inv"):
            note_bits.append("invariants raw absent")
        note = "; ".join(note_bits) or ""
        lines.append(
            f"| {cell} "
            f"| {_fmt(r['orig_schema_r'])}/{_fmt(r['orig_schema_p'])} "
            f"| {_fmt(r['orig_inv_r'])}/{_fmt(r['orig_inv_p'])} "
            f"| {_fmt(r['strict_schema_r'])}/{_fmt(r['strict_schema_p'])} "
            f"| {_fmt(r['strict_inv_r'])}/{_fmt(r['strict_inv_p'])} "
            f"| {_fmt(r['tol_schema_r'])}/{_fmt(r['tol_schema_p'])} "
            f"| {_fmt(r['tol_inv_r'])}/{_fmt(r['tol_inv_p'])} "
            f"| {note} |"
        )
    lines.append("")

    # Absent table
    absent_rows = [r for r in rows if r["status"] != "rescored"]
    if absent_rows:
        lines.append("## Not re-scoreable (raw output absent)")
        lines.append("")
        lines.append("| cell | VU sch r/p | VU inv r/p | reason |")
        lines.append("|---|---|---|---|")
        for r in absent_rows:
            cell = f"{r['strategy']}/{r['engine']}/{r['version']}"
            lines.append(
                f"| {cell} "
                f"| {_fmt(r['orig_schema_r'])}/{_fmt(r['orig_schema_p'])} "
                f"| {_fmt(r['orig_inv_r'])}/{_fmt(r['orig_inv_p'])} "
                f"| {r['status']} |"
            )
        lines.append("")

    out.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
