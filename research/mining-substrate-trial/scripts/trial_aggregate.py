"""Aggregator for the mining-substrate empirical trial.

Phase 1 Day 5 stub. Reference: ``.planning/mining-substrate-empirical-trial.md``
§ "Tooling (shared)" + ``research/mining-substrate-trial/findings/trial_epistemic_framing.md`` § "Synthesis
is multi-axis, not a leaderboard".

Reads per-cell score JSONs from ``research/mining-substrate-trial/findings/trial_scores/`` and emits
a synthesis matrix:

1. ``trial_matrix.md`` - human-readable strategy x cell tables + summaries.
2. ``trial_matrix.csv`` - flat, machine-parseable.

Sections in the markdown output:

* Strategy x cell matrix: rows = strategies, columns = (engine, version)
  cells. Cell values: recall + precision + failure-mode-tag.
* Per-strategy aggregate: recall mean/median, precision mean/median,
  wall-clock mean. Brittleness rollup (avg pass-through rate).
* Per-engine aggregate: same shape, grouped by engine.
* Per-bump-distance aggregate: rolls the brittleness story
  (active vs v-1 vs v+1 vs v+major; degradation curves).
* Adjacent-observations rollup: collected observations per strategy,
  deduplicated, lightly themed.

Update-in-place semantics: rerunning the aggregator overwrites
``trial_matrix.md`` and ``trial_matrix.csv`` with the union of all
score files currently on disk. No cell is ever silently dropped.

CLI
---
::

    python -m research/mining-substrate-trial/scripts.trial_aggregate              # default paths
    python -m research/mining-substrate-trial/scripts.trial_aggregate --scores-dir /custom --out /tmp/trial
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


DEFAULT_SCORES_DIR = _PROJECT_ROOT / "research" / "mining-substrate-trial" / "findings" / "trial_scores"
DEFAULT_MD_OUT = _PROJECT_ROOT / "research" / "mining-substrate-trial" / "findings" / "trial_matrix.md"
DEFAULT_CSV_OUT = _PROJECT_ROOT / "research" / "mining-substrate-trial" / "findings" / "trial_matrix.csv"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_scores(scores_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` score record under ``scores_dir``.

    Each file is one CellScore (per the runner contract). Defensive against
    legacy / malformed records: warns + skips entries that don't carry the
    minimum identity fields (strategy / engine / version_slug).
    """
    out: list[dict[str, Any]] = []
    for path in sorted(scores_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            print(f"[trial_aggregate] skipping malformed {path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"[trial_aggregate] skipping non-object {path}", file=sys.stderr)
            continue
        for k in ("strategy", "engine", "version_slug"):
            if k not in data:
                print(f"[trial_aggregate] skipping {path}: missing {k}", file=sys.stderr)
                break
        else:
            out.append(data)
    return out


# ---------------------------------------------------------------------------
# Aggregation primitives
# ---------------------------------------------------------------------------


@dataclass
class StrategyAggregate:
    strategy: str
    cell_count: int = 0
    schema_recall_mean: float = 0.0
    schema_recall_median: float = 0.0
    schema_precision_mean: float = 0.0
    schema_precision_median: float = 0.0
    invariant_recall_mean: float = 0.0
    invariant_recall_median: float = 0.0
    invariant_precision_mean: float = 0.0
    invariant_precision_median: float = 0.0
    wall_clock_mean: float = 0.0
    energy_wh_mean: float = 0.0
    brittleness_pass_through_mean: float | None = None
    silent_fail_total: int = 0
    detectable_fail_total: int = 0
    crash_count: int = 0


@dataclass
class EngineAggregate:
    engine: str
    cell_count: int = 0
    schema_recall_mean: float = 0.0
    invariant_recall_mean: float = 0.0
    wall_clock_mean: float = 0.0


@dataclass
class BumpAggregate:
    bump_distance: str  # "v-2" | "v-1" | "active" | "v+1" | "v+major"
    cell_count: int = 0
    schema_recall_mean: float = 0.0
    invariant_recall_mean: float = 0.0
    pass_through_mean: float | None = None


@dataclass
class TrialMatrix:
    """The full assembled aggregation. Serialised to .md + .csv."""

    scores: list[dict[str, Any]] = field(default_factory=list)
    strategy_aggs: dict[str, StrategyAggregate] = field(default_factory=dict)
    engine_aggs: dict[str, EngineAggregate] = field(default_factory=dict)
    bump_aggs: dict[str, BumpAggregate] = field(default_factory=dict)
    observations_by_strategy: dict[str, list[str]] = field(default_factory=dict)


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _quality_scores(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter to scores not flagged CRASH (Phase 3 aggregator convention).

    Cells with CRASH failure_modes get reported in crash_count but excluded
    from quality mean/median rollups.
    """
    return [s for s in group if "crash" not in (s.get("failure_modes") or [])]


def aggregate_strategies(scores: list[dict[str, Any]]) -> dict[str, StrategyAggregate]:
    """Roll per-strategy aggregates: recall mean/median, etc.

    Excludes cells flagged ``FailureMode.CRASH`` from the recall/precision
    rollup (a crash isn't a quality datum), but counts them in
    ``crash_count`` so the synthesis can see them.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        groups[s.get("strategy", "<unknown>")].append(s)
    out: dict[str, StrategyAggregate] = {}
    for strat, group in groups.items():
        quality = _quality_scores(group)
        sr = [float(s.get("schema_recall", 0.0)) for s in quality]
        sp = [float(s.get("schema_precision", 0.0)) for s in quality]
        ir = [float(s.get("invariant_recall", 0.0)) for s in quality]
        ip = [float(s.get("invariant_precision", 0.0)) for s in quality]
        wc = [float(s.get("wall_clock_sec", 0.0)) for s in quality]
        ew = [float(s.get("energy_wh", 0.0)) for s in quality]
        pt = [
            float(s["brittleness_pass_through_rate"])
            for s in quality
            if s.get("brittleness_pass_through_rate") is not None
        ]
        silent = sum(int(s.get("brittleness_silent_fail_count") or 0) for s in quality)
        detectable = sum(int(s.get("brittleness_detectable_fail_count") or 0) for s in quality)
        crashes = sum(1 for s in group if "crash" in (s.get("failure_modes") or []))
        out[strat] = StrategyAggregate(
            strategy=strat,
            cell_count=len(group),
            schema_recall_mean=_mean(sr),
            schema_recall_median=_median(sr),
            schema_precision_mean=_mean(sp),
            schema_precision_median=_median(sp),
            invariant_recall_mean=_mean(ir),
            invariant_recall_median=_median(ir),
            invariant_precision_mean=_mean(ip),
            invariant_precision_median=_median(ip),
            wall_clock_mean=_mean(wc),
            energy_wh_mean=_mean(ew),
            brittleness_pass_through_mean=_mean(pt) if pt else None,
            silent_fail_total=silent,
            detectable_fail_total=detectable,
            crash_count=crashes,
        )
    return out


def aggregate_engines(scores: list[dict[str, Any]]) -> dict[str, EngineAggregate]:
    """Roll per-engine aggregates."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        groups[s.get("engine", "<unknown>")].append(s)
    out: dict[str, EngineAggregate] = {}
    for engine, group in groups.items():
        quality = _quality_scores(group)
        sr = [float(s.get("schema_recall", 0.0)) for s in quality]
        ir = [float(s.get("invariant_recall", 0.0)) for s in quality]
        wc = [float(s.get("wall_clock_sec", 0.0)) for s in quality]
        out[engine] = EngineAggregate(
            engine=engine,
            cell_count=len(group),
            schema_recall_mean=_mean(sr),
            invariant_recall_mean=_mean(ir),
            wall_clock_mean=_mean(wc),
        )
    return out


def aggregate_bumps(scores: list[dict[str, Any]]) -> dict[str, BumpAggregate]:
    """Roll per-bump-distance aggregates - the brittleness signal."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in scores:
        groups[s.get("bump_distance", "<unknown>")].append(s)
    out: dict[str, BumpAggregate] = {}
    for bd, group in groups.items():
        quality = _quality_scores(group)
        sr = [float(s.get("schema_recall", 0.0)) for s in quality]
        ir = [float(s.get("invariant_recall", 0.0)) for s in quality]
        pt = [
            float(s["brittleness_pass_through_rate"])
            for s in quality
            if s.get("brittleness_pass_through_rate") is not None
        ]
        out[bd] = BumpAggregate(
            bump_distance=bd,
            cell_count=len(group),
            schema_recall_mean=_mean(sr),
            invariant_recall_mean=_mean(ir),
            pass_through_mean=_mean(pt) if pt else None,
        )
    return out


def rollup_observations(scores: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Collect, dedupe and roughly cluster adjacent observations per strategy.

    Phase 3 implementation deduplicates by exact-match (no semantic
    clustering in v1.0). The Phase 4 synthesis is where humans semantically
    cluster these into themes.
    """
    out: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for s in scores:
        strat = s.get("strategy", "<unknown>")
        for obs in s.get("observations", []) or []:
            if obs in seen[strat]:
                continue
            seen[strat].add(obs)
            out[strat].append(obs)
    return dict(out)


def build_matrix(scores: list[dict[str, Any]]) -> TrialMatrix:
    """Top-level: load + aggregate + build the matrix object."""
    matrix = TrialMatrix(scores=scores)
    matrix.strategy_aggs = aggregate_strategies(scores)
    matrix.engine_aggs = aggregate_engines(scores)
    matrix.bump_aggs = aggregate_bumps(scores)
    matrix.observations_by_strategy = rollup_observations(scores)
    return matrix


# ---------------------------------------------------------------------------
# Markdown emission
# ---------------------------------------------------------------------------


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _render_strategy_x_cell_matrix(matrix: TrialMatrix) -> list[str]:
    """Render a per-cell table sorted by strategy, engine, version_slug."""
    lines: list[str] = []
    lines.append("## Per-cell matrix")
    lines.append("")
    lines.append(
        "| strategy | engine | version | bump | schema_recall | schema_prec | "
        "inv_recall | inv_prec | sev_acc | wall_s | energy_wh | failure_modes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    sorted_scores = sorted(
        matrix.scores,
        key=lambda s: (
            s.get("strategy", ""),
            s.get("engine", ""),
            s.get("version_slug", ""),
        ),
    )
    for s in sorted_scores:
        modes = ";".join(s.get("failure_modes") or [])
        lines.append(
            "| {strat} | {eng} | {ver} | {bump} | {sr} | {sp} | {ir} | {ip} | "
            "{sev} | {wc:.1f} | {ew:.2f} | {modes} |".format(
                strat=s.get("strategy", ""),
                eng=s.get("engine", ""),
                ver=s.get("version_slug", ""),
                bump=s.get("bump_distance", ""),
                sr=_pct(float(s.get("schema_recall", 0.0))),
                sp=_pct(float(s.get("schema_precision", 0.0))),
                ir=_pct(float(s.get("invariant_recall", 0.0))),
                ip=_pct(float(s.get("invariant_precision", 0.0))),
                sev=_pct(float(s.get("invariant_severity_accuracy", 0.0))),
                wc=float(s.get("wall_clock_sec", 0.0)),
                ew=float(s.get("energy_wh", 0.0)),
                modes=modes,
            )
        )
    lines.append("")
    return lines


def _render_strategy_agg_table(matrix: TrialMatrix) -> list[str]:
    lines: list[str] = []
    lines.append("## Per-strategy aggregates")
    lines.append("")
    lines.append(
        "| strategy | cells | schema_recall_mean | schema_recall_median | "
        "inv_recall_mean | inv_recall_median | wall_mean_s | energy_mean_wh | "
        "crashes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for strat in sorted(matrix.strategy_aggs):
        a = matrix.strategy_aggs[strat]
        lines.append(
            f"| {a.strategy} | {a.cell_count} | {_pct(a.schema_recall_mean)} | {_pct(a.schema_recall_median)} | {_pct(a.invariant_recall_mean)} | {_pct(a.invariant_recall_median)} | {a.wall_clock_mean:.1f} | {a.energy_wh_mean:.2f} | {a.crash_count} |"
        )
    lines.append("")
    return lines


def _render_engine_agg_table(matrix: TrialMatrix) -> list[str]:
    lines: list[str] = []
    lines.append("## Per-engine aggregates")
    lines.append("")
    lines.append("| engine | cells | schema_recall_mean | inv_recall_mean | wall_mean_s |")
    lines.append("|---|---|---|---|---|")
    for engine in sorted(matrix.engine_aggs):
        a = matrix.engine_aggs[engine]
        lines.append(
            f"| {a.engine} | {a.cell_count} | {_pct(a.schema_recall_mean)} | {_pct(a.invariant_recall_mean)} | {a.wall_clock_mean:.1f} |"
        )
    lines.append("")
    return lines


def _render_bump_agg_table(matrix: TrialMatrix) -> list[str]:
    lines: list[str] = []
    lines.append("## Per-bump-distance aggregates")
    lines.append("")
    lines.append("| bump | cells | schema_recall_mean | inv_recall_mean | pass_through_mean |")
    lines.append("|---|---|---|---|---|")
    bump_order = ["v-2", "v-1", "active", "v+1", "v+major", "<unknown>"]
    order_index = {b: i for i, b in enumerate(bump_order)}
    for bd in sorted(matrix.bump_aggs, key=lambda b: order_index.get(b, 99)):
        a = matrix.bump_aggs[bd]
        pt = _pct(a.pass_through_mean) if a.pass_through_mean is not None else "-"
        lines.append(
            f"| {a.bump_distance} | {a.cell_count} | {_pct(a.schema_recall_mean)} | {_pct(a.invariant_recall_mean)} | {pt} |"
        )
    lines.append("")
    return lines


def _render_observations_rollup(matrix: TrialMatrix) -> list[str]:
    lines: list[str] = []
    lines.append("## Adjacent observations (deduped per strategy)")
    lines.append("")
    for strat in sorted(matrix.observations_by_strategy):
        lines.append(f"### strategy {strat}")
        lines.append("")
        for obs in matrix.observations_by_strategy[strat]:
            lines.append(f"- {obs}")
        lines.append("")
    return lines


def emit_markdown(matrix: TrialMatrix, out_path: Path) -> None:
    """Write the human-readable matrix synthesis to a Markdown file."""
    from datetime import datetime, timezone

    lines: list[str] = []
    lines.append("# Empirical trial matrix")
    lines.append("")
    lines.append(f"_generated at {datetime.now(timezone.utc).isoformat()}_")
    lines.append("")
    lines.append(f"_score files aggregated: {len(matrix.scores)}_")
    lines.append("")
    lines.extend(_render_strategy_x_cell_matrix(matrix))
    lines.extend(_render_strategy_agg_table(matrix))
    lines.extend(_render_engine_agg_table(matrix))
    lines.extend(_render_bump_agg_table(matrix))
    lines.extend(_render_observations_rollup(matrix))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------


CSV_COLUMNS = [
    "strategy",
    "engine",
    "version_slug",
    "bump_distance",
    "schema_recall",
    "schema_precision",
    "schema_type_accuracy",
    "invariant_recall",
    "invariant_precision",
    "invariant_severity_accuracy",
    "wall_clock_sec",
    "energy_wh",
    "failure_modes",
    "brittleness_pass_through_rate",
    "brittleness_silent_fail_count",
    "brittleness_detectable_fail_count",
    "brittleness_patch_cost_loc",
    "schema_reference_count",
    "schema_cell_count",
    "schema_intersection_count",
    "invariant_reference_count",
    "invariant_cell_count",
    "invariant_intersection_count",
]


def _row_from_score(s: dict[str, Any]) -> dict[str, Any]:
    row = {col: s.get(col, "") for col in CSV_COLUMNS}
    modes = s.get("failure_modes") or []
    row["failure_modes"] = ";".join(modes) if isinstance(modes, list) else str(modes)
    return row


def emit_csv(matrix: TrialMatrix, out_path: Path) -> None:
    """Write the flat per-cell matrix to CSV.

    Stable column order for downstream stats (pandas read_csv) and
    diff-friendly for git tracking.
    """
    import csv

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_scores = sorted(
        matrix.scores,
        key=lambda s: (
            s.get("strategy", ""),
            s.get("engine", ""),
            s.get("version_slug", ""),
        ),
    )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for s in sorted_scores:
            writer.writerow(_row_from_score(s))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument(
        "--scores-dir",
        type=Path,
        default=DEFAULT_SCORES_DIR,
        help=f"Directory of per-cell score JSON files. Default: {DEFAULT_SCORES_DIR}",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=DEFAULT_MD_OUT,
        help=f"Output markdown path. Default: {DEFAULT_MD_OUT}",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=DEFAULT_CSV_OUT,
        help=f"Output CSV path. Default: {DEFAULT_CSV_OUT}",
    )

    args = parser.parse_args(argv)

    if not args.scores_dir.exists():
        print(f"[trial_aggregate] scores dir not found: {args.scores_dir}", file=sys.stderr)
        return 1

    scores = load_scores(args.scores_dir)
    if not scores:
        print(f"[trial_aggregate] no score files in {args.scores_dir}", file=sys.stderr)
        return 1

    try:
        matrix = build_matrix(scores)
        emit_markdown(matrix, args.md_out)
        emit_csv(matrix, args.csv_out)
    except NotImplementedError as exc:
        print(
            f"[trial_aggregate] stub: {exc.args[0].splitlines()[0] if exc.args else exc}",
            file=sys.stderr,
        )
        return 2

    print(
        f"[trial_aggregate] wrote {args.md_out} and {args.csv_out} from {len(scores)} cell(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
