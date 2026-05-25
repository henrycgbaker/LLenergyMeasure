"""Phase 4.0 driver - build validated unions + re-score all cells.

Per the trial DECISIONS_LOG entry "treat the union of all discovered
across all methods as the upper bound": the corrected ground truth for
each cell is the UNION of every strategy's mined invariants, filtered to
the entries that pass runtime validation.

This script does five things:

1. Builds the validated union for each active cell (3 cells -
   transformers v4_57_3, vllm v0_7_3, tensorrt v1_2_1) via the
   per-engine container dispatcher.
2. For off-active cells (12 cells), the union INHERITS the active cell's
   union as reference per the trial's "active-cell-is-the-ground-truth-
   anchor" convention - off-active cells get scored against the same
   per-engine union, with the bump-distance interpretation supplying the
   brittleness story.
3. Re-scores every (a)/(b)/(d-ab)/h*/e* cell against the validated union
   (instead of (a)'s output) for both schema + invariant axes.
4. Writes `_spike/findings/trial_scores/<strategy>__<engine>__<vslug>__vu.json`
   files (the `_vu` suffix = validated-union scoring) without touching
   the original (a)-based score files.
5. Refreshes `_spike/findings/trial_matrix_vu.{md,csv}` + writes
   `_spike/findings/phase4_0_validated_union_summary.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Imports after sys.path insertion.
from _spike.scripts.trial_scoring import (  # noqa: E402
    FailureMode,
    ScoringConfig,
    build_validated_union,
    score_invariants,
    score_schema,
)


# Per Phase 1 lock: the active cell per engine maps to the canonical
# container image. Off-active cells fall back to the active cell's
# union for scoring (the active-anchor convention).
#
# Note: tensorrt's active version_slug is v0_21_0 per current.toml, but
# the production container image is v1_2_1 (per Phase 1 Day 2 - 0.21.0
# wasn't packaged as a standalone image; the production validator runs
# inside the 1.2.1 container which has all 0.21+ types available). The
# version_slug here is what the strategy outputs use on disk; the image
# override per cell handles the container choice.
ACTIVE_CELLS: list[tuple[str, str]] = [
    ("transformers", "v4_57_3"),
    ("vllm", "v0_7_3"),
    ("tensorrt", "v0_21_0"),
]

# All 15 cells per Phase 1 lock (engine, version_slug).
ALL_CELLS: list[tuple[str, str, str]] = [
    ("transformers", "v4_55_4", "v-2"),
    ("transformers", "v4_56_2", "v-1"),
    ("transformers", "v4_57_3", "active"),
    ("transformers", "v4_57_6", "v+1"),
    ("transformers", "v5_9_0", "v+major"),
    ("vllm", "v0_6_0", "v-2"),
    ("vllm", "v0_6_6_post1", "v-1"),
    ("vllm", "v0_7_3", "active"),
    ("vllm", "v0_9_2", "v+1"),
    ("vllm", "v0_19_1", "v+major"),
    ("tensorrt", "v0_19_0", "v-2"),
    ("tensorrt", "v0_20_0", "v-1"),
    ("tensorrt", "v0_21_0", "active"),
    ("tensorrt", "v1_0_0", "v+1"),
    ("tensorrt", "v1_2_1", "v+major"),
]


# Active version slug per engine - used to resolve which union file to
# score off-active cells against.
ACTIVE_SLUG_BY_ENGINE = {
    "transformers": "v4_57_3",
    "vllm": "v0_7_3",
    "tensorrt": "v0_21_0",
}


def active_image_override(engine: str, version_slug: str) -> str | None:
    """For the active tensorrt cell, validation runs against the v+major
    image (per Phase 1 lock - 1.2.1 is the production image; 0.21.0
    wasn't packaged) so the union reflects what the production validator
    confirms.

    The active tensorrt cell mines outputs at version_slug v0_21_0 but
    the production validator container is v1_2_1.
    """
    if engine == "tensorrt" and version_slug == "v0_21_0":
        return "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
    return None


def _active_image(engine: str) -> str | None:
    """Resolve the production image for an engine's active cell."""
    if engine == "transformers":
        return "llenergymeasure:transformers-4.57.3"
    if engine == "vllm":
        return "llenergymeasure:vllm-v0.7.3"
    if engine == "tensorrt":
        return "nvcr.io/nvidia/tensorrt-llm/release:1.2.1"
    return None


@dataclass
class UnionSummary:
    engine: str
    version_slug: str
    total_unioned: int = 0
    """Identities collected across every strategy for this cell."""

    validated_count: int = 0
    """Identities that passed BOTH positive_confirmed AND negative_confirmed."""

    validation_errors: int = 0
    """Identities that hit validation-infra errors (count NOT as confirmed
    positives/negatives - separate bucket per Phase 1 Day 2 tensorrt finding)."""

    partial_count: int = 0
    """Identities where positive_confirmed XOR negative_confirmed (or
    neither) and no infra error. Not in the union; surfaced as a
    bucket so failure modes are visible."""

    contributors_by_strategy: dict[str, int] = field(default_factory=dict)
    """Per-strategy contributor count: how many union entries had this
    strategy in its contributor list."""


# ---------------------------------------------------------------------------
# Step 1: build validated unions for active cells
# ---------------------------------------------------------------------------


def build_unions(*, dry_run: bool = False) -> dict[tuple[str, str], UnionSummary]:
    """Build the validated union for each active cell.

    The active cell holds the per-engine union; off-active cells inherit
    it as reference (the trial's active-anchor convention).
    """
    summaries: dict[tuple[str, str], UnionSummary] = {}
    for engine, version_slug in ACTIVE_CELLS:
        summary = UnionSummary(engine=engine, version_slug=version_slug)
        print(f"[phase4_0] building union for {engine}/{version_slug}...", flush=True)
        if dry_run:
            summaries[(engine, version_slug)] = summary
            continue
        union_path = build_validated_union(
            engine=engine,
            version_slug=version_slug,
            image_override=active_image_override(engine, version_slug),
            timeout_sec=900,
        )
        envelope = _load_yaml(union_path)
        contributors = envelope.get("contributors_by_identity", {}) or {}
        errors = envelope.get("validation_errors_by_identity", {}) or {}
        partials = envelope.get("partial_validations_by_identity", {}) or {}

        summary.total_unioned = len(contributors) + len(errors) + len(partials)
        summary.validated_count = len(envelope.get("invariants") or [])
        summary.validation_errors = len(errors)
        summary.partial_count = len(partials)
        strategy_counts: dict[str, int] = defaultdict(int)
        for _, strategies in contributors.items():
            for s in strategies or []:
                strategy_counts[s] += 1
        summary.contributors_by_strategy = dict(strategy_counts)
        summaries[(engine, version_slug)] = summary
        print(
            f"[phase4_0]   {engine}/{version_slug}: total={summary.total_unioned} "
            f"validated={summary.validated_count} partial={summary.partial_count} "
            f"infra_err={summary.validation_errors}",
            flush=True,
        )
    return summaries


# ---------------------------------------------------------------------------
# Step 2 + 3: re-score every cell against its engine's validated union
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return yaml.safe_load(path.read_text()) or {}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _union_path(engine: str) -> Path:
    """Return the validated-union YAML path for the engine's active cell."""
    active_slug = ACTIVE_SLUG_BY_ENGINE[engine]
    return (
        _PROJECT_ROOT
        / "_spike"
        / "findings"
        / "validated_union"
        / f"{engine}__{active_slug}.yaml"
    )


def _existing_score_path(strategy: str, engine: str, version_slug: str) -> Path | None:
    """Find the existing (a-as-reference) score JSON file for this cell."""
    candidate = (
        _PROJECT_ROOT
        / "_spike"
        / "findings"
        / "trial_scores"
        / f"{strategy}__{engine}__{version_slug}.json"
    )
    return candidate if candidate.exists() else None


def rescore_all_cells(summaries: dict[tuple[str, str], UnionSummary]) -> list[dict[str, Any]]:
    """Re-score every existing cell score against the validated union.

    Reads the original score JSON to recover (strategy, engine,
    version_slug, bump_distance, wall_clock, energy) + the cell's
    schema_path + invariants_path. Re-computes recall/precision against
    the engine's union YAML.

    Writes new `<strategy>__<engine>__<vslug>__vu.json` files. The
    original (a)-as-reference scores are untouched.
    """
    new_scores: list[dict[str, Any]] = []
    scores_dir = _PROJECT_ROOT / "_spike" / "findings" / "trial_scores"
    cfg = ScoringConfig()

    # Find every existing score file that doesn't already have a `_vu` suffix.
    for original_path in sorted(scores_dir.glob("*.json")):
        if original_path.stem.endswith("__vu"):
            continue
        try:
            original = _load_json(original_path)
        except json.JSONDecodeError:
            print(f"[phase4_0]   skip malformed {original_path}", file=sys.stderr)
            continue

        engine = original.get("engine")
        strategy = original.get("strategy")
        version_slug = original.get("version_slug")
        if engine not in ACTIVE_SLUG_BY_ENGINE or not strategy or not version_slug:
            continue

        # Resolve the union path for this cell's engine.
        union_path = _union_path(engine)
        if not union_path.exists():
            print(f"[phase4_0]   union missing for {engine}; skip {original_path.name}")
            continue
        union_invs = _load_yaml(union_path)
        if not union_invs.get("invariants"):
            print(
                f"[phase4_0]   empty union for {engine}; "
                f"vu score will be empty for {original_path.name}"
            )

        # Locate the cell's schema + invariants outputs (the paths recorded
        # in the original score live on the MAIN worktree's project root;
        # rewrite to the trial worktree.)
        cell_schema_path = _rewrite_to_trial_worktree(original.get("cell_schema_path", ""))
        cell_inv_path = _rewrite_to_trial_worktree(original.get("cell_invariants_path", ""))

        if not cell_schema_path or not cell_inv_path:
            print(f"[phase4_0]   no cell paths for {original_path.name}; skip")
            continue
        if not cell_schema_path.exists() or not cell_inv_path.exists():
            print(
                f"[phase4_0]   cell artefacts missing for {original_path.name}; skip"
                f" (schema={cell_schema_path.exists()}, inv={cell_inv_path.exists()})"
            )
            continue

        cell_schema = _load_json(cell_schema_path)
        cell_invs = _load_yaml(cell_inv_path)
        # Reference schema = the engine's canonical engine_versions schema
        # (schema_recall/precision is best-effort vs the static-mined
        # schema; the VU change targets invariants only, but we keep
        # schema scoring against the existing canonical schema for now).
        active_slug = ACTIVE_SLUG_BY_ENGINE[engine]
        ref_schema_path = (
            _PROJECT_ROOT
            / "engine_versions"
            / engine
            / active_slug
            / "outputs"
            / "schema.discovered.json"
        )
        if not ref_schema_path.exists():
            print(f"[phase4_0]   no canonical schema for {engine}; skip {original_path.name}")
            continue
        ref_schema = _load_json(ref_schema_path)

        # Compute new schema + invariant scores
        (
            schema_recall,
            schema_precision,
            schema_type_accuracy,
            schema_mode,
            s_ref_n,
            s_cell_n,
            s_int_n,
            _smiss,
            _sspur,
            _sdiff,
        ) = score_schema(reference_schema=ref_schema, cell_schema=cell_schema, config=cfg)
        (
            inv_recall,
            inv_precision,
            inv_sev_accuracy,
            inv_mode,
            i_ref_n,
            i_cell_n,
            i_int_n,
            _imiss,
            _ispur,
            _idiff,
        ) = score_invariants(
            reference_invariants=union_invs, cell_invariants=cell_invs, config=cfg
        )

        # Build vu-scored record, preserving identity + cost from original.
        vu_score = dict(original)
        vu_score["scoring_reference"] = "validated_union"
        vu_score["schema_recall"] = schema_recall
        vu_score["schema_precision"] = schema_precision
        vu_score["schema_type_accuracy"] = schema_type_accuracy
        vu_score["schema_failure_mode"] = schema_mode.value
        vu_score["schema_reference_count"] = s_ref_n
        vu_score["schema_cell_count"] = s_cell_n
        vu_score["schema_intersection_count"] = s_int_n
        vu_score["invariant_recall"] = inv_recall
        vu_score["invariant_precision"] = inv_precision
        vu_score["invariant_severity_accuracy"] = inv_sev_accuracy
        vu_score["invariant_failure_mode"] = inv_mode.value
        vu_score["invariant_reference_count"] = i_ref_n
        vu_score["invariant_cell_count"] = i_cell_n
        vu_score["invariant_intersection_count"] = i_int_n
        # Aggregate failure_modes deduped
        all_modes = (original.get("failure_modes") or []) + [
            schema_mode.value,
            inv_mode.value,
        ]
        seen: set[str] = set()
        deduped: list[str] = []
        for m in all_modes:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        vu_score["failure_modes"] = deduped
        vu_score["reference_path"] = str(union_path)
        vu_score["scored_at"] = datetime.now(timezone.utc).isoformat()

        # Carry over observations + append vu-specific note
        observations = list(original.get("observations") or [])
        observations.append(
            f"vu_rescore: reference={engine} union; ref_count={i_ref_n}, "
            f"cell={i_cell_n}, intersect={i_int_n}"
        )
        vu_score["observations"] = observations

        out_path = (
            scores_dir
            / f"{strategy}__{engine}__{version_slug}__vu.json"
        )
        out_path.write_text(json.dumps(vu_score, indent=2, sort_keys=True))
        new_scores.append(vu_score)
        print(
            f"[phase4_0]   vu-rescored {strategy}/{engine}/{version_slug}: "
            f"inv recall={inv_recall:.3f} prec={inv_precision:.3f} ref_count={i_ref_n}",
            flush=True,
        )
    return new_scores


def _rewrite_to_trial_worktree(path_str: str) -> Path | None:
    """The score JSONs sometimes record paths under the MAIN worktree
    `/home/.../workspace/llenergymeasure/...`. Rewrite to the current
    trial worktree if needed."""
    if not path_str:
        return None
    p = Path(path_str)
    main_root = "/home/h.baker@hertie-school.lan/workspace/llenergymeasure"
    trial_root = str(_PROJECT_ROOT)
    if path_str.startswith(main_root + "/"):
        rewritten = trial_root + path_str[len(main_root) :]
        return Path(rewritten)
    return p


# ---------------------------------------------------------------------------
# Step 4: matrix synthesis (vu-suffixed)
# ---------------------------------------------------------------------------


def emit_vu_matrix(vu_scores: list[dict[str, Any]]) -> None:
    """Refresh `_spike/findings/trial_matrix_vu.{md,csv}` from the
    `*__vu.json` score files."""
    import csv

    out_md = _PROJECT_ROOT / "_spike" / "findings" / "trial_matrix_vu.md"
    out_csv = _PROJECT_ROOT / "_spike" / "findings" / "trial_matrix_vu.csv"

    # Per-strategy aggregate (recall/precision against the validated union).
    by_strategy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in vu_scores:
        by_strategy[s.get("strategy", "<unknown>")].append(s)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _pct(v: float) -> str:
        return f"{v * 100:.1f}%"

    lines: list[str] = []
    lines.append("# Empirical trial matrix - validated-union scoring")
    lines.append("")
    lines.append(
        f"_generated at {datetime.now(timezone.utc).isoformat()}; "
        f"score files: {len(vu_scores)}_"
    )
    lines.append("")
    lines.append(
        "Per the trial DECISIONS_LOG entry on the validated-union ground truth: "
        "this matrix scores every cell against the UNION of all strategies' "
        "outputs filtered to runtime-validated entries, NOT against (a)'s output. "
        "See `_spike/findings/phase4_0_validated_union_summary.md` for the "
        "methodology + per-cell union sizes."
    )
    lines.append("")

    lines.append("## Per-strategy aggregate (vu scoring)")
    lines.append("")
    lines.append(
        "| strategy | cells | inv_recall_mean | inv_precision_mean | "
        "schema_recall_mean | wall_mean_s |"
    )
    lines.append("|---|---|---|---|---|---|")
    for strat in sorted(by_strategy):
        cells = by_strategy[strat]
        qual = [c for c in cells if "crash" not in (c.get("failure_modes") or [])]
        ir = _mean([float(c.get("invariant_recall", 0.0)) for c in qual])
        ip = _mean([float(c.get("invariant_precision", 0.0)) for c in qual])
        sr = _mean([float(c.get("schema_recall", 0.0)) for c in qual])
        wc = _mean([float(c.get("wall_clock_sec", 0.0)) for c in qual])
        lines.append(
            f"| {strat} | {len(cells)} | {_pct(ir)} | {_pct(ip)} | {_pct(sr)} | {wc:.1f} |"
        )
    lines.append("")

    # Per-cell rows
    lines.append("## Per-cell vu scores")
    lines.append("")
    lines.append(
        "| strategy | engine | version | bump | inv_recall | inv_prec | "
        "inv_ref | inv_cell | inv_int | failure_modes |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    sorted_scores = sorted(
        vu_scores,
        key=lambda s: (s.get("strategy", ""), s.get("engine", ""), s.get("version_slug", "")),
    )
    for s in sorted_scores:
        modes = ";".join(s.get("failure_modes") or [])
        lines.append(
            f"| {s.get('strategy', '')} | {s.get('engine', '')} | "
            f"{s.get('version_slug', '')} | {s.get('bump_distance', '')} | "
            f"{_pct(float(s.get('invariant_recall', 0.0)))} | "
            f"{_pct(float(s.get('invariant_precision', 0.0)))} | "
            f"{s.get('invariant_reference_count', 0)} | "
            f"{s.get('invariant_cell_count', 0)} | "
            f"{s.get('invariant_intersection_count', 0)} | {modes} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines))

    csv_columns = [
        "strategy",
        "engine",
        "version_slug",
        "bump_distance",
        "schema_recall",
        "schema_precision",
        "invariant_recall",
        "invariant_precision",
        "invariant_severity_accuracy",
        "wall_clock_sec",
        "energy_wh",
        "invariant_reference_count",
        "invariant_cell_count",
        "invariant_intersection_count",
        "schema_reference_count",
        "schema_cell_count",
        "schema_intersection_count",
        "failure_modes",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for s in sorted_scores:
            row = {c: s.get(c, "") for c in csv_columns}
            modes = s.get("failure_modes") or []
            row["failure_modes"] = ";".join(modes) if isinstance(modes, list) else str(modes)
            writer.writerow(row)
    print(f"[phase4_0] wrote {out_md} and {out_csv}", flush=True)


# ---------------------------------------------------------------------------
# Step 5: per-strategy aggregate deltas + summary doc
# ---------------------------------------------------------------------------


def emit_summary(
    *,
    summaries: dict[tuple[str, str], UnionSummary],
    vu_scores: list[dict[str, Any]],
) -> None:
    """Write phase4_0_validated_union_summary.md.

    Includes:
    - per-engine union sizes + delta vs (a) alone
    - per-strategy aggregate deltas from (a)-as-reference to vu scoring
    - failure-mode breakdown for validation infra
    - cells where the union shifted the picture
    """
    summary_path = (
        _PROJECT_ROOT / "_spike" / "findings" / "phase4_0_validated_union_summary.md"
    )

    # Load the original (a-as-reference) scores so we can compute deltas.
    scores_dir = _PROJECT_ROOT / "_spike" / "findings" / "trial_scores"
    original_scores: list[dict[str, Any]] = []
    for original_path in sorted(scores_dir.glob("*.json")):
        if original_path.stem.endswith("__vu"):
            continue
        try:
            original_scores.append(_load_json(original_path))
        except json.JSONDecodeError:
            continue

    def _strategy_mean(scores: list[dict[str, Any]], strategy: str, field_name: str) -> float:
        vals = [
            float(s.get(field_name, 0.0))
            for s in scores
            if s.get("strategy") == strategy
            and "crash" not in (s.get("failure_modes") or [])
        ]
        return sum(vals) / len(vals) if vals else 0.0

    lines: list[str] = []
    lines.append("# Phase 4.0 - validated-union ground truth summary")
    lines.append("")
    lines.append(
        f"_generated at {datetime.now(timezone.utc).isoformat()}_"
    )
    lines.append("")
    lines.append(
        "Per the DECISIONS_LOG entry framing: the corrected ground truth "
        "for the mining-substrate trial is the UNION of every strategy's "
        "mined invariants, filtered to entries that pass runtime validation. "
        "(a)'s output is one input among several; equal weight in the union."
    )
    lines.append("")
    lines.append("## Per-cell union sizes (active cells)")
    lines.append("")
    lines.append(
        "| engine | version | unique_unioned | validated_both | infra_errors | "
        "a_alone_ref | union_delta_vs_a |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for (engine, version), summary in summaries.items():
        # (a)-alone count = canonical engine_versions invariants.proposed count
        a_ref = (
            _PROJECT_ROOT
            / "engine_versions"
            / engine
            / version
            / "outputs"
            / "invariants.proposed.yaml"
        )
        a_count = 0
        if a_ref.exists():
            a_inv = _load_yaml(a_ref)
            a_count = len(a_inv.get("invariants") or [])
        delta = summary.validated_count - a_count
        lines.append(
            f"| {engine} | {version} | {summary.total_unioned} | "
            f"{summary.validated_count} | {summary.validation_errors} | "
            f"{a_count} | {delta:+d} |"
        )
    lines.append("")

    lines.append("## Per-strategy contributor breakdown (active cells)")
    lines.append("")
    lines.append("Which strategies contributed at least once to the validated union for each cell.")
    lines.append("")
    lines.append("| engine | version | a | b | d-ab | h2 | h3 | h6 | e6 | e9 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for (engine, version), summary in summaries.items():
        c = summary.contributors_by_strategy
        lines.append(
            f"| {engine} | {version} | "
            f"{c.get('a', 0)} | {c.get('b', 0)} | {c.get('d-ab', 0)} | "
            f"{c.get('h2', 0)} | {c.get('h3', 0)} | {c.get('h6', 0)} | "
            f"{c.get('e6', 0)} | {c.get('e9', 0)} |"
        )
    lines.append("")

    # Unique-contributor analysis - which strategies found things NO
    # other strategy found?
    lines.append("## Unique-contributor count per strategy (active cells)")
    lines.append("")
    lines.append("Number of validated union entries where this strategy is the ONLY")
    lines.append("contributor. High counts indicate distinctive coverage; zero counts")
    lines.append("indicate the strategy's outputs were entirely captured by other strategies.")
    lines.append("")
    lines.append("| engine | version | a | b | d-ab | h2 | h3 | h6 | e6 | e9 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for engine, version in [(e, v) for (e, v), _ in summaries.items()]:
        union_path = _union_path(engine)
        if not union_path.exists():
            continue
        envelope = _load_yaml(union_path)
        contribs = envelope.get("contributors_by_identity", {}) or {}
        unique_counts: dict[str, int] = defaultdict(int)
        for _ident, strategies in contribs.items():
            if len(strategies) == 1:
                unique_counts[strategies[0]] += 1
        lines.append(
            f"| {engine} | {version} | "
            f"{unique_counts.get('a', 0)} | {unique_counts.get('b', 0)} | "
            f"{unique_counts.get('d-ab', 0)} | {unique_counts.get('h2', 0)} | "
            f"{unique_counts.get('h3', 0)} | {unique_counts.get('h6', 0)} | "
            f"{unique_counts.get('e6', 0)} | {unique_counts.get('e9', 0)} |"
        )
    lines.append("")

    lines.append("## Per-strategy aggregate deltas: (a)-as-reference -> validated-union")
    lines.append("")
    lines.append(
        "Recall + precision computed against the new reference. Strategies "
        "that beat (a) on the union (find things (a) missed) shift down on "
        "(a)-relative recall but reveal their true coverage."
    )
    lines.append("")
    lines.append(
        "| strategy | cells | inv_recall_a | inv_recall_vu | delta | "
        "inv_precision_a | inv_precision_vu | delta |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    all_strategies = sorted(
        {
            *(s.get("strategy", "<unknown>") for s in original_scores),
            *(s.get("strategy", "<unknown>") for s in vu_scores),
        }
    )
    for strat in all_strategies:
        cells = [s for s in original_scores if s.get("strategy") == strat]
        vu_cells = [s for s in vu_scores if s.get("strategy") == strat]
        ir_a = _strategy_mean(original_scores, strat, "invariant_recall")
        ir_vu = _strategy_mean(vu_scores, strat, "invariant_recall")
        ip_a = _strategy_mean(original_scores, strat, "invariant_precision")
        ip_vu = _strategy_mean(vu_scores, strat, "invariant_precision")
        lines.append(
            f"| {strat} | {len(vu_cells)}/{len(cells)} | "
            f"{ir_a * 100:.1f}% | {ir_vu * 100:.1f}% | {(ir_vu - ir_a) * 100:+.1f}pp | "
            f"{ip_a * 100:.1f}% | {ip_vu * 100:.1f}% | {(ip_vu - ip_a) * 100:+.1f}pp |"
        )
    lines.append("")

    lines.append("## Validation infrastructure errors")
    lines.append("")
    lines.append("Per Phase 1 Day 2 tensorrt finding: type-blind probe synthesis leaves")
    lines.append("entries that pass static identity comparison but fail when the live")
    lines.append("library tries to construct from kwargs. These DON'T count as failed")
    lines.append("union entries; they count as `validation_error` (separate bucket).")
    lines.append("")
    for (engine, version), summary in summaries.items():
        if summary.validation_errors:
            lines.append(
                f"- {engine}/{version}: {summary.validation_errors} infra errors out "
                f"of {summary.total_unioned} unique unioned entries "
                f"({summary.validation_errors / max(1, summary.total_unioned) * 100:.1f}%)"
            )
        else:
            lines.append(f"- {engine}/{version}: 0 infra errors")
    lines.append("")

    lines.append("## Cells where the union shifted the picture")
    lines.append("")
    lines.append(
        "Cells where vu scoring changed the headline recall by >5pp (positive or "
        "negative). Positive shift = strategy looks BETTER under vu (it found "
        "things (a) missed); negative shift = strategy looks WORSE (its prior "
        "recall was inflated by being measured against (a)'s narrow reference)."
    )
    lines.append("")
    lines.append("| strategy | engine | version | inv_recall_a | inv_recall_vu | delta |")
    lines.append("|---|---|---|---|---|---|")
    pairs: dict[tuple[str, str, str], tuple[float, float]] = {}
    for s in original_scores:
        key = (s.get("strategy", ""), s.get("engine", ""), s.get("version_slug", ""))
        pairs[key] = (float(s.get("invariant_recall", 0.0)), 0.0)
    for s in vu_scores:
        key = (s.get("strategy", ""), s.get("engine", ""), s.get("version_slug", ""))
        if key in pairs:
            pairs[key] = (pairs[key][0], float(s.get("invariant_recall", 0.0)))
    shifted = sorted(
        ((k, vals) for k, vals in pairs.items() if abs(vals[1] - vals[0]) > 0.05),
        key=lambda kv: kv[1][1] - kv[1][0],
    )
    for (strat, engine, version), (r_a, r_vu) in shifted:
        lines.append(
            f"| {strat} | {engine} | {version} | "
            f"{r_a * 100:.1f}% | {r_vu * 100:.1f}% | "
            f"{(r_vu - r_a) * 100:+.1f}pp |"
        )
    lines.append("")

    summary_path.write_text("\n".join(lines))
    print(f"[phase4_0] wrote {summary_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip the union build step (use pre-existing union YAMLs).",
    )
    parser.add_argument(
        "--skip-rescore",
        action="store_true",
        help="Skip the per-cell rescoring step.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't run the per-engine validators (use for smoke).",
    )
    args = parser.parse_args(argv)

    print(
        f"[phase4_0] start at {datetime.now(timezone.utc).isoformat()}; "
        "active cells:",
        ACTIVE_CELLS,
        flush=True,
    )

    if args.skip_build:
        # Rebuild summaries from disk
        summaries: dict[tuple[str, str], UnionSummary] = {}
        for engine, version_slug in ACTIVE_CELLS:
            union_path = _union_path(engine)
            if not union_path.exists():
                summaries[(engine, version_slug)] = UnionSummary(engine, version_slug)
                continue
            envelope = _load_yaml(union_path)
            contributors = envelope.get("contributors_by_identity", {}) or {}
            errors = envelope.get("validation_errors_by_identity", {}) or {}
            partials = envelope.get("partial_validations_by_identity", {}) or {}
            summary = UnionSummary(
                engine=engine,
                version_slug=version_slug,
                total_unioned=len(contributors) + len(errors) + len(partials),
                validated_count=len(envelope.get("invariants") or []),
                validation_errors=len(errors),
                partial_count=len(partials),
            )
            strategy_counts: dict[str, int] = defaultdict(int)
            for _, strategies in contributors.items():
                for s in strategies or []:
                    strategy_counts[s] += 1
            summary.contributors_by_strategy = dict(strategy_counts)
            summaries[(engine, version_slug)] = summary
    else:
        summaries = build_unions(dry_run=args.dry_run)

    if args.skip_rescore:
        # Load existing vu scores from disk for the summary step
        scores_dir = _PROJECT_ROOT / "_spike" / "findings" / "trial_scores"
        vu_scores = []
        for p in sorted(scores_dir.glob("*__vu.json")):
            try:
                vu_scores.append(_load_json(p))
            except json.JSONDecodeError:
                pass
    else:
        vu_scores = rescore_all_cells(summaries)

    emit_vu_matrix(vu_scores)
    emit_summary(summaries=summaries, vu_scores=vu_scores)

    print(f"[phase4_0] complete; vu_scores={len(vu_scores)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
