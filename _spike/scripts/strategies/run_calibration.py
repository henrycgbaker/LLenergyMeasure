"""Calibration runner for strategy (b) on transformers v4_57_3.

Used by Phase 2 to iterate prompts (max 3 rounds) and produce the
"locked at round N" final score.

Usage::

    uv run python -m _spike.scripts.strategies.run_calibration \\
        --round 1 \\
        --out-dir _spike/findings/trial_runs/b/transformers/v4_57_3 \\
        --model llama3.1:70b

Outputs:
- ``<out_dir>/schema.json``
- ``<out_dir>/invariants.proposed.yaml``
- ``<out_dir>/raw_llm_transcripts/*.md``
- ``<out_dir>/calibration_round_<N>_report.md`` (this round's summary)
- ``<out_dir>/calibration_summary.md`` (all rounds; appended to)

Designed to run end-to-end UNATTENDED. Logs to stderr; exit code
reflects whether the cell scored (recall available) or crashed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts import trial_scoring  # noqa: E402
from _spike.scripts.strategies.llm_b_oss import run_b_on_transformers_active  # noqa: E402
from _spike.scripts.strategies.llm_extractor import OllamaBackend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--round", type=int, required=True, help="Calibration round number (1-3).")
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Per-cell run directory (will be created).",
    )
    parser.add_argument("--model", default="llama3.1:70b", help="Ollama model tag.")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11435",
        help="Ollama HTTP base URL (container Ollama on 11435; host on 11434).",
    )
    parser.add_argument(
        "--num-ctx", type=int, default=32768, help="Ollama num_ctx (32k keeps 70B on-GPU)."
    )
    parser.add_argument("--max-retries", type=int, default=2, help="Per-chunk retry budget.")
    parser.add_argument(
        "--engine-version", default="4.57.3", help="Engine version label for the envelope."
    )
    parser.add_argument(
        "--multipass",
        action="store_true",
        help="Run invariants through Phase 2.5 multi-pass pipeline (extract -> verify -> extend).",
    )
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Archive previous round's transcripts (don't overwrite)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    if transcripts_dir.exists() and any(transcripts_dir.iterdir()):
        archive = out_dir / f"raw_llm_transcripts_round_{args.round - 1}"
        if not archive.exists():
            import shutil

            shutil.move(str(transcripts_dir), str(archive))
            transcripts_dir.mkdir(parents=True, exist_ok=True)

    backend = OllamaBackend(model=args.model, url=args.ollama_url, num_ctx=args.num_ctx)

    # Energy sampling
    energy_wh = 0.0
    try:
        from llenergymeasure.energy import select_energy_sampler

        sampler = select_energy_sampler("auto")
    except Exception as exc:
        print(f"[calibration] energy sampler unavailable: {exc}", file=sys.stderr)
        sampler = None

    tracker = None
    if sampler is not None:
        try:
            tracker = sampler.start_tracking()
        except Exception as exc:
            print(f"[calibration] energy tracker failed: {exc}", file=sys.stderr)
            tracker = None

    t0 = time.perf_counter()
    try:
        outputs = run_b_on_transformers_active(
            out_dir=out_dir,
            engine_version=args.engine_version,
            backend=backend,
            max_retries=args.max_retries,
            multipass=args.multipass,
        )
    except Exception as exc:
        wall = time.perf_counter() - t0
        if sampler is not None and tracker is not None:
            try:
                meas = sampler.stop_tracking(tracker)
                # energy.NVMLSampler returns an EnergyMeasurement with total_j
                if hasattr(meas, "total_j"):
                    energy_wh = meas.total_j / 3600.0
            except Exception:
                pass
        print(
            f"[calibration] cell CRASHED after {wall:.1f}s, energy {energy_wh:.2f}Wh: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    wall = time.perf_counter() - t0

    if sampler is not None and tracker is not None:
        try:
            meas = sampler.stop_tracking(tracker)
            if hasattr(meas, "total_j"):
                energy_wh = meas.total_j / 3600.0
            elif hasattr(meas, "energy_wh"):
                energy_wh = float(meas.energy_wh)
        except Exception as exc:
            print(f"[calibration] energy stop failed: {exc}", file=sys.stderr)

    # Score
    ref_schema = (
        _PROJECT_ROOT / "engine_versions/transformers/v4_57_3/outputs/schema.discovered.json"
    )
    ref_inv = (
        _PROJECT_ROOT / "engine_versions/transformers/v4_57_3/outputs/invariants.proposed.yaml"
    )

    score, schema_diffs, inv_diffs, type_or_sev_diffs = trial_scoring.score_cell(
        strategy="b",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="active",
        schema_output_path=outputs.schema_path,
        invariants_output_path=outputs.invariants_path,
        reference_schema_path=ref_schema,
        reference_invariants_path=ref_inv,
        wall_clock_sec=wall,
        energy_wh=energy_wh,
        extra_observations=outputs.observations,
    )

    # Write score JSON
    score_path = out_dir / f"calibration_round_{args.round}_score.json"
    score_path.write_text(score.to_json())

    # Write diff artefacts
    trial_scoring.write_diff_artefact(
        diffs=[d for d in schema_diffs if d.kind == "recall_miss"],
        out_path=out_dir / f"calibration_round_{args.round}_schema_recall_misses.yaml",
        kind_label="recall_misses",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in schema_diffs if d.kind == "precision_spurious"],
        out_path=out_dir / f"calibration_round_{args.round}_schema_precision_spurious.yaml",
        kind_label="precision_spurious",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in inv_diffs if d.kind == "recall_miss"],
        out_path=out_dir / f"calibration_round_{args.round}_invariants_recall_misses.yaml",
        kind_label="recall_misses",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in inv_diffs if d.kind == "precision_spurious"],
        out_path=out_dir / f"calibration_round_{args.round}_invariants_precision_spurious.yaml",
        kind_label="precision_spurious",
    )
    trial_scoring.write_diff_artefact(
        diffs=type_or_sev_diffs,
        out_path=out_dir / f"calibration_round_{args.round}_type_or_severity_mismatches.yaml",
        kind_label="type_or_severity_mismatches",
    )

    # Write round report
    report_lines = [
        f"# Calibration round {args.round} - transformers v{args.engine_version}",
        "",
        f"- model: `{args.model}` (num_ctx={args.num_ctx})",
        f"- max_retries: {args.max_retries}",
        f"- wall_clock: {wall:.1f}s "
        f"(schema {outputs.schema_wall_sec:.1f}s + invariants {outputs.invariants_wall_sec:.1f}s)",
        f"- energy: {energy_wh:.2f} Wh",
        "",
        "## Schema",
        f"- recall: **{score.schema_recall:.1%}** ({score.schema_intersection_count}/{score.schema_reference_count})",
        f"- precision: {score.schema_precision:.1%} ({score.schema_intersection_count}/{score.schema_cell_count})",
        f"- type_accuracy: {score.schema_type_accuracy:.1%}",
        f"- failure_mode: {score.schema_failure_mode}",
        "",
        "## Invariants",
        f"- recall: **{score.invariant_recall:.1%}** ({score.invariant_intersection_count}/{score.invariant_reference_count})",
        f"- precision: {score.invariant_precision:.1%} ({score.invariant_intersection_count}/{score.invariant_cell_count})",
        f"- severity_accuracy: {score.invariant_severity_accuracy:.1%}",
        f"- failure_mode: {score.invariant_failure_mode}",
        "",
        "## Observations",
        "",
    ]
    for obs in outputs.observations:
        report_lines.append(f"- {obs}")
    (out_dir / f"calibration_round_{args.round}_report.md").write_text("\n".join(report_lines))

    # Append to summary
    summary_path = out_dir / "calibration_summary.md"
    if not summary_path.exists():
        summary_path.write_text(
            "# Calibration summary - transformers v4_57_3 strategy (b)\n\n"
            "| Round | Schema recall | Schema precision | Inv recall | Inv precision | Wall (s) | Notes |\n"
            "|---|---|---|---|---|---|---|\n"
        )
    with summary_path.open("a") as f:
        f.write(
            f"| {args.round} | {score.schema_recall:.1%} | {score.schema_precision:.1%} | "
            f"{score.invariant_recall:.1%} | {score.invariant_precision:.1%} | "
            f"{wall:.0f} | model={args.model}, ctx={args.num_ctx} |\n"
        )

    print(
        f"[calibration] round {args.round}: "
        f"schema_recall={score.schema_recall:.1%} "
        f"schema_precision={score.schema_precision:.1%} "
        f"inv_recall={score.invariant_recall:.1%} "
        f"inv_precision={score.invariant_precision:.1%} "
        f"wall={wall:.0f}s energy={energy_wh:.2f}Wh",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
