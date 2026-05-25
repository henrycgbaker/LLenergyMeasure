"""Phase 2.5 multi-pass calibration runner for strategy (b) on transformers v4_57_3.

Runs ONLY the invariants multi-pass pipeline (skipping schema re-extraction
since the round-3 schema output is already locked at 83.0% recall and
nothing changed there). Reuses the existing schema.json artefact from
the round-3 trial run.

Scores with the FIXED rubric (4-tuple identity including secondary_field).

Usage::

    uv run python -m research/mining-substrate-trial/scripts.strategies.run_multipass_calibration \\
        --out-dir research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/phase2_5 \\
        --schema-source research/mining-substrate-trial/findings/trial_runs/b/transformers/v4_57_3/calibration_round_3_schema.json

Outputs:
- ``<out_dir>/invariants.proposed.yaml`` (multi-pass output)
- ``<out_dir>/schema.json``              (copied from --schema-source)
- ``<out_dir>/raw_llm_transcripts/*.md``
- ``<out_dir>/phase2_5_score.json``
- ``<out_dir>/phase2_5_report.md``
- ``<out_dir>/phase2_5_invariants_recall_misses.yaml``
- ``<out_dir>/phase2_5_invariants_precision_spurious.yaml``
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
import yaml  # noqa: E402

import trial_scoring  # noqa: E402
from strategies.llm_b_oss import extract_invariants_multipass  # noqa: E402
from strategies.llm_extractor import OllamaBackend  # noqa: E402
from strategies.transformers_chunker import invariants_chunks  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Per-cell run directory (will be created).",
    )
    parser.add_argument(
        "--schema-source",
        type=Path,
        required=True,
        help="Path to existing schema.json (e.g. calibration_round_3_schema.json).",
    )
    parser.add_argument("--model", default="llama3.1:70b", help="Ollama model tag.")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11435",
        help="Ollama HTTP base URL.",
    )
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--engine-version", default="4.57.3")
    parser.add_argument("--no-verify", action="store_true", help="Skip pass 2 (verify).")
    parser.add_argument("--no-extend", action="store_true", help="Skip pass 3 (extend).")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    transcripts_dir = out_dir / "raw_llm_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    # Copy schema artefact
    schema_target = out_dir / "schema.json"
    if not args.schema_source.exists():
        print(f"ERROR: schema source not found: {args.schema_source}", file=sys.stderr)
        return 1
    shutil.copy2(args.schema_source, schema_target)
    print(f"[phase2_5] copied schema from {args.schema_source} -> {schema_target}", file=sys.stderr)

    backend = OllamaBackend(model=args.model, url=args.ollama_url, num_ctx=args.num_ctx)

    chunks = invariants_chunks()
    print(f"[phase2_5] invariants chunks: {len(chunks)}", file=sys.stderr)

    # Energy
    energy_wh = 0.0
    try:
        from llenergymeasure.energy import select_energy_sampler

        sampler = select_energy_sampler("auto")
    except Exception as exc:
        print(f"[phase2_5] energy sampler unavailable: {exc}", file=sys.stderr)
        sampler = None
    tracker = None
    if sampler is not None:
        try:
            tracker = sampler.start_tracking()
        except Exception as exc:
            print(f"[phase2_5] energy tracker failed: {exc}", file=sys.stderr)
            tracker = None

    t0 = time.perf_counter()
    try:
        envelope, inv_wall, observations = extract_invariants_multipass(
            backend=backend,
            chunks=chunks,
            engine="transformers",
            engine_version=args.engine_version,
            transcripts_dir=transcripts_dir,
            max_retries=args.max_retries,
            enable_verify=not args.no_verify,
            enable_extend=not args.no_extend,
        )
    except Exception as exc:
        wall = time.perf_counter() - t0
        if sampler is not None and tracker is not None:
            try:
                meas = sampler.stop_tracking(tracker)
                if hasattr(meas, "total_j"):
                    energy_wh = meas.total_j / 3600.0
            except Exception:
                pass
        print(
            f"[phase2_5] multipass CRASHED after {wall:.1f}s, energy {energy_wh:.2f}Wh: "
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
            print(f"[phase2_5] energy stop failed: {exc}", file=sys.stderr)

    # Write invariants output
    inv_path = out_dir / "invariants.proposed.yaml"
    inv_path.write_text(yaml.safe_dump(envelope, sort_keys=False, default_flow_style=False))
    print(
        f"[phase2_5] wrote {inv_path} ({len(envelope['invariants'])} invariants)", file=sys.stderr
    )

    # Score with the FIXED rubric
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
        schema_output_path=schema_target,
        invariants_output_path=inv_path,
        reference_schema_path=ref_schema,
        reference_invariants_path=ref_inv,
        wall_clock_sec=wall,
        energy_wh=energy_wh,
        extra_observations=observations,
    )

    score_path = out_dir / "phase2_5_score.json"
    score_path.write_text(score.to_json())

    # Per-item diffs
    trial_scoring.write_diff_artefact(
        diffs=[d for d in inv_diffs if d.kind == "recall_miss"],
        out_path=out_dir / "phase2_5_invariants_recall_misses.yaml",
        kind_label="recall_misses",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in inv_diffs if d.kind == "precision_spurious"],
        out_path=out_dir / "phase2_5_invariants_precision_spurious.yaml",
        kind_label="precision_spurious",
    )
    trial_scoring.write_diff_artefact(
        diffs=type_or_sev_diffs,
        out_path=out_dir / "phase2_5_type_or_severity_mismatches.yaml",
        kind_label="type_or_severity_mismatches",
    )

    # Report
    report = [
        "# Phase 2.5 multi-pass calibration - transformers v4_57_3 (b)",
        "",
        f"- model: `{args.model}` (num_ctx={args.num_ctx})",
        f"- multi-pass: verify={'on' if not args.no_verify else 'off'} "
        f"extend={'on' if not args.no_extend else 'off'}",
        f"- invariants_wall: {inv_wall:.1f}s (total wall {wall:.1f}s)",
        f"- energy: {energy_wh:.2f} Wh",
        "",
        "## Schema (carried over from round 3 - unchanged)",
        f"- recall: **{score.schema_recall:.1%}** ({score.schema_intersection_count}/{score.schema_reference_count})",
        f"- precision: {score.schema_precision:.1%}",
        f"- type_accuracy: {score.schema_type_accuracy:.1%}",
        "",
        "## Invariants (multi-pass + Phase 2.5 rubric fix)",
        f"- recall: **{score.invariant_recall:.1%}** ({score.invariant_intersection_count}/{score.invariant_reference_count})",
        f"- precision: {score.invariant_precision:.1%} ({score.invariant_intersection_count}/{score.invariant_cell_count})",
        f"- severity_accuracy: {score.invariant_severity_accuracy:.1%}",
        f"- failure_mode: {score.invariant_failure_mode}",
        "",
        "## Observations",
        "",
    ]
    for obs in observations:
        report.append(f"- {obs}")
    (out_dir / "phase2_5_report.md").write_text("\n".join(report))

    print(
        f"[phase2_5] DONE: "
        f"inv_recall={score.invariant_recall:.1%} ({score.invariant_intersection_count}/{score.invariant_reference_count}) "
        f"inv_precision={score.invariant_precision:.1%} "
        f"severity={score.invariant_severity_accuracy:.1%} "
        f"wall={wall:.0f}s energy={energy_wh:.2f}Wh",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
