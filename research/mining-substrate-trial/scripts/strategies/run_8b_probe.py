"""M5 8B vs 70B sub-probe runner.

Runs the SAME final calibrated prompts through llama3.1:8b on the
SAME transformers v4_57_3 cell. Measures recall delta, wall-clock
delta, energy delta.

Per plan: "If 8B is 80%+ quality at 10x speed, that materially
changes (b)'s economics for the full matrix."

Output: ``research/mining-substrate-trial/findings/trial_runs/b_8b/transformers/v4_57_3/`` -
parallel to the 70B b/ cell so the scoring + aggregator can compare.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
import trial_scoring  # noqa: E402
from strategies.llm_b_oss import run_b_on_transformers_active  # noqa: E402
from strategies.llm_extractor import OllamaBackend  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_PROJECT_ROOT / "research" / "mining-substrate-trial"
        / "findings"
        / "trial_runs"
        / "b_8b"
        / "transformers"
        / "v4_57_3",
        help="Per-cell run directory.",
    )
    parser.add_argument("--model", default="llama3.1:8b", help="Ollama model tag.")
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11435",
        help="Ollama HTTP base URL.",
    )
    parser.add_argument("--num-ctx", type=int, default=32768, help="Ollama num_ctx.")
    parser.add_argument("--max-retries", type=int, default=2, help="Per-chunk retry budget.")
    args = parser.parse_args(argv)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = OllamaBackend(model=args.model, url=args.ollama_url, num_ctx=args.num_ctx)

    # Energy sampling
    energy_wh = 0.0
    try:
        from llenergymeasure.energy import select_energy_sampler

        sampler = select_energy_sampler("auto")
    except Exception:
        sampler = None
    tracker = None
    if sampler is not None:
        try:
            tracker = sampler.start_tracking()
        except Exception:
            tracker = None

    t0 = time.perf_counter()
    outputs = run_b_on_transformers_active(
        out_dir=out_dir,
        engine_version="4.57.3",
        backend=backend,
        max_retries=args.max_retries,
    )
    wall = time.perf_counter() - t0
    if sampler is not None and tracker is not None:
        try:
            meas = sampler.stop_tracking(tracker)
            if hasattr(meas, "total_j"):
                energy_wh = float(meas.total_j) / 3600.0
        except Exception:
            pass

    ref_schema = (
        _PROJECT_ROOT / "engine_versions/transformers/v4_57_3/outputs/schema.discovered.json"
    )
    ref_inv = (
        _PROJECT_ROOT / "engine_versions/transformers/v4_57_3/outputs/invariants.proposed.yaml"
    )

    score, schema_diffs, inv_diffs, type_or_sev_diffs = trial_scoring.score_cell(
        strategy="b_8b",
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

    score_path = out_dir / "score.json"
    score_path.write_text(score.to_json())

    print(
        f"[8b-probe] model={args.model} "
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
