"""Wave 2c hybrid H11: self-consistency on (b).

Pattern: run the standard (b) extraction k=5 times at temperature 0.7,
take entries appearing in >= 3/5 runs as the final output. Tests whether
sampling-then-vote denoises q4 LLM stochasticity.

Hypothesis: variance in (b)'s output is partly stochastic at the q4 model
scale; aggregating across runs filters that variance. If the resulting
recall lifts >5pp without precision loss, self-consistency becomes a
cheap precision-boosting wrapper.

Cost: 5x the (b) wall-clock per cell. Worth it if recall lifts.
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from strategies.wave2._base import CellOutput, standard_out_dir

K_RUNS = 5
TEMPERATURE = 0.7
VOTE_THRESHOLD = 3  # entries appearing in >= 3 of 5 runs


def _load_invariants(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict) and "invariants" in data:
        return data["invariants"] or []
    if isinstance(data, list):
        return data
    return []


def _invariant_identity(inv: dict[str, Any]) -> tuple[str, str, str, str]:
    """Four-tuple identity per Wave 1 scoring rubric.

    Matches `trial_scoring._invariant_identity`.
    """
    return (
        inv.get("namespace", ""),
        inv.get("native_field", ""),
        inv.get("predicate_kind", ""),
        inv.get("secondary_field", ""),
    )


def _vote_aggregate(
    runs: list[list[dict[str, Any]]],
    *,
    threshold: int = VOTE_THRESHOLD,
) -> list[dict[str, Any]]:
    """Take entries appearing in >= threshold runs as the final output.

    When multiple runs contribute different bodies for the same identity,
    the body from the earliest run wins (run order is deterministic in
    the runner). Acceptable for v1; can be revised to "longest body" or
    "consensus body" if needed.
    """
    counter: Counter[tuple[str, str, str, str]] = Counter()
    bodies: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for run in runs:
        seen_in_run: set[tuple[str, str, str, str]] = set()
        for inv in run:
            ident = _invariant_identity(inv)
            if ident in seen_in_run:
                continue
            seen_in_run.add(ident)
            counter[ident] += 1
            bodies.setdefault(ident, inv)
    return [bodies[ident] for ident, count in counter.items() if count >= threshold]


def run_cell(
    *,
    engine: str,
    version: str,
    ollama_host: str = "http://localhost:11435",
    model: str = "llama3.1:70b",
    k: int = K_RUNS,
    temperature: float = TEMPERATURE,
    threshold: int = VOTE_THRESHOLD,
) -> CellOutput:
    """Execute one Wave 2c H11 cell.

    Runs the (b) pipeline k times with non-zero temperature, then
    vote-aggregates the per-run outputs. Records per-run paths for
    post-hoc analysis (variance-per-cell).
    """
    strategy_id = "w2-h11-self-consistency"
    version_slug = version.replace(".", "_").lstrip("v")
    if not version_slug.startswith("v"):
        version_slug = f"v{version_slug}"
    out_dir = standard_out_dir(strategy_id, engine, version_slug)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    per_run_outputs: list[list[dict[str, Any]]] = []
    per_run_wall: list[float] = []

    for run_idx in range(k):
        run_t0 = time.perf_counter()
        run_dir = runs_dir / f"run_{run_idx:02d}"
        run_dir.mkdir(exist_ok=True)
        # TODO: dispatch to llm_b_oss.run_b_pipeline with temperature override.
        # Per-run output drops into run_dir/invariants.proposed.yaml.
        # Implementation deferred per WAVE2_PROTOCOL section 4 step 12.
        raise NotImplementedError(
            "wave2 H11 self-consistency: call llm_b_oss.run_b_pipeline "
            f"with temperature={temperature}, seed=run_idx, "
            "output to run_dir. Scaffolded; not yet implemented."
        )
        # noqa: unreachable
        per_run_invs = _load_invariants(run_dir / "invariants.proposed.yaml")
        per_run_outputs.append(per_run_invs)
        per_run_wall.append(time.perf_counter() - run_t0)

    voted = _vote_aggregate(per_run_outputs, threshold=threshold)

    out_invariants = out_dir / "invariants.proposed.yaml"
    out_invariants.write_text(yaml.safe_dump({"invariants": voted}, sort_keys=False))

    return CellOutput(
        strategy_id=strategy_id,
        engine=engine,
        engine_version=version,
        schema_path=None,  # H11 is invariant-only; schema reuses (b)'s
        invariants_path=out_invariants,
        observations=[
            f"k={k} temperature={temperature} threshold={threshold}",
            f"per_run_counts={[len(r) for r in per_run_outputs]}",
            f"voted_count={len(voted)}",
        ],
        wall_sec={
            "total": time.perf_counter() - t0,
            "per_run_mean": sum(per_run_wall) / max(len(per_run_wall), 1),
        },
        extra={"per_run_counts": [len(r) for r in per_run_outputs]},
    )
