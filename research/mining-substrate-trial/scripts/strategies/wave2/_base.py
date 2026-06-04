"""Shared abstractions for Wave 2 strategy modules.

All Wave 2 strategies return a uniform ``CellOutput`` so the Wave 2 runner
can dispatch + score them without per-strategy glue. Mirrors the Wave 1
``StrategyBOutputs`` shape but generalised.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@dataclass
class CellOutput:
    """Uniform per-cell output across all Wave 2 strategies.

    The schema_path / invariants_path point at the canonical artefact
    files the scorer reads. observations is free-text per-cell notes
    (failure modes, anomalies). wall_sec breaks down per-phase wall
    clock so the aggregator can attribute cost.
    """

    strategy_id: str
    engine: str
    engine_version: str
    schema_path: Path | None
    invariants_path: Path
    observations: list[str] = field(default_factory=list)
    wall_sec: dict[str, float] = field(default_factory=dict)
    energy_wh: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "schema_path": str(self.schema_path) if self.schema_path else None,
            "invariants_path": str(self.invariants_path),
            "observations": self.observations,
            "wall_sec": self.wall_sec,
            "energy_wh": self.energy_wh,
            "extra": self.extra,
        }


def standard_out_dir(strategy_id: str, engine: str, version_slug: str) -> Path:
    """Canonical Wave 2 output location.

    research/mining-substrate-trial/findings/trial_runs/wave2/<strategy>/<engine>/<version>/
    """
    base = (
        _PROJECT_ROOT
        / "research"
        / "mining-substrate-trial"
        / "findings"
        / "trial_runs"
        / "wave2"
        / strategy_id
        / engine
        / version_slug
    )
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_cell_record(output: CellOutput) -> Path:
    """Drop a per-cell record JSON into trial_scores/wave2/.

    Mirrors the Wave 1 trial_scores/ layout so the aggregator can glob
    across both waves.
    """
    scores_dir = (
        _PROJECT_ROOT
        / "research"
        / "mining-substrate-trial"
        / "findings"
        / "trial_scores"
        / "wave2"
    )
    scores_dir.mkdir(parents=True, exist_ok=True)
    version_slug = output.engine_version.replace(".", "_")
    record_path = scores_dir / f"{output.strategy_id}__{output.engine}__{version_slug}.json"
    record_path.write_text(json.dumps(output.to_dict(), indent=2, default=str))
    return record_path
