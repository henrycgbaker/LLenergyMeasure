"""Wave 2 runner: thin dispatcher over scripts/strategies/wave2/ modules.

Mirrors the contract of `trial_runner.run_cell` so Wave 2 cells score
through the same `trial_scoring.score_cell` path and drop records into
the same `findings/trial_scores/` tree. Differs only in dispatch:
Wave 2 strategies live under `strategies.wave2.*` and use the
`_base.CellOutput` interface instead of the original
`StrategyOutputPaths`.

Layout:
- Wave 2 score records: `findings/trial_scores/wave2/<strategy_id>__<engine>__<version_slug>.json`
- Wave 2 cell artefacts: `findings/trial_runs/wave2/<strategy_id>/<engine>/<version_slug>/`

CLI shape:

    python research/mining-substrate-trial/scripts/wave2_runner.py \\
        --strategy w2-a-pydantic-native --engine vllm --version v0.7.3

    python research/mining-substrate-trial/scripts/wave2_runner.py --list

Smoke-test discipline: this script intentionally does NOT auto-build
venvs or pull models. Infrastructure setup is per
`WAVE2_INFRA_SETUP.md`.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_TRIAL_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_TRIAL_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRIAL_SCRIPTS_DIR))

import trial_scoring  # noqa: E402
from strategies.wave2._base import CellOutput, write_cell_record  # noqa: E402

TRIAL_SCORES_ROOT = (
    _PROJECT_ROOT / "research" / "mining-substrate-trial" / "findings" / "trial_scores" / "wave2"
)
ENGINE_VERSIONS_ROOT = _PROJECT_ROOT / "engine_versions"


@dataclass(frozen=True)
class Wave2StrategyEntry:
    """Registry entry for one Wave 2 strategy.

    `module` is the import path under `strategies.wave2`; the module
    must expose `run_cell` matching the contract documented in each
    strategy's docstring.
    """

    strategy_id: str
    module: str
    engines: tuple[str, ...]
    tier: str  # "A" | "B" | "C"
    notes: str


WAVE2_REGISTRY: dict[str, Wave2StrategyEntry] = {
    "w2-a-pydantic-native": Wave2StrategyEntry(
        strategy_id="w2-a-pydantic-native",
        module="strategies.wave2.a_pydantic_native",
        engines=("vllm",),
        tier="A",
        notes="Framework-native reflection (dataclass / msgspec / Pydantic). vllm v0.7.3 pilot.",
    ),
    "w2-a-runtime-trace": Wave2StrategyEntry(
        strategy_id="w2-a-runtime-trace",
        module="strategies.wave2.a_runtime_trace",
        engines=("vllm",),
        tier="A",
        notes="Monkey-patch + perturbation invariant discovery. vllm v0.7.3 pilot.",
    ),
    "w2-a-treesitter": Wave2StrategyEntry(
        strategy_id="w2-a-treesitter",
        module="strategies.wave2.a_treesitter",
        engines=("transformers", "vllm", "tensorrt-llm"),
        tier="A",
        notes="Universal tree-sitter query walker. NOT YET IMPLEMENTED (raises NotImplementedError).",
    ),
    "w2-h15-closed-loop": Wave2StrategyEntry(
        strategy_id="w2-h15-closed-loop",
        module="strategies.wave2.h15_closed_loop",
        engines=("transformers",),
        tier="B",
        notes="Closed-loop H3+H2 hybrid. LLM dispatch wiring deferred (no Ollama).",
    ),
    "w2-h11-self-consistency": Wave2StrategyEntry(
        strategy_id="w2-h11-self-consistency",
        module="strategies.wave2.h11_self_consistency",
        engines=("transformers", "vllm", "tensorrt-llm"),
        tier="B",
        notes="k=3 vote on small-LLM (b). LLM dispatch wiring deferred.",
    ),
    "w2-b-stub-bench": Wave2StrategyEntry(
        strategy_id="w2-b-stub-bench",
        module="strategies.wave2.b_stub_bench",
        engines=("transformers",),
        tier="C",
        notes="Single-cell pyright-stub substrate benchmark. LLM dispatch deferred.",
    ),
    "w2-b-tree-bench": Wave2StrategyEntry(
        strategy_id="w2-b-tree-bench",
        module="strategies.wave2.b_tree_bench",
        engines=("transformers",),
        tier="C",
        notes="Single-cell tree-sitter-AST substrate benchmark. LLM dispatch deferred.",
    ),
}


def _resolve_run_cell(entry: Wave2StrategyEntry) -> Callable[..., CellOutput]:
    """Import the strategy module and return its run_cell callable.

    Raises ImportError with a clear message if the module can't load
    (likely cause: a missing optional dep like tree-sitter or vllm).
    """
    module = importlib.import_module(entry.module)
    run_cell = getattr(module, "run_cell", None)
    if run_cell is None or not callable(run_cell):
        raise AttributeError(
            f"strategy {entry.strategy_id!r} module {entry.module!r} "
            "does not expose a callable run_cell"
        )
    return run_cell


def _version_slug(version: str) -> str:
    """Normalise a version string to the canonical engine_versions slug.

    Accepts both `4.57.3` and `v4_57_3`; emits `v4_57_3`.
    """
    if version.startswith("v") and "." not in version:
        return version
    cleaned = version.lstrip("v").replace(".", "_")
    return f"v{cleaned}"


def _resolve_engine_source_root(engine: str, version_slug: str) -> Path | None:
    """Locate the unpacked engine source, if available.

    Returns the canonical /tmp/trial_<engine>_<version_slug>_venv/src/<engine>/
    path when venv_setup has materialised it; None otherwise. Strategies
    that need source (a_treesitter, b_stub_bench, b_tree_bench) accept
    this kwarg; the runner filters it out for strategies that don't.
    """
    candidates = [
        Path(f"/tmp/trial_{engine}_{version_slug}_venv/src/{engine}"),
        Path(f"/tmp/{engine}-unpacked/{engine}"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _reference_paths(engine: str, version_slug: str) -> tuple[Path, Path]:
    """Resolve canonical reference schema + invariants paths for a cell."""
    outputs_dir = ENGINE_VERSIONS_ROOT / engine / version_slug / "outputs"
    schema_path = outputs_dir / "schema.discovered.json"
    invariants_path = outputs_dir / "invariants.validated.yaml"
    if not invariants_path.exists():
        invariants_path = outputs_dir / "invariants.proposed.yaml"
    return schema_path, invariants_path


def run_one(
    *,
    strategy_id: str,
    engine: str,
    version: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one Wave 2 cell end-to-end and write its score record."""
    if strategy_id not in WAVE2_REGISTRY:
        raise KeyError(f"unknown wave2 strategy {strategy_id!r}; known: {list(WAVE2_REGISTRY)}")
    entry = WAVE2_REGISTRY[strategy_id]
    if engine not in entry.engines:
        raise ValueError(
            f"strategy {strategy_id!r} does not support engine {engine!r}; "
            f"supported: {entry.engines}"
        )

    version_slug = _version_slug(version)
    run_cell = _resolve_run_cell(entry)
    extras = extra_kwargs or {}

    # Auto-resolve common per-strategy kwargs from the cell context.
    # Strategies declare which they accept via run_cell's signature;
    # we filter against that so unsupported kwargs don't TypeError.
    auto_kwargs = {
        "engine_source_root": _resolve_engine_source_root(engine, version_slug),
    }
    sig_params = inspect.signature(run_cell).parameters
    forwarded = {
        k: v for k, v in {**auto_kwargs, **extras}.items() if k in sig_params and v is not None
    }

    t0 = time.perf_counter()
    try:
        output: CellOutput = run_cell(engine=engine, version=version, **forwarded)
    except NotImplementedError as exc:
        return _emit_deferred_record(
            strategy_id=strategy_id,
            engine=engine,
            version_slug=version_slug,
            reason=str(exc),
        )
    except Exception as exc:
        return _emit_crash_record(
            strategy_id=strategy_id,
            engine=engine,
            version_slug=version_slug,
            exc=exc,
        )

    wall_total = time.perf_counter() - t0
    write_cell_record(output)

    schema_ref, invariants_ref = _reference_paths(engine, version_slug)
    if not schema_ref.exists() or not invariants_ref.exists():
        return {
            "status": "reference_missing",
            "strategy_id": strategy_id,
            "engine": engine,
            "version_slug": version_slug,
            "schema_ref": str(schema_ref),
            "invariants_ref": str(invariants_ref),
        }

    score, schema_diffs, inv_diffs, type_or_sev_diffs = trial_scoring.score_cell(
        strategy=strategy_id,
        engine=engine,
        version_slug=version_slug,
        bump_distance="active",
        schema_output_path=output.schema_path,
        invariants_output_path=output.invariants_path,
        reference_schema_path=schema_ref,
        reference_invariants_path=invariants_ref,
        wall_clock_sec=output.wall_sec.get("total", wall_total),
        energy_wh=output.energy_wh or 0.0,
        extra_observations=output.observations,
    )

    TRIAL_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    score_path = TRIAL_SCORES_ROOT / f"{strategy_id}__{engine}__{version_slug}.json"
    score_path.write_text(score.to_json())

    return {
        "status": "scored",
        "strategy_id": strategy_id,
        "engine": engine,
        "version_slug": version_slug,
        "score_path": str(score_path),
        "schema_recall": score.schema_recall,
        "invariant_recall": score.invariant_recall,
        "schema_precision": score.schema_precision,
        "invariant_precision": score.invariant_precision,
        "wall_sec": output.wall_sec,
    }


def _emit_deferred_record(
    *,
    strategy_id: str,
    engine: str,
    version_slug: str,
    reason: str,
) -> dict[str, Any]:
    """Persist a 'deferred' marker so the aggregator sees the cell tried."""
    TRIAL_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "deferred",
        "strategy_id": strategy_id,
        "engine": engine,
        "version_slug": version_slug,
        "reason": reason,
    }
    path = TRIAL_SCORES_ROOT / f"{strategy_id}__{engine}__{version_slug}.deferred.json"
    path.write_text(json.dumps(record, indent=2))
    return record


def _emit_crash_record(
    *,
    strategy_id: str,
    engine: str,
    version_slug: str,
    exc: BaseException,
) -> dict[str, Any]:
    """Persist a 'crashed' marker so failures are visible in the matrix."""
    TRIAL_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "status": "crashed",
        "strategy_id": strategy_id,
        "engine": engine,
        "version_slug": version_slug,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "traceback": traceback.format_exception(type(exc), exc, exc.__traceback__),
    }
    path = TRIAL_SCORES_ROOT / f"{strategy_id}__{engine}__{version_slug}.crashed.json"
    path.write_text(json.dumps(record, indent=2))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Wave 2 mining-substrate trial runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--strategy", help="strategy id from the wave2 registry")
    parser.add_argument("--engine", help="engine name (transformers / vllm / tensorrt-llm)")
    parser.add_argument("--version", help="engine version (e.g. v0.7.3)")
    parser.add_argument(
        "--list",
        action="store_true",
        help="list registered wave2 strategies and exit",
    )
    args = parser.parse_args(argv)

    if args.list:
        for entry in WAVE2_REGISTRY.values():
            print(
                f"{entry.strategy_id:<28} tier {entry.tier}  "
                f"engines={','.join(entry.engines):<35} {entry.notes}"
            )
        return 0

    if not (args.strategy and args.engine and args.version):
        parser.error("--strategy + --engine + --version are required unless --list is given")

    result = run_one(
        strategy_id=args.strategy,
        engine=args.engine,
        version=args.version,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"scored", "deferred"} else 1


if __name__ == "__main__":
    sys.exit(main())
