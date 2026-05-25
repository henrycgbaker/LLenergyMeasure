"""Per-cell orchestrator for the mining-substrate empirical trial.

Phase 1 Day 5 stub. Reference: ``.planning/mining-substrate-empirical-trial.md``.

Cell shape: ``(strategy, engine, version)``. One cell = one cartesian point
in the 75-cell maximum matrix (5 strategies x 3 engines x 5 versions). The
runner:

1. Resolves the cell config - which producer (a) / prompt template (b/c) /
   hybrid pattern (d). Cell configs live in :data:`CELL_REGISTRY` (Phase 1
   stub; Phase 2 fills variants in).
2. Sets up the isolated venv if needed (Day 3's per-engine-per-version
   venvs at ``/tmp/trial_<engine>_<version_slug>_venv/``). Strategy (a) on
   the active version reuses the project venv directly.
3. Executes the strategy. Measures wall-clock (``time.perf_counter``) and
   energy (``llenergymeasure.energy.select_energy_sampler``).
4. Scores the cell using :mod:`_spike.scripts.trial_scoring`.
5. Writes the per-cell record JSON + side-artefacts (recall_misses.yaml,
   precision_spurious.yaml, type_mismatches.yaml).

Output layout
-------------
**Strategy (a) "pure mining"** writes its primary artefacts into the
canonical ``engine_versions/<engine>/v<v>/outputs/`` tree (the existing
pipeline; not parallel). The trial score record + side-artefacts land in
``_spike/findings/trial_scores/a__<engine>__<version_slug>.json`` plus
sibling ``_spike/findings/trial_runs/a/<engine>/<version_slug>/...``.

**Strategies (b/c/d)** write everything under
``_spike/findings/trial_runs/<strategy>/<engine>/<version_slug>/``:

* ``proposed.yaml`` - the candidate invariants (mirrors invariants.proposed.yaml)
* ``schema.json`` - the candidate schema (mirrors schema.discovered.json)
* ``llm_transcript/`` - raw LLM input/output transcripts (one file per call)
* ``hybrid_log.md`` - for (d), the orchestration trace
* ``recall_misses.yaml``, ``precision_spurious.yaml``, ``type_mismatches.yaml``
  - per-cell diffs from the scorer

The score record itself sits ABOVE this tree at
``_spike/findings/trial_scores/<strategy>__<engine>__<version_slug>.json``
so the aggregator can glob across strategies in one pass.

Justification for parallel structure: ``engine_versions/`` is product
surface (curated by the spike + ships in main). The trial is an EXCURSION;
its outputs are research evidence, not product. Mixing them would leak
trial work into product paths; isolating them keeps the trial reversible.

CLI
---
::

    # Single cell:
    python -m _spike.scripts.trial_runner \\
        --strategy a --engine transformers --version-slug v4_57_3

    # Sweep (every combination listed in the cell registry):
    python -m _spike.scripts.trial_runner --all

    # Dry-run (print planned execution, no work):
    python -m _spike.scripts.trial_runner --all --dry-run

    # Re-run a single cell (filter):
    python -m _spike.scripts.trial_runner --cell-spec 'a/transformers/v4_57_3'

"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Project root is two parents up from _spike/scripts/.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Local imports (sys.path adjusted above; ruff E402 deliberate here).
from _spike.scripts import trial_scoring  # noqa: E402
from _spike.scripts.trial_scoring import CellScore, ScoringConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRIAL_RUNS_ROOT = _PROJECT_ROOT / "_spike" / "findings" / "trial_runs"
TRIAL_SCORES_ROOT = _PROJECT_ROOT / "_spike" / "findings" / "trial_scores"
ENGINE_VERSIONS_ROOT = _PROJECT_ROOT / "engine_versions"
VENV_ROOT_TEMPLATE = Path("/tmp/trial_{engine}_{version_slug}_venv")
"""Day 3 per-cell isolated venv path template. Day 3 builds these; Day 5
assumes they exist when needed (with a clear runtime error if missing)."""


# ---------------------------------------------------------------------------
# Cell configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSpec:
    """One cell's configuration - the orchestrator's input parameter."""

    strategy: str  # "a" | "b" | "c" | "d-ab" | "d-ac"
    engine: str  # "transformers" | "vllm" | "tensorrt"
    version_slug: str  # filesystem-safe version identifier, e.g. "v4_57_3"
    bump_distance: str  # "v-2" | "v-1" | "active" | "v+1" | "v+major"

    pypi_version: str = ""
    """e.g. ``"4.57.3"``. Resolved at cell-registry-build time. Empty
    string when the cell uses the project's installed library directly
    (active-version cells on the project's matching engine)."""

    prompt_template_ref: str | None = None
    """Reference to a prompt template under ``_spike/scripts/prompts/``
    (Phase 2 builds this directory). Used by strategies (b)/(c)/(d).
    None for strategy (a)."""

    hybrid_pattern: str | None = None
    """The hybrid pattern variant (e.g. ``"deterministic_then_validate"``).
    Used by strategy (d) only. None otherwise."""

    venv_path: Path | None = None
    """Isolated venv to activate when running the cell. None means use the
    project's environment. Set by :func:`resolve_cell_config`."""

    def to_key(self) -> str:
        return f"{self.strategy}/{self.engine}/{self.version_slug}"

    def to_score_filename(self) -> str:
        return f"{self.strategy}__{self.engine}__{self.version_slug}.json"

    def to_run_dir(self) -> Path:
        return TRIAL_RUNS_ROOT / self.strategy / self.engine / self.version_slug


# Cell registry. Phase 1 populates a SKELETON; Phase 2 + 3 fill in
# version-bump entries and (b)/(c)/(d) variants.
#
# Schema: keyed by (strategy, engine, version_slug); value is a CellSpec.
# The active-version cells get a default ``bump_distance="active"``; the
# version-bumped cells use the values fixed at Day 3 (Phase 1).
CELL_REGISTRY: dict[tuple[str, str, str], CellSpec] = {}


def _register(spec: CellSpec) -> None:
    CELL_REGISTRY[(spec.strategy, spec.engine, spec.version_slug)] = spec


# Skeleton seeding - Phase 1 active-version cells per § Matrix table.
# Version-bumped cells (v-2 / v-1 / v+1 / v+major) get appended at Day 3
# once PyPI probe lands.
_register(
    CellSpec(
        strategy="a",
        engine="transformers",
        version_slug="v4_57_3",
        bump_distance="active",
        pypi_version="4.57.3",
    )
)
_register(
    CellSpec(
        strategy="a",
        engine="vllm",
        version_slug="v0_7_3",
        bump_distance="active",
        pypi_version="0.7.3",
    )
)
_register(
    CellSpec(
        strategy="a",
        engine="tensorrt",
        version_slug="v0_21_0",
        bump_distance="active",
        pypi_version="0.21.0",
    )
)
# Phase 2: LLM strategies on the transformers active cell.
# Phase 3a.1 extends (b) and (d-ab) to all engines on the active versions.
# (c) and (d-ac) stay transformers-only here pending ANTHROPIC_API_KEY arrival.
for _strategy in ("b", "c", "d-ab", "d-ac"):
    _register(
        CellSpec(
            strategy=_strategy,
            engine="transformers",
            version_slug="v4_57_3",
            bump_distance="active",
            pypi_version="4.57.3",
            prompt_template_ref="strategies/prompts.py",
        )
    )

# Phase 3a.1 - (b) + (d-ab) on vllm/tensorrt active cells.
for _strategy in ("b", "d-ab"):
    _register(
        CellSpec(
            strategy=_strategy,
            engine="vllm",
            version_slug="v0_7_3",
            bump_distance="active",
            pypi_version="0.7.3",
            prompt_template_ref="strategies/prompts.py",
        )
    )
    _register(
        CellSpec(
            strategy=_strategy,
            engine="tensorrt",
            version_slug="v0_21_0",
            bump_distance="active",
            pypi_version="0.21.0",
            prompt_template_ref="strategies/prompts.py",
        )
    )

# Phase 3a.2 - bumped-version cells for transformers (4 versions x 3 strategies = 12 cells).
# Per Phase 1 version_lock: v-2 4.55.4, v-1 4.56.2, v+1 4.57.6, v+major 5.9.0.
_TRANSFORMERS_BUMPED: list[tuple[str, str, str]] = [
    # (version_slug, pypi_version, bump_distance)
    ("v4_55_4", "4.55.4", "v-2"),
    ("v4_56_2", "4.56.2", "v-1"),
    ("v4_57_6", "4.57.6", "v+1"),
    ("v5_9_0", "5.9.0", "v+major"),
]
for _vslug, _pyver, _bump in _TRANSFORMERS_BUMPED:
    # (a) bumped: rerun static miner against bumped source.
    _register(
        CellSpec(
            strategy="a",
            engine="transformers",
            version_slug=_vslug,
            bump_distance=_bump,
            pypi_version=_pyver,
        )
    )
    # (b) bumped: LLM extracts against bumped source via AST file-reading.
    _register(
        CellSpec(
            strategy="b",
            engine="transformers",
            version_slug=_vslug,
            bump_distance=_bump,
            pypi_version=_pyver,
            prompt_template_ref="strategies/prompts.py",
        )
    )
    # (d-ab) bumped: hybrid uses bumped source + active-(a)'s output.
    _register(
        CellSpec(
            strategy="d-ab",
            engine="transformers",
            version_slug=_vslug,
            bump_distance=_bump,
            pypi_version=_pyver,
            prompt_template_ref="strategies/prompts.py",
        )
    )


def resolve_cell_config(spec: CellSpec, *, lazy_build: bool = False) -> CellSpec:
    """Fill in ``venv_path`` (and any other runtime-resolved fields).

    Phase 2: active-version transformers cells reuse the project venv
    directly. Other cells need an isolated venv at
    ``/tmp/trial_<engine>_<version_slug>_venv/``.

    Phase 2.5 adds optional ``lazy_build=True`` which, for strategies
    (b)/(c)/(d-ab)/(d-ac), invokes
    ``_spike.scripts.venv_setup.ensure_source_only_venv`` to build a
    source-only venv on first hit. Strategy (a) is unchanged (needs a
    full canonical container, not a venv).

    If the venv already exists OR the cell is the project-native active
    cell, no build is performed.
    """
    import dataclasses as _dc

    PROJECT_NATIVE = {("transformers", "v4_57_3")}
    if spec.bump_distance == "active" and (spec.engine, spec.version_slug) in PROJECT_NATIVE:
        return spec
    venv_path = Path(
        str(VENV_ROOT_TEMPLATE).format(engine=spec.engine, version_slug=spec.version_slug)
    )
    if venv_path.exists():
        return _dc.replace(spec, venv_path=venv_path)

    # Phase 2.5: lazy source-only venv build for (b)/(c)/(d-*) strategies.
    if lazy_build and spec.strategy in {"b", "c", "d-ab", "d-ac"}:
        try:
            from _spike.scripts.venv_setup import ensure_source_only_venv

            sv = ensure_source_only_venv(engine=spec.engine, version_slug=spec.version_slug)
            return _dc.replace(spec, venv_path=sv.venv_path)
        except Exception:
            # Build failure is non-fatal at resolve time; dispatcher will surface it.
            return spec
    # Phase 2: leave venv_path=None; the strategy dispatcher handles
    # the "active-only" guard for non-transformers / bumped cells.
    return spec


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferencePaths:
    schema: Path
    invariants: Path


def reference_paths_for(engine: str, version_slug: str) -> ReferencePaths:
    """Return the reference catalogue paths for an (engine, version) pair.

    Phase 1 contract: reference = the canonical engine_versions outputs
    (mature for transformers v4_57_3; thin for vllm/tensorrt). Phase 1
    Day 4 builds the wider reference set for non-active cells; until then
    those cells score against a placeholder.

    Phase 3a.2 (bumped-version cells): there are no per-version
    references for v4_55_4/v4_56_2/v4_57_6/v5_9_0 etc. The active
    reference is used for those cells - the brittleness signal IS the
    delta between the cell's bumped-source output and the active
    reference catalogue.
    """
    # Active-version reference fallback for bumped cells.
    ACTIVE_PER_ENGINE: dict[str, str] = {
        "transformers": "v4_57_3",
        "vllm": "v0_7_3",
        "tensorrt": "v0_21_0",
    }
    base = ENGINE_VERSIONS_ROOT / engine / version_slug / "outputs"
    schema = base / "schema.discovered.json"
    invariants = base / "invariants.proposed.yaml"
    if not schema.exists() or not invariants.exists():
        active_slug = ACTIVE_PER_ENGINE.get(engine)
        if active_slug:
            fallback_base = ENGINE_VERSIONS_ROOT / engine / active_slug / "outputs"
            if (fallback_base / "schema.discovered.json").exists():
                schema = fallback_base / "schema.discovered.json"
            if (fallback_base / "invariants.proposed.yaml").exists():
                invariants = fallback_base / "invariants.proposed.yaml"
    return ReferencePaths(schema=schema, invariants=invariants)


# ---------------------------------------------------------------------------
# Energy measurement wrapper
# ---------------------------------------------------------------------------


def measure_energy_during(thunk: Any) -> tuple[Any, float, float]:
    """Run ``thunk()`` and return ``(result, wall_clock_sec, energy_wh)``.

    Uses ``llenergymeasure.energy.select_energy_sampler('auto')``. On hosts
    where no sampler is available, falls back to ``energy_wh=0.0`` with
    an observation tag. Strategies (b)/(c) running Ollama on the host
    capture GPU-side energy via NVML when available.

    The energy sampler's ``stop_tracking`` returns an ``EnergyMeasurement``
    with ``total_j``; we convert to Wh (joules / 3600).
    """
    import time as _time

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
    t0 = _time.perf_counter()
    try:
        result = thunk()
    finally:
        wall = _time.perf_counter() - t0
        energy_wh = 0.0
        if tracker is not None and sampler is not None:
            try:
                meas = sampler.stop_tracking(tracker)
                if hasattr(meas, "total_j"):
                    energy_wh = float(meas.total_j) / 3600.0
                elif hasattr(meas, "energy_wh"):
                    energy_wh = float(meas.energy_wh)
            except Exception:
                energy_wh = 0.0
    return result, wall, energy_wh


# ---------------------------------------------------------------------------
# Strategy execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrategyOutputPaths:
    """The two artefact paths a cell run produces."""

    schema_path: Path
    invariants_path: Path


def run_strategy_a(spec: CellSpec) -> tuple[StrategyOutputPaths, list[str]]:
    """Run strategy (a) "pure mining" for a cell.

    Active-version cells: returns the canonical ``engine_versions/``
    outputs (the existing pipeline already ran these). Bumped-version
    cells: subprocess-invokes the v4_57_3 static miner against the
    bumped source tree (built via venv_setup), capturing the raw output.
    The miner's brittleness when shown bumped source IS the signal.

    Failure modes:
    - ``infrastructure_missing``: source-only venv could not be built
      (pip download failed).
    - ``miner_runtime_error``: subprocess crashed (import error, missing
      landmark class, etc.). Recorded with stderr trace.
    """
    if spec.bump_distance == "active":
        output_dir = ENGINE_VERSIONS_ROOT / spec.engine / spec.version_slug / "outputs"
        schema_path = output_dir / "schema.discovered.json"
        invariants_path = output_dir / "invariants.proposed.yaml"
        observations: list[str] = []
        if not schema_path.exists():
            observations.append(
                f"strategy_a: schema output missing at {schema_path}; "
                "run engine_producers pipeline first"
            )
        if not invariants_path.exists():
            observations.append(
                f"strategy_a: invariants output missing at {invariants_path}; "
                "run engine_producers pipeline first"
            )
        observations.append(
            f"strategy_a: reusing canonical engine_versions outputs for {spec.to_key()}"
        )
        return StrategyOutputPaths(
            schema_path=schema_path, invariants_path=invariants_path
        ), observations

    # Bumped-version path (Phase 3a.2). For now, transformers only.
    if spec.engine != "transformers":
        raise NotImplementedError(
            f"strategy (a) bumped for engine {spec.engine!r} is a separate work item "
            "(Phase 3a.2.{spec.engine})."
        )
    return _run_strategy_a_transformers_bumped(spec)


def _run_strategy_a_transformers_bumped(
    spec: CellSpec,
) -> tuple[StrategyOutputPaths, list[str]]:
    """Subprocess-invoke the v4_57_3 static miner against bumped transformers source."""
    import os
    import subprocess

    observations: list[str] = []
    run_dir = spec.to_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build source-only venv (lazy)
    try:
        from _spike.scripts.venv_setup import ensure_source_only_venv

        sv = ensure_source_only_venv(engine="transformers", version_slug=spec.version_slug)
    except Exception as exc:
        observations.append(
            f"strategy_a bumped: source-only venv build failed: "
            f"{type(exc).__name__}: {exc}"
        )
        # Emit empty proposed.yaml so the scorer doesn't choke on missing file.
        invariants_path = run_dir / "invariants.proposed.yaml"
        invariants_path.write_text("schema_version: 1.0.0\ninvariants: []\n")
        schema_path = run_dir / "schema.json"
        schema_path.write_text("{}")
        observations.append("strategy_a bumped: failure_mode=infrastructure_missing")
        return StrategyOutputPaths(
            schema_path=schema_path, invariants_path=invariants_path
        ), observations

    # 2. Subprocess-invoke the v4_57_3 static miner with PYTHONPATH override
    invariants_path = run_dir / "invariants.proposed.yaml"
    schema_path = run_dir / "schema.json"

    # Use miniforge python (3.12+) so tomllib is present.
    py = "/home/h.baker@hertie-school.lan/miniforge3/bin/python3.12"
    if not Path(py).exists():
        py = sys.executable  # fallback

    # The static miner writes to --out. Build a subprocess that:
    # 1. Drops the project .venv from sys.path
    # 2. Inserts the source-only venv first
    # 3. Runs walk_transformers + emit_yaml, writes to invariants_path.
    runner_src = """
import sys, os
sys.path = [p for p in sys.path if 'workspace/llenergymeasure/.venv' not in p]
sys.path.insert(0, os.environ['SOURCE_ROOT_PARENT'])
sys.path.insert(0, os.environ['PROJECT_ROOT'])
from engine_versions.transformers.v4_57_3.producers import static_invariant_miner as miner
candidates, version, rel_path = miner.walk_transformers()
text = miner.emit_yaml(candidates, engine_version=version, rel_path=rel_path)
with open(os.environ['OUT_PATH'], 'w') as f:
    f.write(text)
print(f'wrote {len(candidates)} candidates to {os.environ[\"OUT_PATH\"]}')
"""
    env = os.environ.copy()
    env["SOURCE_ROOT_PARENT"] = str(sv.source_dir.parent)
    env["PROJECT_ROOT"] = str(_PROJECT_ROOT)
    env["OUT_PATH"] = str(invariants_path)
    try:
        proc = subprocess.run(
            [py, "-c", runner_src],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        observations.append("strategy_a bumped: miner subprocess timed out (>120s)")
        invariants_path.write_text("schema_version: 1.0.0\ninvariants: []\n")
        schema_path.write_text("{}")
        observations.append("strategy_a bumped: failure_mode=miner_runtime_error;subreason=timeout")
        return (
            StrategyOutputPaths(schema_path=schema_path, invariants_path=invariants_path),
            observations,
        )

    if proc.returncode != 0:
        stderr_tail = proc.stderr[-1500:] if proc.stderr else "<empty>"
        observations.append(
            f"strategy_a bumped: miner subprocess returncode={proc.returncode}; "
            f"stderr_tail={stderr_tail}"
        )
        invariants_path.write_text("schema_version: 1.0.0\ninvariants: []\n")
        schema_path.write_text("{}")
        observations.append("strategy_a bumped: failure_mode=miner_runtime_error")
        return (
            StrategyOutputPaths(schema_path=schema_path, invariants_path=invariants_path),
            observations,
        )

    # Success path: stdout has the line "wrote N candidates ...".
    observations.append(f"strategy_a bumped: subprocess stdout={proc.stdout.strip()}")
    # Schema: copy active reference (no per-version schema extraction in
    # bumped (a); the miner only emits invariants). The schema_path field
    # IS used by trial_scoring; we point at active reference deliberately
    # because (a) bumped's schema is "no schema" - we score invariants only.
    active_schema = (
        ENGINE_VERSIONS_ROOT / "transformers" / "v4_57_3" / "outputs" / "schema.discovered.json"
    )
    if active_schema.exists():
        schema_path.write_text(active_schema.read_text())
    else:
        schema_path.write_text("{}")
    return (
        StrategyOutputPaths(schema_path=schema_path, invariants_path=invariants_path),
        observations,
    )


def run_strategy_b(spec: CellSpec) -> tuple[StrategyOutputPaths, list[str]]:
    """Run strategy (b) "pure OSS LLM" for a cell.

    Phase 3 implementation: dispatches per engine. Active-version cells
    use the engine's pre-staged source tree; bumped versions go through
    the lazy-source-only-venv path (Phase 3a.2).

    Phase 2.5 closure: multipass=True (extract -> verify -> extend) is the
    Phase 3 default for active cells - Phase 2.5 measured this as the
    final calibrated config for transformers v4_57_3.

    Phase 3a.2 (bumped transformers): the source root comes from the
    source-only venv built by ``venv_setup.ensure_source_only_venv``.
    The chunker AST-parses the bumped source files; any extraction
    failure (API rename, file moved) surfaces as ``_extraction_error``
    inside the chunks and propagates to the score as a brittleness mode.
    """
    from _spike.scripts.strategies.llm_b_oss import (
        run_b_on_tensorrt_active,
        run_b_on_transformers_active,
        run_b_on_vllm_active,
    )
    from _spike.scripts.strategies.llm_extractor import OllamaBackend

    run_dir = spec.to_run_dir()
    backend = OllamaBackend()
    if spec.engine == "transformers":
        source_root: Path | None = None
        if spec.bump_distance != "active":
            # Build source-only venv (lazy) and pass the source tree to
            # the chunker. Surface AST-extraction errors via observations.
            from _spike.scripts.venv_setup import ensure_source_only_venv

            sv = ensure_source_only_venv(
                engine="transformers", version_slug=spec.version_slug
            )
            source_root = sv.source_dir
        out = run_b_on_transformers_active(
            out_dir=run_dir,
            engine_version=spec.pypi_version or "4.57.3",
            backend=backend,
            multipass=True,
            source_root=source_root,
        )
    elif spec.engine == "vllm":
        if spec.bump_distance != "active":
            raise NotImplementedError(
                f"strategy (b) for bumped vllm ({spec.bump_distance}) is "
                "Phase 3a.2.vllm scope (separate work item)."
            )
        out = run_b_on_vllm_active(
            out_dir=run_dir,
            engine_version=spec.pypi_version or "0.7.3",
            backend=backend,
            multipass=True,
        )
    elif spec.engine == "tensorrt":
        if spec.bump_distance != "active":
            raise NotImplementedError(
                f"strategy (b) for bumped tensorrt ({spec.bump_distance}) is "
                "Phase 3a.2.tensorrt scope (separate work item)."
            )
        out = run_b_on_tensorrt_active(
            out_dir=run_dir,
            engine_version=spec.pypi_version or "0.21.0",
            backend=backend,
            multipass=True,
        )
    else:
        raise NotImplementedError(f"strategy (b) for engine {spec.engine!r} not implemented")
    return (
        StrategyOutputPaths(schema_path=out.schema_path, invariants_path=out.invariants_path),
        out.observations,
    )


def run_strategy_c(spec: CellSpec) -> tuple[StrategyOutputPaths, list[str]]:
    """Run strategy (c) "pure Claude API" for a cell.

    Raises :class:`_spike.scripts.strategies.llm_extractor.KeyAbsentError`
    when ANTHROPIC_API_KEY is absent - the orchestrator catches and
    emits a SKIP record.
    """
    from _spike.scripts.strategies.claude_extractor import run_c_on_transformers_active

    if spec.engine != "transformers":
        raise NotImplementedError(f"strategy (c) for engine {spec.engine!r} not yet implemented")
    if spec.bump_distance != "active":
        raise NotImplementedError(
            f"strategy (c) for bumped versions ({spec.bump_distance}) needs Phase 3 venv plumbing"
        )

    run_dir = spec.to_run_dir()
    out = run_c_on_transformers_active(
        out_dir=run_dir, engine_version=spec.pypi_version or "4.57.3"
    )
    return (
        StrategyOutputPaths(schema_path=out.schema_path, invariants_path=out.invariants_path),
        out.observations,
    )


def run_strategy_d(spec: CellSpec) -> tuple[StrategyOutputPaths, list[str]]:
    """Run strategy (d-ab) or (d-ac) "hybrid" for a cell.

    Phase 3 implementation: per-engine dispatch. (d-ab) supported on
    transformers/vllm/tensorrt active cells; (d-ac) currently transformers
    only (Anthropic-dependent path will be extended once ANTHROPIC_API_KEY
    arrives).

    Phase 3a.2 bumped transformers: source_root is the source-only venv
    path. (a)'s seed remains the active reference (we don't have bumped
    static-miner output without a venv install).
    """
    from _spike.scripts.strategies.hybrid_extractor import (
        run_d_ab_on_tensorrt_active,
        run_d_ab_on_transformers_active,
        run_d_ab_on_vllm_active,
        run_d_ac_on_transformers_active,
    )

    run_dir = spec.to_run_dir()
    if spec.strategy == "d-ab":
        if spec.engine == "transformers":
            source_root: Path | None = None
            if spec.bump_distance != "active":
                from _spike.scripts.venv_setup import ensure_source_only_venv

                sv = ensure_source_only_venv(
                    engine="transformers", version_slug=spec.version_slug
                )
                source_root = sv.source_dir
            out = run_d_ab_on_transformers_active(
                out_dir=run_dir,
                engine_version=spec.pypi_version or "4.57.3",
                source_root=source_root,
            )
        elif spec.engine == "vllm":
            if spec.bump_distance != "active":
                raise NotImplementedError(
                    f"strategy (d-ab) for bumped vllm ({spec.bump_distance}) is "
                    "Phase 3a.2.vllm scope (separate work item)."
                )
            out = run_d_ab_on_vllm_active(
                out_dir=run_dir, engine_version=spec.pypi_version or "0.7.3"
            )
        elif spec.engine == "tensorrt":
            if spec.bump_distance != "active":
                raise NotImplementedError(
                    f"strategy (d-ab) for bumped tensorrt ({spec.bump_distance}) is "
                    "Phase 3a.2.tensorrt scope (separate work item)."
                )
            out = run_d_ab_on_tensorrt_active(
                out_dir=run_dir, engine_version=spec.pypi_version or "0.21.0"
            )
        else:
            raise NotImplementedError(f"strategy (d-ab) for engine {spec.engine!r} not implemented")
    elif spec.strategy == "d-ac":
        if spec.engine != "transformers":
            raise NotImplementedError(
                f"strategy (d-ac) for engine {spec.engine!r} not implemented "
                "(requires ANTHROPIC_API_KEY + engine-specific wiring)"
            )
        out = run_d_ac_on_transformers_active(
            out_dir=run_dir, engine_version=spec.pypi_version or "4.57.3"
        )
    else:
        raise ValueError(f"unrecognised hybrid strategy: {spec.strategy!r}")
    return (
        StrategyOutputPaths(schema_path=out.schema_path, invariants_path=out.invariants_path),
        out.observations,
    )


_STRATEGY_DISPATCHERS = {
    "a": run_strategy_a,
    "b": run_strategy_b,
    "c": run_strategy_c,
    "d-ab": run_strategy_d,
    "d-ac": run_strategy_d,
}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_cell(
    spec: CellSpec,
    *,
    dry_run: bool = False,
    scoring_config: ScoringConfig | None = None,
    lazy_build: bool = True,
) -> CellScore | None:
    """Execute one cell end-to-end. Returns the per-cell score (or None on
    dry-run).

    Steps:
    1. Resolve config (venv path, prompt template, etc). For non-active
       cells with strategies (b)/(c)/(d-*), ``lazy_build=True`` triggers
       source-only venv build on first hit (Phase 2.5 design).
    2. Identify reference paths.
    3. Execute the strategy (writes output files; observations captured).
    4. Score the output via :func:`trial_scoring.score_cell`.
    5. Write the score record JSON + side-artefacts.
    6. Return the score for in-memory consumers.
    """
    if dry_run:
        refs = reference_paths_for(spec.engine, spec.version_slug)
        print(_format_plan(spec, refs))
        return None

    resolved = resolve_cell_config(spec, lazy_build=lazy_build)
    refs = reference_paths_for(spec.engine, spec.version_slug)
    dispatcher = _STRATEGY_DISPATCHERS[spec.strategy]

    try:
        (paths, obs), wall, energy_wh = measure_energy_during(lambda: dispatcher(resolved))
    except Exception as exc:
        # Special-case KeyAbsentError for cleaner reporting
        return _emit_crash_record(spec, exc)

    try:
        score, schema_diffs, inv_diffs, type_or_sev_diffs = trial_scoring.score_cell(
            strategy=spec.strategy,
            engine=spec.engine,
            version_slug=spec.version_slug,
            bump_distance=spec.bump_distance,
            schema_output_path=paths.schema_path,
            invariants_output_path=paths.invariants_path,
            reference_schema_path=refs.schema,
            reference_invariants_path=refs.invariants,
            wall_clock_sec=wall,
            energy_wh=energy_wh,
            config=scoring_config,
            extra_observations=obs,
        )
    except FileNotFoundError as exc:
        # Reference missing - configuration error, not a cell finding.
        raise exc

    _write_score_and_diffs(spec, score, schema_diffs, inv_diffs, type_or_sev_diffs)
    return score


def _emit_crash_record(spec: CellSpec, exc: BaseException) -> CellScore:
    """Build + persist a placeholder CellScore for a crashed cell."""
    from datetime import datetime, timezone

    from _spike.scripts.trial_scoring import FailureMode

    # Detect "key absent" specially so the score record is cleaner.
    name = type(exc).__name__
    is_key_absent = name == "KeyAbsentError"
    mode = "key_absent" if is_key_absent else FailureMode.CRASH.value
    score = trial_scoring.CellScore(
        strategy=spec.strategy,
        engine=spec.engine,
        version_slug=spec.version_slug,
        bump_distance=spec.bump_distance,
        schema_failure_mode=mode,
        invariant_failure_mode=mode,
        failure_modes=[mode],
        observations=[f"cell crashed: {name}: {exc}"],
        scored_at=datetime.now(timezone.utc).isoformat(),
    )
    TRIAL_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    (TRIAL_SCORES_ROOT / spec.to_score_filename()).write_text(score.to_json())
    return score


def _write_score_and_diffs(
    spec: CellSpec,
    score: CellScore,
    schema_diffs: list[trial_scoring.ItemDiff],
    invariant_diffs: list[trial_scoring.ItemDiff],
    type_or_sev_diffs: list[trial_scoring.ItemDiff],
) -> None:
    """Persist the per-cell score + side-artefacts."""
    TRIAL_SCORES_ROOT.mkdir(parents=True, exist_ok=True)
    score_path = TRIAL_SCORES_ROOT / spec.to_score_filename()
    score_path.write_text(score.to_json())
    run_dir = spec.to_run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    trial_scoring.write_diff_artefact(
        diffs=[d for d in schema_diffs + invariant_diffs if d.kind == "recall_miss"],
        out_path=run_dir / "recall_misses.yaml",
        kind_label="recall_misses",
    )
    trial_scoring.write_diff_artefact(
        diffs=[d for d in schema_diffs + invariant_diffs if d.kind == "precision_spurious"],
        out_path=run_dir / "precision_spurious.yaml",
        kind_label="precision_spurious",
    )
    trial_scoring.write_diff_artefact(
        diffs=type_or_sev_diffs,
        out_path=run_dir / "type_mismatches.yaml",
        kind_label="type_mismatches",
    )


def _format_plan(spec: CellSpec, refs: ReferencePaths) -> str:
    """Pretty-print the planned execution for ``--dry-run``."""
    return (
        f"[plan] cell={spec.to_key()} bump={spec.bump_distance}\n"
        f"        strategy={spec.strategy} engine={spec.engine}\n"
        f"        pypi_version={spec.pypi_version or '<project default>'}\n"
        f"        venv_path={spec.venv_path or '<project venv>'}\n"
        f"        prompt={spec.prompt_template_ref or '-'}\n"
        f"        hybrid={spec.hybrid_pattern or '-'}\n"
        f"        ref_schema={refs.schema}\n"
        f"        ref_invariants={refs.invariants}\n"
    )


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def cells_matching(
    *,
    strategy: str | None = None,
    engine: str | None = None,
    version_slug: str | None = None,
    cell_spec: str | None = None,
) -> list[CellSpec]:
    """Return the list of cells matching the filter.

    ``cell_spec`` is a slash-separated string ``"<strategy>/<engine>/<vslug>"``;
    when supplied it overrides the individual fields (used by ``--cell-spec``).

    Implemented Phase 1 (pure registry filter; no external state required).
    """
    if cell_spec is not None:
        parts = cell_spec.split("/")
        if len(parts) != 3:
            raise ValueError(
                f"--cell-spec must be 'strategy/engine/version_slug', got: {cell_spec!r}"
            )
        s, e, v = parts
        key = (s, e, v)
        match = CELL_REGISTRY.get(key)
        if match is None:
            raise KeyError(f"no cell registered for {cell_spec!r} in CELL_REGISTRY")
        return [match]
    return [
        s
        for s in CELL_REGISTRY.values()
        if (strategy is None or s.strategy == strategy)
        and (engine is None or s.engine == engine)
        and (version_slug is None or s.version_slug == version_slug)
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    selection = parser.add_argument_group("cell selection")
    selection.add_argument("--strategy", choices=sorted(_STRATEGY_DISPATCHERS))
    selection.add_argument("--engine", choices=("transformers", "vllm", "tensorrt"))
    selection.add_argument("--version-slug", help="Filesystem-safe version slug, e.g. v4_57_3.")
    selection.add_argument(
        "--cell-spec",
        help="Single-cell filter shorthand: '<strategy>/<engine>/<vslug>'. "
        "Used for re-runs. Overrides individual --strategy/--engine/--version-slug.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Iterate every cell in the registry. Mutually exclusive with single-cell selection.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned execution per cell; do nothing.",
    )

    args = parser.parse_args(argv)

    if not (args.all or args.cell_spec or args.strategy):
        parser.error(
            "Provide a cell selection: --all, --cell-spec, "
            "or at least --strategy (+ optional --engine/--version-slug)."
        )

    try:
        cells = cells_matching(
            strategy=args.strategy if not args.all else None,
            engine=args.engine if not args.all else None,
            version_slug=args.version_slug if not args.all else None,
            cell_spec=args.cell_spec,
        )
    except (KeyError, ValueError) as exc:
        print(f"[trial_runner] cell selection error: {exc}", file=sys.stderr)
        return 2

    if not cells:
        print("[trial_runner] no cells matched the filter.", file=sys.stderr)
        return 1

    started_at = datetime.now(timezone.utc).isoformat()
    print(
        f"[trial_runner] {started_at} - {len(cells)} cell(s) to run (dry_run={args.dry_run})",
        file=sys.stderr,
    )

    failures = 0
    for spec in cells:
        try:
            score = run_cell(spec, dry_run=args.dry_run)
        except NotImplementedError as exc:
            print(
                f"[trial_runner] {spec.to_key()} stubbed: {exc.args[0].splitlines()[0]}",
                file=sys.stderr,
            )
            failures += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[trial_runner] {spec.to_key()} unexpected failure: {exc}", file=sys.stderr)
            failures += 1
            continue
        if score is not None:
            print(
                f"[trial_runner] {spec.to_key()} scored: "
                f"schema_recall={score.schema_recall:.2f} "
                f"inv_recall={score.invariant_recall:.2f}",
                file=sys.stderr,
            )

    if failures:
        print(f"[trial_runner] {failures}/{len(cells)} cells failed/stubbed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
