---
phase: 01-measurement-foundations
plan: 03
subsystem: infra
tags: [state-machine, subprocess, lifecycle, atomic-persistence, platformdirs]

# Dependency graph
requires:
  - "llenergymeasure.exceptions (InvalidStateTransitionError, ConfigError)"
  - "llenergymeasure.security (sanitize_experiment_id, is_safe_path)"
provides:
  - "3-state machine: ExperimentPhase (INITIALISING, MEASURING, DONE)"
  - "ExperimentState: Pydantic model with failed:bool flag orthogonal to phase"
  - "StateManager: atomic persistence, find_by_config_hash, cleanup_stale"
  - "compute_config_hash: SHA-256 16-char stable config hash"
  - "SubprocessRunner: signal handling, process group management, no CLI deps"
  - "build_subprocess_env: backend-specific env construction"
affects:
  - phase 03 (config models use ExperimentState for resume)
  - phase 04 (PyTorch runner uses StateManager and SubprocessRunner)

# Tech tracking
tech-stack:
  added:
    - "platformdirs (user_state_path for cross-platform state dir)"
  patterns:
    - "3-state + failed:bool orthogonal to phase (replaces 6-state machine)"
    - "Atomic write: temp file then os.rename() for state persistence"
    - "infra/ package for infrastructure utilities without CLI/backend deps"
    - "stdlib logging throughout (no loguru)"

key-files:
  created:
    - src/llenergymeasure/core/state.py
    - src/llenergymeasure/infra/__init__.py
    - src/llenergymeasure/infra/subprocess.py
  modified:
    - src/llenergymeasure/orchestration/__init__.py
    - src/llenergymeasure/orchestration/launcher.py
  deleted:
    - src/llenergymeasure/state/experiment_state.py
    - src/llenergymeasure/cli/lifecycle.py
    - src/llenergymeasure/orchestration/lifecycle.py

key-decisions:
  - "3-state machine (INITIALISING, MEASURING, DONE) + failed:bool replaces 6-state v1.x design"
  - "failed flag orthogonal to phase: mark_failed() does not change phase"
  - "platformdirs.user_state_path used for default state dir (cross-platform)"
  - "orchestration/lifecycle.py deleted: torch+loguru specific, rebuilt in Phase 4"
  - "SubprocessRunner raises SystemExit(130) not typer.Exit(130)"

requirements-completed:
  - INF-07
  - INF-08
  - INF-19

# Metrics
duration: 4min
completed: 2026-02-26
---

# Phase 1 Plan 03: State Machine and Subprocess Lifecycle Summary

**v2.0 3-state machine (INITIALISING, MEASURING, DONE) with atomic StateManager, and clean infra/subprocess.py SubprocessRunner with no CLI dependencies**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-02-26T11:44:33Z
- **Completed:** 2026-02-26T11:48:07Z
- **Tasks:** 2
- **Files created:** 3
- **Files modified:** 2
- **Files deleted:** 3

## Accomplishments

- Replaced v1.x 6-state `ExperimentStatus` (INITIALISED, RUNNING, COMPLETED, AGGREGATED, FAILED, INTERRUPTED) with a v2.0 3-state `ExperimentPhase` (INITIALISING, MEASURING, DONE) plus an orthogonal `failed: bool` flag
- Implemented `StateManager` with atomic persistence (write-to-temp-then-rename), `find_by_config_hash()` for experiment deduplication/resume, and `cleanup_stale()` for stale MEASURING states
- Used `platformdirs.user_state_path("llenergymeasure")` for default state directory (cross-platform, replaces hardcoded `.state`)
- Created `infra/subprocess.py` with `SubprocessRunner`: process group management, SIGINT/SIGTERM handling, SIGKILL escalation — with zero typer/Rich/loguru dependencies
- Deleted 3 old files and updated 2 more to remove references to deleted modules

## Task Commits

Each task was committed atomically:

1. **Task 1: Create 3-state machine in core/state.py** — `1e95005` (feat)
2. **Task 2: Carry-forward subprocess lifecycle into infra/subprocess.py** — `53c2fc9` (feat)

## Files Created/Modified

- `src/llenergymeasure/core/state.py` — ExperimentPhase (3 states), ExperimentState (failed:bool, Pydantic), StateManager (atomic, find_by_config_hash, cleanup_stale), compute_config_hash
- `src/llenergymeasure/infra/__init__.py` — Package init exporting SubprocessRunner, build_subprocess_env
- `src/llenergymeasure/infra/subprocess.py` — SubprocessRunner with signal handling, no CLI/loguru deps
- `src/llenergymeasure/orchestration/__init__.py` — Removed lifecycle imports (module deleted)
- `src/llenergymeasure/orchestration/launcher.py` — Removed `ensure_clean_start()` reference

## Decisions Made

- `failed: bool` flag is orthogonal to phase: `mark_failed()` sets the flag without changing `phase`. An experiment can fail during MEASURING and still be in MEASURING phase (no terminal FAILED state).
- `platformdirs.user_state_path` used instead of hardcoded `.state` — cross-platform, respects OS conventions
- `orchestration/lifecycle.py` (torch/loguru/accelerate) deleted in full — its functionality (CUDA cleanup, distributed teardown) is torch-specific and will be rebuilt in Phase 4 alongside the PyTorch backend
- `SubprocessRunner` raises `SystemExit(130)` instead of `typer.Exit(130)` — no Typer dependency in infrastructure layer

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Removed stale lifecycle imports from orchestration/__init__.py and launcher.py**

- **Found during:** Task 2 (deleting orchestration/lifecycle.py)
- **Issue:** Deleting `orchestration/lifecycle.py` would have broken `orchestration/__init__.py` (re-exports 6 functions from it) and `orchestration/launcher.py` (lazy import of `ensure_clean_start`)
- **Fix:** Removed all lifecycle imports from `orchestration/__init__.py`; removed `ensure_clean_start()` call from `orchestration/launcher.py` lazy import block
- **Files modified:** `src/llenergymeasure/orchestration/__init__.py`, `src/llenergymeasure/orchestration/launcher.py`
- **Commit:** `53c2fc9`

### Notes

The plan verification command (`python -c "from llenergymeasure.core.state import ..."`) fails due to a pre-existing issue: `core/__init__.py` eagerly imports v1.x modules that reference the old `ConfigurationError` (renamed to `ConfigError` in Plan 02). This is out of scope for this plan and is logged as a deferred item. Verification was confirmed by loading modules directly via `importlib.util`.

## User Setup Required

None.

## Next Phase Readiness

- `core/state.py` is ready for import by Phase 2 (config models) and Phase 4 (PyTorch runner)
- `infra/subprocess.py` is ready for Phase 4 (Docker multi-backend launch)
- 19 v1.x source files still import old exception names — updated as each phase rewrites those modules
- `core/__init__.py` still eagerly imports broken v1.x modules — will be cleaned up in Phase 2

---
*Phase: 01-measurement-foundations*
*Completed: 2026-02-26*
