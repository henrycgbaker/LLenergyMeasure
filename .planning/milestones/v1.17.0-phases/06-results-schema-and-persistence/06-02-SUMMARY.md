---
phase: 06-results-schema-and-persistence
plan: 02
subsystem: results
tags: [persistence, atomic-write, collision-handling, parquet, sidecar, pydantic]

requires:
  - phase: 06-01
    provides: ExperimentResult v2.0 schema with frozen model_config and timeseries field

provides:
  - save_result() function with atomic writes and collision-safe directory creation
  - load_result() with graceful sidecar degradation
  - ExperimentResult.save() and ExperimentResult.from_json() thin delegation methods
  - 16 passing unit tests for persistence API

affects: [06-03-aggregation, library-api, cli-run]

tech-stack:
  added: []
  patterns:
    - deferred-import-to-break-circular-dependency
    - posix-atomic-write-via-temp-plus-os-replace
    - collision-safe-directory-via-counter-suffix

key-files:
  created:
    - src/llenergymeasure/results/persistence.py
    - tests/unit/test_persistence_v2.py
  modified:
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/results/__init__.py

key-decisions:
  - "ExperimentResult.save() and from_json() use deferred imports inside method bodies to avoid circular import (experiment.py -> persistence.py -> experiment.py)"
  - "timeseries_source uses shutil.copy2 (copy, not move) — caller retains the source file"
  - "load_result() warns on missing sidecar but preserves timeseries field value (graceful degradation, not data loss)"

patterns-established:
  - "Persistence module (_save_result, _load_result) separate from domain model — domain methods are thin delegation wrappers"
  - "Collision-free directory: _find_collision_free_dir() tries base, base_1, base_2 etc. — always creates before returning"

requirements-completed: [RES-16, RES-17, RES-18, RES-19]

duration: 3min
completed: "2026-02-27"
---

# Phase 6 Plan 02: Results Persistence Summary

**v2.0 persistence API: save_result() with atomic JSON writes, collision-safe {model}_{backend}_{timestamp}/ dirs, timeseries.parquet sidecar copy, load_result() with graceful degradation, and 16 passing unit tests.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-27T08:56:10Z
- **Completed:** 2026-02-27T08:59:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- New `results/persistence.py` module: `save_result()`, `load_result()`, `_atomic_write()`, `_find_collision_free_dir()`
- `ExperimentResult.save()` and `ExperimentResult.from_json()` wired in `domain/experiment.py` as thin delegation wrappers with deferred imports to avoid circular dependency
- 16 unit tests covering all behaviours including round-trip fidelity for datetime, `steady_state_window` tuple, `effective_config`, and sidecar management

## Task Commits

Each task was committed atomically:

1. **Task 1: Create results/persistence.py and wire save()/from_json() onto ExperimentResult** - `d355820` (feat)
2. **Task 2: Unit tests for persistence API** - `2ac9b84` (test)

## Files Created/Modified

- `src/llenergymeasure/results/persistence.py` — v2.0 persistence API: save_result(), load_result(), atomic writes, collision handling
- `src/llenergymeasure/domain/experiment.py` — added save() and from_json() class/instance methods
- `tests/unit/test_persistence_v2.py` — 16 unit tests for full API surface
- `src/llenergymeasure/results/__init__.py` — removed stale `calculate_efficiency_metrics` import (pre-existing bug)

## Decisions Made

- `ExperimentResult.save()` and `from_json()` use deferred imports inside method bodies to break the circular import chain (`experiment.py` imports from `persistence.py` which imports `ExperimentResult`)
- `timeseries_source` uses `shutil.copy2` (not move) — caller retains source file for reuse or cleanup
- `load_result()` warns on missing sidecar with `UserWarning` but preserves the `timeseries` field value — graceful degradation, not data loss

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stale `calculate_efficiency_metrics` import in `results/__init__.py`**
- **Found during:** Task 2 (running tests)
- **Issue:** `results/__init__.py` imported `calculate_efficiency_metrics` from `aggregation.py`, but that function does not exist. Any import of `llenergymeasure.results` (triggered by `persistence.py` import) raised `ImportError: cannot import name 'calculate_efficiency_metrics'`.
- **Fix:** Removed the stale import and its entry from `__all__` in `results/__init__.py`.
- **Files modified:** `src/llenergymeasure/results/__init__.py`
- **Verification:** All 16 tests pass after fix
- **Committed in:** `2ac9b84` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (blocking)
**Impact on plan:** Fix was necessary for any code that imports `llenergymeasure.results` — pre-existing bug, not introduced by this plan.

## Issues Encountered

- `.gitignore` contains `results` as a bare pattern, which matches `src/llenergymeasure/results/` package directory. New files in that directory require `git add -f`. Pre-existing `results/` files were already tracked (added with `-f` at initial commit). Logged for future gitignore cleanup.

## Next Phase Readiness

- Persistence API complete — `result.save(output_dir)` → `result.json`, `ExperimentResult.from_json(path)` round-trip
- Plan 06-03 (aggregation) and CLI `llem run` can both use `result.save()` directly
- No blockers

---
*Phase: 06-results-schema-and-persistence*
*Completed: 2026-02-27*
