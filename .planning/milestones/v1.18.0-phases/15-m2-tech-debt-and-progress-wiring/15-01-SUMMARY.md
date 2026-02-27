---
phase: 15-m2-tech-debt-and-progress-wiring
plan: 01
subsystem: study
tags: [study, runner, progress, display, subprocess, multiprocessing]

# Dependency graph
requires:
  - phase: 12-integration
    provides: print_study_progress() in cli/_display.py
  - phase: 11-subprocess-isolation-and-studyrunner
    provides: _consume_progress_events() stub and _run_experiment_worker
provides:
  - _consume_progress_events() wired to print_study_progress() — progress visible during study
  - Phantom experiment_timeout_seconds getattr removed — _calculate_timeout() sole timeout source
affects: [16-onwards, any phase testing study execution output]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy import inside event handler: print_study_progress imported inside _consume_progress_events() body, not at module top level — keeps study/runner.py decoupled from CLI display layer"

key-files:
  created: []
  modified:
    - src/llenergymeasure/study/runner.py
    - tests/unit/test_study_runner.py

key-decisions:
  - "Lazy import pattern for print_study_progress inside event handlers: avoids coupling study/runner.py to CLI layer at import time"
  - "Phantom getattr(execution, 'experiment_timeout_seconds', None) removed — field never existed on ExecutionConfig (extra=forbid), always returned None"
  - "_run_one() now takes explicit index and total params passed from run() loop — cleaner than threading through closure or class state"

patterns-established:
  - "Progress consumer thread receives index/total/config from _run_one() caller, not from queue events — event payload is not the source of truth for position"

requirements-completed: []

# Metrics
duration: 2min
completed: 2026-02-27
---

# Phase 15 Plan 01: M2 Tech Debt and Progress Wiring Summary

**Progress display wired from orphaned stub to print_study_progress(), and phantom experiment_timeout_seconds getattr removed — two targeted tech debt fixes closing one broken flow**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-27T21:09:19Z
- **Completed:** 2026-02-27T21:11:03Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `_consume_progress_events()` rewritten to accept `index`, `total`, `config` and forward started/completed/failed events to `print_study_progress()` via lazy import
- `_run_one()` updated to accept and pass `index` and `total` to the consumer thread; `run()` passes 1-based index and `len(ordered)` total
- Phantom `getattr(self.study.execution, "experiment_timeout_seconds", None)` removed — `_calculate_timeout()` is now the sole timeout source
- `_calculate_timeout()` docstring cleaned: misleading "escape hatch" sentence removed
- `test_progress_events_forwarded` added — verifies started and completed events produce the correct `print_study_progress()` calls
- 536 tests pass (1 new)

## Task Commits

1. **Task 1: Wire progress display in _consume_progress_events()** - `c8e41ea` (feat)
2. **Task 2: Remove phantom experiment_timeout_seconds reference** - `e637e67` (fix)

## Files Created/Modified

- `src/llenergymeasure/study/runner.py` - Rewrote `_consume_progress_events()`, updated `_run_one()` signature, removed phantom getattr, fixed docstring
- `tests/unit/test_study_runner.py` - Added `test_progress_events_forwarded`; updated `sigint_during_run_one` wrapper to accept/pass `index` and `total`

## Decisions Made

- Lazy import pattern: `print_study_progress` imported inside each event handler branch, not at module top level. This keeps `study/runner.py` decoupled from the CLI display layer — no import cost if progress display is unused.
- `index` and `total` passed as explicit parameters to `_run_one()` rather than accessed via closure or class state — explicit is cleaner and testable.
- No new `ExecutionConfig` field for user timeout — phantom getattr was always returning None; a proper field can be added in a future milestone if needed.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. The SIGINT test's `sigint_during_run_one` wrapper needed updating to accept `index` and `total` keyword args (Rule 1 auto-fix scope — directly caused by the signature change), but this was anticipated by the plan's instructions.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Progress lines `[3/12] ... model backend precision` now visible on stderr during study execution
- Plan 02 (15-02) can proceed; tech debt items closed
- 536 unit tests pass, no regressions

---
*Phase: 15-m2-tech-debt-and-progress-wiring*
*Completed: 2026-02-27*
