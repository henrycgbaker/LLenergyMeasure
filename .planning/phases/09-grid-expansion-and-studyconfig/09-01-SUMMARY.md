---
phase: 09-grid-expansion-and-studyconfig
plan: 01
subsystem: config
tags: [pydantic, sweep, grid-expansion, study-config, tdd]

requires:
  - phase: M1-completion
    provides: ExperimentConfig model with backend sections, ConfigError exception hierarchy

provides:
  - ExecutionConfig Pydantic model (n_cycles, cycle_order, gaps, shuffle_seed)
  - Expanded StudyConfig model (execution, study_design_hash, skipped_configs)
  - study/ package with grid.py (expand_grid, apply_cycles, compute_study_design_hash, CycleOrder, SkippedConfig)
  - Full TDD coverage: 38 unit tests covering all sweep modes, cycle ordering, hash, invalid handling, base: resolution

affects:
  - 09-02 (manifest writer — consumes StudyConfig, study_design_hash)
  - 09-03 (subprocess runner — consumes expanded experiment list from expand_grid)
  - 10-12 (integration and CLI wiring — depend on these models and functions)

tech-stack:
  added: []
  patterns:
    - "TDD: RED (stubs + failing tests) → GREEN (implementation) committed separately"
    - "Cartesian product sweep: itertools.product over sweep dimensions"
    - "Backend-scoped dotted keys: pytorch.batch_size routed to backend section dict"
    - "study_design_hash: SHA-256[:16] of json.dumps([exp.model_dump() for exp in experiments], sort_keys=True)"
    - "Cycle ordering: sequential=[A,A,B,B], interleaved=[A,B,A,B], shuffled=random.Random(seed)"
    - "StrEnum backport pattern for Python 3.10 compatibility"
    - "SkippedConfig dataclass: collects invalid combos without raising, enables post-hoc display"

key-files:
  created:
    - src/llenergymeasure/study/__init__.py
    - src/llenergymeasure/study/grid.py
    - tests/unit/test_study_grid.py
  modified:
    - src/llenergymeasure/config/models.py

key-decisions:
  - "StrEnum backport for Python 3.10: sys.version_info guard + str+Enum fallback (project uses Python 3.10 test runner)"
  - "errors cast: [dict(e) for e in exc.errors()] to satisfy mypy (Pydantic ErrorDetails not dict[str, Any])"
  - "Multi-backend sweep: independent Cartesian grids per backend — pytorch gets [precision x batch_size], vllm gets [precision x max_num_seqs], no cross-product between backends"

patterns-established:
  - "Sweep expansion: _extract_fixed() strips study-only keys, _expand_sweep() handles dotted notation, _load_base() resolves relative to study YAML parent"
  - "Invalid combination handling: collect as SkippedConfig, hard ConfigError only if all invalid"

requirements-completed: [CFG-11, CFG-13, CFG-14, CFG-15, CFG-16]

duration: 6min
completed: 2026-02-27
---

# Phase 09 Plan 01: Grid Expansion and StudyConfig Summary

**ExecutionConfig + StudyConfig Pydantic models with full sweep grid expansion (Cartesian product, backend-scoped dotted keys, base: inheritance, invalid collection) and cycle ordering via TDD**

## Performance

- **Duration:** 6 min
- **Started:** 2026-02-27T11:11:59Z
- **Completed:** 2026-02-27T11:18:08Z
- **Tasks:** 2 (TDD RED + GREEN)
- **Files modified:** 4

## Accomplishments

- `ExecutionConfig` model: n_cycles, cycle_order (sequential/interleaved/shuffled), config_gap_seconds, cycle_gap_seconds, shuffle_seed — all with correct Pydantic validation
- `StudyConfig` expanded from M1 stub: adds execution, study_design_hash, skipped_configs fields
- `study/` package created with `grid.py` implementing `expand_grid()`, `compute_study_design_hash()`, `apply_cycles()`
- 38 unit tests pass covering all 9 test groups; 443 total unit tests pass (no regressions); mypy clean

## Task Commits

1. **Task 1: Models + stubs + failing tests (RED)** - `af85b4a` (test)
2. **Task 2: Implement grid functions (GREEN)** - `92a1618` (feat)

## Files Created/Modified

- `src/llenergymeasure/config/models.py` — Added `ExecutionConfig`; expanded `StudyConfig` with execution, study_design_hash, skipped_configs fields
- `src/llenergymeasure/study/__init__.py` — study package init
- `src/llenergymeasure/study/grid.py` — `expand_grid`, `apply_cycles`, `compute_study_design_hash`, `CycleOrder`, `SkippedConfig`
- `tests/unit/test_study_grid.py` — 38 unit tests covering all behaviours

## Decisions Made

- **Python 3.10 StrEnum backport:** test runner is Python 3.10 (system python3), which lacks `StrEnum`. Added `sys.version_info` guard with `str+Enum` fallback — no dependency added, just stdlib compatibility shim.
- **Multi-backend grid independence:** When `backend: [pytorch, vllm]` with backend-scoped dims, each backend gets its own Cartesian product (not a cross-product between backends). pytorch gets `[precision x pytorch.batch_size]`, vllm gets `[precision x vllm.max_num_seqs]`.
- **Pydantic ErrorDetails type:** `exc.errors()` returns `list[ErrorDetails]`, not `list[dict[str, Any]]`. Used `[dict(e) for e in exc.errors()]` to satisfy mypy without ignoring the type error.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] StrEnum Python 3.10 incompatibility**
- **Found during:** Task 1 (test collection)
- **Issue:** `StrEnum` not available in Python 3.10 stdlib; test runner uses Python 3.10
- **Fix:** Added `sys.version_info` guard with `str+Enum` backport class in grid.py
- **Files modified:** `src/llenergymeasure/study/grid.py`
- **Verification:** Tests collect and run correctly under Python 3.10
- **Committed in:** `af85b4a` (Task 1 commit — fixed before commit)

**2. [Rule 1 - Bug] Pydantic ErrorDetails type mismatch**
- **Found during:** Task 2 (mypy type check)
- **Issue:** `exc.errors()` returns `list[ErrorDetails]` not `list[dict[str, Any]]` — mypy assignment error
- **Fix:** Cast via `[dict(e) for e in exc.errors()]`
- **Files modified:** `src/llenergymeasure/study/grid.py`, `src/llenergymeasure/config/models.py` (dict -> dict[str, Any])
- **Verification:** mypy reports no issues in 2 source files
- **Committed in:** `92a1618` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes required for Python 3.10 compatibility and type correctness. No scope creep.

## Issues Encountered

None — plan executed as designed. Grid expansion logic aligned well with planned algorithm; all test cases passed first attempt after implementation.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `expand_grid()`, `compute_study_design_hash()`, `apply_cycles()` are fully tested and ready for consumption
- Phase 09-02 (manifest writer) can import `StudyConfig`, `study_design_hash`, and `SkippedConfig` directly
- Phase 12 (CLI) should set effective defaults: n_cycles=3, cycle_order="interleaved" (Pydantic defaults are conservative: 1 cycle, sequential)
- No blockers

---
*Phase: 09-grid-expansion-and-studyconfig*
*Completed: 2026-02-27*
