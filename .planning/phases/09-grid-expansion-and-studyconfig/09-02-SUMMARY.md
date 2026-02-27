---
phase: 09-grid-expansion-and-studyconfig
plan: 02
subsystem: config
tags: [config-loading, study-config, preflight, yaml, integration-tests]

requires:
  - phase: 09-01
    provides: expand_grid, apply_cycles, compute_study_design_hash, CycleOrder, SkippedConfig, ExecutionConfig, StudyConfig

provides:
  - load_study_config() public function in config/loader.py (CFG-12 contract)
  - format_preflight_summary() display function in study/grid.py
  - 10 integration tests for load_study_config() in test_config_loader.py
  - 8 unit tests for format_preflight_summary() in test_study_grid.py

affects:
  - 09-03 (subprocess runner — calls load_study_config to get resolved StudyConfig)
  - 12 (CLI — calls load_study_config + format_preflight_summary for llem run)

tech-stack:
  added: []
  patterns:
    - "CFG-12: sweep resolution at YAML parse time via expand_grid() before individual Pydantic validation"
    - "TYPE_CHECKING guard for StudyConfig import in grid.py to avoid circular import"
    - "load_study_config guards: empty study (ConfigError) and all-invalid (ConfigError) — expand_grid already handles these but double-guard for clarity"
    - "format_preflight_summary: n_configs derived as n_runs // n_cycles (integer division)"
    - "skip_rate > 0.5 triggers WARNING line in preflight summary"

key-files:
  modified:
    - src/llenergymeasure/config/loader.py
    - src/llenergymeasure/study/grid.py
    - tests/unit/test_config_loader.py
    - tests/unit/test_study_grid.py

key-decisions:
  - "TYPE_CHECKING import for StudyConfig in grid.py: format_preflight_summary() type annotation uses StudyConfig at runtime but from __future__ import annotations makes it a string; TYPE_CHECKING guard avoids circular import loader->grid->loader"
  - "load_study_config delegates empty/all-invalid guard to expand_grid() (which already raises ConfigError) but adds a post-hoc guard for the case where valid_experiments is empty after expand_grid returns (belt-and-suspenders)"
  - "format_preflight_summary derives n_configs = n_runs // n_cycles to recover unique config count without storing it separately in StudyConfig"

requirements-completed: [CFG-12]

duration: 4min
completed: 2026-02-27
---

# Phase 09 Plan 02: load_study_config() and format_preflight_summary() Summary

**load_study_config() wiring YAML→StudyConfig through the full CFG-12 pipeline (expand_grid at parse time), plus format_preflight_summary() terminal display function and 18 integration/display tests**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-02-27T11:20:55Z
- **Completed:** 2026-02-27T11:24:50Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- `load_study_config()` added to `config/loader.py` and exported in `__all__`: full 8-step resolution pipeline (load YAML, apply CLI overrides, parse execution block, expand_grid, guard empty/all-invalid, hash, apply_cycles, construct StudyConfig)
- `format_preflight_summary()` added to `study/grid.py`: produces the locked terminal format (hash, config count, cycle count, total runs, order mode, per-skip reasons, >50% skip WARNING)
- 10 integration tests for `load_study_config()` covering all paths: grid sweep, explicit, combined, execution block, CLI overrides, base: inheritance, empty study error, all-invalid error, file-not-found, hash stability across execution changes
- 8 unit tests for `format_preflight_summary()`: basic format, hash display, skip lines, high-skip WARNING, low-skip (no WARNING), `skipped` argument override, single-cycle
- 461 total unit tests pass (was 443 before plan 01, +18 new tests across plans 01 and 02); mypy clean on both modified files

## Task Commits

1. **Task 1: load_study_config() and format_preflight_summary()** - `1c58c1b` (feat)
2. **Task 2: Integration and display tests** - `b280eac` (test)

## Files Created/Modified

- `src/llenergymeasure/config/loader.py` — Added `load_study_config()` + imports; `load_study_config` added to `__all__`
- `src/llenergymeasure/study/grid.py` — Added `format_preflight_summary()`; `TYPE_CHECKING` import for `StudyConfig`
- `tests/unit/test_config_loader.py` — 10 new `load_study_config` tests appended
- `tests/unit/test_study_grid.py` — 8 new `format_preflight_summary` tests in `TestFormatPreflightSummary` class

## Decisions Made

- **TYPE_CHECKING import pattern:** `StudyConfig` is needed as a type annotation in `grid.py`, but `loader.py` imports from `grid.py`, creating a potential circular import. `from __future__ import annotations` (already present) makes all annotations strings, so `if TYPE_CHECKING:` import is sufficient — no circular import at runtime.
- **n_configs derivation:** `format_preflight_summary()` recovers the unique config count as `n_runs // n_cycles` rather than storing it separately in `StudyConfig`. This avoids adding a new field and is always correct because `apply_cycles()` produces exactly `n_configs * n_cycles` entries.
- **Double guard in load_study_config:** `expand_grid()` already raises `ConfigError` for empty and all-invalid cases, but `load_study_config()` adds post-hoc guards for the zero-experiment and zero-valid cases — belt-and-suspenders for defensive programming at the public API boundary.

## Deviations from Plan

None — plan executed exactly as written. No auto-fixes required; all tests passed first attempt.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- `load_study_config()` is fully tested and ready for Phase 12 (CLI) to call with `cli_overrides`
- `format_preflight_summary()` is ready for Phase 12 `llem run` pre-flight display
- Phase 09-03 (subprocess isolation / manifest writer) can import `StudyConfig` from loader
- No blockers

---
*Phase: 09-grid-expansion-and-studyconfig*
*Completed: 2026-02-27*
