---
phase: 03-library-api
plan: 01
subsystem: domain
tags: [pydantic, domain-models, result-types, config-types]

requires:
  - phase: 02-config-system
    provides: ExperimentConfig fully built with forward-reference resolution

provides:
  - ExperimentResult class (renamed from AggregatedResult) in domain/experiment.py
  - StudyResult stub (experiments + name) in domain/experiment.py
  - AggregatedResult = ExperimentResult alias for v1.x compatibility
  - StudyConfig stub (experiments + name) in config/models.py
  - ExperimentResult and StudyResult exported from domain/__init__.py

affects:
  - 03-library-api (Plan 02 depends on these types for _api.py + __init__.py)
  - Any consumer of AggregatedResult (alias keeps them working)

tech-stack:
  added: []
  patterns:
    - "Compatibility alias pattern: NewName = OldName at module bottom, excluded from __all__ removal until v3.0"
    - "Stub class pattern: minimal Pydantic BaseModel with only M1 fields; M2+ fields documented in docstring"

key-files:
  created: []
  modified:
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/domain/__init__.py
    - src/llenergymeasure/config/models.py

key-decisions:
  - "AggregatedResult kept as alias (not removed) to avoid breaking v1.x consumers before Phase 7 CLI rewrite"
  - "StudyResult is mutable (no frozen model_config) - result containers don't need immutability"
  - "StudyConfig placed after _rebuild_experiment_config() call to ensure ExperimentConfig forward refs are resolved"

patterns-established:
  - "Stub class: document M2+ fields in docstring rather than adding placeholder fields"

requirements-completed: [CFG-17, LA-09]

duration: 8min
completed: 2026-02-26
---

# Phase 3 Plan 01: Library API - Domain Type Contracts Summary

**ExperimentResult (renamed from AggregatedResult) + StudyResult stub + StudyConfig stub establishing the type contracts that Plan 02's public API depends on.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-26T16:56:08Z
- **Completed:** 2026-02-26T17:04:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Renamed `AggregatedResult` class to `ExperimentResult` with v1.x compatibility alias
- Added `StudyResult` stub with `experiments: list[ExperimentResult]` and `name: str | None`
- Added `StudyConfig` stub with `experiments: list[ExperimentConfig]` (min_length=1) and `name: str | None`
- All 28 existing domain experiment tests pass unchanged (alias ensures no breakage)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rename AggregatedResult to ExperimentResult, add StudyResult stub** - `213bfef` (feat)
2. **Task 2: Add StudyConfig stub to config/models.py** - `bc92e0b` (feat)

## Files Created/Modified
- `src/llenergymeasure/domain/experiment.py` - Renamed class, added StudyResult stub, added AggregatedResult alias
- `src/llenergymeasure/domain/__init__.py` - Added ExperimentResult and StudyResult to imports and __all__
- `src/llenergymeasure/config/models.py` - Added StudyConfig after _rebuild_experiment_config() call

## Decisions Made
- `StudyResult` is not frozen (no `model_config = {"frozen": True}`) — result containers need to accumulate experiments
- `StudyConfig` placed after `_rebuild_experiment_config()` call so `ExperimentConfig` has resolved forward references when `StudyConfig` is defined
- `AggregatedResult` alias kept in `__all__` (via import in `domain/__init__.py`) to avoid silently breaking callers

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- `test_config_models.py` imports `BUILTIN_DATASETS` which does not exist in `config/models.py` — confirmed pre-existing failure unrelated to this plan (reproduced on unmodified codebase via `git stash`).

## Next Phase Readiness
- `ExperimentResult`, `StudyResult`, and `StudyConfig` all importable and validating correctly
- Plan 02 (`_api.py` + `__init__.py`) can now import these types to wire the public API surface
- No blockers

---
*Phase: 03-library-api*
*Completed: 2026-02-26*
