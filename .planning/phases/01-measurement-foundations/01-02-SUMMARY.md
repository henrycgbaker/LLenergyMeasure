---
phase: 01-measurement-foundations
plan: 02
subsystem: infra
tags: [protocols, exceptions, resilience, security, dependency-injection, stdlib]

# Dependency graph
requires: []
provides:
  - "LLEMError exception hierarchy (LLEMError + 5 direct subclasses)"
  - "5 runtime-checkable Protocol interfaces for DI (ModelLoader, InferenceEngine, MetricsCollector, EnergyBackend, ResultsRepository)"
  - "retry_on_error decorator with exponential backoff, no external deps"
  - "Path sanitisation utilities (validate_path, is_safe_path, sanitize_experiment_id)"
affects:
  - all later phases (exceptions, protocols imported everywhere)
  - phase 02 (config models use ConfigError)
  - phase 04 (PyTorch backend implements ModelLoader, InferenceEngine protocols)
  - phase 06 (ExperimentResult referenced in ResultsRepository)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flat exception hierarchy: LLEMError -> 5 direct subclasses (no sub-sub-classes)"
    - "runtime_checkable Protocol for all DI interfaces"
    - "from __future__ import annotations + TYPE_CHECKING for forward refs"
    - "stdlib logging (logging.getLogger) not loguru in base package"

key-files:
  created: []
  modified:
    - src/llenergymeasure/exceptions.py
    - src/llenergymeasure/protocols.py
    - src/llenergymeasure/resilience.py
    - src/llenergymeasure/security.py

key-decisions:
  - "Flat exception hierarchy: only 5 LLEMError subclasses, no deeper nesting"
  - "InvalidStateTransitionError kept as ExperimentError subclass (experiment lifecycle)"
  - "Pydantic ValidationError passes through unchanged — not wrapped in ConfigError"
  - "ResultsRepository Protocol: save/load only (v2.0 API) — old save_raw/list_raw/load_raw dropped"
  - "MetricsCollector.collect returns Any (not CombinedMetrics) — v2.0 type defined later"
  - "resilience.py uses stdlib logging; loguru not a base package dependency"

patterns-established:
  - "Protocol-first DI: all pluggable components implement a Protocol"
  - "TYPE_CHECKING imports for forward refs to Phase 2/6 types"
  - "Base package has zero optional-dependency imports (no torch, no loguru)"

requirements-completed:
  - INF-06
  - INF-18
  - INF-20

# Metrics
duration: 8min
completed: 2026-02-26
---

# Phase 1 Plan 02: Protocols, Exceptions, Resilience, Security Summary

**v2.0 error hierarchy (LLEMError + 5 subclasses), 5 runtime-checkable Protocol interfaces, stdlib-only retry decorator, and path sanitisation utilities**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-26T11:39:26Z
- **Completed:** 2026-02-26T11:47:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Replaced v1.x `LLMBenchError` tree (13 classes) with flat `LLEMError + 5` hierarchy matching product decisions
- Rewrote `protocols.py` with 5 `@runtime_checkable` Protocol classes using v2.0 signatures and forward references
- Stripped `resilience.py` of loguru and torch deps; `retry_on_error` now uses stdlib logging and defaults to `LLEMError`
- Removed dead `check_env_for_secrets()` from `security.py`; updated `ConfigError` import

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite protocols.py and exceptions.py** - `d16d87c` (refactor)
2. **Task 2: Rewrite resilience.py and security.py** - `c782223` (refactor, via pre-commit hook)

## Files Created/Modified

- `src/llenergymeasure/exceptions.py` - v2.0 flat hierarchy: LLEMError, ConfigError, BackendError, PreFlightError, ExperimentError, StudyError, InvalidStateTransitionError
- `src/llenergymeasure/protocols.py` - 5 runtime-checkable DI protocols for v2.0 (ModelLoader, InferenceEngine, MetricsCollector, EnergyBackend, ResultsRepository)
- `src/llenergymeasure/resilience.py` - retry_on_error with stdlib logging, LLEMError default, no torch/loguru
- `src/llenergymeasure/security.py` - validate_path, is_safe_path, sanitize_experiment_id with ConfigError

## Decisions Made

- `InvalidStateTransitionError` retained as `ExperimentError` subclass (experiment lifecycle semantics)
- `ResultsRepository` Protocol changed from `save_raw/list_raw/load_raw/save_aggregated` to clean `save/load` (v2.0 API)
- `MetricsCollector.collect` returns `Any` not `CombinedMetrics` — v2.0 metrics type doesn't exist yet (Phase 6)
- `from __future__ import annotations` used in protocols.py to resolve forward references cleanly

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

The pre-commit hook auto-staged and committed `resilience.py` and `security.py` changes as part of a larger commit (`c782223`) that also covered dead code deletions from Plan 01. Both files have the correct v2.0 content.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All 4 infrastructure modules are clean and v2.0-ready
- `exceptions.py` and `protocols.py` ready for import by all subsequent phases
- Phase 2 (config models) can import `ConfigError` immediately
- Phase 4 (PyTorch backend) can implement `ModelLoader` and `InferenceEngine` protocols
- Note: 19 v1.x source files still import old exception names — will be updated as each phase rewrites those modules

---
*Phase: 01-measurement-foundations*
*Completed: 2026-02-26*
