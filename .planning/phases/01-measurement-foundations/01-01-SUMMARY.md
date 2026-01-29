---
phase: 01-measurement-foundations
plan: 01
subsystem: domain
tags: [pydantic, schema-v3, energy-breakdown, thermal-throttle, warmup, environment-metadata]

# Dependency graph
requires: []
provides:
  - "Schema v3 domain models (EnergyBreakdown, ThermalThrottleInfo, WarmupResult, EnvironmentMetadata)"
  - "Configuration extensions (WarmupConfig, BaselineConfig, TimeSeriesConfig)"
  - "SCHEMA_VERSION 3.0.0"
affects:
  - 01-02 (warmup convergence engine uses WarmupConfig + WarmupResult)
  - 01-03 (baseline power uses BaselineConfig + EnergyBreakdown)
  - 01-04 (environment capture uses EnvironmentMetadata)
  - 01-05 (thermal monitoring uses ThermalThrottleInfo)
  - 01-06 (time-series uses TimeSeriesConfig)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Optional fields with defaults for backwards-compatible schema evolution"
    - "Sub-configuration models on ExperimentConfig for measurement features"

key-files:
  created:
    - src/llenergymeasure/domain/environment.py
  modified:
    - src/llenergymeasure/constants.py
    - src/llenergymeasure/domain/metrics.py
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/config/models.py
    - docs/generated/config-reference.md
    - docs/generated/invalid-combos.md
    - docs/generated/parameter-support-matrix.md

key-decisions:
  - "Removed from __future__ import annotations from domain models to avoid Pydantic v2 model rebuild issues"
  - "Used runtime imports (not TYPE_CHECKING) for new types in experiment.py so Pydantic can resolve them"

patterns-established:
  - "Schema v3 additive evolution: all new fields have default=None, existing code unchanged"
  - "Sub-config pattern: WarmupConfig/BaselineConfig/TimeSeriesConfig as fields on ExperimentConfig with default_factory"

# Metrics
duration: 7min
completed: 2026-01-29
---

# Phase 1 Plan 1: Schema v3 Domain Models Summary

**Schema v3 with EnergyBreakdown, ThermalThrottleInfo, WarmupResult, EnvironmentMetadata models and WarmupConfig/BaselineConfig/TimeSeriesConfig on ExperimentConfig**

## Performance

- **Duration:** 7 min
- **Started:** 2026-01-29T14:09:35Z
- **Completed:** 2026-01-29T14:16:26Z
- **Tasks:** 2/2
- **Files modified:** 8 (4 source + 1 new + 3 auto-generated docs)

## Accomplishments

- Schema v3 domain models: EnergyBreakdown (baseline-adjusted energy), ThermalThrottleInfo (GPU throttling), WarmupResult (convergence detection)
- EnvironmentMetadata with GPU, CUDA, thermal, CPU, container sub-models and summary_line property
- WarmupConfig with CV-based convergence, BaselineConfig with cache TTL, TimeSeriesConfig for data collection
- SCHEMA_VERSION bumped to 3.0.0; all 791 existing unit tests pass

## Task Commits

1. **Task 1: Schema v3 domain models** - `296e0eb` (feat)
2. **Task 2: Configuration extensions** - `507497a` (feat)

## Files Created/Modified

- `src/llenergymeasure/constants.py` - SCHEMA_VERSION bumped to 3.0.0
- `src/llenergymeasure/domain/metrics.py` - Added EnergyBreakdown, ThermalThrottleInfo, WarmupResult
- `src/llenergymeasure/domain/environment.py` - New file: EnvironmentMetadata with sub-models
- `src/llenergymeasure/domain/experiment.py` - Added v3 optional fields to RawProcessResult and AggregatedResult
- `src/llenergymeasure/config/models.py` - Added WarmupConfig, BaselineConfig, TimeSeriesConfig
- `docs/generated/config-reference.md` - Auto-regenerated (SSOT)
- `docs/generated/invalid-combos.md` - Auto-regenerated (SSOT)
- `docs/generated/parameter-support-matrix.md` - Auto-regenerated (SSOT)

## Decisions Made

- Removed `from __future__ import annotations` from domain model files to avoid Pydantic v2 model resolution issues with `datetime` in nested models
- Used string forward references (`"Timestamps"`, `"AggregatedResult"`) only where needed for self-referential types, runtime imports elsewhere

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed Pydantic v2 incompatibility with `from __future__ import annotations`**
- **Found during:** Task 1 (Schema v3 domain models)
- **Issue:** Adding `from __future__ import annotations` to metrics.py and environment.py caused Pydantic v2 to fail resolving `datetime` during model rebuild
- **Fix:** Removed `from __future__ import annotations` from domain files; used string forward refs only for self-referential types
- **Files modified:** metrics.py, environment.py, experiment.py
- **Verification:** All imports and model construction verified working
- **Committed in:** 296e0eb (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary fix for Pydantic v2 runtime compatibility. No scope creep.

## Issues Encountered

- Pre-commit SSOT hooks regenerated documentation files when config/models.py changed -- required re-staging and recommitting (expected behaviour from the SSOT auto-generation pipeline)

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All domain models and config extensions ready for Phase 1 plans 02-06
- WarmupConfig ready for 01-02 (warmup convergence engine)
- BaselineConfig + EnergyBreakdown ready for 01-03 (baseline power measurement)
- EnvironmentMetadata ready for 01-04 (environment capture)
- ThermalThrottleInfo ready for 01-05 (thermal monitoring)
- TimeSeriesConfig ready for 01-06 (time-series collection)

---
*Phase: 01-measurement-foundations*
*Completed: 2026-01-29*
