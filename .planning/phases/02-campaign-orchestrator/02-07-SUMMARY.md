---
phase: 02-campaign-orchestrator
plan: 07
subsystem: config-introspection
tags: [ssot, introspection, campaign-config, pydantic]
completed: 2026-01-29
duration: 2 min
requires: [02-01]
provides: [campaign-config-introspection]
affects: [02-08]
tech-stack:
  added: []
  patterns: [ssot-introspection, lazy-import]
key-files:
  created: []
  modified:
    - src/llenergymeasure/config/introspection.py
    - docs/generated/invalid-combos.md
    - docs/generated/parameter-support-matrix.md
decisions: []
---

# Phase 2 Plan 7: SSOT Campaign Config Introspection Summary

Extended introspection.py with 3 functions to auto-discover all Phase 2 CampaignConfig fields (33 params across 7 nested models).

## What Was Done

### Task 1: Add campaign config introspection functions

Added three new functions to `src/llenergymeasure/config/introspection.py`:

1. **`get_campaign_params()`** - Introspects full CampaignConfig with all nested models (CampaignGridConfig, CampaignHealthCheckConfig, CampaignColdStartConfig, CampaignDaemonConfig, CampaignIOConfig, CampaignExecutionConfig, CampaignScheduleConfig). Returns 33 params with metadata.

2. **`get_campaign_grid_params()`** - Subset for grid-related fields only (4 params: backends, models, shared, backend_overrides).

3. **`get_campaign_health_check_params()`** - Subset for health check fields only (5 params: enabled, interval_experiments, gpu_memory_threshold_pct, restart_on_failure, max_restarts).

All functions reuse the existing `get_params_from_model()` helper which handles recursive nested model introspection. Campaign config models are lazily imported inside functions to avoid circular imports.

## Deviations from Plan

None - plan executed exactly as written.

## Verification Results

- All 33 campaign params discovered (grid, health_check, cold_start, daemon, io, execution, schedule sections)
- Grid params: 4 fields
- Health check params: 5 fields
- Existing 44 introspection unit tests still pass
- ruff check + format + mypy all pass
- Pre-commit doc generation hooks pass (regenerated invalid-combos.md and parameter-support-matrix.md)

## Commits

| Hash | Description |
|------|-------------|
| 2579a75 | feat(02-07): add campaign config SSOT introspection functions |

## Next Phase Readiness

No blockers. Campaign config fields are now discoverable via SSOT introspection for use in CLI help, generated docs, and runtime validation.
