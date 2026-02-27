---
phase: 02-config-system
plan: 03
subsystem: config
tags: [pydantic, platformdirs, yaml, env-vars, user-config]

requires:
  - phase: 02-config-system/02-01
    provides: ExperimentConfig, ConfigError, exceptions hierarchy

provides:
  - UserConfig Pydantic model with 6 nested sub-configs (extra=forbid)
  - get_user_config_path() XDG-compliant path via platformdirs
  - load_user_config() with missing-file-safe defaults and ConfigError on bad schema
  - _apply_env_overrides() for LLEM_* env var override layer

affects: [cli, runner-selection, execution-gaps, energy-backend-selection]

tech-stack:
  added: []
  patterns:
    - "XDG config path via platformdirs.user_config_dir"
    - "Env var override layer applied after file load, before CLI flags"
    - "model_copy(update=...) for immutable Pydantic v2 updates"
    - "ConfigError wraps ValidationError with field path context for researcher clarity"

key-files:
  created: []
  modified:
    - src/llenergymeasure/config/user_config.py

key-decisions:
  - "Silently ignore invalid LLEM_CARBON_INTENSITY / LLEM_DATACENTER_PUE env var values (same as not set)"
  - "Runner format validator checks singularity: prefix separately for explicit not-yet-supported message"
  - "get_user_config_path() defers platformdirs import to function body (no import-time cost)"

patterns-established:
  - "load_user_config: missing file = zero-config (return defaults), invalid schema = ConfigError"
  - "Env var overrides use model_copy(update=...) — preserves immutability, no mutation"

requirements-completed: [CFG-23, CFG-24, CFG-25, CFG-26]

duration: 2min
completed: 2026-02-26
---

# Phase 2 Plan 3: UserConfig Summary

**XDG-aware UserConfig Pydantic model with runner/measurement/execution sections and LLEM_* env var override layer replacing v1.x .lem-config.yaml schema**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-26T16:22:23Z
- **Completed:** 2026-02-26T16:23:34Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Replaced v1.x `ThermalGapConfig`/`DockerConfig`/`NotificationsConfig` with v2.0 schema (6 sub-configs)
- XDG-compliant config path via `platformdirs.user_config_dir("llenergymeasure")` instead of `.lem-config.yaml` in cwd
- `load_user_config()` with silent missing-file behaviour and `ConfigError` (not `ValueError`) for bad YAML/schema
- Env var override layer: `LLEM_RUNNER_{PYTORCH,VLLM,TENSORRT}`, `LLEM_CARBON_INTENSITY`, `LLEM_DATACENTER_PUE`, `LLEM_NO_PROMPT`
- Runner format validator: `local` or `docker:<image>` only, explicit error for `singularity:` prefix

## Task Commits

1. **Task 1: Rewrite UserConfig and load_user_config** - `f2d1415` (feat)

## Files Created/Modified

- `src/llenergymeasure/config/user_config.py` - Complete rewrite: v2.0 schema, XDG path, env var overrides, ConfigError error handling

## Decisions Made

- Silent ignore on unparseable `LLEM_CARBON_INTENSITY` / `LLEM_DATACENTER_PUE` float values (matches "same as not set" semantics for env vars)
- `singularity:` prefix check placed before the generic format check so it emits a "not yet supported" message rather than a generic format error

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `UserConfig` and `load_user_config()` ready for Plan 04 (sweep resolution / StudyConfig)
- Env var override layer in place for CI/HPC use cases
- `get_user_config_path()` available for `llem config` display command

---
*Phase: 02-config-system*
*Completed: 2026-02-26*
