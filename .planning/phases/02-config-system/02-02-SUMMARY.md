---
phase: 02-config-system
plan: "02"
subsystem: config
tags: [yaml, pydantic, validation, config-loading, error-handling]

requires:
  - phase: 02-config-system/02-01
    provides: ExperimentConfig v2.0 schema (extra=forbid, renamed fields), ConfigError exception

provides:
  - load_experiment_config(path, cli_overrides, user_config_defaults) -> ExperimentConfig
  - Collect-all-errors: all unknown field errors reported together (not one-at-a-time)
  - did-you-mean suggestions via Levenshtein distance for unknown YAML keys
  - CLI override merging at highest priority; user_config_defaults at lowest
  - Native YAML anchor support (yaml.safe_load)
  - deep_merge() utility for recursive dict merging

affects:
  - 02-03 (user config — calls load_experiment_config with user_config_defaults)
  - Phase 3 (library API — run_experiment wraps load_experiment_config)
  - Phase 7 (CLI — passes cli_overrides dict to load_experiment_config)

tech-stack:
  added: []
  patterns:
    - "Collect-all-errors before raise: unknown fields accumulated into list, single ConfigError raised"
    - "ValidationError pass-through: Pydantic field errors not wrapped — they are Pydantic's domain"
    - "version field stripped silently: optional YAML version: '2.0' key removed before Pydantic"

key-files:
  created: []
  modified:
    - src/llenergymeasure/config/loader.py

key-decisions:
  - "Drop _extends inheritance: replaced by native YAML anchors — yaml.safe_load handles &/* natively, no custom resolution needed"
  - "ConfigError for unknown keys, ValidationError for bad values: clear boundary — structural/schema errors are ConfigError, field constraint errors pass through as ValidationError"
  - "did-you-mean Levenshtein max_distance=3: catches common typos (modell, bakend) without false positives on completely unrelated keys"
  - "version field stripped silently: allows researchers to annotate YAML with version: '2.0' without triggering unknown-field error"

patterns-established:
  - "Error collection pattern: accumulate errors list, join and raise as single ConfigError for UX"
  - "_unflatten for dotted CLI keys: pytorch.batch_size -> {pytorch: {batch_size: ...}} before merge"

requirements-completed: [CFG-07, CFG-18, CFG-19, CFG-20, CFG-21, CFG-22]

duration: 3min
completed: 2026-02-26
---

# Phase 2 Plan 02: Config Loader Summary

**YAML loader with collect-all-errors, ConfigError + did-you-mean, CLI override merging, and native YAML anchor support — removing ~275 lines of v1.x provenance/inheritance code**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-26T16:20:00Z
- **Completed:** 2026-02-26T16:23:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Rewrote `loader.py` from v1.x (provenance tracking, `_extends` inheritance, warnings mode) to v2.0 (clean `load_experiment_config()`, collect-all-errors, ConfigError, did-you-mean)
- Removed 275 lines of v1.x code: `resolve_inheritance`, `validate_config`, `load_config`, `load_config_with_provenance`, `has_blocking_warnings`, `get_pydantic_defaults`, `ConfigWarning` usage
- Added `_load_file`, `_unflatten`, `_did_you_mean`, `_levenshtein` helpers as clean private functions

## Task Commits

1. **Task 1: Rewrite loader.py with v2.0 loading contract** - `fe1dd9b` (feat)

## Files Created/Modified

- `src/llenergymeasure/config/loader.py` - Complete rewrite: `load_experiment_config()` public API + `deep_merge()` + private helpers; exports `["load_experiment_config", "deep_merge"]`

## Decisions Made

- **Drop `_extends` inheritance entirely**: YAML native anchors (`&base`, `*base`) already handle config reuse natively. Custom `_extends` resolution was 60 lines of code that yaml.safe_load renders unnecessary.
- **`_extends` in YAML now triggers ConfigError**: It's an unknown field (not a special key anymore) so researchers with old-style configs get a clear error with did-you-mean pointing to the correct YAML anchor syntax.
- **`version` field stripped silently**: `merged.pop("version", None)` before Pydantic construction. Researchers can annotate their YAML files with `version: "2.0"` for their own documentation without causing errors.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `load_experiment_config()` is ready for both the CLI (Phase 7, passes `cli_overrides`) and library API (Phase 3, wraps with `run_experiment()`)
- `deep_merge()` is available for Plan 03 (user config defaults merging)
- Plan 03 (user config) can now call `load_experiment_config(path, user_config_defaults=user_defaults)`

---
*Phase: 02-config-system*
*Completed: 2026-02-26*
