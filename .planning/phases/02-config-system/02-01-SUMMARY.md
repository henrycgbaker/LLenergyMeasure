---
phase: 02-config-system
plan: "01"
subsystem: config
tags: [pydantic, config, schema, validation]

requires:
  - phase: 01-measurement-foundations
    provides: exceptions hierarchy, state machine, core infrastructure

provides:
  - v2.0 ExperimentConfig with renamed fields (model, precision, n, passthrough_kwargs), extra=forbid, and three cross-validators
  - PyTorchConfig, VLLMConfig, TensorRTConfig with None-as-default pattern and extra=forbid
  - config/ssot.py with PRECISION_SUPPORT, DECODING_SUPPORT, DECODER_PARAM_SUPPORT dicts as single source of truth

affects:
  - 02-02 (YAML loader — depends on ExperimentConfig schema)
  - 02-03 (user config — depends on ExperimentConfig defaults)
  - 02-04 (introspection — depends on backend config models)

tech-stack:
  added: []
  patterns:
    - "None-as-default: backend config fields default to None; None means use backend's own runtime default"
    - "extra=forbid: all Pydantic models reject unknown fields, surfacing YAML typos immediately"
    - "forward-reference rebuild: ExperimentConfig uses TYPE_CHECKING imports + model_rebuild() to avoid circular deps with backend_configs"

key-files:
  created:
    - src/llenergymeasure/config/ssot.py
  modified:
    - src/llenergymeasure/config/models.py
    - src/llenergymeasure/config/backend_configs.py
    - src/llenergymeasure/config/__init__.py

key-decisions:
  - "None-as-default pattern for backend configs: all fields None by default, distinguishes researcher intent from backend defaults"
  - "extra=forbid on all config models: unknown YAML keys raise ValidationError immediately (catches typos)"
  - "Pydantic ValidationError passes through unchanged: not wrapped in ConfigError (per global decision)"
  - "Three cross-validators only: backend-section/backend mismatch, passthrough_kwargs key collision; cpu-precision validator deferred to Phase 4 (all current backends are GPU-only)"
  - "WarmupConfig simplified to n_warmup + thermal_floor_seconds: CV-based convergence detection is measurement concern (Phase 5), not config"
  - "BaselineConfig simplified to enabled + duration_seconds: cache_ttl and sample_interval are measurement concerns"
  - "SyntheticDatasetConfig and LoRAConfig added as new v2.0 sub-configs"

patterns-established:
  - "None-as-default: backend section fields use None to mean 'use backend runtime default'"
  - "SSOT dicts in ssot.py: capability maps imported by validators and introspection, never inlined"

requirements-completed: [CFG-01, CFG-02, CFG-03, CFG-04, CFG-05, CFG-06, CFG-08, CFG-09, CFG-10]

duration: 3min
completed: 2026-02-26
---

# Phase 2 Plan 01: Config Schema Summary

**v2.0 ExperimentConfig with field renames (model/precision/n/passthrough_kwargs), extra=forbid, three cross-validators, minimal backend configs with None-as-default pattern, and PRECISION_SUPPORT/DECODING_SUPPORT SSOT dicts**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-26T16:13:56Z
- **Completed:** 2026-02-26T16:16:54Z
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments

- Rewrote `ExperimentConfig` from v1.x (extra=allow, v1 field names) to v2.0 (extra=forbid, renamed fields, three structural cross-validators)
- Rewrote all three backend configs to the None-as-default pattern with extra=forbid, removing ~700 lines of v1.x-only content
- Created `config/ssot.py` as the single source of truth for backend capability constants (PRECISION_SUPPORT, DECODING_SUPPORT, DECODER_PARAM_SUPPORT)

## Task Commits

1. **Task 1: Rewrite ExperimentConfig with v2.0 schema** - `55845cf` (feat)
2. **Task 2: Rewrite backend configs with v2.0 schema** - `a301a23` (feat)
3. **Task 3: Create config/ssot.py** - `01bdefa` (feat)

## Files Created/Modified

- `src/llenergymeasure/config/models.py` - v2.0 ExperimentConfig with field renames, extra=forbid, cross-validators; new SyntheticDatasetConfig, LoRAConfig; simplified WarmupConfig/BaselineConfig
- `src/llenergymeasure/config/backend_configs.py` - PyTorchConfig, VLLMConfig, TensorRTConfig with None-as-default and extra=forbid; removed all v1.x sub-configs
- `src/llenergymeasure/config/ssot.py` - PRECISION_SUPPORT, DECODING_SUPPORT, DECODER_PARAM_SUPPORT dicts (created)
- `src/llenergymeasure/config/__init__.py` - Updated exports to match v2.0 classes (removed TrafficSimulation, introspection imports; added new sub-configs)

## Decisions Made

- **None-as-default for backend sections**: All backend config fields default to `None`. This clearly distinguishes "researcher explicitly set this" from "use backend runtime default" — important for result attribution and reproducibility.
- **WarmupConfig simplified**: CV-based convergence detection removed — that is a measurement algorithm concern (Phase 5), not a user-facing config concern. Config only specifies `n_warmup` and `thermal_floor_seconds`.
- **BaselineConfig simplified**: `cache_ttl_sec` and `sample_interval_ms` removed — sampling mechanics belong in the measurement engine, not the user config.
- **cpu-precision cross-validator not implemented**: All three current backends (pytorch, vllm, tensorrt) are GPU-only. The cpu backend is future scope. A comment documents this for Phase 4 pre-flight.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed `config/__init__.py` importing removed class `TrafficSimulation`**
- **Found during:** Task 1 verification
- **Issue:** `config/__init__.py` imported `TrafficSimulation` from models.py, which was removed in the v2.0 rewrite. This blocked import of any config module.
- **Fix:** Updated `__init__.py` to export only v2.0 classes (ExperimentConfig, DecoderConfig, WarmupConfig, BaselineConfig, SyntheticDatasetConfig, LoRAConfig, SAMPLING_PRESETS). Removed stale imports of introspection, loader, validation modules.
- **Files modified:** `src/llenergymeasure/config/__init__.py`
- **Verification:** `from llenergymeasure.config.models import ExperimentConfig` succeeds
- **Committed in:** `55845cf` (Task 1 commit)

**2. [Rule 3 - Blocking] Fixed `.git/info/exclude` pattern blocking `src/llenergymeasure/config/`**
- **Found during:** Task 1 commit
- **Issue:** The exclude file contained `config` (bare pattern) intended to exclude the git-internal `config` file. Git interpreted this as matching any path component named `config`, blocking `git add` for the entire config/ source directory.
- **Fix:** Changed `config` to `/config` (root-anchored pattern) so it only excludes the top-level `config` file created by the sandbox git environment.
- **Files modified:** `.git/info/exclude`
- **Verification:** `git add src/llenergymeasure/config/models.py` succeeds
- **Committed in:** N/A (exclude file is not tracked)

---

**Total deviations:** 2 auto-fixed (2 Rule 3 blocking issues)
**Impact on plan:** Both fixes were necessary to complete the tasks. No scope creep.

## Issues Encountered

None beyond the two blocking issues documented above.

## Next Phase Readiness

- `ExperimentConfig` v2.0 schema is in place and fully verified — Wave 2 plans (loader, user config, introspection) can proceed
- `config/ssot.py` ready for introspection module to consume
- Backend configs cleaned to M1 minimal; Phase 4.1 parameter audit will expand them

---
*Phase: 02-config-system*
*Completed: 2026-02-26*
