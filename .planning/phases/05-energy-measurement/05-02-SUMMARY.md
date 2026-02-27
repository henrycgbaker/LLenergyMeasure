---
phase: 05-energy-measurement
plan: 02
subsystem: config, domain, core
tags: [warmup, energy-config, flops, palm-formula, unit-tests]
dependency_graph:
  requires: []
  provides:
    - WarmupConfig v2 with CV opt-in fields and n_warmup=5
    - EnergyConfig sub-model on ExperimentConfig
    - measurement_warnings field on ExperimentResult
    - thermal_floor_wait_s field on WarmupResult
    - PaLM FLOPs formula (estimate_flops_palm) as v2.0 primary API
    - palm_formula Literal on FlopsResult.method
  affects:
    - src/llenergymeasure/config/models.py
    - src/llenergymeasure/domain/metrics.py
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/core/flops.py
    - src/llenergymeasure/core/warmup.py
tech_stack:
  added: []
  patterns:
    - PaLM/Chinchilla 2*N_non_embed*T FLOPs formula (primary v2.0 method)
    - Opt-in CV convergence detection (additive to fixed warmup)
    - stdlib logging throughout (no loguru in base package)
key_files:
  created:
    - tests/unit/test_warmup_v2.py
    - tests/unit/test_flops_v2.py
  modified:
    - src/llenergymeasure/config/models.py
    - src/llenergymeasure/domain/metrics.py
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/core/flops.py
    - src/llenergymeasure/core/warmup.py
decisions:
  - "WarmupConfig.max_prompts used (not max_warmup_prompts) for field name consistency with warmup.py loop variable"
  - "Fixed-mode warmup iterates n_warmup times (not max_prompts) — n_warmup is the researcher-facing count"
  - "estimate_flops_palm precision='n/a' — floating-point precision does not affect FLOPs count (forward pass)"
metrics:
  duration_min: 5
  completed_date: "2026-02-26"
  tasks_completed: 2
  files_modified: 7
---

# Phase 05 Plan 02: WarmupConfig v2, EnergyConfig, PaLM FLOPs Summary

**One-liner:** WarmupConfig upgraded to n_warmup=5 with opt-in CV convergence, EnergyConfig added to ExperimentConfig, PaLM formula replaces calflops fallback chain as v2.0 primary FLOPs estimator.

## What Was Built

### Task 1: Config/domain model updates and flops.py rewrite

**WarmupConfig v2** (`src/llenergymeasure/config/models.py`):
- `n_warmup` default changed from 3 → 5 (HIGH confidence: DeepSpeed/Zeus/AIEnergyScore all use 5–10)
- `thermal_floor_seconds` constraint changed from `ge=0.0` → `ge=30.0` (MLPerf Power mandates 60s; 30s is the minimum enforced floor)
- Added `enabled: bool = True` for early-return path
- Added CV convergence opt-in fields: `convergence_detection`, `cv_threshold`, `max_prompts`, `window_size`, `min_prompts`

**EnergyConfig** (new sub-model):
- `backend: Literal["auto", "nvml", "zeus", "codecarbon"] | None = "auto"`
- Wired onto `ExperimentConfig.energy` field
- `backend=None` (YAML `null`) disables energy measurement

**WarmupResult** (`domain/metrics.py`):
- Added `thermal_floor_wait_s: float = 0.0` — populated by caller after `time.sleep()`, not by `warmup_until_converged()`

**FlopsResult** (`domain/metrics.py`):
- Added `"palm_formula"` to `method` Literal

**ExperimentResult** (`domain/experiment.py`):
- Added `measurement_warnings: list[str] = []`

**flops.py rewrite** (`core/flops.py`):
- Removed module-level `import torch` and `from loguru import logger`
- Added `_count_non_embedding_params()` — excludes layers with "embed" in name (case-insensitive)
- Added `estimate_flops_palm()` as v2.0 primary API
- `FlopsEstimator` kept as LEGACY fallback (docstring updated)
- `estimate_flops()` kept as backward-compat wrapper
- All `logger.debug/info/error` calls use stdlib `%s` format (not f-strings)

**warmup.py** (`core/warmup.py`):
- Replaced `from loguru import logger` with stdlib `logging.getLogger(__name__)`
- Fixed-mode loop now uses `config.n_warmup` iterations (was `config.max_prompts`)
- CV mode still uses `config.max_prompts` as safety cap

### Task 2: Unit tests (48 passing)

**test_warmup_v2.py** (31 tests):
- WarmupConfig defaults, CV field bounds, extra=forbid
- `warmup_until_converged()` fixed mode (exact n_warmup iterations) and CV mode convergence
- `WarmupResult.thermal_floor_wait_s` defaults and round-trips
- EnergyConfig backend values (auto, nvml, zeus, codecarbon, None)
- ExperimentConfig.energy field wiring
- ExperimentResult.measurement_warnings defaults to empty list

**test_flops_v2.py** (17 tests):
- `_count_non_embedding_params()` correctness and embedding exclusion
- `estimate_flops_palm()` formula: `2 * non_embed_params * batch_size * (n_input + n_output)`
- Method/confidence/precision fields
- Batch size multiplier
- `FlopsResult` literal validation for `palm_formula` and legacy methods
- `estimate_flops` backward-compat import

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed-mode warmup iterated max_prompts instead of n_warmup**
- **Found during:** Task 1 — reading existing warmup.py after WarmupConfig update
- **Issue:** The for-loop in `warmup_until_converged()` used `range(config.max_prompts)` for both fixed and CV modes. After separating `n_warmup` (fixed count) from `max_prompts` (CV safety cap), fixed mode must iterate `n_warmup` times, not `max_prompts`.
- **Fix:** Added `iteration_limit = config.n_warmup if fixed_mode else config.max_prompts` and updated the loop range accordingly.
- **Files modified:** `src/llenergymeasure/core/warmup.py`
- **Commit:** 83190a5

## Self-Check: PASSED

- FOUND: tests/unit/test_warmup_v2.py
- FOUND: tests/unit/test_flops_v2.py
- FOUND: .planning/phases/05-energy-measurement/05-02-SUMMARY.md
- FOUND: commit 83190a5 (Task 1)
- FOUND: commit 0ff5392 (Task 2)
