---
phase: 05-energy-measurement
plan: "03"
subsystem: core, backends
tags: [energy-integration, timeseries, measurement-warnings, pytorch-backend, parquet]
dependency_graph:
  requires:
    - core/energy_backends/__init__.py (select_energy_backend — Plan 01)
    - core/energy_backends/nvml.py (EnergyMeasurement — Plan 01)
    - core/flops.py (estimate_flops_palm — Plan 02)
    - core/warmup.py (warmup_until_converged, create_warmup_inference_fn — Plan 02)
    - core/baseline.py (measure_baseline_power, create_energy_breakdown)
    - config/models.py (EnergyConfig, WarmupConfig — Plan 02)
    - domain/experiment.py (ExperimentResult)
  provides:
    - core/timeseries.py: write_timeseries_parquet() — 1 Hz Parquet sidecar writer
    - core/measurement_warnings.py: collect_measurement_warnings() — four quality flags
    - core/backends/pytorch.py: full energy lifecycle wired into run()
  affects:
    - PyTorchBackend.run() now produces real energy/FLOPs/timeseries (no 0.0 placeholders)
tech_stack:
  added:
    - pyarrow (direct, no pandas) for Parquet timeseries writes
  patterns:
    - 1 Hz Parquet sidecar with locked 8-column schema
    - four informational quality warnings (never block experiments)
    - CUDA sync before energy stop (CM-15)
    - baseline measurement before model load (CM-17)
    - warmup tokens excluded from FLOPs (CM-28)
key_files:
  created:
    - src/llenergymeasure/core/timeseries.py
    - src/llenergymeasure/core/measurement_warnings.py
    - tests/unit/test_measurement_integration.py
  modified:
    - src/llenergymeasure/core/backends/pytorch.py
    - src/llenergymeasure/core/baseline.py
    - src/llenergymeasure/core/power_thermal.py
decisions:
  - "warmup_result not stored on ExperimentResult — it lives on RawProcessResult; accepted in _build_result() signature for future use when process_results are assembled"
  - "timeseries_path stored as file basename (not full path) to remain portable across storage locations"
  - "persistence mode unknown returns True (no spurious warning) — pynvml absence should not generate noise"
  - "total_tokens includes input+output to maintain backward compat; avg_energy_per_token_j uses output_tokens only (generation cost, not decoding)"
metrics:
  duration_minutes: 8
  tasks_completed: 2
  files_created: 3
  files_modified: 3
  tests_added: 20
  completed_date: "2026-02-26"
---

# Phase 05 Plan 03: Measurement Integration Summary

**One-liner:** PyTorchBackend.run() wired with full energy lifecycle — select_energy_backend + CUDA sync + create_energy_breakdown + estimate_flops_palm + write_timeseries_parquet + collect_measurement_warnings, replacing all 0.0 placeholders.

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Create timeseries writer and measurement warnings module | bba4bd5 | Done |
| 2 | Wire energy, FLOPs, timeseries, and warnings into PyTorchBackend.run() | 629fa93 | Done |

## Verification Results

1. `from llenergymeasure.core.timeseries import write_timeseries_parquet` — PASS
2. `from llenergymeasure.core.measurement_warnings import collect_measurement_warnings` — PASS
3. `pytest tests/unit/test_measurement_integration.py -x -v` — 20 passed, 0 failed
4. PyTorchBackend.run() lifecycle reviewed:
   - Baseline measured before model load: PASS
   - Energy tracking starts after warmup + thermal floor: PASS
   - torch.cuda.synchronize() before energy stop: PASS
   - FLOPs use measurement-only token counts (input_tokens, output_tokens from _run_batch): PASS
   - All 0.0 placeholders removed: PASS

## Files Produced

### Created
- `src/llenergymeasure/core/timeseries.py` — 1 Hz Parquet sidecar writer with locked 8-column schema
- `src/llenergymeasure/core/measurement_warnings.py` — four informational quality warnings with remediation advice
- `tests/unit/test_measurement_integration.py` — 20 unit tests (timeseries, warnings, backend wiring)

### Modified
- `src/llenergymeasure/core/backends/pytorch.py` — full energy lifecycle, WarmupResult return, separated input/output token tracking
- `src/llenergymeasure/core/baseline.py` — loguru replaced with stdlib logging (Rule 1 auto-fix)
- `src/llenergymeasure/core/power_thermal.py` — loguru replaced with stdlib logging (Rule 1 auto-fix)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] baseline.py used loguru (not a base dependency)**
- **Found during:** Task 2 — reading baseline.py before wiring it in
- **Issue:** `from loguru import logger` in baseline.py; base package uses stdlib logging only per project decisions
- **Fix:** Replaced with `import logging; logger = logging.getLogger(__name__)`; f-strings converted to `%s` format
- **Files modified:** `src/llenergymeasure/core/baseline.py`
- **Commit:** 629fa93

**2. [Rule 1 - Bug] power_thermal.py used loguru (not a base dependency)**
- **Found during:** Task 2 — reading power_thermal.py imports
- **Issue:** Same as above — loguru used in core module
- **Fix:** Replaced with stdlib logging
- **Files modified:** `src/llenergymeasure/core/power_thermal.py`
- **Commit:** 629fa93

**3. [Rule 1 - Bug] warmup_result not a field on ExperimentResult**
- **Found during:** Task 2 test execution — Pydantic validation error
- **Issue:** `warmup_result` field exists on `RawProcessResult` but NOT on `ExperimentResult`. Plan specified storing it in result but the model doesn't support it.
- **Fix:** `_build_result()` accepts `warmup_result` in signature (reserved for future process_results assembly) but does not pass it to `ExperimentResult()`. Test assertions updated accordingly.
- **Files modified:** `src/llenergymeasure/core/backends/pytorch.py`, `tests/unit/test_measurement_integration.py`
- **Commit:** 629fa93

## Self-Check: PASSED

- FOUND: `src/llenergymeasure/core/timeseries.py`
- FOUND: `src/llenergymeasure/core/measurement_warnings.py`
- FOUND: `tests/unit/test_measurement_integration.py`
- FOUND: `.planning/phases/05-energy-measurement/05-03-SUMMARY.md`
- FOUND: commit bba4bd5 (Task 1 — timeseries + warnings)
- FOUND: commit 629fa93 (Task 2 — pytorch backend wiring)
