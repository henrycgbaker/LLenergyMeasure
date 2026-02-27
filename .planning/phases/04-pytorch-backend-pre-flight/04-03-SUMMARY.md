---
phase: 04-pytorch-backend-pre-flight
plan: 03
subsystem: runner
tags: [runner, wiring, preflight, thermal-throttle, pynvml, api, tests]

# Dependency graph
requires:
  - phase: 04-pytorch-backend-pre-flight
    plan: 01
    provides: run_preflight() function
  - phase: 04-pytorch-backend-pre-flight
    plan: 02
    provides: get_backend(), PyTorchBackend, InferenceBackend Protocol

provides:
  - Real _run() implementation: preflight → get_backend().run() → StudyResult
  - PowerThermalSampler integrated into PyTorchBackend measurement path (CM-34)
  - ExperimentResult.thermal_throttle populated from sampler output
  - 6 new _run() wiring tests (preflight-per-config, backend name, return type, error propagation, e2e)

affects:
  - Phase 5 (energy backends) — _run() is the measurement entry point
  - CLI (Phase 7) — run_experiment() is now end-to-end functional with mocked/real backends

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Deferred imports inside _run(): no new module-level imports in _api.py"
    - "Error propagation: PreFlightError and BackendError pass through _run() unchanged"
    - "Tuple return from _run_measurement(): (MeasurementData, ThermalThrottleInfo)"
    - "PowerThermalSampler as context manager wrapping the full measurement loop"

key-files:
  created: []
  modified:
    - src/llenergymeasure/_api.py
    - src/llenergymeasure/core/backends/pytorch.py
    - tests/unit/test_api.py

key-decisions:
  - "_run() uses deferred imports — no new module-level imports added to _api.py"
  - "PowerThermalSampler wraps the entire measurement loop (not per-batch) — single ThermalThrottleInfo per experiment"
  - "_run_measurement() returns tuple (MeasurementData, ThermalThrottleInfo) — clean separation of concerns"
  - "ThermalThrottleInfo imported at module level in pytorch.py for type annotation clarity"
  - "No try/except in _run() — errors propagate naturally to caller (CLI phase adds display)"

requirements-completed: [CM-34]

# Metrics
duration: 3min
completed: 2026-02-26
---

# Phase 4 Plan 03: Runner Integration Summary

**_run() wired to preflight + PyTorchBackend with PowerThermalSampler integration (CM-34) — run_experiment() is now end-to-end functional**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T18:35:07Z
- **Completed:** 2026-02-26T18:38:13Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `_run()` stub replaced with real pipeline: `run_preflight(config)` → `get_backend(config.backend).run(config)` → `StudyResult`
- Pre-flight runs once per experiment config (each config may differ in model/backend)
- `PowerThermalSampler` from `core/power_thermal.py` integrated as context manager wrapping the full measurement loop in `PyTorchBackend._run_measurement()`
- `_run_measurement()` now returns `(MeasurementData, ThermalThrottleInfo)` tuple
- `_build_result()` populates `ExperimentResult.thermal_throttle` from the sampler output
- `ExperimentResult.thermal_throttle` field confirmed present in `domain/experiment.py` (line 238) — no change needed
- 6 new wiring tests added; all 75 Phase 4 tests pass (32 preflight + 13 env snapshot + 25 backend protocol + 18 API)
- `run_experiment(ExperimentConfig(model="gpt2"))` flows end-to-end through the real pipeline (mocked backend in tests, real backend on GPU)

## Task Commits

1. **Task 1: Replace _run() stub and integrate thermal throttle detection** — `0145000` (feat)
2. **Task 2: Update API tests for real _run() pipeline** — `68c45d2` (test)

## Files Modified

- `src/llenergymeasure/_api.py` — `_run()` stub replaced with real implementation: per-config preflight + get_backend().run() loop; all imports deferred inside function body
- `src/llenergymeasure/core/backends/pytorch.py` — `_run_measurement()` returns `(MeasurementData, ThermalThrottleInfo)` tuple via `PowerThermalSampler` context manager; `_build_result()` accepts and sets `thermal_throttle`; `ThermalThrottleInfo` imported at module level for type annotations; `from __future__ import annotations` added
- `tests/unit/test_api.py` — 6 new `_run()` wiring tests: `test_run_calls_preflight_once_per_config`, `test_run_calls_get_backend_with_correct_name`, `test_run_returns_study_result`, `test_run_propagates_preflight_error`, `test_run_propagates_backend_error`, `test_run_experiment_end_to_end_mocked`

## Decisions Made

- **Deferred imports in `_run()`** — all imports (`get_backend`, `run_preflight`) stay inside the function body; no new module-level imports added to `_api.py`
- **`PowerThermalSampler` wraps the entire measurement loop** — single `ThermalThrottleInfo` per experiment (not per batch). The sampler runs during the full batch loop.
- **`_run_measurement()` returns a tuple** — clean separation of `_MeasurementData` (throughput) from `ThermalThrottleInfo` (thermal). `_build_result()` accepts both as explicit parameters.
- **`ThermalThrottleInfo` imported at module level in `pytorch.py`** — used in type annotations for `_run_measurement()` and `_build_result()`. `from __future__ import annotations` added to enable forward references if needed.
- **No try/except in `_run()`** — `PreFlightError` and `BackendError` propagate unchanged to the caller. Phase 7 (CLI) adds user-facing error display.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None.

## Next Phase Readiness

- `run_experiment()` is end-to-end functional: `run_experiment(ExperimentConfig(model="gpt2"))` on a GPU machine produces a real `ExperimentResult`
- Phase 4 complete — all 3 plans done (pre-flight, backend protocol, runner wiring)
- Phase 5 (energy backends) replaces the `0.0` placeholder energy fields with real Zeus/NVML/CodeCarbon measurements

---

*Phase: 04-pytorch-backend-pre-flight*
*Completed: 2026-02-26*

## Self-Check: PASSED

- FOUND: src/llenergymeasure/_api.py
- FOUND: src/llenergymeasure/core/backends/pytorch.py
- FOUND: tests/unit/test_api.py
- FOUND: .planning/phases/04-pytorch-backend-pre-flight/04-03-SUMMARY.md
- FOUND: commit 0145000 (feat - _run() wiring)
- FOUND: commit 68c45d2 (test - API wiring tests)
