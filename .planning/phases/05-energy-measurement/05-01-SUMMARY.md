---
phase: 05-energy-measurement
plan: "01"
subsystem: energy-backends
tags: [energy, nvml, zeus, codecarbon, auto-selection, trapezoidal-integration]
dependency_graph:
  requires:
    - core/power_thermal.py (PowerThermalSampler)
    - exceptions.py (ConfigError)
    - protocols.py (EnergyBackend protocol)
  provides:
    - NVMLBackend: poll + trapezoidal integrate to joules
    - ZeusBackend: ZeusMonitor wrapper
    - EnergyMeasurement: shared result dataclass
    - select_energy_backend(): auto-selection with explicit override and null disable
  affects:
    - Plan 03 (measurement integration calls select_energy_backend)
tech_stack:
  added:
    - zeus>=0.13.1 (fixes abandoned zeus-ml package)
  patterns:
    - deferred imports (no module-level pynvml/zeus at import time)
    - trapezoidal integration of consecutive power samples
    - importlib.util.find_spec for soft dependency probing
key_files:
  created:
    - src/llenergymeasure/core/energy_backends/nvml.py
    - src/llenergymeasure/core/energy_backends/zeus.py
    - tests/unit/test_energy_backends_v2.py
  modified:
    - src/llenergymeasure/core/energy_backends/__init__.py
    - pyproject.toml
decisions:
  - "NVMLBackend and ZeusBackend both return EnergyMeasurement (shared dataclass in nvml.py)"
  - "ZeusBackend imports EnergyMeasurement from .nvml — avoids circular import and keeps dataclass co-located with primary backend"
  - "select_energy_backend() probes zeus via importlib.util.find_spec (not is_available()) to avoid instantiation overhead in auto mode"
  - "Legacy registry retained for backward compat — NVMLBackend added to _register_default_backends()"
metrics:
  duration_minutes: 3
  tasks_completed: 2
  files_created: 3
  files_modified: 2
  tests_added: 17
  completed_date: "2026-02-26"
---

# Phase 5 Plan 01: Energy Backends Summary

NVMLBackend (trapezoidal power integration) + ZeusBackend (ZeusMonitor wrapper) with auto-selection function implementing Zeus > NVML > CodeCarbon priority, explicit override, and null disable.

## Tasks Completed

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | NVMLBackend, ZeusBackend, auto-selection, pyproject.toml | 29ca5a2 | Done |
| 2 | Unit tests for energy backends and auto-selection | 7f78dcb | Done |

## Verification Results

1. `from llenergymeasure.core.energy_backends import select_energy_backend, NVMLBackend, ZeusBackend, EnergyMeasurement` — PASS
2. `select_energy_backend(None) is None` — PASS
3. `grep 'zeus>=0.13.1' pyproject.toml` — PASS
4. `pytest tests/unit/test_energy_backends_v2.py -x -v` — 17 passed, 0 failed

## Files Produced

### Created
- `src/llenergymeasure/core/energy_backends/nvml.py` — NVMLBackend + EnergyMeasurement dataclass
- `src/llenergymeasure/core/energy_backends/zeus.py` — ZeusBackend wrapping ZeusMonitor
- `tests/unit/test_energy_backends_v2.py` — 17 unit tests

### Modified
- `src/llenergymeasure/core/energy_backends/__init__.py` — Added NVMLBackend, ZeusBackend, select_energy_backend, EnergyMeasurement to exports; NVMLBackend registered in legacy registry
- `pyproject.toml` — Fixed zeus extra: `zeus-ml>=0.10` -> `zeus>=0.13.1`

## Deviations from Plan

None — plan executed exactly as written. The 17 tests exceed the 9-test minimum specified in the plan (added 8 additional edge-case tests: `test_nvml_is_available_import_error`, `test_select_backend_explicit_zeus_unavailable_raises`, `test_select_backend_unknown_name_raises`, `test_select_backend_auto_priority_codecarbon_fallback`, `test_select_backend_auto_returns_none_when_nothing_available`, `test_nvml_trapezoidal_skips_none_power`, `test_nvml_trapezoidal_single_sample`).

## Self-Check: PASSED

- [x] `src/llenergymeasure/core/energy_backends/nvml.py` — exists
- [x] `src/llenergymeasure/core/energy_backends/zeus.py` — exists
- [x] `tests/unit/test_energy_backends_v2.py` — exists
- [x] Commit 29ca5a2 — exists (feat: NVMLBackend, ZeusBackend)
- [x] Commit 7f78dcb — exists (test: energy backend tests)
