---
phase: 16-gpu-memory-verification
plan: "01"
subsystem: measurement
tags: [gpu, nvml, measurement-quality, pynvml, study-runner]
dependency_graph:
  requires: []
  provides: [check_gpu_memory_residual, nvml-pre-dispatch-check]
  affects: [study/runner.py]
tech_stack:
  added: []
  patterns: [lazy-local-import, graceful-degradation, stdlib-logging]
key_files:
  created:
    - src/llenergymeasure/study/gpu_memory.py
    - tests/unit/test_gpu_memory.py
  modified:
    - src/llenergymeasure/study/runner.py
    - tests/unit/test_study_runner.py
decisions:
  - "Local import in _run_one() keeps pynvml dependency lazy; avoids module-level ImportError when pynvml not installed"
  - "Patch llenergymeasure.study.gpu_memory.check_gpu_memory_residual (source) not runner module attribute (local import)"
  - "100 MB threshold hardcoded for M3; configurability deferred until researcher demand"
metrics:
  duration_seconds: 173
  completed_date: "2026-02-27"
  tasks_completed: 2
  tasks_planned: 2
  files_created: 2
  files_modified: 2
---

# Phase 16 Plan 01: GPU Memory Residual Check Summary

NVML-based pre-dispatch GPU memory check with warning threshold, wired into StudyRunner before each experiment subprocess starts.

## What Was Built

`gpu_memory.py` adds a single public function `check_gpu_memory_residual()` that queries pynvml before each experiment dispatch. If residual GPU memory exceeds 100 MB, a warning is logged — catching driver-level leaks or third-party GPU processes that could inflate `peak_memory_mb`. If pynvml is absent or any NVML error occurs, the check skips silently.

The function is called in `StudyRunner._run_one()` immediately before `p.start()`, ensuring it runs in the parent process where host GPU state is visible. Subprocess isolation via `spawn` guarantees each child gets a clean CUDA context regardless.

## Tasks Completed

| Task | Description | Commit | Files |
|------|-------------|--------|-------|
| 1 | Create gpu_memory module with NVML residual check | 1c61208 | gpu_memory.py, test_gpu_memory.py |
| 2 | Wire GPU memory check into StudyRunner._run_one() | e2a0bee | runner.py, test_study_runner.py |

## Decisions Made

1. **Local import in `_run_one()`** — `from llenergymeasure.study.gpu_memory import check_gpu_memory_residual` is inside the method body. This keeps pynvml import lazy, avoiding module-level ImportError on hosts without pynvml installed.

2. **Patch source module not runner attribute** — Tests must patch `llenergymeasure.study.gpu_memory.check_gpu_memory_residual` (the function's own module) rather than `llenergymeasure.study.runner.check_gpu_memory_residual`, since local imports don't bind to the importing module's namespace.

3. **100 MB threshold hardcoded** — NVML driver overhead is 50-80 MB on modern GPUs; 100 MB accommodates that while flagging real residuals. Not configurable in M3; deferred to when researcher demand arises.

## Verification

- `check_gpu_memory_residual` importable from `llenergymeasure.study.gpu_memory`: confirmed
- Type check (mypy): no issues
- Lint (ruff): clean
- 6 unit tests pass: clean state, warning, custom threshold, pynvml absent, NVML error, shutdown called
- 1 wiring test confirms GPU check fires before `p.start()`
- Full unit suite: 543 passed (no regression, +7 new tests)

## Requirements Satisfied

- MEAS-01: NVML GPU memory queried before each experiment dispatch
- MEAS-02: Warning logged when residual memory exceeds 100 MB threshold

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- gpu_memory.py: FOUND
- test_gpu_memory.py: FOUND
- commit 1c61208: FOUND
- commit e2a0bee: FOUND
