---
plan: 11-01
phase: 11-subprocess-isolation-and-studyrunner
subsystem: study
status: complete
tags: [subprocess, multiprocessing, spawn, cuda-isolation, tdd]
files_modified:
  - src/llenergymeasure/study/runner.py
  - src/llenergymeasure/study/__init__.py
  - tests/unit/test_study_runner.py
tests_added: 10
tests_total: 494
dependency_graph:
  requires:
    - llenergymeasure.study.manifest (ManifestWriter)
    - llenergymeasure.study.grid (apply_cycles, CycleOrder)
    - llenergymeasure.domain.experiment (compute_measurement_config_hash)
  provides:
    - StudyRunner (subprocess dispatch core)
    - _run_experiment_worker (child process entry point stub)
    - _calculate_timeout (heuristic: max(n*2, 600))
  affects:
    - src/llenergymeasure/study/__init__.py (StudyRunner added to exports)
tech_stack:
  added:
    - multiprocessing.get_context('spawn') for CUDA-safe subprocess isolation
    - multiprocessing.Pipe for IPC (child→parent result transport)
    - threading.Thread (daemon consumer for progress queue)
    - signal.SIG_IGN (child ignores SIGINT; parent owns the signal)
  patterns:
    - spawn-not-fork for all subprocess creation
    - daemon=False for clean CUDA teardown
    - SIGKILL (not SIGTERM) on timeout
    - structured error dict {type, message, traceback} through Pipe on failure
key_files:
  created:
    - src/llenergymeasure/study/runner.py
    - tests/unit/test_study_runner.py
  modified:
    - src/llenergymeasure/study/__init__.py
decisions:
  - spawn context (not fork): fork causes silent CUDA state corruption (CP-1 decision)
  - daemon=False: ensures clean CUDA teardown when parent exits unexpectedly (CP-4)
  - SIGKILL on timeout: SIGTERM may be ignored by hung CUDA operations
  - Pipe-only IPC (no temp files): ExperimentResult fits in Pipe buffer for M2 sizes
  - _run_experiment_worker is a stub (raises NotImplementedError); wired to real backend in Phase 12
  - cycle tracking (cycle=1 hardcoded): full per-cycle tracking deferred to Phase 12 wiring
requirements-completed: [STU-01, STU-02, STU-03, STU-04]
metrics:
  duration: "~3 min"
  completed: "2026-02-27T18:11:32Z"
  tasks: 3
  files: 3
---

# Phase 11 Plan 01: StudyRunner Subprocess Isolation Summary

**One-liner:** StudyRunner using multiprocessing spawn context with Pipe IPC, SIGKILL timeouts, structured failure dicts, and ManifestWriter integration — all paths covered by 10 mock-based unit tests.

## What Was Built

`StudyRunner` is the subprocess dispatch core for Phase 11. Every experiment in a study runs in a freshly spawned subprocess with a clean CUDA context. Results travel parent←child via `multiprocessing.Pipe`. Failures are structured (never fatal to the study). Timeouts use SIGKILL.

### Key components

**`src/llenergymeasure/study/runner.py`**
- `StudyRunner.run()` — resolves ordered sequence via `apply_cycles`, spawns one subprocess per config
- `StudyRunner._run_one()` — subprocess lifecycle: Pipe + Queue creation, Process spawn, join with timeout, result collection, manifest update
- `_collect_result()` — post-join: timeout path (SIGKILL), crash path (non-zero exit), success path (Pipe recv)
- `_run_experiment_worker()` — child entry point: installs SIGINT→SIG_IGN, sends progress events, sends error dict on failure. **Stub only** — raises `NotImplementedError` until Phase 12 wires the real backend.
- `_calculate_timeout()` — `max(n * 2, 600)` heuristic; respects `experiment_timeout_seconds` override via `getattr` (forward-compat for Phase 12)
- `_consume_progress_events()` — daemon thread draining progress queue (display wiring is Phase 12)

**`tests/unit/test_study_runner.py`**
- 10 test cases; 0 real subprocesses spawned; 0 GPU required
- Mock injection: `multiprocessing.get_context` patched to return a controlled mock context
- Covers: timeout→SIGKILL, crash (non-zero exit + empty pipe), subprocess exception, success path, spawn context enforced, daemon=False enforced, interleaved ordering, sequential ordering, timeout minimum, timeout scaling

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check: PASSED

- `src/llenergymeasure/study/runner.py` — FOUND
- `tests/unit/test_study_runner.py` — FOUND
- `src/llenergymeasure/study/__init__.py` — FOUND
- Commit `794d69e` (RED) — FOUND
- Commit `af8f5de` (GREEN) — FOUND
- Commit `4b995e6` (REFACTOR) — FOUND
