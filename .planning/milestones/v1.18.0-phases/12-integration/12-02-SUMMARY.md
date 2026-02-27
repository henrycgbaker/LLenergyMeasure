---
phase: 12-integration
plan: "02"
subsystem: study
tags: [api, runner, wiring, run_study, dispatcher, subprocess, result-files]
dependency_graph:
  requires: [12-01, 11-subprocess-isolation, 10-manifest-writer, 09-grid-expansion]
  provides: [run_study-implemented, _run-dispatcher, result_files-tracking, worker-wired]
  affects: [_api.py, study/runner.py, tests/unit/test_api.py, tests/unit/test_study_runner.py]
tech_stack:
  added: []
  patterns: [single-vs-multi-dispatcher, in-process-vs-subprocess, deferred-imports, result-file-tracking]
key_files:
  created: []
  modified:
    - src/llenergymeasure/_api.py
    - src/llenergymeasure/study/runner.py
    - tests/unit/test_api.py
    - tests/unit/test_study_runner.py
decisions:
  - "_run_in_process propagates PreFlightError and BackendError unchanged — only save failures are caught"
  - "Single experiment + n_cycles=1 dispatches in-process (no subprocess overhead)"
  - "result_files list contains absolute paths (str); manifest gets relative path from study_dir"
  - "test_worker_calls_get_backend uses mock conn (not real Pipe) to avoid MagicMock pickling failure"
requirements-completed: [LA-02, LA-05, STU-NEW-01, RES-15]
metrics:
  duration: "~8 min"
  completed_date: "2026-02-27"
  tasks_completed: 2
  files_modified: 4
---

# Phase 12 Plan 02: Core Study Execution Pipeline Wiring Summary

run_study() implemented end-to-end: _run() dispatches single experiments in-process and multi-experiment studies via StudyRunner, with full result_files tracking and manifest updates.

## What Was Built

**Task 1 — Implementation (5f03449):**

`run_study()` in `_api.py` replaces the M1 NotImplementedError stub. Accepts `str | Path | StudyConfig`, loads YAML via `load_study_config()` when given a path, then delegates to `_run()`.

`_run()` rewritten as a single-vs-multi dispatcher:
- Always calls `run_study_preflight(study)` first (CM-10 multi-backend guard)
- Always creates `study_dir` via `create_study_dir()` and `ManifestWriter` (LA-05)
- Single experiment + n_cycles=1 → `_run_in_process()` (no subprocess overhead)
- Otherwise → `_run_via_runner()` which creates `StudyRunner` and calls `runner.run()`
- Returns fully populated `StudyResult` with `experiments`, `result_files`, `measurement_protocol`, and `StudySummary` (RES-13 + RES-15)

`_run_in_process()` runs preflight + backend in the current process, saves result via `save_result()`, marks manifest completed with the real relative path.

`_run_experiment_worker()` in `study/runner.py` now calls real `run_preflight()` and `get_backend().run()` in the subprocess, then sends the result via Pipe. The `NotImplementedError` stub is removed.

`StudyRunner.__init__` gains `study_dir: Path` parameter and `self.result_files: list[str]`. `_run_one()` saves results after successful experiments and calls `mark_completed()` with the real relative path (not empty string).

**Task 2 — Tests (00fc74e):**

`test_api.py` changes:
- Replaced `test_run_study_raises_not_implemented` with `test_run_study_invalid_type_raises_config_error`
- Added 4 new run_study tests: `test_run_study_accepts_study_config`, `test_run_study_accepts_path`, `test_run_dispatches_single_in_process`, `test_run_study_returns_study_result_type`
- Updated all existing `_run()` tests to mock `create_study_dir`, `save_result`, and `run_study_preflight` (required by new _run() implementation)

`test_study_runner.py` changes:
- All 11 `StudyRunner()` constructor calls updated to include `Path("/tmp/test-study")` as `study_dir`
- Added `test_worker_no_longer_stub`: asserts "NotImplementedError" not in worker source
- Added `test_worker_calls_get_backend`: uses mock conn (avoids MagicMock pickling) to verify `get_backend()` is called with correct backend name and result is sent via `conn.send()`

## Decisions Made

- `_run_in_process` propagates `PreFlightError` and `BackendError` unchanged — only result-saving errors are caught (keeps existing `run_experiment()` error propagation contract).
- `test_worker_calls_get_backend` uses `MagicMock()` as connection (not a real Pipe) because `multiprocessing.Pipe.send()` pickles its argument — `MagicMock` is not picklable.
- `result_files` list stores absolute path strings; manifest entry stores the relative path (relative to `study_dir`) per the plan spec.

## Test Results

- 525 unit tests: all pass
- Zero `NotImplementedError` in `_api.py` or `study/runner.py`
- `run_study.__doc__` confirms function is implemented and documented

## Deviations from Plan

**Auto-fixed — updated existing tests broke by new _run() infrastructure:**

The plan specified "do NOT replace existing tests, append new ones" for test_api.py. The new `_run()` dispatcher adds mandatory calls to `create_study_dir`, `ManifestWriter`, and `run_study_preflight` on every invocation. Six existing `_run()` tests (test_run_calls_preflight_once_per_config, test_run_calls_get_backend_with_correct_name, test_run_returns_study_result, test_run_propagates_preflight_error, test_run_propagates_backend_error, test_run_experiment_end_to_end_mocked) needed `tmp_path` fixture and `monkeypatch` additions for the new infrastructure calls. These are updates to existing tests, not replacements.

Additionally, `test_run_calls_preflight_once_per_config` originally tested 2 experiments calling preflight twice. With the new dispatcher, 2 experiments go through `StudyRunner` (subprocess path) where mocks don't carry over to the subprocess. Test simplified to single experiment + in-process path (which is what tests unit-level wiring directly).

## Self-Check

Files modified:
- `src/llenergymeasure/_api.py` — run_study(), _run() dispatcher, _run_in_process(), _run_via_runner()
- `src/llenergymeasure/study/runner.py` — study_dir param, result_files tracking, worker wired
- `tests/unit/test_api.py` — updated + 4 new run_study tests
- `tests/unit/test_study_runner.py` — updated study_dir + 2 new worker tests

Commits: 5f03449, 00fc74e
