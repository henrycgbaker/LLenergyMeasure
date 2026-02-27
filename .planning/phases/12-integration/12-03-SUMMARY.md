---
phase: 12-integration
plan: "03"
subsystem: cli
tags: [cli, study, flags, display, progress, summary]
dependency_graph:
  requires: [12-02, 12-01]
  provides: [study-cli-flags, study-detection, print_study_summary, print_study_progress]
  affects: [cli/run.py, cli/_display.py, tests/unit/test_cli_run.py]
tech_stack:
  added: []
  patterns: [study-detection, yaml-key-check, cli-effective-defaults, model_construct-test-pattern]
key_files:
  created: []
  modified:
    - src/llenergymeasure/cli/run.py
    - src/llenergymeasure/cli/_display.py
    - tests/unit/test_cli_run.py
decisions:
  - "CLI effective defaults n_cycles=3 and cycle_order=shuffled applied only when YAML execution block omits those keys"
  - "quiet suppresses CLI-side progress and summary; gap countdown suppression deferred (M2 limitation, subprocess-level)"
  - "test_print_study_summary_basic uses model_construct() to bypass Pydantic validation when constructing StudyResult with MagicMock experiments"
metrics:
  duration: "~3 min"
  completed_date: "2026-02-27"
  tasks_completed: 2
  files_modified: 3
---

# Phase 12 Plan 03: Study-Mode CLI Integration Summary

Study-aware `llem run` with `--cycles`/`--order`/`--no-gaps` flags, YAML study detection, per-experiment progress display, and summary table.

## What Was Built

**Task 1 — Study detection and flags (18b0a1e):**

Added `--cycles`, `--order`, `--no-gaps` flags to `run()` and `_run_impl()` in `cli/run.py`. Study detection in `_run_impl()` reads the YAML with `yaml.safe_load` and checks for `sweep:` or `experiments:` top-level keys. When detected, execution is routed to `_run_study_impl()` before the experiment path.

`_run_study_impl()` applies CLI effective defaults (`n_cycles=3`, `cycle_order="shuffled"`) only when the YAML `execution:` block omits those keys. `--no-gaps` sets both `experiment_gap_seconds=0` and `cycle_gap_seconds=0`. Calls `format_preflight_summary()` before execution (unless `--quiet`), then `run_study()`, then `print_study_summary()`.

`StudyError` added to the error handler in `run()` (exit code 1).

**Task 2 — Display functions and tests (977520f):**

Added to `cli/_display.py`:

- `print_study_progress(index, total, config, status, elapsed, energy)` — prints `[3/12] OK model backend precision -- 4m 32s (123 J)` to stderr. Icons: `...` (running), `OK` (completed), `FAIL` (failed).
- `print_study_summary(result: StudyResult)` — prints table with columns `#`, `Config`, `Status`, `Time`, `Energy`, `tok/s` plus footer with totals. Truncates long model names to 20 chars. Shows warnings and result file paths.

`StudyResult` added to the `_display.py` top-level import.

5 new tests in `tests/unit/test_cli_run.py`:
- `test_study_detection_with_sweep_key` — sweep: key triggers study mode
- `test_study_detection_with_experiments_key` — experiments: key triggers study mode
- `test_cli_flags_present` — --cycles, --order, --no-gaps all in signature
- `test_print_study_summary_basic` — summary table renders without error
- `test_print_study_progress` — progress line format correct

## Decisions Made

- CLI effective defaults (`n_cycles=3`, `cycle_order="shuffled"`) are checked against `yaml_execution` dict keys before applying, so explicit YAML values are never overridden by defaults.
- `--quiet` suppresses CLI-level output (pre-flight summary, study summary table). Gap countdown suppression is a documented M2 limitation: `run_gap()` writes directly to stdout in the subprocess and cannot be suppressed by the parent process without redesigning the gap API. Document in Phase 13.
- `test_print_study_summary_basic` uses `StudyResult.model_construct()` to bypass Pydantic's strict `experiments` validation, allowing a `MagicMock` in the experiments list. This avoids constructing a full `ExperimentResult` with 10+ required fields just to test display logic.

## Test Results

- 5 new tests: all pass
- 525 existing tests: all still pass
- Total: 530 unit tests passing

## Deviations from Plan

**Auto-fixed — test used MagicMock for experiments field:**

The plan's test template passed a raw `MagicMock()` to `StudyResult(experiments=[exp])`. Pydantic's strict type-checking rejects non-ExperimentResult values. Fixed by using `StudyResult.model_construct()` instead, which skips validation. This is the correct pattern for unit tests that test display logic, not model construction.

## Self-Check

Files modified exist:
- `src/llenergymeasure/cli/run.py` — cycles/order/no_gaps flags, _run_study_impl
- `src/llenergymeasure/cli/_display.py` — print_study_progress, print_study_summary
- `tests/unit/test_cli_run.py` — 5 new tests

Commits: 18b0a1e, 977520f
