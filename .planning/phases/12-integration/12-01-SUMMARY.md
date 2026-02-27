---
phase: 12-integration
plan: "01"
subsystem: study
tags: [schema, preflight, rename, results]
dependency_graph:
  requires: [11-subprocess-isolation]
  provides: [StudyResult-full-schema, StudySummary, run_study_preflight, experiment_gap_seconds]
  affects: [study/runner.py, domain/experiment.py, orchestration/preflight.py, config/models.py]
tech_stack:
  added: []
  patterns: [Pydantic-BaseModel, PreFlightError, multi-backend-guard]
key_files:
  created:
    - tests/unit/test_study_result.py
    - tests/unit/test_study_preflight.py
  modified:
    - src/llenergymeasure/domain/experiment.py
    - src/llenergymeasure/config/models.py
    - src/llenergymeasure/config/user_config.py
    - src/llenergymeasure/orchestration/preflight.py
    - src/llenergymeasure/study/runner.py
    - tests/unit/test_study_runner.py
    - tests/unit/test_study_grid.py
decisions:
  - "StudyResult has no extra='forbid' — internal model, not user-visible output"
  - "run_study_preflight raises PreFlightError immediately for multi-backend (CM-10, Docker is M3)"
  - "experiment_gap_seconds replaces config_gap_seconds in both ExecutionConfig and UserExecutionConfig"
metrics:
  duration: "~10 min"
  completed_date: "2026-02-27"
  tasks_completed: 3
  files_modified: 9
---

# Phase 12 Plan 01: StudyResult Schema, Gap Rename, Multi-backend Pre-flight Summary

Full StudyResult RES-13 schema with StudySummary, experiment_gap_seconds rename, and run_study_preflight() guard for multi-backend studies.

## What Was Built

**Task 1 — StudyResult full schema (9e97911):**
Added `StudySummary` model and upgraded `StudyResult` from M1 stub to full RES-13 schema. `StudySummary` provides aggregate stats (total_experiments, completed, failed, total_wall_time_s, total_energy_j, warnings). `StudyResult` gains `study_design_hash`, `measurement_protocol`, `result_files`, and `summary` fields. No `extra="forbid"` — this is an internal model.

**Task 2 — Rename and multi-backend guard (bffb189):**
Renamed `config_gap_seconds` to `experiment_gap_seconds` in `ExecutionConfig` (models.py), `UserExecutionConfig` (user_config.py), and all call sites in `study/runner.py`. Gap label in runner updated from "Config gap" to "Experiment gap". Added `run_study_preflight(study: StudyConfig)` to `orchestration/preflight.py` that raises `PreFlightError` for multi-backend studies with a clear Docker isolation direction (CM-10).

**Task 3 — Unit tests (64b133d):**
8 new unit tests across two files: 4 for StudyResult schema (full schema, backwards compat, defaults, paths-not-embedded) and 4 for run_study_preflight (single-backend passes, multi-backend raises, Docker message, backend list in error). All 88 study-related tests pass.

## Decisions Made

- `StudyResult` has no `extra="forbid"` — it is internal, not user-visible output (unlike `ExperimentResult` which does have it).
- `run_study_preflight` raises immediately for multi-backend studies — Docker isolation is M3. Clear error message includes backend list and Docker direction.
- Both `ExecutionConfig` and `UserExecutionConfig` renamed simultaneously to keep user config YAML and internal config in sync.

## Test Results

- 8 new tests: all pass
- 80 existing study tests: all still pass
- Zero `config_gap_seconds` references remaining in Python source or tests

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

Files created/modified exist and are committed:
- `src/llenergymeasure/domain/experiment.py` — StudySummary + full StudyResult
- `src/llenergymeasure/config/models.py` — experiment_gap_seconds
- `src/llenergymeasure/config/user_config.py` — experiment_gap_seconds
- `src/llenergymeasure/orchestration/preflight.py` — run_study_preflight
- `src/llenergymeasure/study/runner.py` — experiment_gap_seconds usage
- `tests/unit/test_study_result.py` — 4 schema tests
- `tests/unit/test_study_preflight.py` — 4 preflight tests
- `tests/unit/test_study_runner.py` — updated fixture
- `tests/unit/test_study_grid.py` — updated assertions

Commits: 9e97911, bffb189, 64b133d
