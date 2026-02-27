---
phase: 12-integration
verified: 2026-02-27T00:00:00Z
status: passed
score: 13/13 must-haves verified
human_verification:
  - test: "Run llem run study.yaml with a real multi-experiment YAML"
    expected: "CLI detects study mode, shows preflight summary, runs experiments via subprocess, prints summary table"
    why_human: "Full end-to-end requires GPU + real model; subprocess spawn cannot be exercised without hardware"
  - test: "Ctrl+C during a study run (thermal gap pause)"
    expected: "SIGINT interrupts gap countdown, marks manifest interrupted, exits with code 130"
    why_human: "Requires interactive terminal with active subprocess; cannot simulate in unit tests"
  - test: "llem run study.yaml --quiet"
    expected: "No pre-flight summary or study summary table printed; gap countdown still appears (documented M2 limitation)"
    why_human: "Requires live execution; gap countdown suppression limitation needs human confirmation it is documented"
---

# Phase 12: Integration Verification Report

**Phase Goal:** Wire config → runner → subprocess → results into end-to-end study execution with CLI surface
**Verified:** 2026-02-27
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | StudyResult has study_design_hash, measurement_protocol dict, result_files list, and summary (StudySummary) | VERIFIED | `domain/experiment.py` lines 323–349: all four fields present with correct types |
| 2 | StudySummary has total_experiments, completed, failed, total_wall_time_s, total_energy_j, warnings list | VERIFIED | `domain/experiment.py` lines 309–320: all six fields present |
| 3 | A study YAML with backend: [pytorch, vllm] raises PreFlightError with Docker runner message before any subprocess is spawned | VERIFIED | `orchestration/preflight.py` lines 153–172: `run_study_preflight()` raises PreFlightError with "Multi-backend studies require Docker isolation" |
| 4 | config_gap_seconds is renamed to experiment_gap_seconds throughout the codebase | VERIFIED | Zero matches for `config_gap_seconds` in `src/`; `models.py` line 455 and `user_config.py` line 98 use new name |
| 5 | run_study() accepts str, Path, or StudyConfig and returns StudyResult with populated result_files, measurement_protocol, and summary | VERIFIED | `_api.py` lines 85–110: accepts all three forms, delegates to `_run()` which populates all fields |
| 6 | _run() dispatches single-experiment in-process and multi-experiment to StudyRunner | VERIFIED | `_api.py` lines 152–212: `is_single` branch routes to `_run_in_process()`, else to `_run_via_runner()` |
| 7 | _run_experiment_worker sends ExperimentResult through Pipe (no longer raises NotImplementedError) | VERIFIED | `study/runner.py` lines 79–91: calls `run_preflight`, `get_backend().run()`, `conn.send(result)` |
| 8 | run_study() always writes manifest.json to disk (LA-05 side-effect contract) | VERIFIED | `_api.py` lines 165–172: `create_study_dir()` + `ManifestWriter()` called on every `_run()` invocation |
| 9 | result_files in StudyResult contains path strings to per-experiment result.json files (RES-15) | VERIFIED | `_api.py` lines 246–248 (in-process) and `runner.py` lines 381–383 (subprocess): `result_files.append(str(result_path))` |
| 10 | mark_completed receives the actual result_file path (not empty string) | VERIFIED | `_api.py` line 248: `manifest.mark_completed(config_hash, cycle, rel_path)` with real path |
| 11 | llem run study.yaml detects study mode (YAML has sweep: or experiments: keys) and routes to run_study() | VERIFIED | `cli/run.py` lines 188–211: yaml.safe_load check, routes to `_run_study_impl()` which calls `run_study()` |
| 12 | --cycles, --order, --no-gaps CLI flags added; CLI effective defaults n_cycles=3 and cycle_order=shuffled apply when neither YAML nor CLI specifies | VERIFIED | `cli/run.py` lines 84–97 (flags), lines 282–293 (defaults logic) |
| 13 | Study summary table displays after completion; print_study_progress and print_study_summary exist in _display.py | VERIFIED | `cli/_display.py` lines 297–398: both functions present with correct format and column headers |

**Score:** 13/13 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/domain/experiment.py` | StudyResult full schema with StudySummary | VERIFIED | `StudySummary` at line 309, `StudyResult` at line 323; both substantive with all required fields |
| `src/llenergymeasure/orchestration/preflight.py` | Multi-backend study pre-flight check | VERIFIED | `run_study_preflight()` at line 153; raises PreFlightError with "Multi-backend" + "Docker" message |
| `tests/unit/test_study_result.py` | StudyResult schema tests | VERIFIED | 4 tests; all pass: full schema, backwards compat, defaults, paths-not-embedded |
| `tests/unit/test_study_preflight.py` | Multi-backend pre-flight tests | VERIFIED | 4 tests; all pass: single passes, multi raises, Docker message, backend list in error |
| `src/llenergymeasure/_api.py` | run_study() implementation and _run() dispatcher | VERIFIED | `run_study()` at line 85; `_run()` at line 152; `_run_in_process()` and `_run_via_runner()` present |
| `src/llenergymeasure/study/runner.py` | Wired _run_experiment_worker with real backend | VERIFIED | Worker calls `get_backend()` at line 86, `backend.run()` at line 87, `conn.send(result)` at line 90 |
| `tests/unit/test_api.py` | Tests for run_study() and _run() dispatch | VERIFIED | Includes `test_run_study_accepts_study_config`, `test_run_dispatches_single_in_process`, and others |
| `tests/unit/test_study_runner.py` | Updated runner tests with study_dir parameter | VERIFIED | All 11 `StudyRunner()` calls include `study_dir`; `test_worker_no_longer_stub` and `test_worker_calls_get_backend` pass |
| `src/llenergymeasure/cli/run.py` | Study detection, study flags, study execution path | VERIFIED | `--cycles`, `--order`, `--no-gaps` at lines 84–97; `_run_study_impl()` at line 256 |
| `src/llenergymeasure/cli/_display.py` | Study progress display and summary table | VERIFIED | `print_study_progress()` at line 297; `print_study_summary()` at line 333 |
| `tests/unit/test_cli_run.py` | CLI study-mode tests | VERIFIED | 5 new tests: detection (sweep/experiments keys), flag presence, summary render, progress format |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `_api.py` | `orchestration/preflight.run_study_preflight` | `_run()` calls before dispatch | VERIFIED | `_api.py` line 168: `run_study_preflight(study)` |
| `_api.py` | `study/runner.StudyRunner` | `_run_via_runner()` creates StudyRunner when multi | VERIFIED | `_api.py` line 264: `runner = StudyRunner(study, manifest, study_dir)` |
| `study/runner.py` | `core/backends.get_backend` | `_run_experiment_worker` calls get_backend + backend.run() | VERIFIED | `runner.py` lines 79–87: `from llenergymeasure.core.backends import get_backend; get_backend(config.backend); backend.run(config)` |
| `cli/run.py` | `llenergymeasure._api.run_study` | Study detection routes to run_study() | VERIFIED | `cli/run.py` lines 268, 316: `from llenergymeasure import run_study; result = run_study(study_config)` |
| `cli/run.py` | `config/loader.load_study_config` | Study YAML detection checks for sweep/experiments keys | VERIFIED | `cli/run.py` line 304: `study_config = load_study_config(path=config, ...)` |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| RES-13 | 12-01 | `StudyResult`: study_design_hash, measurement_protocol, result_files, summary | SATISFIED | `domain/experiment.py` lines 323–349 |
| CM-10 | 12-01 | Multi-backend study without Docker → hard error at pre-flight | SATISFIED | `orchestration/preflight.py` lines 153–172; `_api.py` line 168 calls it before dispatch |
| LA-02 | 12-02 | `run_study(config: str | Path | StudyConfig) -> StudyResult` | SATISFIED | `_api.py` lines 85–110: all three input types handled |
| LA-05 | 12-02 | `run_study()` always writes manifest to disk | SATISFIED | `_api.py` lines 170–172: `create_study_dir` + `ManifestWriter` called unconditionally |
| STU-NEW-01 | 12-02 | `_run()` body implemented — dispatches single vs study, returns StudyResult | SATISFIED | `_api.py` lines 152–212: full dispatcher with both paths |
| RES-15 | 12-02 | `result_files` contains paths, not embedded results | SATISFIED | `_api.py` line 247: appends `str(result_path)`; `StudyResult.result_files: list[str]` |
| CLI-05 | 12-03 | Study-mode flags: --cycles, --no-gaps, --order | SATISFIED | `cli/run.py` lines 84–97; passed through to `_run_study_impl()` |
| CLI-11 | 12-03 | Thermal gap countdown display during inter-experiment pauses | SATISFIED | `study/runner.py` lines 24, 287, 295: `run_gap()` called with "Experiment gap" / "Cycle gap" labels; countdown is visible during execution. `--quiet` does not suppress gap countdown (subprocess-level, documented M2 limitation) |

All 8 required requirement IDs are accounted for. No orphaned requirements detected.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `study/runner.py` | 131 | Comment `# Phase 12: forward to Rich display layer here` in `_consume_progress_events` | Info | Progress events are drained (consumer thread drains queue) but not forwarded to display. This means the print_study_progress function added to _display.py has no caller in the study execution loop. Not a blocker — the plan specifies display wiring is best-effort. |

**Note on progress display wiring:** `print_study_progress()` exists in `_display.py` and has tests, but `_consume_progress_events()` in `runner.py` still only drains the queue (line 131 comment says "Phase 12: forward to Rich display layer"). The per-experiment progress line is not actually printed during study execution. This is an incomplete wiring — the function exists but the caller is not connected. However, this does not block any requirement ID: CLI-11 specifically covers thermal gap countdown (which is wired via `run_gap()`), and no requirement mandates per-experiment progress line wiring in M2.

### Human Verification Required

**1. End-to-end study execution with hardware**

**Test:** `llem run study.yaml` with a YAML containing `sweep: {precision: [fp32, fp16]}` against a small model
**Expected:** CLI detects study mode, prints preflight summary, spawns subprocesses, prints table with Config/Status/Time/Energy/tok/s columns after completion
**Why human:** Requires GPU + real model weights; subprocess spawn path with real backend cannot be unit-tested

**2. Ctrl+C interrupt handling during a study**

**Test:** Run a multi-experiment study, press Ctrl+C during a gap countdown
**Expected:** Gap exits immediately, study marks manifest as interrupted, exits with code 130
**Why human:** Requires interactive terminal and active subprocess; signal delivery to subprocess cannot be simulated in unit tests

**3. --quiet flag suppresses correct output**

**Test:** `llem run study.yaml --quiet`
**Expected:** No preflight summary or study summary table; gap countdown may still appear (M2 limitation)
**Why human:** Gap countdown suppression limitation needs human confirmation it is acceptable and documented

### Gaps Summary

No blocking gaps found. All 13 observable truths verified. All 8 requirement IDs satisfied. All key links wired. The only incomplete item is `print_study_progress()` having no caller in the run loop (the progress consumer thread drains but does not forward events), but this is not required by any requirement ID in this phase.

---

_Verified: 2026-02-27_
_Verifier: Claude (gsd-verifier)_
