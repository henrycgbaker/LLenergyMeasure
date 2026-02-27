---
phase: 07-cli
plan: 01
subsystem: cli
tags: [cli, display, vram, formatting, pydantic, huggingface-hub, pynvml]

requires:
  - phase: 06-results-schema-and-persistence
    provides: ExperimentResult domain model with process_results and latency_stats fields
  - phase: 02-config-system
    provides: ExperimentConfig with precision/backend/pytorch section
  - phase: 03-library-api
    provides: LLEMError exception hierarchy

provides:
  - Plain-text display utilities for result summaries (stdout) and experiment headers (stderr)
  - VRAM estimator using HF Hub metadata with non-blocking timeout
  - _sig3() 3-significant-figure formatter
  - _format_duration() human-readable duration formatter
  - format_validation_error() Pydantic error wrapper with did-you-mean suggestions
  - 19 unit tests covering all pure display functions and DTYPE_BYTES

affects: [07-cli plan 02 (llem run command), 07-cli plan 03 (llem config command)]

tech-stack:
  added: [difflib (stdlib), math (stdlib), pynvml (optional), huggingface_hub (optional)]
  patterns:
    - All result output to stdout, progress/header output to stderr
    - No Rich imports anywhere in new CLI modules
    - Non-blocking pattern: HF Hub and pynvml wrapped in try/except with 5s socket timeout
    - TYPE_CHECKING guards for heavy imports (_display.py uses only string annotations at runtime)

key-files:
  created:
    - src/llenergymeasure/cli/_display.py
    - src/llenergymeasure/cli/_vram.py
    - tests/unit/test_cli_display.py
  modified: []

key-decisions:
  - "stdout for results, stderr for progress/headers — result summary is the scientific record"
  - "HF Hub metadata fetch uses 5s socket timeout to avoid blocking CLI on network failure"
  - "VRAM estimation returns None (not error) when model metadata unavailable — non-blocking"
  - "format_validation_error uses difflib.get_close_matches on backend/precision pools for did-you-mean"
  - "_sig3 uses math.log10 + rounding to 3 significant figures; zero handled as special case"

patterns-established:
  - "CLI output modules prefixed with _ (private): _display.py, _vram.py"
  - "All network/GPU operations wrapped in try/except returning None on failure"
  - "Pydantic ValidationError passed through unchanged — only formatted, never wrapped"

requirements-completed: [CLI-08, CLI-09, CLI-14]

duration: 3min
completed: 2026-02-27
---

# Phase 7 Plan 01: CLI Display Utilities Summary

**Plain-text result summary printer, VRAM estimator, and Pydantic error formatter providing the display foundation for `llem run` and `llem config` commands**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-02-26T23:56:46Z
- **Completed:** 2026-02-27T00:00:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `_display.py` with 7 functions: `_sig3`, `_format_duration`, `print_result_summary`, `print_dry_run`, `format_error`, `format_validation_error`, `print_experiment_header`
- `_vram.py` with VRAM estimator using HuggingFace Hub metadata and pynvml GPU query — both non-blocking
- 19 unit tests covering all pure functions (no GPU, no network required)

## Task Commits

1. **Task 1: Create _display.py — plain-text formatting utilities** - `dca54da` (feat)
2. **Task 2: Create _vram.py + unit tests for display and VRAM** - `1abb93c` (feat)

## Files Created/Modified

- `src/llenergymeasure/cli/_display.py` — 7 display functions: result summary, dry-run echo, error formatting, duration/number formatting, experiment header
- `src/llenergymeasure/cli/_vram.py` — VRAM estimator (weights, KV cache, overhead, total) + `DTYPE_BYTES` constant + `get_gpu_vram_gb()`
- `tests/unit/test_cli_display.py` — 19 unit tests for all pure functions

## Decisions Made

- `stdout` for result summaries (scientific record), `stderr` for progress headers (transient) — matches CONTEXT.md decision
- `_sig3()` implementation uses `math.log10` magnitude detection + rounding — handles zero, large, and tiny values correctly
- VRAM estimator returns `None` rather than raising on network failure — non-blocking per research doc requirement
- `format_validation_error` uses `difflib.get_close_matches` on backend and precision pools for did-you-mean suggestions
- `TYPE_CHECKING` guards on all heavy imports in `_display.py` — zero runtime cost at import time

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `_display.py` and `_vram.py` are ready for consumption by `llem run` (plan 02) and `llem config` (plan 03)
- `print_result_summary` reads `result.process_results[0].compute_metrics` for FLOPs method/confidence
- `print_dry_run` accepts pre-computed `vram` dict from `_vram.estimate_vram()` + GPU memory from `_vram.get_gpu_vram_gb()`

---
*Phase: 07-cli*
*Completed: 2026-02-27*
