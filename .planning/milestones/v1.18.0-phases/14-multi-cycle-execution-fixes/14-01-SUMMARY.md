---
phase: 14-multi-cycle-execution-fixes
plan: "01"
subsystem: study
tags: [bugfix, multi-cycle, manifest, runner]
dependency_graph:
  requires: []
  provides: [correct-multi-cycle-execution, manifest-completed-status]
  affects: [study/runner.py, study/manifest.py, _api.py]
tech_stack:
  added: []
  patterns: [per-config-hash-cycle-counters, deduplication-before-cycling]
key_files:
  created: []
  modified:
    - src/llenergymeasure/study/runner.py
    - src/llenergymeasure/study/manifest.py
    - src/llenergymeasure/_api.py
    - tests/unit/test_study_runner.py
    - tests/unit/test_study_manifest.py
decisions:
  - "Remove apply_cycles() from runner entirely: study.experiments is already cycled by load_study_config() — runner must consume the pre-ordered list as-is"
  - "Per-config_hash cycle counters (_cycle_counters dict): reset per run(), increment on each _run_one() call for a given hash — correct regardless of cycle_order"
  - "_build_entries deduplication: deduplicate study.experiments by config_hash before looping n_cycles — recovers correct n_unique*n_cycles entry count from the pre-cycled list"
  - "mark_study_completed only reached on success path: SIGINT path calls sys.exit(130) before _run() returns, so no guard needed"
metrics:
  duration: "~13 min"
  completed_date: "2026-02-27"
  tasks: 2
  files: 5
---

# Phase 14 Plan 01: Multi-Cycle Execution Fixes Summary

Fixed 4 integration defects that made multi-cycle study execution non-functional: double `apply_cycles()` inflating experiment count 3x, `_build_entries()` double-multiplying manifest entries, hardcoded `cycle=1` causing `KeyError` for cycles 2+, and missing `mark_study_completed()` leaving manifest status stuck at "running".

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix double apply_cycles and cycle tracking in runner.py | cb7ab69 | runner.py, test_study_runner.py |
| 2 | Add mark_study_completed() and wire it in _run() | c489d39 | manifest.py, _api.py, test_study_manifest.py |

## What Was Built

**Task 1 — runner.py fixes (STU-07, STU-08):**

Removed the entire `apply_cycles()` call block from `StudyRunner.run()`. `study.experiments` arriving in the runner is already the fully-ordered execution sequence produced by `load_study_config()` → calling `apply_cycles()` again multiplied a 6-entry list by 3 to produce 18 experiments.

Replaced `n_unique = len(self.study.experiments)` with a set comprehension over `config_hash` values so cycle-gap detection still fires correctly after each complete round.

Replaced `cycle = 1` (hardcoded) in `_run_one()` with a `_cycle_counters: dict[str, int]` that increments per `config_hash`. First time hash "abc" appears: `cycle=1`; second time: `cycle=2`; etc. This matches exactly the manifest entries built by `_build_entries()`.

Updated the two ordering test helpers (`_make_ordering_study`) to pass a pre-cycled `experiments` list — matching what `load_study_config()` produces — rather than an uncycled `[exp_a, exp_b]`.

**Task 2 — manifest.py and _api.py fixes (STU-09):**

Rewrote `_build_entries()` to deduplicate `study.experiments` by `config_hash` before looping over `n_cycles`. The pre-cycled 6-entry list (for 2 configs x 3 cycles) was being iterated and multiplied by `n_cycles=3` again, producing 18 entries. Deduplication recovers 2 unique configs, then `for cycle in range(1, n_cycles+1)` produces the correct 6 entries.

Added `ManifestWriter.mark_study_completed()` — sets `status="completed"` and `completed_at` using the same `model_copy(update=...)` + `_write()` pattern as `mark_interrupted()`.

Wired `manifest.mark_study_completed()` in `_api._run()` after `_run_via_runner()` or `_run_in_process()` returns and wall time is captured. The SIGINT path calls `mark_interrupted()` then `sys.exit(130)` inside `StudyRunner.run()` — it never returns to `_run()` — so no guard is needed.

## Tests Added

**test_study_runner.py (+5 tests, 17 total):**
- `test_multi_cycle_correct_experiment_count` — 2-config x 3-cycle study produces 6 experiments (not 18), Process() called 6 times
- `test_cycle_counter_increments_per_config_hash` — for A,B,A,B ordering: hash_A gets cycles [1,2], hash_B gets cycles [1,2]
- Ordering tests updated: `_make_ordering_study` now passes pre-cycled lists

**test_study_manifest.py (+3 tests, 24 total):**
- `test_build_entries_deduplicates_cycled_experiments` — 2-config x 3-cycle pre-cycled list produces 6 entries; each hash has cycles [1,2,3]
- `test_mark_study_completed` — status="completed", completed_at set in-memory and on-disk
- `test_manifest_status_after_all_experiments_complete` — full lifecycle: pending→running→completed per entry, then mark_study_completed()→status="completed"

**Full suite:** 535 passed (up from 530, no regressions).

## Verification

```
grep -n "apply_cycles" runner.py     # only in comment — no call
grep -n "seen" manifest.py           # dedup dict in _build_entries
grep -n "_cycle_counters" runner.py  # dict init in __init__ and run(), usage in _run_one
grep -n "mark_study_completed" manifest.py   # def mark_study_completed
grep -n "mark_study_completed" _api.py       # call after runner returns
```

All 5 checks confirm the fixes are in place.

## Deviations from Plan

**1. [Rule 2 - Missing critical functionality] Updated ordering test helpers to pass pre-cycled lists**
- **Found during:** Task 1
- **Issue:** `_make_ordering_study` built a `StudyConfig` with `experiments=[exp_a, exp_b]` (uncycled). After removing `apply_cycles()` from the runner, these tests expected 4 executions (2 configs x 2 cycles) but would only see 2 (one per entry in the uncycled list).
- **Fix:** Added `apply_cycles()` call inside `_make_ordering_study` to mirror what `load_study_config()` produces. Tests now pass a pre-cycled 4-entry list and continue to assert `call_order == ["model-a", "model-b", "model-a", "model-b"]` (interleaved) and `["model-a", "model-a", "model-b", "model-b"]` (sequential).
- **Files modified:** `tests/unit/test_study_runner.py`
- **Commit:** cb7ab69

## Self-Check: PASSED

- src/llenergymeasure/study/runner.py: FOUND
- src/llenergymeasure/study/manifest.py: FOUND
- src/llenergymeasure/_api.py: FOUND
- tests/unit/test_study_runner.py: FOUND
- tests/unit/test_study_manifest.py: FOUND
- Commit cb7ab69: FOUND
- Commit c489d39: FOUND
- 535 unit tests pass: CONFIRMED
