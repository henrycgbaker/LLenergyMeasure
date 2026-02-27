---
phase: 14-multi-cycle-execution-fixes
verified: 2026-02-27T12:00:00Z
status: passed
score: 5/5 must-haves verified
---

# Phase 14: Multi-Cycle Execution Fixes — Verification Report

**Phase Goal:** Fix the 4 integration defects that break multi-cycle study execution: double apply_cycles(), hardcoded cycle=1, missing mark_study_completed(), and _build_entries() over-inflation.
**Verified:** 2026-02-27
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | A 2-config x 3-cycle study runs exactly 6 experiments (not 18) — apply_cycles() called once, not twice | VERIFIED | runner.py line 246: `ordered = self.study.experiments` — no apply_cycles() call. test_multi_cycle_correct_experiment_count asserts 6 results and 6 Process() calls. |
| 2 | Manifest entries for cycles 2+ transition correctly — per-config_hash cycle counter replaces hardcoded cycle=1 | VERIFIED | runner.py lines 229, 260, 330-332: `_cycle_counters: dict[str, int]` initialised in __init__, reset in run(), incremented per config_hash in _run_one(). test_cycle_counter_increments_per_config_hash asserts hash_A=[1,2], hash_B=[1,2]. |
| 3 | _build_entries() deduplicates study.experiments by config_hash before multiplying by n_cycles — 2-config x 3-cycle = 6 entries, not 18 | VERIFIED | manifest.py lines 261-269: `seen: dict[str, ExperimentConfig] = {}` deduplicates before cycling. test_build_entries_deduplicates_cycled_experiments asserts 6 entries and 2 unique hashes each with cycles [1,2,3]. |
| 4 | StudyManifest.status is 'completed' after a successful study run — mark_study_completed() exists and is called by _run() | VERIFIED | manifest.py lines 189-202: mark_study_completed() method exists; _api.py line 186: manifest.mark_study_completed() called after _run_via_runner() returns. test_mark_study_completed and test_manifest_status_after_all_experiments_complete both pass. |
| 5 | All existing multi-cycle and manifest tests pass with the fixes applied | VERIFIED | 41 tests pass across test_study_runner.py (17 tests) and test_study_manifest.py (24 tests). No regressions. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/study/runner.py` | apply_cycles() removed, per-config_hash cycle counters | VERIFIED | Line 246: `ordered = self.study.experiments`. Lines 229/260/330-332: `_cycle_counters` dict. No apply_cycles import or call in run(). |
| `src/llenergymeasure/study/manifest.py` | _build_entries() deduplicates by config_hash; mark_study_completed() method | VERIFIED | Lines 261-269: `seen` dict deduplication. Lines 189-202: `mark_study_completed()` method with model_copy + _write(). |
| `src/llenergymeasure/_api.py` | _run() calls manifest.mark_study_completed() on success | VERIFIED | Line 186: `manifest.mark_study_completed()` called after wall_time computed, before StudyResult assembly. |
| `tests/unit/test_study_runner.py` | Tests for correct experiment count and cycle tracking | VERIFIED | test_multi_cycle_correct_experiment_count (line 570) and test_cycle_counter_increments_per_config_hash (line 621) both present and passing. |
| `tests/unit/test_study_manifest.py` | Tests for mark_study_completed() and _build_entries() correct entry count | VERIFIED | test_build_entries_deduplicates_cycled_experiments (line 402), test_mark_study_completed (line 444), test_manifest_status_after_all_experiments_complete (line 461) — all present and passing. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `src/llenergymeasure/_api.py` | `src/llenergymeasure/study/manifest.py` | ManifestWriter.mark_study_completed() called after successful run | WIRED | `manifest.mark_study_completed()` at _api.py:186, reached only on success path (SIGINT calls sys.exit(130) inside StudyRunner.run() before returning). |
| `src/llenergymeasure/study/runner.py` | `src/llenergymeasure/study/manifest.py` | mark_running/mark_completed/mark_failed with correct cycle numbers | WIRED | runner.py:354 `self.manifest.mark_running(config_hash, cycle)`, :381 `self.manifest.mark_failed(config_hash, cycle, ...)`, :390/393 `self.manifest.mark_completed(config_hash, cycle, ...)`. Cycle comes from `_cycle_counters` increment. |
| `src/llenergymeasure/study/manifest.py` | `src/llenergymeasure/study/manifest.py` | _build_entries() deduplicates experiments by config_hash before cycling | WIRED | `seen` dict at lines 261-269 filters study.experiments to unique config_hashes before the `for cycle in range(1, n_cycles + 1)` loop. |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| STU-07 | 14-01-PLAN.md | cycle_order: sequential or interleaved — interleaved round-robins across configs | SATISFIED | Correct cycle ordering preserved: runner now uses pre-cycled study.experiments list produced by load_study_config(). _make_ordering_study in tests updated to pass pre-cycled lists; test_interleaved_ordering and test_sequential_ordering pass. |
| STU-08 | 14-01-PLAN.md | StudyManifest written after each experiment completes (checkpoint pattern) | SATISFIED | _build_entries() now produces exactly n_unique * n_cycles entries (deduplication fix). Per-config_hash cycle counters ensure mark_running/mark_completed/mark_failed are called with the correct cycle number for every entry, including cycles 2+. |
| STU-09 | 14-01-PLAN.md | ManifestWriter: mark_running(), mark_completed(), mark_failed() — atomic writes via os.replace() | SATISFIED | mark_study_completed() added to ManifestWriter (manifest.py:189-202). _run() wired to call it at _api.py:186. All three new tests pass on-disk and in-memory. |

No orphaned requirements: all three IDs from the PLAN frontmatter are defined in REQUIREMENTS.md and accounted for above.

**Requirement ID note:** STU-07, STU-08, STU-09 as defined in REQUIREMENTS.md describe the broad features (cycle ordering, manifest checkpointing, atomic writes). Phase 14 fixed the integration defects that prevented those features from working correctly. The traceability table in REQUIREMENTS.md explicitly marks all three as "Phase 14 | Complete".

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `src/llenergymeasure/study/runner.py` | 121, 128 | "Phase 12: forward to Rich display layer here" comment in _consume_progress_events | Info | Pre-existing deferred work from Phase 11/12, not introduced by this phase. Progress events are consumed (queue drained) but not displayed. Does not affect correctness of multi-cycle execution. |

No blockers or warnings introduced by this phase.

### Human Verification Required

None. All observable truths are verifiable programmatically via:
- Static code inspection (no apply_cycles call, _cycle_counters dict, mark_study_completed method, manifest.mark_study_completed() call)
- Unit test pass/fail (41 tests, all passing)

The only deferred item — Rich display for progress events — is pre-existing and scoped to Phase 12, not Phase 14.

### Gaps Summary

No gaps. All 5 must-have truths verified, all 5 artifacts substantive and wired, all 3 key links confirmed, all 3 requirements satisfied.

---

_Verified: 2026-02-27_
_Verifier: Claude (gsd-verifier)_
