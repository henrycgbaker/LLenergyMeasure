---
phase: 11-subprocess-isolation-and-studyrunner
verified: 2026-02-27T18:35:00Z
status: passed
score: 13/13 must-haves verified
re_verification: false
---

# Phase 11: Subprocess Isolation and StudyRunner Verification Report

**Phase Goal:** Each experiment in a study runs in a freshly spawned subprocess with a clean CUDA state, results are returned via Pipe, progress events flow via Queue, and the study survives experiment failures, timeouts, and SIGINT without data corruption.
**Verified:** 2026-02-27T18:35:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | StudyRunner spawns each experiment via `multiprocessing.get_context('spawn')` — never fork | VERIFIED | `runner.py:239` — `mp_ctx = multiprocessing.get_context("spawn")`; `test_spawn_context_used` asserts this at runtime |
| 2 | Results travel parent←child exclusively via `multiprocessing.Pipe` | VERIFIED | `runner.py:320-321` — `Pipe(duplex=False)`, `_collect_result` reads from `parent_conn`; no temp-file path |
| 3 | Child processes created with `daemon=False` — clean CUDA teardown guaranteed | VERIFIED | `runner.py:326` — `daemon=False` explicit kwarg; `test_daemon_false` asserts it |
| 4 | Timeout uses `p.kill()` (SIGKILL), not `p.terminate()` — `_run_one` returns failure dict | VERIFIED | `runner.py:151` (in `_collect_result`) and `runner.py:348` (grace-period fallback) both call `p.kill()`; `test_study_runner_timeout` asserts `p.terminate.assert_not_called()` |
| 5 | Subprocess exception sends structured error dict `{type, message, traceback}` via Pipe; runner marks failed and continues | VERIFIED | `runner.py:86-102` — worker sends error dict on exception; `_collect_result` detects it; `_run_one` calls `mark_failed`; `test_study_runner_subprocess_exception` confirms non-fatal |
| 6 | `cycle_order=interleaved` produces A,B,A,B; `sequential` produces A,A,B,B | VERIFIED | `test_interleaved_ordering` and `test_sequential_ordering` both pass; wired via `apply_cycles` call in `runner.py:230-236` |
| 7 | First Ctrl+C: SIGTERM to child, 2s grace, SIGKILL if alive; manifest status = "interrupted"; exits code 130 | VERIFIED | `runner.py:254-259` (handler sends SIGTERM), `runner.py:345-349` (grace + SIGKILL), `runner.py:300-301` (`mark_interrupted` + `sys.exit(130)`); `test_sigint_first_ctrl_c_marks_manifest_interrupted` passes |
| 8 | Second Ctrl+C escalates immediately to SIGKILL | VERIFIED | `runner.py:258-259` — handler checks `_interrupt_count > 1` and calls `p.kill()`; `test_sigint_second_ctrl_c_kills_immediately` passes |
| 9 | Ctrl+C during gap: immediately aborts gap via `interrupt_event`, stops study | VERIFIED | `run_gap` in `gaps.py:80-84` checks `interrupt_event.is_set()` each tick and returns; `test_sigint_during_gap_exits_immediately` passes |
| 10 | Config gap fires between every consecutive experiment pair; cycle gap after every N-experiment round | VERIFIED | `runner.py:272-285` — config gap at `i > 0`, cycle gap at `i > 0 and i % n_unique == 0` |
| 11 | Gap display shows inline countdown (`\r`-overwrite) with Enter-to-skip; degrades in non-TTY | VERIFIED | `gaps.py:97-101` — `\r{label}: {format_gap_duration(remaining)} (Enter to skip)`; TTY check at `gaps.py:60`; `test_run_gap_non_tty_no_crash` passes |
| 12 | Manifest status = "interrupted" (distinct from "failed") on SIGINT | VERIFIED | `manifest.py:58-60` — `status: Literal["running", "completed", "interrupted", "failed"]`; `mark_interrupted()` at `manifest.py:184-187`; `test_mark_interrupted_sets_status` verifies in-memory and on-disk |
| 13 | Worker installs `signal.SIGINT = SIG_IGN` at entry — parent owns SIGINT | VERIFIED | `runner.py:69` — `signal.signal(signal.SIGINT, signal.SIG_IGN)` is first statement in worker |

**Score:** 13/13 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `src/llenergymeasure/study/runner.py` | `StudyRunner`, `_run_experiment_worker`, `_calculate_timeout`, `_collect_result`, `_consume_progress_events` | VERIFIED | All 5 symbols present; `__all__` exports `StudyRunner`, `_calculate_timeout`, `_run_experiment_worker`; 368 lines |
| `src/llenergymeasure/study/gaps.py` | `run_gap`, `format_gap_duration` | VERIFIED | Both symbols present; `__all__` exports both; 107 lines |
| `src/llenergymeasure/study/manifest.py` | `StudyManifest.status` field; `ManifestWriter.mark_interrupted()` | VERIFIED | `status` field at line 58 with default `"running"`; `mark_interrupted()` at line 184 |
| `tests/unit/test_study_runner.py` | 10 Plan-01 tests + 3 SIGINT tests | VERIFIED | 13 test functions, all pass, no GPU required |
| `tests/unit/test_study_gaps.py` | gap display tests — TTY, non-TTY, interrupt | VERIFIED | 12 test functions, all pass |
| `tests/unit/test_study_manifest.py` | `test_manifest_initial_status_is_running`, `test_mark_interrupted_sets_status` (new) | VERIFIED | Both new tests present and passing; 21 total manifest tests |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `runner.py` | `multiprocessing.get_context('spawn')` | `mp_ctx = multiprocessing.get_context("spawn")` in `run()` | WIRED | Line 239; pattern confirmed |
| `runner.py` | `manifest.py` | `mark_running`, `mark_completed`, `mark_failed` called in `_run_one()` | WIRED | Lines 336, 363, 365; all three transition calls present |
| `runner.py` | `gaps.py` | `run_gap(gap_secs, "Config gap", self._interrupt_event)` between experiments | WIRED | Lines 275, 283; imported at line 23 |
| `runner.py` | `manifest.py` | `manifest_writer.mark_interrupted()` before `sys.exit(130)` | WIRED | Lines 300-301 |
| `runner.py` | `signal` module | `threading.Event` set by `_sigint_handler`; checked in run loop and `run_gap` | WIRED | `_interrupt_event` at line 216; handler at lines 246-259; checked at lines 268, 276, 284, 293 |
| `tests/test_study_runner.py` | `runner.py` | FakeWorker injection via `multiprocessing.get_context` mock | WIRED | `patch("multiprocessing.get_context", return_value=ctx)` pattern in all subprocess tests |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| STU-01 | 11-01-PLAN.md | `StudyRunner` iterates experiments, spawns via `get_context("spawn")` | SATISFIED | `runner.py:239`; `test_spawn_context_used` |
| STU-02 | 11-01-PLAN.md | IPC via `multiprocessing.Pipe` (Pipe-only; file fallback dropped) | SATISFIED | `runner.py:320-321`; `_collect_result` reads from pipe exclusively |
| STU-03 | 11-01-PLAN.md | `daemon=False` on subprocess | SATISFIED | `runner.py:326`; `test_daemon_false` |
| STU-04 | 11-01-PLAN.md | Timeout via `p.join(timeout=...)` + SIGKILL on timeout | SATISFIED | `runner.py:341` (join), `runner.py:151` (SIGKILL in `_collect_result`); `test_study_runner_timeout` asserts `p.terminate.assert_not_called()` |
| STU-06 | 11-02-PLAN.md | Config gap between experiments from user config | SATISFIED | `runner.py:272-277`; `execution.config_gap_seconds` read; `test_sigint_during_gap_exits_immediately` covers gap call |
| STU-07 | 11-01-PLAN.md, 11-02-PLAN.md | `cycle_order: sequential | interleaved` — interleaved round-robins | SATISFIED | `apply_cycles` wired at `runner.py:230-236`; `test_interleaved_ordering`, `test_sequential_ordering` |

**Note on STU-05:** STU-05 does not exist in `REQUIREMENTS.md` (the list is STU-01, STU-02, STU-03, STU-04, STU-06, STU-07 — STU-05 is absent). No orphaned requirements for this phase.

**Phase-10 requirements (STU-08, STU-09) are complete and not regressed:** `ManifestWriter.mark_running/completed/failed` all present; atomic write via `_atomic_write`; 511 unit tests pass.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `runner.py` | 80-83 | `_run_experiment_worker` raises `NotImplementedError` | Info | Intentional documented stub — backend wiring is Phase 12. The subprocess isolation infrastructure (Pipe, Queue, SIGKILL, manifest) is fully implemented and tested. Worker stub raises so tests can detect the structured-error path; this is correct Phase 11 behaviour. |
| `runner.py` | 116, 123 | `_consume_progress_events` discards events (display is Phase 12) | Info | Intentional — progress display wiring deferred to Phase 12. Queue is drained correctly so child never blocks. |
| `runner.py` | 314 | `cycle = 1` hardcoded | Info | Intentional — per-cycle tracking deferred to Phase 12 wiring. |

No blockers. No unexpected TODOs or placeholders. All deferred items are explicitly documented with Phase 12 references in inline comments.

---

### Human Verification Required

#### 1. Real CUDA Subprocess Isolation

**Test:** Run `llem run config.yaml` with a real GPU and two models; observe that each experiment spawns a separate process (visible in `ps aux` or `/proc`) with no CUDA context leakage between runs.
**Expected:** Each subprocess starts fresh with no inherited CUDA state; GPU memory returns to baseline between experiments.
**Why human:** Requires GPU hardware. CUDA state isolation cannot be verified programmatically without actual CUDA initialisation.

#### 2. Real Ctrl+C Behaviour During Active Experiment

**Test:** Start a multi-experiment study on GPU hardware; press Ctrl+C during an active experiment; verify "Interrupt received..." message appears; verify manifest.json on disk shows `status: "interrupted"` and exit code is 130.
**Expected:** Clean termination, no corrupted manifest, correct exit code, partial results preserved.
**Why human:** Real SIGINT delivery to a spawned subprocess requires an actual running process; unit tests simulate the handler but not true OS signal dispatch.

#### 3. Gap Countdown Display

**Test:** Run a study with `config_gap_seconds: 5` in a TTY; verify the countdown overwrites in-place ("Config gap: 5s remaining (Enter to skip)") and pressing Enter skips the remainder.
**Expected:** Single updating line, no duplicated output; Enter terminates gap early; gap marked "done".
**Why human:** Terminal rendering and keyboard input require a real TTY.

---

### Gaps Summary

No gaps. All 13 observable truths are verified with code evidence. All 6 requirement IDs (STU-01 through STU-04, STU-06, STU-07) are satisfied by substantive, wired implementations. 511 unit tests pass with no regressions.

The three items marked for human verification (CUDA isolation, real Ctrl+C, TTY display) are expected and noted in `11-02-PLAN.md`'s success criteria as requiring manual verification during Phase 12 integration testing.

---

_Verified: 2026-02-27T18:35:00Z_
_Verifier: Claude (gsd-verifier)_
