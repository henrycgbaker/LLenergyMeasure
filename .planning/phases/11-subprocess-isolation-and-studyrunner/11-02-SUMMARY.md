---
plan: 11-02
phase: 11-subprocess-isolation-and-studyrunner
subsystem: study
status: complete
tags: [sigint, gap-countdown, interrupt-handling, thermal-gaps, tty]
files_modified:
  - src/llenergymeasure/study/manifest.py
  - src/llenergymeasure/study/gaps.py
  - src/llenergymeasure/study/runner.py
  - tests/unit/test_study_manifest.py
  - tests/unit/test_study_gaps.py
  - tests/unit/test_study_runner.py
tests_added: 17
tests_total: 511
dependency_graph:
  requires:
    - llenergymeasure.study.manifest (StudyManifest, ManifestWriter)
    - llenergymeasure.study.runner (StudyRunner — from Plan 01)
    - llenergymeasure.config.models (ExecutionConfig.config_gap_seconds, cycle_gap_seconds)
  provides:
    - StudyManifest.status field ("running"|"completed"|"interrupted"|"failed")
    - ManifestWriter.mark_interrupted() — atomic status write
    - gaps.run_gap() — SIGINT-safe countdown with Enter-to-skip
    - gaps.format_gap_duration() — display formatting helper
    - StudyRunner SIGINT handler (two-stage: SIGTERM → SIGKILL)
    - StudyRunner gap calls between experiments
  affects:
    - src/llenergymeasure/study/__init__.py (gaps not yet exported — internal use only)
tech_stack:
  added:
    - threading.Event for interrupt coordination (interrupt_event shared between SIGINT handler and run_gap)
    - signal.signal() install/restore pattern (SIGINT handler scoped to run() lifetime)
    - daemon readline thread for Enter-to-skip (TTY-only, degrades gracefully)
    - contextlib.suppress() for best-effort Pipe/Queue sends in worker (Ruff SIM105 auto-fix)
  patterns:
    - interrupt_event passed by reference to run_gap() — clean SIGINT propagation into gap
    - Two-stage SIGTERM→SIGKILL: 2s grace for CUDA teardown after SIGTERM before SIGKILL
    - try/finally around experiment loop ensures signal.signal() is always restored
    - sys.exit(130) standard POSIX convention for SIGINT-terminated process
key_files:
  created:
    - src/llenergymeasure/study/gaps.py
    - tests/unit/test_study_gaps.py
  modified:
    - src/llenergymeasure/study/manifest.py
    - src/llenergymeasure/study/runner.py
    - tests/unit/test_study_manifest.py
    - tests/unit/test_study_runner.py
decisions:
  - "Grace period 2s (within 2-3s range per CONTEXT.md): chosen for balance between clean teardown and responsiveness"
  - "Enter-to-skip uses daemon readline thread (not select/termios): simplest TTY-degrading approach"
  - "interrupt_event.clear() at run() start: allows runner to be re-used; SIGINT state is per-run"
  - "SIGINT tests use _run_one side_effect injection: avoids patching signal.signal() globally which is fragile"
metrics:
  duration: "~5 min"
  completed: "2026-02-27T18:18:24Z"
  tasks: 3
  files: 6
---

# Phase 11 Plan 02: SIGINT Handling and Gap Countdown Summary

**One-liner:** StudyRunner SIGINT handling with two-stage SIGTERM→SIGKILL escalation, manifest.mark_interrupted(), sys.exit(130), and thermal gap countdown (Enter-to-skip, TTY-safe) via threading.Event coordination.

## What Was Built

Three connected additions that together give researchers a clean Ctrl+C experience during multi-hour studies.

### Task 1: StudyManifest.status + ManifestWriter.mark_interrupted()

**`src/llenergymeasure/study/manifest.py`**
- `StudyManifest.status` field: `Literal["running", "completed", "interrupted", "failed"]`, default `"running"`. Placed after `completed_at` in field order.
- `ManifestWriter.mark_interrupted()`: atomic model_copy + _write(). Called by StudyRunner before sys.exit(130).
- 2 new tests confirming initial status is "running" and mark_interrupted() writes "interrupted" to disk.

### Task 2: gaps.py — countdown display

**`src/llenergymeasure/study/gaps.py`**
- `format_gap_duration(seconds)`: under 120s → "47s remaining"; 120s+ → "4m 32s remaining"
- `run_gap(seconds, label, interrupt_event)`: inline `\r`-overwrite countdown. Enter-to-skip launches a daemon readline thread (TTY only, no-op in CI/piped). `interrupt_event` checked every tick — exits immediately when set by SIGINT handler.
- 12 unit tests covering formatting edge cases, 0-second gaps, pre-set interrupt event, mid-gap interrupt, and non-TTY environments.

### Task 3: StudyRunner SIGINT wiring and gap calls

**`src/llenergymeasure/study/runner.py`**

SIGINT handler (`_sigint_handler`) installed in `run()` before the experiment loop:
- **First Ctrl+C**: prints "Interrupt received...", calls SIGTERM on active subprocess, sets `_interrupt_event`
- **Second Ctrl+C**: prints "Force-killing...", calls SIGKILL on active subprocess

Post-join interrupt handling in `_run_one()`:
- After `p.join(timeout=timeout)`, if `interrupt_event.is_set()` and `p.is_alive()`: 2s grace join → SIGKILL if still alive

Gap calls inserted in the ordered experiment loop:
- Config gap: between every consecutive experiment pair (`i > 0`)
- Cycle gap: after every N-experiment round (`i > 0 and i % n_unique == 0`)

Exit path: `try/finally` restores original SIGINT handler; if `interrupt_event.is_set()` prints summary, calls `manifest.mark_interrupted()`, and `sys.exit(130)`.

3 new SIGINT tests:
- Test A: interrupt after first experiment → exit 130 + mark_interrupted called
- Test B: interrupt during gap (via fake_run_gap side_effect) → exit 130 + mark_interrupted called
- Test C: second Ctrl+C simulation → p.kill() called

## Deviations from Plan

**[Rule 3 - Auto-fix] `signal` import missing in test file**
- **Found during:** Task 3 test writing
- **Issue:** Test used `signal.signal` without importing `signal`
- **Fix:** Added `import signal` and `import threading` to test imports; also simplified test approach (direct `_run_one` side_effect rather than capturing signal handler)
- **Files modified:** `tests/unit/test_study_runner.py`
- **Commit:** 7757863

**[Rule 2 - Auto-fix] Ruff SIM105: `try/except/pass` → `contextlib.suppress()`**
- **Found during:** Pre-commit hook on Tasks 2 and 3 commits
- **Issue:** Ruff lint rule SIM105 flagged `try/except/pass` patterns in `gaps.py` and `runner.py`
- **Fix:** Applied automatically by Ruff formatter during pre-commit
- **Files modified:** `gaps.py`, `runner.py`

## Self-Check: PASSED

Files:
- `src/llenergymeasure/study/manifest.py` — FOUND (status field + mark_interrupted)
- `src/llenergymeasure/study/gaps.py` — FOUND
- `src/llenergymeasure/study/runner.py` — FOUND (SIGINT handler + gap calls)
- `tests/unit/test_study_manifest.py` — FOUND (21 tests)
- `tests/unit/test_study_gaps.py` — FOUND (12 tests)
- `tests/unit/test_study_runner.py` — FOUND (13 tests)

Commits:
- `589d880` — Task 1: status field + mark_interrupted
- `95140f4` — Task 2: gaps.py
- `7757863` — Task 3: SIGINT + gap wiring

All 511 unit tests pass.
