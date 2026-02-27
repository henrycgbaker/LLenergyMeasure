---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: unknown
last_updated: "2026-02-27T21:11:55.721Z"
progress:
  total_phases: 27
  completed_phases: 22
  total_plans: 86
  completed_plans: 78
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** M2 — Study / Sweep (Phase 10 in progress)

## Current Position

Phase: 15
Plan: 15-01 complete (Phase 15 in progress)
Status: Phase 15 Plan 02 next
Last activity: 2026-02-27 — Phase 15 Plan 01 complete: progress display wired (_consume_progress_events → print_study_progress), phantom experiment_timeout_seconds getattr removed, 536 tests pass

Progress: [██░░░░░░░░] ~10%

## Performance Metrics

**Velocity (M1 reference):**
- M1 plans completed: 29
- Average duration: ~7 min/plan
- M1 total execution time: ~3.5 hours

**M2 By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 09-grid-expansion | 2 completed | 10 min | 5 min |
| 10-manifest-writer | 1 completed | 3 min | 3 min |
| 11-subprocess-isolation | 2 completed | 8 min | 4 min |
| 12-integration | 1 completed | TBD | - |
| 08.1-pytorch-result-wiring | 1 completed | 3 min | 3 min |

*Updated after each plan completion*
| Phase 08.2-m1-tech-debt-cleanup P01 | 3 | 2 tasks | 3 files |
| Phase 08.2 P02 | 4 | 2 tasks | 6 files |
| Phase 11 P02 | 5 min | 3 tasks | 6 files |
| Phase 12-integration P02 | 8 | 2 tasks | 4 files |
| Phase 12-integration P03 | 3 | 2 tasks | 3 files |
| Phase 14 P01 | 13 | 2 tasks | 5 files |
| Phase 15 P02 | 1 | 2 tasks | 6 files |
| Phase 15 P01 | 2 | 2 tasks | 2 files |

## Accumulated Context

### Decisions

All M2 decisions pre-confirmed in `.product/decisions/`. Key points for execution:

- `multiprocessing.get_context("spawn")` — never `set_start_method()` globally (CP-1: fork causes silent CUDA corruption)
- Pipe-only IPC (file fallback dropped) — `ExperimentResult` fits in Pipe buffer for M2 experiment sizes
- `daemon=False` on subprocess — clean CUDA teardown (CP-4)
- `os.replace()` for manifest writes — atomic POSIX rename from day one (CP-6)
- STU-NEW-01: `_run()` body implements dispatcher in Phase 12 (M1 left it as `NotImplementedError`)
- CM-10: Multi-backend without Docker is a hard `PreFlightError` in M2 (Docker is M3)
- `--resume` deferred to M4 — manifest always-on in M2 enables it later
- StrEnum backport pattern for Python 3.10 (sys.version_info guard + str+Enum fallback)
- Multi-backend sweep: independent Cartesian grids per backend (not cross-product between backends)
- [Phase 09]: TYPE_CHECKING import for StudyConfig in grid.py avoids circular import with loader.py at runtime
- [Phase 08.1-01]: Add extra='forbid' only to ExperimentResult (user-visible output), not RawProcessResult
- [Phase 08.1-01]: ts_source.unlink(missing_ok=True) after co-location: safe because save_result() copies the file before deletion
- [Phase 08.2-01]: cli/CLAUDE.md is excluded from git tracking by machine-local .git/info/exclude — corrected locally only
- [Phase 08.2-01]: Retroactive Phase 2 verification uses UAT 11/11 results and SUMMARY frontmatter as evidence — no re-execution required
- [Phase 08.2]: Removed only state.experiment_state import from cli/experiment.py; left rest of dead module intact to minimise change risk
- [Phase 08.2]: Inline field access replaces calculate_efficiency_metrics() in cli/results.py and cli/display/results.py
- [Phase 10-01]: Lazy-import __version__ inside _build_manifest() to break circular import (study.__init__ -> manifest -> llenergymeasure)
- [Phase 10-01]: study_design_hash used in StudyManifest (not study_yaml_hash) per CONTEXT.md CP-6 decision
- [Phase 11-01]: _run_experiment_worker is a stub (raises NotImplementedError) — wired to real backend in Phase 12
- [Phase 11-01]: cycle=1 hardcoded in _run_one — full per-cycle tracking deferred to Phase 12
- [Phase 11-01]: getattr(execution, 'experiment_timeout_seconds', None) forward-compat — field added in Phase 12
- [Phase 11]: Grace period 2s (2-3s range): balance between CUDA teardown and responsiveness
- [Phase 11]: Enter-to-skip uses daemon readline thread (not select/termios): simplest TTY-degrading approach
- [Phase 11]: interrupt_event.clear() at run() start: allows StudyRunner re-use; SIGINT state is per-run
- [Phase 12-01]: StudyResult has no extra='forbid' — internal model, not user-visible output
- [Phase 12-01]: run_study_preflight raises PreFlightError immediately for multi-backend (CM-10); Docker isolation is M3
- [Phase 12-01]: experiment_gap_seconds replaces config_gap_seconds in both ExecutionConfig and UserExecutionConfig
- [Phase 12-integration]: _run_in_process propagates PreFlightError and BackendError unchanged; only save failures are caught
- [Phase 12-integration]: Single experiment + n_cycles=1 dispatches in-process; multi-experiment via StudyRunner subprocess
- [Phase 12-integration]: result_files list contains absolute path strings; manifest entry stores relative path from study_dir
- [Phase 12-integration]: CLI effective defaults n_cycles=3 and cycle_order=shuffled applied only when YAML execution block omits those keys
- [Phase 12-integration]: quiet suppresses CLI-side progress and summary; gap countdown suppression deferred (M2 limitation, subprocess-level)
- [Phase 14]: Remove apply_cycles() from runner: study.experiments is already cycled by load_study_config() — runner consumes the pre-ordered list as-is
- [Phase 14]: _cycle_counters dict in StudyRunner: per-config_hash cycle numbers, reset per run(), incremented per _run_one() call — correct regardless of cycle_order
- [Phase 14]: _build_entries deduplication: deduplicate study.experiments by config_hash before looping n_cycles — recovers correct n_unique*n_cycles entry count from the pre-cycled list
- [Phase 14]: mark_study_completed only reached on success path: SIGINT path calls sys.exit(130) before _run() returns, so no guard needed
- [Phase 15]: STU-07 credited to 11-02 (original cycle ordering implementation) despite Phase 14 fixing the double-apply bug
- [Phase 15]: Lazy import pattern for print_study_progress inside event handlers: avoids coupling study/runner.py to CLI layer at import time
- [Phase 15]: Phantom getattr(execution, experiment_timeout_seconds, None) removed — field never existed on ExecutionConfig (extra=forbid), always returned None
- [Phase 15]: _run_one() now takes explicit index and total params passed from run() loop — cleaner than threading through closure or class state

### Pending Todos

1. **Create `aienergyscore.jsonl` built-in dataset file** — carried from M1, resolve before M2 complete
2. **Confirm `peak_memory_mb` measurement semantics** — carried from M1, resolve before M2 complete
3. **Manual Ctrl+C SIGINT test on GPU hardware** — required in Phase 11 (not fully unit-testable)

### Roadmap Evolution

- Phase 13 added: Documentation — M1 backfill and M2 updates

### Blockers/Concerns

- [Phase 11] SIGINT/Ctrl+C interaction with CUDA teardown is the highest-risk path — requires manual GPU hardware verification, cannot be fully unit-tested

## Session Continuity

Last session: 2026-02-27
Stopped at: Completed 15-01-PLAN.md — progress display wired, phantom getattr removed: 536 tests pass
Resume file: None
Next action: Phase 13 — Documentation (M1 backfill and M2 updates)
