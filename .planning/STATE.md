---
gsd_state_version: 1.0
milestone: v1.19
milestone_name: "Docker + vLLM"
status: in_progress
last_updated: "2026-02-28T11:00:00.000Z"
progress:
  total_phases: 9
  completed_phases: 1
  total_plans: 1
  completed_plans: 1
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** Phase 17 — Docker Runner Infrastructure

## Current Position

Phase: 16 of 23 complete (GPU Memory Verification — merged to main)
Next: Phase 17 (Docker Runner Infrastructure)
Status: Phase 16 shipped, Phase 17 not started
Last activity: 2026-02-28 — Phase 16 squash-merged to main (PR #24)

Progress: [█░░░░░░░░░] 11%

## Performance Metrics

**Velocity:**
- Total plans completed (M3): 1
- Average duration: 173s
- Total execution time: 173s

*Updated after each plan completion*

## Accumulated Context

### Decisions

- StudyRunner dispatches via `multiprocessing.spawn` + Pipe IPC (local path, M2)
- Docker path is a parallel dispatch method in StudyRunner — not a separate runner class
- Multi-backend without Docker → hard error at pre-flight (M2, DOCK-05 extends this to auto-elevation)
- One backend per milestone: M3=vLLM, M4=TRT-LLM, M5=SGLang
- Phase 13 (docs) folded into Phase 22 of M3 — write docs once against final backend story
- Local import in _run_one() keeps pynvml lazy; avoids module-level ImportError when pynvml not installed (Phase 16)
- GPU memory threshold hardcoded at 100 MB for M3; configurability deferred until researcher demand (Phase 16)

### Carried Items

1. `aienergyscore.jsonl` built-in dataset — Phase 21 (MEAS-03)
2. `peak_memory_mb` semantics confirmation — Phase 21 (MEAS-04)
3. Manual Ctrl+C SIGINT test on GPU hardware — Phase 23 (TEST-01)

### Blockers/Concerns

- CUDA/GPU only available inside containers on this host — Docker pre-flight (Phase 18) and vLLM tests (Phase 19) require container execution to verify

## Session Continuity

Last session: 2026-02-28
Stopped at: Phase 16 merged to main via PR #24. Ready for Phase 17.
Resume file: None
