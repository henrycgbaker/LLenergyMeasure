---
gsd_state_version: 1.0
milestone: v1.17
milestone_name: milestone
status: unknown
last_updated: "2026-02-28T03:08:43.745Z"
progress:
  total_phases: 4
  completed_phases: 2
  total_plans: 6
  completed_plans: 5
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-27)

**Core value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.
**Current focus:** Phase 18 — Docker Pre-flight Checks

## Current Position

Phase: 18 of 23 in progress (Docker Pre-flight — Plan 01 complete)
Next: Phase 18 Plan 02 (or Phase 19 — vLLM backend activation)
Status: Phase 18 Plan 01 shipped (Docker pre-flight module, --skip-preflight CLI, CUDA/driver compat checks, 63 unit tests)
Last activity: 2026-02-28 — Phase 18 Plan 01 complete on gsd/phase-18-docker-pre-flight

Progress: [████░░░░░░] 27%

## Performance Metrics

**Velocity:**
- Total plans completed (M3): 6
- Average duration: 248s
- Total execution time: 3078s (173s + 492s + 300s + 300s + 1020s + 793s)

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
- [Phase 17]: parse_runner_value raises ValueError on empty 'docker:' and unrecognised values — strict contract prevents silent fallbacks
- [Phase 17]: Container entrypoint uses core.backends.get_backend path (same as StudyRunner worker) — not orchestration factory — for identical measurement behaviour
- [Phase 17]: Error JSON format {type, message, traceback} mirrors StudyRunner worker payloads — consistent upstream consumer handling
- [Phase 17-02]: user_config=None enables auto-detection; user_config provided (even with default 'local') blocks auto-detection — explicit presence beats inference
- [Phase 17-02]: UserRunnersConfig accepts bare "docker" (not just "docker:<image>") to match YAML runner config syntax
- [Phase 17-03]: DockerRunner returns error payload dicts as-is (no effective_config injection) — error dicts have no result fields to annotate
- [Phase 17-03]: exchange_dir=None sentinel in finally-block prevents double-cleanup on unexpected exceptions
- [Phase 17-04]: Auto-elevation is info-log-only (no user prompt) — multi-backend with Docker proceeds automatically
- [Phase 17-04]: DockerErrors caught in _run_one_docker() and _run_in_process() — converted to non-fatal failure dicts, study continues
- [Phase 17-04]: test_study_preflight.py tests must mock is_docker_available() — host machine has Docker + NVIDIA CT installed
- [Phase 18-01]: DockerPreFlightError inherits PreFlightError (not DockerError) so existing CLI error handler catches it without changes
- [Phase 18-01]: CUDA compat detection uses specific patterns (cuda+version, driver/library mismatch, nvml+driver) to avoid false positives from generic GPU access errors containing "device driver"
- [Phase 18-01]: run_docker_preflight imported inside function body in run_study_preflight — lazy import prevents circular dependency
- [Phase 18-01]: resolve_study_runners called without yaml_runners/user_config in pre-flight — uses auto-detection to check if Docker runners are active

### Carried Items

1. `aienergyscore.jsonl` built-in dataset — Phase 21 (MEAS-03)
2. `peak_memory_mb` semantics confirmation — Phase 21 (MEAS-04)
3. Manual Ctrl+C SIGINT test on GPU hardware — Phase 23 (TEST-01)

### Blockers/Concerns

- CUDA/GPU only available inside containers on this host — Docker pre-flight (Phase 18) and vLLM tests (Phase 19) require container execution to verify

## Session Continuity

Last session: 2026-02-28
Stopped at: Phase 18 Plan 01 complete (Docker pre-flight checks, --skip-preflight CLI, 63 tests). Ready for Phase 18 Plan 02 or Phase 19 (vLLM backend).
Resume file: None
