---
phase: 17-docker-runner-infrastructure
plan: "03"
subsystem: infra
tags: [docker, subprocess, container, dispatch, exchange-dir, lifecycle]

# Dependency graph
requires:
  - phase: 17-01
    provides: DockerError hierarchy, translate_docker_error, container_entrypoint, image_registry
  - phase: 17-02
    provides: RunnerSpec, resolve_runner, is_docker_available
provides:
  - DockerRunner class — dispatches experiments to ephemeral Docker containers
  - Full dispatch lifecycle: temp dir creation, config write, docker run, result read, cleanup
  - Runner metadata injection into ExperimentResult.effective_config
  - 15 unit tests covering success, failure, timeout, OOM, permission, missing result, command structure
affects: [study-runner, orchestration, phase-18-docker-preflight, phase-19-vllm-backend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Exchange dir pattern: tempfile.mkdtemp(prefix='llem-') for config/result file handoff
    - Preserve-on-failure: exchange dir kept on error, cleaned on success (debug artifact retention)
    - Lazy domain imports: heavy models imported at method call time, not module load
    - Runner metadata as result annotation: model_copy(update=...) on frozen ExperimentResult

key-files:
  created:
    - src/llenergymeasure/infra/docker_runner.py
    - tests/unit/test_docker_runner.py
  modified: []

key-decisions:
  - "DockerRunner.run() returns dict for error payloads (container exit 0 with error JSON) — mirrors StudyRunner worker contract, no effective_config injection on error dicts"
  - "Exchange dir preserved on all failure paths (timeout, non-zero exit, missing result) — debug artifacts must survive failures"
  - "HF_TOKEN propagated into docker run -e flag — gated model support without requiring user to rebuild images"

patterns-established:
  - "Exchange dir lifecycle: create → write config → run container → read result → cleanup on success only"
  - "exchange_dir = None sentinel: signals to finally-block that cleanup has been handled or should be skipped"
  - "Separate _build_docker_cmd helper: enables subclass customisation of docker flags"

requirements-completed: [DOCK-01, DOCK-04]

# Metrics
duration: 5min
completed: 2026-02-28
---

# Phase 17 Plan 03: DockerRunner Summary

**DockerRunner class dispatching experiments to ephemeral containers via subprocess.run with temp-dir config/result exchange and categorised error translation**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-28T02:51:04Z
- **Completed:** 2026-02-28T02:56:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- DockerRunner class with complete dispatch lifecycle: create temp dir, write config JSON, docker run --rm --gpus all --shm-size 8g, read result JSON, clean up on success / preserve on failure
- 15 unit tests covering all paths: success, image pull error, OOM, timeout, permission denied, missing result file, error payload dict, command structure verification, HF_TOKEN propagation, runner metadata injection, cleanup warning
- Runner metadata (runner_type, runner_image, runner_source) injected into ExperimentResult.effective_config via model_copy on frozen model
- Exchange dir sentinel pattern: `exchange_dir = None` in finally-block avoids double-cleanup on unexpected exceptions

## Task Commits

1. **Task 1: DockerRunner class with full dispatch lifecycle** - `5c1df19` (feat) + `be7c3d6` (fix: ruff B904)
2. **Task 2: DockerRunner unit tests** - `9f194f3` (feat: tests + docker_runner.py success-path refinement)

## Files Created/Modified

- `src/llenergymeasure/infra/docker_runner.py` — DockerRunner class with run(), _build_docker_cmd(), _read_result(), _cleanup_exchange_dir(), _inject_runner_metadata()
- `tests/unit/test_docker_runner.py` — 15 unit tests, all passing, no real Docker required

## Decisions Made

- Error payload dicts returned as-is without runner metadata injection — they have no `effective_config` field (container wrote error before producing a result)
- `exchange_dir = None` sentinel in the success path avoids the finally-block attempting cleanup again after explicit `_cleanup_exchange_dir()` call
- Separate `_read_result()` helper to isolate file I/O and error-payload detection

## Deviations from Plan

None — plan executed exactly as written. Task 2 tests include 15 scenarios (4 more than the 11 listed in the plan) covering additional `make_result()` variants and HF_TOKEN absent case.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- DockerRunner ready for consumption by StudyRunner (Phase 20) to dispatch docker-mode experiments
- Phase 18 (Docker pre-flight) can now be tested against the real DockerRunner interface
- All DOCK-01 and DOCK-04 requirements met: ephemeral container dispatch via subprocess.run blocking call

---
*Phase: 17-docker-runner-infrastructure*
*Completed: 2026-02-28*
