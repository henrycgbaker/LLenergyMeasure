---
phase: 18-docker-pre-flight
plan: 01
subsystem: infra
tags: [docker, preflight, nvidia, cuda, gpu, subprocess, exceptions]

# Dependency graph
requires:
  - phase: 17-docker-runner-infrastructure
    provides: DockerRunner, runner_resolution, docker_errors — infrastructure this pre-flight protects

provides:
  - DockerPreFlightError exception class in exceptions hierarchy (PreFlightError subclass)
  - docker_preflight.py: run_docker_preflight() with tiered Tier 1 (host) + Tier 2 (container) checks
  - --skip-preflight CLI flag on llem run
  - execution.skip_preflight YAML config field in ExecutionConfig
  - run_study_preflight() wired to Docker pre-flight for Docker runner studies

affects:
  - phase-19-vllm-backend (uses Docker runner, benefits from pre-flight validation)
  - phase-20-tensorrt-backend (same)
  - phase-22-documentation (docs Docker pre-flight flags and error messages)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Tiered check pattern: collect all Tier 1 failures before attempting Tier 2 (container probe)
    - Warn-only host check: missing host nvidia-smi logs warning but does not block (remote daemon)
    - CLI flag OR YAML config bypass: effective_skip = cli_flag OR yaml_config
    - Local function imports for Docker pre-flight (lazy, avoids circular deps)

key-files:
  created:
    - src/llenergymeasure/infra/docker_preflight.py
    - tests/unit/test_docker_preflight.py
  modified:
    - src/llenergymeasure/exceptions.py
    - src/llenergymeasure/config/models.py
    - src/llenergymeasure/orchestration/preflight.py
    - src/llenergymeasure/cli/run.py
    - src/llenergymeasure/_api.py
    - tests/unit/test_api.py

key-decisions:
  - "DockerPreFlightError inherits PreFlightError (not DockerError) so CLI error handler catches it without changes"
  - "CUDA compat error detection uses specific patterns (cuda+version, driver/library version mismatch, nvml+driver) to avoid false positives from generic GPU access errors that mention 'device driver'"
  - "run_docker_preflight uses local import inside run_study_preflight — avoids circular import and keeps pre-flight lazy"
  - "resolve_study_runners called without yaml_runners/user_config in pre-flight — uses auto-detection path for whether Docker runners are active"

requirements-completed: [DOCK-07, DOCK-08, DOCK-09]

# Metrics
duration: 13min
completed: 2026-02-28
---

# Phase 18 Plan 01: Docker Pre-flight Checks Summary

**Tiered Docker pre-flight validation with GPU container probe, CUDA/driver compat check, --skip-preflight CLI flag, and full unit test coverage**

## Performance

- **Duration:** 13 min
- **Started:** 2026-02-28T03:15:11Z
- **Completed:** 2026-02-28T03:28:24Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Tiered Docker pre-flight module: Tier 1 checks host environment (Docker CLI, NVIDIA Container Toolkit, nvidia-smi), Tier 2 validates GPU access inside container
- Missing host nvidia-smi warns but does not block — supports remote Docker daemon scenarios
- All Tier 1 failures reported together (numbered list) before aborting, Tier 2 only reached if Tier 1 passes
- --skip-preflight CLI flag and execution.skip_preflight YAML config field both bypass all checks; CLI takes priority
- run_study_preflight() wired to call run_docker_preflight() when Docker runners are resolved
- 63 GPU-free unit tests covering all check paths, skip behaviour, inheritance, wiring, CLI flag existence

## Task Commits

1. **Task 1: Docker pre-flight check module with tiered execution and unit tests** — `765a251` (feat)
2. **Task 2: Wire pre-flight into CLI and study execution path** — `2757b94` (feat)

## Files Created/Modified

- `src/llenergymeasure/infra/docker_preflight.py` — Tiered Docker pre-flight checks with run_docker_preflight()
- `src/llenergymeasure/exceptions.py` — DockerPreFlightError(PreFlightError) added
- `src/llenergymeasure/config/models.py` — ExecutionConfig.skip_preflight field added
- `src/llenergymeasure/orchestration/preflight.py` — run_study_preflight() wired to Docker pre-flight
- `src/llenergymeasure/cli/run.py` — --skip-preflight flag added, threaded through
- `src/llenergymeasure/_api.py` — skip_preflight threaded through run_experiment(), run_study(), _run()
- `tests/unit/test_docker_preflight.py` — 63 GPU-free unit tests (created)
- `tests/unit/test_api.py` — Mock signatures updated to accept skip_preflight keyword arg

## Decisions Made

- DockerPreFlightError inherits PreFlightError rather than DockerError, so existing CLI `except (PreFlightError, ...)` handler catches it without modification.
- CUDA compatibility error detection was tightened to avoid false positives: the phrase "device driver" (from GPU access errors) does not trigger the CUDA compat path; only explicit CUDA version/incompatible/NVML mismatch patterns do.
- run_docker_preflight is imported inside the function body in run_study_preflight (local import pattern) to keep the import lazy and avoid potential circular dependencies.
- resolve_study_runners is called without yaml_runners/user_config in the preflight wiring — uses auto-detection to determine if any runner will be Docker mode.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] CUDA compat detection pattern over-matched "device driver" in generic GPU errors**
- **Found during:** Task 1 (unit test TestTier2GPUVisibility)
- **Issue:** Original logic triggered CUDA compat error path for any stderr containing "driver", which incorrectly matched the generic "could not select device driver" GPU access error
- **Fix:** Replaced broad "driver in stderr" check with specific CUDA compat patterns: cuda+version, cuda+incompatible, "driver/library version mismatch", nvml+driver
- **Files modified:** src/llenergymeasure/infra/docker_preflight.py
- **Verification:** TestTier2GPUVisibility and TestTier2CUDADriverCompat all pass
- **Committed in:** 765a251 (Task 1 commit, amended)

**2. [Rule 1 - Bug] test_api.py mock functions lacked skip_preflight keyword arg acceptance**
- **Found during:** Task 2 (full unit suite run)
- **Issue:** _run() and run_study_preflight() mock lambdas/functions in test_api.py used single positional arg, so the new skip_preflight=False keyword argument caused TypeError
- **Fix:** Updated all _run mocks to `lambda study, **kw:` and all mock_run/mock_preflight function defs to accept `**kw`
- **Files modified:** tests/unit/test_api.py
- **Verification:** 704 unit tests pass
- **Committed in:** 2757b94 (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 logic bug, 1 test compatibility bug)
**Impact on plan:** Both auto-fixes necessary for correctness. No scope creep.

## Issues Encountered

- Git branching confusion: initial Task 1 edits were made while on `main` branch, then branch was switched. All changes were recovered via git stash and re-applied on the correct `gsd/phase-18-docker-pre-flight` branch.
- Patch targets for wiring tests: `resolve_study_runners` and `run_docker_preflight` are imported inside function bodies (local imports), so tests must patch them at their source module (`llenergymeasure.infra.runner_resolution`, `llenergymeasure.infra.docker_preflight`), not at the preflight module level.

## Next Phase Readiness

- Docker pre-flight validation complete — any backend using Docker runner will get host + container GPU checks before container launch
- Phase 19 (vLLM backend activation) can rely on pre-flight to validate environment before vLLM container runs
- --skip-preflight available for CI/CD environments and remote Docker daemon setups

## Self-Check: PASSED

- docker_preflight.py: FOUND
- test_docker_preflight.py: FOUND
- 18-01-SUMMARY.md: FOUND
- Task 1 commit 765a251: FOUND
- Task 2 commit 2757b94: FOUND

---
*Phase: 18-docker-pre-flight*
*Completed: 2026-02-28*
