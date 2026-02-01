---
phase: 02-campaign-orchestrator
plan: 02
subsystem: orchestration
tags: [docker, python-on-whales, container-lifecycle, gpu-health, nvml]

requires:
  - phase: 02-01
    provides: CampaignConfig with backend field and campaign execution config
provides:
  - ContainerManager class with start/exec/health/restart/teardown lifecycle
  - ContainerHealthStatus dataclass for GPU memory reporting
affects: [02-03, 02-04, 02-05, 02-08]

tech-stack:
  added: [python-on-whales]
  patterns: [lazy-optional-import, docker-compose-exec-over-run, context-manager-teardown]

key-files:
  created:
    - src/llenergymeasure/orchestration/container.py
  modified: []

key-decisions:
  - "Lazy import with _is_docker_exception() type-check pattern instead of catching dynamic exception type — avoids mypy errors while keeping python-on-whales optional"
  - "Did not add ContainerManager to orchestration __init__.py — optional dependency would break imports for users without python-on-whales"

patterns-established:
  - "Lazy optional dependency: _create_docker_client() and _is_docker_exception() helpers for type-safe dynamic imports"
  - "Docker exec dispatch: execute_experiment() uses compose.execute() not compose.run() for long-running containers"

duration: 6min
completed: 2026-01-29
---

# Phase 02 Plan 02: ContainerManager Summary

**Docker container lifecycle manager using python-on-whales with up/exec/health-check/restart/teardown and GPU memory monitoring via NVML**

## Performance

- **Duration:** 6 min
- **Started:** 2026-01-29T22:27:40Z
- **Completed:** 2026-01-29T22:33:43Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- ContainerManager class replacing docker compose run --rm with long-running container lifecycle
- GPU health checks via NVML exec inside containers with configurable memory threshold
- Auto-restart with max retry tracking and health verification after restart
- Context manager support ensuring teardown on any failure path

## Task Commits

Each task was committed atomically:

1. **Task 1: Create ContainerManager with full lifecycle** - `c339005` (feat)
2. **Task 2: Update orchestration __init__ exports** - No commit (no changes needed; optional dependency cannot be top-level imported)

## Files Created/Modified
- `src/llenergymeasure/orchestration/container.py` - ContainerManager with 7 lifecycle methods + context manager, ContainerHealthStatus dataclass

## Decisions Made
- Used lazy import pattern with `_create_docker_client()` factory and `_is_docker_exception()` type-checker to avoid mypy errors from dynamic exception catching while keeping python-on-whales fully optional
- Did not add ContainerManager to `orchestration/__init__.py` top-level exports — importing would require python-on-whales at module load time, breaking the package for users without the optional dependency

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed mypy errors with dynamic exception catching**
- **Found during:** Task 1 (pre-commit hook)
- **Issue:** Original pattern `DockerException = _get_docker_exception()` then `except DockerException` produced mypy errors: "Returning Any from function declared to return type" and "Exception type must be derived from BaseException"
- **Fix:** Replaced with `_is_docker_exception()` type-checker function, catching `Exception` and re-raising if not a DockerException
- **Files modified:** src/llenergymeasure/orchestration/container.py
- **Verification:** mypy passes with no errors
- **Committed in:** c339005

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Mypy-safe pattern for dynamic exception handling. No scope creep.

## Issues Encountered
None beyond the mypy fix above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ContainerManager ready for integration with CampaignOrchestrator (02-04)
- Health check and restart patterns ready for campaign execution loop (02-05)
- No blockers

---
*Phase: 02-campaign-orchestrator*
*Completed: 2026-01-29*
