---
phase: 17-docker-runner-infrastructure
plan: "04"
subsystem: infra
tags: [docker, runner, study, preflight, dispatch, isolation]

# Dependency graph
requires:
  - phase: 17-01
    provides: DockerError hierarchy and container entrypoint
  - phase: 17-02
    provides: runner_resolution, RunnerSpec, is_docker_available, resolve_study_runners
  - phase: 17-03
    provides: DockerRunner.run() dispatch mechanic

provides:
  - Docker dispatch path wired into StudyRunner._run_one() and _api._run_in_process()
  - Auto-elevation for multi-backend studies in run_study_preflight()
  - Mixed runner warning in _api._run() when backends use different runner modes
  - runner_specs propagated from _api through _run_via_runner to StudyRunner

affects:
  - phase 18 (Docker pre-flight checks): can now test docker runner end-to-end
  - phase 19 (vLLM backend): dispatch path is ready
  - any future backend: Docker dispatch already wired

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Runner spec dispatch: check spec.mode before subprocess path in _run_one()"
    - "DockerError → failure dict: DockerErrors caught in _run_one_docker() and _run_in_process(), converted to non-fatal failure dicts"
    - "Auto-elevation: multi-backend preflight checks is_docker_available() before raising"
    - "runner_specs threaded through: _run() → _run_via_runner() / _run_in_process() → StudyRunner"

key-files:
  created: []
  modified:
    - src/llenergymeasure/orchestration/preflight.py
    - src/llenergymeasure/study/runner.py
    - src/llenergymeasure/_api.py
    - tests/unit/test_preflight.py
    - tests/unit/test_study_runner.py
    - tests/unit/test_api.py
    - tests/unit/test_study_preflight.py

key-decisions:
  - "Auto-elevation in preflight: log info and return (no error) when Docker available for multi-backend study; raise PreFlightError only when Docker not available"
  - "DockerRunner called blocking (no thread) in _run_one_docker() — Docker run is inherently blocking, no progress queue needed"
  - "_run_in_process() also gets Docker path: single-experiment API calls can dispatch to Docker containers directly without subprocess overhead"
  - "test_study_preflight.py tests must mock is_docker_available() — test machine has Docker+NVIDIA CT installed so real call auto-elevates"

patterns-established:
  - "Dispatch check pattern: spec = self._runner_specs.get(config.backend) if self._runner_specs else None; if spec and spec.mode == 'docker': use DockerRunner"
  - "Image resolution: use spec.image if spec.image is not None else get_default_image(config.backend)"

requirements-completed: [DOCK-05]

# Metrics
duration: 17min
completed: 2026-02-28
---

# Phase 17 Plan 04: Docker Runner Integration Summary

**Docker dispatch wired end-to-end: StudyRunner routes docker-spec backends through DockerRunner, preflight auto-elevates multi-backend studies, and _api resolves + propagates runner specs**

## Performance

- **Duration:** ~17 min
- **Started:** 2026-02-28
- **Completed:** 2026-02-28
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- `run_study_preflight()` upgraded from hard error to auto-elevation: Docker available → proceed with info log; Docker absent → PreFlightError with install guidance
- `StudyRunner` accepts `runner_specs: dict[str, RunnerSpec] | None` and dispatches to `_run_one_docker()` for docker-mode backends; subprocess path unchanged as fallback
- `_api._run()` resolves runners via `resolve_study_runners()` after preflight and passes specs through `_run_via_runner()` and `_run_in_process()`
- `_run_in_process()` handles docker-spec single experiments via DockerRunner directly (no subprocess spawn overhead)
- Mixed-runner warning logged when a study has backends across both local and docker modes
- 28 new tests across test_preflight.py, test_study_runner.py, test_api.py, test_study_preflight.py; 664 total unit tests passing

## Task Commits

1. **Task 1: Update preflight with auto-elevation logic** - `31dcea0` (feat)
2. **Task 2: Wire Docker dispatch into StudyRunner and _api** - `beee98a` (feat)

## Files Created/Modified

- `src/llenergymeasure/orchestration/preflight.py` — auto-elevation logic in run_study_preflight() using is_docker_available()
- `src/llenergymeasure/study/runner.py` — runner_specs param, _run_one() docker check, _run_one_docker() method
- `src/llenergymeasure/_api.py` — runner resolution in _run(), docker path in _run_in_process(), runner_specs threading
- `tests/unit/test_preflight.py` — 4 new run_study_preflight tests (single backend, auto-elevation, no-docker error, backend listing)
- `tests/unit/test_study_runner.py` — 4 new Docker dispatch tests (docker dispatch, local fallback, no specs fallback, DockerError non-fatal)
- `tests/unit/test_api.py` — 2 new tests (runner wiring, mixed runner warning)
- `tests/unit/test_study_preflight.py` — Fixed existing tests to mock is_docker_available() (machine has Docker+NVIDIA CT)

## Decisions Made

- Auto-elevation is silent (info log only, no user prompt) per 17-CONTEXT.md: minimal one-liner, proceed automatically
- `_run_one_docker()` calls `print_study_progress()` directly (no progress queue thread) — Docker dispatch is blocking, no IPC needed
- DockerErrors caught in both `_run_one_docker()` and `_run_in_process()` and returned as failure dicts to maintain non-fatal study execution
- Existing test_study_preflight.py tests needed monkeypatching for `is_docker_available()` because the host machine actually has Docker + NVIDIA CT installed

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_study_preflight.py tests failing due to real Docker detection**
- **Found during:** Task 1 (full regression run after preflight changes)
- **Issue:** test_multi_backend_raises_preflight_error and related tests called real is_docker_available() which returned True on this machine (Docker + nvidia-container-runtime present), causing auto-elevation instead of PreFlightError
- **Fix:** Added monkeypatch for is_docker_available() in all three tests that expect PreFlightError
- **Files modified:** tests/unit/test_study_preflight.py
- **Verification:** 664 unit tests pass
- **Committed in:** beee98a (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in existing tests exposed by behaviour change)
**Impact on plan:** Required fix for correctness. Tests must mock external detection to be deterministic.

## Issues Encountered

None beyond the test_study_preflight.py fix above.

## Next Phase Readiness

- Docker dispatch path is complete: setting `runner: docker` in config now routes experiments through DockerRunner containers
- Phase 18 (Docker pre-flight checks) can build on this — the pre-flight gap for container-level GPU validation is ready to fill
- Multi-backend studies will auto-elevate to Docker when available, satisfying DOCK-05

---
*Phase: 17-docker-runner-infrastructure*
*Completed: 2026-02-28*

## Self-Check: PASSED

- FOUND: src/llenergymeasure/orchestration/preflight.py
- FOUND: src/llenergymeasure/study/runner.py
- FOUND: src/llenergymeasure/_api.py
- FOUND: .planning/phases/17-docker-runner-infrastructure/17-04-SUMMARY.md
- FOUND: commit 31dcea0 (Task 1: preflight auto-elevation)
- FOUND: commit beee98a (Task 2: Docker dispatch wiring)
