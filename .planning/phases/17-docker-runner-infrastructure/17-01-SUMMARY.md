---
phase: 17-docker-runner-infrastructure
plan: "01"
subsystem: infra
tags: [docker, containers, error-handling, image-registry, runner-resolution]

# Dependency graph
requires:
  - phase: 15-study-runner-core
    provides: StudyRunner worker pattern (core.backends.get_backend + orchestration.preflight)
  - phase: 16-gpu-memory-verification
    provides: Pre-flight hook pattern used in container entrypoint

provides:
  - DockerError hierarchy with 5 categorised subclasses and fix suggestions
  - translate_docker_error() mapping container stderr to actionable error types
  - capture_stderr_snippet() for last-N-lines extraction
  - run_container_experiment() container-side entry point via library API (DOCK-11)
  - main() container entry point reading LLEM_CONFIG_PATH env var
  - get_default_image() resolving backend to ghcr.io image with version + CUDA tags
  - parse_runner_value() parsing "local"/"docker"/"docker:image" with validation
  - get_cuda_major_version() detecting host CUDA major via nvcc/pynvml
  - RunnerSpec dataclass with precedence-chain source tracking
  - resolve_runner() full precedence chain (env > yaml > user_config > auto > default)
  - resolve_study_runners() multi-backend runner resolution
  - StudyConfig.runners field for per-backend runner configuration
  - UserRunnersConfig per-backend runner fields

affects:
  - 17-02-PLAN (DockerRunner dispatch — uses DockerError, image_registry, runner_resolution)
  - 17-04-PLAN (StudyRunner integration — uses runner_resolution, container_entrypoint)
  - 18-docker-pre-flight (uses image_registry, runner_resolution)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Docker error hierarchy with categorised subclasses and fix_suggestion attribute"
    - "Container entrypoint via library API (not CLI re-entry) — DOCK-11"
    - "Image template with {backend}/{version}-cuda{cuda_major} naming"
    - "Precedence chain: env > yaml > user_config > auto-detection > default"
    - "RunnerSpec dataclass with source tracking for debugging resolution"
    - "parse_runner_value raises ValueError on empty/unrecognised — strict contract"

key-files:
  created:
    - src/llenergymeasure/exceptions.py (DockerError base class added)
    - src/llenergymeasure/infra/docker_errors.py
    - src/llenergymeasure/infra/container_entrypoint.py
    - src/llenergymeasure/infra/image_registry.py
    - src/llenergymeasure/infra/runner_resolution.py
    - tests/unit/test_docker_errors.py
    - tests/unit/test_container_entrypoint.py
    - tests/unit/test_image_registry.py
    - tests/unit/test_runner_resolution.py
  modified:
    - src/llenergymeasure/config/models.py (StudyConfig.runners field)
    - src/llenergymeasure/config/user_config.py (UserRunnersConfig per-backend fields)
    - src/llenergymeasure/config/loader.py (wire runners field through study loading)
    - src/llenergymeasure/study/grid.py (pass runners through grid expansion)

key-decisions:
  - "parse_runner_value raises ValueError on empty 'docker:' and unrecognised values — strict contract (not silent fallback)"
  - "Container entrypoint uses core.backends.get_backend path (same as StudyRunner worker) — not orchestration factory path"
  - "Error JSON format {type, message, traceback} mirrors StudyRunner worker error payloads for consistent handling"
  - "get_cuda_major_version uses lru_cache — single detection per process, cleared in tests"
  - "runner_resolution.py re-exports parse_runner_value from image_registry for consumer convenience"
  - "StudyConfig.runners is metadata, not part of experiment config hash"

patterns-established:
  - "Docker error: every subclass carries fix_suggestion and stderr_snippet attributes"
  - "Container volume: /run/llem/{hash}_config.json in, /run/llem/{hash}_result.json out"
  - "Runner test patches: patch source module (llenergymeasure.core.backends.get_backend), not container_entrypoint references, due to function-local imports"

requirements-completed: [DOCK-02, DOCK-03, DOCK-11]

# Metrics
duration: 8min
completed: 2026-02-28
---

# Phase 17 Plan 01: Docker Foundation Types Summary

**DockerError hierarchy with categorised stderr translation, container-side entry point running experiments via library API, and built-in image registry with CUDA detection and runner precedence chain**

## Performance

- **Duration:** 8 min
- **Started:** 2026-02-28T02:28:00Z
- **Completed:** 2026-02-28T02:36:12Z
- **Tasks:** 2
- **Files modified:** 13

## Accomplishments

- DockerError hierarchy with 5 categorised subclasses (ImagePull, GPUAccess, OOM, Permission, Timeout) and a generic fallback, each with `fix_suggestion` and `stderr_snippet` attributes
- `translate_docker_error()` maps Docker container stderr to actionable error types via case-insensitive pattern matching (returncode 124/-9/-15 → timeout, nvidia patterns → GPU error, etc.)
- `run_container_experiment()` reads ExperimentConfig JSON, runs via `core.backends.get_backend` (same path as StudyRunner worker, not CLI re-entry — DOCK-11), writes result JSON to shared volume
- `main()` container entry point reads `LLEM_CONFIG_PATH` env var, writes error JSON on failure matching StudyRunner worker error payload format
- `get_default_image()` resolves backend → `ghcr.io/llenergymeasure/{backend}:{version}-cuda{major}` with CUDA detection via nvcc/pynvml, falling back to "latest"
- `parse_runner_value()` with strict contract (raises on empty "docker:" and unrecognised values)
- `resolve_runner()` with full 5-layer precedence chain and `RunnerSpec` dataclass tracking resolution source
- `StudyConfig.runners` field wired through loader and grid expansion

## Task Commits

1. **Task 1: Docker error hierarchy and stderr translation** - `a43c458` (feat)
2. **Task 2: Container entrypoint and built-in image registry** - `5d4f812` (feat)

## Files Created/Modified

- `src/llenergymeasure/exceptions.py` — DockerError base class added
- `src/llenergymeasure/infra/docker_errors.py` — Error hierarchy and translate_docker_error()
- `src/llenergymeasure/infra/container_entrypoint.py` — Container-side entry point (DOCK-11)
- `src/llenergymeasure/infra/image_registry.py` — Image registry, CUDA detection, parse_runner_value()
- `src/llenergymeasure/infra/runner_resolution.py` — Full precedence chain runner resolution
- `src/llenergymeasure/config/models.py` — StudyConfig.runners field
- `src/llenergymeasure/config/user_config.py` — UserRunnersConfig per-backend fields
- `src/llenergymeasure/config/loader.py` — Wire runners through study loading
- `src/llenergymeasure/study/grid.py` — Pass runners through grid expansion
- `tests/unit/test_docker_errors.py` — 42 tests
- `tests/unit/test_container_entrypoint.py` — 8 tests
- `tests/unit/test_image_registry.py` — 13 tests
- `tests/unit/test_runner_resolution.py` — 34 tests (pre-existing tests now unblocked)

## Decisions Made

- `parse_runner_value` raises `ValueError` on empty image (`"docker:"`) and unrecognised runner types — strict contract matches the pre-existing test expectations in `test_runner_resolution.py`
- Container entrypoint uses `core.backends.get_backend` path (matches StudyRunner worker) rather than the orchestration factory path — identical measurement behaviour inside and outside container
- Error JSON format `{type, message, traceback}` mirrors StudyRunner worker format — consistent upstream consumer handling
- `get_cuda_major_version` uses `lru_cache(maxsize=1)` — single probe per process, manually cleared in tests

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Aligned parse_runner_value with pre-existing test contract**
- **Found during:** Task 2 (image_registry + runner_resolution)
- **Issue:** `runner_resolution.py` and `test_runner_resolution.py` were already on branch (from docs/phase-17-context work). The existing tests expected `parse_runner_value` to raise `ValueError` on empty image and unrecognised values. Initial implementation silently returned fallback values.
- **Fix:** Made `parse_runner_value` strict — raises `ValueError` with actionable messages for both cases. Updated `test_image_registry.py` to match the contract.
- **Files modified:** `src/llenergymeasure/infra/image_registry.py`, `tests/unit/test_image_registry.py`
- **Verification:** All 639 unit tests pass
- **Committed in:** `5d4f812` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 2 — missing critical validation contract alignment)
**Impact on plan:** Necessary for correctness — `parse_runner_value` is a public API and silent fallbacks would cause downstream confusion. No scope creep.

## Issues Encountered

- Pre-existing `runner_resolution.py`, `test_runner_resolution.py`, and several modified config files were already on the `docs/phase-17-context` branch. These are companion runner infrastructure changes that belong with this plan. All included in Task 2 commit after verifying they pass tests.

## Next Phase Readiness

- DockerError hierarchy ready for DockerRunner (Plan 02) to use in `translate_docker_error()` calls
- `image_registry.get_default_image()` and `parse_runner_value()` ready for DockerRunner dispatch
- `runner_resolution.resolve_study_runners()` ready for StudyRunner integration (Plan 04)
- Container entrypoint ready for integration testing once Docker infrastructure is in place (Phase 18)

---
*Phase: 17-docker-runner-infrastructure*
*Completed: 2026-02-28*
