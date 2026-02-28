---
phase: 17-docker-runner-infrastructure
plan: "02"
subsystem: infra
tags: [docker, runner-resolution, config, user-config, study-config]

# Dependency graph
requires:
  - phase: 17-01
    provides: parse_runner_value() from image_registry; DockerError hierarchy; RunnerSpec dataclass

provides:
  - resolve_runner() with full 5-layer precedence chain (env > yaml > user_config > auto_detect > default)
  - is_docker_available() host-level Docker + NVIDIA CT detection via shutil.which
  - resolve_study_runners() multi-backend runner resolution
  - StudyConfig.runners field for per-backend runner configuration from study YAML
  - UserRunnersConfig accepts bare "docker" (not just "docker:<image>")
  - "runners" key treated as study-only — not passed to ExperimentConfig
  - 31 unit tests covering all precedence layers and edge cases

affects:
  - 17-03-PLAN (DockerRunner dispatch — consumes resolve_runner() to determine mode)
  - 17-04-PLAN (StudyRunner integration — consumes resolve_study_runners())

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Runner resolution: user_config=None signals auto-detection; user_config=provided (even with 'local') blocks auto-detection"
    - "parse_runner_value re-exported from image_registry via runner_resolution for consumer convenience"
    - "StudyConfig.runners is metadata — not part of experiment config hash (per CONTEXT.md)"

key-files:
  created:
    - src/llenergymeasure/infra/runner_resolution.py
    - tests/unit/test_runner_resolution.py
  modified:
    - src/llenergymeasure/config/models.py (StudyConfig.runners field)
    - src/llenergymeasure/config/user_config.py (UserRunnersConfig accepts bare 'docker')
    - src/llenergymeasure/config/loader.py (extract and pass runners field)
    - src/llenergymeasure/study/grid.py (add 'runners' to _STUDY_ONLY_KEYS)

key-decisions:
  - "user_config=None enables auto-detection; user_config provided (any value, even 'local') blocks auto-detection — explicit presence beats inference"
  - "parse_runner_value imported from image_registry (canonical home), re-exported from runner_resolution for test convenience"
  - "UserRunnersConfig now accepts bare 'docker' in addition to 'docker:<image>' — aligns with CONTEXT.md runner config syntax"
  - "'runners' added to _STUDY_ONLY_KEYS so it is not propagated to individual ExperimentConfig objects"

patterns-established:
  - "Runner resolution test pattern: mock is_docker_available at source module path; use monkeypatch for env vars"
  - "user_config=None as 'not set' sentinel vs user_config=UserRunnersConfig() as 'explicitly set' distinction"

requirements-completed: [DOCK-06]

# Metrics
duration: 5min
completed: 2026-02-28
---

# Phase 17 Plan 02: Runner Resolution Summary

**resolve_runner() implementing full 5-layer precedence chain (env > YAML > user config > auto-detection > local fallback) with Docker-first default when NVIDIA Container Toolkit detected**

## Performance

- **Duration:** 5 min
- **Started:** 2026-02-28T02:28:34Z
- **Completed:** 2026-02-28T02:33:00Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- `resolve_runner()` implements full precedence chain: env var wins, then YAML `runners:` section, then user config, then auto-detection (Docker + NVIDIA CT on PATH), then local fallback with nudge
- `is_docker_available()` checks for `docker` CLI and any of `nvidia-container-runtime`, `nvidia-ctk`, or `nvidia-container-cli` on PATH — a lightweight host-level check (container GPU validation is Phase 18)
- Docker-first default when available: auto-detected docker prevents measurement environment drift without any user configuration
- `user_config=None` semantics: passing None means "no user config file present" → enables auto-detection. Passing a UserRunnersConfig object (even with default values) is treated as explicit user intent
- `StudyConfig.runners` field added and wired through loader; `"runners"` added to `_STUDY_ONLY_KEYS` so it is not propagated to individual `ExperimentConfig` objects
- `UserRunnersConfig` updated to accept bare `"docker"` (not just `"docker:<image>"`), aligning with CONTEXT.md runner config syntax
- 31 unit tests covering all 5 precedence layers plus `is_docker_available` and `resolve_study_runners`

## Task Commits

All deliverables for Plans 01 and 02 were built together in the Plan 01 session and landed in commit `5d4f812`. This plan's SUMMARY documents the Plan 02 scope already present in that commit.

1. **Task 1: Runner resolution module with precedence chain** - `5d4f812` (feat)
2. **Task 2: Add runners section to study YAML parsing** - `5d4f812` (feat)

## Files Created/Modified

- `src/llenergymeasure/infra/runner_resolution.py` — `RunnerSpec`, `is_docker_available`, `resolve_runner`, `resolve_study_runners`; re-exports `parse_runner_value` from image_registry
- `src/llenergymeasure/config/models.py` — `StudyConfig.runners: dict[str, str] | None` field
- `src/llenergymeasure/config/user_config.py` — `UserRunnersConfig` validator updated to accept bare `"docker"`; field descriptions updated
- `src/llenergymeasure/config/loader.py` — Extract `runners` from raw YAML and pass to `StudyConfig`
- `src/llenergymeasure/study/grid.py` — `"runners"` added to `_STUDY_ONLY_KEYS`
- `tests/unit/test_runner_resolution.py` — 31 tests for all precedence layers

## Decisions Made

- `user_config=None` as the "auto-detection enabled" sentinel — when the caller has no user config file, they pass None. When they have a UserRunnersConfig (even factory-default), they pass the object. This gives unambiguous control over auto-detection without needing a separate "was this user-set?" flag.
- Bare `"docker"` accepted in `UserRunnersConfig` to match the study YAML syntax — researchers should be able to write `docker` in both the YAML `runners:` section and their `~/.config/llenergymeasure/config.yaml` file.

## Deviations from Plan

None — plan executed as specified. All deliverables were pre-implemented in Plan 01's session and verified passing (639 unit tests).

## Issues Encountered

The Plan 02 deliverables were implemented in the Plan 01 session (commit `5d4f812`) alongside the Plan 01 artifacts. This is expected for wave-1 plans executed together. All tests pass; no rework required.

## Next Phase Readiness

- `resolve_runner()` ready for consumption by DockerRunner (Plan 03) — call `resolve_runner(backend)` to get `RunnerSpec`, then dispatch based on `spec.mode`
- `StudyConfig.runners` passes the study-level runner dict into the resolution chain
- `is_docker_available()` available for Plan 03 to guard Docker dispatch
- When `spec.mode == "docker"` and `spec.image is None`, Plan 03 should call `get_default_image(backend)` from `image_registry` to resolve the built-in image

---
*Phase: 17-docker-runner-infrastructure*
*Completed: 2026-02-28*
