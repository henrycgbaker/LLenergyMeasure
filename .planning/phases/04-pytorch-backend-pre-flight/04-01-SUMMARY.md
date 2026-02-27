---
phase: 04-pytorch-backend-pre-flight
plan: 01
subsystem: infra
tags: [preflight, environment, cuda, validation, pydantic, stdlib-logging]

# Dependency graph
requires:
  - phase: 03-library-api
    provides: ExperimentConfig, ExperimentResult domain models already in place
provides:
  - run_preflight() function that collects all failures into a single PreFlightError
  - EnvironmentSnapshot model with full CM-32 fields
  - detect_cuda_version_with_source() with 4-source fallback chain (CM-33)
  - collect_environment_snapshot() pre-inference capture function
  - ExperimentResult.environment_snapshot field (v2.0 schema)
  - 32 GPU-free unit tests covering all new code
affects:
  - 04-02 (PyTorch backend — calls run_preflight() before inference)
  - 04-03 and beyond (all backends share the same preflight and snapshot machinery)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Collect-all failures pattern: all checks run before raising a single error"
    - "Deferred imports: importlib.util.find_spec() before conditional torch import at module level"
    - "Multi-source fallback chain: torch → version.txt → nvcc → None for CUDA detection"
    - "Stdlib logging throughout: import logging; logger = logging.getLogger(__name__)"

key-files:
  created:
    - src/llenergymeasure/orchestration/preflight.py
    - tests/unit/test_preflight.py
    - tests/unit/test_environment_snapshot.py
  modified:
    - src/llenergymeasure/domain/environment.py
    - src/llenergymeasure/domain/experiment.py

key-decisions:
  - "Persistence mode off is a warning (not an error) — first-run latency issue, not a blocker"
  - "Network errors in model accessibility check are non-blocking (None return, not error string)"
  - "Module-level torch import forbidden in preflight — importlib.util.find_spec() first"
  - "collect_environment_snapshot() uses deferred imports for __version__ and core/environment to avoid circular imports"

patterns-established:
  - "Preflight check helper: _check_X() → bool or str | None, caller collects into failures list"
  - "GPU-free testing: all torch/pynvml/huggingface_hub access via monkeypatch, no direct imports in tests"

requirements-completed: [CM-29, CM-30, CM-31, CM-32, CM-33]

# Metrics
duration: 4min
completed: 2026-02-26
---

# Phase 4 Plan 01: Pre-flight Validation and EnvironmentSnapshot Summary

**Pre-flight validation module with collect-all PreFlightError pattern, EnvironmentSnapshot model with 4-source CUDA detection chain, and 32 GPU-free unit tests**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-26T18:19:16Z
- **Completed:** 2026-02-26T18:23:18Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- `run_preflight(config)` checks CUDA availability, backend installation, and model accessibility — all before GPU allocation — collecting every failure into a single `PreFlightError` with the exact error format from CONTEXT.md
- `EnvironmentSnapshot` model and `collect_environment_snapshot()` capture the complete software+hardware context (Python version, pip freeze, optional conda list, tool version, CUDA version with source) at experiment start
- `detect_cuda_version_with_source()` implements the `torch → version.txt → nvcc → None` fallback chain (CM-33)
- `ExperimentResult.environment_snapshot` field added (v2.0 schema, default None)
- 32 GPU-free unit tests: all torch, pynvml, and huggingface_hub access fully monkeypatched

## Task Commits

1. **Task 1: Create pre-flight module and EnvironmentSnapshot** — `efa8612` (feat)
2. **Task 2: GPU-free unit tests** — `5a76f29` (test)

## Files Created/Modified

- `src/llenergymeasure/orchestration/preflight.py` — `run_preflight()` with collect-all pattern; `_check_cuda_available()`, `_check_backend_installed()`, `_check_model_accessible()`, `_warn_if_persistence_mode_off()`
- `src/llenergymeasure/domain/environment.py` — `EnvironmentSnapshot` model, `collect_environment_snapshot()`, `detect_cuda_version_with_source()`, `_capture_pip_freeze()`, `_capture_conda_list()`
- `src/llenergymeasure/domain/experiment.py` — `environment_snapshot: EnvironmentSnapshot | None` field added to `ExperimentResult`
- `tests/unit/test_preflight.py` — 19 tests: collect-all pattern, each check type, error format, persistence mode warning, all internal helpers
- `tests/unit/test_environment_snapshot.py` — 13 tests: CUDA fallback chain, pip freeze, conda list, full snapshot construction

## Decisions Made

- Persistence mode off is a **warning** not an error — avoids blocking experiments on a non-critical configuration detail
- Network errors in HF Hub model check are **non-blocking** — returns `None` rather than an error string, as network timeouts should not prevent experiments on local or cached models
- `importlib.util.find_spec()` used before any conditional torch import — preflight.py has **no module-level torch import**
- `collect_environment_snapshot()` uses deferred imports (`from llenergymeasure.core.environment import ...` and `from llenergymeasure import __version__`) to avoid circular dependency

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Test for persistence mode warning fixed after initial run**

- **Found during:** Task 2 (test_preflight_persistence_mode_warning_not_blocking)
- **Issue:** Original test used `importlib.reload(preflight_module)` after monkeypatching the old module reference, but the checks-pass patches applied to the old object and the reload created a fresh module — causing `_check_cuda_available` to run unpatched and fail CUDA check
- **Fix:** Replaced reload approach with a direct monkeypatch of `_warn_if_persistence_mode_off` to call the real logger, avoiding module reload entirely
- **Files modified:** tests/unit/test_preflight.py
- **Verification:** All 32 tests pass
- **Committed in:** 5a76f29 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug in test implementation)
**Impact on plan:** Test logic fix only — no production code changes. No scope creep.

## Issues Encountered

None beyond the test fix documented above.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `run_preflight()` is ready to be called at the start of the PyTorch backend `_run()` implementation (Plan 02)
- `collect_environment_snapshot()` is ready to be called before model loading in the PyTorch runner
- All pre-flight checks are GPU-free and will work in CI

---

*Phase: 04-pytorch-backend-pre-flight*
*Completed: 2026-02-26*
