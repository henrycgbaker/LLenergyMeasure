---
phase: 08-testing-and-integration
plan: 01
subsystem: testing
tags: [pytest, fakes, protocol-injection, test-infrastructure, unit-tests]

# Dependency graph
requires:
  - phase: 07-cli
    provides: "Completed v2.0 CLI and all source code"
  - phase: 06-results-schema-and-persistence
    provides: "ExperimentResult with measurement_config_hash/measurement_methodology fields"
provides:
  - "Clean test suite: 258 GPU-free v2.0 unit tests, 0 collection errors"
  - "tests/fakes.py: FakeInferenceBackend, FakeEnergyBackend, FakeResultsRepository"
  - "tests/conftest.py: make_config() and make_result() factories + shared fixtures"
  - "pyproject.toml: 'gpu' mark registered, addopts excludes gpu tests by default"
affects:
  - "08-02 (unit test writing uses fakes/factories)"
  - "08-03 (CI workflows run the clean test suite)"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Protocol injection fakes: explicit class-body behaviour, no MagicMock"
    - "Factory functions in conftest.py: override only what you care about"
    - "Two-tier test structure: unit/ (GPU-free) + integration/ (@pytest.mark.gpu)"

key-files:
  created:
    - tests/fakes.py
  modified:
    - tests/conftest.py
    - tests/CLAUDE.md
    - tests/unit/test_api.py
    - tests/unit/test_backend_protocol.py
    - tests/unit/test_warmup_v2.py
    - src/llenergymeasure/core/backends/pytorch.py
    - pyproject.toml

key-decisions:
  - "start_time and end_time are required fields on ExperimentResult — make_result() factory includes them"
  - "load_in_4bit/load_in_8bit tests updated to assert BitsAndBytesConfig wrapper (v2.0 API), not raw kwargs (v1.x legacy)"
  - "_build_result() in pytorch.py fixed with compute_measurement_config_hash(config) and measurement_methodology='total'"

patterns-established:
  - "Protocol injection fakes in tests/fakes.py: implement protocol structurally, record calls, raise explicitly on unconfigured state"
  - "Factory pattern: make_config(**overrides) and make_result(**overrides) in conftest.py — all required fields included in defaults"
  - "GPU tests excluded by default via addopts = '-v --tb=short -m \"not gpu\"'"

requirements-completed: [INF-09, INF-10]

# Metrics
duration: 4min
completed: 2026-02-27
---

# Phase 8 Plan 01: Test Infrastructure Summary

**Deleted ~90 v1.x test files, created protocol injection fakes and factory functions, fixed 16 failing v2.0 tests: pytest tests/unit/ exits 0 with 258 passing**

## Performance

- **Duration:** 4 min
- **Started:** 2026-02-27T00:35:09Z
- **Completed:** 2026-02-27T00:40:07Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Deleted all v1.x test artefacts: 5 subdirectories, ~55 unit test files, 7 integration test files, e2e/, runtime/, fixtures/, configs/ directories
- Created tests/fakes.py with FakeInferenceBackend, FakeEnergyBackend, FakeResultsRepository (protocol injection, no MagicMock)
- Rewrote tests/conftest.py with make_config()/make_result() factories (all required ExperimentResult fields included)
- Fixed all 16 failing v2.0 tests across test_api.py, test_backend_protocol.py, test_warmup_v2.py, and source fix in pytorch.py

## Task Commits

1. **Task 1: Delete v1.x test files and directories, register pytest gpu mark** - `2c83a5e` (chore)
2. **Task 2: Create shared test infrastructure and fix all 16 failing v2.0 tests** - `78b4b09` (feat)

## Files Created/Modified

- `tests/fakes.py` - Protocol injection fakes: FakeInferenceBackend, FakeEnergyBackend, FakeResultsRepository
- `tests/conftest.py` - Rewritten with make_config(), make_result() factories and sample_config/sample_result/tmp_results_dir fixtures
- `tests/CLAUDE.md` - Updated to v2.0 two-tier test structure (removed all v1.x references)
- `tests/unit/test_api.py` - Added measurement_config_hash, measurement_methodology, start_time, end_time to _make_experiment_result()
- `tests/unit/test_backend_protocol.py` - Updated load_in_4bit/load_in_8bit tests to assert BitsAndBytesConfig instead of raw kwargs
- `tests/unit/test_warmup_v2.py` - Added missing required fields to two ExperimentResult constructions
- `src/llenergymeasure/core/backends/pytorch.py` - Added measurement_config_hash and measurement_methodology to _build_result() return
- `pyproject.toml` - Added gpu marker and -m "not gpu" to addopts

## Decisions Made

- `start_time` and `end_time` are required fields on `ExperimentResult` — make_result() factory includes them with fixed epoch values
- `load_in_4bit`/`load_in_8bit` tests corrected to assert `BitsAndBytesConfig` wrapper (v2.0 modern HuggingFace API), not raw kwargs (v1.x legacy pattern recorded in STATE.md)
- Fixed `_build_result()` in pytorch.py as a Rule 1 bug fix — source code was omitting required fields introduced in Phase 6

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] start_time and end_time missing from make_result() defaults**
- **Found during:** Task 2 (creating conftest.py and running tests)
- **Issue:** ExperimentResult requires start_time and end_time (required fields), but the plan's factory template omitted them — would cause ValidationError on every make_result() call
- **Fix:** Added `start_time` and `end_time` with fixed epoch values to make_result() defaults; also fixed _make_experiment_result() in test_api.py and the ExperimentResult constructions in test_warmup_v2.py
- **Files modified:** tests/conftest.py, tests/unit/test_api.py, tests/unit/test_warmup_v2.py
- **Verification:** pytest tests/unit/ reports 258 passed
- **Committed in:** 78b4b09 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — missing required fields in factory)
**Impact on plan:** Necessary for correctness. The plan's make_result() template was incomplete; all fields correctly identified in the plan's context but omitted from the factory template body.

## Issues Encountered

None — both tasks executed cleanly. The 16 failing tests fell exactly into the two categories described in the plan: missing required ExperimentResult fields (14 tests) and stale load_in_4bit/load_in_8bit assertions (2 tests).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 258 GPU-free unit tests pass with 0 failures and 0 collection errors
- tests/fakes.py and tests/conftest.py ready for use in Plan 02 (unit test expansion)
- pytest gpu mark registered, integration/ directory clean (only __init__.py) — ready for Plan 03 CI wiring
- No blockers

---
*Phase: 08-testing-and-integration*
*Completed: 2026-02-27*
