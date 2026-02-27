---
phase: 03-library-api
plan: 02
subsystem: api
tags: [public-api, library-interface, __init__, overloads, unit-tests]

requires:
  - phase: 03-library-api
    plan: 01
    provides: ExperimentResult, StudyResult, StudyConfig type contracts

provides:
  - run_experiment() public function (three call forms) in _api.py
  - run_study() stub (NotImplementedError, M2 message) in _api.py
  - _to_study_config() internal converter in _api.py
  - _run() internal stub (Phase 4 implementation hook) in _api.py
  - __init__.py rewritten as stable public API surface with __all__ and SemVer contract
  - 12 unit tests in tests/unit/test_api.py (no GPU required)

affects:
  - Any consumer doing `from llenergymeasure import ...` now gets the v2.0 surface
  - Phase 4 (PyTorch backend) replaces _run() body only

tech-stack:
  added: []
  patterns:
    - "@overload stubs for three-form public function: all three grouped together before implementation"
    - "Internal module pattern: _api.py (underscore prefix) imported only via __init__.py"
    - "Degenerate StudyConfig pattern: single-experiment run wraps ExperimentConfig in StudyConfig(experiments=[...])"
    - "Monkeypatching _run() for GPU-free unit tests"

key-files:
  created:
    - src/llenergymeasure/_api.py
    - tests/unit/test_api.py
  modified:
    - src/llenergymeasure/__init__.py

key-decisions:
  - "_to_study_config kwargs form omits backend=None to let Pydantic default ('pytorch') apply — avoids _detect_default_backend() in M1"
  - "_run() raises NotImplementedError in M1 — Phase 4 replaces the body; tests monkeypatch it"
  - "__version__ in __all__ — required by LA-10, enables `from llenergymeasure import __version__`"
  - "Internal names (load_experiment_config, ConfigError, AggregatedResult) NOT in __init__ — accessible via their module paths only"

metrics:
  duration: 3min
  completed: 2026-02-26
  tasks: 2
  files_modified: 3
  tests_added: 12
---

# Phase 3 Plan 02: Library API - Public API Surface Summary

**`_api.py` with three-form `run_experiment()`, stub `run_study()`, and `__init__.py` as the stable v2.0 public surface gated by `__all__` with SemVer contract.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-02-26T17:01:50Z
- **Completed:** 2026-02-26T17:04:49Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- Created `_api.py` with `run_experiment()` (three call forms via `@overload`), `run_study()` (NotImplementedError stub), `_to_study_config()` (input normalisation), `_run()` (Phase 4 hook)
- Rewrote `__init__.py` with `__all__` (7 names), stability contract docstring, SemVer guarantee
- `from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult, __version__` all resolve
- `llenergymeasure.__version__ == "2.0.0"` and `"__version__"` is in `__all__`
- Internal names (`load_experiment_config`, `ConfigError`, `AggregatedResult`) raise `AttributeError` on module access
- 12 unit tests pass — all without GPU hardware, using monkeypatching of `_run()`

## Task Commits

1. **Task 1: Create _api.py** - `87934a5` (feat)
2. **Task 2: Wire __init__.py + write tests** - `f6b28bf` (feat)

## Files Created/Modified

- `src/llenergymeasure/_api.py` (new) — `run_experiment()`, `run_study()`, `_to_study_config()`, `_run()`
- `src/llenergymeasure/__init__.py` (rewrite) — `__all__`, stability docstring, `__version__`
- `tests/unit/test_api.py` (new) — 12 tests covering all Phase 3 success criteria

## Decisions Made

- `_to_study_config` omits `backend=None` when constructing `ExperimentConfig` — lets Pydantic default (`"pytorch"`) apply, avoids needing `_detect_default_backend()` in M1
- `_run()` raises `NotImplementedError` — Phase 4 replaces the body; tests monkeypatch it for isolation
- `__version__` included in `__all__` — satisfies LA-10, `from llenergymeasure import __version__` works
- `__init__.py` contains nothing except the 7 public names + `__all__` + `__version__` + docstring — no internal leakage

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- `test_config_models.py`, `test_config_loader.py`, and `test_user_config.py` still fail at collection time due to pre-existing import errors (`BUILTIN_DATASETS`, `load_config`, `DockerConfig` missing). Confirmed pre-existing (documented in 03-01-SUMMARY.md). Not caused by this plan.

## Next Phase Readiness

- `from llenergymeasure import run_experiment` works correctly
- Phase 4 (PyTorch backend) only needs to implement `_run(StudyConfig) -> StudyResult`; no API changes required
- API contract is established and tested

---
*Phase: 03-library-api*
*Completed: 2026-02-26*
