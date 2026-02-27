# Phase 3: Library API - Context

**Gathered:** 2026-02-26
**Status:** Ready for planning
**Source:** Product design docs (.product/decisions/, .product/designs/)

<domain>
## Phase Boundary

This phase delivers the stable public API surface for `llenergymeasure`. After this phase:
- `from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult` resolves
- `run_experiment(config)` returns `ExperimentResult` (no union types, no None)
- `__init__.py.__all__` is the stability contract; everything else is internal
- `__version__ == "2.0.0"` (already set in Phase 1)

**Not in scope:** `run_study()` full implementation (M2), sweep grammar, Docker execution,
study manifest persistence. Only the stub `StudyConfig` for CFG-17 degenerate case.

</domain>

<decisions>
## Implementation Decisions

### Public API Surface (LA-07, LA-08)
- `__init__.py` exports exactly: `run_experiment`, `run_study`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult`, `__version__`
- `run_study` and `StudyResult` exported now (stubs) so the surface is complete from day one
- Everything NOT in `__init__.py.__all__` is internal — no SemVer guarantee (LA-08)

### run_experiment() — Three Overloaded Forms (LA-01)
- Form 1: `run_experiment(config: str | Path)` — YAML path
- Form 2: `run_experiment(config: ExperimentConfig)` — Pydantic object
- Form 3: `run_experiment(model="X", backend="Y", **kwargs)` — kwargs convenience
- Uses `@overload` for type-checker resolution
- Returns exactly `ExperimentResult` — no union types (LA-06)

### Side-Effect Free (LA-04)
- `run_experiment()` with no `output_dir` produces no disk writes
- Result persistence is explicit: `result.to_json()`, `result.to_parquet()`
- The CLI writes to disk; the library does not

### Internal Runner Pattern — CFG-17
- Internally, `_run(StudyConfig) -> StudyResult` always (LA-03, Option C architecture)
- `run_experiment()` wraps input into `StudyConfig(experiments=[config])`, runs, unwraps `result.experiments[0]`
- Single run = degenerate StudyConfig (CFG-17)
- For M1, single experiment runs **in-process** — no subprocess (STU-05)

### Stub StudyConfig (CFG-17 dependency)
- Create minimal `StudyConfig` in `config/models.py`: `experiments: list[ExperimentConfig]`, `name: str | None`
- No `ExecutionConfig`, no sweep grammar — those are M2 (CFG-11 through CFG-16)
- Export from `__init__.py` as part of the stable surface

### run_study() — Stub for M1 (surface completeness)
- Exported from `__init__.py` but raises `NotImplementedError("Study execution is available in M2")`
- This ensures the import surface is stable from v2.0.0 — no new exports needed in M2

### ExperimentResult Rename
- Rename `AggregatedResult` → `ExperimentResult` in `domain/experiment.py`
- Keep `AggregatedResult` as a compatibility alias (same pattern as exceptions.py v1.x aliases)
- Phase 3 owns the rename so the public API uses the correct name from day one

### StudyResult Stub
- Create minimal `StudyResult` in `domain/results.py` (or `experiment.py`): `experiments: list[ExperimentResult]`, `name: str | None`
- Exported from `__init__.py`; fleshed out in M2 (RES-13 through RES-15)

### _api.py Module
- New file: `src/llenergymeasure/_api.py`
- Contains `run_experiment()` and `run_study()` implementations
- Imported into `__init__.py` — the module itself is internal (prefixed with `_`)
- `_to_study_config()` helper converts all input forms to a `StudyConfig`

### Deprecation Contract (LA-09)
- One minor version deprecation window before removing any `__all__` export
- Document this in module docstring

### Stability Contract Enforcement
- Any name NOT in `__init__.py.__all__` must raise `AttributeError` on direct `from llenergymeasure import X`
- Test this explicitly (success criterion #4)

### Claude's Discretion
- Internal structure of `_api.py` (helper functions, error handling flow)
- Exact implementation of config-to-StudyConfig conversion for kwargs form
- Whether `StudyResult` goes in `domain/experiment.py` or a new `domain/study.py`
- Test file organisation (one file or split by concern)
- Whether `_run()` lives in `_api.py` or a separate internal module

</decisions>

<specifics>
## Specific Implementation References

### __init__.py Target Shape (from designs/architecture.md)
```python
from llenergymeasure._api import run_experiment, run_study
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.results import ExperimentResult, StudyResult

__version__: str = "2.0.0"

__all__ = [
    "run_experiment",
    "run_study",
    "ExperimentConfig",
    "StudyConfig",
    "ExperimentResult",
    "StudyResult",
    "__version__",
]
```

### run_experiment Type Signature (from designs/library-api.md)
```python
@overload
def run_experiment(config: str | Path) -> ExperimentResult: ...
@overload
def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...
@overload
def run_experiment(
    model: str,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "alpaca",
    **kwargs,
) -> ExperimentResult: ...
```

### Error Handling (from decisions/error-handling.md)
- `ConfigError` for YAML parse errors, file not found
- `Pydantic ValidationError` passes through unchanged
- `BackendError` for backend not installed
- All error messages must be instructive (what went wrong + how to fix)

### Existing Code State
- `ExperimentConfig` exists in `config/models.py` (Phase 2 complete)
- `load_experiment_config()` exists in `config/loader.py`
- Exception hierarchy exists in `exceptions.py`
- `AggregatedResult` exists in `domain/experiment.py` (needs rename)
- No `_api.py`, no `StudyConfig`, no `StudyResult` yet

</specifics>

<deferred>
## Deferred Ideas

- `run_study()` full implementation — M2
- `StudyConfig` with `ExecutionConfig` and sweep grammar — M2
- `StudyResult` with `study_design_hash`, `measurement_protocol`, `result_files` — M2
- `StudyManifest` checkpoint pattern — M2
- Docker execution and multi-backend — M3

</deferred>

---

*Phase: 03-library-api*
*Context gathered: 2026-02-26 from .product/ design documents*
