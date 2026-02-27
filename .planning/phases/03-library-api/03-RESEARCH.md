# Phase 3: Library API - Research

**Researched:** 2026-02-26
**Domain:** Python public API surface, `@overload` typing, module `__all__` stability contracts
**Confidence:** HIGH — all decisions are locked, codebase state confirmed by direct inspection

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Public API Surface (LA-07, LA-08)**
- `__init__.py` exports exactly: `run_experiment`, `run_study`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult`, `__version__`
- `run_study` and `StudyResult` exported now (stubs) so the surface is complete from day one
- Everything NOT in `__init__.py.__all__` is internal — no SemVer guarantee (LA-08)

**run_experiment() — Three Overloaded Forms (LA-01)**
- Form 1: `run_experiment(config: str | Path)` — YAML path
- Form 2: `run_experiment(config: ExperimentConfig)` — Pydantic object
- Form 3: `run_experiment(model="X", backend="Y", **kwargs)` — kwargs convenience
- Uses `@overload` for type-checker resolution
- Returns exactly `ExperimentResult` — no union types (LA-06)

**Side-Effect Free (LA-04)**
- `run_experiment()` with no `output_dir` produces no disk writes
- Result persistence is explicit: `result.to_json()`, `result.to_parquet()`
- The CLI writes to disk; the library does not

**Internal Runner Pattern — CFG-17**
- Internally, `_run(StudyConfig) -> StudyResult` always (LA-03, Option C architecture)
- `run_experiment()` wraps input into `StudyConfig(experiments=[config])`, runs, unwraps `result.experiments[0]`
- Single run = degenerate StudyConfig (CFG-17)
- For M1, single experiment runs **in-process** — no subprocess (STU-05)

**Stub StudyConfig (CFG-17 dependency)**
- Create minimal `StudyConfig` in `config/models.py`: `experiments: list[ExperimentConfig]`, `name: str | None`
- No `ExecutionConfig`, no sweep grammar — those are M2 (CFG-11 through CFG-16)
- Export from `__init__.py` as part of the stable surface

**run_study() — Stub for M1 (surface completeness)**
- Exported from `__init__.py` but raises `NotImplementedError("Study execution is available in M2")`
- This ensures the import surface is stable from v2.0.0 — no new exports needed in M2

**ExperimentResult Rename**
- Rename `AggregatedResult` → `ExperimentResult` in `domain/experiment.py`
- Keep `AggregatedResult` as a compatibility alias (same pattern as exceptions.py v1.x aliases)
- Phase 3 owns the rename so the public API uses the correct name from day one

**StudyResult Stub**
- Create minimal `StudyResult` in `domain/results.py` (or `experiment.py`): `experiments: list[ExperimentResult]`, `name: str | None`
- Exported from `__init__.py`; fleshed out in M2 (RES-13 through RES-15)

**_api.py Module**
- New file: `src/llenergymeasure/_api.py`
- Contains `run_experiment()` and `run_study()` implementations
- Imported into `__init__.py` — the module itself is internal (prefixed with `_`)
- `_to_study_config()` helper converts all input forms to a `StudyConfig`

**Deprecation Contract (LA-09)**
- One minor version deprecation window before removing any `__all__` export
- Document this in module docstring

**Stability Contract Enforcement**
- Any name NOT in `__init__.py.__all__` must raise `AttributeError` on direct `from llenergymeasure import X`
- Test this explicitly (success criterion #4)

### Claude's Discretion

- Internal structure of `_api.py` (helper functions, error handling flow)
- Exact implementation of config-to-StudyConfig conversion for kwargs form
- Whether `StudyResult` goes in `domain/experiment.py` or a new `domain/study.py`
- Test file organisation (one file or split by concern)
- Whether `_run()` lives in `_api.py` or a separate internal module

### Deferred Ideas (OUT OF SCOPE)

- `run_study()` full implementation — M2
- `StudyConfig` with `ExecutionConfig` and sweep grammar — M2
- `StudyResult` with `study_design_hash`, `measurement_protocol`, `result_files` — M2
- `StudyManifest` checkpoint pattern — M2
- Docker execution and multi-backend — M3
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| LA-01 | `run_experiment(config: str \| Path \| ExperimentConfig \| None, **kwargs) -> ExperimentResult` | `@overload` pattern covers all three call forms; `_to_study_config()` converts each |
| LA-03 | Internal `_run(StudyConfig) -> StudyResult` always. Both public functions wrap/unwrap. | Stub `_run()` raises `NotImplementedError` for M1 (no inference engine yet); returns `ExperimentResult` placeholder |
| LA-04 | `run_experiment()` side-effect-free when `output_dir` not specified | `_api.py` never touches disk; result object owns persistence via `to_json()` / `to_parquet()` |
| LA-06 | No union return types. Each function returns exactly one type. | `@overload` gives type-checkers exact return type; implementation always returns concrete type |
| LA-07 | `__init__.py` exports all 7 names | Direct import and re-export; `__all__` list drives AttributeError on unlisted names |
| LA-08 | Everything NOT in `__init__.py` is internal | Python `__all__` + `AttributeError` on direct sub-module import not in `__all__` |
| LA-09 | One minor version deprecation window before removing any `__all__` export | Docstring stability contract; compatibility alias pattern (matches `exceptions.py`) |
| LA-10 | `__version__: str = "2.0.0"` | Already set in current `__init__.py` — verify it's in `__all__` |
| CFG-17 | Single run = degenerate `StudyConfig(experiments=[config])` | Minimal `StudyConfig(experiments: list[ExperimentConfig], name: str \| None)` added to `config/models.py` |
</phase_requirements>

---

## Summary

Phase 3 is a **wiring and naming phase**, not an algorithmic one. The decisions are locked and the codebase is in a known state after Phase 2. The core work is: add `StudyConfig` + `StudyResult` stubs, rename `AggregatedResult` → `ExperimentResult`, create `_api.py` with `run_experiment()` / `run_study()`, and stitch it all together in `__init__.py`.

The biggest technical subtlety is the **three-form `run_experiment()` overload** — Python's `@overload` mechanism requires careful implementation to satisfy both type-checkers (mypy) and runtime dispatch. The kwargs form (`run_experiment(model="X")`) must detect that no positional `config` was passed and treat all kwargs as `ExperimentConfig` field values. This requires a sentinel default or `None` first argument strategy.

The second consideration is the **`_run()` stub for M1**. In M1, there is no real inference engine yet — `_run(StudyConfig)` cannot actually produce measurements. The stub must raise `NotImplementedError` or return a placeholder `StudyResult` that satisfies the type signature. Since Phase 4 (Core Measurement) implements the real engine, `_run()` in M1 should raise `NotImplementedError("Core measurement not yet implemented")` so that test code can mock it and the public API surface is fully testable at import/call time without GPU hardware.

**Primary recommendation:** Implement the three `@overload` stubs + runtime dispatch function (`_to_study_config`) in `_api.py`. Keep `_run()` as a stub that raises `NotImplementedError`. All success criteria are achievable without GPU hardware.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `typing.overload` | stdlib | Type-checker dispatch for multiple call signatures | Standard Python — no alternative |
| `pathlib.Path` | stdlib | Path input normalisation | Project convention (CLAUDE.md) |
| `pydantic` | >=2.0 (already dep) | `StudyConfig`, `StudyResult` as `BaseModel` | Consistent with all other models |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `typing_extensions` | via pydantic | `@overload` compatibility on older Python | Already transitive dep |
| `pytest` | >=8.0 (already dev dep) | Unit tests for API surface | All test targets |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `@overload` | `functools.singledispatch` | `singledispatch` cannot handle kwargs-only form; `@overload` is correct for mixed positional/kwargs dispatch |
| `__all__` for stability | Module `__init__` guards | `__all__` is the standard Python mechanism; no custom guards needed |

---

## Architecture Patterns

### Recommended Module Layout (after Phase 3)

```
src/llenergymeasure/
├── __init__.py               # PUBLIC SURFACE — all 7 exports + stability docstring
├── _api.py                   # run_experiment(), run_study(), _to_study_config(), _run()
├── config/
│   └── models.py             # ExperimentConfig + NEW: StudyConfig
├── domain/
│   ├── experiment.py         # AggregatedResult → ExperimentResult (alias kept)
│   └── results.py  [NEW]     # StudyResult stub (or added to experiment.py)
└── exceptions.py             # unchanged
```

The choice between `domain/results.py` (new file) vs adding to `domain/experiment.py` is **Claude's discretion**. Recommendation: add `StudyResult` to `domain/experiment.py` for now — M2 will likely reorganise when `StudyResult` gains its full schema. Creating a new file for a 3-field stub adds friction without benefit.

### Pattern 1: `@overload` Dispatch for `run_experiment()`

**What:** Python's `@overload` decorator provides type-checker resolution for multiple call signatures. The actual implementation uses a single un-decorated function that handles all forms at runtime.

**When to use:** When a function accepts multiple distinct argument patterns that result in the same return type but need different type-checker handling.

**Example:**
```python
# Source: Python typing docs (https://docs.python.org/3/library/typing.html#typing.overload)
from __future__ import annotations

from pathlib import Path
from typing import overload

from llenergymeasure.config.models import ExperimentConfig
from llenergymeasure.domain.experiment import ExperimentResult


@overload
def run_experiment(config: str | Path) -> ExperimentResult: ...
@overload
def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...
@overload
def run_experiment(
    config: None = None,
    *,
    model: str,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs,
) -> ExperimentResult: ...


def run_experiment(
    config: str | Path | ExperimentConfig | None = None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs,
) -> ExperimentResult:
    """Run a single experiment. Side-effect free — no disk writes without output_dir."""
    study = _to_study_config(config, model=model, backend=backend, n=n, dataset=dataset, **kwargs)
    study_result = _run(study)
    return study_result.experiments[0]
```

**Critical detail:** The third overload uses `config: None = None` (not omitting `config`) to allow mypy to correctly resolve the signature when `config` is absent. The `model` kwarg must be keyword-only (after `*`) to prevent positional ambiguity with `config`.

### Pattern 2: `_to_study_config()` Converter

**What:** Internal helper normalises all three input forms into a `StudyConfig`.

**Example:**
```python
def _to_study_config(
    config: str | Path | ExperimentConfig | None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs,
) -> StudyConfig:
    """Convert any run_experiment() input form to a degenerate StudyConfig."""
    if isinstance(config, ExperimentConfig):
        experiment = config
    elif isinstance(config, (str, Path)):
        from llenergymeasure.config.loader import load_experiment_config
        experiment = load_experiment_config(path=Path(config))
    elif config is None:
        if model is None:
            raise ConfigError(
                "run_experiment() requires either a config argument or model= keyword. "
                "Example: run_experiment(model='meta-llama/Llama-3.1-8B')"
            )
        experiment = ExperimentConfig(
            model=model,
            backend=backend or _detect_default_backend(),
            n=n,
            dataset=dataset,
            **kwargs,
        )
    else:
        raise ConfigError(f"Unexpected config type: {type(config).__name__}")

    return StudyConfig(experiments=[experiment])
```

### Pattern 3: `__init__.py` Stability Contract

**What:** `__init__.py` exports exactly the public surface. Anything not in `__all__` raises `AttributeError` when accessed via `from llenergymeasure import X`.

**Example:**
```python
"""LLenergyMeasure — LLM inference efficiency measurement framework.

Public API (stable from v2.0.0):
    run_experiment, run_study, ExperimentConfig, StudyConfig,
    ExperimentResult, StudyResult, __version__

Stability contract: exports in __all__ follow SemVer. Names not in __all__ are
internal and may change without notice. One minor version deprecation window
before removing any __all__ export (i.e., removed in v2.x+1 or v3.0 at earliest).
"""

from llenergymeasure._api import run_experiment, run_study
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult

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

**Note:** `__all__` controls `from module import *` but does NOT automatically raise `AttributeError` on `from llenergymeasure import SomeInternalName` — that still works unless the name is not actually imported into `__init__.py`. The success criterion says "raises AttributeError on direct import" — this is naturally satisfied by simply not importing internal names into `__init__.py`. No custom `__getattr__` needed.

### Pattern 4: Compatibility Alias

**What:** Keep `AggregatedResult = ExperimentResult` in `domain/experiment.py` so existing v1.x code does not break immediately. Same pattern already used in `exceptions.py`.

```python
# At bottom of domain/experiment.py — AFTER the ExperimentResult class definition
AggregatedResult = ExperimentResult  # v1.x compatibility alias — remove in v3.0
```

### Anti-Patterns to Avoid

- **Importing internal submodules in `__init__.py`:** Do not `from llenergymeasure.config import load_experiment_config` in `__init__.py` — only the 7 stable exports go there. Users who need `load_experiment_config` access the internal module directly (no SemVer guarantee).
- **Union return types:** `run_experiment() -> ExperimentResult | None` violates LA-06. The function always returns `ExperimentResult` or raises.
- **Putting `_run()` logic in `__init__.py`:** `__init__.py` is a surface, not an implementation file.
- **`@overload` on the implementation function:** Only the stubs get `@overload`. The actual implementation is undecorated and follows immediately after all stubs.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Type-checker dispatch | Custom `isinstance` dispatcher + Union types | `@overload` | Standard Python; IDEs understand it; type-checkers validate it |
| `__all__` enforcement | Custom `__getattr__` raising `AttributeError` | Just don't import internals into `__init__.py` | Python's normal name lookup handles it |
| Config file loading | Duplicate YAML loading in `_api.py` | Call existing `load_experiment_config()` from `config/loader.py` | Already implemented and tested in Phase 2 |

**Key insight:** Python's `@overload` + `__all__` gives the stability guarantees without custom machinery. The main implementation work is in the conversion logic and stub structure, not tooling.

---

## Common Pitfalls

### Pitfall 1: `@overload` Implementation Function Placement

**What goes wrong:** The runtime implementation function (no `@overload`) must come immediately after all `@overload`-decorated stubs, with the same name. If there is any non-`@overload` code between stubs and implementation, type-checkers may not resolve correctly.

**Why it happens:** `@overload` works by registering stubs; the last un-decorated definition with the same name is the runtime function. Code between stubs can confuse some type-checkers.

**How to avoid:** Group all `@overload` stubs together, then immediately define the implementation. No intervening code.

**Warning signs:** mypy reports "No overload variant of X matches argument type" when the call should match.

### Pitfall 2: `config: None = None` vs Omitting `config`

**What goes wrong:** If the third overload does not include `config` in its signature at all, mypy cannot resolve `run_experiment(model="X")` — it only sees the first two overloads, which both require a `config` positional argument.

**Why it happens:** Python overload resolution requires that each overload signature be complete. The implementation's signature `config: str | Path | ExperimentConfig | None = None` covers all forms, but the overload stubs are what type-checkers see.

**How to avoid:** The third overload stub must have `config: None = None` (with default) so type-checkers know it can be called without `config`. Confirmed pattern from Python docs (HIGH confidence).

**Warning signs:** `mypy` error: "Too few arguments for run_experiment" when calling `run_experiment(model="X")`.

### Pitfall 3: `StudyResult` Location Causing Circular Import

**What goes wrong:** `StudyResult` contains `list[ExperimentResult]`. If placed in a new file `domain/results.py` that imports from `domain/experiment.py`, and `domain/experiment.py` also imports from `domain/results.py`, circular import occurs.

**Why it happens:** Python's module import system executes module code linearly; circular imports fail at runtime.

**How to avoid:** Place `StudyResult` in `domain/experiment.py` alongside `ExperimentResult` (no circular import possible). If a separate file is preferred, use `TYPE_CHECKING` guards and string forward references.

**Warning signs:** `ImportError: cannot import name 'ExperimentResult' from partially initialised module`.

### Pitfall 4: `ExperimentConfig.model_rebuild()` After Adding `StudyConfig`

**What goes wrong:** `ExperimentConfig` uses forward references for `PyTorchConfig | None` etc., resolved by `_rebuild_experiment_config()` at import time. If `StudyConfig` is added to the same `models.py` file after `ExperimentConfig`, and it also uses forward references, those must also be rebuilt.

**Why it happens:** Pydantic v2 requires `model_rebuild()` for models with forward references that reference types defined after the model itself.

**How to avoid:** `StudyConfig` only references `ExperimentConfig` (already defined above it in the same file) — no forward references needed. This pitfall does not apply if `StudyConfig` is defined after `ExperimentConfig` in `models.py`.

**Warning signs:** `PydanticUserError: `model_rebuild()` must be called when models refer to each other`.

### Pitfall 5: `__version__` Not in `__all__`

**What goes wrong:** `llenergymeasure.__version__` is importable but if `__version__` is not in `__all__`, `from llenergymeasure import *` in user code does not include it.

**Why it happens:** `__all__` controls what `import *` exports. Not including `__version__` is a common omission.

**How to avoid:** Explicitly include `"__version__"` in `__all__`. Confirmed required by LA-07 (includes `__version__` in the named exports).

---

## Codebase State (Confirmed by Direct Inspection)

### What Exists

| Item | Location | State |
|------|----------|-------|
| `ExperimentConfig` | `config/models.py` | Complete, v2.0 schema, Phase 2 done |
| `load_experiment_config()` | `config/loader.py` | Complete, handles YAML/JSON, CLI overrides |
| `AggregatedResult` | `domain/experiment.py` | Exists — needs rename to `ExperimentResult` |
| `LLEMError` hierarchy | `exceptions.py` | Complete with v1.x aliases |
| `__version__ = "2.0.0"` | `__init__.py` | Set but `__all__` is missing entirely |
| Test infrastructure | `tests/unit/` | `pytest>=8.0`, `mypy>=1.0`, many existing test files |

### What Does Not Exist Yet

| Item | Needed For |
|------|------------|
| `StudyConfig` | CFG-17, `__init__.py` export |
| `ExperimentResult` name | LA-07, public API |
| `StudyResult` | LA-07, `__init__.py` export |
| `_api.py` | LA-01, LA-03, LA-04 |
| `__all__` in `__init__.py` | LA-07, LA-08 |
| `tests/unit/test_api.py` | Validation of success criteria |

### Import Chain (After Phase 3)

```
__init__.py
  ├── from ._api import run_experiment, run_study
  │     └── _api.py
  │           ├── from .config.models import ExperimentConfig, StudyConfig
  │           ├── from .config.loader import load_experiment_config
  │           └── from .domain.experiment import ExperimentResult, StudyResult
  ├── from .config.models import ExperimentConfig, StudyConfig
  └── from .domain.experiment import ExperimentResult, StudyResult
```

No circular imports — all dependencies flow in one direction.

---

## Code Examples

### Complete `_api.py` skeleton

```python
# Source: designs/library-api.md + decisions/experiment-study-architecture.md
"""Public API functions for llenergymeasure.

This module is internal (_api.py prefix). Import via llenergymeasure.__init__ only.
"""
from __future__ import annotations

from pathlib import Path
from typing import overload

from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult
from llenergymeasure.exceptions import ConfigError


@overload
def run_experiment(config: str | Path) -> ExperimentResult: ...
@overload
def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...
@overload
def run_experiment(
    config: None = None,
    *,
    model: str,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs,
) -> ExperimentResult: ...


def run_experiment(
    config: str | Path | ExperimentConfig | None = None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs,
) -> ExperimentResult:
    """Run a single LLM inference efficiency experiment.

    Side-effect free: no disk writes unless output_dir is specified.

    Three call forms:
      run_experiment("config.yaml")             # YAML path
      run_experiment(ExperimentConfig(...))      # config object
      run_experiment(model="X", backend="Y")    # kwargs convenience
    """
    study = _to_study_config(
        config, model=model, backend=backend, n=n, dataset=dataset, **kwargs
    )
    study_result = _run(study)
    return study_result.experiments[0]


def run_study(study: str | Path | StudyConfig) -> StudyResult:
    """Run a study (multiple experiments). Available in M2.

    Raises:
        NotImplementedError: Study execution is not yet available (M2).
    """
    raise NotImplementedError(
        "Study execution is available in M2. "
        "Use run_experiment() for single experiments."
    )


def _to_study_config(
    config: str | Path | ExperimentConfig | None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs,
) -> StudyConfig:
    """Convert any run_experiment() input form to a degenerate StudyConfig."""
    if isinstance(config, ExperimentConfig):
        experiment = config
    elif isinstance(config, (str, Path)):
        experiment = load_experiment_config(path=Path(config))
    elif config is None:
        if model is None:
            raise ConfigError(
                "run_experiment() requires either a config argument or model= keyword.\n"
                "Example: run_experiment(model='meta-llama/Llama-3.1-8B')"
            )
        experiment = ExperimentConfig(
            model=model,
            **({"backend": backend} if backend is not None else {}),
            n=n,
            dataset=dataset,
            **kwargs,
        )
    else:
        raise ConfigError(
            f"Expected str, Path, ExperimentConfig, or None; got {type(config).__name__}"
        )
    return StudyConfig(experiments=[experiment])


def _run(study: StudyConfig) -> StudyResult:
    """Internal runner — always receives StudyConfig, returns StudyResult.

    M1 stub: raises NotImplementedError. Phase 4 (Core Measurement) implements this.
    """
    raise NotImplementedError(
        "Core measurement engine not yet implemented (Phase 4). "
        "This stub exists to satisfy the type contract."
    )
```

### `StudyConfig` addition to `config/models.py`

```python
# Add at bottom of config/models.py, after ExperimentConfig
class StudyConfig(BaseModel):
    """Thin resolved container for a study (list of experiments + execution config).

    M1 stub: only experiments + name. ExecutionConfig and sweep grammar added in M2.
    """

    model_config = {"extra": "forbid"}

    experiments: list[ExperimentConfig] = Field(
        ..., min_length=1, description="Resolved list of experiments to run"
    )
    name: str | None = Field(
        default=None, description="Study name (used in output directory naming)"
    )
```

No `model_rebuild()` needed — `ExperimentConfig` is already defined above.

### `StudyResult` stub

```python
# Add to domain/experiment.py, after ExperimentResult
class StudyResult(BaseModel):
    """Container for study results.

    M1 stub: experiments list + name only. Full schema (study_design_hash,
    measurement_protocol, result_files, StudySummary) added in M2 (RES-13..15).
    """

    experiments: list[ExperimentResult] = Field(
        default_factory=list, description="Results for each experiment in the study"
    )
    name: str | None = Field(default=None, description="Study name")
```

### Success Criteria Test Patterns

```python
# tests/unit/test_api.py

def test_public_imports_resolve():
    from llenergymeasure import (
        run_experiment, run_study, ExperimentConfig, StudyConfig,
        ExperimentResult, StudyResult,
    )
    import llenergymeasure
    assert llenergymeasure.__version__ == "2.0.0"


def test_internal_name_raises_attribute_error():
    import llenergymeasure
    with pytest.raises(AttributeError):
        _ = llenergymeasure.load_experiment_config  # internal, not in __all__


def test_run_experiment_returns_experiment_result(monkeypatch):
    """run_experiment() return type is exactly ExperimentResult."""
    from llenergymeasure import run_experiment, ExperimentConfig, ExperimentResult
    from llenergymeasure import _api

    stub_result = StudyResult(experiments=[ExperimentResult(...)])
    monkeypatch.setattr(_api, "_run", lambda study: stub_result)

    result = run_experiment(ExperimentConfig(model="test-model"))
    assert isinstance(result, ExperimentResult)


def test_run_experiment_no_disk_writes(tmp_path, monkeypatch):
    """run_experiment() without output_dir writes nothing to disk."""
    # ... assert tmp_path is empty after call ...
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `AggregatedResult` | `ExperimentResult` | Phase 3 (this phase) | Public API uses correct domain name |
| No `__all__` in `__init__.py` | Explicit `__all__` with 7 stable exports | Phase 3 (this phase) | Stability contract established |
| `__init__.py` with only `__version__` | Full public API surface | Phase 3 (this phase) | Package is importable as a library |

---

## Open Questions

1. **Default backend detection for kwargs form**
   - What we know: `ExperimentConfig.backend` has default `"pytorch"`. `_to_study_config()` must handle `backend=None` in kwargs form.
   - What's unclear: Should `_detect_default_backend()` inspect installed packages (e.g., check if `torch` is importable), or should `ExperimentConfig`'s Pydantic default of `"pytorch"` simply be used as-is?
   - Recommendation: For M1 (PyTorch only milestone), omit `_detect_default_backend()` entirely. If `backend=None` is passed in the kwargs form, do not pass it to `ExperimentConfig` — let Pydantic default apply. This is simpler and correct for M1.

2. **`_run()` stub vs placeholder return**
   - What we know: Phase 4 implements the real `_run()`. M1 `_run()` must raise `NotImplementedError`.
   - What's unclear: Tests for `run_experiment()` need to mock `_run()`. Is `monkeypatch` sufficient or do tests need a Protocol-based injection point?
   - Recommendation: `monkeypatch.setattr(llenergymeasure._api, "_run", mock_fn)` is sufficient for unit tests. No injection point needed at this stage.

3. **`ExperimentResult` v2.0 field shape**
   - What we know: `AggregatedResult` (to be renamed) has many v1.x fields. The v2.0 `ExperimentResult` schema (RES-01..12) is defined in `designs/result-schema.md` but not yet implemented.
   - What's unclear: Should Phase 3 rename `AggregatedResult` → `ExperimentResult` in-place (keeping all v1.x fields), or should it also reshape the model to v2.0 schema?
   - Recommendation: Phase 3 renames only — do not reshape `ExperimentResult` fields. The full v2.0 schema implementation is Phase 6 (Results). Rename now so the public API name is correct; field changes follow when the schema phase runs.

---

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` — this section is skipped.

---

## Sources

### Primary (HIGH confidence)

- Direct codebase inspection — `src/llenergymeasure/__init__.py`, `config/models.py`, `domain/experiment.py`, `exceptions.py`, `config/loader.py`
- `.product/designs/library-api.md` — complete API design with peer comparison
- `.product/decisions/experiment-study-architecture.md` — Option C architecture decision (ADR)
- `.product/REQUIREMENTS.md` — authoritative LA-* and CFG-17 requirements
- `.planning/phases/03-library-api/03-CONTEXT.md` — locked decisions for this phase
- Python documentation on `@overload` — https://docs.python.org/3/library/typing.html#typing.overload

### Secondary (MEDIUM confidence)

- `tests/unit/test_exceptions.py` — confirmed test file naming and import patterns used in this project
- `src/llenergymeasure/config/__init__.py` — confirms `__all__` pattern already used in config subpackage

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new libraries; all patterns from Python stdlib + existing deps
- Architecture: HIGH — all decisions locked; codebase state confirmed by direct inspection
- Pitfalls: HIGH — `@overload` mechanics are stable Python stdlib; circular import risk confirmed by code inspection

**Research date:** 2026-02-26
**Valid until:** 2026-03-28 (stable Python stdlib patterns; no fast-moving dependencies)
