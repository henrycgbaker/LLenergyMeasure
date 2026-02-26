# Unified API vs Split API: Empirical Peer Evidence

**Researched:** 2026-02-25
**Domain:** Python library API surface design for "one vs many" execution patterns
**Confidence:** HIGH — based on actual source code from 10 peer tools + official Python typing guidance
**Motivation:** Resolve contradiction between `experiment-study-architecture.md` (Q3: unified `run()`) and `designs/library-api.md` (split `run_experiment()` / `run_study()`)

---

## Summary

The overwhelming industry pattern is: **always return the same type, regardless of input count.** No major Python ML/benchmarking tool uses a union return type that varies based on the number of items processed. The consistent approach is either (a) always return the collection type (which degenerates naturally to a single-element collection), or (b) provide separate functions with distinct return types.

The Python typing community explicitly advises against union return types. Guido van Rossum, the mypy documentation, and the official Python typing best practices all state that union return types force `isinstance()` checks on callers and degrade developer experience.

**Primary recommendation:** Use **Option 3 (always return `StudyResult`)** for the internal runner, combined with **two thin public convenience functions** (`run_experiment()` / `run_study()`) that provide unambiguous return types. This matches the pattern used by Ray Tune, scikit-learn, lm-eval, and optimum-benchmark, while avoiding the DX problems of union return types.

---

## The Three Options Under Evaluation

### Option 1 — Unified `run()` with union return

```python
def run(config) -> ExperimentResult | StudyResult:
    # Single -> unwraps to ExperimentResult
    # Multi -> returns StudyResult
```

### Option 2 — Two separate functions

```python
def run_experiment(config) -> ExperimentResult: ...
def run_study(config) -> StudyResult: ...
```

### Option 3 — Always return one type

```python
def run(config) -> StudyResult:
    # Single experiment -> StudyResult with len(experiments) == 1
```

---

## Peer Tool Evidence

### 1. pytest

**Pattern:** Always returns ONE type regardless of test count.

`pytest.main()` returns `int | ExitCode` — the same type whether running 1 test or 10,000 tests. The `Session` object wraps all test `Item`s uniformly. A single-test session is structurally identical to a multi-test session.

```python
# Source: docs.pytest.org/en/latest/_modules/_pytest/main.html
def wrap_session(config, doit) -> int | ExitCode:
    session = Session.from_config(config)
    session.exitstatus = ExitCode.OK
    # ... runs all tests ...
    return session.exitstatus
```

**Verdict:** Single item is a degenerate collection. Same API, same return type. No union based on item count.

### 2. Optuna

**Pattern:** `optimize()` returns `None`. Results accessed via study object properties. Single trial = `n_trials=1`.

```python
# Source: github.com/optuna/optuna — study.py
def optimize(
    self,
    func: ObjectiveFuncType,
    n_trials: int | None = None,
    ...
) -> None:
```

A single trial is indeed just `study.optimize(objective, n_trials=1)`. The `Study` object is always the container — there is no separate "run one trial" function that returns a different type. Access is via `study.best_trial` (returns `FrozenTrial`) and `study.trials` (returns `list[FrozenTrial]`).

**Verdict:** Always the collection type (Study). Single trial = degenerate study. No union return. No unwrapping.

### 3. Ray Tune

**Pattern:** `Tuner.fit()` ALWAYS returns `ResultGrid` — even for a single trial.

```python
# Source: github.com/ray-project/ray — tuner.py
def fit(self) -> ResultGrid:
    ...
```

`ResultGrid` always contains a list of `Result` objects internally. A single trial = `ResultGrid` with one element. The class even has an optimisation for single-result grids in `get_best_result()` — but it still returns a `ResultGrid`, not a bare `Result`.

```python
# Source: github.com/ray-project/ray — result_grid.py
class ResultGrid:
    _results: list[Result]  # Always a list, even for single trial

    def __len__(self) -> int:
        return len(self._results)

    def __getitem__(self, i: int) -> Result:
        return self._results[i]
```

**Verdict:** Always the collection type. Single trial = `ResultGrid` with `len() == 1`. No union return. No unwrapping.

### 4. lm-eval (EleutherAI)

**Pattern:** `simple_evaluate()` returns `EvalResults | None` — the SAME type regardless of number of tasks.

```python
# Source: github.com/EleutherAI/lm-evaluation-harness — evaluator.py
def simple_evaluate(
    model: str | LM,
    tasks: list[str | dict[str, Any] | Task] | None = None,
    ...
) -> EvalResults | None:
```

The `| None` is for distributed computing (non-rank-0 processes return `None`), NOT for single-vs-multi task distinction. Whether you pass `tasks=["hellaswag"]` or `tasks=["hellaswag", "arc_easy", "mmlu"]`, the return type is the same `EvalResults` dict with one or many keys.

**Verdict:** Same return type regardless of task count. The collection structure scales naturally. No union based on item count.

### 5. MLflow

**Pattern:** Separate APIs for different concepts. `mlflow.start_run()` returns `ActiveRun`. `mlflow.search_runs()` returns `DataFrame`. No unified function.

MLflow treats "Run" and "Experiment" as separate organisational concepts — an Experiment is a container for Runs, like a folder. The API never returns `Run | Experiment` from a single function. Instead:
- `mlflow.start_run()` -> `ActiveRun` (always)
- `mlflow.search_runs()` -> `pd.DataFrame` (always)
- `mlflow.get_experiment()` -> `Experiment` (always)

**Verdict:** Separate functions, separate return types. Never a union. The Run/Experiment hierarchy is navigated through distinct API calls, not through a single polymorphic function.

### 6. optimum-benchmark (Hugging Face)

**Pattern:** `Benchmark.run()` and `Benchmark.launch()` BOTH return `BenchmarkReport`.

```python
# Source: github.com/huggingface/optimum-benchmark — base.py
def run(config) -> BenchmarkReport:
    ...

def launch(config) -> BenchmarkReport:
    ...
```

A single benchmark returns a `BenchmarkReport`. A sweep (via `--multirun` which delegates to Hydra) also produces `BenchmarkReport` objects — one per run. The report structure is uniform.

**Verdict:** Same return type for single and sweep. Uses Hydra's multirun for sweeps, which runs the same function multiple times rather than returning a different type.

### 7. W&B (Weights & Biases)

**Pattern:** Completely separate APIs for single runs vs sweeps. No unified function.

- `wandb.init()` -> `Run` (single run tracking)
- `wandb.sweep()` -> `str` (sweep ID)
- `wandb.agent()` -> `None` (runs sweep, calls init() internally)

W&B never attempts to unify these. A sweep is a fundamentally different orchestration mode that invokes multiple `wandb.init()` calls. The return types are completely different because the operations are different.

**Verdict:** Separate APIs, separate types. A sweep is NOT a degenerate run, and they never pretend otherwise.

### 8. scikit-learn

**Pattern:** `GridSearchCV.fit()` ALWAYS returns `self` — regardless of parameter grid size.

```python
# Source: github.com/scikit-learn/scikit-learn — _search.py
def fit(self, X, y=None, **params):
    ...
    return self
```

The `cv_results_` dict always has the same structure whether you test 1 parameter combination or 1,000. A single-parameter grid is just `ParameterGrid` with one entry. The API is completely uniform.

`cross_val_score()` similarly always returns `np.ndarray`, regardless of fold count.

**Verdict:** Always the same return type. Single item = collection with one element. No union.

### 9. Hydra

**Pattern:** The Compose API does NOT support multirun. There is an **open feature request** (issue #2673, opened 2023, still open) proposing either:
- (a) Add a `mode` parameter to `compose()` (would create union return)
- (b) Create a separate `compose_multirun()` function (avoids union)

The Hydra team has not implemented either, but the proposed Option (b) — a separate function — is notable.

`hydra_zen.launch()` (third-party) handles this with a `multirun: bool` parameter:
- `multirun=False` -> returns job output
- `multirun=True` -> returns list of job outputs

This IS effectively a union return, and it is the only example found in any tool studied.

**Verdict:** Hydra avoids the problem entirely (no multirun in compose API). The community-proposed solution leans toward separate functions.

### 10. Zeus

**Pattern:** `ZeusMonitor.end_window()` ALWAYS returns `Measurement`.

```python
# Source: github.com/ml-energy/zeus — energy.py
@dataclass
class Measurement:
    time: float
    gpu_energy: dict[int, float]
    cpu_energy: dict[int, float] | None = None
    dram_energy: dict[int, float] | None = None

class ZeusMonitor:
    def end_window(self, key: str, ...) -> Measurement:
        ...
```

Zeus has NO higher-level API that wraps multiple measurement windows into a study. Each `begin_window()` / `end_window()` pair is independent. Aggregation of multiple windows is the caller's responsibility.

**Verdict:** Single-level API only. Always returns `Measurement`. No collection concept at this layer.

---

## Summary Table: How Peers Handle "One vs Many"

| Tool | Single item uses same API as many? | Return type changes based on count? | Union return type? | Pattern |
|------|-------------------------------------|--------------------------------------|---------------------|---------|
| **pytest** | Yes | No | No | Always same type |
| **Optuna** | Yes (n_trials=1) | No | No | Always Study (collection) |
| **Ray Tune** | Yes | No | No | Always ResultGrid (collection) |
| **lm-eval** | Yes | No | No | Always EvalResults (dict) |
| **MLflow** | No (separate APIs) | N/A | No | Separate functions |
| **optimum-benchmark** | Yes | No | No | Always BenchmarkReport |
| **W&B** | No (separate APIs) | N/A | No | Separate functions |
| **scikit-learn** | Yes | No | No | Always same type |
| **Hydra** | N/A (compose API lacks multirun) | N/A | No | Unsolved |
| **Zeus** | N/A (single-level only) | N/A | No | No collection concept |

**Zero out of ten tools use a union return type that varies based on input count.**

---

## Python Typing Community Guidance on Union Returns

### Official Python Typing Best Practices

Source: [typing.python.org/en/latest/reference/best_practices.html](https://typing.python.org/en/latest/reference/best_practices.html)

> "Avoid union return types, since they require `isinstance()` checks."

This is unambiguous. The official typing documentation explicitly advises against union return types.

### Guido van Rossum (mypy issue #1693)

Source: [github.com/python/mypy/issues/1693](https://github.com/python/mypy/issues/1693)

> "Every caller will have to use `isinstance()` to pick the union apart before its value can be used, right? And that's often unnecessary because there's some reason why the runtime value is always one or the other."

### mypy Documentation

Source: [mypy.readthedocs.io/en/stable/kinds_of_types.html](https://mypy.readthedocs.io/en/stable/kinds_of_types.html)

> "It is recommended to avoid union types as function return types, since the caller may have to use `isinstance()` before doing anything interesting with the value."

### The `@overload` Alternative

The typing community recommends `@overload` when the return type can be determined from the input type at static analysis time:

```python
@overload
def run(config: ExperimentConfig) -> ExperimentResult: ...
@overload
def run(config: StudyConfig) -> StudyResult: ...
@overload
def run(config: str | Path) -> ExperimentResult | StudyResult: ...  # Can't resolve statically!

def run(config):
    ...
```

**Critical limitation:** `@overload` works when the input TYPE determines the output type. But with `llem.run("config.yaml")`, the type checker cannot determine from a `str` whether the YAML contains an experiment config or a study config. The overload for `str | Path` would STILL need a union return type — the very thing we are trying to avoid.

This means `@overload` does NOT fully solve the problem for the unified `run()` approach when the input is a file path.

---

## Analysis of the Three Options

### Option 1 — Unified `run()` with union return

```python
def run(config) -> ExperimentResult | StudyResult:
```

**Evidence against:**
- Zero peer tools use this pattern
- Official Python typing guidance explicitly recommends against union returns
- `@overload` cannot resolve the union for `str | Path` inputs (the most common usage)
- Every caller must `isinstance()` check: `if isinstance(result, ExperimentResult): ...`
- IDE autocompletion shows the union of all fields, not the specific type's fields
- mypy/pyright produce wider inferred types, leading to cascading type-narrowing requirements

**Sole argument for:**
- Fewer functions to remember

**Verdict: REJECT.** Contradicts typing best practices, has no peer precedent, and degrades DX.

### Option 2 — Two separate functions

```python
def run_experiment(config: str | Path | ExperimentConfig) -> ExperimentResult: ...
def run_study(config: str | Path | StudyConfig) -> StudyResult: ...
```

**Evidence for:**
- MLflow and W&B use separate APIs for single/collection concepts
- Return types are unambiguous — no `isinstance()` needed
- `@overload` works perfectly within each function (vary by config type)
- The existing `designs/library-api.md` already documents this approach in detail
- Clear naming convention: function name tells you what you get back

**Evidence against:**
- Requires the user to know whether their YAML is an experiment or study
- For `str | Path` inputs, the function must validate that the YAML matches expectations
- Two functions where one would suffice (if the return type problem were solved)

**Design tension with Option C architecture:**
Under Option C, a single experiment is internally `StudyConfig(experiments=[config])`. If the internal architecture treats everything as a study, why expose two public functions? Answer: because the **return type** matters to the caller, and two functions give unambiguous types.

### Option 3 — Always return `StudyResult`

```python
def run(config) -> StudyResult:
    # Single experiment -> StudyResult with len(experiments) == 1
```

**Evidence for:**
- This is exactly what Ray Tune, Optuna, scikit-learn, and pytest do
- Matches the internal architecture (Option C: everything is a study)
- No union return type
- One function, one return type, fully type-safe
- Callers always know what they get: `result.experiments[0]` for single

**Evidence against:**
- Callers running single experiments must write `result.experiments[0]` — mildly verbose
- "I just want to measure one model" → getting a StudyResult feels heavyweight
- Breaks the ergonomic ideal of `result = llem.run(model="X"); print(result.energy_total_j)`

**But this can be mitigated:**
If `StudyResult` provides convenience accessors that delegate to the single experiment when `len(experiments) == 1`:

```python
class StudyResult:
    experiments: list[ExperimentResult]

    @property
    def energy_total_j(self) -> float:
        if len(self.experiments) != 1:
            raise ValueError("Use .experiments[i].energy_total_j for multi-experiment studies")
        return self.experiments[0].energy_total_j
```

This is the pattern used by pandas (`DataFrame` with one row still returns `DataFrame` from most operations, but `.item()` or `.squeeze()` extracts the scalar).

---

## Recommended Architecture: Hybrid of Options 2 + 3

Based on the evidence, the cleanest architecture is:

### Internal layer: Option 3 (always `StudyResult`)

```python
# Internal — not public API
def _run(study: StudyConfig) -> StudyResult:
    """Single runner. Always returns StudyResult."""
    ...
```

This matches Option C's architecture perfectly. One runner, one return type, no branching.

### Public layer: Option 2 (two functions with unambiguous return types)

```python
# Public API — exported from __init__.py
def run_experiment(
    config: str | Path | ExperimentConfig | None = None,
    **kwargs,
) -> ExperimentResult:
    """Run a single experiment. Always returns ExperimentResult."""
    study_config = _to_study_config(config, **kwargs)
    assert len(study_config.experiments) == 1
    study_result = _run(study_config)
    return study_result.experiments[0]

def run_study(
    config: str | Path | StudyConfig,
    *,
    output_dir: Path | None = _UNSET,
) -> StudyResult:
    """Run a study (one or more experiments). Always returns StudyResult."""
    study_config = _to_study_config(config)
    return _run(study_config)
```

This gives:
- **Internally:** Always `StudyResult` (Option 3, matches Option C architecture)
- **Publicly:** Unambiguous return types (Option 2, matches typing best practices)
- **No union return types anywhere in the public API**
- **`@overload` works correctly** within each function for config type variants

### Why this is better than a unified `run()`

| Concern | Unified `run()` | Split functions |
|---------|-----------------|-----------------|
| Return type clarity | Union — caller must check | Unambiguous — no checking needed |
| IDE autocompletion | Shows union of all fields | Shows exact type's fields |
| mypy/pyright | Wider inferred types, cascading narrowing | Exact types, no narrowing needed |
| `str | Path` input | Cannot resolve return type statically | Each function knows its return type |
| User mental model | "What will I get back?" | Function name tells you |
| Peer precedent | Zero tools do this | MLflow, W&B, existing library-api.md |
| Python typing guidance | Explicitly advised against | Recommended approach |

---

## Addressing the `experiment-study-architecture.md` Decision

The Q3 decision in `experiment-study-architecture.md` states:

> **Q3. Library API:** `llem.run(config) -> ExperimentResult | StudyResult`. Single function.

This directly contradicts:
1. The existing `designs/library-api.md` (which documents `run_experiment()` / `run_study()`)
2. All peer tool evidence (no tool uses union return based on item count)
3. Official Python typing guidance (avoid union returns)

**Recommendation:** Update Q3 to align with the evidence. The internal architecture (Option C: everything is a StudyConfig internally) is correct and should remain. But the public API surface should be two functions with unambiguous return types, as documented in `library-api.md`.

The Q1 architecture decision (single runner `_run(StudyConfig)`) is sound and unchanged. Only Q3 (public API surface) needs revision.

---

## Liskov Substitution Angle

The question was raised: if `StudyResult` contains `ExperimentResult`, should the API always return `StudyResult`?

**Analysis:** `StudyResult` is NOT a supertype of `ExperimentResult` in the Liskov sense. They are sibling types with different semantics:
- `ExperimentResult` has `.energy_total_j`, `.tokens_per_second` (direct metrics)
- `StudyResult` has `.experiments: list[ExperimentResult]`, `.summary()` (aggregate metadata)

They do not share an interface. You cannot substitute one for the other. Liskov substitution does not apply here because there is no inheritance relationship and no shared protocol.

If we WANTED to make Liskov apply, we would need:
```python
class Result(Protocol):
    """Common protocol for all results."""
    def to_json(self, path: Path) -> None: ...
    def to_parquet(self, path: Path) -> None: ...

class ExperimentResult(Result): ...
class StudyResult(Result): ...
```

But even then, the caller needs to know which type they have to access the actual data (energy, throughput, etc.). The protocol only covers serialisation, not data access.

**Verdict:** Liskov substitution does not provide a reason to always return `StudyResult`. The two types have fundamentally different access patterns. Returning the specific type the caller expects is the correct design.

---

## Real-World DX with Union Return Types in ML Libraries

### pandas: The cautionary tale

`pandas.DataFrame.squeeze()` returns `DataFrame | Series | scalar` depending on the shape. This is widely regarded as one of the worst API decisions in pandas — it makes type checking nearly impossible and forces runtime checks everywhere.

### numpy: Intentionally avoids it

`np.mean()` with `keepdims=True` always returns `ndarray`. Without `keepdims`, it returns `ndarray | scalar` — and this is considered a DX problem that `keepdims` was specifically introduced to solve.

### The pattern that works

Tools like Ray Tune, scikit-learn, and Optuna demonstrate that **always returning the collection type** (even for single items) is the accepted ML ecosystem convention. Users adapt quickly to `result.experiments[0]` or `results[0]`, and the type safety benefits compound across the codebase.

---

## Confidence Assessment

| Finding | Confidence | Basis |
|---------|------------|-------|
| No peer tool uses union return based on count | HIGH | Verified source code of 10 tools |
| Python typing community advises against union returns | HIGH | Official docs + Guido + mypy team |
| `@overload` cannot resolve `str | Path` to specific return type | HIGH | Typing spec, verified |
| Two public functions is better DX than unified `run()` | HIGH | Typing guidance + peer evidence + existing library-api.md |
| Internal `_run(StudyConfig) -> StudyResult` is correct | HIGH | Matches Option C architecture + 4 peer tools |
| Liskov substitution does not apply to ExperimentResult/StudyResult | HIGH | Types have different interfaces |

---

## Sources

### Primary (HIGH confidence)
- [pytest source: `_pytest/main.py`](https://docs.pytest.org/en/latest/_modules/_pytest/main.html)
- [Optuna source: `study.py`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html)
- [Ray Tune source: `tuner.py`, `result_grid.py`](https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.fit.html)
- [lm-eval source: `evaluator.py`](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/evaluator.py)
- [scikit-learn source: `_search.py`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
- [optimum-benchmark source](https://github.com/huggingface/optimum-benchmark)
- [Zeus source: `energy.py`](https://github.com/ml-energy/zeus)
- [Python typing best practices](https://typing.python.org/en/latest/reference/best_practices.html)
- [Python overload spec](https://typing.python.org/en/latest/spec/overload.html)
- [mypy issue #1693: union return types](https://github.com/python/mypy/issues/1693)

### Secondary (MEDIUM confidence)
- [MLflow API docs](https://mlflow.org/docs/latest/python_api/mlflow.html)
- [W&B sweep walkthrough](https://docs.wandb.ai/models/sweeps/walkthrough)
- [Hydra compose API issue #2673](https://github.com/facebookresearch/hydra/issues/2673)
- [Python typing issue #566: AnyOf](https://github.com/python/typing/issues/566)
- [Adam Johnson: @overload patterns](https://adamj.eu/tech/2021/05/29/python-type-hints-how-to-use-overload/)
