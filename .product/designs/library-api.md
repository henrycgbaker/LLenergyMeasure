# Library API Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/architecture.md](../decisions/architecture.md) (sub-decision H),
                     [../decisions/experiment-study-architecture.md](../decisions/experiment-study-architecture.md) (Q3)
**Status**: Confirmed (session 3, naming updated session 4, API shape confirmed 2026-02-25)

---

## Core Principle: Side-Effect Free (with one documented exception)

`run_experiment()` is **side-effect free** — it returns data, it does not write to disk.
The CLI writes to disk as part of its job; the library does not.

```python
result = llem.run_experiment("config.yaml")
result.to_json("results/my_run.json")    # explicit — user decides
result.to_parquet("results/my_run.parquet")  # explicit
```

**`run_study()` is the exception — DECIDED 2026-02-19:**
`run_study()` writes a `study_manifest.json` checkpoint during execution for study recovery.
This is the only library function that writes to disk, and it is intentional: long-running
studies (hours to days) need checkpointing for resume-on-failure. The same need exists for
library users as for CLI users — moving manifest writing to the CLI layer would leave library
users without recovery capability.

- Default: writes to `{output_dir}/{study_name}/study_manifest.json`
- `output_dir` defaults to user config's `output.results_dir`; `./results/` if not configured
- Pass `output_dir=None` explicitly to suppress all disk writes (side-effect-free mode)
- The manifest lives **alongside result files**, not in a separate dotdir (MLflow pattern)

**Why not a dotdir (`.llem_checkpoints/`)?** Checkpoint data is study metadata, not a hidden
system cache. MLflow writes `meta.yaml` alongside run artifacts for the same reason: the state
and the results travel together. Dotdirs imply "tool internals, not user data" — that's wrong
here.

**Industry comparison:**
- lm-eval `simple_evaluate()` — no disk writes (single short evaluation, no need)
- Optuna `create_study()` — writes to SQLite by default (long-running optimisation)
- Ray Tune — writes to `~/ray_results/<trial>/` by default (long-running HPO)
- MLflow — writes `mlruns/<exp_id>/meta.yaml` alongside run artifacts
- Zeus `ZeusMonitor.end_window()` — returns a value; no disk write (single measurement)

llem `run_study()` is closer to Optuna/Ray Tune in character (long-running, needs recovery)
than to Zeus (instantaneous measurement). The exception is principled, not a compromise.

**Peer reference:** lm-eval `simple_evaluate()` returns a dict, no disk writes.
Zeus `ZeusMonitor.end_window()` returns a measurement object, no disk writes.
CodeCarbon `tracker.stop()` returns a value — CSV write is opt-in.

---

## Public API Surface (`llenergymeasure/__init__.py`)

Everything exported from `__init__.py` is **stable API** — SemVer-guaranteed from v2.0.
Everything not in `__init__.py` is internal and may change without notice.

```python
# llenergymeasure/__init__.py

# Primary entry points
from llenergymeasure._api import run_experiment, run_study

# Config types
from llenergymeasure.config import ExperimentConfig, StudyConfig

# Result types (for type annotations in user code)
from llenergymeasure.domain import ExperimentResult, StudyResult

# Version
__version__: str
```

**Why CLI names differ from library names:**
The CLI uses unified `llem run` (YAML determines scope). The library uses
`run_experiment` / `run_study` (Python verb-phrase convention, unambiguous return types).
No peer tool mirrors CLI command names in its Python API — lm-eval, MLflow, W&B, Zeus all
use different naming in CLI vs library. They serve different users with different conventions.

---

## `run_experiment` — Run a Single Experiment

Overloaded on argument type. All three call forms are equivalent — same function, different
input:

```python
import llenergymeasure as llem

# Form 1: kwargs — zero-config convenience (notebooks, quick scripts)
result = llem.run_experiment(
    model="meta-llama/Llama-3.1-8B",
    backend="pytorch",    # optional if only one backend installed
    n=100,                # optional — default: 100
    dataset="alpaca",     # optional — default: "alpaca"
)

# Form 2: path — YAML-driven reproducible experiment
result = llem.run_experiment("configs/my_experiment.yaml")

# Form 3: config object — programmatic, fully type-checked
config = ExperimentConfig(model="meta-llama/Llama-3.1-8B", backend="pytorch", n=100)
result = llem.run_experiment(config)

# ExperimentConfig.from_yaml() for the config-object form of path loading
config = ExperimentConfig.from_yaml("configs/my_experiment.yaml")
result = llem.run_experiment(config)
```

Type signature using `@overload`:

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

**`backend` is optional** when exactly one backend is installed — matches CLI zero-config
behaviour. Raises `ValueError` (not `TypeError`) if multiple backends are installed and
`backend` is not specified.

---

## `run_study` — Run a Study

```python
# Form 1: path (most common) — writes manifest to user config's results_dir by default
results = llem.run_study("study.yaml")

# Form 2: explicit output_dir
results = llem.run_study("study.yaml", output_dir=Path("./my-results/"))

# Form 3: config object
study = StudyConfig.from_yaml("study.yaml")
results = llem.run_study(study)

# Form 4: suppress disk writes (side-effect-free mode)
results = llem.run_study("study.yaml", output_dir=None)
```

Type signature:

```python
_UNSET = object()  # private sentinel distinguishes "not passed" from "explicitly None"

@overload
def run_study(study: str | Path, *, output_dir: Path | None = _UNSET) -> StudyResult: ...
@overload
def run_study(study: StudyConfig, *, output_dir: Path | None = _UNSET) -> StudyResult: ...
```

`output_dir` semantics:
- `_UNSET` (default, not passed): reads from user config's `output.results_dir`; `./results/` if not configured. Writes manifest.
- `Path(...)`: writes manifest to this directory. Individual result files also land here.
- `None`: no manifest written, no disk writes at all (purely side-effect-free).

Manifest path when writing: `{output_dir}/{study.name}/study_manifest.json`
Result files: `{output_dir}/{study.name}/{model_slug}_{backend}_{timestamp}.json`

Returns a `StudyResult` — see [result-schema.md](result-schema.md) for the full schema.
`StudyResult.result_files` contains paths to individual `ExperimentResult` JSON files.

**Why not `run("config.yaml")`?**
A single `run()` has an ambiguous return type — the type checker cannot statically determine
whether to return `ExperimentResult` or `StudyResult` from a path string. This forces
`isinstance` checks everywhere in user code and degrades IDE autocompletion.
`run_experiment` / `run_study` give unambiguous types with no inference overhead.

---

## `ExperimentConfig` — Experiment Definition

```python
from llenergymeasure import ExperimentConfig

# From kwargs — Pydantic validates on construction
config = ExperimentConfig(
    model="meta-llama/Llama-3.1-8B",
    backend="pytorch",
    n=100,
    dataset="alpaca",
    batch_size=1,
    precision="bf16",
)

# From YAML
config = ExperimentConfig.from_yaml("experiment.yaml")

# Validation is eager — raises ValidationError on construction if config is invalid.
# Backend × precision constraints and cross-field constraints (e.g. speculative_decoding
# requires draft_model) are enforced here, before any GPU time is spent.
```

---

## `StudyConfig` — Study Definition

```python
from llenergymeasure import StudyConfig

study = StudyConfig.from_yaml("study.yaml")

# After parsing and grid expansion:
# study.experiments  -> list[ExperimentConfig]  (all valid combinations)
# study.skipped      -> list[dict]               (L1 invalid configs with reasons)
# study.runner       -> "local" | "docker" | "auto"
```

Grid expansion and Pydantic constraint validation both run at `from_yaml()` time — before any
experiment starts. Invalid combinations are enumerated upfront and logged.

---

## `ExperimentResult` — Working with Results

```python
result: ExperimentResult = llem.run_experiment("config.yaml")

# Human-readable summary
print(result.summary())

# Direct metric access
print(result.energy_total_j)
print(result.tokens_per_second)
print(result.flops_total)
print(result.measurement_methodology)   # "total" | "steady_state" | "windowed"
print(result.measurement_config_hash)   # 16-char SHA-256 for reproducibility checking (environment snapshot not in hash)

# Persistence — explicit; library does not write to disk by default
result.to_json("results/my_run.json")
result.to_parquet("results/my_run.parquet")

# Load from file
result = ExperimentResult.from_json("results/my_run.json")
```

---

## Internal: Composable Components

For advanced use — testing, custom backends, research extensions. Importable from
submodules but **not exported from `__init__.py`**. No SemVer stability guarantee.

```python
from llenergymeasure.core import EnergyBackend, InferenceBackend
from llenergymeasure.orchestration import ExperimentOrchestrator

orchestrator = ExperimentOrchestrator(
    inference=MyCustomInferenceBackend(),
    energy=ZeusBackend(),
    config=config,
)
result = orchestrator.run()
```

`EnergyBackend` and `InferenceBackend` are **Protocol classes** (structural subtyping) —
custom backends only need to implement the protocol methods, not inherit from a base class.

**Peer reference:** lm-eval exposes `LM` as an internal Protocol; users implement custom
models by matching the protocol without inheriting. MLflow `MlflowClient` is the composable
layer beneath the fluent API.

---

## Usage Patterns by User Type

| User | Entry point | When |
|---|---|---|
| Notebook / quick script | `llem.run_experiment(model="X", ...)` | Exploratory, one-off |
| Reproducible experiment | `llem.run_experiment("config.yaml")` | Published research |
| Sweep / study | `llem.run_study("study.yaml")` | Parameter sweeps |
| Library integrator | `run_experiment(ExperimentConfig(...))` | Building on top of llem |
| Custom backend / testing | `ExperimentOrchestrator(inference=..., energy=...)` | Extension / testing |

---

## Peer Comparison

| Pattern | lm-eval | MLflow | Zeus | LLenergyMeasure |
|---|---|---|---|---|
| Top-level convenience | `simple_evaluate(model, tasks)` | `mlflow.log_metric()` | `ZeusMonitor()` | `run_experiment(model=...)` |
| Config-object API | `Evaluator` + `Task` objects | `MlflowClient` | `ZeusDataLoader` | `run_experiment(ExperimentConfig)` |
| Composable internals | `LM` Protocol | `MlflowClient` | `Measurement` objects | `ExperimentOrchestrator` |
| Writes to disk by default | ✗ | ✓ (tracking server) | ✗ | ✗ (`run_experiment`); ✓ (`run_study` manifest — opt-out via `output_dir=None`) |
| CLI name ≠ Python name | ✓ | ✓ | n/a | ✓ |
| Overloaded entry point | ✓ (`tasks: list[str\|Task]`) | ✗ | ✗ | ✓ (`str\|Path\|Config`) |
