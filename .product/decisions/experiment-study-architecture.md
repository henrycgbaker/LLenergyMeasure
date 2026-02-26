# Experiment / Study Architecture

**Status**: Accepted
**Date decided**: 2026-02-25
**Last updated**: 2026-02-26 (Q5 superseded — always subdirectory per output-storage.md J1; Q3 revised 2026-02-25)
**Supersedes**: Decision D in [study-execution-model.md](study-execution-model.md)
**Prerequisite for**: Phase 5 implementation (ExperimentConfig refactor, runner design)

## Decision

**Option C accepted.** `ExperimentConfig` = pure data type (one measurement point). `StudyConfig` = thin resolved container (`list[ExperimentConfig]` + `ExecutionConfig`). Sweep resolution at YAML parse time — Pydantic models never see sweep grammar.

| Sub-decision | Resolution |
|---|---|
| **Q1. Architecture** | Option C — clean separation. Single runner `_run(StudyConfig)`. |
| **Q2. CLI** | Unified `llem run`. YAML determines scope. 2 commands + 1 flag. |
| **Q3. Library API** | `run_experiment(ExperimentConfig) -> ExperimentResult` + `run_study(StudyConfig) -> StudyResult`. Two functions, unambiguous return types. Internal: `_run(StudyConfig) -> StudyResult` always. |
| **Q4. Type names** | Keep `ExperimentConfig` + `StudyConfig`. |
| **Q5. Output** | **Superseded (2026-02-26):** Q5 output contract revised — see output-storage.md J1. Always subdirectory for both single and multi-experiment runs (`{name}_{timestamp}/` with `result.json` + `timeseries.parquet`). Flat JSON for single experiments is no longer correct. ~~Single → flat JSON. Multi → `{name}_{timestamp}/` subdirectory.~~ |

---

## Context

Decision D (study-execution-model.md, 2026-02-20) was written quickly and recorded "reject unification" as the outcome. On closer examination, this was premature. A third architectural option (Option C) was identified that is strictly better than both "keep two separate types" and "naive collapse."

The decision affects:
- Internal type hierarchy (ExperimentConfig, StudyConfig, or one unified type)
- Library API surface (`run_experiment()` / `run_study()` or unified `run()`)
- CLI command structure
- Runner code architecture (one code path vs two)
- Config schema complexity and maintenance burden going forward

---

## Peer Tool Survey

| Tool | Single unit | Collection | CLI pattern | Collection adds genuine complexity? |
|------|------------|-----------|-------------|-------------------------------------|
| **Optuna** | Trial | Study | API; `study.optimize(n_trials=N)` | Yes — optimizer, search algo, pruner |
| **Ray Tune** | Trial | Experiment | API; `Tuner.fit()` | Yes — scheduler, search, parallelism |
| **W&B** | Run | Sweep | Two commands: `wandb sweep` + `wandb agent` | Yes — search strategy, Bayesian/grid |
| **MLflow** | Run | Experiment | `mlflow run` (same cmd) | No — experiment is org namespace only |
| **lm-eval** | Task | Group | `lm-eval run` (same cmd) | No — group has no protocol, just labelling |
| **pytest** | Test | Suite | `pytest` (same cmd) | No — config file adds complexity, not a second command |
| **vLLM bench** | Benchmark run | Sweep | Two commands: `bench latency` / `bench sweep` | Yes — parameter grid, multiple runs |

Tools that use a single unified command (MLflow, lm-eval, pytest) do so because either (a) the collection adds no structural complexity, or (b) the config file carries the complexity signal, not the command name. With Option C's architecture, llem falls into category (b): the YAML file carries the signal.

---

## Considered Options

### Option A — Two Separate Types (Status Quo, rejected)

```python
class ExperimentConfig(BaseModel):
    model: str
    backend: Literal["pytorch", "vllm", "tensorrt"]
    precision: str = "bf16"
    n: int = 100
    # ... ~15 fields total ...

class StudyConfig(BaseModel):
    # ← RE-DECLARATION of ExperimentConfig fields as optional defaults
    model: str | None = None
    backend: str | list[str] | None = None
    precision: str | None = None
    n: int | None = None
    # ... ~15 fields re-declared ...

    # Study-specific additions
    sweep: dict[str, list[Any]] | None = None
    experiments: list[ExperimentConfig] | None = None
    execution: ExecutionConfig = ExecutionConfig()
```

Two CLI commands: `llem run` (single) and `llem study` (multi).
Two library functions: `run_experiment()` and `run_study()`.
Two runner code paths sharing underlying GPU measurement logic.

**Pros:**
- Two commands communicate scope clearly at the CLI level
- Users with simple needs never see sweep/execution concepts
- "experiment" and "study" map naturally to research methodology

**Cons:**
- ExperimentConfig fields are re-declared in StudyConfig — DRY violation that grows with every new feature. Adding a field to ExperimentConfig = two file edits, minimum.
- StudyConfig is structurally messy: it is simultaneously a "base defaults config" and a "sweep generator" and a "list container" — three jobs
- Two runner code paths that diverge over time or require a shared internal `_run_single()` helper anyway
- Boundary maintenance: as new params are added, someone must decide whether they go in ExperimentConfig only, StudyConfig only, or both

### Option B — Naive Collapse (One Big Type, rejected)

```python
class RunConfig(BaseModel):
    model: str
    backend: str | list[str]
    # ... ~15 fields ...

    # Optional study fields (absent = single experiment)
    sweep: dict[str, list[Any]] | None = None
    experiments: list[Self] | None = None
    execution: ExecutionConfig = ExecutionConfig()

    @property
    def is_study(self) -> bool:
        return bool(self.sweep or self.experiments)
```

**Pros:** Simpler surface area, no duplication, single code path.

**Cons:** `is_study` checks scattered through codebase; single-experiment YAML sees sweep/execution as dead weight; validation logic becomes conditional; loses clarity of the two-concept distinction.

### Option C — Clean Separation of Responsibilities (Chosen)

**Key insight**: The DRY violation in Option A comes from `StudyConfig` re-declaring `ExperimentConfig` fields as optional base defaults. This is unnecessary. The sweep grammar is a *YAML syntax feature*, not a *Pydantic model feature*. If sweep resolution happens at **parse time** (before the Pydantic model is constructed), `StudyConfig` never needs to hold a sweep dict at all.

```python
class ExperimentConfig(BaseModel):
    """One measurement point. Pure. Zero knowledge of studies."""
    model: str
    backend: Literal["pytorch", "vllm", "tensorrt"]
    precision: str = "bf16"
    n: int = 100
    # ... ~15 fields — unchanged, no study concepts ever touch this ...

class ExecutionConfig(BaseModel):
    n_cycles: int = 1
    cycle_order: str = "sequential"
    config_gap_seconds: int | None = None
    cycle_gap_seconds: int | None = None

class StudyConfig(BaseModel):
    """A structured investigation. Always a resolved flat list of experiments."""
    experiments: list[ExperimentConfig]   # sweep resolved BEFORE this is constructed
    execution: ExecutionConfig = ExecutionConfig()
    name: str | None = None
```

### Rejected Options

**Rejected (2026-02-25): Option A — Two separate types** — DRY violation grows with every new experiment parameter. StudyConfig has three conflicting jobs (base defaults, sweep generator, list container). Two runner code paths diverge over time. The maintenance burden compounds.

**Rejected (2026-02-25): Option B — Naive collapse** — `is_study` checks scattered through codebase. Single-experiment YAML sees sweep/execution as dead weight. Validation logic becomes conditional. Both concepts pollute each other.

> **Superseded (2026-02-25):** Decision D in study-execution-model.md (2026-02-20) rejected unification prematurely. Option C resolves the concerns that motivated that rejection (DRY violation, unclear responsibility) while preserving the clean conceptual separation between experiment and study.

---

## Decision

### Q1: Internal Architecture — Option C

`ExperimentConfig` is a pure data type describing one measurement point (~15 fields). It has zero knowledge of studies, sweeps, or execution protocol.

`StudyConfig` is a thin resolved container: `list[ExperimentConfig]` + `ExecutionConfig`. It never holds sweep grammar — that is resolved at YAML parse time.

The dependency runs one direction only:
```
ExperimentConfig  →  (nothing)
StudyConfig       →  ExperimentConfig (via list[ExperimentConfig])
```

The YAML loader resolves sweeps before constructing StudyConfig:
```
experiment.yaml  →  load_yaml()  →  ExperimentConfig
                                          ↓
                         StudyConfig(experiments=[config], execution=...)
                                          ↓
                                 single runner: _run(StudyConfig)

study.yaml  →  load_yaml()  →  resolve sweep → list[ExperimentConfig]
                                          ↓
                         StudyConfig(experiments=[...], execution=...)
                                          ↓
                                 single runner: _run(StudyConfig)
```

A single run is a natural degenerate case — `StudyConfig(experiments=[config])`. Same runner, same result pipeline, same measurement infrastructure.

### Q2: CLI — Unified `llem run`

One command handles both modes. The input determines the behaviour:

```bash
llem run --model meta-llama/Llama-3.1-8B                  # quick, CLI flags, defaults
llem run --model X --backend pytorch --precision fp16      # more specific, still flags
llem run experiment.yaml                                   # full YAML, sweeps, execution
```

The YAML file is the complexity signal — a second command name is redundant when the internal architecture is unified (Option C). This matches `pytest`, `lm-eval`, `mlflow run`.

CLI surface: **`llem run`** + **`llem config`** + **`llem --version`**. Two commands + one flag.

**Defaults by input mode:**
- `llem run --model X` (no YAML) → CLI effective default n_cycles=3, sensible for quick measurement
- `llem run experiment.yaml` → whatever the YAML specifies; Pydantic default n_cycles=1 if omitted

### Q3: Library API — Two functions, unambiguous return types

> **Superseded (2026-02-25):** The original Q3 specified a unified `llem.run()` with union
> return type `ExperimentResult | StudyResult`. Deep peer research (`.planning/research/UNIFIED-API-RESEARCH.md`)
> found that **zero out of ten peer tools** use a union return type based on input count, and
> official Python typing guidance (Guido, mypy docs) explicitly advises against union returns.
> Revised to two public functions with unambiguous return types.

```python
import llenergymeasure as llem

# Quick single measurement
result = llem.run_experiment(model="meta-llama/Llama-3.1-8B")

# From config object
result = llem.run_experiment(ExperimentConfig(model="X", backend="pytorch"))

# From YAML file
result = llem.run_experiment("experiment.yaml")

# Full study from YAML
result = llem.run_study("study.yaml")
```

Internally, all paths construct a `StudyConfig` and call `_run(StudyConfig) -> StudyResult`.
`run_experiment()` wraps input into a single-experiment `StudyConfig`, runs it, and unwraps
`StudyResult.experiments[0]` to return `ExperimentResult` directly.

**Why two functions, not one:**
- No peer tool uses a union return type that varies based on input count (0/10 studied)
- `@overload` cannot resolve `str | Path` inputs — the type checker can't know if a YAML file
  contains an experiment or study config from the filename alone
- `isinstance()` tax on every caller degrades DX and IDE autocompletion
- MLflow, W&B, and the existing `designs/library-api.md` all use separate functions
- See `.planning/research/UNIFIED-API-RESEARCH.md` for full evidence

### Q4: Config Type Names — Keep ExperimentConfig + StudyConfig

The internal type names stay as `ExperimentConfig` and `StudyConfig`. These are the correct domain terms. Users constructing configs programmatically see them; users writing YAML never see them.

### Q5: Output Contracts

> **Superseded (2026-02-26):** Q5 output contract revised — see output-storage.md J1. Always subdirectory for both single and multi-experiment runs (`{name}_{timestamp}/` with `result.json` + `timeseries.parquet`). Flat JSON for single experiments is no longer correct.

- Single experiment → flat JSON file: `results/{model}_{backend}_{timestamp}.json`
- Multi-experiment → subdirectory: `results/{name}_{timestamp}/` with per-experiment JSON files + study summary

The output format is determined by `len(study.experiments)`, not by the command name.

---

## Consequences

Positive:
- No DRY violation — `ExperimentConfig` fields declared once, automatically propagate to sweeps and studies
- `StudyConfig` is tiny and stable — it won't grow with new experiment features
- Single runner code path — no diverging implementations to maintain
- One CLI command to learn — the YAML file carries the complexity signal
- Library API has unambiguous return types: `run_experiment() -> ExperimentResult`, `run_study() -> StudyResult`
- Adding a new experiment parameter = one edit to `ExperimentConfig`

Negative / Trade-offs:
- Sweep resolution at parse time requires a clean YAML loader layer — design work needed for this boundary
- Losing the explicit "I'm in study mode" CLI signal — mitigated by the YAML file being that signal
- Two public functions to maintain (`run_experiment`, `run_study`) — mitigated by both delegating to single internal `_run(StudyConfig)`

Neutral / Follow-up decisions triggered:
- YAML loader boundary (sweep resolution layer) needs detailed design
- `cli-ux.md` needs updating (3 commands → 2 + flag)
- `designs/cli-commands.md` needs updating
- `study-execution-model.md` Decision D is now superseded by this ADR

---

## Design Details (Resolved)

### Sweep resolution at parse time

The YAML loader layer sits between raw YAML and Pydantic model construction:

```python
def load_yaml(path: Path) -> StudyConfig:
    """Load YAML, resolve sweeps, return validated StudyConfig."""
    raw = yaml.safe_load(path.read_text())

    if "sweep" in raw or "experiments" in raw:
        # Multi-experiment: resolve sweep grammar → flat list
        experiments = resolve_sweep(raw)
        execution = ExecutionConfig(**raw.get("execution", {}))
        return StudyConfig(experiments=experiments, execution=execution)
    else:
        # Single experiment: wrap in degenerate StudyConfig
        config = ExperimentConfig(**raw)
        return StudyConfig(experiments=[config])
```

The `sweep:` block and dotted notation (`pytorch.batch_size: [1, 8]`) are YAML syntax features. The Pydantic models never see them.

### Library user constructs ExperimentConfig directly

```python
config = ExperimentConfig(model="X", backend="pytorch", precision="fp16")
result = llem.run_experiment(config)  # internally wraps in StudyConfig, runs, unwraps result
```

`run_experiment()` wraps the input in a single-experiment `StudyConfig`, calls `_run()`, and
returns `study_result.experiments[0]`.

### Public API delegation to internal runner

```python
def run_experiment(config, **kwargs) -> ExperimentResult:
    study = _to_study_config(config, **kwargs)
    assert len(study.experiments) == 1
    study_result = _run(study)
    return study_result.experiments[0]

def run_study(config) -> StudyResult:
    study = _to_study_config(config)
    return _run(study)
```

Each public function has an unambiguous return type. Both delegate to the same `_run(StudyConfig) -> StudyResult` internal runner.

---

## Related

- [study-execution-model.md § Decision D](study-execution-model.md) — superseded by this ADR
- [cli-ux.md](cli-ux.md) — needs updating (3 commands → 2 + flag)
- [architecture.md](architecture.md) — library-first architecture
- [config-architecture.md](config-architecture.md) — composition vs inheritance (settled)
- [zero-config-defaults.md](zero-config-defaults.md) — related UX decision stub
- [progressive-disclosure.md](progressive-disclosure.md) — related UX decision stub
- [../designs/cli-commands.md](../designs/cli-commands.md) — needs updating
