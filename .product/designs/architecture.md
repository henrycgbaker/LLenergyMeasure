# Architecture Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/architecture.md](../decisions/architecture.md),
                     [../decisions/experiment-study-architecture.md](../decisions/experiment-study-architecture.md)
**Status**: Confirmed (v2.0 target)

---

## Module Layout (`src/` layout)

```
src/
  llenergymeasure/                   ← library package (primary interface)
    __init__.py                      ← PUBLIC API: run_experiment, run_study,
                                     ←   ExperimentConfig, StudyConfig,
                                     ←   ExperimentResult, StudyResult
    _api.py                          ← implementation of run_experiment / run_study
                                     ←   (imported into __init__.py, not public itself)
    protocols.py                     ← DI interfaces: ModelLoader, InferenceEngine,
                                     ←   MetricsCollector, EnergyBackend, ResultsRepository
                                     ←   (carry-forward from v1.x; see preservation audit N-X09)
    config/                          ← YAML parsing + Pydantic config models
      __init__.py
      models.py                      ← ExperimentConfig, StudyConfig, DecoderConfig, etc.
      backend_configs.py             ← PyTorchConfig, VLLMConfig, TensorRTConfig
      user_config.py                 ← UserConfig (runner profiles, execution profiles)
      loader.py                      ← load_experiment_config(), load_study_config()
      ssot.py                        ← PRECISION_SUPPORT, DECODING_SUPPORT dicts (SSOT)
      introspection.py               ← SSOT: model_json_schema() → test values, docs, CLI
                                     ←   (carry-forward from v1.x, lean on Pydantic v2 API;
                                     ←    see preservation audit P-03)
    core/                            ← inference engine, energy backends, metrics
      __init__.py
      backends/                      ← inference backends
        protocol.py                  ← InferenceBackend Protocol
        pytorch.py
        vllm.py
        tensorrt.py
      energy/                        ← energy measurement backends
        protocol.py                  ← EnergyBackend Protocol
        nvml.py                      ← base NVML polling (always available)
        zeus.py                      ← optional: ZeusMonitor (llenergymeasure[zeus])
        codecarbon.py                ← optional: CodeCarbon (llenergymeasure[codecarbon])
      metrics.py                     ← FLOPs calculation, throughput, latency statistics
    domain/                          ← Pydantic result + domain models
      __init__.py
      results.py                     ← ExperimentResult, StudyResult, RawProcessResult
      environment.py                 ← EnvironmentSnapshot
      co2.py                         ← CO2Estimate
    orchestration/                   ← experiment lifecycle
      __init__.py
      orchestrator.py                ← ExperimentOrchestrator (INTERNAL — not exported)
      preflight.py                   ← pre-flight validation logic
    results/                         ← persistence and aggregation
      __init__.py
      persistence.py                 ← to_json(), to_parquet(), from_json()
      aggregation.py                 ← late aggregation (per-process → ExperimentResult)
      filenames.py                   ← human-readable filename generation
    state/                           ← state machine (3 states: target from current 6)
      __init__.py
      machine.py                     ← ExperimentState: INITIALISING → MEASURING → DONE
    study/                           ← study module (ships with core at v2.0)
      __init__.py
      runner.py                      ← StudyRunner (subprocess orchestration)
      grid.py                        ← sweep: → list[ExperimentConfig] expansion
      manifest.py                    ← study_manifest.json read/write
    cli/                             ← thin Typer wrapper (2 commands + 1 flag)
      __init__.py
      app.py                         ← Typer app definition
      commands/
        run.py                       ← `llem run` command (handles both experiments and studies)
        config.py                    ← `llem config` command
      display.py                     ← Plain text output (~200 LOC, tqdm progress;
                                     ←   see preservation audit N-X04 — Rich dropped)
    datasets/                        ← built-in datasets (ship with package)
      __init__.py
      loader.py                      ← load_dataset() dispatcher
      builtin/
        aienergyscore.jsonl          ← default built-in (1000 prompts)
      synthetic.py                   ← SyntheticDatasetGenerator
```

---

## Public API (`__init__.py`)

```python
# src/llenergymeasure/__init__.py
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

**Stability contract**: everything in `__all__` is SemVer-stable from v2.0.
Everything NOT in `__init__.py` is internal — may change without notice.

`ExperimentOrchestrator` is intentionally NOT exported. It is accessible at
`llenergymeasure.orchestration.orchestrator.ExperimentOrchestrator` for advanced users
building custom integrations, but carries no SemVer guarantee.

---

## Call Graph: `run_experiment()` vs `run_study()`

Both public functions delegate to `_run(StudyConfig) -> StudyResult` internally.
`run_experiment()` wraps input into a single-experiment `StudyConfig` and unwraps the result.

```
run_experiment(config) -> ExperimentResult     run_study(config) -> StudyResult
      │                                               │
      ▼                                               ▼
  _to_study_config(config)                     _to_study_config(config)
      │                                               │
      ▼                                               ▼
  _run(StudyConfig) ──────────────────────── _run(StudyConfig)
      │                                               │
      ▼                                               ▼
  StudyRunner                                 StudyRunner
  (single experiment)                         │
      │                                       │  grid.py: sweep → list[ExperimentConfig]
      │                                       │
      ▼                                       │  for each experiment_config:
  ExperimentOrchestrator                      │    sleep(config_gap_seconds)
  (in-process for single)                     │    mp_ctx.Process(
      │                                       │        target=_run_experiment_worker,
      ▼                                       │        args=(config, pipe, queue),
  unwrap: result.experiments[0]               │    )
      │                                       │    result ← Pipe
      ▼                                       │
  ExperimentResult                            ▼
                                          StudyResult
```

`run_experiment()` for a single experiment may run in-process (no subprocess needed —
clean GPU state at start is guaranteed).

`run_study()` wraps each `ExperimentOrchestrator` call in a `multiprocessing.Process` to
guarantee clean GPU state between experiments.

---

## Config Model (Two Sources + Auto-Capture)

Three distinct concerns that must not be mixed:

```
User Config (HOW to execute)
  Location: ~/.config/llenergymeasure/config.yaml   ← user-local, never committed
  Contents: Backend → execution environment mapping, execution defaults
  Example:
    runners:
      pytorch: local
      vllm:    docker:ghcr.io/llenergymeasure/vllm:2.2.0-cuda12.4

Experiment / Study YAML (WHAT to measure)
  Location: experiment.yaml / study.yaml             ← versioned, shareable
  Contents: Model, dataset, backend, hyperparams, sweep definition
  Example:
    model: meta-llama/Llama-3.1-8B
    backend: pytorch
    sweep:
      precision: [fp16, bf16]

Environment Snapshot (WHAT you're measuring ON)
  Location: auto-detected at runtime; stored with results
  Contents:
    Auto-detected: GPU model, VRAM, count, TDP, CPU model, CUDA version, driver
    User-specified defaults (from user config): carbon_intensity_gco2_kwh, datacenter_pue
  Stored: In ExperimentResult — must travel with published data for reproducibility
```

**Reference pattern**: Nextflow (`nextflow.config` profiles), Snakemake (`--profile`),
DVC (`~/.dvc/config`). Same separation: portable pipeline + user-local environment.

See [user-config.md](user-config.md) for user config schema.
See [experiment-config.md](experiment-config.md) for experiment config schema.
See [config-model.md](config-model.md) for the SSOT field placement and data flow.

---

## State Machine (3 States)

Current codebase has 6 states. v2.0 target: 3 states.

```python
class ExperimentState(str, Enum):
    INITIALISING = "initialising"   # pre-flight, model load, warmup
    MEASURING    = "measuring"      # active inference measurement window
    DONE         = "done"           # complete or failed
```

<!-- TODO: Review current 6-state machine to confirm the 3-state target is sufficient.
     The current states are: INITIALISING, WARMING_UP, MEASURING, AGGREGATING, SAVING, DONE.
     Collapsing WARMING_UP into INITIALISING and AGGREGATING+SAVING into DONE should be fine,
     but confirm no external code depends on the current state names before removal. -->

---

## NVML Single-Session Owner

Only one NVML session can be active at a time. When Zeus is installed and active, the base
NVML poller must yield — they cannot both run simultaneously.

```python
# src/llenergymeasure/core/energy/__init__.py
def get_active_energy_backend(config: ExperimentConfig) -> EnergyBackend:
    """Returns the active energy backend, enforcing single-session ownership."""
    if _zeus_available() and config.energy_backend != "nvml":
        return ZeusBackend()   # Zeus owns NVML session — base poller must not start
    return NVMLBackend()       # Base NVML poller owns the session
```

The backend layer enforces mutual exclusion. `ExperimentOrchestrator` calls
`get_active_energy_backend()` and uses the result — it does not make its own NVML decisions.

---

## StudyRunner: Subprocess Lifecycle

See [experiment-isolation.md](experiment-isolation.md) for the full subprocess pattern.
See [../decisions/experiment-isolation.md](../decisions/experiment-isolation.md) for the decision.
Key structural points:

```python
# src/llenergymeasure/study/runner.py (pseudocode)

class StudyRunner:
    def __init__(self, study: StudyConfig, user_config: UserConfig): ...

    def run(self) -> StudyResult:
        experiments = grid.expand(self.study)   # sweep → list[ExperimentConfig]
        study_result = StudyResult(...)
        mp_ctx = multiprocessing.get_context("spawn")   # CUDA requires spawn, not fork

        for config in experiments:
            self._write_manifest(study_result)           # checkpoint after each experiment
            time.sleep(self.study.execution.config_gap_seconds)
            result_or_error = self._run_one(config, mp_ctx)
            study_result.add(result_or_error)

        return study_result

    def _run_one(self, config, mp_ctx) -> ExperimentResult | StudyFailed:
        # Local: multiprocessing.Process + Pipe
        # Docker: subprocess.run(["docker", "run", ...]) + shared volume
        ...
```

---

## Peer References

| Tool | Matches our pattern |
|---|---|
| **lm-eval-harness** | Library-first (`simple_evaluate` returns dict); CLI is thin wrapper; `lm_eval/` inside package | ✓ |
| **MLflow** | `mlflow.*` top-level public API; `mlflow.client`, `mlflow.tracking` are internal | ✓ |
| **optimum-benchmark** | `src/` layout; one `Process` per benchmark; Protocol-based backends | ✓ |
| **Zeus** | Library-first; `ZeusMonitor` is the API; no CLI | ✓ (library pattern) |
| **CodeCarbon** | `EmissionsTracker` as library API; `carbonboard` as separate CLI tool | ✓ |

---

## Related

- [../decisions/architecture.md](../decisions/architecture.md): Confirmed decisions
- [experiment-config.md](experiment-config.md): ExperimentConfig schema
- [experiment-isolation.md](experiment-isolation.md): Subprocess isolation pattern
- [library-api.md](library-api.md): Public API design
- [user-config.md](user-config.md): User config schema
