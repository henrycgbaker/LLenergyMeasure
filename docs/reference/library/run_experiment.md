---
title: run_experiment
description: Run a single LLM inference efficiency measurement and return an ExperimentResult.
---

# `run_experiment`

```python
from llenergymeasure import run_experiment
```

## Concept

`run_experiment` is the simplest entry point for a one-off measurement. It accepts a model
name, an engine, and optional measurement settings, runs inference against the
`aienergyscore` prompt set (or a dataset you specify), and returns an
[`ExperimentResult`](./ExperimentResult) containing energy, throughput, FLOPs, and
timing data.

Use `run_experiment` when you want to measure a single model-engine pair. When you need to
sweep across multiple models, engines, or parameter axes - or run the same configuration
more than once for statistical reliability - use [`run_study`](./run_study) instead.
Internally, `run_experiment` wraps a degenerate `StudyConfig` containing exactly one
experiment and unwraps the result before returning it.

---

## Simple usage

```python
from llenergymeasure import run_experiment

result = run_experiment(model="gpt2", engine="transformers")

print(f"Energy:     {result.total_energy_j:.1f} J")
print(f"Throughput: {result.avg_tokens_per_second:.1f} tok/s")
print(f"Efficiency: {result.energy_per_token_mj_total:.3f} mJ/tok")
```

The return value is an [`ExperimentResult`](./ExperimentResult). For field definitions and
the on-disk JSON schema it mirrors, see
[Results schema](/reference/results-schema).

---

## Config-from-file usage

```python
result = run_experiment("experiment.yaml")
```

`experiment.yaml`:

```yaml
serving_mode: offline

task:
  model: meta-llama/Llama-3.1-8B
  dataset:
    source: aienergyscore
    n_prompts: 100

engine: transformers

measurement:
  baseline:
    enabled: true
    strategy: validated
  energy_sampler: auto

offline:
  warmup:
    n_prompts: 5
```

Any kwarg you pass alongside a YAML path overrides the corresponding field:

```python
# Load the YAML but force a different model
result = run_experiment("experiment.yaml", model="gpt2")
```

---

## Parameter table

`run_experiment` accepts three mutually exclusive call forms:

| Form | First argument | Required kwargs |
|------|---------------|-----------------|
| YAML path | `str` or `Path` to a YAML file | none |
| Config object | `ExperimentConfig` instance | none |
| kwargs | `None` (omit) | `model=` |

### Shared keyword arguments

These apply to all three call forms:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `skip_preflight` | `bool` | `False` | Skip Docker pre-flight checks (GPU visibility, CUDA/driver compatibility). Useful in CI or when using a remote Docker daemon. |
| `progress` | `ProgressCallback \| None` | `None` | Step-by-step progress callback. Receives `on_step_start` / `on_step_done` events during preflight, warmup, and inference phases. |
| `output_dir` | `str \| Path \| None` | `None` | Override the base directory for results. A timestamped study subdirectory is created within this path. `None` defers to `results/` (built-in default). |

### kwargs-form-only parameters

Used when `config` is `None` (omitted):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | _(required)_ | HuggingFace Hub ID or local path (e.g. `"gpt2"`, `"meta-llama/Llama-3.1-8B"`). |
| `engine` | `str \| None` | `None` | Inference engine: `"transformers"`, `"vllm"`, or `"tensorrt"`. `None` falls back to the `ExperimentConfig` default (`"transformers"`). |
| `n_prompts` | `int` | `100` | Number of prompts to run. Matches `DatasetConfig.n_prompts` default. |
| `dataset` | `str` | `"aienergyscore"` | Dataset source: built-in alias or path to a `.jsonl` file. |
| `**kwargs` | `Any` | - | Additional `ExperimentConfig` fields. Task-level fields (`max_input_tokens`, `max_output_tokens`, `random_seed`) and measurement-level fields (`energy_sampler`, etc.) are routed automatically. |

---

## Returns

[`ExperimentResult`](./ExperimentResult) - a frozen Pydantic model containing all
measurements. Key fields:

```python
result.total_energy_j          # float  - total GPU energy in joules
result.energy_adjusted_j       # float | None - baseline-subtracted energy
result.avg_tokens_per_second   # float  - throughput
result.energy_per_token_mj_total        # float | None - millijoules per token (total)
result.energy_per_token_mj_adjusted     # float | None - millijoules per token (baseline-adjusted)
result.total_flops             # float  - estimated FLOPs (reference)
result.total_inference_time_sec # float - wall time
```

See [`ExperimentResult`](./ExperimentResult) for the full field list.

---

## Common patterns

### Override a single field from a YAML file

The `model=` and other keyword fields belong to the keyword call form only; they are
ignored when a config file or object is passed. To swap one field of a file, load it and
edit the object:

```python
from llenergymeasure import run_experiment
from llenergymeasure.config.loader import load_experiment_config

config = load_experiment_config("base_config.yaml")
config = config.model_copy(
    update={"task": config.task.model_copy(update={"model": "meta-llama/Llama-3.1-8B"})}
)
result = run_experiment(config)
```

### Pass an ExperimentConfig object directly

The config you pass is resolved before it runs, exactly as a study file is: any field the
engine is known to ignore for this configuration is normalised away first, and it is that
normalised form that is dispatched and named in the results. So the experiment id recorded
for a config carrying an engine-ignored field is the id of the normalised config, not of
the config as you wrote it - `llem study plan` lists any such normalisation up front. Your
own object is not modified; the run works on a copy.

```python
from llenergymeasure import run_experiment, ExperimentConfig
from llenergymeasure.config.models import TaskConfig, MeasurementConfig

config = ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    engine="transformers",
    serving_mode="offline",
)
result = run_experiment(config)
```

### Capture results to a custom directory

```python
result = run_experiment(
    model="gpt2",
    engine="transformers",
    output_dir="/data/experiments/gpt2-baseline",
)
```

### Compare two single measurements

```python
result_a = run_experiment(model="gpt2", engine="transformers")
result_b = run_experiment(model="gpt2-xl", engine="transformers")

delta_pct = (result_b.energy_per_token_mj_total - result_a.energy_per_token_mj_total) / result_a.energy_per_token_mj_total * 100
print(f"gpt2-xl uses {delta_pct:+.1f}% more energy per token than gpt2")
```

---

## Raises

| Exception | When |
|-----------|------|
| `ConfigError` | No `config` argument and no `model=` kwarg; invalid YAML path; wrong argument type. |
| `pydantic.ValidationError` | Field value fails validation (e.g. `engine` not a known string, `n_prompts < 1`). Passes through unchanged. |
| `PreFlightError` | Engine requires Docker but Docker is not available or not running. |
| `ExperimentError` | Experiment ran but produced no results (e.g. engine crashed without raising). |

---

## Pitfalls

**When to use `run_study` instead.** If you are running the same configuration
multiple times to get a stable mean, or sweeping over a parameter axis, use
[`run_study`](./run_study). Calling `run_experiment` in a loop bypasses study-level
gap controls, cycle ordering, circuit-breaker logic, and the result manifest.

**Engines run in Docker.** Engine libraries are not installed on the host;
each engine runs inside its own Docker image. Docker with the NVIDIA Container
Toolkit must be available, and the engine image must be present locally or
pullable. See [Docker setup](/how-to/docker-setup) and
[engine configuration](/reference/engines/configuration). When Docker is
unavailable, `run_experiment` raises `PreFlightError` at preflight, before any
inference starts.

**kwargs routing is automatic but transparent.** When using the kwargs form, known
`TaskConfig` fields (`max_input_tokens`, `max_output_tokens`, `random_seed`) and
`MeasurementConfig` fields (`energy_sampler`, etc.) are routed to the correct sub-model.
Unknown keys land on `ExperimentConfig` directly and raise `ValidationError` if unrecognised.

---

## See also

- [`run_study`](./run_study) - multi-experiment sweeps with cycles, ordering, and manifests
- [`ExperimentConfig`](./ExperimentConfig) - the config object accepted in the config-object form
- [`ExperimentResult`](./ExperimentResult) - the return type
- [Study config reference](/reference/study-config) - YAML syntax for experiment files
- [Results schema](/reference/results-schema) - on-disk JSON schema
