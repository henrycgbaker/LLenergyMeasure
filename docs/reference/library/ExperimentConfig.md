---
title: ExperimentConfig
description: Pydantic model representing one LLM measurement point - model, engine, dataset, and measurement settings.
---

# `ExperimentConfig`

```python
from llenergymeasure import ExperimentConfig
```

## Concept

`ExperimentConfig` is the pure scientific specification for a single measurement. It captures
everything that defines *what is being measured* and *how it is measured*, with no knowledge
of studies, sweeps, cycles, or output paths (those live on
[`StudyConfig`](./StudyConfig)).

The design is intentional: two `ExperimentConfig` objects with identical field values always
represent the same experiment, regardless of when or how many times they are run. This
property powers the deduplication and reproducibility tracking that the study layer builds
on top of it.

`ExperimentConfig` is a Pydantic `BaseModel` with `extra="forbid"` and a frozen instance
(immutable after construction). Validation runs at construction time; invalid configs raise
`pydantic.ValidationError` immediately.

---

## Construction

### From YAML (via loader)

The standard path - the loader handles file I/O, sweep expansion, and returns validated
objects ready for execution:

```python
from llenergymeasure.config.loader import load_experiment_config
from pathlib import Path

config = load_experiment_config(path=Path("experiment.yaml"))
```

### From kwargs

```python
from llenergymeasure import ExperimentConfig
from llenergymeasure.config.models import TaskConfig

config = ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    engine="transformers",
    serving_mode="offline",
)
```

### From another config (override pattern)

Pydantic's `model_copy(update=...)` creates a new instance with selected fields changed.
Useful for building a family of configs from a base:

```python
base = ExperimentConfig(task=TaskConfig(model="gpt2"), engine="transformers", serving_mode="offline")

# Derive a variant with a different model
large = base.model_copy(update={"task": base.task.model_copy(update={"model": "gpt2-large"})})
```

---

## Fields

### Top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `task` | `TaskConfig` | _(required)_ | What to measure: model, dataset, token limits, seed. |
| `engine` | `str` | `"transformers"` | Inference engine: `"transformers"`, `"vllm"`, or `"tensorrt"`. |
| `serving_mode` | `"offline" \| "server"` | _(required)_ | Serving mode discriminator, no default. `"offline"` measures batch inference over a fixed prompt set (the only mode with an execution path today). `"server"` selects online serving measurement and requires `server` to be set with a traffic spec - see [Study config reference](/reference/study-config#server-mode-server) for the full field reference. |
| `measurement` | `MeasurementConfig` | `MeasurementConfig()` | How to measure: warmup, baseline, energy sampler. |
| `sampling_preset` | `"deterministic" \| "standard" \| "creative" \| "factual" \| None` | `None` | When set, preset values are merged into the active engine's sampling section at parse time. Explicit YAML values take precedence. |
| `transformers` | `TransformersConfig \| None` | `None` | Transformers-specific settings (only used when `engine="transformers"`). |
| `vllm` | `VLLMConfig \| None` | `None` | vLLM-specific settings (only used when `engine="vllm"`). |
| `tensorrt` | `TensorRTConfig \| None` | `None` | TensorRT-LLM settings (only used when `engine="tensorrt"`). |
| `server` | `ServerSection \| None` | `None` | Online-serving traffic spec (only used when `serving_mode="server"`). |
| `passthrough_kwargs` | `dict[str, Any] \| None` | `None` | Extra kwargs forwarded to the engine at execution time. Keys must not collide with top-level `ExperimentConfig` fields. |

### `TaskConfig` fields

Nested under `task:` in YAML, or `TaskConfig(...)` in Python:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `str` | _(required)_ | HuggingFace Hub ID or local path. |
| `dataset` | `DatasetConfig` | `DatasetConfig()` | Dataset source and prompt count. |
| `max_input_tokens` | `int \| None` | `256` | Truncate input prompts to this many tokens. `None` = no truncation. |
| `max_output_tokens` | `int \| None` | `256` | Maximum generated tokens (`max_new_tokens`). `None` = generate until EOS. |
| `random_seed` | `int` | `42` | Seed for all stochasticity: inference RNG and dataset ordering. |

### `MeasurementConfig` fields

Nested under `measurement:` in YAML, or `MeasurementConfig(...)` in Python:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `baseline` | `BaselineConfig` | `BaselineConfig()` | Idle power baseline settings (see below). |
| `energy_sampler` | `"auto" \| "nvml" \| "zeus" \| "codecarbon" \| None` | `"auto"` | Energy measurement sampler. `"auto"` selects the best available (Zeus > NVML > CodeCarbon). `None` disables energy measurement. |

### `WarmupConfig` fields (nested under `offline.warmup`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable warmup phase. |
| `n_prompts` | `int` | `5` | Number of full-length warmup prompts in fixed mode. Minimum 1. |
| `thermal_floor_seconds` | `float` | `60.0` | Minimum seconds to wait after warmup for thermal stabilisation. Minimum 30s enforced. |
| `convergence_detection` | `bool` | `False` | Enable CV-based adaptive convergence detection (replaces the fixed `n_prompts` count). |
| `cv_threshold` | `float` | `0.05` | CV target for convergence (0.01-0.50, only used when `convergence_detection=True`). |
| `max_prompts` | `int` | `20` | Safety cap on warmup prompts in CV mode. |

### `BaselineConfig` fields (nested under `measurement.baseline`)

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `True` | Enable idle power baseline measurement. |
| `duration_seconds` | `float` | `30.0` | Baseline measurement duration (5-120s). |
| `strategy` | `"cached" \| "validated" \| "fresh"` | `"validated"` | `"cached"`: disk-persisted with TTL. `"validated"`: cached with periodic spot-checks. `"fresh"`: measure every experiment (most accurate, ~30s overhead per experiment). |
| `cache_ttl_seconds` | `float` | `7200.0` | How long a cached baseline remains valid. Used with `"cached"` or `"validated"`. |
| `validation_interval` | `int` | `5` | Re-validate every N experiments. Used with `"validated"` only. |
| `drift_threshold` | `float` | `0.10` | Power drift fraction to trigger re-measurement. Used with `"validated"` only. |

---

## Validation

Pydantic validation runs at construction time. Common errors:

**Engine-section mismatch.** Providing a `transformers:` section when `engine="vllm"` is a
configuration error and raises `ValidationError`. The engine section must match the `engine`
field:

```python
# Raises: transformers: config section provided but engine='vllm'
ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    engine="vllm",
    serving_mode="offline",
    transformers=TransformersConfig(batch_size=4),  # wrong engine
)
```

**`passthrough_kwargs` collision.** Keys in `passthrough_kwargs` must not overlap with
top-level `ExperimentConfig` field names:

```python
# Raises: passthrough_kwargs keys collide with ExperimentConfig fields: ['engine']
ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    serving_mode="offline",
    passthrough_kwargs={"engine": "custom"},  # collides
)
```

**FlashAttention dtype.** `attn_implementation="flash_attention_2"` or `"flash_attention_3"`
require `dtype="float16"` or `dtype="bfloat16"`. Combining with `dtype="float32"` raises
`ValidationError`.

**Engine rules.** The engine's shipped `rules.yaml` is checked when the config is parsed,
before the engine starts. A rule at `error` severity raises `ValidationError` for an
invalid combination; a rule at `dormant` severity records a field the engine silently
normalises, which the study planner uses to deduplicate configs that resolve to the same
effective configuration. A separate `ConfigValidationWarning` is emitted for an unknown or
misspelled engine field, with a suggestion for the closest valid name.

---

## Common patterns

### Programmatic config building

```python
from llenergymeasure import ExperimentConfig
from llenergymeasure.config.models import TaskConfig, MeasurementConfig, WarmupConfig

config = ExperimentConfig(
    task=TaskConfig(
        model="meta-llama/Llama-3.1-8B",
        max_input_tokens=512,
        max_output_tokens=256,
    ),
    engine="transformers",
    serving_mode="offline",
    measurement=MeasurementConfig(
        warmup=WarmupConfig(n_prompts=10, thermal_floor_seconds=90.0),
        energy_sampler="nvml",
    ),
)
```

### Override pattern - build a family from a base

```python
base = ExperimentConfig(task=TaskConfig(model="gpt2"), engine="transformers", serving_mode="offline")

variants = [
    base.model_copy(update={"task": base.task.model_copy(update={"model": m})})
    for m in ["gpt2", "gpt2-medium", "gpt2-large"]
]
```

### Inspect the config hash

```python
from llenergymeasure.domain.experiment import compute_declared_config_hash

h = compute_declared_config_hash(config)
print(h)  # 16-char hex, e.g. "a3f2b19c7e4d0a81"
```

Two configs with identical fields produce the same hash, regardless of construction order.

---

## See also

- [`StudyConfig`](./StudyConfig) - thin container for a resolved list of `ExperimentConfig` objects
- [`run_experiment`](./run_experiment) - runs a single `ExperimentConfig`
- [`ExperimentResult`](./ExperimentResult) - the result returned after running
- [Study config reference](/reference/study-config) - YAML syntax for experiment files
