---
title: StudyConfig
description: Pydantic model holding a resolved list of ExperimentConfig objects plus execution controls.
---

# `StudyConfig`

```python
from llenergymeasure import StudyConfig
```

## Concept

`StudyConfig` is the thin resolved container that the study layer operates on. It holds a flat
list of fully-validated [`ExperimentConfig`](./ExperimentConfig) objects (the sweep is already
expanded - no axes, no templates) plus two operational blocks: `output` for where to write
results, and `study_execution` for how to sequence experiments across cycles.

The key design point is that sweep resolution happens *at YAML parse time* in the config
loader, before the `StudyConfig` is constructed. By the time a `StudyConfig` exists, every
item in `experiments` is a concrete, validated configuration ready to run. The study runner
and harness never perform expansion - they iterate the list.

Loading a `StudyConfig` from YAML is a two-step pipeline. The config loader
(`config.loader.load_study_config`) does pure parse + sweep-expansion and returns a
`LoadedStudyRaw`; study finalisation (`study.loading.finalise_study`) applies dedup,
computes `study_design_hash`, and orders cycles to produce the resolved `StudyConfig`. The
public entry that runs both steps is `api.load_study`. `run_study` calls it for you.

`StudyConfig` is rarely constructed directly in userland. The normal path is to pass a YAML
path to [`run_study`](./run_study) and let it load and finalise the config. Direct
construction is useful for programmatic test generation or pipeline integration.

---

## Construction

### From YAML (via `run_study` - standard path)

```python
from llenergymeasure import run_study

result = run_study("study.yaml")
```

### From YAML (via `load_study`, for inspection)

```python
from llenergymeasure.api import load_study
from pathlib import Path

study = load_study(Path("study.yaml"))
print(f"Expanded to {len(study.experiments)} experiments")
```

`load_study` returns the resolved `StudyConfig`. For the raw pre-finalisation material
(sweep-expanded but before dedup/hash/cycles), call `config.loader.load_study_config`
directly; it returns a `LoadedStudyRaw` whose `valid_experiments` list holds the expanded
configs.

### Programmatic construction

```python
from llenergymeasure import StudyConfig, ExperimentConfig
from llenergymeasure.config.models import TaskConfig

study = StudyConfig(
    experiments=[
        ExperimentConfig(task=TaskConfig(model="gpt2"), engine="transformers"),
        ExperimentConfig(task=TaskConfig(model="gpt2-medium"), engine="transformers"),
    ],
    study_name="gpt2-family",
)
```

---

## Fields

### Core fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `experiments` | `list[ExperimentConfig]` | _(required)_ | Resolved list of experiments to run. Minimum length 1. |
| `study_name` | `str \| None` | `None` | Study name used in output directory naming (e.g. `gpt2-sweep_2026-05-07T14-00-00`). |
| `study_design_hash` | `str \| None` | `None` | 16-char SHA-256 hex of the resolved experiment list (execution block excluded). Set by study finalisation after dedup; `None` when constructed programmatically. |

### `output` block (`OutputConfig`)

Controls where results are written:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `results_dir` | `str \| None` | `None` | Base directory for study results. `None` defers to user config or the built-in default (`./results`). A timestamped subdirectory is created inside this path. |
| `save_timeseries` | `bool` | `True` | Persist GPU power/thermal/memory samples as a Parquet sidecar per experiment. Set to `False` to save disk space on long sweeps. |

### `study_execution` block (`ExecutionConfig`)

Controls sequencing, cycles, and failure handling:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `n_cycles` | `int` | `1` | Number of times to repeat the full experiment list. |
| `experiment_order` | `"sequential" \| "interleave" \| "shuffle" \| "reverse" \| "latin_square"` | `"sequential"` | Ordering across cycles. `"interleave"` runs A,B,A,B; `"shuffle"` randomises per-cycle. |
| `experiment_gap_seconds` | `float \| None` | `None` | Seconds to wait between individual experiments. `None` uses machine default from user config. |
| `cycle_gap_seconds` | `float \| None` | `None` | Seconds to wait between full cycles. `None` uses machine default. |
| `shuffle_seed` | `int \| None` | `None` | Explicit seed for `shuffle` ordering. `None` derives seed from `study_design_hash` (same study always shuffles identically). |
| `skip_preflight` | `bool` | `False` | Skip Docker pre-flight checks in the execution block (the `--skip-preflight` CLI flag always overrides). |
| `max_consecutive_failures` | `int` | `10` | Circuit breaker: abort after N consecutive failures. `0` = disabled. |
| `circuit_breaker_cooldown_seconds` | `float` | `60.0` | Cooldown before a half-open probe experiment after the circuit breaker trips. |
| `wall_clock_timeout_hours` | `float \| None` | `None` | Study-level wall-clock timeout. `None` = no limit. |
| `experiment_timeout_seconds` | `float` | `600.0` | Per-experiment timeout. Applies to both local and Docker paths. |
| `stdout_silence_timeout_seconds` | `float` | `300.0` | Docker watchdog: kill the container if it emits no stdout for this many seconds. Raise to 600-900s for TRT-LLM builds with infrequent compile progress output. |
| `deduplicate_equivalent` | `bool` | `True` | When `True`, configs that are library-equivalent (same resolved config hash after engine-invariants dormant resolution) are deduplicated. Set to `false` to run every declared config regardless of measurement equivalence. |

### Runner and image overrides

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `runners` | `dict[str, str] \| None` | `None` | Per-engine runner: `{"transformers": "local", "vllm": "docker"}`. `None` = use user config or auto-detection. |
| `images` | `dict[str, str] \| None` | `None` | Per-engine Docker image overrides: `{"vllm": "ghcr.io/org/img:tag"}`. `None` = smart default (local build then registry fallback). |

### Finalisation-populated metadata fields

These are written by study finalisation and the study runner; you rarely set them manually:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `skipped_configs` | `list[dict]` | `[]` | Grid points that failed Pydantic validation at sweep-expansion time. |
| `dedup_mode` | `"resolved" \| "off"` | `"resolved"` | Current deduplication mode. Set via `ExecutionConfig.deduplicate_equivalent`. |
| `pre_run_equivalence_groups` | `list[dict]` | `[]` | Equivalence groups computed at expansion time and written to `equivalence_groups.json` in the results bundle. |
| `declared_resolved_config_hashes` | `list[str]` | `[]` | Resolved config hashes parallel to the pre-expansion sweep input. Used by the harness for sidecar tagging. |

---

## Validation

`StudyConfig` enforces `extra="forbid"`. Unknown top-level keys raise `ValidationError`.

`experiments` must be non-empty (Pydantic `min_length=1`). A `StudyConfig` with zero
experiments cannot be constructed.

Each item in `experiments` is individually validated as an `ExperimentConfig` (see
[`ExperimentConfig`](./ExperimentConfig) for per-experiment validation rules, including
engine-section matching and engine-invariants checking).

---

## Common patterns

### Inspect expanded experiments before running

```python
from llenergymeasure.api import load_study
from pathlib import Path

study = load_study(Path("sweep.yaml"))
print(f"{len(study.experiments)} experiments after expansion")
for i, exp in enumerate(study.experiments, 1):
    print(f"  {i:>3}. {exp.task.model} / {exp.engine}")
```

### Override output directory programmatically

```python
study = load_study(Path("study.yaml"))
study = study.model_copy(
    update={"output": study.output.model_copy(update={"results_dir": "/data/results"})}
)
result = run_study(study)
```

### Pass a pre-built StudyConfig to `run_study`

```python
from llenergymeasure import run_study, StudyConfig, ExperimentConfig
from llenergymeasure.config.models import TaskConfig, ExecutionConfig

study = StudyConfig(
    experiments=[ExperimentConfig(task=TaskConfig(model="gpt2"), engine="transformers")],
    study_name="quick-check",
    study_execution=ExecutionConfig(n_cycles=3, experiment_order="interleave"),
)
result = run_study(study)
```

---

## See also

- [`ExperimentConfig`](./ExperimentConfig) - the per-experiment config model
- [`run_study`](./run_study) - the function that accepts a `StudyConfig`
- `api.load_study` - load a YAML path into a resolved `StudyConfig` (parse + finalise)
- [`ExperimentResult`](./ExperimentResult) - result type for each experiment
- [Study config reference](/reference/study-config) - YAML syntax (sweep, execution block)
- [Results schema](/reference/results-schema) - on-disk layout
