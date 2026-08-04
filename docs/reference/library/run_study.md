---
title: run_study
description: Run a multi-experiment study and return a StudyResult containing all measurements.
---

# `run_study`

```python
from llenergymeasure import run_study
```

## Concept

`run_study` runs a structured set of experiments defined in a YAML study config and returns a
`StudyResult` containing all measurements together with summary statistics. It is the right
entry point whenever you need more than one experiment - sweeping over models, engines, or
parameters, repeating a configuration multiple times for statistical reliability, or comparing
across an axis in a single reproducible bundle.

The distinction from [`run_experiment`](./run_experiment) is straightforward: `run_experiment`
measures one configuration and returns a single result; `run_study` measures many configurations
(expanded from sweep declarations at YAML parse time) and returns them together with study-level
metadata, a manifest on disk, and a result bundle you can share or archive. In offline mode each
study cell contributes one result to `StudyResult.experiments`; in server mode a single cell is one
server lifetime that contributes one result per measurement window.

`run_study` always writes a `manifest.json` to disk as a documented side-effect. The manifest is
both a resumption checkpoint (if the study is interrupted) and an audit trail linking each result
file to its config hash.

---

## Simple usage

The authoritative home for engine identity and model name is the `config.json`
sidecar next to each `result.json` (they are configuration inputs; `result.json`
keeps `engine` and `model_name` as convenience copies only), so analysis code
reads them from disk via `result_files`:

```python
import json
from pathlib import Path

from llenergymeasure import run_study

study_result = run_study("study.yaml")

for result_file in study_result.result_files:
    cell = Path(result_file).parent
    result = json.loads((cell / "result.json").read_text())
    config = json.loads((cell / "config.json").read_text())
    print(f"{config['model_name']} / {config['engine']}: {result['energy_per_token_mj_total']:.3f} mJ/tok")
```

`study.yaml` (minimal multi-experiment form):

```yaml
study_name: gpt2-comparison
serving_mode: offline

experiments:
  - task:
      model: gpt2
    engine: transformers

  - task:
      model: gpt2-medium
    engine: transformers
```

---

## Sweep usage

Sweeps are declared with a `sweep:` key in the YAML. The loader expands the Cartesian product
at parse time into a flat `experiments` list before any experiment runs.

```yaml
study_name: model-sweep
serving_mode: offline

sweep:
  axes:
    - field: task.model
      values:
        - gpt2
        - gpt2-medium
        - gpt2-large

    - field: engine
      values:
        - transformers
```

```python
study_result = run_study("sweep.yaml")

print(f"Ran {study_result.summary.completed} / {study_result.summary.total_experiments} experiments")
print(f"Total energy: {study_result.summary.total_energy_j:.1f} J")
```

---

## Parameter table

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config` | `str \| Path \| StudyConfig` | _(required)_ | YAML file path or a pre-built `StudyConfig` object. |
| `skip_preflight` | `bool` | `False` | Skip Docker pre-flight checks. Useful in CI or remote-daemon setups. |
| `progress` | `ProgressCallback \| None` | `None` | Progress callback. Receives per-experiment begin/end events and per-step events from worker processes. |
| `resume_dir` | `Path \| None` | `None` | Explicit study directory to resume. Overrides `resume`. |
| `resume` | `bool` | `False` | Auto-detect the most recent resumable study in `output_dir` and resume from the last checkpoint. |
| `output_dir` | `Path \| None` | `None` | Dual role by run mode. Fresh run: results-dir override (precedence `output_dir` > YAML `output.results_dir` > user config > `./results`). Auto-detect resume: base directory searched for the most recent resumable study. Ignored when `resume_dir` is given. |
| `skip_set` | `set[tuple[str, int]] \| None` | `None` | Set of `(config_hash, cycle)` pairs to skip. Populated automatically when resuming; callers rarely need to set this. |
| `no_lock` | `bool` | `False` | Skip GPU advisory lock acquisition. Equivalent to the `--no-lock` CLI flag. |
| `config_path` | `Path \| None` | `None` | Original YAML path for artefact copying when `config` is a `StudyConfig` object. Preserved in `_study-artefacts/` for reproducibility. |
| `cli_overrides` | `dict[str, Any] \| None` | `None` | Flat dict of CLI flag overrides recorded in the per-experiment `config.json` provenance section. Rarely needed outside the CLI. |

---

## Returns

`StudyResult` - a Pydantic model:

```python
study_result.experiments          # list[ExperimentResult] - one per completed experiment
study_result.summary.completed    # int   - number of experiments that succeeded
study_result.summary.failed       # int   - number that failed
study_result.summary.total_energy_j  # float - summed energy across all experiments
study_result.summary.total_wall_time_s  # float - total wall-clock time
study_result.result_files         # list[str] - paths to result.json files on disk
study_result.study_name           # str | None
study_result.study_design_hash    # str | None - 16-char SHA-256 of the experiment list
study_result.measurement_protocol # dict - execution config snapshot (n_cycles, order, etc.)
study_result.skipped_experiments  # list[dict] - configs that failed validation at expand time
```

Each item in `experiments` is an [`ExperimentResult`](./ExperimentResult). See
[Results schema](/reference/results-schema) for the on-disk layout.

---

## Common patterns

These patterns join each `result.json` with its `config.json` sidecar (the
authoritative home of `engine` and `model_name`) via `result_files`.

### Filter results by engine

```python
import json
from pathlib import Path

transformers_cells = [
    Path(f).parent
    for f in study_result.result_files
    if json.loads((Path(f).parent / "config.json").read_text())["engine"] == "transformers"
]
```

### Compare energy across models

```python
import json
import statistics
from pathlib import Path

by_model: dict[str, list[float]] = {}
for result_file in study_result.result_files:
    cell = Path(result_file).parent
    result = json.loads((cell / "result.json").read_text())
    config = json.loads((cell / "config.json").read_text())
    by_model.setdefault(config["model_name"], []).append(result["energy_per_token_mj_total"] or 0.0)

for model, values in by_model.items():
    print(f"{model}: mean {statistics.mean(values):.3f} mJ/tok (n={len(values)})")
```

### Export to a DataFrame

```python
import json
from pathlib import Path
import pandas as pd

rows = []
for result_file in study_result.result_files:
    cell = Path(result_file).parent
    result = json.loads((cell / "result.json").read_text())
    config = json.loads((cell / "config.json").read_text())
    rows.append(
        {
            "model": config["model_name"],
            "engine": config["engine"],
            "energy_j": result["total_energy_j"],
            "throughput": result["avg_tokens_per_second"],
            "energy_per_token_mj": result["energy_per_token_mj_total"],
        }
    )
df = pd.DataFrame(rows)
```

### Resume an interrupted study

```python
# Picks up from the last completed experiment automatically
study_result = run_study("sweep.yaml", resume=True)
```

---

## Raises

| Exception | When |
|-----------|------|
| `ConfigError` | Invalid config path or YAML parse error. |
| `PreFlightError` | Multi-engine study where an auto-resolved engine needs Docker elevation but Docker is unavailable, or an engine pinned to `process` is not importable on the host. |
| `StudyError` | `resume=True` but no resumable study found; config drift detected on resume (study hash changed). |
| `pydantic.ValidationError` | A field value fails validation. Passes through unchanged. |

---

## Pitfalls

**Multi-engine Docker elevation is precedence-based.** In a study that references more than
one engine (e.g. both `engine: transformers` and `engine: vllm`), each engine still runs in
its own subprocess, so process isolation always holds. Docker elevation guards *environment
feasibility* - divergent engine dependency closures cannot coexist on one host - not
isolation. An engine whose runner you pin explicitly (via env var, the study `runners:`
section, or user config) keeps that pin; only engines whose runner resolved from
auto-detection or the default are elevated to a container. Because runner choice is machine-binding
and recorded per result, an explicit `runners: {vllm: process}` is a reproducibility assertion
that this host can run that engine, so `run_study` verifies the engine imports on the host at
preflight. It raises `PreFlightError` before any inference begins when an engine pinned to
`process` is not importable, or when an auto-resolved engine needs Docker elevation but Docker is
unavailable. An all-explicit-`process` multi-engine study runs without Docker.

**Result bundle on disk.** Every `run_study` call creates a timestamped directory under
`./results/` (or `output.results_dir` from the YAML). That directory is not cleaned up
automatically. Budget for disk space when sweeping large grids. See
[Results schema](/reference/results-schema) for the exact layout.

**Skipped configs.** If a sweep axis combination fails Pydantic validation (e.g.
`engine=tensorrt` with a `dtype` that is not supported), the invalid combination is
recorded in `study_result.skipped_experiments` and in `_study-artefacts/skipped_configs.log`,
but the rest of the study continues.

**`n_cycles` vs list length.** `study_result.summary.total_experiments` reflects the expanded
cycle count (`len(experiments) * n_cycles`). `summary.unique_configurations` is the number of
distinct configurations (pre-cycle). Both are in the summary.

---

## See also

- [`run_experiment`](./run_experiment) - single-experiment convenience wrapper
- [`StudyConfig`](./StudyConfig) - the config model accepted by `run_study` and returned by `api.load_study`
- [`ExperimentResult`](./ExperimentResult) - the per-experiment result type
- [Study config reference](/reference/study-config) - YAML syntax (sweep axes, execution block)
- [Results schema](/reference/results-schema) - on-disk layout
