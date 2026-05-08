---
title: Results schema (result.json + manifest.json + timeseries.parquet)
description: Field-by-field reference for everything llem writes to disk after a measurement.
---

# Results schema

Reference for everything `llem` writes to disk after a measurement. Three artefacts ship per study: a per-experiment `result.json`, a study-level `manifest.json`, and an optional `timeseries.parquet` sidecar.

For a guided walkthrough of how to *read* these files (with worked examples), see [How to interpret results](/how-to/interpret-results). For the methodology behind each metric, see [What we measure](/explanation/methodology/what-we-measure) and [Energy measurement](/explanation/methodology/energy-measurement).

## Output layout

A study run produces a directory tree like this:

```
results/
└── <study-name>_<UTC-timestamp>/
    ├── manifest.json                            # study-level checkpoint + summary
    ├── 001_c0_<model>-<engine>_<hash>/          # one experiment cell
    │   ├── result.json                          # all metrics + resolved config
    │   ├── effective_config.json                # final config used (post-expansion)
    │   └── timeseries.parquet                   # GPU power/thermal/memory samples
    ├── 002_c0_.../
    ├── ...
    └── _study-artefacts/
        ├── equivalence_groups.json              # dedup equivalence groups
        └── baseline_cache_<key>.json            # per-engine baseline cache
```

`<UTC-timestamp>` is ISO-8601 (e.g. `2026-05-07T14-32-08`). Cell directory names encode `<NNN>_c<cycle>_<model>-<engine>_<config-hash>` so they sort sensibly and you can tell sibling cycles apart at a glance.

## `result.json` - per-experiment record

The scientific record. One JSON file per experiment cell. Schema version `3.0`.

### Identification

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | str | Result schema version (currently `"3.0"`) |
| `experiment_id` | str | Unique experiment identifier (`{model}_{YYYYMMDD_HHMMSS}` for single experiments; study-level cells inherit a richer per-cell identifier) |
| `measurement_config_hash` | str | SHA-256[:16] of `ExperimentConfig` with environment fields excluded; same hash -> logically identical experiments |
| `llenergymeasure_version` | str &#124; null | Package version that produced this result |
| `engine` | str | Inference engine: `transformers` &#124; `vllm` &#124; `tensorrt` |
| `engine_version` | str &#124; null | Engine library version (e.g. `4.57.0` for transformers) |
| `model_name` | str | Model name or HuggingFace path used |

### Measurement methodology

| Field | Type | Description |
|-------|------|-------------|
| `measurement_methodology` | `"total"` &#124; `"steady_state"` &#124; `"windowed"` | Which slice of the run produced the headline metrics |
| `warmup_excluded_samples` | int &#124; null | Prompts excluded during warmup; `null` when `methodology = "total"` |
| `reproducibility_notes` | str | Free-text caveats (default mentions NVML accuracy +/-5 %, thermal drift) |

### Aggregate metrics

These are the totals across all processes / GPUs (post-aggregation, post-warmup-exclusion when applicable).

| Field | Type | Description |
|-------|------|-------------|
| `total_tokens` | int | Total output tokens generated across all prompts |
| `total_energy_j` | float | Total GPU energy in joules (raw, no baseline subtraction) |
| `total_inference_time_sec` | float | Total wall-clock inference time |
| `avg_tokens_per_second` | float | Throughput: `total_tokens / total_inference_time_sec` |
| `avg_energy_per_token_j` | float | Energy per output token in joules |

### Per-token energy (millijoules)

| Field | Type | Description |
|-------|------|-------------|
| `mj_per_tok_total` | float &#124; null | Millijoules per token from raw (unadjusted) energy |
| `mj_per_tok_adjusted` | float &#124; null | Millijoules per token from baseline-adjusted energy. `null` when no baseline was measured. **This is the right field for cross-experiment comparisons.** |

:::note Why adjusted beats total for comparisons
`mj_per_tok_adjusted` subtracts idle GPU power before dividing by token count. Two experiments running on hardware with different idle power (or at different thermal states) will show a spurious difference in `mj_per_tok_total` even when inference is identical. See [Energy measurement](/explanation/methodology/energy-measurement) for the full reasoning.
:::

### FLOPs

`total_flops` is an estimate (not measurable directly during inference). The derived per-token / per-second fields are `null` when the divisor is zero.

| Field | Type | Description |
|-------|------|-------------|
| `total_flops` | float | Total FLOPs estimate for this experiment |
| `flops_per_output_token` | float &#124; null | FLOPs per decode token. `null` if `total_flops = 0` or `output_tokens = 0` |
| `flops_per_input_token` | float &#124; null | FLOPs per prefill token |
| `flops_per_second` | float &#124; null | FLOPs throughput (`total_flops / inference_time_sec`) |

### Baseline (idle GPU power)

| Field | Type | Description |
|-------|------|-------------|
| `baseline_power_w` | float &#124; null | Idle GPU power in watts, measured before this experiment |
| `energy_adjusted_j` | float &#124; null | Total energy minus `baseline_power_w x total_inference_time_sec`. The "net inference work" energy figure. |
| `energy_per_device_j` | list[float] &#124; null | Per-GPU energy breakdown (length = `num_processes`) |

For the methodology that motivates baseline subtraction, see [Methodology &gt; Baseline power](/explanation/methodology/methodology#baseline-power).

### Sidecar reference

| Field | Type | Description |
|-------|------|-------------|
| `timeseries` | str &#124; null | Relative filename of the timeseries sidecar (e.g. `"timeseries.parquet"`); `null` when `output.save_timeseries: false` |

### Effective config (sibling file)

`effective_config.json` lives next to `result.json` in each experiment directory. It contains the fully resolved `ExperimentConfig` - every parameter value used, including engine defaults that were not explicitly specified. **This is what reproduces the experiment.**

## `manifest.json` - study-level checkpoint

Written and updated as a study runs (resume support reads from it). Once the study completes, manifest's `summary` field is essentially the same as the returned `StudyResult.summary`.

### Top-level

| Field | Type | Description |
|-------|------|-------------|
| `study_name` | str &#124; null | Study name (used in directory naming) |
| `study_design_hash` | str &#124; null | 16-char SHA-256 of the resolved experiment list (execution block excluded). Same YAML -> same hash. |
| `start_time` | datetime | Study start (ISO-8601 UTC) |
| `end_time` | datetime | Study end (ISO-8601 UTC, populated on completion) |
| `experiments` | list[dict] | Per-experiment resolved config + status (running &#124; completed &#124; failed) |
| `summary` | `StudySummary` | Aggregate counters (see below) |

### `summary` block

| Field | Type | Description |
|-------|------|-------------|
| `total_experiments` | int | Total experiments planned for this study |
| `completed` | int | Number of successfully completed experiments |
| `failed` | int | Number of failed experiments |
| `total_wall_time_s` | float | Total wall-clock time in seconds |
| `total_energy_j` | float | Total energy across all experiments in joules |
| `unique_configurations` | int &#124; null | Distinct experiment configs: `total_experiments / n_cycles` |
| `warnings` | list[str] | Runtime warnings emitted during the study |

## `timeseries.parquet` - sample-level sidecar

Written when `output.save_timeseries: true` (the default). One Parquet file per experiment, columnar layout, suitable for direct loading into Pandas / Polars / DuckDB.

| Column | Type | Description |
|--------|------|-------------|
| `t` | float64 | Wall-clock seconds since experiment start |
| `gpu_idx` | int32 | GPU device index (0, 1, ...) for multi-GPU runs |
| `power_w` | float64 | Instantaneous GPU power draw in watts |
| `temperature_c` | float64 | GPU temperature in degC |
| `memory_used_mib` | float64 | GPU memory used in MiB |
| `sm_clock_mhz` | float64 | SM clock in MHz (when available) |

LLenergyMeasure polls NVML at 100 ms intervals; thermal-throttle events shorter than the polling interval may be missed - see [Methodology &gt; Known limitations](/explanation/methodology/methodology#known-limitations).

## `StudyResult` - final return value (Python API)

Returned by `run_study(...)`. Distinct from `manifest.json`: this is the fully-assembled object handed back to the caller after the study completes.

| Field | Type | Description |
|-------|------|-------------|
| `experiments` | list[`ExperimentResult`] | One entry per experiment cell (same fields as the per-experiment `result.json`) |
| `study_name` | str &#124; null | Same as manifest |
| `study_design_hash` | str &#124; null | Same as manifest |
| `measurement_protocol` | dict | Flat snapshot of `ExecutionConfig`: `n_cycles`, `experiment_order`, `experiment_gap_seconds`, `cycle_gap_seconds`, `shuffle_seed`, `experiment_timeout_seconds` |
| `result_files` | list[str] | Paths to per-experiment `result.json` files (paths, not embedded payload) |
| `summary` | `StudySummary` | Same shape as in the manifest |
| `skipped_experiments` | list[dict] | Grid points skipped due to validation errors. Each entry: `{raw_config, reason, errors}` |

## Loading from disk

```python
import json
from pathlib import Path

study = Path("results/tutorial-multi-engine_2026-05-07T14-32-08")

# Load study manifest
with (study / "manifest.json").open() as f:
    manifest = json.load(f)

# Load every experiment result
results = []
for cell in sorted(study.glob("*/result.json")):
    with cell.open() as f:
        results.append(json.load(f))

# Load timeseries (Pandas)
import pandas as pd
ts = pd.read_parquet(study / "001_c0_qwen-transformers_a1b2c3" / "timeseries.parquet")
```

For the Python API equivalent (`StudyResult` object), see [Reference &gt; Library API](/reference/api/llenergymeasure).

## Schema versioning

`result.json.schema_version` follows semantic versioning: minor bumps add fields without breaking existing readers, major bumps signal breaking changes. Pre-1.0 the policy is conservative - new fields land as `Optional` with `default = null` so existing parsers don't break.

## See also

- [Tutorial - Multi-engine study](/tutorials/multi-engine-study) walks through writing analysis code against this schema.
- [How to - Interpret results](/how-to/interpret-results) reads a real `result.json` field by field with worked examples.
- [Methodology - Energy measurement](/explanation/methodology/energy-measurement) explains where the numbers come from.
