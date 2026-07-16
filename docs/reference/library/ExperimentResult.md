---
title: ExperimentResult
description: Pydantic model returned by run_experiment - all measurements from one LLM inference run.
---

# `ExperimentResult`

```python
from llenergymeasure import ExperimentResult
```

## Concept

`ExperimentResult` is the data structure returned by [`run_experiment`](./run_experiment) and
contained in [`StudyResult.experiments`](./run_study#returns). It is a frozen Pydantic model
produced once per measurement run, holding all metrics from that run directly.

`ExperimentResult` mirrors the on-disk `result.json` schema closely - the JSON on disk is
produced by `model.model_dump(mode="json")` and shares the same field names and units. See
[Results schema](/reference/results-schema) for the full on-disk layout including
`manifest.json` and `timeseries.parquet`.

`ExperimentResult` is almost always returned by the harness, not constructed by users.

---

## Fields

### Identity

| Field | Type | Description |
|-------|------|-------------|
| `schema_version` | `str` | Result schema version (current: `"4.0"`). |
| `experiment_id` | `str` | Unique identifier for this experiment run. |
| `measurement_config_hash` | `str` | 16-char SHA-256 hex of the `ExperimentConfig` (environment fields excluded). Matches the hash in the result directory name on disk. |
| `llenergymeasure_version` | `str \| None` | Package version that produced this result. |

### Engine and model

| Field | Type | Description |
|-------|------|-------------|
| `engine` | `str` | Inference engine used: `"transformers"`, `"vllm"`, or `"tensorrt"`. |
| `engine_version` | `str \| None` | Engine version string for reproducibility (e.g. `"4.47.0"` for Transformers). |
| `model_name` | `str` | Model name or path used. |

### Measurement methodology

| Field | Type | Description |
|-------|------|-------------|
| `measurement_methodology` | `"total" \| "steady_state" \| "windowed"` | What was measured: the full run, the steady-state window after warmup, or an explicit time window. |
| `steady_state_window` | `tuple[float, float] \| None` | `(start_sec, end_sec)` relative to inference start. The single-process path sets `(0.0, inference_time_sec)`. `None` only when no inference window was recorded. |

### Core metrics

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `total_tokens` | `int` | tokens | Total tokens generated during the run. |
| `total_energy_j` | `float` | joules | Total GPU energy for the run. |
| `energy_adjusted_j` | `float \| None` | joules | Baseline-subtracted energy attributable to inference. `None` when no baseline was taken. |
| `total_inference_time_sec` | `float` | seconds | Wall time for the inference phase. |
| `avg_tokens_per_second` | `float` | tok/s | Throughput. |
| `avg_energy_per_token_j` | `float` | J/tok | Mean energy per token. |
| `mj_per_tok_total` | `float \| None` | mJ/tok | Millijoules per token from total (unadjusted) energy. |
| `mj_per_tok_adjusted` | `float \| None` | mJ/tok | Millijoules per token from baseline-adjusted energy. `None` when `energy_adjusted_j` is `None`. |

### FLOPs metrics

| Field | Type | Description |
|-------|------|-------------|
| `total_flops` | `float` | Estimated FLOPs. Derived from model config (reference metadata, not measured). |
| `flops_per_output_token` | `float \| None` | FLOPs per decode token. `None` when `total_flops=0` or `output_tokens=0`. |
| `flops_per_input_token` | `float \| None` | FLOPs per prefill token. `None` when `total_flops=0` or `input_tokens=0`. |
| `flops_per_second` | `float \| None` | FLOPs throughput (`total_flops / inference_time_sec`). `None` when `time=0` or `flops=0`. |

### Energy detail

| Field | Type | Description |
|-------|------|-------------|
| `baseline_power_w` | `float \| None` | Idle GPU power in watts measured before the experiment. `None` when baseline measurement is disabled. |
| `energy_per_device_j` | `list[float] \| None` | Per-GPU energy breakdown. Currently populated by the Zeus sampler only. `None` for NVML and CodeCarbon. |
| `energy_breakdown` | `EnergyBreakdown \| None` | Detailed breakdown with baseline adjustment intervals. |

### Multi-GPU

| Field | Type | Description |
|-------|------|-------------|
| `multi_gpu` | `MultiGPUMetrics \| None` | Multi-GPU aggregate metrics. `None` for single-GPU experiments. |
| `aggregation` | `AggregationMetadata \| None` | Aggregation method and quality flags. |

### Quality and reproducibility

| Field | Type | Description |
|-------|------|-------------|
| `measurement_warnings` | `list[str]` | Quality warnings (e.g. short duration, thermal drift detected). |
| `warmup_excluded_samples` | `int \| None` | Warmup iterations run before the measurement window (from `warmup_result.iterations_completed`). `None` when no warmup result is available. |
| `reproducibility_notes` | `str` | Fixed disclaimer about NVML measurement accuracy (+/- 5%). |
| `thermal_throttle` | `ThermalThrottleInfo \| None` | GPU thermal and power throttle events during the run. |
| `warmup_result` | `WarmupResult \| None` | Warmup convergence result (populated when CV convergence detection is enabled). |

### Timing

| Field | Type | Description |
|-------|------|-------------|
| `start_time` | `datetime` | Earliest process start time (UTC). |
| `end_time` | `datetime` | Latest process end time (UTC). |

### Sidecar

| Field | Type | Description |
|-------|------|-------------|
| `timeseries` | `str \| None` | Relative filename of the timeseries Parquet sidecar (e.g. `"timeseries.parquet"`). `None` when timeseries saving is disabled. |
| `latency_stats` | `LatencyStatistics \| None` | TTFT/ITL statistics. `None` when latency capture did not run. vLLM records TTFT on every run; transformers and tensorrt populate stats only under `measurement.latency_profiling`. |
| `extended_metrics` | `ExtendedEfficiencyMetrics \| None` | Extended efficiency metrics (TPOT, memory, GPU utilisation). Always present when the harness runs successfully; fields within are `None` when not computable. |

---

## Properties

`ExperimentResult` exposes two computed properties:

| Property | Type | Description |
|----------|------|-------------|
| `duration_sec` | `float` | Total experiment duration (`end_time - start_time`). |
| `tokens_per_joule` | `float` | Overall energy efficiency (`total_tokens / total_energy_j`). `0.0` when `total_energy_j` is zero. |

---

## Common patterns

### Extract the headline efficiency metrics

```python
result = run_experiment(model="gpt2", engine="transformers")

print(f"Energy (total):    {result.total_energy_j:.2f} J")
print(f"Energy (adjusted): {result.energy_adjusted_j or 'N/A'}")
print(f"mJ/tok (total):    {result.mj_per_tok_total:.3f}")
print(f"mJ/tok (adjusted): {result.mj_per_tok_adjusted or 'N/A'}")
print(f"Throughput:        {result.avg_tokens_per_second:.1f} tok/s")
print(f"FLOPs/s:           {result.flops_per_second or 'N/A'}")
```

### Compare two results

```python
a = run_experiment(model="gpt2", engine="transformers")
b = run_experiment(model="gpt2-medium", engine="transformers")

ratio = b.mj_per_tok_total / a.mj_per_tok_total
print(f"gpt2-medium is {ratio:.2f}x more expensive per token than gpt2")
```

### Serialise to JSON

```python
import json

with open("result.json", "w") as f:
    json.dump(result.model_dump(mode="json"), f, indent=2, default=str)
```

The on-disk `result.json` written by `run_experiment` / `run_study` uses this same
serialisation. Loading it back:

```python
data = json.loads(Path("results/study_name/001_c0_.../result.json").read_text())
loaded = ExperimentResult(**data)
```

### Check for quality warnings

```python
if result.measurement_warnings:
    for w in result.measurement_warnings:
        print(f"Warning: {w}")

if result.thermal_throttle and result.thermal_throttle.throttle_detected:
    print("Thermal throttling detected - results may be unreliable")
```

---

## Pitfalls

**`energy_adjusted_j` and `mj_per_tok_adjusted` are `None` when baseline is disabled.**
If `measurement.baseline.enabled=False`, neither field is populated. Always guard with
`if result.energy_adjusted_j is not None` before using them.

**`energy_per_device_j` is only populated by the Zeus sampler.** With `energy_sampler="nvml"`
or `energy_sampler="codecarbon"` (or `"auto"` resolving to either), `energy_per_device_j`
is `None`; only the run-level `total_energy_j` is available.

**`extended_metrics` fields can be `None` within an otherwise-present object.** The
`ExtendedEfficiencyMetrics` object is always attached but individual sub-fields (e.g. TTFT,
memory bandwidth) are `None` when the data required to compute them was not available
(non-streaming inference, no memory bandwidth counters, etc.).

**`flops_per_*` are reference estimates, not measured values.** FLOPs are estimated from model
config (parameter count and sequence lengths) via `AutoConfig`, not from hardware counters.
They are useful for relative comparisons but not for absolute roofline analysis.

**Frozen model - no mutation.** `ExperimentResult` has `frozen=True`. Attempting to set a
field raises `ValidationError`. Use `model_copy(update=...)` to derive a modified copy.

---

## See also

- [`run_experiment`](./run_experiment) - the function that returns an `ExperimentResult`
- [`run_study`](./run_study) - returns `StudyResult` containing `list[ExperimentResult]`
- [Results schema](/reference/results-schema) - the on-disk `result.json` schema this mirrors
