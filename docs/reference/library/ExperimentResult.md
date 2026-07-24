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
| `bundle_version` | `str` | Results-bundle version (current: `"2.0"`), shared across `result.json`, `config.json`, and `system.json` as one contract. |
| `experiment_id` | `str` | Unique identifier for this experiment run. |
| `declared_config_hash` | `str` | 16-char SHA-256 hex of the `ExperimentConfig` (environment fields excluded). Matches the hash in the result directory name on disk. |
| `llenergymeasure_version` | `str \| None` | Package version that produced this result. |
| `serving_mode` | `str` | The serving mode that produced this result: the offline/server discriminator, mirroring the config-side `ExperimentConfig.serving_mode`. `"offline"` for batch measurement (the only mode today); `"server"` arrives with server mode (v0.8.0). A plain string, not a closed vocabulary. |
| `engine` | `str` | Inference engine used. Convenience copy; authoritative home is the `config.json` sidecar. |
| `model_name` | `str` | Model name/path measured. Convenience copy; authoritative home is the `config.json` sidecar. |

### Engine, model, and methodology

`ExperimentResult` is measurement output. `engine_version` and the measurement-methodology fields (`measurement_methodology`, `steady_state_window`, `measurement_window_discard_fraction`, `steady_state_not_detected`) are configuration inputs and live in the `config.json` sidecar next to `result.json`, not on this model. The model keeps `model_name` and `engine` as convenience copies so a result stays self-describing when separated from its directory; the authoritative home for both is `config.json`.

### Core metrics

| Field | Type | Units | Description |
|-------|------|-------|-------------|
| `input_tokens` | `int` | tokens | Actual input (prefill) tokens observed by the engine after tokenisation. `total_tokens = input_tokens + output_tokens`. |
| `output_tokens` | `int` | tokens | Actual output (decode) tokens observed by the engine. `total_tokens = input_tokens + output_tokens`. |
| `total_tokens` | `int` | tokens | Total tokens generated during the run. |
| `total_energy_j` | `float` | joules | Total GPU energy for the run. |
| `energy_adjusted_j` | `float \| None` | joules | Baseline-subtracted energy attributable to inference. `None` when no baseline was taken. |
| `total_inference_time_sec` | `float` | seconds | Wall time for the inference phase. |
| `avg_tokens_per_second` | `float` | tok/s | Throughput. |
| `avg_energy_per_token_j` | `float` | J/tok | Mean energy per token. |
| `energy_per_token_mj_total` | `float \| None` | mJ/tok | Millijoules per token from total (unadjusted) energy. |
| `energy_per_token_mj_adjusted` | `float \| None` | mJ/tok | Millijoules per token from baseline-adjusted energy. `None` when `energy_adjusted_j` is `None`. |

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
| `energy_per_device_j` | `list[float] \| None` | Per-GPU energy breakdown. Currently populated by the Zeus sampler only. `None` for NVML and CodeCarbon. |
| `energy_breakdown` | `EnergyBreakdown \| None` | Detailed breakdown with baseline adjustment intervals. Carries `baseline_power_w` (idle GPU power in watts), the single home for the baseline reading since bundle 2.0 retired the former top-level `baseline_power_w` copy. |

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
| `throttle` | `ThrottleInfo \| None` | GPU throttling during the run, symmetric across a `thermal` and a `power` axis (each a `ThrottleAxis` with `any` / `hw` / `sw` flags). |
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
| `latency_stats` | `LatencyStatistics \| None` | TTFT/ITL statistics. `None` when latency capture did not run. All engines populate stats only under `measurement.latency_profiling` (vLLM V1 records `RequestOutput.metrics` only when `disable_log_stats=False`, which profiling sets). |
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
print(f"mJ/tok (total):    {result.energy_per_token_mj_total:.3f}")
print(f"mJ/tok (adjusted): {result.energy_per_token_mj_adjusted or 'N/A'}")
print(f"Throughput:        {result.avg_tokens_per_second:.1f} tok/s")
print(f"FLOPs/s:           {result.flops_per_second or 'N/A'}")
```

### Compare two results

```python
a = run_experiment(model="gpt2", engine="transformers")
b = run_experiment(model="gpt2-medium", engine="transformers")

ratio = b.energy_per_token_mj_total / a.energy_per_token_mj_total
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

if result.throttle and result.throttle.detected:
    print("Thermal or power throttling detected - results may be unreliable")
```

---

## Pitfalls

**`energy_adjusted_j` and `energy_per_token_mj_adjusted` are `None` when baseline is disabled.**
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
