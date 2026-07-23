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
    │   ├── result.json                          # measurement metrics (bundle_version 2.0)
    │   ├── config.json                          # engine/model/methodology + resolved config + provenance
    │   ├── environment.json                     # hardware/runtime snapshot + runner provenance
    │   └── timeseries.parquet                   # GPU power/thermal/memory samples
    ├── 002_c0_.../
    ├── ...
    └── _study-artefacts/
        ├── equivalence_groups.json              # dedup equivalence groups
        └── baseline_cache_<key>.json            # per-engine baseline cache
```

`<UTC-timestamp>` is ISO-8601 (e.g. `2026-05-07T14-32-08`). Cell directory names encode `<NNN>_c<cycle>_<model>-<engine>_<config-hash>` so they sort sensibly and you can tell sibling cycles apart at a glance.

## `result.json` - per-experiment record

The scientific record. One JSON file per experiment cell. Stamped with `bundle_version` `"2.0"` (the single version shared across every artefact in the bundle).

`result.json` is measurement output. Configuration inputs and methodology - `engine_version`, `measurement_methodology`, `steady_state_window`, `measurement_window_discard_fraction`, `steady_state_not_detected` - live in the `config.json` sidecar, not here. The one deliberate duplication: `result.json` keeps `model_name` and `engine` as convenience copies so a result file is self-describing when separated from its directory; the authoritative home for both is `config.json`.

### Identification

| Field | Type | Description |
|-------|------|-------------|
| `bundle_version` | str | Results-bundle version (currently `"2.0"`), shared across `result.json`, `config.json`, and `environment.json` as one contract |
| `experiment_id` | str | Unique experiment identifier (`{model}_{YYYYMMDD_HHMMSS}` for single experiments; study-level cells inherit a richer per-cell identifier) |
| `measurement_config_hash` | str | SHA-256[:16] of `ExperimentConfig` with environment fields excluded; same hash -> logically identical experiments |
| `llenergymeasure_version` | str &#124; null | Package version that produced this result |
| `serving_mode` | str | The serving mode that produced this result: the offline/server discriminator, mirroring the config-side `ExperimentConfig.serving_mode`. `"offline"` for batch measurement (the only mode today); server mode (v0.8.0) will stamp `"server"`. A plain string, not a closed vocabulary |
| `engine` | str | Inference engine used. Convenience copy; authoritative home is the `config.json` sidecar |
| `model_name` | str | Model name/path measured. Convenience copy; authoritative home is the `config.json` sidecar |

### Measurement methodology

| Field | Type | Description |
|-------|------|-------------|
| `warmup_excluded_samples` | int &#124; null | Number of warmup iterations run before the measurement window (from `warmup_result.iterations_completed`); `null` when no warmup result is available |
| `model_load_time_sec` | float &#124; null | Wall-clock seconds spent in `engine.load_model()`: model load plus any engine build/compile performed there (e.g. the tensorrt trt backend's TRT engine build, vLLM torch.compile / CUDA-graph capture). Non-energy metadata: this phase completes before the energy measurement window opens and contributes nothing to `total_energy_j` |
| `engine_build_cache_hit` | bool &#124; null | Whether the tensorrt trt-backend engine build was served from the on-disk build cache (`true`) or compiled fresh (`false`). `null` when the build cache is not in play: the pytorch backend, other engines, an `engine_path` override, or the cache disabled. Annotates `model_load_time_sec` (a cache hit skips the compile) |
| `reproducibility_notes` | str | Free-text caveats (default mentions NVML accuracy +/-5 %, thermal drift) |

### Aggregate metrics

These are the run totals (post-warmup-exclusion when applicable).

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | int | Actual input (prefill) tokens observed by the engine after tokenisation (`total_tokens = input_tokens + output_tokens`) |
| `output_tokens` | int | Actual output (decode) tokens observed by the engine (`total_tokens = input_tokens + output_tokens`) |
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
| `energy_breakdown.baseline_power_w` | float &#124; null | Idle GPU power in watts, measured before this experiment. The single home for the baseline power reading (the former top-level `baseline_power_w` copy was retired in bundle 2.0) |
| `energy_adjusted_j` | float &#124; null | Total energy minus `energy_breakdown.baseline_power_w x total_inference_time_sec`. The "net inference work" energy figure. |
| `energy_per_device_j` | list[float] &#124; null | Per-GPU energy breakdown (length = `num_processes`) |

For the methodology that motivates baseline subtraction, see [Methodology &gt; Baseline power](/explanation/methodology/methodology#baseline-power).

### Extended efficiency metrics

`extended_metrics` is a nested object with five always-present sub-objects
(`memory`, `gpu_utilisation`, `batch`, `kv_cache`, `request_latency`) plus two
scalars (`tpot_ms`, `token_efficiency_index`). Every leaf is `null` when it
cannot be computed for the engine/run; the harness fills what each engine can
provide. `latency_stats` and `warmup_excluded_samples` live at the top level of
`result.json`; the measurement window itself (`steady_state_window`) lives in the
`config.json` sidecar.

| Field | Type | Description |
|-------|------|-------------|
| `extended_metrics.tpot_ms` | float &#124; null | Time per output token (ITL mean). Populated only when `measurement.latency_profiling=true` (transformers via streamer; vLLM via decode-average ITL); `null` otherwise. |
| `extended_metrics.token_efficiency_index` | float &#124; null | Composite `throughput x tokens_per_joule x precision_factor`. |
| `extended_metrics.memory.model_memory_utilisation` | float &#124; null | Model weights / total VRAM (0-1). |
| `extended_metrics.memory.tokens_per_gb_vram` | float &#124; null | Output tokens per GB of peak VRAM. |
| `extended_metrics.memory.kv_cache_mb` / `kv_cache_memory_ratio` | float &#124; null | KV-cache size and its share of peak memory (vLLM only, when exposed). |
| `extended_metrics.gpu_utilisation.sm_utilisation_mean` | float &#124; null | Mean SM utilisation (0-100) over NVML samples. |
| `extended_metrics.gpu_utilisation.memory_bandwidth_utilisation` | float &#124; null | Mean memory-controller activity (0-100). NVML proxy: percent of time a read/write was issued, **not** achieved bandwidth. |
| `extended_metrics.batch.num_batches` / `effective_batch_size` / `batch_utilisation` / `padding_overhead` | int/float &#124; null | Static-batching efficiency. `null` for vLLM (continuous batching). |
| `extended_metrics.kv_cache.*` | float/int &#124; null | Prefix-cache hit rate and block occupancy (vLLM only). |
| `extended_metrics.request_latency.e2e_latency_{mean,median,p95,p99}_ms` | float &#124; null | Per-request end-to-end latency distribution. |
| `latency_stats` | object &#124; null | TTFT/ITL statistics. vLLM populates per-request TTFT/E2E plus a decode-average ITL only under `measurement.latency_profiling=true`, in `proportional` mode (V1 records `RequestOutput.metrics` only when `disable_log_stats=False`, which profiling sets); `null` otherwise. transformers populates `latency_stats` (TTFT + ITL) only under profiling. tensorrt populates per-request TTFT/E2E plus an average-TPOT-derived ITL only under profiling, in `per_request_batch` mode; `null` otherwise. |
| `latency_stats.measurement_mode` | str | Provenance of the latency capture: `true_streaming` (real per-token timestamps, transformers under profiling), `proportional` (decode-average ITL estimate, vLLM under profiling), or `per_request_batch` (tensorrt). The mode reflects the weakest signal present. |

#### Per-engine support matrix

A check means the engine populates the field in the single-process path; a dash
means it stays `null` for that engine.

| Metric group | vLLM | transformers | tensorrt |
|--------------|:----:|:------------:|:--------:|
| `request_latency.*` (per-request E2E) | profiling only (from `RequestOutput.metrics` at 0.19.1) | yes (per-batch approximation) | profiling only (from `RequestOutput.metrics_dict` at 1.2.1) |
| `latency_stats` TTFT | profiling only | profiling only | profiling only |
| `latency_stats` ITL / `tpot_ms` | profiling only (`proportional`) | profiling only (`true_streaming`) | profiling only (`per_request_batch`) |
| `kv_cache.*` | yes (best-effort) | dash | dash |
| `gpu_utilisation.*` (SM + mem-bw) | yes | yes | yes |
| `memory.*` ratios | yes | yes | yes |
| `batch.*` (num_batches/padding/utilisation) | dash (continuous batching) | yes | `num_batches=1` only; padding/utilisation dash |

**Latency profiling is opt-in.** Set `measurement.latency_profiling: true` to
capture inter-token latency (and hence `tpot_ms`). Per-engine semantics:

- **transformers**: a custom generation streamer records true per-token arrival
  times. Profiling forces `batch_size=1` (one streamed token maps to one
  request) and is incompatible with beam search (`num_beams > 1` falls back to
  the non-profiled path). Mode = `true_streaming`. With profiling off,
  `latency_stats` is `null`.
- **vLLM**: under profiling the plugin builds the engine with
  `disable_log_stats=False` so vLLM records per-request timing in
  `RequestOutput.metrics` (a V1 `RequestStateStats`; `metrics` is `None` on the
  default offline path, which forces `disable_log_stats=True`). TTFT is the
  engine-recorded `first_token_latency`; E2E is TTFT plus the monotonic decode
  interval (`last_token_ts - first_token_ts`); a decode-average ITL is derived
  from that interval over the longest output's tokens. Because the ITL averages
  over the decode phase rather than timing each token, the mode is
  `proportional`. With profiling off, `latency_stats` is `null`.
- **tensorrt**: under profiling the plugin sets
  `SamplingParams(return_perf_metrics=True)` and extracts per-request TTFT / E2E
  / average TPOT from `RequestOutput.metrics_dict` (the TRT-LLM 1.x surface,
  live-verified on both backends at 1.2.1), in mode `per_request_batch`. With
  profiling off, `latency_stats` is `null`. The `latency_profiling_unsupported`
  warning now fires only when profiling was requested but the engine returned no
  metrics.

**Energy caveat.** Per-token timing capture adds overhead that can perturb both
energy and latency. Energy figures from a profiled run are emitted as-is and are
**not** directly comparable to non-profiled runs; every profiled run records a
disclaimer in `measurement_warnings` (the flag is also part of the config hash,
so profiled and non-profiled runs are distinct experiments).

**transformers non-profiled latency is approximated.** Without profiling, a
non-streaming `generate()` only exposes per-batch wall time, so each prompt in a
batch is attributed `batch_time / batch_size` (the `PER_REQUEST_BATCH` mode in
`request_latency`). This is an estimate, not a true per-request timestamp.

### Sidecar reference

| Field | Type | Description |
|-------|------|-------------|
| `timeseries` | str &#124; null | Relative filename of the timeseries sidecar (e.g. `"timeseries.parquet"`); `null` when `output.save_timeseries: false` |

### Config sidecar (sibling file)

`config.json` lives next to `result.json` in each experiment directory. It is the authoritative home of engine/model/methodology identity (`engine`, `engine_version`, `model_name`, `measurement_methodology`; `result.json` carries convenience copies of `engine` and `model_name` only) and carries the full user-declared `ExperimentConfig` under `declared_config` - every parameter value used, including engine defaults that were not explicitly specified - plus per-field `provenance` (where each non-default value came from: CLI flag, sweep, or YAML) and the observed post-construction engine state. **This is what reproduces the experiment.**

### Environment sidecar (sibling file)

`environment.json` lives next to `result.json` and records the hardware/runtime environment the experiment ran in (stamped with the shared `bundle_version` `2.0`): GPU, CUDA, driver, CPU, platform, Python and tool versions. Under Docker dispatch it captures the *in-container* environment, not the dispatching host.

Two distinct CUDA facts are recorded and must not be conflated: `hardware.cuda.driver_supported_version` is the maximum CUDA version the installed NVIDIA driver supports (from NVML, matching the `CUDA Version` in the `nvidia-smi` header), while the top-level `cuda_version` (with `cuda_version_source`) is the runtime CUDA version the software stack was actually built against (torch / version.txt / nvcc).

It also carries a `runner` block - the reproducibility anchor for cross-run comparability. It is the same unified `RunnerProvenance` model `result.json` carries as `runner_provenance`, serialised into both artefacts:

| Field | Type | Meaning |
| --- | --- | --- |
| `mode` | str | `"container"` (containerized) or `"process"` (host process) - a first-order variable for energy/latency comparability. Legacy bundles' `"docker"`/`"local"` are read as `"container"`/`"process"` |
| `image` | str &#124; null | Docker image reference that ran (`null` for process mode) |
| `source` | str | Which precedence layer selected the runner (`env`, `yaml`, `user_config`, `auto_detected`, `default`, `multi_engine_elevation`, or `local`) |
| `image_source` | str &#124; null | Where the Docker image was resolved from (`null` for process runs or when unresolved) |
| `image_digest` | str &#124; null | Resolved registry digest (`repo@sha256:...`) pinning the full stack (base image, CUDA, torch, patches). `null` for process runs, and for locally-built images with no registry digest |

Sidecars written before this block existed load with `runner: null`.

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

## Bundle versioning

The whole bundle carries one `bundle_version` (currently `"2.0"`), stamped identically into `result.json`, `config.json`, and `environment.json` (`timeseries.parquet` is self-describing columnar data and stays unversioned). It versions the on-disk layout, the artefact set, and each artefact's schema as a single contract, so there is one number to bump and one changelog line per documented break. This replaces the three independent per-artefact `schema_version` counters that earlier releases carried. It follows semantic versioning: minor bumps add fields without breaking existing readers, major bumps signal breaking changes. Pre-1.0 the policy is conservative - new fields land as `Optional` with `default = null` so existing parsers don't break.

Older bundles remain readable best-effort: the canonical reader (`llenergymeasure.results.bundle.BundleReader`, wrapped by `load_result`) tolerates a legacy shape rather than rejecting it, emitting a single warning per bundle. A bundle 1.0 (the provenance-unification break) reads with its retired top-level `baseline_power_w` copy and `schema_version` key dropped on load, its pre-rename `hardware.cuda.version` mapped onto `driver_supported_version`, its never-populated hardware fields (`pcie_gen`, `mig_enabled`, `cudnn_version`, `fan_speed_pct`) ignored, and its old separate runner block read into the unified `RunnerProvenance` model. There is no in-place migration and no converter tooling.

## See also

- [Tutorial - Multi-engine study](/tutorials/multi-engine-study) walks through writing analysis code against this schema.
- [How to - Interpret results](/how-to/interpret-results) reads a real `result.json` field by field with worked examples.
- [Methodology - Energy measurement](/explanation/methodology/energy-measurement) explains where the numbers come from.
