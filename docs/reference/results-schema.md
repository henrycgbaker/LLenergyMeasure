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
    │   ├── system.json                          # hardware/runtime snapshot + runner provenance
    │   └── timeseries.parquet                   # GPU power/thermal/memory samples
    ├── 002_c0_.../
    ├── ...
    └── _study-artefacts/
        ├── equivalence_groups.json              # dedup equivalence groups
        └── baseline_cache_<key>.json            # per-engine baseline cache
```

`<UTC-timestamp>` is ISO-8601 (e.g. `2026-05-07T14-32-08`). The timestamp has one-second resolution, so a study started in the same second as another takes the next free numbered sibling instead (`<study-name>_<UTC-timestamp>-2`); two studies never share one directory. Cell directory names encode `<NNN>_c<cycle>_<model>-<engine>_<config-hash>` so they sort sensibly and you can tell sibling cycles apart at a glance.

## `result.json` - per-experiment record

The scientific record. One JSON file per experiment cell. Stamped with `bundle_version` `"2.0"` (the single version shared across every artefact in the bundle).

`result.json` is measurement output. Configuration inputs and methodology - `engine_version`, `measurement_methodology`, `steady_state_window`, `measurement_window_discard_fraction`, `steady_state_not_detected` - live in the `config.json` sidecar, not here. The one deliberate duplication: `result.json` keeps `model_name` and `engine` as convenience copies so a result file is self-describing when separated from its directory; the authoritative home for both is `config.json`.

### Identification

| Field | Type | Description |
|-------|------|-------------|
| `bundle_version` | str | Results-bundle version (currently `"2.0"`), shared across `result.json`, `config.json`, and `system.json` as one contract |
| `experiment_id` | str | Unique experiment identifier (`{model}_{YYYYMMDD_HHMMSS}` for single experiments; study-level cells inherit a richer per-cell identifier) |
| `declared_config_hash` | str | SHA-256[:16] of `ExperimentConfig` with environment fields excluded; same hash -> logically identical experiments |
| `llenergymeasure_version` | str &#124; null | Package version that produced this result |
| `serving_mode` | str | The serving mode that produced this result: the offline/server discriminator, mirroring the config-side `ExperimentConfig.serving_mode`. `"offline"` for batch measurement; `"server"` for a server-mode measurement window (v0.7). A plain string, not a closed vocabulary. See [Server-mode results](#server-mode-results) for the server surfaces |
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
| `energy_per_token_mj_total` | float &#124; null | Millijoules per token from raw (unadjusted) energy |
| `energy_per_token_mj_adjusted` | float &#124; null | Millijoules per token from baseline-adjusted energy. `null` when no baseline was measured. **This is the right field for cross-experiment comparisons.** |

:::note Why adjusted beats total for comparisons
`energy_per_token_mj_adjusted` subtracts idle GPU power before dividing by token count. Two experiments running on hardware with different idle power (or at different thermal states) will show a spurious difference in `energy_per_token_mj_total` even when inference is identical. See [Energy measurement](/explanation/methodology/energy-measurement) for the full reasoning.
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

### Throttle indicators

`throttle` (renamed from the earlier flat `thermal_throttle`) is a symmetric object over the two throttling axes NVML reports. Each axis carries a `hw`/`sw` cause split plus a combined `any` flag, so `throttle.thermal.any` and `throttle.power.any` are the two top-level "did this axis throttle" questions. The former flat `.power` field reflected only the software power cap; `throttle.power.any` is the previously-missing combined power indicator. Any throttling can invalidate energy and performance measurements. `null` when no throttle sampling was performed.

| Field | Type | Description |
|-------|------|-------------|
| `throttle.thermal` | `ThrottleAxis` | Thermal throttling axis (hardware/software thermal slowdown) |
| `throttle.power` | `ThrottleAxis` | Power throttling axis (hardware power brake / software power cap) |
| `throttle.<axis>.hw` | bool | Hardware slowdown on this axis was seen |
| `throttle.<axis>.sw` | bool | Software slowdown on this axis was seen |
| `throttle.<axis>.any` | bool | Computed: `hw or sw` (either cause seen on this axis) |
| `throttle.detected` | bool | Computed: `thermal.any or power.any` (any throttling occurred) |
| `throttle.throttle_duration_sec` | float | Estimated total duration of throttling in seconds |
| `throttle.max_temperature_c` | float &#124; null | Peak GPU temperature during the experiment in Celsius |
| `throttle.throttle_timestamps` | list[float] | Seconds-from-start timestamps at which a throttle was detected |

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

### System sidecar (sibling file)

`system.json` lives next to `result.json` and records the hardware/runtime environment the experiment ran in (stamped with the shared `bundle_version` `2.0`): GPU, CUDA, driver, CPU, platform, Python and tool versions. Under Docker dispatch it captures the *in-container* environment, not the dispatching host. The rename from the earlier `environment.json` is a clean break: a sidecar written under the old name is not read - the reader looks only for `system.json`, so on a pre-rename bundle the system sidecar is simply treated as absent.

Two distinct CUDA facts are recorded and must not be conflated: `hardware.cuda.driver_supported_version` is the maximum CUDA version the installed NVIDIA driver supports (from NVML, matching the `CUDA Version` in the `nvidia-smi` header), while the top-level `cuda_version` (with `cuda_version_source`) is the runtime CUDA version the software stack was actually built against (torch / version.txt / nvcc).

It also carries a `runner` block - the reproducibility anchor for cross-run comparability. It is the same unified `RunnerProvenance` model `result.json` carries as `runner_provenance`, serialised into both artefacts:

| Field | Type | Meaning |
| --- | --- | --- |
| `mode` | str | `"container"` (containerized) or `"process"` (host process) - a first-order variable for energy/latency comparability. A closed vocabulary renamed from `"docker"`/`"local"` in v0.7 (clean break - a pre-v0.7 value fails validation loudly on read) |
| `image` | str &#124; null | Docker image reference that ran (`null` for process mode) |
| `source` | str | Which precedence layer selected the runner (`env`, `yaml`, `user_config`, `auto_detected`, `default`, `multi_engine_elevation`, or `implicit` when no spec was resolved). The no-spec sentinel was renamed from `local` in v0.7 (clean break - no read-time translation) |
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

## Server-mode results

A study run with `serving_mode: server` writes the same per-cell artefacts described above, plus a per-window request log, and it produces **one result bundle per measurement window** rather than one per experiment. This section documents the server-specific surfaces. For the measurement model behind them see [Methodology &gt; Server-mode measurement](/explanation/methodology/methodology#server-mode-measurement).

### Per-window bundles

A server *session* is one server lifetime that measures several windows (each rate level runs three windows by default), and each window is written as its own result bundle. Each rate cell contributes one result per measurement window; under the default sequential order a session covers one cell, while a grouped (interleaved) session folds a rate sweep's cells into one lifetime. The layout below is a sequential-order study (one session per rate):

```
results/
└── <study-name>_<UTC-timestamp>/
    ├── manifest.json
    ├── 001_c0_<model>-<engine>_<hashA>/         # rate A, window 0
    │   ├── result.json                          # window metrics (serving_mode "server")
    │   ├── config.json                          # resolved config + provenance
    │   ├── system.json                          # environment + session facts
    │   └── requests.parquet                     # per-window request log
    ├── 001_c0_<model>-<engine>_<hashA>_1/       # rate A, window 1 (collision suffix)
    ├── 001_c0_<model>-<engine>_<hashA>_2/       # rate A, window 2
    ├── 002_c0_<model>-<engine>_<hashB>/         # rate B, window 0
    └── ...
```

A cell's window bundles share the cell's index prefix and are disambiguated by a numeric collision suffix (`_1`, `_2`); the config hash differs per rate, not per window; under a grouped (interleaved) session every window bundle shares one index prefix. So a sweep of three rates measuring three windows each produces nine bundles from three study cells. The `manifest.json` still tracks one entry per **cell** (grid point), reflecting that cell's rate level outcome; the per-window `ExperimentResult` objects are what flow into `StudyResult.experiments`. If a grouped server session is fully invalid (a warmup abort, or no valid window), it is counted as its full cell count of failures rather than a single failure, so the manifest's `total`/`failed` accounting stays correct.

A window `result.json` carries `serving_mode: "server"`, the shared energy metrics, and a distinguishing convention on token counts:

| Field | Type | Description |
|-------|------|-------------|
| `output_tokens` | int | Client-side canonical output-token count: streamed content deltas received in the steady-state span. This is the J/token denominator |
| `input_tokens` | int &#124; null | `null` in server mode. Client-side input-token counting is post-v0.7; the engine's self-reported prompt tokens ride in `requests.parquet` only |
| `total_tokens` | int &#124; null | `null` in server mode (undefined while `input_tokens` is uncounted) |

### Server provenance (`server` block)

The window `result.json` carries a `server` block that locates the window within its rate level (`level_index`, `window_index`, `level_valid`) and records its warmup and attribution disclosures. It also discloses whether the concurrency cap materially shaped the load:

| Field | Type | Description |
|-------|------|-------------|
| `cap_bound_fraction` | float &#124; null | Fraction of the level's scheduled issuances the concurrency cap delayed beyond a small tolerance (or never dispatched); `0.0` when the level ran uncapped or the cap never materially bound, `null` when it was not captured (the level aborted before its issuer report was recorded). Level-wide, so every window of a level carries the same value. The cap stays legal (a hashed user choice); this stamps its effect |

### Session facts (`system.json`)

Each server-mode bundle's `system.json` carries a `session` block recording the raw per-phase quantities of the server lifetime the window belongs to:

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Identifier shared by every window bundle of one server lifetime |
| `window_count` | int | Total windows measured in this session |
| `level_count` | int &#124; null | Rate levels in this session (`null` for offline, which degenerates to one window) |
| `launch_duration_s` / `launch_energy_j` | float &#124; null | Launch-to-ready phase (model load rides inside it) |
| `warmup_total_duration_s` / `warmup_total_energy_j` | float &#124; null | Summed warmup across the session's levels |
| `drain_duration_s` / `drain_energy_j` | float &#124; null | Post-window drain (see the asymmetry note below) |

The block holds raw quantities only. A phase whose energy could not be measured stamps `null`, never `0.0`, so a null reads as "unmeasured", not "zero joules".

:::note In-memory vs on-disk drain asymmetry (by design)
A window bundle is written at level close carrying a **preliminary** session block with the drain fields `null` (the drain has not happened yet). On a clean session close the drain raws become known and are patched into each already-written on-disk bundle, so the on-disk `drain_duration_s` / `drain_energy_j` are populated. The in-memory `ExperimentResult.session` returned to a Python caller keeps the preliminary drain-`null` block: the drain is a post-return, session-close measurement. The on-disk bundles are the system of record. On the interrupt path the drain is never measured, so the drain fields stay `null` on disk too.
:::

### `requests.parquet` - per-window request log

One Parquet file per window, one row per issued request. Every column states a physical fact for every row regardless of terminal status (the raw-record discipline): an `error` or `timeout` row still reports the real receipts, first-token latency, and to-failure latency it observed. Filtering failed requests out of latency percentiles is the consumer's job, keyed on `status`.

| Column | Type | Description |
|--------|------|-------------|
| `request_index` | int64 | Ordinal index of the request within the window |
| `issued_at` | float64 | Monotonic-seconds time the request was issued (the ideal scheduled time; the latency anchor) |
| `dispatched_at` | float64 &#124; null | Monotonic-seconds time the request was dispatched to the transport |
| `first_token_at` | float64 &#124; null | Monotonic-seconds first-token receipt. Real whenever a token physically arrived, regardless of status (`== output_token_times[0]`) |
| `completed_at` | float64 &#124; null | Monotonic-seconds time-to-terminal. An `error` row carries its failure time; a `timeout` row never completed, so `null` |
| `ttft_ms` | float64 &#124; null | Time-to-first-token latency in ms |
| `e2e_latency_ms` | float64 &#124; null | End-to-end latency in ms (time-to-terminal, including to-failure latency for an `error` row) |
| `client_output_tokens` | int64 | Client-counted output tokens (length of `output_token_times`). Real for all statuses; the J/token denominator |
| `server_prompt_tokens` | int64 &#124; null | Engine's self-reported prompt-token usage. Auxiliary provenance only |
| `server_completion_tokens` | int64 &#124; null | Engine's self-reported completion-token usage. Auxiliary provenance only |
| `status` | string | Terminal status: `"ok"` (transport returned), `"error"` (it raised), `"timeout"` (never completed) |
| `finish_reason` | string &#124; null | Stream's terminal reason (e.g. `"stop"` vs `"length"`). Real only when a finish chunk arrived, else `null` |
| `level_index` | int32 | Which rate level the request belongs to |
| `window_index` | int32 | Which measurement window (within the level) the request belongs to |
| `in_measurement_window` | bool | Issued within the steady-state span `[span_start, span_end]` |
| `is_ramp` | bool | Issued during the level's ramp; never counted toward steady-state metrics |
| `completed_in_drain` | bool | Issued in-span but completed after `span_end`: full latency kept, only in-span tokens counted |
| `output_token_times` | list(float64) | Client-side per-token receipt series (one entry per streamed content delta). Receipt-unclipped and real for all statuses |

The window's measured span bounds are stored as **file-level Parquet metadata** (`span_start`, `span_end`, plus `experiment_id` and `declared_config_hash`), not per row. Because the per-row token series is issue-partitioned and receipt-unclipped, those file-level bounds let an alternative attribution be re-clipped and re-derived offline without re-running the study; the authoritative in-span denominator remains the window `result.json` `output_tokens`.

The two boundaries are used differently. The **energy denominator** counts client-side output tokens delivered within the span across all requests, regardless of terminal status (a mid-stream failure still delivered those tokens). **Latency percentiles** filter to `status == "ok"` on the consumer side; the rows never pre-filter.

### Derived server metrics and the SLO overlay

Per-window server metrics are derived by a pure function of the request rows (and, when SLO bounds are configured, the bounds). The derivation constructs no backend, reads no clock, and never re-clips receipts:

| Field | Description |
|-------|-------------|
| `request_throughput_req_s` | Completed requests per second over the span |
| `completed_count` / `error_count` / `timeout_count` | Terminal-status tallies |
| `completion_rate` | Completed / total in-window |
| `length_truncated_count` | Completed requests that stopped at the output-token budget (`finish_reason == "length"`) |
| `ttft` / `tpot` / `itl` / `e2e` | Latency percentile blocks (p50 / p90 / p99) over `status == "ok"` requests |
| `energy_at_operating_point_j_per_token` | The window's energy per token at this rate |
| `server_reported_client_token_ratio` | Engine-reported vs client-counted token divergence |
| `slo` | The SLO evaluation overlay (below), or `null` when no bounds were configured |

The SLO overlay is a **pure post-hoc overlay**. The bounds only classify a window after the fact: no SLO value enters the window's identity or its physical measurement, so every measurement field is byte-stable across an SLO re-judgement and only the `slo` block moves. The same rows judged against different bounds (offline, over a loaded `requests.parquet`) yield a different verdict with every physical measurement untouched.

| SLO field | Description |
|-----------|-------------|
| `ttft_bound_ms` / `tpot_bound_ms` / `percentile` | The bounds this window was judged against |
| `ttft_at_percentile_ms` / `tpot_at_percentile_ms` | Observed tail values at the shared percentile, for cross-checking |
| `attainment_fraction` | Fraction of completed requests meeting all configured bounds jointly |
| `slo_pass` | Window verdict: `attainment_fraction >= percentile` |
| `goodput_tokens_s` | SLO-meeting throughput (direct join; see the note below) |
| `energy_at_operating_point_valid` | Whether this window's operating-point energy is usable (`slo_pass and level_valid`) |

**Attainment is a per-request joint evaluation**, not a per-metric percentile: a completed request counts iff its TTFT is within `ttft_ms` **and** its per-request TPOT is within `tpot_ms` at once. A latency exactly equal to a bound meets it; only strictly-greater-than violates. A missing TTFT under a configured TTFT bound fails; an undefined TPOT (fewer than two tokens) passes the TPOT bound vacuously.

**Length-truncated requests are attainment-eligible.** A completion that stopped at its output-token budget (`finish_reason == "length"`) is a normal completion (`status == "ok"`) - the workload fixes the output budget, so a length stop is a served request, not a failure. Such completions are counted and evaluated against the bounds like any other, and their tally is disclosed in `length_truncated_count` so a caller can re-segment for a different reading.

**Goodput.** `goodput_tokens_s` is a direct join: the in-span output tokens of the requests that **both** completed (`status == "ok"`) **and** met every configured bound, divided by the span duration (DistServe, OSDI'24, arXiv:2401.09670; Wang et al., arXiv:2410.14257 Eq. 5). No failed request's tokens enter it at any weight. The numerator is in-span-clipped (each qualifying request's receipts up to the window's `span_end`), a disclosed deviation from a full per-request count that keeps the span discipline and guarantees `goodput_tokens_s <= avg_tokens_per_second`. It is `null` when no request completed or the span duration is non-positive, and `0.0` when requests completed but none qualified; read it alongside `completion_rate`.

### Files as a record: querying per-window results

Because each window writes its own `requests.parquet`, a whole study's request logs are one glob. For example, TTFT percentiles per rate level with DuckDB:

```python
import duckdb

df = duckdb.sql("""
    SELECT level_index,
           count(*) FILTER (WHERE status = 'ok') AS ok_requests,
           quantile_cont(ttft_ms, 0.99) FILTER (WHERE status = 'ok') AS p99_ttft_ms
    FROM 'results/<study>_<ts>/*/requests.parquet'
    WHERE in_measurement_window AND NOT is_ramp
    GROUP BY level_index
    ORDER BY level_index
""").df()
print(df)
```

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

The whole bundle carries one `bundle_version` (currently `"2.0"`), stamped identically into `result.json`, `config.json`, and `system.json` (`timeseries.parquet` is self-describing columnar data and stays unversioned). It versions the on-disk layout, the artefact set, and each artefact's schema as a single contract, so there is one number to bump and one changelog line per documented break. This replaces the three independent per-artefact `schema_version` counters that earlier releases carried. It follows semantic versioning: minor bumps add fields without breaking existing readers, major bumps signal breaking changes. Pre-1.0 the policy is conservative - new fields land as `Optional` with `default = null` so existing parsers don't break.

Older bundles remain readable best-effort: the canonical reader (`llenergymeasure.results.bundle.BundleReader`, wrapped by `load_result`) tolerates a legacy shape rather than rejecting it, emitting a single warning per bundle. A bundle 1.0 (the provenance-unification break) reads with its retired top-level `baseline_power_w` copy and `schema_version` key dropped on load, its pre-rename `hardware.cuda.version` mapped onto `driver_supported_version`, its never-populated hardware fields (`pcie_gen`, `mig_enabled`, `cudnn_version`, `fan_speed_pct`) ignored, and its old separate runner block read into the unified `RunnerProvenance` model. The system sidecar's earlier filename `environment.json` is a clean break, not a tolerance: it is not read, so a pre-rename bundle's system sidecar is treated as absent. There is no in-place migration and no converter tooling.

The one exception to best-effort tolerance is the runner-mode vocabulary: `RunnerProvenance.mode` is a closed `Literal["process", "container"]`, so any bundle (1.0 or 2.0-era) whose runner block carries the pre-v0.7 `docker`/`local` mode fails validation on read rather than loading a stale value. This is the deliberate v0.7 clean break; the other 1.0 tolerances above are unaffected.

## See also

- [Tutorial - Multi-engine study](/tutorials/multi-engine-study) walks through writing analysis code against this schema.
- [How to - Interpret results](/how-to/interpret-results) reads a real `result.json` field by field with worked examples.
- [Methodology - Energy measurement](/explanation/methodology/energy-measurement) explains where the numbers come from.
