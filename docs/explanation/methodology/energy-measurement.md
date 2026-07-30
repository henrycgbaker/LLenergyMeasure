# Energy Measurement

What LLenergyMeasure measures and how it works.

---

## What We Measure

LLenergyMeasure collects three categories of metrics during inference:

### Energy (Joules)

GPU power draw integrated over inference time. This is the primary metric.

Energy in Joules = integral of Power(t) dt over the inference window.

LLenergyMeasure reports:
- `total_energy_j` - total GPU energy during inference
- `energy_adjusted_j` - total energy minus baseline idle power (isolates the
  inference-attributable component)
- `baseline_power_w` - idle GPU power measured before inference
- `energy_per_token_mj_adjusted` - per-token form of the adjusted figure (preferred for
  cross-experiment comparison)

### Throughput

Tokens generated per second.

- `tokens_per_second` - total output tokens divided by total inference time
- Per-prompt throughput is available in the detailed per-prompt result records

Throughput measurements are sampler-independent and do not require GPU access - they
work from wall-clock timing.

### FLOPs

Floating point operations estimated from model architecture.

- Based on the standard transformer FLOPs formula: `~6 * N * T` where N is non-embedding
  parameter count and T is the total token count (input + output)
- **Approximate** - uses architecture-based estimation (parameter count, sequence lengths),
  not hardware performance counters
- Useful for efficiency normalisation (energy per FLOP) but not for direct hardware
  utilisation analysis

### Peak Memory

- `inference_memory_mb` - peak GPU memory used during inference only (not model loading)
- Measured as the delta between pre-inference and peak NVML memory readings
- Reflects the KV cache and activation memory cost of the specific inference configuration

---

## Configuration

Energy measurement is configured with a single field in the `measurement:` section of
your experiment YAML:

```yaml
measurement:
  energy_sampler: auto   # auto | nvml | zeus | codecarbon | null
```

The field is a flat scalar - no per-sampler sub-config. All sampler-specific parameters
(polling intervals, sampling modes, CPU measurement) are resolved internally by the
harness. See [Design Rationale](#design-rationale) for why.

### GPU Telemetry (Parquet Sidecar)

The study-level `output.save_timeseries` field controls whether the NVML
power/thermal/memory timeseries is persisted to a Parquet sidecar file:

```yaml
output:
  save_timeseries: true    # default: persist timeseries.parquet alongside result JSON
  # save_timeseries: false # skip parquet output (useful for large sweeps)
```

It lives on the study's `output:` block, not on individual experiments, because it only
controls disk output - an operational concern, not part of the scientific specification.
For single experiments the sidecar is written alongside the result whenever an output
directory is given (CLI `--output` or `run_experiment(output_dir=...)`); with no output
directory, nothing is written to disk at all.

NVML telemetry is always collected during inference for throttle detection and measurement
quality warnings - `save_timeseries` only controls whether the data is written to disk.
The Parquet sidecar is independent of which energy sampler is selected: even with
`energy_sampler: zeus` or `energy_sampler: null`, the NVML timeseries still runs.

The two systems are independent:

| System | Purpose | Configured by |
|--------|---------|---------------|
| Energy sampler (Zeus/NVML/CodeCarbon) | Total energy in joules | `measurement.energy_sampler:` |
| NVML telemetry (PowerThermalSampler) | Power/thermal/memory timeseries + throttle detection | `output.save_timeseries:` (disk output only) |

---

## Energy Samplers

### `auto` (default)

Auto-selects the best available sampler in priority order: **Zeus > NVML > CodeCarbon**.

Zeus is preferred because it reads hardware energy counters directly (zero overhead,
no integration error). NVML is the fallback (polls power at 100ms intervals). CodeCarbon
is the last resort (coarser sampling, lower accuracy). If no sampler is available
(CPU-only machine), energy measurement is silently disabled.

### `nvml`

NVIDIA Management Library. Polls GPU power draw at 100ms intervals during inference.

- Ships with LLenergyMeasure (no additional install) - `pynvml` is a base dependency
- Works on any NVIDIA GPU (Kepler and newer)
- Measures GPU power only (not CPU or RAM)
- Uses trapezoidal integration of power samples to compute energy in joules
- Hardware sensor accuracy: +/-5% (proportional, same physical sensor as Zeus)
- On Volta+ GPUs, NVML and Zeus read the same sensor; Zeus eliminates integration error

### `zeus`

Zeus GPU energy monitoring via hardware energy counters.

- **Most accurate** software-accessible GPU energy measurement available
- On Volta+ GPUs (V100, T4, A100, H100, etc.): reads `nvmlDeviceGetTotalEnergyConsumption`
  hardware counter - two NVML calls total, zero overhead during measurement, 1mJ resolution
- On pre-Volta GPUs: automatically falls back to power polling (equivalent to the NVML sampler)
- Requires additional install: `pip install "llenergymeasure[zeus]"`
- GPU-only (CPU/DRAM measurement via RAPL is disabled - requires root privileges and is
  <5% of total energy for GPU-dominated LLM inference)

### `codecarbon`

CodeCarbon tracker. Estimates total system energy including CPU, GPU, and RAM.

- Broadest scope - captures GPU, CPU, and RAM energy
- Adds CO2 emissions estimate based on grid carbon intensity
- Requires additional install: `pip install "llenergymeasure[codecarbon]"`
- **Least accurate** for GPU energy: ~25-40% error vs physical power meters
  (Hessenthaler et al., 2025), compared to ~5-10% for NVML polling and ~1-2% for Zeus
- Uses the same `nvmlDeviceGetPowerUsage` NVML call as the NVML sampler, but at coarser
  intervals (1s vs 100ms) with simpler integration
- Best used when you need CPU/RAM energy or carbon emissions estimates

### `null` (disabled)

Set `energy_sampler: null` (YAML `null`) to disable energy measurement entirely.
The tool runs in throughput-only mode - no energy metrics, but faster execution.

---

## Accuracy Hierarchy

```
Most accurate                                        Least accurate
     |                                                       |
     Zeus (hw counter)  >  NVML (100ms poll)  >  CodeCarbon
     ~1-2% error           ~5-10% error           ~25-40% error
     GPU only              GPU only               GPU + CPU + RAM (estimated)
```

All three samplers ultimately read the same NVIDIA power sensor (shunt resistor on the
GPU power rails). The accuracy difference comes from how they process the readings:

- **Zeus** reads the hardware energy counter (`nvmlDeviceGetTotalEnergyConsumption`) which
  accumulates energy continuously on-die - no software polling, no integration error
- **NVML** polls `nvmlDeviceGetPowerUsage` at 100ms intervals and integrates via
  trapezoidal rule - subject to polling gaps and integration approximation
- **CodeCarbon** polls the same NVML function at 1s intervals with simple multiplication -
  coarser resolution, plus CPU/RAM estimates add further uncertainty

The +/-5% floor on all NVML-based measurements comes from the physical sensor accuracy
(NVIDIA-documented). Software integration adds 0-5% on top depending on sampler.

---

## How Energy is Measured

### Zeus (hardware energy counter)

On Volta+ GPUs, Zeus reads hardware energy counters at the start and end of the
measurement window. Energy = end_counter - start_counter. No polling thread runs during
inference - zero CPU overhead.

### NVML (power polling)

1. A `PowerThermalSampler` thread starts immediately before inference begins
2. The sampler polls `nvmlDeviceGetPowerUsage()` at 100ms intervals
3. Each sample records timestamp, power (watts), temperature, memory, and utilisation
4. When inference completes, the sampler thread is stopped
5. Consecutive sample pairs are integrated using the **trapezoidal rule**:
   ```
   energy_j += (power[i] + power[i+1]) / 2 * (time[i+1] - time[i])
   ```
6. The result is total GPU energy in joules over the inference window
7. If baseline measurement is enabled:
   `energy_adjusted_j = total_energy_j - (baseline_power_w * duration_s)`

The raw power timeseries is available in the result JSON for detailed post-hoc analysis.

For very short inference runs (< 200ms), the 100ms polling interval means only 1-2 samples
may be collected. Energy estimates for sub-200ms runs have higher relative uncertainty.
Use `n >= 50` prompts to ensure sufficient measurement duration.

---

## Measurement Window

By default LLenergyMeasure integrates energy over the **whole** inference run
(`measurement_methodology: total`). Two opt-in modes restrict the measured region so that
warm-up transients (kernel autotuning, clock ramp, cache fill) do not bias energy-per-token
and throughput:

```yaml
measurement:
  measurement_methodology: total        # total | windowed | steady_state  (default: total)

  # windowed mode: explicit window relative to inference start
  measurement_window: [2.0, 8.0]        # (start_sec, end_sec)

  # steady_state mode: discard a warm-up prefix, measure the remainder
  warmup_discard_fraction: 0.1          # discard first 10% (default)
  warmup_discard_seconds: 1.5           # OR discard a fixed number of seconds
  steady_state_auto_detect: false       # opt-in sliding-window stability detector
```

- **`total`** - whole run, unchanged default. No pre-cleaning or windowing is applied.
- **`windowed`** - integrate energy and attribute tokens over an explicit
  `[start_sec, end_sec]` window.
- **`steady_state`** - discard a deterministic warm-up prefix (a fraction by default, or a
  fixed number of seconds) and measure the remainder. Setting `steady_state_auto_detect`
  enables a lightweight sliding-window stability test (coefficient-of-variation /
  variance-ratio) that finds the steady-state onset; on failure it falls back to the fixed
  discard and sets the `steady_state_not_detected` flag in the `config.json` sidecar.

For `windowed` and `steady_state`, energy is **re-integrated** by feeding the trapezoidal
integrator only the power samples inside the window (after dropping zero / impossible
samples and median-filtering transient NVML dropouts). The realised window, the
methodology actually used, the discard fraction, and the `steady_state_not_detected` flag
are recorded in the `config.json` sidecar for auditability.

Tokens and throughput are attributed to the sub-window **proportionally by time**: the
harness captures per-request and inter-token durations but not absolute per-token
timestamps, so a window covering 60% of the cleaned inference span is credited with 60% of
the output tokens. A measurement warning records this approximation, and a second warning
fires when the realised window is too short to be meaningful.

This design is grounded in a survey of prior measurement methodology: MLPerf Power,
SPECpower, the VM-warmup steady-state study, and the Cao-Rhinehart detector choice.

---

## Baseline Power

Before inference, LLenergyMeasure measures idle GPU power draw. This baseline represents
the GPU's steady-state power consumption when not doing inference work.

**Why baseline matters:** GPU power draw is never zero. A 300W GPU consuming 50W at idle
means that a 100-second inference consuming 200J of total energy is attributable to only
`200J - (50W * duration_s)` of inference work. Without baseline subtraction, results
conflate background power draw with inference energy.

**How it works:**
- The sampler polls GPU power for `baseline.duration_seconds` (default: 30s) before
  the first experiment
- The mean is stored as `baseline_power_w`
- Baseline results are persisted to disk (`_study-artefacts/baseline_cache.json`) and
  shared across experiments in a study, including Docker containers via bind-mount

**Caching strategies** (`baseline.strategy`):

| Strategy | Behaviour | When to use |
|:---------|:----------|:------------|
| `cached` | Measure once, persist to disk with configurable TTL. After `cache_ttl_seconds` the baseline is re-measured automatically. Docker containers load the host's cached measurement via bind-mount. | Short studies where thermal conditions are stable. |
| `validated` (default) | Same as `cached`, but periodically spot-checks (5s quick measurement) every N experiments. If power drift exceeds the threshold, re-measures the full baseline. | Most studies - catches thermal drift with negligible overhead (~5s per spot-check). |
| `fresh` | Every experiment measures its own baseline independently. No study-level caching. | Maximum accuracy when measurement isolation matters more than speed. |

Configure via the `baseline:` section:

```yaml
baseline:
  enabled: true
  duration_seconds: 30        # 5-120s accepted
  strategy: validated          # or "cached" or "fresh"
  cache_ttl_seconds: 7200     # 2 hour TTL (strategy: cached/validated)
  validation_interval: 5      # spot-check every 5 experiments (strategy: validated)
  drift_threshold: 0.10       # 10% drift triggers re-measurement (strategy: validated)
```

---

## What the Harness Resolves Internally

The following parameters are resolved automatically by the measurement harness. They are
not exposed in YAML config because they are either hardware-determined, sampler-specific
implementation details, or have single correct values:

| Parameter | Value | Rationale |
|:----------|:------|:----------|
| NVML polling interval | 100ms | Matches A100 hardware update period. Polling faster reads stale values. |
| Zeus CPU measurement | Disabled | Requires root privileges; <5% of total energy for GPU workloads. |
| Zeus sync mode | `torch` | All engines use PyTorch/CUDA. Always correct. |
| CodeCarbon polling interval | 1s | Appropriate for inference (default 15s is for training). |
| CodeCarbon tracking mode | `process` | Better attribution than `machine` for single-workload benchmarks. |
| CodeCarbon file output | Disabled | We extract metrics programmatically; prevents stray `emissions.csv`. |
| GPU indices | Auto-resolved | Derived from engine config (tensor_parallel_size, device_map, etc.). |
| Baseline cache TTL | 2 hours | Configurable via `baseline.cache_ttl_seconds`. Disk-persisted and shared with Docker containers. |
| Integration method | Trapezoidal rule | Standard for non-uniform timesteps; Simpson's offers no practical gain given +/-5% sensor noise. |
| Power reading mode | Instantaneous | Uses least-smoothed NVML reading for best temporal resolution. |

The provenance of these sampler choices, along with the baseline-measurement
location and default token lengths, is catalogued in
[Empirical grounding](/explanation/methodology/empirical-grounding).

---

## Design Rationale

### Why `energy_sampler` is a flat field (not nested)

Research into all three samplers (NVML, Zeus, CodeCarbon) confirmed that:

1. **Zeus** has zero user-tunable parameters - all constructor args are internally resolved
2. **CodeCarbon** has 30+ constructor params, all of which should be internally resolved
3. **NVML** has one potentially tunable param (polling interval), but the 100ms default
   matches A100 hardware and is correct for the vast majority of use cases

Since no sampler exposes parameters worth configuring in YAML, a nested
`energy: { sampler: auto }` structure would add indirection with zero value. A flat
`energy_sampler: auto` field is semantically identical and simpler.

### Why Zeus is preferred over NVML

On Volta+ GPUs (V100, A100, H100, and all consumer GPUs since RTX 2000 series), Zeus
reads hardware energy counters that accumulate continuously on the GPU die. This approach:

- Has **zero CPU overhead** during measurement (no polling thread)
- Eliminates **integration error** entirely (hardware does the accumulation)
- Works reliably for **any duration** (sub-second to hours)
- Uses the **same physical sensor** as NVML (so accuracy floor is identical)

NVML polling is kept as a fallback because:
- `pynvml` is a base dependency (zero-install energy measurement)
- Pre-Volta GPUs lack the energy counter API
- The `PowerThermalSampler` still runs during measurement for thermal monitoring
  and timeseries telemetry regardless of which energy sampler is active

### External references

- **NVML** - [NVIDIA Management Library Reference](https://docs.nvidia.com/deploy/nvml-api/index.html) (hardware sensor architecture, polling intervals, power reading modes)
- **Zeus** - [Zeus: Understanding and Optimizing GPU Energy Consumption (NSDI '23)](https://www.usenix.org/conference/nsdi23/presentation/you) and the [Zeus repo](https://github.com/ml-energy/zeus) (hardware counters, CPU/DRAM measurement)
- **CodeCarbon** - [CodeCarbon docs](https://mlco2.github.io/codecarbon/) (estimation modes, accuracy validation, regional carbon intensity)

---

## Limitations

**All samplers measure GPU only (by default).** CPU and RAM power are not included in
the primary energy metric. CodeCarbon can estimate CPU/RAM energy but with significant
uncertainty (~25-40% total error).

**NVML sensor has a +/-5% accuracy floor.** This is a hardware limitation of the on-board
shunt resistor and ADC. All software samplers (Zeus, NVML, CodeCarbon) share this floor.
For publication-quality results, report energy with appropriate uncertainty bounds.

**A100 power sensor observes 25% of runtime.** The A100's power sensor averages over a
25ms window within each 100ms update period (Bridges et al., 2023). During the remaining
75ms, the GPU could draw substantially different power. Zeus's hardware energy counter
mitigates this by accumulating continuously.

**Multi-GPU measurement sums per-device energy.** For tensor-parallel runs across multiple
GPUs, LLenergyMeasure sums per-device energy. All participating GPUs are automatically
monitored based on the engine's parallelism config.

**Container isolation.** Inside Docker containers, pynvml accesses the same physical GPU
as the host via the NVIDIA Container Toolkit. Energy readings represent the full device
power draw, including any other processes sharing the GPU.

**MIG instances.** Multi-Instance GPU partitions report the parent GPU's power, not
per-instance power. MIG energy readings include all instances on the physical GPU.
