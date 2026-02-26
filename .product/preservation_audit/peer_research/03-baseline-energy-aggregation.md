# Peer Research: Baseline Power & Energy Aggregation
> Generated 2026-02-26. Peer evidence for preservation audit items N-X01 + N-R02.

## Evidence Per Tool

### 1. Zeus (ml-energy/zeus)

**Source**: [Zeus Measuring Energy docs](https://ml.energy/zeus/measure/), [energy API reference](https://ml.energy/zeus/reference/monitor/energy/), [power API reference](https://ml.energy/zeus/reference/monitor/power/), [GitHub](https://github.com/ml-energy/zeus)

**Baseline/idle power measurement**: None. Zeus does not measure, subtract, or even mention
baseline/idle power anywhere in its documentation or source code. The `ZeusMonitor` class
measures raw energy deltas between `begin_window()` and `end_window()` calls without any idle
correction. The reported energy includes all power drawn during the window, including idle
components.

**Caching**: No baseline caching (no baseline to cache). The `Measurement` class uses
`@cached_property` on the `total_energy` field, but this is a computed-sum cache (sum of
per-GPU energies), not a baseline measurement cache.

**Multi-GPU energy aggregation**: Per-GPU energy is stored in `gpu_energy: dict[int, float]`
(Joules). `total_energy` is a `@cached_property` that returns `sum(self.gpu_energy.values())` --
a simple sum across all monitored GPUs. Each GPU's energy is measured independently via NVML's
`nvmlDeviceGetTotalEnergyConsumption` (Volta+) as `end_energy - start_energy`, or via a
background `PowerMonitor` process (older GPUs) that polls `nvmlDeviceGetPowerUsage` and
integrates power samples using AUC (`sklearn.metrics.auc(timestamps, powers)`). The
`PowerMonitor` spawns separate daemon processes per power domain and uses deduplication to skip
unchanged readings. Update period is auto-detected by sampling 1000 measurements and halving
the detected counter update interval.

**Negative energy / noise handling**: Not addressed. If `end_energy - start_energy` produces a
negative or zero value (possible with counter rollover or very short windows), Zeus has a
fallback: `approx_instant_energy` parameter estimates energy as `instant_power * window_duration`
for very short windows. But there is no explicit floor-at-zero guard.

**Thermal throttling**: Not tracked. ZeusMonitor does not query throttle reasons or temperature.

**Key quote from Zeus docs**: "All GPUs are measured simultaneously if `gpu_indices` is not
given." Energy is framework-index-aware and remaps through `CUDA_VISIBLE_DEVICES`.

---

### 2. CodeCarbon (mlco2/codecarbon)

**Source**: [GitHub](https://github.com/mlco2/codecarbon), [methodology docs](http://docs.codecarbon.io/methodology.html), source file `codecarbon/core/gpu.py`

**Baseline/idle power measurement**: None. CodeCarbon measures total GPU energy consumption
during the tracked period using `pynvml.nvmlDeviceGetTotalEnergyConsumption()` (Volta+),
computing deltas between successive polling intervals. There is no idle measurement phase and
no baseline subtraction. The `delta()` method computes `current_energy - last_energy` between
polling cycles (default every 15 seconds). This raw delta includes all power drawn, including
idle components.

**Caching**: Static GPU properties (name, UUID, power limit, total memory) are cached during
initialisation in `_init_static_details()`. No measurement caching across runs.

**Multi-GPU energy aggregation**: `AllGPUDevices` enumerates all GPUs via
`pynvml.nvmlDeviceGetCount()` and creates a `GPUDevice` object per index. Each GPU is measured
independently. Energy and power are returned as a list of per-GPU dictionaries. CodeCarbon sums
energy across all detected GPUs to produce a total, but there is a known limitation: it queries
all GPUs from NVML regardless of `CUDA_VISIBLE_DEVICES`, which can overcount energy on
multi-tenant systems (see [issue #567](https://github.com/mlco2/codecarbon/issues/567)). A
`gpu_ids` parameter allows manual override.

**Negative energy / noise handling**: Not addressed. The delta computation assumes monotonically
increasing energy counters. No floor guard.

**Thermal throttling**: Temperature is read via `pynvml.nvmlDeviceGetTemperature()` and reported
as a raw metric. No throttle detection, no throttle reason bitmask queries, no adjustment of
energy values based on thermal state.

---

### 3. MLPerf Power (mlcommons/power-dev)

**Source**: [MLPerf Power docs](https://docs.mlcommons.org/inference/power/), [inference policies](https://github.com/mlcommons/inference_policies/blob/master/power_measurement.adoc), [arXiv 2410.12032](https://arxiv.org/html/2410.12032v1), [HPCA 2025 tutorial](https://github.com/aryatschand/MLPerf-Power-HPCA-2025/blob/main/measurement_tutorial.md)

**Baseline/idle power measurement**: No explicit idle baseline phase. MLPerf measures total
system power "at the wall" via external power analysers (Yokogawa, SPEC PTDaemon). The
methodology captures gross system power during the performance phase, time-bounded by LoadGen
start/stop timestamps. There is no subtraction of idle power. The philosophy is that total
system power *including* idle components is the relevant metric for benchmarking, since all
system components (fans, memory, interconnect) contribute to the cost of running inference.

MLPerf does include a **ranging phase** (auto-detect current/voltage ranges) and a **testing
phase** (fixed ranges for accuracy), but neither is an idle baseline measurement -- ranging
determines analyser sensitivity, not background power.

For training workloads (arXiv 2410.12032): "The energy for each node is calculated by
integrating power samples over the run's time window, and the total energy is computed by
summing the energy across all compute and interconnect components." No idle subtraction.

**Caching**: N/A -- external hardware analyser, no software caching concept.

**Multi-GPU energy aggregation**: MLPerf measures at the system level, not per-GPU. For
single-SUT inference (v1.0-v2.0), a single power analyser measures the entire system (including
all GPUs, CPUs, memory, fans). Multi-node training sums per-node energy. The 2410.12032 paper
notes support for "multiple analyzers connected to a single SUT or multiple SUTs connected to
a single analyzer" for complex distributed scenarios.

**Negative energy / noise handling**: Not applicable -- external power analysers produce
positive readings by physical constraint. Accuracy is ensured by SPEC PTDaemon calibration and
range-locking.

**Thermal throttling**: Mentioned only as a complicating factor for mobile/edge devices. No
throttle detection in the standard workflow. The assumption is that data centre systems are
thermally managed.

**Key insight**: MLPerf's "no idle subtraction" is a deliberate design choice: the benchmark
reports total energy cost of running inference, not marginal energy above idle. This is
appropriate for facility-level cost accounting but not for isolating workload-specific energy.

---

### 4. ML.ENERGY Blog & Benchmark

**Source**: [Best practices blog post](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/), [ML.ENERGY Benchmark paper arXiv 2505.06371](https://arxiv.org/html/2505.06371v1), [Leaderboard v3.0 blog](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/)

**Baseline/idle power measurement**: Not discussed. The best practices blog post focuses on
three points: (1) don't use TDP as a proxy, (2) use `nvmlDeviceGetTotalEnergyConsumption` on
Volta+, (3) synchronise CPU and GPU with `torch.cuda.synchronize()`. There is no mention of
idle power subtraction or baseline measurement.

The ML.ENERGY Benchmark paper (2505.06371) uses Zeus for measurement and reports raw energy
without idle subtraction. It focuses on steady-state energy during saturated serving rather than
idle-corrected energy.

**Multi-GPU handling**: The benchmark paper studies tensor parallelism scaling and reports that
"when scaling from one GPU to two, energy consumption barely changes, but when scaling further,
energy consumption significantly increases" due to inter-GPU communication overhead. Energy is
summed across GPUs via Zeus's `total_energy` property.

**Critical measurement limitation (from blog)**: On A100 and H100 GPUs, the NVML power sensor
samples only 25% of runtime (25ms averaging window updating every 100ms). During the
unsampled 75%, the GPU may draw very different power. This creates systematic measurement
error for short-duration workloads.

**Best practice recommendations**:
- Use `nvmlDeviceGetTotalEnergyConsumption` (two calls + subtraction), not power polling
- Always call `torch.cuda.synchronize()` before energy measurement boundaries
- Actual power varies from TDP by up to 4.1x (worst-case overestimation if using TDP)

---

### 5. NVIDIA NVML Documentation

**Source**: [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html), [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/nvmldevicegetpowerusage-sampling-rate/276517)

**Relevant APIs**:
- `nvmlDeviceGetPowerUsage(handle)`: Returns instantaneous power in milliwatts. On H100 with
  driver 535.xx, `power.draw` averages over 1 second, `power.draw.instant` samples every 25ms.
  Maximum practical polling rate ~66.7 Hz (15ms minimum interval). The underlying sensor
  samples at ~2 kHz internally.
- `nvmlDeviceGetTotalEnergyConsumption(handle)`: Returns cumulative energy in millijoules since
  driver load. Volta+ only. Requires only two calls (before/after) and a subtraction.
  Significantly more convenient and accurate than power polling + integration.
- `nvmlDeviceGetTemperature(handle, NVML_TEMPERATURE_GPU)`: Returns GPU temperature in Celsius.
- `nvmlDeviceGetCurrentClocksThrottleReasons(handle)`: Returns bitmask of active throttle
  reasons. Relevant bits: `nvmlClocksThrottleReasonSwThermalSlowdown`,
  `nvmlClocksThrottleReasonHwThermalSlowdown`, `nvmlClocksThrottleReasonSwPowerCap`,
  `nvmlClocksThrottleReasonHwPowerBrakeSlowdown`.

**Baseline/idle measurement patterns**: NVML documentation contains no guidance on idle power
measurement methodology. There is no dedicated API for "idle power" or "baseline power".
The standard pattern (used by our v1.x and confirmed by peers) is: poll
`nvmlDeviceGetPowerUsage` during a quiescent period and average the samples.

**Multi-GPU**: Each function takes a device handle obtained via `nvmlDeviceGetHandleByIndex(i)`.
There is no built-in aggregation -- callers must query each GPU independently and aggregate
themselves.

---

### 6. Carbontracker (lfwa/carbontracker)

**Source**: [GitHub](https://github.com/lfwa/carbontracker), source file `carbontracker/components/gpu/nvidia.py`

**Baseline/idle power measurement**: None. Carbontracker polls `pynvml.nvmlDeviceGetPowerUsage`
at 10-second intervals during training epochs. Power values are averaged per epoch
(`np.mean(comp.power_usages[-1], axis=0)`) and multiplied by duration to estimate energy.
There is no idle measurement, no baseline subtraction.

**Caching**: None. Each `power_usage()` call directly queries NVML. No cross-run caching.

**Multi-GPU energy aggregation**: Returns a list of per-GPU power values. The caller
(tracker.py) sums `total_energy += energy_usage` across all components. Supports
`devices_by_pid=True` to filter to GPUs running the tracked process's PIDs, or
`devices_by_pid=False` to measure all available GPUs.

**Negative energy / noise handling**: Not addressed.

**Thermal throttling**: Not tracked. No temperature queries, no throttle detection.

---

### 7. experiment-impact-tracker (Breakend)

**Source**: [GitHub](https://github.com/Breakend/experiment-impact-tracker), source file `experiment_impact_tracker/gpu/nvidia.py`, [Henderson et al. 2020 JMLR](https://jmlr.org/papers/volume21/20-312/20-312.pdf)

**Baseline/idle power measurement**: None. The tool records absolute power draw from GPUs
during experiments. The companion paper (Henderson et al. 2020) advocates for systematic
energy reporting but does not describe idle power subtraction as part of the methodology.

**Multi-GPU energy aggregation**: Two modes in the source code:
1. `absolute_power`: sums raw power across all GPUs (`absolute_power += float(power_draw)`)
2. Process-attributable power: weights each GPU's power by the process's SM utilisation
   percentage (`power += sm_relative_percent * float(power_draw)`)

This is notable: experiment-impact-tracker is the only peer tool that attempts per-process
power attribution on shared GPUs, using SM utilisation as a proxy.

**Measurement method**: Parses `nvidia-smi -q -x` (XML output) and `nvidia-smi pmon -c 5`,
rather than using pynvml directly. This adds subprocess overhead.

**Caching**: None. `nvidia-smi` is invoked fresh on every call via `subprocess.Popen`.

**Negative energy / noise handling**: Not addressed.

**Thermal throttling**: Not tracked.

**Note**: Repository archived October 2025 (read-only).

---

### 8. Academic Literature

#### Yang, Adamek & Armour (SC24): "Accurate and Convenient Energy Measurements for GPUs"

**Source**: [SC24 proceedings](https://dl.acm.org/doi/10.1109/SC41406.2024.00028), [arXiv 2312.02741](https://arxiv.org/html/2312.02741v2)

**Key findings on NVML power sensors**:
- Tested 70+ GPUs across 12 architectural generations
- A100/H100: 25ms averaging window, updates every 100ms (only 25% of runtime sampled)
- Power readings are "shifted 100ms earlier" on some architectures due to sensor lag
- Naive energy measurement can have errors exceeding 70%
- Proposed best practices reduce error by ~35%:
  1. Execute target program for 32 consecutive iterations or until 5-second minimum runtime
  2. Conduct four separate trials with randomised delays
  3. Post-process by discarding ramp-up repetitions and synchronising GPU activity timing

**Baseline/idle**: The paper notes idle power clusters at different P-states but does not
recommend idle subtraction. Focus is on correcting measurement methodology rather than
baseline normalisation.

#### Lannelongue et al. (2023): "How to Estimate Carbon Footprint When Training Deep Learning Models"

**Source**: [arXiv 2306.08323](https://arxiv.org/pdf/2306.08323), [hal-04120582](https://hal.science/hal-04120582/document)

Comparative review of CodeCarbon, Carbontracker, Eco2AI. Confirms: none of these tools
perform idle power subtraction. Notes that "dynamic energy consumption (the difference between
total and idle energy consumption)" is a concept evaluated by external power meters in their
review, but not implemented by any of the software tools studied.

#### Krzywaniak et al. (2024): "Green AI: Energy Consumption in DL Models Across Runtime Infrastructures"

**Source**: [arXiv 2402.13640](https://arxiv.org/html/2402.13640v1)

**Key methodology**: The researchers performed their own idle baseline subtraction external to
any measurement tool:
- "Record the GPU power in idle mode for 10 minutes" at 10ms polling interval
- "Record CPU energy consumption during its idle state for 10 minutes"
- "Subtract the average from our result"

This is a manual research methodology, not implemented in any peer tool. The 10-minute
duration is notably longer than our 30-second default.

#### Hanafy et al. (2024): "Computing Within Limits: Energy Consumption in ML Training and Inference"

**Source**: [arXiv 2406.14328](https://arxiv.org/html/2406.14328v1)

Formalises idle subtraction as: `E_tr = integral(P_tr(t))dt - integral(P_idle(t))dt`. Uses
a "hardcoded time interval" for idle measurement (duration unspecified in paper). Measures
`P(t) = P_CPU(t) + P_GPU(t) + P_DRAM(t)`. Confirms that no off-the-shelf tool performs this;
they built custom instrumentation.

#### Data centre context (US DOE 2024 Report)

AI server idle power ~20% of rated power (validated average 21.4% across published studies).
SPECpower benchmark: active idle ranges 20-60% of max utilisation power (100W-1300W absolute).
This confirms that idle power is a significant and variable fraction, reinforcing the need for
per-session measurement rather than fixed estimates.

---

## Summary Table

| Feature | Zeus | CodeCarbon | MLPerf Power | Carbontracker | experiment-impact-tracker | Our v1.x |
|---------|------|------------|-------------|---------------|--------------------------|----------|
| **Baseline/idle measurement** | No | No | No | No | No | Yes (30s NVML poll) |
| **Idle duration** | -- | -- | -- | -- | -- | 30s (configurable) |
| **Baseline subtraction** | No | No | No (deliberate) | No | No | Yes (`adjusted_j = raw - baseline * t`) |
| **Session-level cache** | No | No | N/A | No | No | Yes (1h TTL per device) |
| **Floor-at-zero guard** | No | No | N/A (hardware) | No | No | Yes (`max(0.0, adjusted)`) |
| **Multi-GPU energy** | Sum per-GPU deltas | Sum all GPUs (overcounts on shared) | System-level (wall) | Sum per-GPU | Sum (+ SM-weighted mode) | Sum raw; baseline from first process |
| **Thermal throttle tracking** | No | Temperature only | No (mentions edge) | No | No | Yes (OR-aggregation, bitmask) |
| **Negative energy handling** | No | No | N/A | No | No | Yes (floor at 0.0) |
| **NVML API for energy** | `GetTotalEnergyConsumption` (Volta+) or `GetPowerUsage` poll | `GetTotalEnergyConsumption` + `GetPowerUsage` | External analyser | `GetPowerUsage` poll | `nvidia-smi` subprocess | `GetPowerUsage` poll |
| **Per-process attribution** | No | No (system-wide) | No (system-wide) | Yes (PID filter) | Yes (SM utilisation weighting) | No (per-device) |

---

## Recommendation

### What our v1.x does that no peer tool does

1. **Baseline idle measurement + subtraction**: Zero out of five peer software tools perform
   idle power subtraction. This is our primary differentiator for scientific accuracy.
   Academic papers that need idle-corrected energy (Krzywaniak 2024, Hanafy 2024) build
   custom instrumentation because no library offers this. Our implementation fills a genuine
   gap.

2. **Session-level baseline caching with TTL**: No peer tool caches any cross-run measurement.
   The 1-hour TTL avoids 30s overhead per experiment in a study while ensuring the baseline
   doesn't go stale (GPU power states can shift over hours).

3. **Floor-at-zero guard**: No peer tool handles the case where baseline-adjusted energy goes
   negative. This is a necessary invariant when performing subtraction on noisy sensor data.

4. **Thermal throttle detection with OR-aggregation**: No peer tool tracks throttle reasons.
   CodeCarbon reads temperature but doesn't interpret it. Our bitmask approach
   (`nvmlDeviceGetCurrentClocksThrottleReasons`) with OR-aggregation across GPUs correctly
   captures "any GPU throttled" semantics.

### Carry forward unchanged

- `baseline.py`: 30s idle NVML poll, session cache with 1h TTL, floor-at-zero. All correct.
- `power_thermal.py`: Background sampler with throttle bitmask detection. All correct.
- Aggregation logic in `results/aggregation.py`: Sum raw energy, take baseline from first
  process, OR-aggregate thermal flags, MAX temperature/duration. All correct.

### Consider for v2.0

1. **Migrate to `nvmlDeviceGetTotalEnergyConsumption`** for inference energy measurement
   (not baseline -- baseline still needs power polling since we measure idle *power*, not
   idle *energy*). Zeus and CodeCarbon both use this API on Volta+. It avoids the 25%
   sampling limitation on A100/H100 documented by Yang et al. (SC24) and eliminates the
   need for a background polling thread during inference.

2. **Add `torch.cuda.synchronize()` fence** before energy measurement boundaries, as
   recommended by ML.ENERGY best practices. Without this, GPU work may still be in-flight
   when the energy snapshot is taken.

3. **Document the baseline duration choice**: 30 seconds is reasonable but not peer-validated
   (no peer does this). Academic baselines use 10 minutes (Krzywaniak 2024). Consider making
   the default configurable with a documented rationale for why 30s is sufficient (enough
   samples at 100ms interval = ~300 samples; GPU idle power is stable within seconds of
   quiescence).

4. **Per-device baseline in multi-GPU**: Our v1.x takes baseline from the first process only
   and applies it as a shared measurement. This is correct for homogeneous GPU setups
   (identical hardware = identical idle power) but would be incorrect for heterogeneous
   setups. For v2.0 single-machine scope this is fine. Flag as a consideration for Docker
   multi-node in future milestones.
