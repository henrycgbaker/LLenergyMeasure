# Measurement Methodology & Pitfalls Audit

**Domain:** LLM inference energy measurement -- scientific validity for publishable research
**Researched:** 2026-02-25
**Scope:** Audit of `.product/` methodology decisions against current evidence and peer practice
**Overall confidence:** MEDIUM-HIGH (most claims verified via multiple sources; NVML accuracy specs remain under-documented by NVIDIA)

---

## Executive Summary

This audit examines every measurement methodology decision in `.product/` against current evidence from the ML.ENERGY Benchmark (NeurIPS D&B 2025), MLPerf Power (IEEE HPCA 2025), the "Part-time Power Measurements" paper (arXiv:2312.02741), and peer tool implementations. The existing decisions are broadly sound in direction but several contain specific calibration errors, under-specified parameters, or unaddressed systematic biases that would compromise publishable-quality results.

**The three most serious findings:**

1. **The 30-second thermal floor is under-calibrated.** NVIDIA A100/H100 GPUs take 100ms update periods with 25% sampling coverage; thermal and power state stabilisation requires 60+ seconds under load, not 30 seconds at idle before load.
2. **The NVML accuracy numbers cited in `.product/` are wrong.** The existing docs claim Zeus/NVML is "~5% accurate" and CodeCarbon is "~15% accurate". In reality, NVML accuracy is +/-5 *watts* (not percent), and the percentage error depends on GPU power draw. For a 300W A100 this is ~1.7%; for a 40W idle GPU it is ~12.5%. CodeCarbon uses the same NVML readings with added overhead -- the accuracy gap between Zeus and CodeCarbon is primarily in *what* they measure (energy counter vs power polling), not a fixed percentage difference.
3. **FLOPs as a "primary metric" is scientifically misleading for this tool's stated purpose.** FLOPs are deterministic for a given model+input -- they do not vary between backends, batch sizes, or deployment configurations. For a tool that measures "how implementation choices affect efficiency", FLOPs provide zero discriminatory power. Energy-per-token and tokens-per-second are the metrics that actually vary.

---

## Critical Pitfalls

### CP-1: NVML Power Measurement Accuracy Is Worse Than Documented

**Current state in `.product/`:**
- `designs/energy-backends.md` claims Zeus/NVML energy counter is "~5% accurate"
- `designs/energy-backends.md` claims NVML power polling is "~10-15% accurate"
- `designs/energy-backends.md` claims CodeCarbon (NVML) is "~10-15% accurate"
- These figures are treated as fixed percentages

**What the evidence actually shows:**

NVIDIA documents `nvmlDeviceGetPowerUsage` accuracy as +/-5 *watts*, not +/-5 percent (NVML API Reference Guide). The percentage error therefore depends on the absolute power draw:

| GPU State | Typical Power | +/-5W Error | Percentage |
|-----------|--------------|-------------|------------|
| A100 idle | ~40W | +/-5W | **12.5%** |
| A100 under load | ~300W | +/-5W | **1.7%** |
| H100 SXM under load | ~600W | +/-5W | **0.8%** |
| Consumer GPU idle | ~15W | +/-5W | **33%** |

The "Part-time Power Measurements" paper (Burtscher et al., arXiv:2312.02741) found even worse issues:
- A100/H100: 100ms update period, but only 25% of runtime is sampled (75% unmeasured)
- Without correction methodology, power polling errors reached 30% standard deviation on A100
- With their proposed correction (32+ iterations, randomised delays, discarding rise time data), error reduces to ~5%

For `nvmlDeviceGetTotalEnergyConsumption` (the hardware energy counter used by Zeus on Volta+ GPUs):
- Returns millijoules since driver load
- More accurate than power polling because it integrates at the hardware level
- Known discrepancy reported on RTX 6000 Ada: readings ~0.5x expected (NVIDIA forum, unresolved June 2025)
- No formal accuracy specification published by NVIDIA for this counter

**Impact on LLenergyMeasure:**
- The accuracy table in `energy-backends.md` must be rewritten with conditional accuracy (varies by power draw)
- Results must report absolute power draw alongside energy so readers can assess measurement uncertainty
- Short, low-power experiments (batch_size=1 on small models) have the worst accuracy -- precisely the configuration space most interesting to this tool
- The claim that "Zeus is ~5% accurate" must be softened to "Zeus energy counter accuracy depends on GPU and workload; expect 1-5% under sustained load on Volta+ GPUs"

**Recommendation:** Replace the fixed-percentage accuracy table with a formula: `accuracy_pct = 5W / mean_power_W * 100`. Report this per-experiment in `ExperimentResult`. Add a `measurement_uncertainty_pct` field.

**Confidence:** MEDIUM-HIGH (NVIDIA +/-5W spec is documented; energy counter accuracy is under-documented; Burtscher paper is peer-reviewed)

**Sources:**
- [NVML API Reference Guide](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)
- [Part-time Power Measurements (arXiv:2312.02741)](https://arxiv.org/html/2312.02741v2)
- [NVIDIA Forum: energy counter discrepancy](https://forums.developer.nvidia.com/t/value-from-nvmldevicegettotalenergyconsumption-seems-to-be-off-by-a-factor/336318)
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)

---

### CP-2: 30-Second Thermal Floor Is Under-Calibrated

**Current state in `.product/`:**
- `decisions/warmup-strategy.md` specifies a 30-second thermal floor before energy measurement
- Rationale cites "10-30s GPU power state ramp documented in hardware literature"
- No source is actually cited for the 30-second figure

**What the evidence shows:**

The 30-second figure appears to be an estimate, not a measured value. The actual thermal stabilisation time depends on:

1. **GPU power state transitions:** NVIDIA GPUs use Dynamic Voltage and Frequency Scaling (DVFS). Moving from idle (P8) to compute (P0) happens in milliseconds, but the *power draw at P0* continues to change as the GPU thermally stabilises.

2. **Thermal ramp time under load:** Research on A100/H100 shows:
   - Initial power spike within 1-2 seconds of load onset
   - Power draw continues to drift upward for 45-90 seconds as die temperature increases from ambient to operating temperature (~70-83C)
   - The exact duration depends on cooling solution (SXM vs PCIe), ambient temperature, and airflow
   - MLPerf Power requires a **minimum 60-second measurement window** specifically because shorter windows are unreliable

3. **The 30-second floor as currently designed measures idle, not loaded stability.** The warmup strategy description implies 30 seconds of observation before the workload starts. But thermal stabilisation requires the GPU to be *under the representative workload* for the stabilisation period. 5 warmup runs at 2 tokens each do not thermally load the GPU.

**The actual problem:**
- 5 warmup runs x 2 tokens each = maybe 2-3 seconds of actual GPU compute (on a fast model)
- 30-second thermal floor (if measured at idle) does nothing -- the GPU cools back toward idle temperature
- The measurement window starts with the GPU thermally cold, then warms during measurement
- This creates a systematic bias: early measurement samples have lower power draw than later samples
- For parameter sweeps comparing batch sizes, this bias affects small-batch experiments more (shorter duration, less time to thermally stabilise)

**What peer tools do:**
- **MLPerf Power:** 60-second minimum measurement window (workloads loop until 60s reached)
- **ML.ENERGY Benchmark:** Defines steady state as batch-size-saturated period; implicitly excludes ramp-up by only measuring after server is at full utilisation
- **optimum-benchmark:** 10-20 warmup_runs (full inference runs, not 2-token stubs), which naturally provides thermal load
- **AIEnergyScore:** 10 runs of 1000 queries each; no explicit thermal handling but sheer volume provides natural stabilisation

**Recommendation:**
1. Increase warmup_runs default to **10** full-length runs (not reduced-output), matching optimum-benchmark practice. This provides genuine thermal loading.
2. Change thermal floor from 30 to **60 seconds of actual loaded operation** (matching MLPerf Power). This means the warmup phase must run the actual workload, not 2-token stubs.
3. Alternatively, keep reduced-output warmup for JIT/CUDA purposes (5 runs, 2 tokens) but add a separate **thermal conditioning phase** of 60 seconds running the actual workload before measurement begins. This separates the two concerns (JIT warmup vs thermal stabilisation).
4. Record GPU temperature at measurement start and end in `ExperimentResult` so thermal drift can be detected post-hoc.

**Confidence:** HIGH (MLPerf 60s minimum is documented; thermal drift physics are well understood; the 30s figure has no cited source)

**Sources:**
- [MLPerf Power Benchmark (arXiv:2410.12032)](https://arxiv.org/html/2410.12032v2)
- [ML.ENERGY Benchmark (arXiv:2505.06371)](https://arxiv.org/html/2505.06371v1)
- [NeurIPS 2025 Tutorial: Accurately Benchmarking Power & Energy](https://ml.energy/tutorials/neurips25/session-1.html)

---

### CP-3: FLOPs as "Primary Metric" Is Misleading for This Tool's Purpose

**Current state in `.product/`:**
- `decisions/flops-estimation.md` positions FLOPs as a core metric, with `flops_per_output_token` as a "primary cross-run comparison metric"
- `designs/result-schema.md` includes `flops_total`, `flops_per_token` in the Parquet export schema
- The decision rationale is "to enable comparison of energy efficiency across hardware generations"

**Why this is problematic:**

LLenergyMeasure's stated purpose is measuring "how implementation choice(s) alone affect LLM inference efficiency" -- batch size, quantisation, precision, backend, etc. For a given model and input:

```
FLOPs = 2 * N_params * tokens
```

This is a **deterministic function of model architecture and input/output length**. It does not change between:
- PyTorch vs vLLM vs TensorRT-LLM (same model, same FLOPs)
- batch_size=1 vs batch_size=32 (FLOPs per token is identical)
- fp16 vs bf16 vs fp32 (same MAC count; precision affects hardware throughput, not FLOPs)
- Quantisation INT8 vs INT4 (FLOPs are defined for FP operations; quantised ops are not "FLOPs")

The only parameters that change FLOPs per token are model size and sequence length (via attention). **None of the deployment parameters this tool measures affect FLOPs.**

FLOPs are useful for:
- Comparing energy efficiency *across different models* (energy per FLOP)
- Comparing *hardware* (FLOP/s per watt across GPU generations)
- Academic papers that need a normalisation baseline

FLOPs are *not* useful for:
- Comparing deployment configurations on the *same model* (this tool's primary use case)
- Understanding whether vLLM or PyTorch is more energy-efficient (both execute the same FLOPs)
- Diagnosing why batch_size=32 uses less energy per token than batch_size=1 (FLOPs are identical)

Recent literature explicitly argues against FLOPs as a primary inference metric:
- "The bottleneck is bandwidth -- not FLOPs" (arXiv:2503.08311, "Mind the Memory Gap")
- "Model FLOPs Utilisation (MFU) better reflects arithmetic saturation and correlates with dynamic power" (arXiv:2507.11417) -- but MFU requires knowing peak hardware FLOPs, which this tool defers to v2.1
- Databricks' inference engineering guide prioritises tokens/second and TTFT over FLOPs

**Recommendation:**
1. **Demote FLOPs from "primary metric" to "reference metadata".** FLOPs should be stored in results but not surfaced as a primary comparison axis. They are context, not a measurement.
2. **Promote `energy_per_output_token` (joules/token) and `tokens_per_second` as the primary metrics.** These actually vary between deployment configurations and directly answer the tool's research question.
3. **Defer MFU to v2.1 as planned**, but note that MFU is the only FLOPs-derived metric that has diagnostic value for this tool (it shows how well the hardware is utilised, which does vary by backend/config).
4. **Do not report FLOPs in CLI summary output.** Reserve for Parquet export and detailed JSON. Researchers who need it can find it; casual users should not be misled into thinking FLOPs differences explain efficiency differences.

**Confidence:** HIGH (the mathematical argument is irrefutable; peer evidence supports the conclusion)

**Sources:**
- [The Real Cost of LLM Inference: Memory Bandwidth, Not FLOPs](https://dev.to/avik12345678/the-real-cost-of-llm-inference-memory-bandwidth-not-flops-3855)
- [Mind the Memory Gap (arXiv:2503.08311)](https://arxiv.org/html/2503.08311v2)
- [Databricks: LLM Inference Performance Engineering Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [MatX: Optimise for inference too, not just training FLOPs](https://matx.com/research/lifetime_llm_cost)

---

### CP-4: Baseline Power Subtraction Methodology Is Under-Specified

**Current state in `.product/`:**
- `designs/result-schema.md` specifies `baseline_power_w` measured during "30-second window immediately before experiment starts"
- `energy_adjusted_j = energy_total_j - (baseline_power_w * duration_sec)`
- Deferred to v2.1

**Problems with this approach:**

1. **Idle baseline != loaded baseline.** GPU idle power (P8 state, ~40W on A100) is very different from "baseline" power during inference (P0 state, fans running, HBM active, ~70-90W on A100 even between batches). Subtracting idle power overcorrects -- the GPU consumes significant power just being in the compute-ready state.

2. **The subtraction formula assumes constant baseline power over the experiment duration.** In reality, baseline power increases as the GPU heats up. A linear correction `baseline_power_w * duration_sec` assumes steady-state idle power throughout, but:
   - At experiment start: GPU is near idle temperature, baseline is lower
   - At experiment end: GPU is at operating temperature, baseline is higher
   - The error grows with experiment duration

3. **No peer tool publishes baseline-corrected energy** (as the result schema doc itself notes). This is not because it is too hard -- it is because the correction introduces more uncertainty than it removes for most use cases. The ML.ENERGY Benchmark reports raw energy because:
   - Relative comparisons (A vs B on the same hardware) cancel out baseline power
   - Absolute energy figures are useful for cost estimation, where total power matters

4. **When baseline subtraction IS useful:** Comparing across hardware with very different idle power (consumer GPU at 15W vs A100 at 45W). But this is a cross-hardware comparison, not a deployment-parameter comparison (this tool's primary use case).

**Recommendation:**
- Keep `baseline_power_w` as optional metadata (useful for researchers)
- Do NOT make `energy_adjusted_j` a primary or prominently displayed metric
- Instead, report `mean_power_draw_w` during the measurement window (this naturally captures the loaded power level and is directly comparable across configurations)
- If baseline correction is desired, measure it with the GPU in P0 state (loaded but idle between batches), not P8 (fully idle). This requires a brief "hold at P0" period after warmup.

**Confidence:** MEDIUM-HIGH (physics reasoning is sound; no peer validation of the proposed correction exists precisely because no peer does it)

---

### CP-5: Bootstrap CI Methodology Is Under-Specified

**Current state in `.product/`:**
- `designs/result-schema.md` specifies `n_bootstrap: int = 1000` with percentile CIs (2.5th/97.5th)
- Requires `n >= 30` samples
- Deferred to v2.1

**Issues:**

1. **Percentile vs BCa:** The design specifies simple percentile bootstrap, not bias-corrected and accelerated (BCa). For skewed distributions (which energy measurements typically are -- right-skewed due to occasional GC pauses, thermal events, etc.), percentile bootstrap has poor coverage. BCa corrects for both bias and skewness:
   - Percentile: adequate for symmetric distributions
   - BCa: recommended when data may be skewed (Monte Carlo studies show BCa achieves nominal coverage with n >= 20, while percentile requires n >= 50+ for equivalent coverage)
   - BCa costs ~2x computation (jackknife for acceleration factor) but is trivial for 30 samples

2. **1,000 resamples is adequate for 95% CIs but tight.** The statistical literature recommends:
   - 1,000 resamples: adequate for 95% CIs (common recommendation)
   - 2,000 resamples: recommended by modern tools (tidymodels, scipy.stats.bootstrap)
   - 10,000 resamples: recommended for 99% CIs or when precision matters
   - For energy measurements at n=30-100 samples, 2,000 resamples is a better default

3. **n >= 30 threshold is conservative for BCa but necessary for percentile.** BCa achieves acceptable coverage with as few as 20 samples. If BCa is adopted, the threshold could be lowered to n >= 20.

4. **What is being bootstrapped matters.** The design says "per-request samples within the experiment". But per-request energy is problematic:
   - Zeus energy counters measure total GPU energy for a window, not per-request energy
   - Per-request energy is derived: `total_energy / n_requests`
   - This gives a *mean*, not a distribution of per-request values
   - For CIs on energy, you need *per-cycle* energy measurements (multiple experiment cycles), not per-request
   - Per-request latency CIs are valid (TTFT, ITL have true per-request measurements)

**Recommendation:**
1. Use **BCa bootstrap** (scipy.stats.bootstrap supports it natively with `method='BCa'`)
2. Increase default to **2,000 resamples**
3. For energy CIs: require **multi-cycle studies** (n_cycles >= 5) and bootstrap over per-cycle energy totals
4. For latency CIs: bootstrap over per-request measurements (TTFT, ITL) as designed
5. Clearly separate "energy CI" (requires multi-cycle) from "latency CI" (available per-request within single experiment)

**Confidence:** HIGH (bootstrap methodology is well-established statistical practice)

**Sources:**
- [Bootstrap confidence interval variations (Pustejovsky, 2025)](https://jepusto.com/posts/Bootstrap-CI-variations/)
- [Bootstrap BCa intervals (SAS)](https://blogs.sas.com/content/iml/2017/07/12/bootstrap-bca-interval.html)
- [scipy.stats.bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
- [TQMP 2025: Bootstrap BCa confidence intervals](https://www.tqmp.org/RegularArticles/vol21-3/p125/p125.pdf)

---

## Moderate Pitfalls

### MP-1: Docker Container Energy Measurement Overhead Is Unquantified

**Current state in `.product/`:**
- `decisions/docker-execution.md` discusses container lifecycle but says nothing about measurement accuracy impact
- `decisions/experiment-isolation.md` uses multiprocessing for local, Docker for v2.2
- No discussion of whether Docker adds energy measurement overhead or distortion

**What the evidence shows:**

Docker containerisation with `--gpus all` uses the NVIDIA Container Toolkit, which passes GPU devices directly to the container via `/dev/nvidia*` device files. This means:
- NVML queries inside the container hit the same hardware counters as bare metal
- `nvmlDeviceGetTotalEnergyConsumption` returns the same value inside and outside Docker
- There is no "Docker overhead" on the NVML energy counter itself

However, Docker adds:
- CPU overhead for the container runtime (~1-3% CPU overhead per benchmarks)
- Additional memory overhead for the overlay filesystem
- Potential NUMA misalignment if container scheduler places processes on wrong NUMA node
- GPU execution time may be slightly longer (~1-3%) due to container syscall overhead

The energy impact: a 1-3% longer execution time at similar power draw means 1-3% more total energy. This is within the NVML measurement uncertainty for most workloads, but for studies comparing Docker vs bare metal, it must be acknowledged.

**Recommendation:**
- Document that Docker adds approximately 1-3% energy overhead due to slightly longer execution time
- Record `runner: "docker"` vs `runner: "local"` in results (already planned)
- Do NOT attempt to correct for Docker overhead -- it is within measurement uncertainty
- Cross-runner comparisons (Docker A100 vs local A100) should note this caveat in the result interpretation

**Confidence:** MEDIUM (Docker GPU overhead benchmarks exist but none specifically measure energy impact; the reasoning is physical)

**Sources:**
- [Docker GPU Passthrough Performance (Brilliance, 2024)](https://jurnal.itscience.org/index.php/brilliance/article/view/6794)
- [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)

---

### MP-2: Multi-GPU Energy Aggregation Ignores Interconnect Power

**Current state in `.product/`:**
- `decisions/multi-gpu.md` specifies `total_energy = sum(per_gpu_energy)` via NVML/Zeus
- No mention of NVLink, NVSwitch, or PCIe interconnect power

**What the evidence shows:**

NVML energy counters measure per-GPU board power. They do NOT measure:
- NVLink power consumption (NVLink 4.0 on H100: up to 50W per link at full bandwidth)
- NVSwitch power (on DGX systems: ~100W per switch)
- PCIe root complex power
- Host CPU power for memory copies and synchronisation

For tensor parallelism, the all-reduce operations between GPUs consume significant NVLink bandwidth. The energy consumed by NVLink is *not* captured by per-GPU NVML counters. This means:
- Summing per-GPU energy **understates** the true total energy of a TP experiment
- The understatement is proportional to NVLink traffic, which increases with TP degree
- For TP=4 on DGX A100 with NVLink 3.0: estimated 10-30W of NVLink power per GPU not captured
- This is 3-10% of total system energy, depending on workload

**Recommendation:**
- Document the limitation: "Per-GPU NVML energy does not include interconnect power. Multi-GPU experiments understate total energy by approximately 3-10% depending on NVLink traffic."
- Add `interconnect_energy_note: str` to `MultiGPUMetrics` explaining this limitation
- For publishable results, recommend researchers note this limitation in their methodology section
- In v2.3+ (multi-GPU sweep), consider adding a correction factor based on NVLink bandwidth utilisation (if measurable)

**Confidence:** MEDIUM (NVLink power consumption figures are from NVIDIA marketing materials and are approximate; no peer tool measures or corrects for this)

**Sources:**
- [NVIDIA NVLink and NVSwitch (NVIDIA Blog)](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/)
- [MLPerf Power: Myth #1 -- Measuring ML Components in Isolation is Insufficient](https://arxiv.org/html/2410.12032v2)

---

### MP-3: Warmup Reduced-Output Strategy May Not Warm KV Cache Properly

**Current state in `.product/`:**
- `decisions/warmup-strategy.md` specifies 5 warmup runs at `max_new_tokens=2`
- Rationale cites optimum-benchmark's pattern

**The problem:**

The reduced-output warmup (2 tokens) only triggers:
- CUDA kernel JIT compilation (if using `torch.compile`)
- Prefill path execution
- One or two decode steps

It does NOT trigger:
- KV cache at the actual operational size (2-token decode vs 100-token decode uses different KV cache allocation patterns)
- Memory allocation for the full output sequence
- For vLLM: continuous batching scheduler at operational batch depth
- For TensorRT-LLM: engine pages at operational capacity

This means the first "real" measurement run may still encounter:
- KV cache page allocation overhead (vLLM paged attention)
- Memory fragmentation from the mismatch between warmup and measurement allocation patterns
- Scheduler warm-up in vLLM (first full-batch run triggers different scheduling path than 2-token runs)

**optimum-benchmark uses reduced output for speed,** not because it is methodologically ideal. For a tool prioritising scientific rigour over benchmark speed, full-length warmup is defensible.

**Recommendation:**
- For latency benchmarks: reduced-output warmup is acceptable (JIT is the primary concern)
- For energy benchmarks: use full-length warmup runs (the thermal stabilisation concern from CP-2 aligns with this -- full-length runs provide both JIT warmup AND thermal conditioning)
- Make warmup strategy configurable: `warmup_mode: "fast" | "full"` with `fast` = reduced output, `full` = actual workload. Default to `full` for energy measurements.

**Confidence:** MEDIUM (theoretical reasoning; no peer tool has studied the measurement impact of reduced vs full warmup on KV cache allocation)

---

### MP-4: CodeCarbon Accuracy Claims Need Revision

**Current state in `.product/`:**
- Multiple documents cite CodeCarbon accuracy as "~15%" or "~10-20%"
- These appear to originate from the PMC comparison paper that found "up to 400% variation between tools"

**What the evidence shows:**

CodeCarbon v3.2.2 uses *the same NVML readings* as Zeus. The accuracy difference is not in the sensor but in the methodology:

| Factor | Zeus (ZeusMonitor) | CodeCarbon (EmissionsTracker) |
|--------|-------------------|-------------------------------|
| GPU energy source | `nvmlDeviceGetTotalEnergyConsumption` (Volta+) | `nvmlDeviceGetPowerUsage` polled at intervals |
| Measurement approach | Hardware energy counter delta | Power sampling + trapezoidal integration |
| Default polling interval | N/A (hardware counter) | 15 seconds (configurable) |
| Accuracy driver | Hardware counter resolution | Sampling frequency vs workload variability |
| CPU-GPU sync | Explicit (`torch.cuda.synchronize()`) | None (no framework integration) |

The "15% inaccuracy" of CodeCarbon comes from:
1. **15-second default polling interval** -- misses power transients (the Burtscher paper's core finding)
2. **No CPU-GPU synchronisation** -- GPU work may not be complete when power is sampled
3. **TDP fallback** -- when NVML is unavailable, CodeCarbon estimates from TDP tables (20-30% error)

If CodeCarbon is configured with high polling frequency (e.g., 1 second) and NVML is available, its accuracy approaches that of direct NVML polling (~5-10% for sustained workloads). The gap with Zeus is real but smaller than documented.

**Recommendation:**
- Rewrite accuracy claims: "Zeus energy counter: ~1-5% for sustained loads on Volta+. CodeCarbon NVML polling: ~5-15% depending on polling interval and workload duration. CodeCarbon TDP fallback: ~20-30%."
- When CodeCarbon is the energy backend, automatically set `measure_power_secs` to 1 (not the default 15)
- Document that CodeCarbon's CO2 estimation layer adds no measurement error -- it is a multiplication by a carbon intensity constant

**Confidence:** HIGH (CodeCarbon source code confirms NVML usage; polling interval is configurable)

**Sources:**
- [CodeCarbon Methodology](https://mlco2.github.io/codecarbon/methodology.html)
- [PMC: Energy Tools Comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661046/)

---

### MP-5: No Minimum Measurement Duration Specified

**Current state in `.product/`:**
- No minimum duration is specified for the measurement window
- The design allows single-prompt experiments with no lower bound on duration

**Why this matters:**

The Burtscher paper demonstrated that NVML power readings on A100/H100 use 100ms update periods with only 25% sampling coverage. For very short measurement windows:

| Measurement Duration | Expected NVML Samples | Accuracy |
|---------------------|----------------------|----------|
| < 100ms | 0-1 | **Unreliable** -- may miss entirely |
| 100ms - 1s | 1-10 | **High variance** -- insufficient sampling |
| 1s - 10s | 10-100 | **Moderate** -- adequate for relative comparison |
| > 10s | 100+ | **Good** -- statistical averaging reduces error |

For the hardware energy counter (`nvmlDeviceGetTotalEnergyConsumption`), the situation is better but still has a floor: the counter resolution is millijoules, so experiments consuming < 100 mJ may have significant quantisation error.

**Recommendation:**
- Set minimum measurement duration to **10 seconds** (or loop the workload until 10s reached, matching MLPerf's approach)
- Alternatively, require `n >= 10` prompts per experiment to naturally exceed the minimum
- Flag results where measurement duration < 10 seconds with a `short_measurement_warning: true`

**Confidence:** HIGH (NVML sampling behaviour is empirically documented)

---

## Minor Pitfalls

### mP-1: CPU-GPU Synchronisation Not Explicitly Required

**Current state in `.product/`:**
- The existing research docs reference Zeus's `sync_execution_with` parameter
- But `decisions/warmup-strategy.md` and the energy measurement design do not specify a synchronisation requirement

**The problem:** Without `torch.cuda.synchronize()` (or equivalent) before starting and ending the measurement window, the CPU may start/stop the energy measurement timer while GPU work is still in flight. This introduces up to 10-50ms of timing error per measurement boundary.

The ML.ENERGY blog explicitly states: "To accurately measure GPU time and energy consumption, make the CPU wait for GPU work to complete."

**Recommendation:** Add an explicit requirement: energy measurement windows MUST call `torch.cuda.synchronize()` (or backend equivalent) before `begin_window()` and `end_window()`. This is already handled by Zeus when `sync_execution_with="torch"`, but must be explicitly required in the measurement protocol, not left as a Zeus implementation detail.

**Confidence:** HIGH (documented by Zeus and ML.ENERGY)

---

### mP-2: MIG Energy Measurement Gives Whole-GPU, Not Slice-Level Readings

**Current state in `.product/`:** Mentioned briefly in the old PITFALLS.md but not addressed in any decision document.

NVML energy counters report at the *physical GPU* level, not the MIG instance level. If other workloads are running on sibling MIG slices, the energy reading includes their contribution. There is no correction for this.

**Recommendation:** Detect MIG mode at pre-flight. Warn users that energy readings include all MIG slices on the physical GPU. Recommend running without MIG or ensuring no other workloads on sibling slices.

**Confidence:** HIGH (documented NVML limitation)

---

### mP-3: Power Limit Capping Affects Energy Measurements

GPU power limits (`nvidia-smi -pl <watts>`) constrain the GPU's power draw. If a power limit is set below the GPU's natural draw for a workload, the GPU throttles to stay within the limit. This changes both energy and latency measurements. The tool does not detect or record the active power limit.

**Recommendation:** Record `gpu_power_limit_w` in `EnvironmentSnapshot` via `nvmlDeviceGetPowerManagementLimit()`. Flag if it differs from the default limit (`nvmlDeviceGetPowerManagementDefaultLimit()`). This is a single NVML call per GPU, zero overhead.

**Confidence:** HIGH (NVML API for this is well-documented)

---

## Challenged Decisions: Summary Table

| Decision | Document | Current State | Verdict | Recommended Action |
|----------|----------|---------------|---------|-------------------|
| NVML ~5% accuracy | `designs/energy-backends.md` | Fixed percentage | **Wrong** | Replace with conditional accuracy formula |
| CodeCarbon ~15% accuracy | `designs/energy-backends.md` | Fixed percentage | **Misleading** | Rewrite as range dependent on config |
| 30s thermal floor | `decisions/warmup-strategy.md` | 30 seconds at idle | **Under-calibrated** | Increase to 60s under load; or use full-length warmup runs |
| 5 warmup runs, 2 tokens | `decisions/warmup-strategy.md` | Reduced-output warmup | **Questionable for energy** | Use full-length runs for energy benchmarks |
| FLOPs as primary metric | `decisions/flops-estimation.md` | Core metric | **Misleading for this tool** | Demote to reference metadata |
| Baseline power subtraction | `designs/result-schema.md` | Simple linear correction | **Over-simplified** | Measure in P0 state, not P8; do not make primary |
| Bootstrap 1000 percentile | `designs/result-schema.md` | Percentile bootstrap | **Sub-optimal** | Use BCa with 2000 resamples |
| Multi-GPU sum | `decisions/multi-gpu.md` | Sum per-GPU NVML | **Understates** | Document interconnect power gap |
| Process isolation | `decisions/experiment-isolation.md` | multiprocessing.spawn | **Sound** | No changes needed |
| CO2 estimation | `decisions/carbon-intensity.md` | CodeCarbon delegation | **Sound** | No changes needed |
| Config hash | `designs/result-schema.md` | SHA-256 of config | **Sound** | No changes needed |
| Steady state window | `designs/result-schema.md` | Explicit field | **Sound** | No changes needed |

---

## Unaddressed Areas That Should Be

### UA-1: No Persistence Mode Requirement

NVIDIA GPUs not in persistence mode (`nvidia-smi -pm 1`) add ~100ms NVML initialisation overhead per query. More importantly, without persistence mode, the GPU drops to a lower power state between experiments, requiring re-stabilisation. All serious benchmarking assumes persistence mode is enabled.

**Recommendation:** Check persistence mode at pre-flight. Warn (or error) if not enabled. This is a single NVML call.

---

### UA-2: No ECC Memory Status Recording

ECC (Error Correcting Code) memory has a ~3-5% performance overhead and corresponding energy impact. Whether ECC is enabled should be recorded in `EnvironmentSnapshot`. On datacenter GPUs (A100, H100) ECC is enabled by default; disabling it changes both performance and energy characteristics.

**Recommendation:** Record `ecc_enabled: bool` in `EnvironmentSnapshot` via `nvmlDeviceGetEccMode()`.

---

### UA-3: No GPU Clock Frequency Recording

The GPU's actual clock frequency during measurement affects both latency and energy. Boost clocks vary based on thermal state, power limit, and other GPUs in the system (power budget sharing on DGX). Without recording the actual achieved clock frequency, reproducibility is compromised.

**Recommendation:** Record `gpu_clock_mhz` (actual, not max) at measurement start and end via `nvmlDeviceGetClockInfo()`.

---

### UA-4: No CPU Governor Check

On Linux, the CPU frequency governor affects both CPU energy and can introduce measurement variance. The `performance` governor provides consistent results; `powersave` or `ondemand` introduce frequency scaling delays that affect timing measurements.

**Recommendation:** Check and record `/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` in `EnvironmentSnapshot`.

---

## Phase-Specific Warnings

| Phase / Version | Likely Pitfall | Mitigation |
|-----------------|---------------|------------|
| v2.0 Energy measurement | NVML accuracy claims challenged by users reading the same papers | Pre-empt: document conditional accuracy in user-facing docs |
| v2.0 Warmup | 30s thermal floor insufficient for H100 under heavy load | Implement 60s loaded warmup as default |
| v2.0 Results | FLOPs prominently displayed; users confused why "same FLOPs, different energy" | Demote FLOPs; lead with energy/token |
| v2.1 Baseline correction | Simple linear subtraction criticised for ignoring thermal drift | Measure baseline in P0 state; document as estimate |
| v2.1 Confidence intervals | Percentile bootstrap with skewed energy data; CIs have poor coverage | Use BCa; require multi-cycle for energy CIs |
| v2.2 Docker | Users compare Docker vs local energy and attribute difference to Docker overhead | Document ~1-3% overhead; recommend consistent runner within a study |
| v2.3 Multi-GPU sweep | NVLink energy not captured; users underestimate TP energy cost | Document interconnect gap; note in methodology |

---

## Sources

### Peer-Reviewed Papers
- [Part-time Power Measurements: nvidia-smi's Lack of Attention (Burtscher et al., 2023)](https://arxiv.org/html/2312.02741v2) -- NVML accuracy
- [The ML.ENERGY Benchmark (NeurIPS D&B 2025)](https://arxiv.org/html/2505.06371v1) -- steady-state methodology
- [MLPerf Power Benchmark (IEEE HPCA 2025)](https://arxiv.org/html/2410.12032v2) -- 60s minimum, full-system measurement
- [Mind the Memory Gap (arXiv:2503.08311)](https://arxiv.org/html/2503.08311v2) -- FLOPs vs memory bandwidth
- [Quantifying LLM Inference Energy via Simulations (arXiv:2507.11417)](https://arxiv.org/html/2507.11417v1) -- MFU power model
- [Verified Instruction-Level GPU Energy Consumption (ACM CF 2020)](https://dl.acm.org/doi/10.1145/3387902.3392613) -- NVML verification
- [PMC: Energy Tools Comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661046/) -- cross-tool accuracy
- [Bootstrap BCa intervals (TQMP 2025)](https://www.tqmp.org/RegularArticles/vol21-3/p125/p125.pdf)

### Official Documentation
- [NVIDIA NVML API Reference Guide](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html)
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)
- [Zeus Project Documentation](https://ml.energy/zeus/measure/)
- [NeurIPS 2025 Tutorial: Accurately Benchmarking Power & Energy](https://ml.energy/tutorials/neurips25/session-1.html)
- [CodeCarbon Methodology](https://mlco2.github.io/codecarbon/methodology.html)
- [scipy.stats.bootstrap](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

### Peer Tool Implementations
- [optimum-benchmark (HuggingFace)](https://github.com/huggingface/optimum-benchmark)
- [AIEnergyScore v2 (HuggingFace)](https://huggingface.co/blog/sasha/ai-energy-score-v2)
- [vLLM benchmark tooling](https://docs.vllm.ai/en/latest/)

### Industry
- [Databricks: LLM Inference Performance Engineering](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [NVIDIA NVLink/NVSwitch Technical Blog](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/)
- [Docker GPU Passthrough Benchmarks](https://jurnal.itscience.org/index.php/brilliance/article/view/6794)

---

## Confidence Assessment

| Area | Confidence | Reason |
|------|-----------|--------|
| NVML accuracy characterisation | MEDIUM-HIGH | +/-5W spec is documented; energy counter accuracy is not formally specified by NVIDIA |
| Thermal floor calibration | HIGH | MLPerf 60s minimum is peer-reviewed; physics of thermal ramp are well understood |
| FLOPs metric critique | HIGH | Mathematical argument is deterministic; extensive literature support |
| Bootstrap methodology | HIGH | Well-established statistical practice with extensive literature |
| Docker energy overhead | MEDIUM | Reasoning from GPU passthrough benchmarks; no direct energy measurement study |
| NVLink power gap | MEDIUM | NVLink TDP figures from NVIDIA marketing; no peer-measured data |
| CodeCarbon accuracy | HIGH | Source code confirms NVML usage; polling behaviour is configurable and documented |
