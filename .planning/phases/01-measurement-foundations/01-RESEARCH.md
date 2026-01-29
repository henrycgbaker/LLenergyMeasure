# Phase 1: Measurement Foundations - Research

**Researched:** 2026-01-29
**Domain:** GPU energy measurement and system metadata collection for LLM inference benchmarking
**Confidence:** HIGH

## Summary

Phase 1 enhances measurement accuracy for LLM inference energy benchmarking by addressing three critical gaps: baseline power contamination (15-30% systematic overestimation), missing environmental metadata (prevents reproducibility), and lack of thermal monitoring (silent performance degradation). This is a **brownfield project** extending existing infrastructure, not greenfield development.

The existing codebase provides strong foundations: `GPUUtilisationSampler` demonstrates the NVML sampling pattern, Pydantic domain models in `domain/metrics.py` and `domain/experiment.py` define the current schema, `results/exporters.py` handles CSV export, and the config system in `config/models.py` already manages experiment parameters. Phase 1 extends these patterns rather than introducing new architectural concepts.

Standard approach uses **nvidia-ml-py (13.590.48)** — the official NVIDIA Management Library bindings for GPU monitoring. Baseline power measurement follows ML.ENERGY best practices: measure idle power pre-experiment, subtract conservatively (`adjusted_energy = total_energy - baseline_watts × duration`), store both raw and adjusted values. Thermal throttling detection uses NVML's `nvmlDeviceGetCurrentClocksThrottleReasons` API to distinguish thermal vs power vs other throttling causes. Time-series sampling extends the existing `GPUUtilisationSampler` pattern to capture power/memory/utilisation at configurable intervals (1-10Hz). Warmup convergence replaces fixed iteration counts with coefficient-of-variation based detection (CV < 5% threshold).

**Primary recommendation:** Extend existing patterns conservatively. Schema changes must be additive (backwards compatible), new fields should use nested structures (`energy: {raw, adjusted, baseline}`), and time-series data should be optional separate files (not bloating main results JSON).

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nvidia-ml-py | 13.590.48+ | GPU monitoring via NVML | Official NVIDIA bindings, actively maintained, replaces deprecated pynvml, January 2026 release |
| pydantic | 2.0+ | Domain models and validation | Already used throughout codebase, schema v3 changes are model updates |
| numpy | latest | Statistical calculations | CV computation for warmup convergence, percentile calculations for time-series |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| psutil | latest | CPU governor detection | Environment metadata collection (already may be dependency) |
| tqdm | 4.66+ | Progress bars | Warmup convergence feedback (already dependency) |

### Already in Codebase
| Module | Purpose | How Phase 1 Uses It |
|--------|---------|---------------------|
| `core/gpu_utilisation.py` | Background NVML sampling | Reference implementation for power/thermal sampler |
| `domain/metrics.py` | Metrics models | Extend with baseline power fields, thermal flags |
| `domain/experiment.py` | Results models | Schema v3 with environment metadata |
| `results/exporters.py` | CSV export | Add new columns for extended metrics |
| `config/models.py` | Configuration | Add warmup convergence config, baseline config |

**Installation:**
```bash
# Already satisfied by existing dependencies
pip install pydantic>=2.0 numpy tqdm>=4.66
# nvidia-ml-py already pinned at >=12.0.0 in pyproject.toml, bump to >=13.590.48
```

## Architecture Patterns

### Current Codebase Structure (What We're Extending)
```
src/llenergymeasure/
├── core/
│   ├── gpu_utilisation.py         # ← Reference: NVML sampling pattern
│   ├── energy_backends/
│   │   └── codecarbon.py          # ← Extends: baseline measurement hooks
│   └── inference.py               # ← Extends: warmup convergence logic
├── domain/
│   ├── metrics.py                 # ← Extends: add thermal/baseline fields
│   └── experiment.py              # ← Extends: schema v3, environment metadata
├── results/
│   ├── exporters.py               # ← Extends: CSV columns for new metrics
│   └── aggregation.py             # ← Extends: aggregate time-series
└── config/
    └── models.py                  # ← Extends: warmup/baseline config
```

### Pattern 1: Extend GPUUtilisationSampler for Power/Thermal Monitoring
**What:** Existing `GPUUtilisationSampler` demonstrates thread-safe NVML sampling. Create parallel `PowerThermalSampler` following identical pattern.

**When to use:** Time-series power/memory/thermal data collection during inference.

**Example:**
```python
# Source: Existing pattern in core/gpu_utilisation.py
class PowerThermalSampler:
    """Background sampler for power, memory, and thermal metrics.

    Follows GPUUtilisationSampler pattern: thread-safe, graceful degradation.
    """

    def __init__(self, device_index: int = 0, sample_interval_ms: int = 100):
        self._device_index = device_index
        self._sample_interval = sample_interval_ms / 1000.0
        self._samples: list[PowerThermalSample] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def _sample_loop(self) -> None:
        """Background sampling using pynvml."""
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)

        while self._running:
            try:
                # Power
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                power_w = power_mw / 1000.0

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_mb = mem_info.used / (1024**2)

                # Thermal throttling detection
                throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
                thermal_throttle = bool(throttle_reasons & pynvml.nvmlClocksThrottleReasonThermal)

                self._samples.append(PowerThermalSample(
                    timestamp=time.perf_counter(),
                    power_w=power_w,
                    memory_used_mb=mem_used_mb,
                    thermal_throttle=thermal_throttle,
                ))
            except pynvml.NVMLError:
                pass  # Skip failed samples
            time.sleep(self._sample_interval)
```

**Key implementation notes:**
- Follow existing `GPUUtilisationSampler.__enter__/__exit__` context manager pattern
- Graceful degradation if NVML unavailable (return empty samples, don't crash)
- Thread safety: daemon thread with `_running` flag for clean shutdown
- Sample at 100ms (10Hz) default, configurable 1-10Hz via config

### Pattern 2: Baseline Power Measurement (Pre-Experiment Hook)
**What:** Measure idle GPU power before experiment starts. Cache per terminal session to avoid repeated measurement overhead.

**When to use:** Before every experiment, check cache first.

**Example:**
```python
# Source: ML.ENERGY best practices
def measure_baseline_power(
    device_index: int,
    duration_sec: float = 30.0,
    cache_ttl_sec: float = 3600.0,
) -> float:
    """Measure idle GPU baseline power with session caching.

    Args:
        device_index: CUDA device index.
        duration_sec: Measurement duration (default 30s).
        cache_ttl_sec: Cache validity in seconds (default 1 hour).

    Returns:
        Mean baseline power in Watts.
    """
    # Check cache (implementation: pickle to /tmp/baseline_<gpu_id>.pkl)
    cached = _load_cached_baseline(device_index)
    if cached and (time.time() - cached.timestamp) < cache_ttl_sec:
        return cached.power_w

    # Measure: ensure GPU idle (no CUDA kernels running)
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    samples = []
    start = time.time()
    while time.time() - start < duration_sec:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        samples.append(power_mw / 1000.0)
        time.sleep(0.1)  # 10Hz sampling

    baseline_w = sum(samples) / len(samples)
    _cache_baseline(device_index, baseline_w)  # Persist to temp file
    return baseline_w

def adjust_energy_for_baseline(
    total_energy_j: float,
    baseline_power_w: float,
    duration_sec: float,
) -> float:
    """Conservative baseline subtraction.

    Returns:
        Adjusted energy (never negative).
    """
    baseline_energy_j = baseline_power_w * duration_sec
    adjusted = total_energy_j - baseline_energy_j
    return max(0.0, adjusted)  # Conservative: floor at zero
```

**Integration points:**
- Hook into `ExperimentOrchestrator.run()` pre-inference
- Config: `baseline.enabled` (default True), `baseline.required` (default False)
- Results: Store `baseline_power_w`, `baseline_timestamp`, `baseline_method` metadata
- Failure handling: If measurement fails and `baseline.required=False`, warn and continue with raw values only

### Pattern 3: Warmup Convergence Detection
**What:** Replace fixed `num_warmup_runs` with dynamic convergence based on coefficient of variation (CV).

**When to use:** Inference engine warmup phase, before metric collection starts.

**Example:**
```python
# Source: Research best practices
def warmup_until_converged(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_prompts: int = 50,
    cv_threshold: float = 0.05,  # 5%
    window_size: int = 5,
) -> tuple[bool, float]:
    """Run warmup until latency CV stabilises.

    Args:
        model: Loaded model.
        tokenizer: Tokenizer.
        prompt: Warmup prompt.
        max_prompts: Maximum iterations (safety cap).
        cv_threshold: Target CV (default 5%).
        window_size: Rolling window for CV calculation.

    Returns:
        Tuple of (converged: bool, final_cv: float).
    """
    latencies = []
    converged = False

    with tqdm(total=max_prompts, desc="Warmup", unit="prompt") as pbar:
        for i in range(max_prompts):
            # Run inference, measure latency
            start = time.perf_counter()
            _ = model.generate(**tokenizer(prompt, return_tensors="pt"))
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)

            # Check convergence after minimum window
            if len(latencies) >= window_size:
                recent = latencies[-window_size:]
                mean = sum(recent) / len(recent)
                std = (sum((x - mean)**2 for x in recent) / len(recent))**0.5
                cv = std / mean if mean > 0 else 0.0

                pbar.set_postfix(cv=f"{cv:.1%}", target=f"{cv_threshold:.1%}")

                if cv < cv_threshold:
                    converged = True
                    break

            pbar.update(1)

    final_cv = cv if len(latencies) >= window_size else 1.0
    return converged, final_cv
```

**Config integration:**
```yaml
# New fields in config/models.py
warmup:
  enabled: true
  convergence_detection: true  # false = fallback to fixed iterations
  cv_threshold: 0.05           # 5%
  max_prompts: 50              # Safety cap
  window_size: 5               # Rolling window
```

### Pattern 4: Schema v3 with Nested Energy Fields
**What:** Extend `domain/experiment.py` models with backwards-compatible nested structure.

**When to use:** Results schema changes for baseline-adjusted energy.

**Example:**
```python
# Source: domain/experiment.py (extend EnergyMetrics)
class EnergyBreakdown(BaseModel):
    """Energy breakdown with baseline adjustment."""

    raw_j: float = Field(..., description="Total measured energy (Joules)")
    adjusted_j: float | None = Field(
        default=None,
        description="Baseline-adjusted energy (raw - baseline*duration)"
    )
    baseline_power_w: float | None = Field(
        default=None,
        description="Measured baseline power (Watts)"
    )
    baseline_method: str | None = Field(
        default=None,
        description="Baseline measurement method (cached, fresh, unavailable)"
    )
    baseline_timestamp: datetime | None = Field(
        default=None,
        description="When baseline was measured"
    )

class EnergyMetrics(BaseModel):
    """Energy consumption metrics with baseline adjustment."""

    # Legacy field (backwards compatible)
    total_energy_j: float = Field(..., description="Total energy (raw)")

    # New nested structure (schema v3)
    energy_breakdown: EnergyBreakdown | None = Field(
        default=None,
        description="Detailed energy breakdown with baseline adjustment"
    )

    # ... existing fields ...
```

**Migration strategy:**
- Schema version bumps to v3 in `constants.py`
- Old results readers check `schema_version`, adapt field access
- CSV export flattens: `energy_breakdown_raw_j`, `energy_breakdown_adjusted_j`

### Pattern 5: Optional Time-Series Export
**What:** Save time-series data as separate file, not embedded in main results JSON.

**When to use:** User enables `--save-timeseries` flag or YAML config.

**Example:**
```python
# Source: results/exporters.py (new function)
def export_timeseries_to_json(
    samples: list[PowerThermalSample],
    output_path: Path,
    experiment_id: str,
) -> Path:
    """Export time-series samples to separate JSON file.

    Args:
        samples: Power/thermal/memory samples.
        output_path: Output directory.
        experiment_id: Experiment identifier.

    Returns:
        Path to timeseries file.
    """
    timeseries_file = output_path / f"{experiment_id}_timeseries.json"

    data = {
        "experiment_id": experiment_id,
        "sample_count": len(samples),
        "sample_interval_ms": 100,  # From config
        "samples": [
            {
                "timestamp": s.timestamp,
                "power_w": s.power_w,
                "memory_used_mb": s.memory_used_mb,
                "thermal_throttle": s.thermal_throttle,
            }
            for s in samples
        ]
    }

    with timeseries_file.open("w") as f:
        json.dump(data, f, indent=2)

    return timeseries_file
```

**File layout:**
```
results/
├── raw/
│   └── exp_20260129_120000/
│       ├── process_0.json              # Main results (as before)
│       └── process_0_timeseries.json   # Time-series (optional)
└── aggregated/
    ├── exp_20260129_120000.json        # Aggregated results
    └── exp_20260129_120000_timeseries.json  # Aggregated timeseries
```

### Anti-Patterns to Avoid

**❌ Embedding time-series in main results JSON**
- Why bad: Bloats file size (10Hz for 5min = 3000 samples), slows parsing
- Do instead: Separate file, reference by experiment_id

**❌ Negative adjusted energy values**
- Why bad: Physically meaningless, breaks downstream calculations
- Do instead: Conservative floor at zero, document in methodology

**❌ Hard-coded warmup CV threshold**
- Why bad: Optimal threshold varies by hardware (noisy vs stable)
- Do instead: Configurable with sensible default, document rationale

**❌ Breaking schema changes without migration**
- Why bad: Invalidates existing results, breaks tooling
- Do instead: Additive changes only, bump schema version, provide migration guide

**❌ Synchronous baseline measurement blocking experiments**
- Why bad: 30s overhead per experiment if not cached
- Do instead: Session-level caching, async pre-measurement

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| NVML GPU monitoring | Custom C bindings | nvidia-ml-py | Official NVIDIA bindings, handles CUDA version compatibility, error codes documented |
| Statistical CV calculation | Manual variance formulas | numpy.std / numpy.mean | Handles edge cases (zero mean, single sample), numerically stable |
| Time-series aggregation | Custom percentile logic | numpy.percentile | Correct implementation (linear interpolation), handles empty arrays |
| Progress bars during warmup | Custom print statements | tqdm | Clean terminal output, handles Jupyter notebooks, rate limiting built-in |
| Config validation | Manual checks | Pydantic validators | Type safety, clear error messages, already used throughout codebase |

**Key insight:** GPU measurement has subtle failure modes (NVML context conflicts, MIG limitations, permission errors). Official nvidia-ml-py handles these gracefully, custom bindings will hit production edge cases.

## Common Pitfalls

### Pitfall 1: Idle Power Contamination
**What goes wrong:** Measured energy includes baseline GPU power (idle state), causing 15-30% systematic overestimation. Longer experiments penalised disproportionately. Cross-configuration comparisons invalid.

**Why it happens:** GPUs consume 30-50W idle (high-end datacenter), NVML reports total power not delta. CodeCarbon sums total power over duration.

**How to avoid:**
1. Measure baseline power pre-experiment (30s sampling, GPU idle)
2. Subtract conservatively: `adjusted = total - (baseline × duration)`
3. Store both raw and adjusted values for auditability
4. Floor at zero (never negative energy)
5. Document methodology in results metadata

**Warning signs:**
- Energy scales linearly with time even when throughput constant
- Batch size has negligible impact on energy (should reduce per-token energy)
- Low-utilisation configs show unexpectedly high energy

**Code location:** Extends `core/energy_backends/codecarbon.py`, hooks into `orchestration/runner.py`.

### Pitfall 2: NVML Sampling Blind Spots
**What goes wrong:** NVML samples at max 66.7Hz, but typical sampling only captures 25% of runtime. Short experiments (<1 min) show high variance, up to 73% average error. Miss power spikes during prefill phase.

**Why it happens:** NVML updates internal state asynchronously, `nvmlDeviceGetPowerUsage()` returns cached value. High-frequency updates bottleneck system.

**How to avoid:**
1. Sample at 100ms (10Hz) for <5min experiments, 1Hz for longer
2. Multi-cycle execution for statistical robustness (Phase 2)
3. Record sample count and report CV to flag high-variance measurements
4. Never rely on single-cycle short experiments for energy claims

**Warning signs:**
- High CV (>10%) on repeated experiments with identical config
- Sample count << expected (e.g., 50 samples for 5min experiment at 10Hz)
- Energy measurements differ by >20% across cycles

**Code location:** `core/gpu_utilisation.py` (extend sampling pattern), `results/aggregation.py` (CV reporting).

### Pitfall 3: Thermal Throttling Goes Undetected
**What goes wrong:** GPU throttles silently during long experiments, reducing clocks by 2x+ and inflating energy-per-token. Sequential experiments show degrading performance (thermal accumulation). First run fast, tenth run slow.

**Why it happens:** Sustained load elevates temperature, GPU reduces clocks to stay within thermal limits. User unaware unless monitoring.

**How to avoid:**
1. Monitor `nvmlDeviceGetCurrentClocksThrottleReasons()` during inference
2. Log throttle events with timestamps, duration, cause (thermal vs power vs other)
3. Flag experiments where throttling detected: `thermal_throttle_detected: true`
4. Record thermal state in environment metadata (temperature, power limits, fan speed)
5. Warn user post-experiment, suggest thermal gap between runs

**Warning signs:**
- Throughput degrades over campaign duration
- Power consumption increases while throughput decreases (efficiency drops)
- NVML reports throttle reasons but experiment proceeds silently

**Code location:** Extend `PowerThermalSampler`, add fields to `domain/experiment.py`, display in CLI output.

### Pitfall 4: Warmup Convergence Never Reached
**What goes wrong:** Noisy hardware prevents CV from stabilising below threshold, warmup runs to max_prompts cap, user thinks warmup incomplete.

**Why it happens:** Shared GPUs with background load, thermal fluctuations, scheduler preemption cause variance. 5% CV unrealistic on some systems.

**How to avoid:**
1. Make CV threshold configurable (`warmup.cv_threshold`, default 5%)
2. Always cap iterations (`warmup.max_prompts`, default 50)
3. Log non-convergence as warning, not error: `warmup_converged: false, final_cv: 0.08`
4. Progress bar shows current vs target CV, user sees if converging slowly
5. Document that fixed iterations (fallback) acceptable on noisy systems

**Warning signs:**
- Warmup always hits max cap, never converges
- CV oscillates around threshold (9%, 4%, 7%, 5.5%, ...)
- User on shared GPU or system with high background load

**Code location:** `core/inference.py` warmup logic, `config/models.py` warmup config.

### Pitfall 5: CSV Export Column Explosion
**What goes wrong:** Flattening nested structures creates 50+ columns with prefixes like `energy_breakdown_baseline_adjustment_metadata_timestamp`. CSV unreadable, column order arbitrary.

**Why it happens:** Pydantic models nest deeply, naive flattening produces verbose keys.

**How to avoid:**
1. Use grouped prefixes: `energy_raw_j`, `energy_adjusted_j`, `energy_baseline_w`
2. Omit redundant prefixes (don't repeat parent key in child: `energy_energy_...`)
3. Time-series stays in separate file, NOT flattened to CSV
4. Limit CSV to summary statistics (mean, p95, p99), not all samples
5. Test CSV output readability with Excel/Google Sheets

**Warning signs:**
- CSV has >100 columns
- Column names require horizontal scrolling to read
- Prefixes nested 3+ levels (`parent_child_grandchild_field`)

**Code location:** `results/exporters.py`, `_aggregated_to_row()` function.

### Pitfall 6: Baseline Staleness from Long-Running Sessions
**What goes wrong:** Terminal session left open 24hrs, GPU idle power drifts due to driver updates, thermal state changes, background processes. Cached baseline no longer accurate.

**Why it happens:** Session-level caching uses 1-hour TTL by default, but sessions can last much longer.

**How to avoid:**
1. Document cache TTL clearly (`baseline.cache_ttl`, default 3600s)
2. Allow manual invalidation: `lem baseline measure --force`
3. Log cache age in results metadata: `baseline_cache_age_sec`
4. CLI shows baseline cache status at experiment start
5. Consider shorter TTL (30min) or re-measure per campaign

**Warning signs:**
- Baseline measured 8 hours ago still being used
- Energy measurements drift over time without config changes
- System uptime >> cache TTL (user forgot about long-running shell)

**Code location:** Baseline caching logic in `core/energy_backends/`, CLI display in `cli/experiment.py`.

## Code Examples

### Example 1: NVML Thermal Throttling Detection
```python
# Source: NVML API Reference Guide (Jan 2026)
import pynvml

def detect_thermal_throttle(device_index: int) -> dict[str, bool]:
    """Check if GPU is currently throttling and why.

    Returns:
        Dict with throttle reasons: thermal, power, other.
    """
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)

    return {
        "thermal": bool(reasons & pynvml.nvmlClocksThrottleReasonThermal),
        "power": bool(reasons & pynvml.nvmlClocksThrottleReasonPowerBrake),
        "sw_thermal": bool(reasons & pynvml.nvmlClocksThrottleReasonSwThermalSlowdown),
        "hw_thermal": bool(reasons & pynvml.nvmlClocksThrottleReasonHwThermalSlowdown),
        "hw_power": bool(reasons & pynvml.nvmlClocksThrottleReasonHwPowerBrakeSlowdown),
    }
```

### Example 2: Environment Metadata Collection
```python
# Source: Research best practices + codebase patterns
def collect_environment_metadata(device_index: int) -> dict[str, Any]:
    """Collect comprehensive environment metadata.

    Returns:
        Dict with GPU, CUDA, driver, CPU, container info.
    """
    import pynvml
    import platform
    import subprocess

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    # GPU metadata
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)

    # CUDA/driver versions
    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
    cuda_major = cuda_version // 1000
    cuda_minor = (cuda_version % 1000) // 10
    driver_version = pynvml.nvmlSystemGetDriverVersion()

    # Thermal/power state
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000  # W

    # CPU governor (Linux)
    try:
        governor = subprocess.check_output(
            ["cat", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"],
            text=True
        ).strip()
    except:
        governor = "unknown"

    # Container detection
    in_container = os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv")

    return {
        "gpu": {
            "name": gpu_name,
            "vram_total_mb": mem_info.total / (1024**2),
            "cuda_compute_capability": None,  # TODO: parse from gpu_name
        },
        "cuda": {
            "version": f"{cuda_major}.{cuda_minor}",
            "driver_version": driver_version,
        },
        "thermal": {
            "temperature_c": temp,
            "power_limit_w": power_limit,
        },
        "cpu": {
            "governor": governor,
            "platform": platform.system(),
        },
        "container": {
            "detected": in_container,
        }
    }
```

### Example 3: CSV Export with Grouped Prefixes
```python
# Source: results/exporters.py (extend _aggregated_to_row)
def _aggregated_to_row_v3(result: AggregatedResult) -> dict[str, Any]:
    """Convert schema v3 result to CSV row with grouped prefixes."""
    row = {
        "experiment_id": result.experiment_id,

        # Energy with grouped prefixes
        "energy_raw_j": result.total_energy_j,
        "energy_adjusted_j": (
            result.energy_breakdown.adjusted_j
            if result.energy_breakdown else None
        ),
        "energy_baseline_w": (
            result.energy_breakdown.baseline_power_w
            if result.energy_breakdown else None
        ),

        # Thermal flags
        "thermal_throttle_detected": getattr(result, "thermal_throttle_detected", False),
        "thermal_throttle_duration_sec": getattr(result, "thermal_throttle_duration_sec", 0.0),

        # Environment (one-line summary)
        "gpu_name": result.environment.gpu.name if hasattr(result, "environment") else None,
        "cuda_version": result.environment.cuda.version if hasattr(result, "environment") else None,

        # Warmup
        "warmup_converged": getattr(result, "warmup_converged", None),
        "warmup_final_cv": getattr(result, "warmup_final_cv", None),

        # ... existing fields ...
    }
    return row
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Fixed warmup iterations | Convergence detection (CV < 5%) | 2025-2026 research | Adapts to hardware variance, avoids under/over-warming |
| Raw energy reporting | Baseline-adjusted energy | ML.ENERGY 2025 | Corrects 15-30% systematic overestimation |
| Single-file results | Separate time-series files | TokenPowerBench 2025 | Keeps results lightweight, enables phase analysis |
| Manual throttle detection | NVML throttle reasons API | NVML vR590 (Jan 2026) | Distinguishes thermal vs power vs other causes |
| pynvml (deprecated) | nvidia-ml-py | 2024 | Official NVIDIA bindings, actively maintained |

**Deprecated/outdated:**
- **pynvml PyPI package**: Use nvidia-ml-py instead (official NVIDIA bindings)
- **Fixed warmup iterations**: Use CV-based convergence for variable hardware
- **Embedded time-series in JSON**: Use separate files for >1000 samples
- **Environment metadata in logs only**: Must be in results JSON for reproducibility

## Open Questions

### 1. Baseline Power Measurement Duration
- **What we know:** 30s is common in literature, but shorter experiments may need shorter baseline
- **What's unclear:** Optimal balance between measurement accuracy and overhead
- **Recommendation:** Default 30s, make configurable (`baseline.duration_sec`), document tradeoff

### 2. MIG Instance Energy Isolation
- **What we know:** NVML reports parent GPU power, cannot isolate per-MIG-instance
- **What's unclear:** Whether to attempt estimation (divide by instance count) or report limitation
- **Recommendation:** Report parent GPU power with clear flag `gpu_is_mig: true`, document limitation, accept as best-available measurement

### 3. Warmup Convergence on Noisy Hardware
- **What we know:** 5% CV threshold may be unrealistic on shared GPUs
- **What's unclear:** Should we auto-relax threshold based on detected variance?
- **Recommendation:** Fixed threshold (configurable), log non-convergence as warning not error, allow fallback to fixed iterations

### 4. Time-Series Database for Power Data
- **What we know:** InfluxDB/TimescaleDB appropriate for time-series, but adds dependency
- **What's unclear:** Whether file-based storage sufficient for Phase 1 scope
- **Recommendation:** Start with JSON files (Phase 1), defer time-series DB to post-v2.0 if users request, validate storage scaling with 10Hz 5min experiments (3000 samples)

### 5. Thermal Throttling Response Strategy
- **What we know:** Phase 1 detects and flags, Phase 2 handles campaign-level response (retry/skip)
- **What's unclear:** Should Phase 1 pause inference when throttle detected?
- **Recommendation:** Phase 1 logs only (non-invasive), Phase 2 adds configurable response (`on_throttle: continue|pause|abort`)

## Sources

### Primary (HIGH confidence)

**Stack:**
- [NVML API Reference Guide (vR590, January 2026)](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html) — Official NVML documentation
- [nvidia-ml-py PyPI](https://pypi.org/project/nvidia-ml-py/) — Official NVIDIA Python bindings v13.590.48
- [Pydantic Documentation](https://docs.pydantic.dev/latest/) — Schema validation and migration patterns

**Research:**
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/) — Baseline power methodology
- [Part-time Power Measurements arXiv](https://arxiv.org/html/2312.02741v2) — NVML sampling issues
- [TokenPowerBench (Dec 2025) arXiv](https://arxiv.org/html/2512.03024v1) — Time-series power analysis state-of-art

**Codebase:**
- `src/llenergymeasure/core/gpu_utilisation.py` — Reference NVML sampling pattern
- `src/llenergymeasure/domain/metrics.py` — Current metrics schema
- `src/llenergymeasure/domain/experiment.py` — Results models (schema v2)
- `src/llenergymeasure/results/exporters.py` — CSV export logic
- `.planning/research/SUMMARY.md` — Prior domain research

### Secondary (MEDIUM confidence)

- [GPU Idle Power Benchmark Guide](https://www.ywian.com/blog/gpu-idle-power-benchmark-fix-it-guide) — Idle power ranges (30-50W)
- [Warmup convergence detection research](https://www.emergentmind.com/topics/model-warmup-techniques) — CV thresholds
- [NVML ThrottleReasons Rust docs](https://docs.rs/nvml-wrapper/latest/nvml_wrapper/bitmasks/device/struct.ThrottleReasons.html) — Throttle reason flags

### Tertiary (LOW confidence, needs validation)

- Warmup convergence CV threshold <5% — Domain inference, not research-validated
- 30s baseline measurement duration — Common practice, not formally specified
- Time-series JSON vs database — Implementation choice, depends on scale

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — Official NVIDIA bindings verified, existing codebase uses Pydantic/numpy
- Architecture: HIGH — Extends existing patterns (GPUUtilisationSampler, Pydantic models, CSV export), brownfield not greenfield
- Pitfalls: HIGH — Energy measurement issues from research papers, NVML edge cases from official docs, CSV export from codebase experience

**Research date:** 2026-01-29
**Valid until:** 2026-04-29 (90 days — stable domain, NVML API mature)

**Key constraints from CONTEXT.md:**
- Baseline power: Auto + cached, both raw and adjusted fields
- Thermal throttling: Flag + continue (experiment completes)
- Warmup: CV-based with configurable max cap and threshold
- Results schema: Nested structure acceptable (breaking change for v2.0.0)
- Time-series: Separate file, optional export
- CSV: Grouped prefixes for readability

**Brownfield context:**
- Existing `GPUUtilisationSampler` provides NVML sampling reference
- Existing Pydantic models in `domain/` need extending, not replacing
- Existing CSV export in `results/exporters.py` needs new columns
- Existing config system in `config/models.py` needs new fields
- Schema v3 is additive change, v2 results still readable
