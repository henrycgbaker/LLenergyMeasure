# Phase 5: Energy Measurement - Research

**Researched:** 2026-02-26
**Domain:** GPU energy measurement, warmup, FLOPs estimation, Parquet timeseries
**Confidence:** HIGH (verified against codebase + official docs)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Energy backend wiring:**
- Default mode: `auto` — selects best available backend (Zeus > NVML > CodeCarbon)
- Explicit override supported: `energy.backend: auto | nvml | zeus | codecarbon | null` in ExperimentConfig
- `null` is a valid explicit choice — disables energy measurement intentionally (no warnings, energy fields are null)
- When user explicitly requests an unavailable backend: hard `ConfigError` with install guidance (e.g., `pip install llenergymeasure[zeus]`)
- When no energy backend available at all (CPU-only machine): run inference, produce latency/throughput results, but energy fields are null; surface clearly in results
- Backend conflicts/mutual exclusion enforced at selection time (only one backend owns the NVML session)

**Measurement quality signals:**
- Warnings are purely informational — never block the experiment
- Warning flags stored in `result.measurement_warnings` (list of strings) and displayed inline in CLI output
- Four warning flags: `short_measurement_duration` (<10s), `gpu_persistence_mode_off`, `thermal_drift_detected` (>10°C change), `nvml_low_sample_count` (<10 samples)
- No `--strict` mode — warnings annotate, researchers filter results themselves
- Thermal drift threshold: **configurable** with 10°C default (confidence LOW — engineering judgement)
- Warning display format: inline with results, includes actionable remediation advice

**Warmup-to-measurement handoff:**
- Thermal floor: pure time-based wait (sleep for `thermal_floor_seconds` after warmup) — no temperature probing
- Simple, predictable, matches MLPerf Power. Default 60s, minimum 30s (already in WarmupConfig)
- Docker mode: thermal floor still applies intra-experiment; inter-experiment `gap_seconds` auto-skipped
- WarmupResult captures: warmup run count, per-run latencies, total warmup duration, AND thermal floor wait duration
- CV convergence (opt-in) is additive: run AT LEAST `n_warmup` runs, then continue until CV < target OR `max_warmup_prompts` hit
- CLI output: plain-text status lines as each phase completes

**Energy result fields:**
- Always include all three: `baseline_power_w`, `energy_total_j`, `energy_adjusted_j`
- Never omit any — if baseline unavailable: `baseline_power_w` is null, `energy_adjusted_j` is null, but `energy_total_j` is still populated

**Timeseries format:**
- Full telemetry columns: `timestamp_s`, `gpu_index`, `power_w`, `temperature_c`, `memory_used_mb`, `memory_total_mb`, `sm_utilisation_pct`, `throttle_reasons`
- Measurement window only — no warmup or thermal floor samples
- Long format for multi-GPU: one row per GPU per second, `gpu_index` column
- File location: `{name}_{timestamp}/timeseries.parquet` (same directory as result JSON)
- Result JSON references timeseries via relative path

**Default confidence levels:**
- `n_warmup: 5` — HIGH
- `thermal_floor_seconds: 60` — HIGH (MLPerf Power)
- `thermal_drift_threshold: 10°C` — LOW (engineering judgement)
- `baseline_duration: 30s` — MEDIUM
- `sample_interval: 100ms` — HIGH (NVML hardware update period)

### Claude's Discretion

- Internal module structure (where energy backend registry lives, how auto-selection is wired)
- NVML poller implementation details (thread vs async, error recovery)
- How baseline cache invalidation interacts with energy backend switching
- FLOPs implementation: carry forward existing `flops.py` or rewrite — as long as it implements the PaLM formula per the locked decision
- Exact warning message wording (as long as it includes remediation advice)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| CM-11 | NVML poller (`nvml.py`) always available — base install | `nvidia-ml-py>=12.0` already in base deps; `power_thermal.py` has working pynvml pattern; need new `NVMLBackend` that wraps poll-then-integrate into the EnergyBackend protocol |
| CM-12 | Zeus backend optional (`llenergymeasure[zeus]`) | Zeus 0.15.0 on PyPI (package name `zeus`, NOT `zeus-ml`); ZeusMonitor API: `from zeus.monitor import ZeusMonitor`; `begin_window` / `end_window("name")` → Measurement(time, energy={idx: joules}); pyproject.toml must change `zeus-ml` → `zeus>=0.13.1` |
| CM-13 | CodeCarbon backend optional (`llenergymeasure[codecarbon]`) | `CodeCarbonBackend` already exists in `core/energy_backends/codecarbon.py`; needs integration into auto-selection; existing test coverage in `test_core_energy_backends.py` |
| CM-14 | Energy backend priority: Zeus → NVML → CodeCarbon; mutual exclusion enforced | Auto-selection function needed: `select_energy_backend(explicit: str\|None) -> EnergyBackend\|None`; needs import-guarded probes (`importlib.util.find_spec`) for Zeus/CodeCarbon; NVML always available (base dep) |
| CM-15 | `torch.cuda.synchronize()` before every measurement stop | Must wrap stop call in the measurement orchestration: call synchronize BEFORE `stop_tracking()`; deferred import pattern (find_spec check before torch.cuda) to avoid module-level torch import |
| CM-16 | Timeseries: 1 Hz sampling, sidecar `timeseries.parquet` | `PowerThermalSampler` already collects 100ms samples; need downsampling to 1 Hz + Parquet writer; `pyarrow>=14.0` already in base deps; column schema locked; long format for multi-GPU |
| CM-17 | Idle GPU baseline before warmup (`BaselineConfig.enabled: bool = True`) | `baseline.py` already has `measure_baseline_power()` with TTL cache; wiring into `PyTorchBackend.run()` before warmup phase; config field `BaselineConfig.enabled` already exists |
| CM-18 | `baseline_power_w` stored in ExperimentResult | `ExperimentResult` already has `energy_breakdown: EnergyBreakdown\|None` with `baseline_power_w`; Phase 5 populates it (currently 0.0 placeholders in `_build_result`) |
| CM-19 | `energy_adjusted_j = energy_total_j - (baseline_power_w × duration_sec)` | `adjust_energy_for_baseline()` already in `baseline.py`; `create_energy_breakdown()` already assembles the `EnergyBreakdown` model; just needs wiring |
| CM-20 | Baseline cache with session-level TTL | `_baseline_cache: dict[int, BaselineCache]` already in `baseline.py`; module-level dict persists across calls; TTL default 3600s; no change needed |
| CM-21 | Fixed-count default: `n_warmup: 5`, full-length prompts | `WarmupConfig.n_warmup` default is currently `3` — must be changed to `5`; `PyTorchBackend._run_warmup` already runs full-length via `config.max_output_tokens`; need to add warmup duration tracking and return `WarmupResult` |
| CM-22 | Thermal floor: 60s wait after warmup (configurable down to 30s) | `WarmupConfig.thermal_floor_seconds` already exists with `ge=0.0` — needs `ge=30.0` minimum enforcement (or warning); `thermal_floor_seconds=60.0` default already set; Phase 5 adds the actual sleep + CLI status line |
| CM-23 | CV-based convergence as opt-in: `convergence_detection: true` | `warmup.py` has `warmup_until_converged()` with fixed vs CV mode; `WarmupConfig` lacks `convergence_detection` field — must add; CV detection logic already exists in `warmup_until_converged()` |
| CM-24 | `WarmupResult` with 6 fields: `converged`, `final_cv`, `iterations_completed`, `target_cv`, `max_prompts`, `latencies_ms` | `WarmupResult` Pydantic model already has all 6 fields; `warmup_until_converged()` already returns it; just needs proper wiring into `PyTorchBackend._run_warmup()` to return it |
| CM-25 | Primary metrics: `energy_per_output_token` (J/token) and `tokens_per_second` | `ExperimentResult.avg_tokens_per_second` already computed; `avg_energy_per_token_j` is a 0.0 placeholder — Phase 5 populates from real energy measurement |
| CM-26 | FLOPs = reference metadata, PaLM formula (2 × N_params × tokens) | Existing `flops.py` uses calflops→architecture→parameter_estimate fallback chain — does NOT implement PaLM formula as primary; Phase 5 must rewrite/replace with pure PaLM formula (`2 × non_embedding_params × tokens`) as primary, keeping existing test infrastructure |
| CM-27 | `FlopsResult` with `method: str` and `confidence: Literal["high", "medium", "low"]` | `FlopsResult` Pydantic model already exists with both fields; `method` Literal type must add `"palm_formula"` value; existing tests test old fallback methods and will need updating |
| CM-28 | Warmup tokens excluded from FLOPs calculation | New FLOPs implementation takes `(input_tokens, output_tokens)` from measurement phase only; warmup token counts must be tracked and kept separate |
</phase_requirements>

---

## Summary

Phase 5 wires together existing infrastructure into a complete, scientifically credible energy measurement pipeline. The heavy machinery is already built: `NVMLPoller` (via `power_thermal.py`), `CodeCarbonBackend`, `baseline.py`, `warmup.py`, and `FlopsResult`. What Phase 5 adds is the **integration layer** that makes these components work together correctly.

The three highest-risk items are: (1) the FLOPs rewrite — the existing `flops.py` uses a three-strategy fallback chain (calflops/architecture/parameter_estimate) that conflicts with the locked decision to use the PaLM formula as primary; (2) the `NVMLBackend` energy integration — this is currently only a sampler for thermal/power tracking, not an EnergyBackend that integrates a joule total; and (3) the Parquet timeseries sidecar — the existing `timeseries.py` writes JSON (wrong format) and the Parquet writer needs to be added.

The Zeus package name is wrong in `pyproject.toml` — it references `zeus-ml>=0.10` (abandoned at v0.11.0) instead of `zeus>=0.13.1`. This must be fixed as part of CM-12.

**Primary recommendation:** Build Phase 5 as three focused sub-units: (A) energy backend auto-selection + NVMLBackend + Zeus wiring; (B) warmup finalisation (n_warmup=5 default, WarmupResult return, thermal floor sleep, CV opt-in field); (C) FLOPs rewrite (PaLM formula, non-embedding param count) + Parquet timeseries. Wire all three into `PyTorchBackend._build_result()` to replace the 0.0 placeholders.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `nvidia-ml-py` (pynvml) | >=12.0 | NVML GPU power polling | Already in base deps; used in `power_thermal.py` and `baseline.py` |
| `zeus` | >=0.13.1 | Hardware energy counter (accurate) | ZeusMonitor reads NVML total-energy register (not polling); Zeus 0.15.0 released 2026-02-24 |
| `codecarbon` | >=2.8 | Fallback energy tracking | Already implemented; `CodeCarbonBackend` exists |
| `pyarrow` | >=14.0 | Parquet timeseries write | Already in base deps; `pa.Table.from_pydict()` + `pq.write_table()` |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pandas` | - | Optional Parquet intermediary | NOT needed — use `pyarrow` direct API; pandas not in deps |
| `torch.cuda` | (with pytorch extra) | GPU sync before measurement stop | Must be deferred import; only when torch is available |
| `numpy` | - | CV calculation in warmup | Already used in `warmup.py` via `np.std()`/`np.mean()` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `pyarrow` direct API | `pandas.to_parquet()` | pandas not in base deps; pyarrow is simpler for this schema |
| ZeusMonitor hardware counter | NVML polling integration | Zeus is more accurate but optional; NVML poll is always available |
| PaLM formula | calflops runtime profiler | calflops can't model autoregressive KV cache decode loop (locked decision) |

**Installation (fix required):**
```bash
# Fix pyproject.toml: zeus-ml is abandoned, use zeus
# Change: zeus = ["zeus-ml>=0.10"]
# To:     zeus = ["zeus>=0.13.1"]
```

---

## Architecture Patterns

### Recommended Module Structure

The existing `core/energy_backends/` directory gains two new files:

```
src/llenergymeasure/core/energy_backends/
├── __init__.py          # Registry + auto_select_backend() ← extend existing
├── base.py              # EnergyBackend Protocol re-export ← unchanged
├── codecarbon.py        # CodeCarbonBackend ← unchanged
├── nvml.py              # NVMLBackend (NEW) — poll + integrate joules
└── zeus.py              # ZeusBackend (NEW) — ZeusMonitor wrapper
```

The `core/metrics.py` module (new) houses the PaLM FLOPs formula, replacing the old `flops.py` fallback chain. The old `flops.py` either gets a clean rewrite in-place or is replaced. Given the existing tests reference `estimate_flops` from `core/flops.py`, the cleanest approach is to rewrite `core/flops.py` in-place with the PaLM formula as the primary method, preserving the module path and public API.

### Pattern 1: Energy Backend Auto-Selection

**What:** Single function `select_energy_backend(explicit: str | None, config: ExperimentConfig) -> EnergyBackend | None`

**When to use:** Called at the start of every experiment, before baseline measurement

**Example (from locked design decision and Zeus docs):**
```python
# Source: .product/designs/energy-backends.md + zeus.monitor API
import importlib.util

def select_energy_backend(
    explicit: str | None,
) -> EnergyBackend | None:
    """Select energy backend: explicit override or auto (Zeus > NVML > CodeCarbon).

    Returns None when explicit=null or no backend is available.
    Raises ConfigError when explicit backend is unavailable.
    """
    if explicit == "null":
        return None  # intentional disable — no warnings

    if explicit is not None:
        # Hard error if explicitly requested backend is unavailable
        backend = _make_backend(explicit)
        if not backend.is_available():
            raise ConfigError(
                f"Energy backend '{explicit}' is not available. "
                f"Install with: pip install llenergymeasure[{explicit}]"
            )
        return backend

    # Auto-selection: Zeus > NVML > CodeCarbon
    if importlib.util.find_spec("zeus") is not None:
        backend = ZeusBackend()
        if backend.is_available():
            return backend

    nvml_backend = NVMLBackend()
    if nvml_backend.is_available():
        return nvml_backend

    if importlib.util.find_spec("codecarbon") is not None:
        backend = CodeCarbonBackend()
        if backend.is_available():
            return backend

    return None  # CPU-only machine, energy fields will be null
```

### Pattern 2: NVMLBackend (poll + trapezoidal integrate)

**What:** Wraps `PowerThermalSampler` into the `EnergyBackend` protocol — samples power at 100ms, integrates via trapezoidal rule.

**When to use:** Always available (base dep); used when Zeus not installed.

```python
# Source: .product/designs/energy-backends.md + existing power_thermal.py
class NVMLBackend:
    """NVML power polling backend. Always available (base install).

    Polls GPU power at 100ms intervals during measurement window.
    Integrates power samples via trapezoidal rule to get joules.
    Accuracy: ±5W + sampling error (~5-15% depending on workload duration).
    """
    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index

    @property
    def name(self) -> str:
        return "nvml"

    def is_available(self) -> bool:
        try:
            import pynvml  # noqa: PLC0415
            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:
            return False

    def start_tracking(self) -> PowerThermalSampler:
        """Start background NVML polling. Returns sampler as tracker handle."""
        from llenergymeasure.core.power_thermal import PowerThermalSampler  # noqa: PLC0415
        sampler = PowerThermalSampler(device_index=self._device_index, sample_interval_ms=100)
        sampler.start()
        return sampler

    def stop_tracking(self, tracker: PowerThermalSampler) -> EnergyMeasurement:
        """Stop polling and integrate power samples to joules."""
        tracker.stop()
        samples = tracker.get_samples()
        if not samples:
            return EnergyMeasurement(total_j=0.0, duration_sec=0.0, samples=samples)
        # Trapezoidal integration: sum(power_w * dt_s) for each interval
        powers = [s.power_w for s in samples if s.power_w is not None]
        dt = 0.1  # 100ms sample interval
        total_j = sum(p * dt for p in powers)  # simple Euler; trapezoidal uses pairs
        duration = samples[-1].timestamp - samples[0].timestamp
        return EnergyMeasurement(total_j=total_j, duration_sec=duration, samples=samples)
```

### Pattern 3: ZeusBackend (hardware counter)

**What:** Wraps `ZeusMonitor` from `zeus.monitor`. Uses hardware energy register — most accurate.

**When to use:** When `zeus` package is installed (optional extra).

```python
# Source: zeus.monitor API (verified via docs, Zeus 0.15.0)
# Import: from zeus.monitor import ZeusMonitor
# Measurement: monitor.end_window(name) → Measurement(time, energy={gpu_idx: joules})
class ZeusBackend:
    """Zeus hardware energy counter backend (optional, llenergymeasure[zeus]).

    Uses ZeusMonitor which reads NVML total-energy hardware register.
    Most accurate method: ±5W under sustained load on Volta+ GPUs.
    """
    WINDOW_NAME = "llem_measurement"

    def __init__(self, gpu_indices: list[int] | None = None) -> None:
        self._gpu_indices = gpu_indices  # None = all GPUs

    @property
    def name(self) -> str:
        return "zeus"

    def is_available(self) -> bool:
        try:
            from zeus.monitor import ZeusMonitor  # noqa: PLC0415
            return True
        except ImportError:
            return False

    def start_tracking(self) -> Any:
        from zeus.monitor import ZeusMonitor  # noqa: PLC0415
        monitor = ZeusMonitor(gpu_indices=self._gpu_indices)
        monitor.begin_window(self.WINDOW_NAME)
        return monitor

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        measurement = tracker.end_window(self.WINDOW_NAME)
        # measurement.energy = {gpu_idx: joules_float}
        # measurement.time = elapsed seconds
        total_j = sum(measurement.energy.values())
        return EnergyMeasurement(total_j=total_j, duration_sec=measurement.time, per_gpu_j=measurement.energy)
```

### Pattern 4: torch.cuda.synchronize() before stop

**What:** GPU sync call BEFORE stopping energy measurement. Required for correctness (CM-15).

**When to use:** Always, when torch is available and CUDA is active.

```python
# Source: .product/designs/energy-backends.md + Zeus ML blog best practices
def _stop_energy_measurement(backend: EnergyBackend, tracker: Any) -> EnergyMeasurement:
    """Stop measurement with mandatory GPU sync for correctness."""
    import importlib.util  # noqa: PLC0415
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all GPU kernels to finish
        except Exception:
            pass  # Non-fatal — best effort sync
    return backend.stop_tracking(tracker)
```

### Pattern 5: PaLM FLOPs formula

**What:** Replace existing `flops.py` fallback chain with PaLM/Chinchilla formula as primary.

**When to use:** Always for v2.0. No runtime model dependency — pure arithmetic.

```python
# Source: .product/decisions/flops-estimation.md
# Formula: FLOPs = 2 × N_non_embedding_params × total_tokens
# Phase split: prefill (input tokens) + decode (output tokens)

def estimate_flops_palm(
    model: Any,
    n_input_tokens: int,  # measurement window only, warmup excluded
    n_output_tokens: int,  # measurement window only, warmup excluded
    batch_size: int = 1,
) -> FlopsResult:
    """PaLM/Chinchilla inference FLOPs estimate.

    Formula (v2.0, attention FLOPs omitted):
        FLOPs_prefill = 2 × N_params × B × S_input
        FLOPs_decode  = 2 × N_params × B × S_output
        FLOPs_total   = FLOPs_prefill + FLOPs_decode

    N_params = non-embedding parameters only (embeddings are memory-bound lookups).
    Warmup tokens excluded from n_input_tokens and n_output_tokens by caller.
    """
    n_params = _count_non_embedding_params(model)

    flops_prefill = 2 * n_params * batch_size * n_input_tokens
    flops_decode  = 2 * n_params * batch_size * n_output_tokens
    flops_total   = flops_prefill + flops_decode

    return FlopsResult(
        value=float(flops_total),
        method="palm_formula",     # new literal value needed in FlopsResult
        confidence="high",          # formula is deterministic given N_params
        precision="n/a",            # precision doesn't affect FLOPs (forward pass)
        notes=(
            f"PaLM formula: 2×{n_params:,}×({n_input_tokens}+{n_output_tokens}) tokens. "
            f"Attention FLOPs omitted (v2.0 limitation, significant only for seq_len≥2048)."
        ),
    )

def _count_non_embedding_params(model: Any) -> int:
    """Count non-embedding parameters (embeddings excluded per PaLM methodology)."""
    total = 0
    for name, param in model.named_parameters():
        # Skip embedding layers — they are memory-bound lookups, not MAC ops
        if "embed" not in name.lower():
            total += param.numel()
    return total
```

### Pattern 6: Parquet timeseries write

**What:** Replace existing JSON timeseries with Parquet sidecar file. 1 Hz sampling = downsample from 100ms raw samples.

```python
# Source: pyarrow docs + locked column schema from CONTEXT.md
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def write_timeseries_parquet(
    samples: list[PowerThermalSample],
    output_path: Path,
    gpu_index: int = 0,
) -> Path:
    """Write 1 Hz timeseries to Parquet sidecar.

    Downsamples 100ms NVML samples to 1 Hz by grouping into 1-second buckets
    and taking the mean power/temperature/etc within each bucket.

    Columns (locked): timestamp_s, gpu_index, power_w, temperature_c,
                      memory_used_mb, memory_total_mb, sm_utilisation_pct,
                      throttle_reasons
    Format: long (one row per GPU per second).
    """
    if not samples:
        # Write empty Parquet with correct schema
        schema = _timeseries_schema()
        table = pa.table({col: [] for col in schema.names}, schema=schema)
        pq.write_table(table, output_path)
        return output_path

    # Group samples into 1-second buckets (downsample 100ms → 1 Hz)
    base_ts = samples[0].timestamp
    buckets: dict[int, list[PowerThermalSample]] = {}
    for s in samples:
        bucket = int(s.timestamp - base_ts)
        buckets.setdefault(bucket, []).append(s)

    rows = []
    for second, bucket_samples in sorted(buckets.items()):
        powers = [s.power_w for s in bucket_samples if s.power_w is not None]
        temps  = [s.temperature_c for s in bucket_samples if s.temperature_c is not None]
        mems_u = [s.memory_used_mb for s in bucket_samples if s.memory_used_mb is not None]
        mems_t = [s.memory_total_mb for s in bucket_samples if s.memory_total_mb is not None]
        utils  = [s.sm_utilisation for s in bucket_samples if s.sm_utilisation is not None]
        # throttle_reasons: OR of all reason bitmasks in the bucket
        reasons = 0
        for s in bucket_samples:
            reasons |= s.throttle_reasons

        rows.append({
            "timestamp_s": float(second),
            "gpu_index": gpu_index,
            "power_w": sum(powers) / len(powers) if powers else None,
            "temperature_c": sum(temps) / len(temps) if temps else None,
            "memory_used_mb": sum(mems_u) / len(mems_u) if mems_u else None,
            "memory_total_mb": sum(mems_t) / len(mems_t) if mems_t else None,
            "sm_utilisation_pct": sum(utils) / len(utils) if utils else None,
            "throttle_reasons": reasons,
        })

    # Build pyarrow table directly (no pandas)
    schema = _timeseries_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    return output_path

def _timeseries_schema() -> pa.Schema:
    return pa.schema([
        ("timestamp_s", pa.float64()),
        ("gpu_index", pa.int32()),
        ("power_w", pa.float32()),
        ("temperature_c", pa.float32()),
        ("memory_used_mb", pa.float32()),
        ("memory_total_mb", pa.float32()),
        ("sm_utilisation_pct", pa.float32()),
        ("throttle_reasons", pa.int64()),
    ])
```

### Pattern 7: Measurement warnings assembly

**What:** Collect warning flags at the end of measurement and store in `ExperimentResult.measurement_warnings`.

```python
# Source: CONTEXT.md + .product/designs/energy-backends.md
def _collect_measurement_warnings(
    duration_sec: float,
    gpu_persistence_mode: bool,
    temp_start_c: float | None,
    temp_end_c: float | None,
    nvml_sample_count: int,
    thermal_drift_threshold_c: float = 10.0,
) -> list[str]:
    warnings = []
    if duration_sec < 10.0:
        warnings.append(
            "short_measurement_duration: measurement < 10s; "
            "energy values may be unreliable. Use more prompts or longer sequences."
        )
    if not gpu_persistence_mode:
        warnings.append(
            "gpu_persistence_mode_off: power state variation may inflate measurements. "
            "Run `nvidia-smi -pm 1` to enable persistence mode."
        )
    if temp_start_c is not None and temp_end_c is not None:
        drift = abs(temp_end_c - temp_start_c)
        if drift > thermal_drift_threshold_c:
            warnings.append(
                f"thermal_drift_detected: {drift:.1f}°C change during measurement "
                f"(threshold {thermal_drift_threshold_c}°C). "
                "Increase thermal_floor_seconds or check cooling."
            )
    if nvml_sample_count < 10:
        warnings.append(
            "nvml_low_sample_count: fewer than 10 NVML power samples collected; "
            "energy integration may be inaccurate."
        )
    return warnings
```

### Anti-Patterns to Avoid

- **Calling torch.cuda.synchronize() at module level:** Must be deferred import — `flops.py` currently has `import torch` at module level. Use `importlib.util.find_spec("torch")` check before importing.
- **Using `zeus-ml` package:** The old `zeus-ml` package stopped at v0.11.0. Install `zeus>=0.13.1`.
- **Integrating energy during warmup:** The energy measurement window must wrap only the measurement loop, not warmup. Starting `tracker = backend.start_tracking()` AFTER warmup+thermal_floor sleep.
- **Wide-format timeseries for multi-GPU:** Use long format (one row per GPU per second) per locked decision — not wide format.
- **Blocking on thermal floor:** The 60s thermal floor sleep should happen AFTER warmup completes, BEFORE `backend.start_tracking()`. This ensures warmup GPU heat dissipates before the measurement window opens.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Parquet I/O | Custom binary writer | `pyarrow.parquet` | Already in base deps; handles schema, compression |
| NVML GPU energy hardware counter | Direct NVML C API calls | `zeus.ZeusMonitor` | Zeus reads NVML total energy register correctly; handles multi-GPU, sync |
| FLOPs profiling | Runtime torch hooks | PaLM formula (`2×N×T`) | Runtime profilers don't model KV-cache decode loop correctly (locked decision) |
| Background power polling | Custom threading + pynvml | `PowerThermalSampler` (existing) | Already built, tested, handles graceful degradation |
| Baseline caching | Custom TTL dict | `baseline.py` `_baseline_cache` (existing) | Already built with session-level TTL |

**Key insight:** Most of the heavy machinery exists. Phase 5 is primarily wiring + fixing two gaps: NVMLBackend energy integration (existing sampler doesn't sum joules) and PaLM FLOPs formula (existing flops.py uses wrong primary strategy).

---

## Common Pitfalls

### Pitfall 1: Zeus Package Name

**What goes wrong:** `zeus-ml` in pyproject.toml resolves to the abandoned package (last release v0.11.0, 2023). `ZeusMonitor` API changed significantly in v0.13+.

**Why it happens:** The package was renamed from `zeus-ml` to `zeus` at some point.

**How to avoid:** Update `pyproject.toml`: `zeus = ["zeus>=0.13.1"]`. The verified current version is 0.15.0 (released 2026-02-24).

**Warning signs:** ImportError for `from zeus.monitor import ZeusMonitor` when `zeus-ml` is installed instead.

### Pitfall 2: Energy window must wrap measurement only

**What goes wrong:** Starting energy tracking before warmup includes warmup GPU heat in the measurement.

**Why it happens:** Warmup runs the same inference as measurement — large energy consumption that should be excluded.

**How to avoid:** Call `backend.start_tracking()` AFTER the warmup phase AND after the thermal_floor sleep completes.

**Correct sequence:**
```
1. measure_baseline_power()       ← before model load or before warmup
2. load_model()
3. _run_warmup()                  ← warmup runs (energy NOT tracked)
4. time.sleep(thermal_floor_sec)  ← thermal stabilisation (energy NOT tracked)
5. tracker = backend.start_tracking()  ← measurement window BEGINS
6. _run_measurement()
7. torch.cuda.synchronize()
8. measurement = backend.stop_tracking(tracker)  ← measurement window ENDS
```

### Pitfall 3: FlopsResult.method Literal mismatch

**What goes wrong:** Adding `"palm_formula"` as a new method string fails Pydantic validation because `FlopsResult.method` is `Literal["calflops", "architecture", "parameter_estimate"]`.

**Why it happens:** The existing tests reference all three old literal values — adding a new value requires updating both the model and the tests.

**How to avoid:** Update `FlopsResult.method` to include `"palm_formula"`. Update existing test assertions in `test_flops_estimator.py` that test old methods.

### Pitfall 4: WarmupConfig missing fields for Phase 5

**What goes wrong:** `WarmupConfig` currently has only `n_warmup` (default 3, must be 5) and `thermal_floor_seconds`. CM-23 requires `convergence_detection: bool` field.

**Why it happens:** The context doc says "CV convergence detection is Phase 5 measurement concern, not a config concern" — but CM-23 explicitly requires an opt-in field in WarmupConfig.

**How to avoid:** Add `convergence_detection: bool = False` and related CV fields to `WarmupConfig`. The existing `warmup_until_converged()` in `warmup.py` already has the logic — it just needs to be connected.

**Fields to add to WarmupConfig:**
```python
convergence_detection: bool = Field(default=False, description="Enable CV-based convergence detection")
cv_threshold: float = Field(default=0.05, ge=0.01, le=0.5, description="CV target for convergence")
max_warmup_prompts: int = Field(default=20, ge=5, description="Max warmup prompts when CV mode is on")
window_size: int = Field(default=5, ge=3, description="Window size for CV calculation")
min_prompts: int = Field(default=5, ge=1, description="Minimum prompts before checking convergence")
```

### Pitfall 5: n_warmup default mismatch between config and CONTEXT.md

**What goes wrong:** `WarmupConfig.n_warmup` defaults to 3 in the current codebase. CONTEXT.md says default should be 5 (HIGH confidence, per DeepSpeed/Zeus peer review). CM-21 explicitly requires `n_warmup: 5`.

**How to avoid:** Change `WarmupConfig.n_warmup` default from `3` to `5`. Update any tests that rely on the default.

### Pitfall 6: ExperimentConfig missing `energy` section

**What goes wrong:** The locked decision requires `energy.backend: auto | nvml | zeus | codecarbon | null` in `ExperimentConfig`, but no `energy:` section exists in `config/models.py`.

**How to avoid:** Add `EnergyConfig` sub-model to `config/models.py` (similar to `WarmupConfig`/`BaselineConfig`) and add `energy: EnergyConfig` field to `ExperimentConfig`.

```python
class EnergyConfig(BaseModel):
    model_config = {"extra": "forbid"}
    backend: Literal["auto", "nvml", "zeus", "codecarbon"] | None = Field(
        default="auto",
        description="Energy measurement backend. null disables energy measurement."
    )
```

Note: Pydantic represents `null` as Python `None`. The YAML `energy.backend: null` maps to `None` in Pydantic.

### Pitfall 7: measurement_warnings field missing from ExperimentResult

**What goes wrong:** `ExperimentResult` in `domain/experiment.py` does not have a `measurement_warnings: list[str]` field. This field is required by CM-10 (RES-10) and referenced extensively in CONTEXT.md.

**How to avoid:** Add `measurement_warnings: list[str] = Field(default_factory=list, ...)` to `ExperimentResult`. The field is already referenced in the designs but not yet in the model.

---

## Code Examples

Verified patterns from official sources and existing codebase:

### PyArrow Parquet write (no pandas)

```python
# Source: pyarrow docs + pyarrow>=14.0 in base deps
import pyarrow as pa
import pyarrow.parquet as pq

schema = pa.schema([
    ("timestamp_s", pa.float64()),
    ("gpu_index", pa.int32()),
    ("power_w", pa.float32()),
])
rows = [{"timestamp_s": 0.0, "gpu_index": 0, "power_w": 150.3}]
table = pa.Table.from_pylist(rows, schema=schema)
pq.write_table(table, "timeseries.parquet")
```

### Zeus ZeusMonitor API (verified, zeus 0.15.0)

```python
# Source: zeus.monitor official docs + PyPI zeus 0.15.0
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0])  # or None for all GPUs
monitor.begin_window("llem_measurement")
# ... run inference ...
torch.cuda.synchronize()
result = monitor.end_window("llem_measurement")
# result.time: float (seconds)
# result.energy: dict[int, float] (gpu_index -> joules)
total_j = sum(result.energy.values())
```

### NVML power poll to joules (trapezoidal)

```python
# Source: existing power_thermal.py PowerThermalSampler pattern
# Existing sampler already collects power_w at 100ms. Integration:
samples = sampler.get_samples()  # list[PowerThermalSample]
powers = [(s.timestamp, s.power_w) for s in samples if s.power_w is not None]
# Trapezoidal integration
total_j = 0.0
for i in range(1, len(powers)):
    dt = powers[i][0] - powers[i-1][0]
    avg_power = (powers[i][1] + powers[i-1][1]) / 2
    total_j += avg_power * dt
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `calflops` → architecture → parameter FLOPs fallback chain | PaLM formula (`2×N×T`) as primary | Phase 5 (this phase) | Removes runtime model dependency from FLOPs; makes FLOPs reproducible from config alone |
| JSON timeseries (`timeseries.json`) | Parquet sidecar (`timeseries.parquet`) | Phase 5 (this phase) | Parquet more efficient for columnar time data; standard for scientific data pipelines |
| No energy backend selection (CodeCarbon only) | Zeus > NVML > CodeCarbon priority chain | Phase 5 (this phase) | Zeus is most accurate; NVML always available; CodeCarbon fallback |
| Energy fields as 0.0 placeholders | Real energy measurement | Phase 5 (this phase) | Results become scientifically usable |
| `zeus-ml` PyPI package | `zeus` PyPI package (>=0.13.1) | zeus library rename | zeus-ml abandoned at v0.11.0; zeus 0.15.0 current |

**Deprecated/outdated:**
- `zeus-ml`: abandoned at v0.11.0; use `zeus>=0.13.1`
- `calflops` as FLOPs primary: correct approach is PaLM formula for autoregressive inference
- JSON timeseries: replace with Parquet per locked design

---

## Open Questions

1. **`ExperimentResult.measurement_warnings` field missing**
   - What we know: CONTEXT.md and designs reference this field extensively; it must exist in results
   - What's unclear: Whether to add it to `ExperimentResult` only, or also to `RawProcessResult`
   - Recommendation: Add to `ExperimentResult` in Phase 5 (since energy warnings are experiment-level); `RawProcessResult` is less urgent and may be handled in Phase 6 results cleanup

2. **n_warmup default change from 3 → 5: test impact**
   - What we know: Several existing tests may use `WarmupConfig()` default and assert `n_warmup=3`
   - What's unclear: How many tests are affected
   - Recommendation: Check `test_core_warmup.py` and `test_config_*.py` for `n_warmup=3` assertions; update in the same plan that changes the default

3. **`EnergyConfig` in `ExperimentConfig` — introspection.py update needed**
   - What we know: Adding a new sub-config requires updating `config/introspection.py` (the SSOT) per Phase 2 decisions
   - What's unclear: Whether introspection.py auto-discovers new sub-configs or needs explicit registration
   - Recommendation: Check `introspection.py` discovery pattern before implementing `EnergyConfig`; likely needs manual addition of the energy domain layer

4. **Baseline measurement timing: before or after model load?**
   - What we know: `baseline.py` comment says "before model load or before warmup"; CONTEXT.md says "Idle GPU baseline before warmup"
   - What's unclear: Whether model-in-memory affects "idle" GPU power baseline
   - Recommendation: Measure before model loading for cleanest idle baseline. `PyTorchBackend.run()` already calls `collect_environment_snapshot()` first — baseline should go there too.

5. **`FlopsResult.method` Literal vs `str`**
   - What we know: Current type is `Literal["calflops", "architecture", "parameter_estimate"]`; PaLM formula adds `"palm_formula"`
   - What's unclear: Whether to keep Literal (requires updating every time) or switch to `str`
   - Recommendation: Keep as `Literal` but add `"palm_formula"`. The validation is valuable. The old three methods can be kept for backward compatibility (the estimator can fall back to them if model not available for param counting).

---

## Validation Architecture

> `workflow.nyquist_validation` is not set in `.planning/config.json` — section skipped per researcher instructions.

---

## Sources

### Primary (HIGH confidence)

- Codebase audit: `src/llenergymeasure/core/energy_backends/`, `core/baseline.py`, `core/warmup.py`, `core/flops.py`, `core/power_thermal.py`, `domain/experiment.py`, `results/timeseries.py` — direct read
- `.product/decisions/warmup-strategy.md` — locked decisions, verified peer citations (MLPerf Power arXiv:2410.12032)
- `.product/decisions/flops-estimation.md` — PaLM formula decision, rejection of runtime profilers
- `.product/designs/energy-backends.md` — NVMLBackend accuracy, ZeusMonitor pattern, synchronization requirement
- `pyproject.toml` — current deps; confirmed `pyarrow>=14.0` in base, `zeus-ml` (incorrect) in zeus extra
- Zeus PyPI page — confirmed `zeus` package 0.15.0 released 2026-02-24
- Zeus official docs (ml.energy/zeus/measure/) — `from zeus.monitor import ZeusMonitor`; `begin_window`/`end_window`; `Measurement(time, energy={idx: joules})`

### Secondary (MEDIUM confidence)

- `tests/unit/test_core_energy_backends.py` — existing registry API and CodeCarbonBackend test patterns
- `tests/unit/test_core_baseline.py` — baseline module test coverage (all paths verified)
- `tests/unit/test_flops_estimator.py` — existing FLOPs test patterns (will need partial update for PaLM)

### Tertiary (LOW confidence)

- Trapezoidal integration accuracy: ~5-15% error claim — cited from energy-backends.md design doc; original source is NVML API docs + Burtscher et al. (arXiv:2312.02741); not re-verified directly

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — verified against pyproject.toml, codebase, Zeus 0.15.0 PyPI, Zeus docs
- Architecture: HIGH — most patterns are extensions of existing working code
- Pitfalls: HIGH — all identified from direct codebase reading (not speculation)
- FLOPs rewrite: HIGH — formula from locked decision doc; risk is test update scope (LOW effort)
- Zeus API: MEDIUM — `end_window` return shape verified via docs; `energy` field key type (int vs str) not confirmed from source code

**Research date:** 2026-02-26
**Valid until:** 2026-03-28 (stable ecosystem — Zeus 0.15.0, pyarrow 14.0+)
