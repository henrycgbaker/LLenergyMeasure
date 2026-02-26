# Energy Backends Design

**Last updated**: 2026-02-25
**Source research**: [../research/08-energy-plugin-architecture.md](../research/08-energy-plugin-architecture.md)
**Status**: v2.1 target (plugin extension of v2.0 base)

---

## Current State

`EnergyBackend` Protocol + registry exists in `src/llenergymeasure/core/`. One implementation
(CodeCarbon). Factory hardcodes `get_energy_backend("codecarbon")`.

```python
# protocols.py (existing)
class EnergyBackend(Protocol):
    @property
    def name(self) -> str: ...
    def start_tracking(self) -> Any: ...
    def stop_tracking(self, tracker: Any) -> EnergyMetrics: ...
    def is_available(self) -> bool: ...
```

---

## Target State (v2.1)

### `energy:` block in `ExperimentConfig`

```yaml
energy:
  backend: auto     # auto | nvml | codecarbon | zeus | null
  # auto = prefer Zeus if available, fallback to NVML, fallback to CodeCarbon
```

### Backend Registry

| Extra | Backend | Accuracy | Notes |
|-------|---------|----------|-------|
| *(base)* | `NVMLBackend` | ±5W (see below) | Default; polls NVML power at intervals |
| `[codecarbon]` | `CodeCarbonBackend` | ±5W + polling error | NVML polling via CodeCarbon; TDP fallback (~20–30%) when NVML unavailable |
| `[zeus]` | `ZeusBackend` | ±5W (hardware counter) | NVML total energy counter (hardware register); most accurate |
| *(built-in)* | `NullEnergyBackend` | n/a | No-op for testing |

Auto-selection: Zeus if available (installed + NVML accessible), else NVML poller, else CodeCarbon.

**Package note:** The Zeus PyPI package is `zeus` (not `zeus-ml`). The old `zeus-ml` package
stopped at v0.11.0 and is abandoned. Install via `pip install zeus>=0.13.1`.

---

## Accuracy Table

> **Revised (2026-02-25):** `.planning/research/PITFALLS.md` (CP-1) corrects the accuracy
> claims. NVML accuracy is **±5 watts** (not ±5%), per the NVIDIA NVML API Reference.
> Percentage error depends on absolute power draw:
>
> | GPU State | Typical Power | ±5W Error | Percentage |
> |-----------|--------------|-----------|------------|
> | A100 idle | ~40W | ±5W | **~12.5%** |
> | A100 under load | ~300W | ±5W | **~1.7%** |
> | H100 SXM under load | ~600W | ±5W | **~0.8%** |
> | Consumer GPU idle | ~15W | ±5W | **~33%** |
>
> **Formula:** `accuracy_pct = 5W / mean_power_W * 100`

| Method | Accuracy | Mechanism |
|--------|----------|-----------|
| Zeus / NVML total energy counter | ±5W (best: ~1–5% under sustained load on Volta+) | Reads total energy counter (hardware register); integrates at hardware level |
| NVML power polling | ±5W + sampling error (~5–15% depending on polling interval and workload duration) | Polls power at intervals, integrates via trapezoidal rule; 100ms NVML update period with 25% sampling coverage |
| CodeCarbon (NVML polling) | ±5W + polling error (~5–15% at 1s interval; higher at default 15s interval) | Same as NVML polling via CodeCarbon wrapper |
| CodeCarbon (TDP estimation) | ~20–30% | TDP-based fallback when NVML unavailable; no direct measurement |

> **Superseded (2026-02-25):** Previous accuracy figures ("~5%", "~10–15%", "~10–20%") were
> fixed percentages. These have been replaced with conditional accuracy dependent on GPU
> power draw, per NVML API documentation and Burtscher et al. (arXiv:2312.02741).
>
> Sources: [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html),
> [Part-time Power Measurements (arXiv:2312.02741)](https://arxiv.org/html/2312.02741v2),
> [ML.ENERGY Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/).

---

## Measurement Correctness Requirements

### CPU-GPU Synchronisation (hard requirement)

`torch.cuda.synchronize()` must be called before stopping energy measurement. Without it,
the GPU may still be executing CUDA kernels when the measurement window closes, under-counting
energy. This is a correctness requirement, not optional.

```python
# Correct pattern:
monitor.begin_window("measurement")
run_inference(...)
torch.cuda.synchronize()   # ← wait for all GPU kernels to finish
measurement = monitor.end_window("measurement")
```

Source: Zeus `sync_execution_with` parameter, ML.ENERGY blog best practices.

### GPU Persistence Mode (pre-flight warning)

GPU persistence mode (`nvidia-smi -pm 1`) eliminates cold-start latency and power state
variation. All serious benchmarking tools assume it is enabled. Pre-flight check warns with
exact fix command if not enabled. Warning (not error) because many researchers run on shared
machines without root access.

Persistence mode status is recorded in `EnvironmentSnapshot.gpu_persistence_mode: bool`.

### Minimum Measurement Duration (quality warning)

NVML has a 100ms update period with ~25% sampling coverage. Short inference runs (single
batch, <10 seconds total) may produce unreliable energy integration. When measurement
duration < 10s, a `measurement_warnings` flag is added to the result.

### Measurement Quality Flags

`measurement_warnings: list[str]` in `ExperimentResult` — stored in result files (primary)
and displayed in CLI summary (secondary). The result file is the permanent record.

Possible flags:
- `"short_measurement_duration"` — measurement < 10s; energy values may be unreliable
- `"gpu_persistence_mode_off"` — power state variation may affect measurements
- `"thermal_drift_detected"` — >5°C temperature change during measurement window
- `"nvml_low_sample_count"` — fewer than 10 NVML power samples collected

---

## Multi-GPU Energy Aggregation

Zeus supports multi-GPU measurement natively via per-device NVML energy counters:

```python
gpu_indices = [0, 1, 2, 3]
monitor = ZeusMonitor(gpu_indices=gpu_indices)
monitor.begin_window("measurement")
# ... inference ...
measurement = monitor.end_window("measurement")
# measurement.gpu_energy is a dict: {gpu_index: joules}
total_energy = sum(measurement.gpu_energy.values())
per_device   = [measurement.gpu_energy[i] for i in gpu_indices]
```

NVML fallback (base poller): poll each device's power draw separately and sum.
Single-GPU case: `gpu_indices=[0]` — same code path, list of length 1.

Total energy (`sum(gpu_energy.values())`) is the primary efficiency comparison metric.
Per-device breakdown goes into `MultiGPUMetrics.energy_per_gpu_j` for load imbalance
diagnostics. See [result-schema.md](result-schema.md).

---

## NVML Single-Session Owner

`PowerThermalSampler` and `GPUUtilisationSampler` also use `nvidia-ml-py` (NVML bindings)
directly. Multiple NVML sessions are supported by the driver, but concurrent access needs
validation testing.

The `get_active_energy_backend()` factory (see [architecture.md](architecture.md)) enforces
mutual exclusion between Zeus and the base NVML poller — only one can own the NVML session.

The 4-phase migration in `../research/08-energy-plugin-architecture.md` includes a
cross-validation phase (run both backends simultaneously) to surface any conflicts.

### vLLM CUDA Context Interaction (2026-02-26)

> Source: `.planning/codebase/CONCERNS.md` — "pynvml Thread Safety with vLLM";
> `preservation_audit/P-19` — GPUUtilisationSampler graceful degradation.

The v1.x codebase flagged a potential race: vLLM initialises its own CUDA context in worker
processes, while our `GPUUtilisationSampler` and `PowerThermalSampler` call `pynvml` in a
background thread. In practice this is safe:

- The pynvml C library is thread-safe for read operations
- vLLM initialises CUDA in **worker processes** (not the host process)
- v2.0 subprocess isolation means experiments run in child processes — the NVML sampler runs
  in the parent, vLLM in the child. They never share a process.
- The sampler already degrades gracefully: if NVML init fails, it returns empty samples rather
  than crashing (see `preservation_audit/P-19`)

**No design change required.** Subprocess isolation resolves the concern by construction.

---

## Related

- [architecture.md](architecture.md): NVML single-session owner pattern
- [experiment-config.md](experiment-config.md): Full ExperimentConfig schema
- [../research/08-energy-plugin-architecture.md](../research/08-energy-plugin-architecture.md): Full Protocol changes, ZeusBackend implementation
