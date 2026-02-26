# Energy Measurement Plugin Architecture

**Date**: 2026-02-17
**Status**: Research complete, ready for implementation design review
**Author**: Research Scientist Agent

---

## Table of Contents

1. [Current Architecture Analysis](#1-current-architecture-analysis)
2. [Zeus API Analysis](#2-zeus-api-analysis)
3. [CodeCarbon API Analysis](#3-codecarbon-api-analysis)
4. [Zeus vs CodeCarbon Comparison](#4-zeus-vs-codecarbon-comparison)
5. [Other Energy Measurement Tools](#5-other-energy-measurement-tools)
6. [Proposed Plugin Architecture](#6-proposed-plugin-architecture)
7. [Configuration Design](#7-configuration-design)
8. [Migration Path](#8-migration-path)
9. [Cross-Validation Mode](#9-cross-validation-mode)
10. [Open Questions](#10-open-questions)

---

## 1. Current Architecture Analysis

### 1.1 Existing Plugin Infrastructure

LLenergyMeasure already has a plugin registry pattern for energy backends. The
infrastructure is well-designed but currently has only one implementation
(CodeCarbon).

**Key files and their roles:**

| File | Role |
|------|------|
| `src/llenergymeasure/protocols.py` (L86-122) | `EnergyBackend` Protocol definition |
| `src/llenergymeasure/core/energy_backends/base.py` | Re-exports protocol |
| `src/llenergymeasure/core/energy_backends/codecarbon.py` | CodeCarbon implementation |
| `src/llenergymeasure/core/energy_backends/__init__.py` | Plugin registry (`register_backend`, `get_backend`, `list_backends`) |
| `src/llenergymeasure/orchestration/factory.py` (L147) | Hardcoded: `get_energy_backend("codecarbon")` |
| `src/llenergymeasure/orchestration/runner.py` (L224-255) | Start/stop tracking lifecycle |
| `src/llenergymeasure/domain/metrics.py` (L223-256) | `EnergyMetrics` Pydantic model |

### 1.2 Current EnergyBackend Protocol

```python
# From src/llenergymeasure/protocols.py
@runtime_checkable
class EnergyBackend(Protocol):
    @property
    def name(self) -> str: ...
    def start_tracking(self) -> Any: ...
    def stop_tracking(self, tracker: Any) -> EnergyMetrics: ...
    def is_available(self) -> bool: ...
```

**Observations:**
- The protocol is minimal and clean.
- `start_tracking()` returns `Any` (an opaque tracker handle) -- this is
  appropriate for the adapter pattern since different backends need different
  state objects.
- `stop_tracking()` returns our unified `EnergyMetrics` -- good normalisation
  point.
- Missing: no method to query backend capabilities (what metrics it can provide).

### 1.3 Current Data Flow

```
factory.py::create_components()
    --> get_energy_backend("codecarbon")  # HARDCODED
    --> returns CodeCarbonBackend instance

runner.py::ExperimentOrchestrator.run()
    --> tracker = self._energy.start_tracking()
    --> [inference runs]
    --> energy_metrics = self._energy.stop_tracking(tracker)
    --> [metrics fed into RawProcessResult]
```

### 1.4 Parallel Energy Infrastructure (NVML-based)

In addition to the CodeCarbon backend, the codebase has separate NVML-based
measurement infrastructure that operates in parallel:

| Component | File | Purpose |
|-----------|------|---------|
| `PowerThermalSampler` | `core/power_thermal.py` | Background GPU power/temp/throttle sampling |
| `GPUUtilisationSampler` | `core/gpu_utilisation.py` | Background GPU utilisation sampling |
| `measure_baseline_power()` | `core/baseline.py` | Idle power baseline measurement |
| `create_energy_breakdown()` | `core/baseline.py` | Baseline-adjusted energy breakdown |

These components use `pynvml` (NVML) directly and run alongside CodeCarbon.
This is important context because Zeus also uses NVML for NVIDIA GPUs, which
means there could be NVML session conflicts if both the existing samplers and
Zeus try to `nvmlInit()`/`nvmlShutdown()` simultaneously.

### 1.5 EnergyMetrics Schema

```python
class EnergyMetrics(BaseModel):
    total_energy_j: float       # Total energy consumed in Joules
    gpu_energy_j: float         # GPU energy in Joules
    cpu_energy_j: float         # CPU energy in Joules
    ram_energy_j: float         # RAM energy in Joules
    gpu_power_w: float          # Average GPU power in Watts
    cpu_power_w: float          # Average CPU power in Watts
    duration_sec: float         # Measurement duration in seconds
    emissions_kg_co2: float     # Carbon emissions in kg CO2
    energy_per_token_j: float   # Energy per token in Joules
```

This schema already accommodates component-level energy breakdowns (GPU, CPU,
RAM) which both CodeCarbon and Zeus can populate.

### 1.6 Hardcoded Coupling Points

There are exactly **two** hardcoded coupling points to CodeCarbon:

1. **`factory.py` L147**: `energy_backend=get_energy_backend("codecarbon")` --
   always selects CodeCarbon regardless of config.
2. **`__init__.py` L94**: `register_backend("codecarbon", CodeCarbonBackend)` --
   only registers CodeCarbon at import time.

There is **no config field** for selecting the energy measurement backend. This
is the primary gap.

---

## 2. Zeus API Analysis

### 2.1 Overview

Zeus (v0.13.1, from ML.ENERGY/CMU) is a deep learning energy measurement and
optimisation framework. It provides:

- Direct hardware energy counter access (not software estimation)
- Multi-GPU support with `CUDA_VISIBLE_DEVICES` awareness
- CPU and DRAM energy via Intel RAPL
- Apple Silicon SoC energy (via `zeus-apple-silicon` companion package)
- AMD GPU energy via `amdsmi`
- NVIDIA Jetson platform support
- Named measurement windows (concurrent measurements supported)

### 2.2 Core API: ZeusMonitor

```python
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(
    gpu_indices=[0, 1],           # GPUs to monitor (None = all)
    cpu_indices=[0],              # CPU packages (None = all)
    approx_instant_energy=False,  # Approximate for short windows
    log_file=None,                # Optional CSV log
    sync_execution_with="torch",  # Framework sync (torch/jax/cupy)
)

monitor.begin_window("inference")
# ... run inference ...
result = monitor.end_window("inference")

# Result is a Measurement dataclass:
result.time            # float: seconds
result.gpu_energy      # dict[int, float]: GPU index -> Joules
result.cpu_energy      # dict[int, float] | None: CPU index -> Joules
result.dram_energy     # dict[int, float] | None: CPU index -> Joules
result.soc_energy      # SoCMeasurement | None: Apple Silicon metrics
result.total_energy    # float: sum of GPU energy (Joules)
```

### 2.3 Measurement Mechanism

Zeus uses **hardware energy counters** rather than software estimation:

| Hardware | Mechanism | Resolution | Overhead |
|----------|-----------|------------|----------|
| NVIDIA Volta+ | `nvmlDeviceGetTotalEnergyConsumption` | ~10ms counter update | Single NVML call |
| NVIDIA pre-Volta | Power polling subprocess + integration | ~100ms | Separate process |
| AMD GPUs | `amdsmi` energy counters | Hardware-dependent | Direct call |
| Intel CPUs | RAPL `/sys/class/powercap/intel-rapl` | ~1ms | File read |
| Apple Silicon | `zeus-apple-silicon` IOKit | Hardware-dependent | Direct call |
| NVIDIA Jetson | Jetson power sensors | Hardware-dependent | File read |

**Key advantage**: For Volta+ GPUs, Zeus reads the cumulative energy counter at
window start and end, then subtracts. This avoids the polling-and-integration
approach that CodeCarbon uses, eliminating sampling frequency as an error source.

### 2.4 Multi-GPU Support

Zeus is multi-GPU aware by design:
- `gpu_indices` parameter selects which GPUs to monitor
- `CUDA_VISIBLE_DEVICES` is respected
- Energy is reported per-GPU as `dict[int, float]`
- Execution synchronisation (`torch.cuda.synchronize`) at window boundaries

### 2.5 CPU and DRAM Measurement

Zeus provides CPU and DRAM energy via RAPL:
- Requires read access to `/sys/class/powercap/intel-rapl/`
- In containers, requires bind-mounting RAPL sysfs to `/zeus_sys/`
- Reports per-CPU-package and per-DRAM-domain
- Handles RAPL counter wraparound via a background monitoring process

### 2.6 SoC Support (Apple Silicon)

Zeus's `AppleSiliconMeasurement` provides:
- `cpu_total_mj`: Total CPU energy
- `efficiency_cores_mj` / `performance_cores_mj`: Per-core-type energy
- `gpu_mj`: On-chip GPU energy
- `dram_mj`: Memory energy
- `ane_mj`: Apple Neural Engine energy

### 2.7 Important Caveats

1. **NVML conflict risk**: Zeus calls `nvmlInit()` via its `get_gpus()` device
   layer. If our `PowerThermalSampler` also holds an NVML session, there could
   be conflicts. Needs testing -- NVML is generally re-entrant, but care is
   needed with shutdown ordering.

2. **Background process for pre-Volta**: Zeus spawns a `PowerMonitor` subprocess
   for older GPUs. This must be guarded with `if __name__ == "__main__"` or
   used only within an already-spawned worker.

3. **RAPL permissions**: CPU energy requires either root or
   `/sys/class/powercap/intel-rapl` read permission. In Docker, this means
   bind-mounting the sysfs path.

4. **Framework sync**: Zeus calls `torch.cuda.synchronize()` at window
   boundaries. This is desirable for accuracy but adds a small latency (~0.1ms).

5. **No carbon emissions**: Zeus does not calculate CO2 emissions. If we want
   that, we need to keep CodeCarbon's carbon intensity integration or implement
   our own.

---

## 3. CodeCarbon API Analysis

### 3.1 Core API

```python
from codecarbon import EmissionsTracker

tracker = EmissionsTracker(
    measure_power_secs=1,       # Polling interval (seconds)
    allow_multiple_runs=True,
    tracking_mode="process",    # "process" or "machine"
    log_level=logging.ERROR,
)

tracker.start()
# ... run inference ...
emissions = tracker.stop()     # Returns total emissions in kg CO2

# Internal data extraction (our current approach):
data = tracker._prepare_emissions_data()
data.energy_consumed    # Total energy in kWh
data.cpu_power          # CPU power in W
data.gpu_power          # GPU power in W
data.ram_power          # RAM power in W
data.cpu_energy         # CPU energy in kWh
data.gpu_energy         # GPU energy in kWh
data.ram_energy         # RAM energy in kWh
data.emissions          # CO2 in kg
```

### 3.2 Measurement Mechanism

CodeCarbon uses a **polling-and-integration** approach:

| Component | Mechanism | Accuracy |
|-----------|-----------|----------|
| GPU power | NVML `nvmlDeviceGetPowerUsage` polled at interval | +/- 5-10%, depends on polling rate |
| CPU power | Intel RAPL or TDP estimation | Variable |
| RAM power | Estimation based on DIMM count/type | Low accuracy |
| Carbon | Grid carbon intensity APIs | Region-dependent |

### 3.3 Known Limitations

1. **`duration_sec=0.0` bug**: CodeCarbon does not reliably report measurement
   duration. Our codebase already works around this (L183-184 of
   `codecarbon.py`: "Duration is difficult to get from CodeCarbon directly").

2. **Polling overhead**: At `measure_power_secs=1` (default), CodeCarbon spawns
   a background thread polling NVML. At higher frequencies, CPU overhead
   increases. CodeCarbon does not support sub-second polling well.

3. **Private API usage**: Our implementation calls
   `tracker._prepare_emissions_data()` which is a private/internal method.
   This is fragile across CodeCarbon version upgrades.

4. **kWh units**: CodeCarbon reports energy in kWh, requiring conversion to
   Joules. This is a minor inconvenience but adds a conversion step.

5. **Verbose logging**: CodeCarbon generates significant logging output and
   FutureWarnings from pandas, requiring suppression (already handled in our
   codebase).

6. **Container issues**: CodeCarbon can fail in containers without NVML
   permissions, which our codebase already handles gracefully.

7. **No per-GPU breakdown**: CodeCarbon reports aggregate GPU energy, not
   per-GPU. In multi-GPU setups, you cannot see which GPU consumed how much.

### 3.4 Strengths

1. **Carbon emissions**: Built-in CO2 calculation with regional grid data.
2. **RAM estimation**: Attempts RAM power estimation (Zeus does not).
3. **Mature ecosystem**: Widely used, good documentation, active maintenance.
4. **Simple API**: Start/stop is very straightforward.

---

## 4. Zeus vs CodeCarbon Comparison

### 4.1 Feature Matrix

| Feature | Zeus (0.13.1) | CodeCarbon (2.8+) |
|---------|---------------|-------------------|
| **GPU energy (NVIDIA Volta+)** | Direct counter (high accuracy) | Power polling + integration |
| **GPU energy (NVIDIA pre-Volta)** | Power polling subprocess | Power polling thread |
| **GPU energy (AMD)** | Via amdsmi | Not supported |
| **CPU energy** | RAPL direct read | RAPL or TDP estimation |
| **DRAM energy** | RAPL sub-package | Estimation (low accuracy) |
| **Apple Silicon** | Via zeus-apple-silicon | Not supported |
| **NVIDIA Jetson** | Supported | Not supported |
| **CO2 emissions** | Not supported | Built-in (grid APIs) |
| **Per-GPU breakdown** | Yes (dict per GPU) | No (aggregate only) |
| **Multi-GPU awareness** | Native | Limited |
| **Measurement overhead** | ~0.1ms (counter read) | ~5-10ms per poll |
| **Concurrent windows** | Named windows | Single tracker |
| **Energy units** | Joules (native) | kWh (requires conversion) |
| **Framework sync** | torch/jax/cupy sync | None |
| **Container support** | RAPL bind-mount needed | NVML access needed |
| **Python version** | 3.9+ | 3.7+ |
| **License** | Apache 2.0 | MIT |

### 4.2 Accuracy Comparison

| Scenario | Zeus | CodeCarbon |
|----------|------|------------|
| Short inference (<1s) | Accurate (counter diff) | May report 0 (polling missed) |
| Long inference (>10s) | Accurate | Reasonable (~5-10% error) |
| Multi-GPU | Per-GPU accuracy | Aggregate only |
| CPU energy | RAPL accuracy (~2%) | Estimation (~10-20%) |
| DRAM energy | RAPL accuracy (~5%) | Estimation (~30-50%) |

### 4.3 Recommendation

**Zeus should be the recommended backend** for energy measurement when available,
with CodeCarbon as a fallback. The reasons are:

1. Direct hardware counter access is fundamentally more accurate than polling.
2. Per-GPU energy breakdown is essential for multi-GPU experiments.
3. Broader hardware support (AMD, Apple Silicon, Jetson).
4. Lower measurement overhead.
5. Native Joules output matches our `EnergyMetrics` schema.

**CodeCarbon should be retained** for:
1. CO2 emissions calculation (Zeus does not provide this).
2. Environments where Zeus is not installable.
3. Cross-validation against Zeus measurements.

---

## 5. Other Energy Measurement Tools

### 5.1 Assessment Matrix

| Tool | Type | GPU | CPU | Accuracy | Overhead | Maturity | Priority |
|------|------|-----|-----|----------|----------|----------|----------|
| **Zeus** | Library | NVIDIA, AMD, Apple, Jetson | RAPL | High | Low | High | **P0** |
| **CodeCarbon** | Library | NVIDIA | RAPL/est. | Medium | Medium | High | **P0 (current)** |
| **NVIDIA DCGM** | Service | NVIDIA only | No | Very high | Very low | Very high | **P1** |
| **Intel RAPL (direct)** | Sysfs | No | Intel only | High | Very low | N/A | P2 |
| **Scaphandre** | Agent | No | Intel/AMD | High | Low | Medium | P2 |
| **PowerJoular** | Agent | NVIDIA | Intel/AMD | Medium | Low | Medium | P3 |
| **NVIDIA SMI** | CLI | NVIDIA | No | Medium | Medium | Very high | P3 |
| **Intel Power Gadget** | GUI/API | No | Intel only | Medium | Medium | Deprecated | Not recommended |

### 5.2 NVIDIA DCGM (Data Center GPU Manager)

DCGM is NVIDIA's enterprise GPU monitoring solution. It provides:
- High-frequency GPU metrics (power, temperature, utilisation, memory)
- Historical data collection
- Health monitoring and diagnostics
- Multi-GPU, multi-node support
- gRPC and REST APIs

**Integration approach**: DCGM runs as a system service (`nv-hostengine`).
A Python backend would use `dcgm-python` to query metrics. This is particularly
valuable in cluster/HPC environments where DCGM is already deployed.

**Considerations**:
- Requires DCGM service to be running (not just a library import)
- Enterprise NVIDIA feature; availability varies
- Best accuracy for GPU metrics but no CPU/DRAM data
- Would be a natural fit for data centre deployments

### 5.3 Scaphandre

Scaphandre is a Rust-based energy measurement agent that:
- Reads Intel RAPL and AMD RAPL for CPU/DRAM energy
- Provides per-process energy attribution
- Exports to Prometheus, JSON, or stdout
- Runs as a daemon or CLI tool

**Integration approach**: Run Scaphandre alongside experiments, query its
Prometheus endpoint or parse JSON output. More complex than library-based
approaches but provides system-level energy context.

### 5.4 PowerJoular

PowerJoular is a C-based tool that:
- Uses Intel RAPL for CPU energy
- Uses NVIDIA SMI for GPU power
- Per-process energy tracking
- Lightweight daemon

**Integration approach**: Similar to Scaphandre -- external process, parse output.

### 5.5 Direct RAPL Access

For CPU-only energy measurement, reading RAPL sysfs directly is the simplest
approach:
```
/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj  # Package 0 energy
/sys/class/powercap/intel-rapl/intel-rapl:0:0/energy_uj  # Core energy
/sys/class/powercap/intel-rapl/intel-rapl:0:2/energy_uj  # DRAM energy
```

Zeus already does this internally, so a direct RAPL backend would duplicate
Zeus's CPU measurement capability. Only worthwhile if Zeus is not available but
RAPL is.

### 5.6 Recommendations for Plugin Priority

| Phase | Backends | Rationale |
|-------|----------|-----------|
| **Phase 1** | CodeCarbon (existing) + Zeus | Covers 95% of use cases, biggest accuracy improvement |
| **Phase 2** | NVIDIA DCGM | Enterprise/HPC deployments |
| **Phase 3** | Direct RAPL, Scaphandre | Niche use cases, CPU-only environments |

---

## 6. Proposed Plugin Architecture

### 6.1 Design Principles

1. **Backward compatible**: Existing CodeCarbon behaviour is preserved as default.
2. **Auto-detection**: When `energy_backend: auto`, select the best available backend.
3. **Capability-aware**: Backends declare what they can measure.
4. **Normalised output**: All backends produce `EnergyMetrics`.
5. **Extensible**: New backends can be added without touching existing code.
6. **Cross-validation**: Allow running two backends simultaneously.

### 6.2 Enhanced EnergyBackend Protocol

The current protocol needs two enhancements: capability declaration and richer
measurement data.

```python
# src/llenergymeasure/protocols.py

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.domain.metrics import EnergyMetrics


class EnergyCapability(Flag):
    """Capabilities an energy backend can provide."""
    GPU_ENERGY = auto()        # Total GPU energy
    GPU_ENERGY_PER_DEVICE = auto()  # Per-GPU energy breakdown
    CPU_ENERGY = auto()        # CPU package energy
    DRAM_ENERGY = auto()       # DRAM energy
    SOC_ENERGY = auto()        # SoC energy (Apple Silicon, Jetson)
    GPU_POWER = auto()         # Instantaneous GPU power
    CPU_POWER = auto()         # Instantaneous CPU power
    CARBON_EMISSIONS = auto()  # CO2 emissions calculation
    FRAMEWORK_SYNC = auto()    # Synchronises with DL framework


@dataclass
class EnergyBackendInfo:
    """Metadata about an energy measurement backend."""
    name: str
    version: str
    capabilities: EnergyCapability
    supported_hardware: list[str] = field(default_factory=list)
    # e.g., ["nvidia_volta+", "nvidia_pre_volta", "amd", "intel_cpu", "apple_silicon"]
    measurement_method: str = "unknown"
    # e.g., "hardware_counter", "power_polling", "estimation"
    typical_accuracy_pct: float = 10.0
    # Typical measurement error percentage


@runtime_checkable
class EnergyBackend(Protocol):
    """Protocol for energy measurement backends.

    Implementations include CodeCarbon, Zeus, DCGM, etc.
    """

    @property
    def name(self) -> str:
        """Backend name for identification."""
        ...

    def get_info(self) -> EnergyBackendInfo:
        """Return backend metadata and capabilities."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    def start_tracking(self) -> Any:
        """Start energy tracking.

        Returns:
            Tracker handle to pass to stop_tracking.
        """
        ...

    def stop_tracking(self, tracker: Any) -> EnergyMetrics:
        """Stop energy tracking and return metrics.

        Args:
            tracker: Handle from start_tracking.

        Returns:
            Energy metrics from the tracking period.
        """
        ...
```

**Backward compatibility note**: The new `get_info()` method is additive.
The existing CodeCarbon backend can implement it, and the `EnergyCapability`
flag enum defaults gracefully. The old protocol methods (`name`,
`start_tracking`, `stop_tracking`, `is_available`) are unchanged.

### 6.3 Extended EnergyMetrics

The current `EnergyMetrics` schema already supports component-level breakdown.
We should add optional fields for richer data without breaking existing consumers:

```python
# Additions to src/llenergymeasure/domain/metrics.py

class EnergyMetrics(BaseModel):
    # ... existing fields ...

    # NEW: Per-device GPU energy (Zeus provides this)
    gpu_energy_per_device_j: dict[int, float] | None = Field(
        default=None,
        description="Per-GPU energy in Joules, keyed by device index",
    )

    # NEW: DRAM energy (Zeus RAPL provides this)
    dram_energy_j: float = Field(
        default=0.0,
        description="DRAM energy in Joules (via RAPL)",
    )

    # NEW: Measurement backend metadata
    energy_backend_name: str | None = Field(
        default=None,
        description="Name of the energy measurement backend used",
    )
    energy_measurement_method: str | None = Field(
        default=None,
        description="Measurement method (hardware_counter, power_polling, estimation)",
    )
```

**Note**: `ram_energy_j` already exists in `EnergyMetrics`. For CodeCarbon,
this is an estimation. For Zeus, we would populate the new `dram_energy_j` field
with the RAPL-measured value and leave `ram_energy_j` for the CodeCarbon
estimation. Alternatively, we could consolidate these -- this is a schema
decision that needs PM input.

### 6.4 Zeus Backend Implementation

```python
# src/llenergymeasure/core/energy_backends/zeus.py

from __future__ import annotations

import logging
from typing import Any

from loguru import logger

from llenergymeasure.domain.metrics import EnergyMetrics


class ZeusBackend:
    """Energy tracking backend using Zeus (ml-energy/zeus).

    Zeus provides direct hardware energy counter access for NVIDIA (Volta+),
    AMD, Apple Silicon, and Jetson GPUs, plus Intel RAPL for CPU/DRAM.

    Implements the EnergyBackend protocol.
    """

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        cpu_indices: list[int] | None = None,
        approx_instant_energy: bool = True,
        sync_execution_with: str = "torch",
    ) -> None:
        self._gpu_indices = gpu_indices
        self._cpu_indices = cpu_indices
        self._approx_instant = approx_instant_energy
        self._sync_with = sync_execution_with
        self._window_name = "lem_inference"

    @property
    def name(self) -> str:
        return "zeus"

    def get_info(self) -> "EnergyBackendInfo":
        from llenergymeasure.protocols import EnergyBackendInfo, EnergyCapability

        caps = EnergyCapability.GPU_ENERGY | EnergyCapability.GPU_ENERGY_PER_DEVICE
        caps |= EnergyCapability.FRAMEWORK_SYNC

        # Detect CPU/DRAM capability
        try:
            from zeus.device import get_cpus
            cpus = get_cpus()
            if len(cpus) > 0:
                caps |= EnergyCapability.CPU_ENERGY | EnergyCapability.DRAM_ENERGY
        except Exception:
            pass

        # Detect SoC
        try:
            from zeus.device import get_soc
            soc = get_soc()
            caps |= EnergyCapability.SOC_ENERGY
        except Exception:
            pass

        hardware = []
        try:
            from zeus.device import get_gpus
            gpus = get_gpus()
            if len(gpus) > 0:
                hardware.append("nvidia" if hasattr(gpus, '_nvml_available') else "gpu")
        except Exception:
            pass

        import zeus
        return EnergyBackendInfo(
            name="zeus",
            version=getattr(zeus, "__version__", "unknown"),
            capabilities=caps,
            supported_hardware=hardware,
            measurement_method="hardware_counter",
            typical_accuracy_pct=2.0,
        )

    def is_available(self) -> bool:
        try:
            from zeus.monitor import ZeusMonitor
            # Try to instantiate -- this will check for GPU/CPU availability
            monitor = ZeusMonitor(
                gpu_indices=self._gpu_indices,
                cpu_indices=self._cpu_indices or [],
            )
            return len(monitor.gpu_indices) > 0 or len(monitor.cpu_indices) > 0
        except Exception:
            return False

    def start_tracking(self) -> Any:
        try:
            from zeus.monitor import ZeusMonitor

            monitor = ZeusMonitor(
                gpu_indices=self._gpu_indices,
                cpu_indices=self._cpu_indices,
                approx_instant_energy=self._approx_instant,
                sync_execution_with=self._sync_with,
            )
            monitor.begin_window(self._window_name)
            logger.debug("Zeus energy tracking started")
            return monitor
        except Exception as e:
            logger.warning(f"Zeus energy tracking unavailable: {e}")
            return None

    def stop_tracking(self, tracker: Any) -> EnergyMetrics:
        if tracker is None:
            return self._empty_metrics()

        try:
            measurement = tracker.end_window(self._window_name)
            return self._convert_measurement(measurement)
        except Exception as e:
            logger.error(f"Failed to stop Zeus energy tracking: {e}")
            return self._empty_metrics()

    def _convert_measurement(self, measurement: Any) -> EnergyMetrics:
        """Convert Zeus Measurement to our EnergyMetrics."""
        # GPU energy: sum across all GPUs
        total_gpu_energy_j = sum(measurement.gpu_energy.values())

        # CPU energy: sum across all packages
        total_cpu_energy_j = 0.0
        if measurement.cpu_energy:
            total_cpu_energy_j = sum(measurement.cpu_energy.values())

        # DRAM energy: sum across all packages
        total_dram_energy_j = 0.0
        if measurement.dram_energy:
            total_dram_energy_j = sum(measurement.dram_energy.values())

        # SoC energy (Apple Silicon)
        soc_energy_j = 0.0
        # (Would need to extract from measurement.soc_energy if present)

        total_energy_j = total_gpu_energy_j + total_cpu_energy_j + total_dram_energy_j

        # Compute average power from energy and time
        duration = measurement.time
        gpu_power_w = total_gpu_energy_j / duration if duration > 0 else 0.0
        cpu_power_w = total_cpu_energy_j / duration if duration > 0 else 0.0

        return EnergyMetrics(
            total_energy_j=total_energy_j,
            gpu_energy_j=total_gpu_energy_j,
            cpu_energy_j=total_cpu_energy_j,
            ram_energy_j=total_dram_energy_j,  # DRAM from RAPL, not estimation
            gpu_power_w=gpu_power_w,
            cpu_power_w=cpu_power_w,
            duration_sec=duration,
            emissions_kg_co2=0.0,  # Zeus does not calculate CO2
            energy_per_token_j=0.0,  # Caller sets this
            # NEW fields (when schema is extended):
            # gpu_energy_per_device_j=dict(measurement.gpu_energy),
            # energy_backend_name="zeus",
            # energy_measurement_method="hardware_counter",
        )

    def _empty_metrics(self) -> EnergyMetrics:
        return EnergyMetrics(
            total_energy_j=0.0,
            gpu_energy_j=0.0,
            cpu_energy_j=0.0,
            ram_energy_j=0.0,
            gpu_power_w=0.0,
            cpu_power_w=0.0,
            duration_sec=0.0,
            emissions_kg_co2=0.0,
            energy_per_token_j=0.0,
        )
```

### 6.5 Auto-Detection Logic

```python
# src/llenergymeasure/core/energy_backends/__init__.py (enhanced)

def get_best_available_backend(**kwargs: object) -> EnergyBackend:
    """Select the best available energy backend.

    Priority order:
    1. Zeus (highest accuracy, broadest hardware support)
    2. CodeCarbon (good fallback, has CO2 calculation)
    3. Null backend (no energy measurement)

    Returns:
        Best available EnergyBackend instance.
    """
    # Try Zeus first
    if "zeus" in _BACKENDS:
        backend = _BACKENDS["zeus"](**kwargs)
        if backend.is_available():
            logger.info("Auto-selected energy backend: zeus")
            return backend

    # Fall back to CodeCarbon
    if "codecarbon" in _BACKENDS:
        backend = _BACKENDS["codecarbon"](**kwargs)
        if backend.is_available():
            logger.info("Auto-selected energy backend: codecarbon")
            return backend

    # No backend available
    logger.warning("No energy measurement backend available")
    return NullEnergyBackend()
```

### 6.6 Null Backend (for graceful degradation)

```python
# src/llenergymeasure/core/energy_backends/null.py

class NullEnergyBackend:
    """No-op energy backend for when no measurement is available.

    Returns zero-valued metrics. Useful for:
    - Systems without GPU/RAPL support
    - Testing and development
    - Disabling energy measurement via config
    """

    @property
    def name(self) -> str:
        return "null"

    def get_info(self) -> EnergyBackendInfo:
        return EnergyBackendInfo(
            name="null",
            version="1.0.0",
            capabilities=EnergyCapability(0),
            measurement_method="none",
            typical_accuracy_pct=0.0,
        )

    def is_available(self) -> bool:
        return True  # Always available

    def start_tracking(self) -> Any:
        return None

    def stop_tracking(self, tracker: Any) -> EnergyMetrics:
        return EnergyMetrics.placeholder()
```

### 6.7 Registry Enhancement

```python
# Enhanced _register_default_backends() in __init__.py

def _register_default_backends() -> None:
    """Register built-in backends."""
    # Always available
    register_backend("codecarbon", CodeCarbonBackend)
    register_backend("null", NullEnergyBackend)

    # Conditionally register Zeus
    try:
        from llenergymeasure.core.energy_backends.zeus import ZeusBackend
        register_backend("zeus", ZeusBackend)
    except ImportError:
        logger.debug("Zeus not installed, zeus energy backend not available")
```

---

## 7. Configuration Design

### 7.1 Config Schema Addition

Add to `ExperimentConfig` in `src/llenergymeasure/config/models.py`:

```python
class EnergyConfig(BaseModel):
    """Energy measurement configuration."""

    backend: Literal["auto", "zeus", "codecarbon", "null"] = Field(
        default="auto",
        description=(
            "Energy measurement backend. 'auto' selects the best available "
            "(Zeus > CodeCarbon > null). 'null' disables energy measurement."
        ),
    )

    # CodeCarbon-specific
    measure_power_secs: int = Field(
        default=1,
        ge=1,
        description="CodeCarbon polling interval in seconds",
    )
    tracking_mode: Literal["process", "machine"] = Field(
        default="process",
        description="CodeCarbon tracking mode",
    )

    # Zeus-specific
    zeus_approx_instant_energy: bool = Field(
        default=True,
        description="Zeus: approximate energy for very short measurement windows",
    )
    zeus_sync_execution: bool = Field(
        default=True,
        description="Zeus: synchronise DL framework execution at window boundaries",
    )

    # Cross-validation
    cross_validate: bool = Field(
        default=False,
        description="Run both Zeus and CodeCarbon simultaneously for cross-validation",
    )


class ExperimentConfig(BaseModel):
    # ... existing fields ...

    energy: EnergyConfig = Field(
        default_factory=EnergyConfig,
        description="Energy measurement configuration",
    )
```

### 7.2 YAML Config Example

```yaml
config_name: energy-comparison
model_name: meta-llama/Llama-2-7b-hf

energy:
  backend: auto            # or: zeus, codecarbon, null
  cross_validate: false    # set true to run both backends

  # CodeCarbon-specific (used when backend is codecarbon)
  measure_power_secs: 1
  tracking_mode: process

  # Zeus-specific (used when backend is zeus)
  zeus_approx_instant_energy: true
  zeus_sync_execution: true

baseline:
  enabled: true
  duration_sec: 30
```

### 7.3 Factory Integration

Update `src/llenergymeasure/orchestration/factory.py`:

```python
def _create_backend_components(ctx, backend, results_dir=None):
    # ... existing code ...

    # Select energy backend from config
    energy_config = getattr(ctx.config, "energy", None)
    energy_backend_name = energy_config.backend if energy_config else "auto"

    if energy_backend_name == "auto":
        energy_backend = get_best_available_backend()
    else:
        energy_backend = get_energy_backend(energy_backend_name)

    return ExperimentComponents(
        # ... existing ...
        energy_backend=energy_backend,
        # ...
    )
```

---

## 8. Migration Path

### 8.1 Phase 1: Add Zeus Backend (Non-Breaking)

**Changes:**
1. Add `zeus` as optional dependency in `pyproject.toml`
2. Create `src/llenergymeasure/core/energy_backends/zeus.py`
3. Create `src/llenergymeasure/core/energy_backends/null.py`
4. Update registry in `__init__.py` to conditionally register Zeus
5. Update factory to respect `energy_backend` config
6. Add `EnergyConfig` to `ExperimentConfig`
7. Default `backend: "codecarbon"` (not `auto`) for backward compatibility

**No breaking changes**: Existing configs work unchanged. CodeCarbon remains
default.

### 8.2 Phase 2: Switch Default to Auto-Detection

**Changes:**
1. Change default from `"codecarbon"` to `"auto"`
2. Add `EnergyCapability` and `EnergyBackendInfo` to protocol
3. Enhance `EnergyMetrics` with optional per-device and metadata fields
4. Add tests for auto-detection logic

**Migration note**: Users who want CodeCarbon specifically can set
`energy.backend: codecarbon` in their config.

### 8.3 Phase 3: Cross-Validation Mode

**Changes:**
1. Implement `CrossValidationBackend` that wraps two backends
2. Add cross-validation results to `RawProcessResult`
3. Add CLI reporting for cross-validation discrepancies

### 8.4 Phase 4: DCGM and Additional Backends

**Changes:**
1. Add DCGM backend
2. Add direct RAPL backend (for Zeus-unavailable environments)
3. Consolidate PowerThermalSampler with energy backends (reduce NVML duplication)

### 8.5 Dependency Management

```toml
# pyproject.toml additions

[tool.poetry.dependencies]
codecarbon = ">=2.8.0"  # existing, remains required

[tool.poetry.extras]
# Zeus energy measurement (recommended for accuracy)
zeus = ["zeus"]

# All energy backends
energy-all = ["zeus"]
```

Zeus should be an optional dependency because:
- It has platform-specific requirements (NVML, RAPL permissions)
- It pulls in additional dependencies (`sklearn` for power integration on
  pre-Volta GPUs)
- CodeCarbon is sufficient for many use cases

---

## 9. Cross-Validation Mode

### 9.1 Rationale

Running two energy measurement backends simultaneously enables:
- Validating measurement accuracy
- Quantifying CodeCarbon estimation error
- Building confidence in measurements for publications
- Identifying systematic biases

### 9.2 Design

```python
class CrossValidationBackend:
    """Runs two energy backends simultaneously for cross-validation."""

    def __init__(self, primary: EnergyBackend, secondary: EnergyBackend) -> None:
        self._primary = primary
        self._secondary = secondary

    @property
    def name(self) -> str:
        return f"crossval({self._primary.name}+{self._secondary.name})"

    def start_tracking(self) -> tuple[Any, Any]:
        primary_tracker = self._primary.start_tracking()
        secondary_tracker = self._secondary.start_tracking()
        return (primary_tracker, secondary_tracker)

    def stop_tracking(self, tracker: tuple[Any, Any]) -> EnergyMetrics:
        primary_tracker, secondary_tracker = tracker

        primary_metrics = self._primary.stop_tracking(primary_tracker)
        secondary_metrics = self._secondary.stop_tracking(secondary_tracker)

        # Log discrepancy
        if primary_metrics.total_energy_j > 0 and secondary_metrics.total_energy_j > 0:
            ratio = secondary_metrics.total_energy_j / primary_metrics.total_energy_j
            pct_diff = abs(1.0 - ratio) * 100
            logger.info(
                f"Energy cross-validation: {self._primary.name}={primary_metrics.total_energy_j:.2f}J, "
                f"{self._secondary.name}={secondary_metrics.total_energy_j:.2f}J "
                f"(diff={pct_diff:.1f}%)"
            )

        # Return primary metrics (secondary stored for analysis)
        # Could attach secondary as metadata
        return primary_metrics
```

### 9.3 Considerations

- **NVML conflicts**: Both Zeus and CodeCarbon may try to use NVML. Zeus uses
  `nvmlDeviceGetTotalEnergyConsumption` while CodeCarbon uses
  `nvmlDeviceGetPowerUsage`. Both should work concurrently since NVML supports
  multiple sessions, but this needs testing.
- **Overhead**: Running two backends doubles the measurement overhead. For
  CodeCarbon (polling), this means an extra background thread.
- **Timing**: Start/stop order matters. Starting both before inference and
  stopping both after ensures the same time window.

---

## 10. Open Questions

### 10.1 Schema Decisions (Needs PM Input)

1. **`ram_energy_j` vs `dram_energy_j`**: Should we keep both (CodeCarbon
   estimation vs Zeus RAPL) or consolidate? Recommendation: consolidate into
   `ram_energy_j` with a `measurement_method` field to indicate source.

2. **Per-GPU energy in `EnergyMetrics`**: Should `gpu_energy_per_device_j` be
   added to the existing model or stored separately? The late aggregation pattern
   suggests it should be in `RawProcessResult` for per-process data.

3. **Carbon emissions with Zeus**: Zeus does not calculate CO2. Options:
   - (a) Use CodeCarbon solely for CO2 even when Zeus does energy measurement
   - (b) Implement our own CO2 calculation using electricity maps API
   - (c) Make CO2 an optional enrichment step post-measurement
   Recommendation: (c) -- decouple CO2 from energy measurement.

### 10.2 Technical Decisions (Needs Testing)

4. **NVML session management**: Can Zeus's NVML, our PowerThermalSampler's
   NVML, and CodeCarbon's NVML all coexist? Needs empirical testing.

5. **Zeus + vLLM**: vLLM manages its own CUDA context. Zeus calls
   `torch.cuda.synchronize()` at window boundaries. Does this conflict with
   vLLM's async execution model? Our existing CodeCarbon already has this
   issue (L242-255 of runner.py shows energy tracking can fail with vLLM).

6. **Baseline measurement + Zeus**: Our baseline measurement uses `pynvml`
   directly. Zeus also uses `pynvml` (via its device layer). Should baseline
   measurement be integrated into the energy backend protocol?

### 10.3 Architectural Decisions

7. **PowerThermalSampler consolidation**: The PowerThermalSampler does NVML-based
   power/temperature sampling that overlaps with what Zeus provides. Should we:
   - (a) Keep them independent (current approach)
   - (b) Have the energy backend optionally provide time-series data
   - (c) Consolidate the sampler into the Zeus backend
   Recommendation: (a) for now, (b) as a future enhancement.

8. **Plugin discovery**: Should we support third-party energy backends via
   entry points (`importlib.metadata.entry_points`)? This would allow pip-
   installable custom backends. Low priority but architecturally clean.

---

## Summary of Recommended Changes

### Immediate (Phase 1, non-breaking)

| Change | File | Effort |
|--------|------|--------|
| Create `ZeusBackend` class | `core/energy_backends/zeus.py` | Medium |
| Create `NullEnergyBackend` class | `core/energy_backends/null.py` | Small |
| Add `EnergyConfig` to config | `config/models.py` | Small |
| Update registry to register Zeus | `core/energy_backends/__init__.py` | Small |
| Update factory to use config | `orchestration/factory.py` | Small |
| Add `zeus` optional dependency | `pyproject.toml` | Small |
| Add tests for Zeus backend | `tests/unit/test_core_energy_backends.py` | Medium |
| Update protocol with `get_info()` | `protocols.py` | Small |

### Near-term (Phase 2)

| Change | File | Effort |
|--------|------|--------|
| Switch default to `auto` | `config/models.py` | Small |
| Add per-device GPU energy to `EnergyMetrics` | `domain/metrics.py` | Small |
| Add `EnergyCapability` flag enum | `protocols.py` | Small |
| Integration tests with real hardware | `tests/integration/` | Medium |

### Future (Phase 3+)

| Change | File | Effort |
|--------|------|--------|
| Cross-validation backend | `core/energy_backends/crossval.py` | Medium |
| DCGM backend | `core/energy_backends/dcgm.py` | Medium |
| Decouple CO2 calculation | `core/carbon.py` (new) | Medium |
| Entry-point plugin discovery | `core/energy_backends/__init__.py` | Small |
