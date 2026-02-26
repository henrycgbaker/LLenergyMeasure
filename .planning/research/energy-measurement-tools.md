# Energy Measurement Tools

**Sources**: `03-codecarbon-zeus-energy.md`, `05-zeus-deep-dive.md`,
`06-llenergymeasure-vs-zeus.md`, `08-energy-plugin-architecture.md`

## Accuracy Hierarchy

| Method | Accuracy | Hardware | Notes |
|--------|----------|----------|-------|
| SPEC PTD (PDU) | ~1% | Any | Gold standard, requires hardware |
| DCGM (NVIDIA) | ~2% | NVIDIA only | Best software-only for NVIDIA |
| Zeus / NVML direct | ~5% | NVIDIA + AMD + Apple Silicon | Reads total energy counters |
| RAPL (CPU) | ~5% | CPU only | Good for CPU-dominant workloads |
| CodeCarbon | ~10-15% | Any | Polls power and integrates; hardware estimation fallback |
| Parameter estimation | ~20-30% | Any | TDP-based, worst case |

**Key distinction**: Zeus reads `nvmlDeviceGetTotalEnergyConsumption()` (total joules since
reset) — a hardware counter. CodeCarbon polls `nvmlDeviceGetPowerUsage()` at intervals and
integrates — sampling error accumulates.

## Our Existing Infrastructure

`EnergyBackend` Protocol + registry already exists in `src/llenergymeasure/core/`:

```python
class EnergyBackend(Protocol):
    @property
    def name(self) -> str: ...
    def start_tracking(self) -> Any: ...
    def stop_tracking(self, tracker: Any) -> EnergyMetrics: ...
    def is_available(self) -> bool: ...
```

Currently: one implementation (CodeCarbon). Factory hardcodes `get_energy_backend("codecarbon")`.
Adding Zeus is an extension of existing infrastructure, not a redesign.

## Zeus ZeusMonitor API

```python
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0])        # target GPU indices
monitor.begin_window("inference")
# ... run inference ...
measurement = monitor.end_window("inference") # returns Measurement dataclass

measurement.total_energy   # float, Joules
measurement.gpu_energy     # dict[int, float] — per-GPU breakdown
measurement.time           # float, seconds
```

Supports: Volta+ NVIDIA (energy counters), older NVIDIA (power integration), AMD ROCm,
Apple Silicon (Energy Impact).

## NVML Conflict Risk

Our `PowerThermalSampler` and `GPUUtilisationSampler` also use `pynvml` directly. Multiple
NVML sessions are supported by the driver, but concurrent access patterns need testing.

See `08-energy-plugin-architecture.md` for full Protocol changes, ZeusBackend implementation
design, and 4-phase migration path including cross-validation (run both backends simultaneously
to verify accuracy).

## CodeCarbon Modes

CodeCarbon supports 5 integration modes: decorator, context manager, explicit tracker,
background thread, CLI. We use the context manager mode. Migrating to Zeus does not
require removing CodeCarbon — both can coexist as separate extras.
