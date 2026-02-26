# Research: Zeus Deep-Dive (Codebase Architecture)

**Date:** 2026-02-17
**Source:** Research agent transcript (agent-a2b9295.jsonl, 159KB)
**Status:** Agent stopped before final synthesis; findings compiled from web research and GitHub code analysis

---

## Summary

Deep architectural analysis of the Zeus energy measurement library (ML.ENERGY Initiative, University of Michigan). Zeus provides hardware-level energy measurement with <10ms overhead, supporting NVIDIA (NVML), AMD (AMDSMI), Intel/AMD CPUs (RAPL), Apple Silicon, and NVIDIA Jetson. It is the measurement backend for the ML.ENERGY Benchmark/Leaderboard (NeurIPS D&B 2025 spotlight).

---

## 1. Project Metadata

| Field | Value |
|-------|-------|
| Name | `zeus` |
| Version | v0.13.1 (November 2025) |
| Description | Framework for deep learning energy measurement and optimization |
| Licence | Apache-2.0 |
| Python | >= 3.9 (supports 3.9, 3.10, 3.11, 3.12) |
| Repository | https://github.com/ml-energy/zeus |
| Docs | https://ml.energy/zeus |
| GitHub stars | ~332 |
| Forks | ~40 |
| Primary maintainer | Jae-Won Chung (jaywonchung: 338 of 386 commits) |
| Ecosystem | PyTorch ecosystem project, 2024 Mozilla Technology Fund awardee |
| Last updated | 5 Feb 2026 (README.md update) |

---

## 2. Repository Structure

```
zeus/
  monitor/          # Energy & power measurement (ZeusMonitor, PowerMonitor)
  optimizer/        # Energy optimization algorithms (power limit, batch size, Perseus)
  device/           # CPU/GPU/SoC device abstraction (NVIDIA, AMD)
  callback.py       # PyTorch training callback
  metric.py         # Prometheus metrics support
  utils/            # Logging, async, framework helpers
zeusd/              # Zeus daemon (background service for CPU/DRAM via RAPL)
docker/             # Docker images
examples/           # Usage examples (HuggingFace, etc.)
```

---

## 3. Core API: ZeusMonitor

### 3.1 Constructor

```python
def __init__(
    self,
    gpu_indices: list[int] | None = None,
    cpu_indices: list[int] | None = None,
    approx_instant_energy: bool = False,
    log_file: str | Path | None = None,
    sync_execution_with: Literal["torch", "jax", "cupy"] = "torch",
) -> None:
```

Parameters:
- `gpu_indices`: Which GPUs to monitor (defaults to all visible)
- `cpu_indices`: Which CPUs to monitor
- `approx_instant_energy`: For short windows, approximate energy via instantaneous power x time
- `log_file`: CSV output path
- `sync_execution_with`: Framework for CPU-GPU synchronisation (`torch.cuda.synchronize()`, etc.)

### 3.2 Measurement Windows

```python
monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

monitor.begin_window("inference")
# ... workload ...
measurement = monitor.end_window("inference")
```

**`begin_window(key, sync_execution=True)`:**
- Creates a named measurement window
- Optionally synchronises GPU execution before recording baseline energy
- Captures initial timestamps and energy counters for all monitored GPUs/CPUs
- Stores state in `measurement_states` dictionary

**`end_window(key, sync_execution=True, cancel=False)`:**
- Retrieves baseline measurements
- Optionally synchronises GPU execution
- Queries current energy readings and computes deltas from baseline
- Handles legacy GPU architectures using background power monitoring
- Returns a `Measurement` object
- Logs results to optional CSV file

Windows can be **named, overlapping, and nested** with different keys.

### 3.3 Measurement Dataclass

```python
@dataclass
class Measurement:
    time: float                              # Elapsed seconds
    gpu_energy: dict[int, float]             # GPU index -> Joules
    cpu_energy: dict[int, float] | None      # CPU index -> Joules
    dram_energy: dict[int, float] | None     # DRAM -> Joules
    soc_energy: SoCMeasurement | None        # Apple Silicon / Jetson

    @cached_property
    def total_energy(self) -> float:
        """Total energy consumed (in Joules) during the measurement window."""
        return sum(self.gpu_energy.values())
```

### 3.4 MeasurementState Dataclass

Tracks initial state when a window begins: baseline timestamps and energy readings to enable delta calculations at window closure.

### 3.5 CSV Logging

Header: `start_time, window_name, elapsed_time, gpu{i}_energy` for each monitored GPU. Data flushes after each write.

---

## 4. Energy Measurement Strategy

Two approaches based on GPU generation:

| GPU Generation | Method | Mechanism |
|----------------|--------|-----------|
| Volta+ (modern) | Direct query | `nvmlDeviceGetTotalEnergyConsumption` API |
| Pre-Volta (legacy) | Background polling | `PowerMonitor` polls `nvmlDeviceGetPowerUsage`, integrates samples |

The `approx_instant_energy` option approximates energy for very short windows by multiplying instantaneous power by elapsed time.

---

## 5. PowerMonitor

Background power monitoring for timeline analysis:

```python
from zeus.monitor import PowerMonitor

monitor = PowerMonitor(gpu_indices=[0])
timelines = monitor.get_all_power_timelines()
```

### Architecture

- **Multiprocessing:** Spawns separate polling processes for each supported power domain
- **Deduplication:** Only records power samples when values change
- **Queue-based communication:** Each domain has its own queue for inter-process data transfer
- **Automatic initialisation:** Infers optimal polling frequency by detecting GPU power counter update rates

### Three Power Domains

| Domain | Description | Availability |
|--------|-------------|-------------|
| Device Instant | Real-time GPU power draw | All NVIDIA GPUs |
| Device Average | Historical average GPU power | Volta+ |
| Memory Average | Historical average memory power | Hopper+ |

### API Methods

| Method | Purpose |
|--------|---------|
| `get_power_timeline()` | Time-series power data for a domain with filtering |
| `get_all_power_timelines()` | All monitored domains' timelines |
| `get_energy()` | Total energy (joules) between timestamps |
| `get_power()` | Instantaneous power at a specific time or latest |
| `stop()` | Terminate monitoring processes |

### Configuration

Constructor: `gpu_indices`, `update_period`, `max_samples_per_gpu`, `power_domains`.

---

## 6. CarbonEmissionMonitor

Bridges energy measurement with real-time carbon intensity data:

- Tracks elapsed time, GPU/CPU/DRAM energy, carbon emissions (gCO2eq)
- Background subprocess continuously polls energy and carbon intensity
- Supports multiple concurrent measurement windows with unique identifiers
- Framework integration: PyTorch, JAX, CuPy synchronisation

---

## 7. GPU Abstraction Layer

```python
from zeus.device import get_gpus
gpus = get_gpus()  # Returns NVIDIAGPUs or AMDGPUs
```

### Supported Platforms

| Platform | Implementation | Requirements |
|----------|---------------|-------------|
| NVIDIA GPUs | `NVIDIAGPUs` via `pynvml` | Any NVIDIA GPU |
| AMD GPUs | `AMDGPUs` via `amdsmi` | ROCm 6.1+ only |

### API Pattern

Methods map directly to GPU management operations:
```python
constraints = gpus.getPowerManagementLimitConstraints(gpu_index)
```

Non-blocking setter support via `GPU.supports_nonblocking_setters`. Currently only Zeus daemon-based NVIDIA implementation supports this.

### Error Handling

19+ specific exception types:
- `ZeusGPUInitError` -- initialisation failures
- `ZeusGPUNoPermissionError` -- access control
- Various hardware, driver, and API-specific exceptions

---

## 8. Prometheus Metrics Integration

Three metric types exported to Prometheus:

### EnergyHistogram

Records energy distribution for repeated code ranges:
```python
from zeus.metric import EnergyHistogram

histogram = EnergyHistogram(
    gpu_indices=[0],
    cpu_indices=[0],
    prometheus_url='http://localhost:9091',
    job='training_energy_histogram'
)
```

### EnergyCumulativeCounter

Tracks cumulative energy over time with periodic Prometheus pushes.

### PowerGauge

Monitors real-time GPU power consumption.

### Metric Naming Convention

- `energy_monitor_{component}_energy_joules` (histogram/counter)
- `power_monitor_gpu_power_watts` (gauge)
- Labels: `window` (user-defined), `index` (device index)

---

## 9. Optimisers

### Power Limit Optimizer
Finds optimal GPU power cap balancing performance vs. energy.

### Batch Size Optimizer
Optimises batch size for energy efficiency.

### Perseus (SOSP '24)
Pipeline frequency optimiser for large model training. Reduces energy with negligible throughput loss. FastAPI server for coordination.

Optional dependencies for optimiser servers:
- `zeus[pfo-server]`: FastAPI, lowtime, aiofiles, torch
- `zeus[bso-server]`: FastAPI, SQLAlchemy, python-dotenv
- `zeus[migration]`: Alembic, SQLAlchemy

---

## 10. Zeus and ML.ENERGY Relationship

Zeus is the measurement library; ML.ENERGY is the initiative and benchmark:

| Component | Purpose |
|-----------|---------|
| Zeus | Open-source measurement and optimisation library |
| ML.ENERGY Benchmark | Suite and tool for measuring inference energy under realistic conditions |
| ML.ENERGY Leaderboard | Public benchmark results display |
| Chase | Carbon-Aware Zeus (ICLR workshop paper) |

### ML.ENERGY Benchmark Details

- Uses Zeus for energy measurement + vLLM for LLM inference + Diffusers for diffusion models
- Hardware: NVIDIA H100 (80GB), B200 on AWS
- Scope (v3.0): 46 models, 7 tasks, 1,858 configurations
- NeurIPS Datasets & Benchmarks 2025 spotlight

**Measurement methodology:**
1. Select model and request dataset
2. Execute configurations independently, measuring time/energy via Zeus
3. Construct time-energy Pareto frontier
4. Recommend energy-optimal configurations per latency constraints

**Two energy accounting methods:**
- Diffusion models: Energy divided equally across batch requests
- LLM text generation: Energy per token during steady-state multiplied by average output tokens

**Derived metrics:** Average power draw (W), monetary cost ($), operational carbon emissions (gCO2e)

### Zeus vs CodeCarbon (Key Distinction)

Zeus provides **direct measurement** (objective, consistent regardless of location/time). CodeCarbon provides **estimation** (carbon emission based on regional intensity). Zeus yields more precise energy values; CodeCarbon provides broader scope with carbon conversion.

Zeus is backend-agnostic for energy measurement -- it measures GPU/CPU energy regardless of what software is running. The ML.ENERGY Benchmark pairs it with vLLM, but Zeus itself has no inference backend dependency.

---

## 11. Dependencies

**Core:** numpy, pandas, scikit-learn, nvidia-ml-py, pydantic, rich, tyro, httpx, amdsmi, python-dateutil

**Optional extras:**
| Extra | Dependencies |
|-------|-------------|
| `prometheus` | prometheus-client |
| `apple` | zeus-apple-silicon |
| `pfo` / `pfo-server` | Pydantic <2, FastAPI, lowtime |
| `bso` / `bso-server` | Pydantic <2, FastAPI, SQLAlchemy |
| `migration` | Alembic, SQLAlchemy |
| `lint` | ruff, pyright, pandas-stubs |
| `test` | pytest, pytest-mock, pytest-xdist, anyio |
| `docs` | mkdocs-material, mkdocstrings |

Build system: setuptools (>=61.0.0) with dynamic versioning.

---

## 12. Docker Support

```bash
docker run -it --gpus all --cap-add SYS_ADMIN --ipc host \
  -v /sys/class/powercap/intel-rapl:/zeus_sys/class/powercap/intel-rapl \
  mlenergy/zeus:latest bash
```

Image: `nvidia/cuda:11.8.0-base-ubuntu22.04` with Miniconda 3, PyTorch, Torchvision. Tags: `latest`, `v*`, `master`.

Requirements:
- `--cap-add SYS_ADMIN` for GPU power management
- RAPL volume mount for CPU/DRAM energy
- NVIDIA Container Toolkit

Verification: `python -m zeus.show_env`

---

## 13. Recent Development Activity

Based on Git log (most recent commits):
- 2026-02-05: README.md update, newsletter embeds
- 2026-01-13: Global variable warning for monitors
- 2025-11-25: PowerMonitor failure handling improvement
- 2025-11-22: Troubleshooting docs (multiprocessing pitfall)
- 2025-11-17: Skip unnecessary API calls, AMDSMI unit fixes
- 2025-11-13: v0.13.1 release, AMD GPU hardening, power management limit implementation

---

## 14. What Zeus Does NOT Measure

Zeus focuses exclusively on energy and power. It does not provide:
- Throughput metrics
- FLOPs estimation
- Latency measurement
- Model accuracy evaluation
- Training performance tracking

These are explicitly left to downstream tools (like LLenergyMeasure or ML.ENERGY Benchmark harness).

---

## Sources

- [Zeus Project](https://ml.energy/zeus/)
- [Zeus GitHub](https://github.com/ml-energy/zeus)
- [Measuring Energy (Zeus docs)](https://ml.energy/zeus/measure/)
- [Zeus on PyTorch Blog](https://pytorch.org/blog/zeus/)
- [Zeus energy module reference](https://ml.energy/zeus/reference/monitor/energy/)
- [GPU Energy Best Practices (ML.ENERGY Blog)](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)
- [ML.ENERGY Benchmark Paper](https://arxiv.org/html/2505.06371v1)
- [ML.ENERGY Leaderboard v3.0 Blog](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/)
- [ML.ENERGY Leaderboard README](https://github.com/ml-energy/leaderboard/blob/master/README.md)
- [zeus-ml on PyPI](https://pypi.org/project/zeus-ml/0.5.0/)
- [Zeus Getting Started](https://ml.energy/zeus/getting_started/)
