# ML.ENERGY Ecosystem Research

**Date**: 2026-02-17
**Author**: Research Scientist Agent
**Purpose**: Comprehensive analysis of the ML.ENERGY ecosystem from Carnegie Mellon University for strategic comparison with LLenergyMeasure.

---

## Table of Contents

1. [Ecosystem Overview](#1-ecosystem-overview)
2. [Zeus Architecture Deep-Dive](#2-zeus-architecture-deep-dive)
3. [ML.ENERGY Benchmark](#3-mlenergy-benchmark)
4. [ML.ENERGY Leaderboard](#4-mlenergy-leaderboard)
5. [Feature Comparison](#5-feature-comparison)
6. [Architecture Best Practices](#6-architecture-best-practices)
7. [Integration Possibilities](#7-integration-possibilities)
8. [Differentiation Strategy](#8-differentiation-strategy)

---

## 1. Ecosystem Overview

### 1.1 Organisation and Academic Context

ML.ENERGY is a research group at Carnegie Mellon University, led by Jae-Won Chung and Mosharaf Chowdhury (University of Michigan). They focus on energy measurement and optimisation for deep learning workloads. Their benchmark paper received a **NeurIPS 2025 Datasets & Benchmarks Spotlight**.

### 1.2 Repository Map

The ml-energy GitHub organisation has 14 repositories:

| Repository | Stars | Language | Purpose | Status |
|------------|-------|----------|---------|--------|
| **zeus** | 332 | Python/Rust | Energy measurement & optimisation library | Active, v0.13.1 |
| **benchmark** | 3 | Python | Benchmark suite for GenAI energy | Active (v3.0) |
| **leaderboard** | 4 | TypeScript | Interactive leaderboard web app (v3) | Active |
| **leaderboard-v2** | 50 | Python | Previous leaderboard (Gradio/HF Spaces) | Superseded |
| **ml-energy.github.io** | 1 | CSS | Homepage | Active |
| **blog** | 1 | Jupyter | Research & tech blog | Active |
| **lowtime** | 11 | Python | Time-cost tradeoff solver | Active |
| **zeus-apple-silicon** | 6 | C++ | Apple Silicon device support for Zeus | Active |
| **amdsmi** | 3 | Python | Python bindings for AMD SMI | Active |
| **text_generation_energy** | 1 | Python | Fork of text_generation with energy info | Archived |
| **merak-zeus** | 3 | Python | Merak integration with Zeus | Low activity |
| **gpu_power_monitor_tui** | 3 | Python | Terminal GPU power visualiser | Low activity |
| **data** | 0 | Python | Dataset utilities | Active |
| **tutorials** | 2 | HTML | Tutorial homepages | Active |

### 1.3 Ecosystem Architecture

```
+------------------------------------------------------------------+
|                     ML.ENERGY Ecosystem                          |
|                                                                  |
|  +------------------+    +-----------------+    +--------------+ |
|  |      Zeus        |    |    Benchmark    |    | Leaderboard  | |
|  | (Library, v0.13) |    |   (Suite, v3)   |    |  (Web, v3)   | |
|  |                  |    |                 |    |              | |
|  | - ZeusMonitor    |--->| - Job generator |--->| - React/TS   | |
|  | - PowerMonitor   |    | - vLLM runner   |    | - JSON data  | |
|  | - Optimisers     |    | - xDiT runner   |    | - Charts     | |
|  | - Device HAL     |    | - Prometheus    |    | - Filters    | |
|  | - zeusd daemon   |    | - Analysis      |    |              | |
|  +------------------+    +-----------------+    +--------------+ |
|           |                       |                    ^         |
|           v                       v                    |         |
|  +------------------+    +-----------------+           |         |
|  | Supporting Libs  |    | Data Pipeline   |-----------+         |
|  | - amdsmi         |    | build_data.py   |                    |
|  | - zeus-apple-si  |    +-----------------+                    |
|  | - lowtime        |                                           |
|  +------------------+                                           |
+------------------------------------------------------------------+
```

### 1.4 Key Differences from LLenergyMeasure at a Glance

- **Zeus** is a general-purpose energy measurement **library** (not LLM-specific)
- **Benchmark** is an LLM/diffusion energy benchmark **suite** (server-based, vLLM-only)
- **Leaderboard** is a **centrally run** leaderboard (CMU runs all benchmarks)
- Zeus focuses heavily on **optimisation** (power limit, batch size, frequency), not just measurement
- Zeus supports **training** workloads (callbacks for training loops), not just inference
- The benchmark uses **server-mode** inference (vLLM behind OpenAI API), not direct model loading

---

## 2. Zeus Architecture Deep-Dive

### 2.1 Package Structure

```
zeus/
  __init__.py              # v0.13.1
  callback.py              # Training callback base class (86 lines)
  exception.py             # Base exception
  metric.py                # Prometheus integration (526 lines)
  show_env.py              # CLI env verification

  device/                  # Hardware Abstraction Layer
    __init__.py             # get_gpus(), get_cpus(), get_soc()
    common.py               # has_sys_admin(), deprecated_alias()
    exception.py            # Base GPU error
    gpu/
      __init__.py           # get_gpus() singleton with vendor detection
      common.py             # GPU and GPUs ABCs (617 lines)
      nvidia.py             # NVML implementation (459 lines)
      amd.py                # AMD SMI implementation
    cpu/
      __init__.py
      common.py             # CPU ABC
      rapl.py               # Intel RAPL implementation
    soc/
      __init__.py
      common.py             # SoC ABC
      apple.py              # Apple Silicon (via zeus-apple-silicon)
      jetson.py             # NVIDIA Jetson

  monitor/                 # Measurement Tools
    __init__.py             # Exports ZeusMonitor, PowerMonitor, TemperatureMonitor
    energy.py               # ZeusMonitor - main energy measurement (468 lines)
    power.py                # PowerMonitor - power time-series (639 lines)
    carbon.py               # CarbonIntensityMonitor
    price.py                # ElectricityPriceMonitor
    temperature.py          # TemperatureMonitor

  optimizer/               # Energy Optimisers
    __init__.py
    power_limit.py          # GlobalPowerLimitOptimizer (551 lines)
    batch_size/             # BatchSizeOptimizer (client-server)
      client.py
      server/               # FastAPI server with SQLAlchemy
        optimizer.py
        router.py
        database/
        ...
    pipeline_frequency/     # PipelineFrequencyOptimizer
      optimizer.py
      frequency_controller.py
      server/
        scheduler.py
        ...

  utils/                   # Utilities
    async_utils.py
    env.py
    framework.py            # sync_execution for torch/jax/cupy
    lat_lon.py
    logging.py
    lr_scaler.py
    metric.py               # zeus_cost() function
    multiprocessing.py
    pydantic_v1.py          # Pydantic v1/v2 compat layer
    testing.py

zeusd/                     # Rust Daemon for privileged operations
  src/
    main.rs
    config.rs
    startup.rs
    routes/
      gpu.rs
      cpu.rs
    devices/
      gpu/
      cpu/
```

**Total codebase size**: Approximately 8,000-10,000 lines of Python + 2,000 lines of Rust.

### 2.2 Device Abstraction Layer (HAL)

Zeus has a clean, vendor-agnostic hardware abstraction. The key insight is a **singleton pattern** with **auto-detection**.

```python
# zeus/device/gpu/__init__.py
_gpus: GPUs | None = None

def get_gpus(ensure_homogeneous: bool = False) -> GPUs:
    global _gpus
    if _gpus is not None:
        return _gpus
    if nvml_is_available():
        _gpus = NVIDIAGPUs(ensure_homogeneous)
    elif amdsmi_is_available():
        _gpus = AMDGPUs(ensure_homogeneous)
    else:
        raise ZeusGPUInitError(...)
    return _gpus
```

**GPU ABC** (`zeus/device/gpu/common.py`) defines 18 abstract methods:

| Method | Category | Units |
|--------|----------|-------|
| `get_name()` | Identity | string |
| `get_power_management_limit_constraints()` | Power | mW |
| `get_power_management_limit()` | Power | mW |
| `set_power_management_limit()` | Power | mW |
| `reset_power_management_limit()` | Power | - |
| `set_persistence_mode()` | Config | bool |
| `get_supported_memory_clocks()` | Frequency | MHz |
| `set_memory_locked_clocks()` | Frequency | MHz |
| `reset_memory_locked_clocks()` | Frequency | - |
| `get_supported_graphics_clocks()` | Frequency | MHz |
| `set_gpu_locked_clocks()` | Frequency | MHz |
| `reset_gpu_locked_clocks()` | Frequency | - |
| `get_average_power_usage()` | Power | mW |
| `get_instant_power_usage()` | Power | mW |
| `get_average_memory_power_usage()` | Power | mW |
| `supports_get_total_energy_consumption()` | Energy | bool |
| `get_total_energy_consumption()` | Energy | mJ |
| `get_gpu_temperature()` | Thermal | Celsius |

**Vendor implementations**:
- **NVIDIA** (`nvidia.py`): Uses `pynvml` (official `nvidia-ml-py`). Handles Volta+ energy counters, Grace Hopper chips, and NVML field value queries.
- **AMD** (`amd.py`): Uses `amdsmi` with their own Python bindings repo.
- **Apple Silicon** (`soc/apple.py`): Via separate `zeus-apple-silicon` C++ package.
- **NVIDIA Jetson** (`soc/jetson.py`): Reads power from sysfs.

**Key design choice**: `GPUs` (plural) wraps a list of `GPU` (singular), forwarding calls. The `EmptyGPUs` class acts as a null object pattern.

### 2.3 ZeusMonitor - The Core Energy API

The `ZeusMonitor` (`zeus/monitor/energy.py`, 468 lines) is the primary user-facing class.

**API Pattern - Named Measurement Windows**:

```python
from zeus.monitor import ZeusMonitor

monitor = ZeusMonitor(gpu_indices=[0, 1])

monitor.begin_window("forward_pass")
# ... do work ...
result = monitor.end_window("forward_pass")

# result.time         -> float (seconds)
# result.gpu_energy   -> dict[int, float] (Joules per GPU)
# result.cpu_energy   -> dict[int, float] | None (Joules per CPU)
# result.dram_energy  -> dict[int, float] | None (Joules per DRAM zone)
# result.soc_energy   -> SoCMeasurement | None
# result.total_energy -> float (sum of GPU Joules)
```

**Energy measurement strategy (NVIDIA)**:

1. **Volta+ GPUs**: Uses `nvmlDeviceGetTotalEnergyConsumption()` - a cumulative energy counter in millijoules. This is a **hardware counter**, very accurate, very cheap to read.
2. **Pre-Volta GPUs**: Falls back to the `PowerMonitor` which polls `nvmlDeviceGetPowerUsage()` in a background process and integrates power over time using `sklearn.metrics.auc()`.
3. **Short windows**: If energy reads as zero (counter update period not elapsed), the `approx_instant_energy` option multiplies instantaneous power by time.

**Critical implementation details**:
- `sync_execution` calls `torch.cuda.synchronize()` (or JAX/CuPy equivalent) before begin/end
- Supports concurrent named windows
- CUDA_VISIBLE_DEVICES-aware indexing
- CPU energy via Intel RAPL (requires root or `/sys/class/powercap` permissions)
- Logs to CSV if `log_file` is specified

**Comparison with LLenergyMeasure's approach (CodeCarbon)**:

| Aspect | Zeus | LLenergyMeasure (CodeCarbon) |
|--------|------|------------------------------|
| GPU energy | NVML direct (mJ counter) | NVML via CodeCarbon |
| CPU energy | RAPL direct | RAPL via CodeCarbon |
| Precision | Hardware counter on Volta+ | Power polling + integration |
| Overhead | Minimal (counter read) | Higher (polling thread) |
| CO2 tracking | Separate module | Integrated |
| Multi-GPU | Native (per-GPU energy) | Via CodeCarbon |
| AMD support | Yes (amdsmi) | No |
| Apple Silicon | Yes | No |

### 2.4 PowerMonitor - Time-Series Power

The `PowerMonitor` (`zeus/monitor/power.py`, 639 lines) provides real-time power monitoring with multiple domains.

**Architecture**:
- Spawns **separate processes per power domain** using `multiprocessing.get_context("spawn")`
- Three power domains: `DEVICE_INSTANT`, `DEVICE_AVERAGE`, `MEMORY_AVERAGE`
- Each process polls at an inferred or specified update period
- Data flows via `multiprocessing.Queue` (one per domain)
- **Deduplication**: Only sends samples when power changes (saves queue bandwidth)
- **Cleanup**: Uses `weakref.finalize()` for graceful process shutdown

**Key APIs**:
```python
monitor = PowerMonitor(gpu_indices=[0], update_period=0.1)

# Get power timeline for a time range
timeline = monitor.get_power_timeline(
    power_domain=PowerDomain.DEVICE_INSTANT,
    gpu_index=0,
    start_time=t0,
    end_time=t1,
)
# Returns: dict[int, list[tuple[float, float]]]
#          gpu_idx -> [(timestamp, power_watts), ...]

# Get energy (integrates power over time using sklearn AUC)
energy = monitor.get_energy(start_time=t0, end_time=t1)
# Returns: dict[int, float] (Joules per GPU)

# Get all timelines across all domains
all_timelines = monitor.get_all_power_timelines()
```

**Counter update period inference** (`infer_counter_update_period()`):
- Polls GPU 1000 times rapidly
- Finds timestamps where power readings changed
- Computes minimum interval between changes
- Uses half that interval as the polling period
- Caps at 0.1s if too slow

### 2.5 Energy Optimisers

Zeus is unique in providing **active energy optimisation**, not just measurement.

#### 2.5.1 GlobalPowerLimitOptimizer

(`zeus/optimizer/power_limit.py`, 551 lines)

A **state machine** that profiles different GPU power limits during training and selects the optimal one.

```
State Machine:
  Ready -> Warmup -> Profiling -> [more power limits? -> Warmup] -> Done
              |           |
              +-- (epoch end, restart) --+
```

**Optimum selection strategies** (pluggable via `OptimumSelector` ABC):
- `Energy`: Minimise energy
- `Time`: Minimise training time
- `ZeusCost`: Minimise eta * Energy + (1-eta) * MaxPower * Time
- `MaxSlowdownConstraint`: Lowest power limit within X slowdown factor

**Integration with HuggingFace**: `HFGlobalPowerLimitOptimizer` wraps as a `TrainerCallback`.

#### 2.5.2 BatchSizeOptimizer

A client-server architecture:
- **Server**: FastAPI with SQLAlchemy, uses Multi-Armed Bandit (MAB) exploration
- **Client**: Sends training cost (time, energy) per batch size
- Implements Thompson Sampling for exploration/exploitation
- Designed for recurring/periodic training jobs

#### 2.5.3 PipelineFrequencyOptimizer (Perseus)

For pipeline-parallel training:
- Profiles computation at different GPU frequencies
- Different pipeline stages can have different optimal frequencies
- Uses a server to coordinate across pipeline stages

### 2.6 Additional Monitors

#### CarbonIntensityMonitor
- Queries real-time carbon intensity from Electricity Maps API
- Uses latitude/longitude of the host
- Can also read from local files (for batch jobs)

#### ElectricityPriceMonitor
- Queries electricity prices from various utility APIs
- Currently supports some US regions

#### TemperatureMonitor
- Polls GPU temperature over time
- Same multiprocessing architecture as PowerMonitor

### 2.7 Prometheus Integration (`metric.py`, 526 lines)

Zeus provides first-class Prometheus metric types:

- **EnergyHistogram**: Energy per window as Prometheus Histogram
- **EnergyCumulativeCounter**: Cumulative energy as Prometheus Counter
- **PowerGauge**: Real-time power as Prometheus Gauge

All push to a Prometheus Pushgateway. This enables **production monitoring** of energy consumption.

### 2.8 The Zeus Daemon (`zeusd`)

Written in **Rust** (Actix Web framework), `zeusd` solves the privilege problem:

- GPU power limit and frequency changes require `SYS_ADMIN`
- `zeusd` runs as root, exposes a Unix domain socket
- Zeus Python library auto-detects `ZEUSD_SOCK_PATH` and uses `ZeusdNVIDIAGPU` (which calls zeusd via HTTP over UDS)
- Supports both UDS and TCP modes
- Non-blocking setter support (returns immediately without waiting for confirmation)

This is a notably elegant solution to a common pain point.

---

## 3. ML.ENERGY Benchmark

### 3.1 Overview

The ML.ENERGY Benchmark (v3.0, NeurIPS 2025 D&B Spotlight) is a **comprehensive inference energy benchmark suite** for generative AI.

**Paper**: "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization" (Chung et al., 2025)

### 3.2 Task Coverage

| Category | Task | Dataset | Description |
|----------|------|---------|-------------|
| **LLM** | lm-arena-chat | LMArena Human Preference | Conversational chatbot |
| **LLM** | gpqa | GPQA Diamond | PhD-level reasoning |
| **LLM** | sourcegraph-fim | Sourcegraph FIM | Code completion (fill-in-middle) |
| **MLLM** | image-chat | LMArena Vision Arena | Image understanding + chat |
| **MLLM** | video-chat | LLaVA-Video-178K | Video understanding + chat |
| **MLLM** | audio-chat | NVIDIA AudioSkills | Audio understanding + chat |
| **Diffusion** | text-to-image | Open Image Preferences | Image generation |
| **Diffusion** | text-to-video | EvalCrafter T2V | Video generation |

### 3.3 Runtime

- **LLM/MLLM**: vLLM v0.11.1 (served via OpenAI-compatible API inside containers)
- **Diffusion**: xDiT v0.4.5 (for parallel diffusion inference)
- Supports both Docker and Singularity containers

### 3.4 Configuration System

Configs live in a structured directory hierarchy:

```
configs/vllm/{task}/{org}/{model}/{gpu}/
  monolithic.config.yaml    # vLLM server arguments
  monolithic.env.yaml       # Environment variables
  num_gpus.txt              # GPU count
  sweeps.yaml               # Parameter sweeps
  extra_body.json           # Additional request body params
```

**Example sweep config** (Qwen3-14B on H100):
```yaml
sweep:
  - num_request_repeats: [1]
    max_num_seqs: [8, 16, 32, 64, 96, 128]
  - num_request_repeats: [2]
    max_num_seqs: [192, 256]
```

**Example vLLM config**:
```yaml
async-scheduling: true
max-num-batched-tokens: 8192
max-model-len: 40960
dtype: bfloat16
trust-remote-code: true
reasoning-parser: qwen3
api-server-count: 3
```

### 3.5 Benchmark Methodology

1. **Server startup**: vLLM launches inside container with specified config
2. **Warmup**: Initial requests establish steady state
3. **Steady-state measurement**: Energy measured only during steady state (excludes startup/cooldown)
4. **Energy measurement**: Zeus ZeusMonitor wraps the benchmark window
5. **Prometheus collection**: 1-second interval time-series from vLLM metrics endpoint
6. **Per-request tracking**: Each request records TTFT, ITL array, latency, energy, success

### 3.6 Energy Measurement Approach

The benchmark measures **steady-state GPU energy** using Zeus:

```
Total energy during steady state / Total output tokens = Energy per token (J/token)
```

Key methodology decisions:
- **Steady-state only**: Excludes startup/shutdown transients
- **GPU energy only**: Does not include CPU, DRAM, or cooling overhead
- **Server-mode**: All inference through vLLM OpenAI API (realistic deployment scenario)
- **Sweep over max_num_seqs**: Tests different concurrent batch sizes to find efficiency frontier

### 3.7 Results Schema

Per benchmark run (`results.json`):
```json
{
  "model_id": "Qwen/Qwen3-8B",
  "gpu_model": "H100",
  "num_gpus": 1,
  "max_num_seqs": 128,
  "num_prompts": 1024,
  "completed": 1024,
  "duration": 135.2,
  "steady_state_duration": 97.6,
  "steady_state_energy": 54631.48,
  "steady_state_energy_per_token": 0.093545,
  "output_throughput": 4890.37,
  "total_output_tokens": 661234,
  "results": [
    {
      "ttft": 0.156,
      "itl": [0.012, 0.011, ...],
      "latency": 8.234,
      "energy": 52.1,
      "input_len": 245,
      "output_len": 646,
      "success": true
    }
  ]
}
```

### 3.8 GPUs Benchmarked

Currently: **H100** and **B200** (NVIDIA's latest datacenter GPUs).

---

## 4. ML.ENERGY Leaderboard

### 4.1 Architecture

The leaderboard (v3) is a **static React/TypeScript web app**:

- Built with Vite + React + Tailwind CSS
- Data stored as static JSON files in `public/data/`
- No backend server (fully client-side)
- Deployed via GitHub Pages

Previous v2 was a Gradio app on HuggingFace Spaces.

### 4.2 Data Pipeline

```
Benchmark runs (benchmark repo)
        |
        v
  build_data.py (scripts/build_data.py)
        |
   +----+----+
   |         |
   v         v
index.json   models/{model}__{task}.json
             tasks/{task}.json
```

The data is **pre-aggregated** and committed to the leaderboard repo. No live data fetching.

### 4.3 Data Schema

**Index** (`public/data/index.json`):
```typescript
{
  last_updated: string;
  tasks: string[];           // ["gpqa", "lm-arena-chat", ...]
  architectures: Record<Architecture, string[]>;
  models: Record<string, ModelInfo>;
}
```

**Per-task data** (`public/data/tasks/{task}.json`):
```typescript
{
  task: string;
  task_display_name: string;
  architecture: "llm" | "mllm" | "diffusion";
  configurations: Configuration[];  // All model configs for this task
}
```

**Configuration (LLM/MLLM)**:
```typescript
{
  model_id: string;            // "Qwen/Qwen3-14B"
  nickname: string;            // "Qwen 3 14B"
  gpu_model: string;           // "B200"
  num_gpus: number;
  total_params_billions: number;
  activated_params_billions: number;
  architecture: string;        // "Dense Transformer"
  weight_precision: string;    // "bfloat16"
  max_num_seqs: number;        // Batch size parameter

  // Parallelisation
  tensor_parallel: number;
  expert_parallel: number;
  data_parallel: number;

  // Energy metrics
  energy_per_token_joules: number;
  energy_per_request_joules: number;
  avg_power_watts: number;

  // Performance metrics
  avg_output_len: number;
  median_itl_ms: number;
  p90_itl_ms: number;
  p95_itl_ms: number;
  p99_itl_ms: number;
  output_throughput_tokens_per_sec: number;
  avg_batch_size: number;
}
```

### 4.4 Visualisations

The leaderboard provides:

1. **Sortable table**: All configs with key metrics, filterable by task/model/GPU
2. **Energy-per-response chart**: Bar chart comparing models
3. **Time-energy tradeoff chart**: Scatter plot showing ITL vs energy as batch size changes
4. **Model detail modal**: Per-model deep dive with output length distribution
5. **Comparison modal**: Side-by-side model comparison
6. **ITL deadline slider**: Filter configs that meet a latency deadline
7. **Energy budget slider**: Filter configs within an energy budget

### 4.5 Models Currently Benchmarked (v3, Feb 2026)

Approximately 70 model-task combinations across:
- **LLM**: Qwen3 (8B, 14B, 32B, 235B MoE), DeepSeek-R1-0528, DeepSeek-V3.1, Llama-3.1 (8B, 70B, 405B), Llama-3.3-70B, Llama-4 (Scout, Maverick), Nemotron, GPT-OSS-20B/120B
- **MLLM**: Qwen3-VL series, Qwen3-Omni, Gemma-3, Llama-4 Scout/Maverick, Nemotron-VL
- **Diffusion**: FLUX.1-dev, SANA1.5, PixArt-Sigma, SD3.5, HunyuanDiT, Wan2.1, CogVideoX

### 4.6 Key Finding: Batch Size Dominates Energy Efficiency

From the leaderboard data, a striking pattern emerges. For Qwen3-14B on B200 (GPQA task):

| max_num_seqs | energy_per_token (J) | throughput (tok/s) | avg_power (W) |
|-------------|---------------------|-------------------|---------------|
| 8 | 1.366 | 234 | 320 |
| 16 | 0.781 | 462 | 361 |
| 32 | 0.418 | 1,010 | 422 |
| 64 | 0.291 | 1,458 | 507 |
| 96 | 0.218 | 2,327 | 509 |
| 128 | 0.183 | 2,771 | 508 |

**Energy per token drops 7.5x** from batch 8 to 128, while power only increases 1.6x. This demonstrates the core thesis: deployment configuration dramatically impacts energy efficiency.

---

## 5. Feature Comparison

### 5.1 Zeus vs LLenergyMeasure - Capabilities Matrix

| Capability | Zeus | LLenergyMeasure | Notes |
|------------|------|-----------------|-------|
| **Energy Measurement** | | | |
| GPU energy (NVML counter) | Yes (Volta+) | Via CodeCarbon | Zeus uses direct counter; more accurate |
| GPU energy (power polling) | Yes (pre-Volta fallback) | Via CodeCarbon | Both poll, Zeus uses sklearn AUC |
| CPU energy (RAPL) | Yes | Via CodeCarbon | Similar capability |
| DRAM energy | Yes | Via CodeCarbon | Similar capability |
| CO2 estimation | Separate module | Integrated (CodeCarbon) | CodeCarbon has better region coverage |
| Apple Silicon | Yes | No | Unique to Zeus |
| AMD GPU | Yes (amdsmi) | No | Unique to Zeus |
| Jetson | Yes | No | Unique to Zeus |
| | | | |
| **Performance Metrics** | | | |
| Latency (total) | User-measured | Yes (p50/p95/p99) | LLenergyMeasure has better stats |
| TTFT | Not built-in | Yes | LLenergyMeasure native |
| ITL | Not built-in | Yes | LLenergyMeasure native |
| Throughput (tok/s) | Not built-in | Yes | LLenergyMeasure native |
| | | | |
| **Compute Metrics** | | | |
| FLOPs estimation | No | Yes | Unique to LLenergyMeasure |
| FLOPs/token | No | Yes | Unique to LLenergyMeasure |
| GPU utilisation | No | Yes | Unique to LLenergyMeasure |
| Memory tracking | No | Yes | Unique to LLenergyMeasure |
| | | | |
| **Temporal Data** | | | |
| Power time-series | Yes (PowerMonitor) | Yes | Both, different implementations |
| Temperature time-series | Yes (TemperatureMonitor) | Yes | Both |
| Utilisation time-series | No | Yes | Unique to LLenergyMeasure |
| Memory time-series | No | Yes | Unique to LLenergyMeasure |
| | | | |
| **Inference Backends** | | | |
| PyTorch (direct) | No (library, not runner) | Yes | LLenergyMeasure runs models |
| vLLM | No (library) | Yes | LLenergyMeasure runs models |
| TensorRT-LLM | No (library) | Yes | LLenergyMeasure runs models |
| | | | |
| **Optimisation** | | | |
| Power limit optimisation | Yes | No | Unique to Zeus |
| Batch size optimisation | Yes (MAB) | No | Unique to Zeus |
| Frequency optimisation | Yes (Perseus) | No | Unique to Zeus |
| | | | |
| **Configuration** | | | |
| Batch size parameter | No (library) | Yes | LLenergyMeasure exposes this |
| Quantisation | No | Yes | LLenergyMeasure configures this |
| Tensor parallelism | No | Yes | LLenergyMeasure configures this |
| | | | |
| **Integration** | | | |
| Prometheus | Yes (native) | No | Zeus has first-class support |
| HuggingFace Trainer | Yes (callback) | No | Zeus integrates with training |
| Training loop callbacks | Yes | No | Zeus is training-focused |
| CLI tool | No | Yes (Typer) | LLenergyMeasure is CLI-first |
| | | | |
| **Infrastructure** | | | |
| Docker | Yes | Yes | Both |
| Privileged daemon | Yes (zeusd, Rust) | No | Zeus solves SYS_ADMIN elegantly |
| Multi-GPU | Yes | Yes | Both handle multi-GPU |

### 5.2 ML.ENERGY Benchmark vs LLenergyMeasure

| Aspect | ML.ENERGY Benchmark | LLenergyMeasure |
|--------|---------------------|-----------------|
| **Approach** | Server-mode (vLLM API) | Direct model loading |
| **Backend** | vLLM only (+ xDiT for diffusion) | PyTorch, vLLM, TensorRT-LLM |
| **Configuration sweep** | max_num_seqs only | batch_size, quantisation, TP, precision, etc. |
| **Workloads** | Real datasets (Arena, GPQA, etc.) | User-configurable prompts/datasets |
| **Focus** | Model comparison at scale | Deployment configuration tradeoffs |
| **Energy scope** | Steady-state GPU only | Total system (GPU+CPU+RAM) |
| **Methodology** | Fixed protocol, centrally run | User-configurable protocol |
| **Who runs it** | CMU (centrally) | Users (self-hosted) |
| **Results** | Public leaderboard | Local results (user owns) |
| **Task types** | LLM, MLLM, Diffusion | LLM only (currently) |
| **Hardware** | H100, B200 (datacenter) | Any NVIDIA GPU |

### 5.3 Overlap Analysis

```
+-------------------+-------------------+-------------------+
|   Zeus Only       |    Overlap        | LLenergyMeasure   |
|                   |                   |      Only         |
| - AMD GPU support | - GPU energy      | - FLOPs tracking  |
| - Apple Silicon   |   measurement     | - FLOPs/token     |
| - Jetson support  | - CPU energy      | - GPU utilisation  |
| - Power limit     |   (RAPL)          | - Memory tracking  |
|   optimisation    | - Power           | - Multiple LLM    |
| - Batch size      |   time-series     |   backends         |
|   optimisation    | - Temperature     | - Configuration    |
| - Frequency       |   monitoring      |   sweeps           |
|   optimisation    | - Multi-GPU       | - Quantisation     |
| - Prometheus      | - Docker support  | - Tensor parallel  |
|   integration     |                   | - CLI interface    |
| - Training        |                   | - Presets          |
|   callbacks       |                   | - Results mgmt    |
| - zeusd daemon    |                   | - Alpaca dataset   |
| - Electricity     |                   | - Multi-cycle      |
|   price tracking  |                   |   experiments      |
| - Carbon          |                   |                   |
|   intensity       |                   |                   |
+-------------------+-------------------+-------------------+
```

---

## 6. Architecture Best Practices

### 6.1 What We Can Learn from Zeus

#### 6.1.1 Hardware Abstraction Layer

Zeus's device HAL is the strongest part of their architecture. Key patterns:

1. **Singleton with auto-detection**: `get_gpus()` detects vendor automatically. We could adopt this for energy backends.

2. **ABC with comprehensive error mapping**: Every NVML error code maps to a specific Zeus exception. Very robust.

3. **Null object pattern**: `EmptyGPUs` class rather than `None` checks everywhere.

4. **Deprecated alias decorator**: Maintains backward compatibility while renaming methods:
   ```python
   @deprecated_alias("getInstantPowerUsage")
   def get_instant_power_usage(self) -> int: ...
   ```

#### 6.1.2 Named Measurement Windows

Zeus's `begin_window("name")` / `end_window("name")` pattern is elegant:
- Multiple concurrent measurements
- Clear start/end semantics
- Enforces naming (no anonymous measurements)
- Easy to correlate with code regions

**Contrast with our approach**: We use CodeCarbon's tracker start/stop, which is less granular.

#### 6.1.3 Process Management for Power Polling

Zeus's PowerMonitor spawns dedicated processes per power domain:
- Uses `multiprocessing.get_context("spawn")` (safer than fork)
- `weakref.finalize()` for cleanup (does not rely on `__del__`)
- Ready events for synchronisation
- Deduplication in the polling process

**Lesson**: Our power monitoring could benefit from this architecture rather than CodeCarbon's thread-based approach.

#### 6.1.4 Framework-Agnostic Sync

```python
def sync_execution(gpu_indices, sync_with="torch"):
    # Calls torch.cuda.synchronize(), jax.device_put(...).block_until_ready(), etc.
```

This ensures measurements capture actual computation, not just dispatched operations.

#### 6.1.5 Pydantic Compatibility Layer

```python
# zeus/utils/pydantic_v1.py
# Provides Pydantic v1 API regardless of installed version
```

Smart approach to handle the Pydantic v1->v2 transition without breaking changes.

### 6.2 What We Can Learn from ML.ENERGY Benchmark

#### 6.2.1 Steady-State Measurement

The benchmark explicitly excludes startup/shutdown transients. This is methodologically critical and something we should formalise.

#### 6.2.2 Per-Request Energy Attribution

Each request in their benchmark gets its own energy measurement. This enables computing distributions, not just averages.

#### 6.2.3 Sweep over Deployment Parameters

Their benchmark sweeps `max_num_seqs` systematically, demonstrating the energy-latency tradeoff. This is directly aligned with our value proposition.

#### 6.2.4 Prometheus Integration for Temporal Data

Collecting vLLM Prometheus metrics at 1-second intervals during benchmarks gives rich operational data (KV cache utilisation, queue depth, etc.).

### 6.3 Testing Patterns

Zeus has moderate test coverage:
- Unit tests for RAPL, device detection, monitors, optimisers
- Profile data fixtures (CSV files with real GPU measurements)
- Uses pytest with `pytest-xdist` for parallelism
- Mock-based tests for GPU interactions
- Separate CI for Python (fmt/lint/test) and Rust (zeusd)

Areas where Zeus's testing is **weaker** than ours:
- No integration tests with real GPUs
- No SSOT/introspection-based test generation
- No invalid combination testing

### 6.4 Documentation Patterns

- MkDocs Material with auto-generated API reference
- Research overview pages linking to papers
- Getting started guide with privilege setup
- Examples directory with working scripts
- Inline documentation in docstrings (Google style)

---

## 7. Integration Possibilities

### 7.1 Could Zeus Replace CodeCarbon?

**Assessment: Yes, partially, and it would be an improvement.**

**Advantages of Zeus over CodeCarbon for our use case:**

| Factor | Zeus | CodeCarbon |
|--------|------|------------|
| GPU energy accuracy | Hardware counter (Volta+) | Power polling + integration |
| Measurement overhead | Minimal | Thread-based polling |
| Per-GPU granularity | Native | Aggregated by default |
| Power time-series | Native PowerMonitor | Requires custom code |
| AMD support | Yes | No |
| Apple Silicon | Yes | No |
| Maintenance | Active (CMU research group) | Community-maintained |
| CO2 estimation | Basic (separate module) | Comprehensive (regional data, cloud providers) |

**What we would lose:**
- CodeCarbon's CO2 estimation is more comprehensive (carbon intensity databases, cloud provider detection)
- CodeCarbon's dashboard / visualisation
- CodeCarbon's TDP-based fallback when RAPL unavailable

**Recommended approach:**
1. Use Zeus `ZeusMonitor` for energy measurement (more accurate, lower overhead)
2. Keep CodeCarbon or a lightweight alternative for CO2 estimation
3. Use Zeus `PowerMonitor` for power time-series (replaces our custom polling)

**Integration sketch:**
```python
# Current (CodeCarbon)
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()
# ... inference ...
emissions = tracker.stop()

# Proposed (Zeus)
from zeus.monitor import ZeusMonitor
monitor = ZeusMonitor(gpu_indices=[0])
monitor.begin_window("inference")
# ... inference ...
result = monitor.end_window("inference")
# result.gpu_energy -> dict[int, float] in Joules
# result.cpu_energy -> dict[int, float] in Joules (if RAPL available)
```

### 7.2 Could We Contribute Results to ML.ENERGY Leaderboard?

**Assessment: Unlikely in current form, but possible with alignment.**

The ML.ENERGY Leaderboard is centrally run by CMU. Results are generated using their specific benchmark methodology (vLLM server mode, specific datasets, steady-state measurement). Our results would not be directly comparable because:

1. We measure **total system energy** (GPU+CPU+RAM); they measure **GPU energy only**
2. We test **multiple backends** (PyTorch, vLLM, TensorRT); they use **vLLM only**
3. We use **different workloads** (configurable); they use **fixed datasets**
4. Our measurement methodology differs (CodeCarbon vs Zeus)

However, if we aligned our vLLM backend measurements to their methodology, we could:
- Generate comparable results
- Submit for inclusion (if they accept external contributions)
- Use their data format for interoperability

### 7.3 Could We Use Their Benchmark Methodology?

**Assessment: Yes, selectively.**

Their methodology has strong elements we should adopt:
1. **Steady-state measurement window** (excluding startup/cooldown)
2. **Per-request energy attribution** (not just total)
3. **Systematic batch size sweeps** with energy-latency tradeoff curves
4. **Real-world datasets** (LMArena, GPQA, etc.)

We should NOT adopt:
1. Server-mode only (we need direct model loading for research)
2. vLLM-only (our multi-backend support is a strength)
3. GPU-energy-only (total system energy is more complete)

### 7.4 Architectural Integration Possibilities

```
Option A: Zeus as Energy Backend
+---------------------------------------------------+
| LLenergyMeasure                                   |
|                                                   |
| config -> orchestrator -> inference backends       |
|                              |                    |
|                              v                    |
|                     +----------------+            |
|                     | Energy Backend |            |
|                     | (abstracted)   |            |
|                     +-------+--------+            |
|                             |                     |
|              +--------------+-------------+       |
|              |              |             |       |
|              v              v             v       |
|        ZeusBackend    CodeCarbonBackend  Custom   |
|        (new)          (existing)                  |
+---------------------------------------------------+

Option B: Zeus as a Dependency (use ZeusMonitor directly)
  - Simpler: just pip install zeus-ml
  - Use ZeusMonitor.begin_window/end_window
  - Use PowerMonitor for time-series
  - Keep our own CO2 estimation

Option C: Adopt Zeus Device HAL Only
  - Fork/vendorise zeus.device
  - Use for NVML/RAPL/AMD access
  - Build our own monitors on top
```

**Recommendation: Option B** is the pragmatic choice. Zeus is a well-maintained library with a clean API. Using it as a dependency for energy measurement while keeping our own orchestration, configuration, and compute metrics preserves our unique value.

---

## 8. Differentiation Strategy

### 8.1 What ML.ENERGY Does NOT Do That We Do

1. **Multi-backend comparison**: We compare PyTorch vs vLLM vs TensorRT-LLM on the same workload. ML.ENERGY uses vLLM only. This is arguably our most unique capability.

2. **Configuration sweep across multiple dimensions**: We sweep batch size, quantisation, tensor parallelism, precision, and other parameters simultaneously. ML.ENERGY sweeps `max_num_seqs` only.

3. **Compute metrics (FLOPs)**: We track FLOPs and FLOPs/token. ML.ENERGY does not. This is important for understanding computational efficiency vs energy efficiency.

4. **Self-hosted measurement**: Users run our tool on their own hardware. ML.ENERGY is centrally run on H100/B200 only. This means we cover a wider range of GPUs (consumer, edge, older datacenter).

5. **Total system energy**: We measure GPU + CPU + RAM energy. ML.ENERGY measures GPU only. Total system energy is more relevant for true carbon footprint.

6. **Research-oriented CLI**: Our CLI is designed for researchers to run controlled experiments with statistical rigour (multi-cycle, warmup, etc.). ML.ENERGY is designed for benchmark operators.

### 8.2 What ML.ENERGY Does Better

1. **Energy measurement accuracy**: Zeus uses NVML hardware energy counters directly. We go through CodeCarbon, which adds overhead and potential inaccuracy.

2. **Hardware breadth**: Zeus supports NVIDIA, AMD, Apple Silicon, and Jetson. We support NVIDIA only.

3. **Scale of benchmarking**: They benchmark 70+ model-task combinations on cutting-edge hardware (B200). We typically benchmark fewer models on available hardware.

4. **Public leaderboard**: Their results are public and interactive. Ours are local files.

5. **Server-mode realism**: vLLM server mode with concurrent requests is more representative of real-world deployment than our direct model loading approach.

6. **Optimisation**: Zeus can actively optimise energy (power limits, frequencies). We only measure.

7. **Production monitoring**: Prometheus integration enables production deployment monitoring.

### 8.3 Competitive Positioning

```
                    Measurement Focus
                    ^
                    |
    LLenergyMeasure |  ML.ENERGY
    (Multi-backend, |  Benchmark
     Config sweeps, |  (Large-scale,
     FLOPs,         |   Server-mode,
     Self-hosted)   |   Leaderboard)
                    |
    ----------------+---------------->
    Library/Tool                Scale/Platform
    Focus                       Focus
                    |
    Zeus            |  Production
    (Library,       |  Monitoring
     Optimisation,  |  (Prometheus,
     HAL)           |   Grafana)
                    |
```

### 8.4 Are We Competitors or Complementary?

**We are complementary, not competitors.** The relationship is:

- **Zeus** is the **energy measurement layer**. It measures energy. We could (and should) use it.
- **ML.ENERGY Benchmark** is a **centralised, large-scale benchmark**. We are a **self-hosted, researcher-controlled tool**.
- **ML.ENERGY Leaderboard** shows **cross-model comparisons on fixed hardware**. We show **cross-configuration comparisons on the user's hardware**.

The real differentiation is in our unique question: **"How do deployment choices (backend, quantisation, batch size, parallelism) affect energy efficiency for a given model on your hardware?"**

ML.ENERGY asks: **"How do different models compare in energy efficiency on standard hardware?"**

These are complementary questions. A researcher might use:
1. ML.ENERGY Leaderboard to **choose which model** to deploy
2. LLenergyMeasure to **optimise how** to deploy that model

### 8.5 Strategic Recommendations

#### 8.5.1 Integrate, Do Not Compete

1. **Adopt Zeus for energy measurement** (Option B above). This gives us better accuracy and broader hardware support while reducing our own energy measurement code.

2. **Align results format** with ML.ENERGY where possible (energy_per_token_joules, energy_per_request_joules, avg_power_watts). This makes our results comparable.

3. **Reference their leaderboard** in our documentation. Position ourselves as the next step: "Found your model on ML.ENERGY? Now optimise its deployment with LLenergyMeasure."

#### 8.5.2 Double Down on Our Unique Value

1. **Multi-backend comparison** is our killer feature. No one else lets you compare PyTorch vs vLLM vs TensorRT-LLM for the same model with the same prompts and see the energy-latency-cost tradeoffs.

2. **Configuration sweep visualisation**: Show how energy changes across batch sizes, quantisation levels, and TP degrees simultaneously. The ML.ENERGY leaderboard only sweeps one dimension.

3. **Self-hosted, any-GPU**: We work on RTX 3090s, A100s, consumer GPUs, etc. ML.ENERGY only benchmarks on H100/B200.

4. **FLOPs integration**: Understanding compute efficiency alongside energy efficiency gives a fuller picture.

#### 8.5.3 Learn From Their Architecture

1. **Steady-state measurement**: Formalise warmup/cooldown exclusion in our measurement protocol
2. **Per-request energy**: Track energy per inference request, not just per experiment
3. **Named measurement windows**: Consider adopting this pattern (or use Zeus directly)
4. **Deployment parameter emphasis**: Their data proves that max_num_seqs (batch size) is the dominant factor in energy efficiency -- validate and extend this finding

#### 8.5.4 Future Integration Path

```
Phase 1 (v5.0): Adopt Zeus as optional energy backend
  - pip install llenergymeasure[zeus]
  - Falls back to CodeCarbon if Zeus unavailable
  - Better accuracy on supported hardware

Phase 2 (v6.0): Align results with ML.ENERGY format
  - Export results in ML.ENERGY-compatible JSON
  - Enable comparison with their leaderboard data
  - Support their benchmark datasets as built-in options

Phase 3 (Web platform): Link to ML.ENERGY
  - "Compare with ML.ENERGY baseline" feature
  - Show how user's deployment compares to leaderboard
  - Potential data contribution pipeline
```

---

## Appendix A: Key Code References

### Zeus Repository (github.com/ml-energy/zeus)

| File | Lines | Key Classes/Functions |
|------|-------|----------------------|
| `zeus/monitor/energy.py` | 468 | `ZeusMonitor`, `Measurement`, `MeasurementState` |
| `zeus/monitor/power.py` | 639 | `PowerMonitor`, `PowerDomain`, `PowerSample` |
| `zeus/device/gpu/common.py` | 617 | `GPU` (ABC), `GPUs` (ABC), `EmptyGPUs` |
| `zeus/device/gpu/nvidia.py` | 459 | `NVIDIAGPU`, `ZeusdNVIDIAGPU`, `NVIDIAGPUs` |
| `zeus/device/gpu/amd.py` | ~400 | `AMDGPU`, `AMDGPUs` |
| `zeus/optimizer/power_limit.py` | 551 | `GlobalPowerLimitOptimizer`, `OptimumSelector` |
| `zeus/metric.py` | 526 | `EnergyHistogram`, `EnergyCumulativeCounter`, `PowerGauge` |
| `zeus/callback.py` | 86 | `Callback`, `CallbackSet` |
| `zeus/device/gpu/__init__.py` | ~100 | `get_gpus()` (singleton factory) |

### Benchmark Repository (github.com/ml-energy/benchmark)

| Path | Purpose |
|------|---------|
| `configs/vllm/{task}/{model}/{gpu}/` | Per-model benchmark configs |
| `configs/vllm/{task}/benchmark.yaml` | Task-level command templates |
| `docs/overview.md` | Task descriptions, datasets |
| `docs/running-benchmarks.md` | Job generation, execution |
| `docs/analyzing-results.md` | Results format, directory structure |

### Leaderboard Repository (github.com/ml-energy/leaderboard)

| File | Purpose |
|------|---------|
| `src/types.ts` | TypeScript type definitions for all data |
| `src/config/tasks.ts` | Task configuration with display names |
| `src/config/columns.ts` | Column definitions for tables |
| `public/data/index.json` | Master index of models and tasks |
| `public/data/tasks/{task}.json` | Per-task aggregated data |
| `public/data/models/{model}__{task}.json` | Per-model detail data |
| `scripts/build_data.py` | Data pipeline from benchmark to leaderboard |

## Appendix B: ML.ENERGY Publications

1. **Zeus (NSDI 2023)**: "Zeus: Understanding and Optimizing GPU Energy Consumption of DNN Training" - foundational paper on GPU energy measurement and power limit optimisation.

2. **Perseus (SOSP 2024)**: "Reducing Energy Bloat in Large Model Training" - pipeline-parallel frequency optimisation.

3. **ML.ENERGY Benchmark (NeurIPS 2025 D&B Spotlight)**: "The ML.ENERGY Benchmark: Toward Automated Inference Energy Measurement and Optimization" - the benchmark methodology paper.

## Appendix C: Version and Dependency Information

### Zeus v0.13.1 Dependencies
```
Core: numpy, pandas, scikit-learn, nvidia-ml-py, pydantic, rich, tyro, httpx, amdsmi, python-dateutil
Optional: fastapi, sqlalchemy, prometheus-client, torch, transformers, lowtime, zeus-apple-silicon
Python: >=3.9
License: Apache-2.0
```

### Benchmark v3.0 Dependencies
```
Runtime: vLLM v0.11.1 (LLM/MLLM), xDiT v0.4.5 (Diffusion)
Infrastructure: Docker/Singularity, Slurm/Pegasus
Energy: Zeus (ZeusMonitor)
```

### Leaderboard v3 Dependencies
```
Frontend: React, TypeScript, Vite, Tailwind CSS
Data: Static JSON files
Deployment: GitHub Pages
```
