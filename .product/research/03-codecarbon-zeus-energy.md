# Research: CodeCarbon, Zeus, and Energy Measurement Tools

**Date:** 2026-02-17
**Source:** Research agent transcript (agent-a8d68b4.jsonl, 159KB)
**Status:** Complete -- agent produced a 17,574-character synthesis

---

## Summary

This research covers the landscape of energy/sustainability measurement tools for ML, with deep focus on CodeCarbon (the tool LLenergyMeasure currently uses) and Zeus (the most precise alternative). Key finding: almost all tools are library-first; web dashboards are rare and late-stage; all credible leaderboards run their own benchmarks centrally rather than accepting user submissions.

---

## 1. CodeCarbon

**Version:** 3.2.2 (released 1 February 2026). MIT licence. Maintained by Mila, DataForGood, BCG GAMMA, Comet.ml, and Haverford College.

### 1.1 Product Architecture -- Three Layers

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Python library | `pip install codecarbon` | Measure GPU + CPU + RAM power, apply regional carbon intensity |
| Local dashboard (Carbonboard) | Plotly Dash app (`pip install codecarbon[carbonboard]`) | Offline visualisation from CSV files |
| Cloud dashboard + API | FastAPI server at `api.codecarbon.io` / `dashboard.codecarbon.io` | Central, persistent, multi-project tracking |

### 1.2 Integration Modes (Five Patterns)

1. **Explicit start/stop:**
```python
tracker = EmissionsTracker()
tracker.start()
# ... workload ...
tracker.stop()
```

2. **Context manager:**
```python
with EmissionsTracker() as tracker:
    # ... workload ...
```

3. **Decorator:**
```python
@track_emissions()
def train_model():
    # ... workload ...
```

4. **Task manager** (granular sub-operation tracking):
```python
tracker.start_task("preprocessing")
# ...
tracker.stop_task()
tracker.start_task("inference")
# ...
tracker.stop_task()
```

5. **CLI monitor** (language-agnostic, system-wide):
```bash
codecarbon monitor  # Ctrl+C to stop
```

Also: `OfflineEmissionsTracker` for air-gapped environments (requires `country_iso_code`).

### 1.3 Configuration Hierarchy

Script parameters > Environment variables (`CODECARBON_*`) > Local `.codecarbon.config` > Global `~/.codecarbon.config`

### 1.4 Output Options

- CSV file (`emissions.csv` by default)
- API upload (`save_to_api=True`, configurable interval via `api_call_interval`)
- Custom loggers (derive from `BaseOutput`)
- Comet.ml integration

### 1.5 Cloud Dashboard Architecture

- Hierarchy: Organisation > Team > Project > Experiment
- FastAPI backend at `api.codecarbon.io` (Swagger docs at `/docs`)
- PostgreSQL database
- Hosted free by Clever Cloud (with usage limits)
- **All data sent to the API is public**
- Dashboard shows: aggregate energy/emissions per org, per-project breakdowns, per-run bubble charts, carbon intensity maps
- No nice web UI for creating orgs/projects -- must use OpenAPI interface directly

### 1.6 Self-Hosting

The repo includes a `docker-compose.yml` with a `carbonserver` FastAPI service (port 8000) + PostgreSQL + PGAdmin. Database URL is configurable via environment variables. Documented in `CONTRIBUTING.md` rather than user-facing docs.

### 1.7 Local Dashboard (Carbonboard)

```bash
carbonboard --filepath="emissions.csv" --port=3333
```

Visualisations: summary with real-world equivalents (km driven, TV hours), regional comparisons of electricity grid composition, cloud region recommendations for lowest emissions.

---

## 2. Zeus (ML.ENERGY Initiative, University of Michigan)

**Version:** v0.13.1 (November 2025). Apache-2.0 licence. PyTorch ecosystem project. 2024 Mozilla Technology Fund awardee.

### 2.1 Product Architecture

```
zeus/
  monitor/        # Energy & power measurement (CLI + programmatic)
  optimizer/      # Energy optimization algorithms
  device/         # CPU/GPU/SoC device abstraction
  callback.py     # PyTorch training callback
  metric.py       # Prometheus metrics support
  utils/          # Logging, async, framework helpers
zeusd/            # Zeus daemon (background service for CPU/DRAM via RAPL)
```

### 2.2 Core Measurement API -- ZeusMonitor

```python
from zeus.monitor import ZeusMonitor
import torch

monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])

for epoch in range(100):
    monitor.begin_window("epoch")
    # ... training/inference code ...
    measurement = monitor.end_window("epoch")
    print(f"Energy: {measurement.total_energy} J, Time: {measurement.time} s")
```

**Constructor parameters:**
- `gpu_indices`: Which GPUs to monitor (defaults to all)
- `cpu_indices`: Which CPUs to monitor
- `approx_instant_energy`: Approximate energy for short windows via instantaneous power x time
- `log_file`: Path for CSV logging
- `sync_execution_with`: `"torch"` | `"jax"` | `"cupy"` -- ensures async GPU operations are captured

**Measurement dataclass:**
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
        return sum(self.gpu_energy.values())
```

**Key features:**
- Named, overlapping/nested measurement windows
- Measurement overhead < 10ms per call
- Automatic CPU-GPU synchronisation for accurate timing
- Respects `CUDA_VISIBLE_DEVICES`
- CSV logging with headers: `start_time, window_name, elapsed_time, gpu{i}_energy`

### 2.3 Energy Measurement Strategy

Two approaches based on GPU generation:
1. **Modern GPUs (Volta+):** Directly queries `nvmlDeviceGetTotalEnergyConsumption` API
2. **Older architectures:** Background `PowerMonitor` process polls `nvmlDeviceGetPowerUsage`, integrates samples over time

### 2.4 PowerMonitor

Background monitoring for power timeline analysis:

```python
from zeus.monitor import PowerMonitor

monitor = PowerMonitor(gpu_indices=[0])
timelines = monitor.get_all_power_timelines()
```

Three power domains (when supported):
- GPU instantaneous power
- GPU average power (windowed, one-second interval)
- GPU memory average power (Hopper or newer)

Architecture: Multiprocessing with deduplication (only records when values change), queue-based communication, automatic polling frequency detection.

| Method | Purpose |
|--------|---------|
| `get_power_timeline()` | Time-series power data with filtering |
| `get_all_power_timelines()` | All monitored domains |
| `get_energy()` | Total energy (joules) between timestamps |
| `get_power()` | Instantaneous power at a specific time |
| `stop()` | Terminate monitoring processes |

### 2.5 CarbonEmissionMonitor

Bridges energy measurement with real-time carbon intensity:
- Tracks elapsed time, GPU/CPU/DRAM energy, carbon emissions (gCO2eq)
- Background subprocess polls energy and carbon intensity data
- Supports multiple concurrent measurement windows
- Framework integration (PyTorch, JAX, CuPy)

### 2.6 Hardware Support

| Platform | Implementation |
|----------|---------------|
| NVIDIA GPUs | NVML (pynvml) |
| AMD GPUs | AMDSMI (ROCm 6.1+ only) |
| Intel/AMD CPUs | RAPL via `zeusd` daemon (requires root) |
| Apple Silicon | `zeus-apple-silicon` package |
| NVIDIA Jetson | CPU, GPU, total chip energy |
| DRAM | On supported CPUs via RAPL |

**Limitation:** AMD GPU support requires ROCm >= 6.1 due to incorrect power/energy values in older versions.

### 2.7 GPU Abstraction Layer

```python
from zeus.device import get_gpus
gpus = get_gpus()  # Returns NVIDIAGPUs or AMDGPUs
constraints = gpus.getPowerManagementLimitConstraints(gpu_index)
```

19+ specific exception types for granular error handling.

### 2.8 Prometheus Integration

Three metric types:
- `EnergyHistogram` -- energy distribution for repeated code ranges
- `EnergyCumulativeCounter` -- cumulative energy over time
- `PowerGauge` -- real-time GPU power

Naming: `energy_monitor_{component}_energy_joules`, `power_monitor_gpu_power_watts`

### 2.9 Optimisers

1. **Power Limit Optimizer** -- finds optimal GPU power cap
2. **Batch Size Optimizer** -- optimises batch size for energy efficiency
3. **Perseus** (SOSP '24) -- pipeline frequency optimiser for large model training; reduces energy with negligible throughput loss

### 2.10 CLI Tools

```bash
python -m zeus.monitor energy   # Total GPU energy consumption
python -m zeus.monitor power    # Real-time power draw display
```

### 2.11 Installation and Dependencies

```bash
pip install zeus                       # Basic
pip install zeus[prometheus]           # With Prometheus
pip install zeus[apple]                # Apple Silicon
pip install -e '.[dev]'                # Development
```

Docker: `mlenergy/zeus:latest` with NVIDIA Container Toolkit, requires `--cap-add SYS_ADMIN` for GPU optimisation and RAPL volume mount for CPU energy.

Core deps: numpy, pandas, scikit-learn, nvidia-ml-py, pydantic, rich, tyro, httpx.

### 2.12 Key Distinction from CodeCarbon

Zeus focuses on **energy measurement** (joules) at high precision (<10ms overhead) and **energy optimisation**. It does not compute carbon emissions itself -- that is left to downstream tools. CodeCarbon provides broader scope (energy + carbon + reporting) but with lower precision.

---

## 3. Other Energy Measurement Tools

### 3.1 Carbontracker

- **Architecture:** Python library, epoch-based tracking
- **Integration:** `tracker = CarbonTracker(epochs=max_epochs)`, then `tracker.epoch_start()` / `tracker.epoch_end()`
- **Unique feature:** Predictive forecasting after `epochs_before_pred` epochs; optional `stop_and_confirm=True` pauses training for user approval
- **Carbon intensity:** ElectricityMaps API for 160+ regions (UK and Denmark in real-time)
- **Output:** Log files; `parser` module aggregates log directories
- **Limitation:** Designed specifically for training loops with epochs; not general-purpose

### 3.2 experiment-impact-tracker (Henderson et al., 2020)

- Context manager: `with ImpactTracker(experiment1): do_something()`
- Launches separate background Python process for monitoring
- Measures power draw (CPU via RAPL, GPU via nvidia-smi), hardware info, Python package versions
- 10-second measurement interval
- **Limitation:** Linux only, Intel CPUs + NVIDIA GPUs. Not actively maintained.

### 3.3 PowerJoular

- **Written in Ada** for minimal energy overhead
- External process monitoring by PID: `powerjoular -p <PID>`
- Uses `/proc/stat` and `/proc/pid/stat` for CPU cycle proportions; applies RAPL proportionally
- Supports Intel/AMD CPUs, NVIDIA GPUs, ARM (Raspberry Pi, Asus TinkerBoard)
- Process-level and multi-process monitoring
- **Key distinction:** Language-agnostic. Monitors any process externally. No Python dependency.

### 3.4 Eco2AI

- Does not require RAPL access (unlike CodeCarbon)
- Internal database of 3,279 CPU models and 365 regional emission coefficients
- 10-second measurement interval
- Local file output only, no dashboard
- Simpler than CodeCarbon; works where RAPL access is restricted

### 3.5 Other Tools

| Tool | Type | Key Feature |
|------|------|-------------|
| Green-Algorithms | Web calculator (green-algorithms.org) | Estimate carbon from hardware specs + runtime |
| MLCO2 | Web calculator (mlco2.github.io/impact) | Simple GPU-hours to CO2 calculator |
| Cumulator | Python library | Includes data transfer emissions |

---

## 4. Patterns Across Energy Measurement Tools

| Pattern | Tools Using It |
|---------|---------------|
| Library-first | CodeCarbon, Zeus, Carbontracker, eco2AI, experiment-impact-tracker |
| Context manager / decorator | CodeCarbon, experiment-impact-tracker, eco2AI |
| Named measurement windows | Zeus (most flexible programmatic API) |
| External CLI monitor | PowerJoular, CodeCarbon (`monitor`), Zeus (`python -m zeus.monitor`) |
| Web calculator | Green-Algorithms, MLCO2 |

**Key observations:**
1. **Almost all are library-first.** CLI and web are secondary.
2. **Web dashboards are rare and late-stage.** Only CodeCarbon has a real web dashboard.
3. **Central servers are the exception.** CodeCarbon's hosted API is the only central aggregation service. All others write to local disk.
4. **Carbon vs energy split.** Zeus measures energy only (joules). CodeCarbon/Carbontracker/eco2AI add carbon conversion. These are architecturally separable concerns.
5. **Accuracy varies wildly.** PMC comparison paper found up to 400% variation between tools measuring the same workload. RAPL-based tools disagree due to different component scoping.
6. **Self-hosting is an afterthought everywhere.** CodeCarbon's `docker-compose.yml` exists but is in `CONTRIBUTING.md`, not user docs.

---

## 5. Leaderboard and Comparison Platforms

### 5.1 Hugging Face Open LLM Leaderboard

- **Frontend:** Gradio Space on Hugging Face (custom component, client-side filtering/sorting)
- **Backend:** Fork of EleutherAI lm-evaluation-harness
- **Compute:** HF's own GPU cluster runs all evaluations centrally
- **Data:** Two HF datasets -- `open-llm-leaderboard/results` (scores) and `open-llm-leaderboard/requests` (queue/status)
- **Benchmarks (v2):** IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-PRO (0-shot or few-shot)
- **Submission:** Users submit model names; HF runs evaluation on their cluster. **No user-submitted results accepted.**
- **Reproducibility:** `lm-eval --model_args="pretrained=<model>" --tasks=leaderboard --batch_size=auto`

### 5.2 ML.ENERGY Leaderboard (University of Michigan)

- **Frontend:** Node.js web app (Vite + TypeScript + Tailwind CSS)
- **Measurement:** Zeus library for energy measurement
- **Inference:** vLLM for LLMs, Diffusers for diffusion models
- **Hardware:** NVIDIA A100 (40GB), H100 (80GB) on AWS. v3.0 added B200 GPUs
- **Scope (v3.0, December 2025):** 46 models, 7 tasks, 1,858 configurations on H100 and B200
- **Metrics:** Per-request energy (joules), time-energy Pareto frontiers, latency targets
- **Four design principles:**
  1. Software-based GPU measurement for portability
  2. Production-grade software (vLLM on real GPUs)
  3. Per-response granularity (not per-token)
  4. Actionable optimisation recommendations
- **NeurIPS D&B 2025 spotlight paper**

### 5.3 HF LLM-Perf Leaderboard

- Built on `optimum-benchmark` (unified multi-backend benchmarking)
- Measures latency, throughput, memory, and energy
- Single GPU per benchmark to avoid communication-dependent variance

### 5.4 Universal Pattern

**All credible leaderboards run their own benchmarks on their own hardware.** None accept user-submitted results. This ensures comparable, reproducible measurements at the cost of scaling being limited by available compute.

| Leaderboard | Data Source | Who Runs |
|-------------|------------|----------|
| HF Open LLM | HF GPU cluster | HF team (centralised) |
| ML.ENERGY | AWS GPU instances | Michigan team (centralised) |
| HF LLM-Perf | HF infrastructure | HF team (centralised) |
| Artificial Analysis | Own infrastructure | Own team (centralised) |

---

## 6. Relevance to LLenergyMeasure

1. **Library-first is the norm.** Every successful tool started as a Python library. CLI and web came later (if at all).

2. **CodeCarbon is the closest architectural precedent** for a tool with library + CLI + dashboard + central API. Its hierarchy (Org > Team > Project > Experiment) maps conceptually to what a leaderboard needs.

3. **Zeus is the closest technical precedent** for precision energy measurement of ML inference. The ML.ENERGY Benchmark uses Zeus + vLLM -- essentially the same measurement stack that LLenergyMeasure targets.

4. **Leaderboards run benchmarks centrally.** No credible leaderboard accepts user-submitted results. The ML.ENERGY Leaderboard is the most relevant model.

5. **Self-hosting support is underdeveloped everywhere.** This is a potential differentiator for LLenergyMeasure.

6. **The carbon conversion layer is separable.** Zeus proves that energy measurement (joules) and carbon accounting (kgCO2e) are distinct concerns. LLenergyMeasure already focuses on energy; carbon can be a thin layer on top.

---

## Sources

- [CodeCarbon PyPI](https://pypi.org/project/codecarbon/)
- [CodeCarbon Quickstart](https://mlco2.github.io/codecarbon/usage.html)
- [CodeCarbon Visualisation](https://mlco2.github.io/codecarbon/visualize.html)
- [CodeCarbon API](https://mlco2.github.io/codecarbon/api.html)
- [CodeCarbon GitHub](https://github.com/mlco2/codecarbon)
- [CodeCarbon Dashboard](https://dashboard.codecarbon.io/)
- [CodeCarbon API Swagger](https://api.codecarbon.io/docs)
- [Zeus Project](https://ml.energy/zeus/)
- [Zeus GitHub](https://github.com/ml-energy/zeus)
- [Zeus Measuring Energy](https://ml.energy/zeus/measure/)
- [Zeus on PyTorch Blog](https://pytorch.org/blog/zeus/)
- [ML.ENERGY Leaderboard](https://ml.energy/leaderboard/)
- [ML.ENERGY Benchmark Paper](https://arxiv.org/html/2505.06371v1)
- [ML.ENERGY Leaderboard v3.0 Blog](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/)
- [Carbontracker](https://carbontracker.info/)
- [experiment-impact-tracker](https://github.com/Breakend/experiment-impact-tracker)
- [PowerJoular](https://github.com/joular/powerjoular)
- [Eco2AI](https://github.com/sb-ai-lab/Eco2AI)
- [Green Algorithms](https://www.green-algorithms.org/)
- [PMC Energy Tools Comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC10661046/)
- [HF Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [HF LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)
