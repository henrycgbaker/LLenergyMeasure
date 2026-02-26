# Research: LLenergyMeasure vs Zeus Comparison

**Date:** 2026-02-17
**Source:** Research agent transcript (agent-ad09208.jsonl, 218KB)
**Status:** Complete -- agent produced a 10,370-character synthesis based on reading both codebases

---

## Summary

Detailed codebase-level comparison of LLenergyMeasure and Zeus/ML.ENERGY. The two tools are **complementary, not competitive**: LLenergyMeasure excels at multi-backend inference orchestration, streaming latencies, FLOPs estimation, and configuration management; Zeus excels at precise multi-platform energy measurement, hardware-level insights, and optimisation. The recommendation is to complement rather than replace the current CodeCarbon energy backend.

---

## 1. What LLenergyMeasure Measures That Zeus Does Not

### 1.1 Detailed FLOPs Estimation with Provenance

- Multi-strategy FLOPs estimation with **confidence tracking** (high/medium/low)
- Fallback chain: calflops -> architecture-based -> parameter-based
- Tracks **estimation method** used for downstream reliability assessment
- Properly handles BitsAndBytes quantization (computes at FP16, not reduced)
- Per-sample FLOPs with timeout protection
- Implementation: `core/flops.py`

### 1.2 Streaming Latency Metrics

- **TTFT (Time-to-First-Token)** and **ITL (Inter-Token Latency)** collection via streaming callbacks
- Per-request latency statistics: mean, median, p95, p99
- Trimmed ITL (excluding first/last tokens per request) for cleaner statistics
- Measurement mode tracking (true_streaming vs proportional_estimate)
- Works across all three backends (PyTorch, vLLM, TensorRT)

### 1.3 Model and GPU Metadata

- Comprehensive GPU topology detection with **MIG (Multi-Instance GPU) support**
- GPU info: memory, compute capability, topology
- Device selection validation
- Model metadata: parameters, architecture type, quantization support
- Implementation: `core/gpu_info.py`, `domain/model_info.py`

### 1.4 Extended Efficiency Analysis

| Metric Category | Specific Metrics |
|-----------------|-----------------|
| Token Efficiency | Token Efficiency Index (throughput x tokens_per_joule x precision_factor) |
| Memory | Tokens/GB VRAM, model memory utilisation, KV cache ratio |
| Batch | Padding overhead, batch utilisation metrics |
| KV Cache | Hit rates, block usage (vLLM-specific) |
| Request Latency | E2E latency percentiles across requests |
| Precision | Weights/activations/compute precision with perplexity degradation |

Implementation: `core/extended_metrics.py`

### 1.5 Thermal Throttling Detection

- Background sampling of GPU temperature and throttle state
- Distinguishes thermal vs. power throttling (SW vs. HW)
- Throttle duration and timestamps
- Critical for result validity assessment
- Implementation: `core/power_thermal.py` (PowerThermalSampler with threading)

### 1.6 Multi-Backend Orchestration

- **Backend-agnostic** measurement architecture via protocols
- Three production-grade backends: PyTorch + Accelerate, vLLM, TensorRT-LLM
- **RuntimeCapabilities** abstraction: each backend declares CUDA management (orchestrator vs. backend-managed)
- Tensor parallelism support detection
- **Late aggregation pattern**: raw per-process results -> aggregation on-demand (statistically correct for distributed runs)
- Implementation: `core/inference_backends/protocols.py`, `orchestration/runner.py`

### 1.7 Configuration System with SSOT

- **Single Source of Truth** via introspection: parameter metadata derived from Pydantic models, not static lists
- Automatic validation of parameter combinations (mutual exclusions)
- Streaming constraints tracking
- Built-in presets (quick-test, benchmark, throughput)
- Config generation/validation CLI
- Implementation: `config/introspection.py`, `config/models.py`, `config/backend_configs.py`

### 1.8 Comprehensive Result Aggregation

- Per-process result persistence
- Aggregation strategies for distributed experiments
- Warmup convergence detection and exclusion
- Energy baseline adjustment (idle power subtraction)
- Implementation: `results/aggregation.py`

---

## 2. What Zeus/ML.ENERGY Measures That LLenergyMeasure Does Not

### 2.1 Multi-Source Energy Attribution

| Source | Zeus | LLenergyMeasure |
|--------|------|-----------------|
| NVIDIA GPU energy | NVML direct | Via CodeCarbon (NVML wrapper) |
| AMD GPU energy | AMDSMI | Not supported |
| CPU energy (Intel RAPL) | Direct, via zeusd daemon | Via CodeCarbon (RAPL wrapper) |
| DRAM energy | Separate tracking | Via CodeCarbon (combined) |
| Apple Silicon | CPU cores (perf/efficiency), GPU, neural engine, DRAM (in mJ) | Not supported |
| NVIDIA Jetson | CPU, GPU, total chip energy | Not supported |

### 2.2 Carbon and Monetary Cost Analysis

- CO2 emissions calculation: Energy x regional carbon intensity
- Monetary cost: Energy x regional electricity rates
- Geographically aware metrics: Different regions/seasons

(LLenergyMeasure gets CO2 via CodeCarbon but not monetary cost.)

### 2.3 Service-Level Energy Accounting for LLMs

- **Per-request energy during steady state**: Designed for inference servers at capacity
- Batch-aware energy accounting: scales per-token energy x output tokens
- Reflects realistic production deployments vs. isolated runs

### 2.4 Pareto Frontier Optimisation Analysis

- Automated energy-optimal configuration recommendations
- Multi-dimensional trade-off analysis (latency vs. energy)
- Reports achievable savings (e.g., "44% energy reduction possible")

### 2.5 Performance Power Percentage Metrics

- Power draw as % of GPU TDP (relative efficiency metric)
- Throughput per watt normalisation

### 2.6 ML.ENERGY Ecosystem

- Public leaderboard: 46 models x 7 tasks (1,858 configurations)
- Comparison data: energy per token/image/video across architectures
- ML.ENERGY Colosseum: Side-by-side response quality vs. energy comparison
- Community benchmarking infrastructure

---

## 3. Areas of Direct Overlap

| Metric | LLenergyMeasure | Zeus/ML.ENERGY |
|--------|----------------|----------------|
| Total energy (Joules) | Via CodeCarbon | Platform-specific APIs |
| GPU power (Watts) | Thermal sampler (NVML) | NVML / platform APIs |
| GPU memory | PyTorch API | NVML |
| Temperature | NVML sampler | NVML / platform |
| Throughput (tok/s) | Calculated | Calculated |
| Latency / ITL | Streaming callbacks | Timing measurement |
| Model parameters | Direct count | For architecture analysis |

---

## 4. LLenergyMeasure's Codebase Structure (Relevant Files)

Based on the agent's reading of actual source files:

```
core/
  inference_backends/
    protocols.py          # RuntimeCapabilities protocol, backend abstraction
    pytorch.py            # PyTorch + Accelerate backend
    vllm.py               # vLLM backend
    tensorrt.py           # TensorRT-LLM backend
    adapters.py           # Backend adapters
    shared.py             # Shared utilities
  energy_backends/
    base.py               # EnergyBackend protocol
    codecarbon.py          # CodeCarbon implementation
  flops.py               # Multi-strategy FLOPs estimation
  compute_metrics.py     # Memory, utilization metrics
  extended_metrics.py    # TPOT, TEI, memory efficiency, GPU utilisation
  gpu_utilisation.py     # NVML-based GPU utilization sampler
  power_thermal.py       # PowerThermalSampler (threading)
  model_loader.py        # Model loading
  dataset_loader.py      # Dataset loading
  inference.py           # Main inference engine
  baseline.py            # Baseline measurement
  warmup.py              # Warmup detection
  traffic.py             # Traffic simulation
```

### CodeCarbonBackend (from codecarbon.py)

```python
class CodeCarbonBackend:
    """Energy tracking backend using CodeCarbon.
    Implements the EnergyBackend protocol."""

    def __init__(self, measure_power_secs=1, tracking_mode="process"):
        ...

    def start_tracking(self) -> Any:
        """Start energy tracking."""
        ...

    def stop_tracking(self, tracker) -> EnergyMetrics:
        """Stop tracking, return EnergyMetrics."""
        ...

    def is_available(self) -> bool:
        """Check if codecarbon package is installed."""
        ...
```

Returns a `CodeCarbonData` dataclass:
```python
@dataclass
class CodeCarbonData:
    cpu_power: float | None
    gpu_power: float | None
    ram_power: float | None
    cpu_energy: float | None
    gpu_energy: float | None
    ram_energy: float | None
    total_energy_kwh: float
    emissions_kg: float | None
```

---

## 5. Integration Possibilities

### 5.1 Could Zeus Replace Our Energy Measurement Layer?

**Short answer: Partially, with significant trade-offs.**

**Advantages of Zeus:**
- More hardware platform coverage (AMD, Apple Silicon, Jetson, CPUs via RAPL)
- Better CPU/DRAM energy attribution (separate tracking)
- Proven production-grade measurement (ML.ENERGY leaderboard)
- Service-level energy accounting
- Lower-level measurement (potentially more accurate)
- <10ms overhead

**Disadvantages:**
- Zeus is not LLM-specific (originally designed for training)
- Lacks FLOPs estimation
- No streaming latency support (TTFT/ITL)
- No thermal throttling detection
- No backend orchestration layer
- No distributed aggregation strategy
- Requires new integration work for CLI/config system

### 5.2 Recommended Integration Approach

**Complement rather than replace.** Zeus could supplement CPU/DRAM energy. ML.ENERGY leaderboard provides cross-model comparison context.

### Phase 1 -- Energy Enhancement
- Keep CodeCarbon for baseline (battery-tested, widely available)
- Add optional Zeus backend for platforms it supports (AMD, Apple Silicon, RAPL CPUs)
- Expose choice via config

### Phase 2 -- Result Enrichment
- Add CO2/monetary cost fields to result schema
- Integrate regional electricity rates for cost calculation
- Use ML.ENERGY leaderboard as reference for cross-model comparisons

### Phase 3 -- Advanced Accounting
- Implement service-level energy accounting for batch inference
- Add Pareto frontier analysis for configuration optimisation
- Compare against ML.ENERGY leaderboard entries

### Phase 4 -- Publication
- Publish aggregated results to public benchmarks
- Consider ML.ENERGY leaderboard submission pipeline

---

## 6. Our Unique Value-Add Beyond Energy Measurement

Based on the codebase comparison, LLenergyMeasure's distinctive capabilities:

### 6.1 Streaming Latency Collection
Industry-standard TTFT/ITL metrics collected across three backends consistently. Per-request statistics with trimming. No other general-purpose tool does this for all three backends.

### 6.2 Multi-Backend Abstraction
Protocol-based architecture enables swapping backends without infrastructure changes. RuntimeCapabilities abstraction hides backend-specific CUDA management. Distributed orchestration (torchrun, accelerate, direct).

### 6.3 Configuration System
SSOT introspection: configs are self-documenting, auto-validated. Config generation utilities (grid search generation). Built-in presets with clear semantics.

### 6.4 Late Aggregation Pattern
Statistically correct handling of distributed experiments. Per-process results archived separately for reproducibility. Aggregation on-demand (compute percentiles from raw samples).

### 6.5 FLOPs Estimation
Multi-strategy with provenance tracking. Correct handling of quantization (BNB computes at FP16). Confidence levels for downstream reliability assessment.

### 6.6 Thermal Safety
Detects thermal/power throttling during runs. Marks results invalid if throttling occurred. Temperature tracking throughout experiment.

---

## 7. Bottom Line

**LLenergyMeasure and Zeus/ML.ENERGY are complementary, not competitive:**

| Dimension | LLenergyMeasure Excels | Zeus/ML.ENERGY Excels |
|-----------|----------------------|----------------------|
| Inference orchestration | Multi-backend (PyTorch, vLLM, TensorRT) | N/A (not an inference tool) |
| Streaming latency | TTFT, ITL, per-request stats | N/A |
| FLOPs estimation | Multi-strategy with confidence | N/A |
| Configuration | SSOT, presets, grid generation | Minimal (library-only) |
| Energy precision | CodeCarbon (process-level) | Direct NVML/AMDSMI (<10ms) |
| Hardware coverage | NVIDIA only (via CodeCarbon) | NVIDIA, AMD, Intel CPU, Apple, Jetson |
| Energy optimisation | N/A | Power limit, batch size, Perseus |
| Production deployment | N/A | Service-level energy accounting |
| Ecosystem | Standalone tool | ML.ENERGY Leaderboard, Benchmark |

**Recommendation:** Use LLenergyMeasure as primary for inference research and optimisation. Consider Zeus as an optional energy backend for version 3.0+ when expanding hardware support. Reference ML.ENERGY leaderboard for cross-model comparison context. No immediate need to replace the CodeCarbon energy layer.

---

## Sources

- [Zeus Project](https://ml.energy/zeus/)
- [ML.ENERGY Leaderboard](https://ml.energy/leaderboard/)
- [ML.ENERGY Benchmark Paper](https://arxiv.org/html/2505.06371v1)
- [Zeus Measurement Documentation](https://ml.energy/zeus/measure/)
- [ML.ENERGY Blog -- Leaderboard v3.0](https://ml.energy/blog/measurement/energy/diagnosing-inference-energy-consumption-with-the-mlenergy-leaderboard-v30/)
- LLenergyMeasure codebase files read directly:
  - `src/llenergymeasure/core/energy_backends/codecarbon.py`
  - `src/llenergymeasure/core/inference_backends/protocols.py`
  - `src/llenergymeasure/core/flops.py`
  - `src/llenergymeasure/core/extended_metrics.py`
  - `src/llenergymeasure/core/power_thermal.py`
  - `src/llenergymeasure/core/gpu_utilisation.py`
  - `src/llenergymeasure/config/introspection.py`
