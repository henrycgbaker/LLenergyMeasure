# Broader Landscape: LLM Efficiency Measurement, Energy Benchmarking, and Inference Optimisation

**Date**: 2026-02-17
**Phase**: 04.5 Strategic Reset
**Purpose**: Comprehensive scan of tools, frameworks, and leaderboards relevant to LLenergyMeasure's positioning

---

## Table of Contents

1. [LLM Inference Benchmarking Tools](#1-llm-inference-benchmarking-tools)
2. [Energy and Carbon Measurement Tools](#2-energy-and-carbon-measurement-tools)
3. [Deployment Configuration and Optimisation Tools](#3-deployment-configuration-and-optimisation-tools)
4. [Academic Tools and Frameworks](#4-academic-tools-and-frameworks)
5. [Industry Leaderboards](#5-industry-leaderboards)
6. [LLM Inference Serving Engines](#6-llm-inference-serving-engines)
7. [Gap Analysis Summary](#7-gap-analysis-summary)
8. [Strategic Recommendations](#8-strategic-recommendations)

---

## 1. LLM Inference Benchmarking Tools

### 1.1 Optimum-Benchmark (HuggingFace)

**Repository**: https://github.com/huggingface/optimum-benchmark
**Stars**: ~329 | **License**: Apache-2.0 | **Status**: Active (updated 2026-02-16)

**What it does**: Unified multi-backend benchmarking for HuggingFace ecosystem (Transformers, Diffusers, PEFT, TIMM, Sentence Transformers). Supports inference and training scenarios with multiple launchers (process, torchrun, inline).

**Key capabilities**:
- **Backends**: PyTorch, ONNX Runtime, OpenVINO, TensorRT-LLM, vLLM, Py-TXI (TGI/TEI), IPEX, llama.cpp
- **Metrics**: Latency, throughput, memory, energy (via CodeCarbon integration)
- **Launchers**: Process isolation, torchrun (distributed), inline
- **Configuration**: Hydra-based config system (YAML files)
- **Python API**: Full programmatic access, not just CLI
- **HuggingFace Hub**: Push results directly to Hub for sharing

**Architecture**:
```python
from optimum_benchmark import Benchmark, BenchmarkConfig, InferenceConfig, PyTorchConfig
launcher_config = TorchrunConfig(nproc_per_node=2)
scenario_config = InferenceConfig(latency=True, memory=True)
backend_config = PyTorchConfig(model="gpt2", device="cuda")
benchmark_config = BenchmarkConfig(name="test", scenario=scenario_config,
                                    launcher=launcher_config, backend=backend_config)
report = Benchmark.launch(benchmark_config)
```

**What it does that we don't**:
- More backend coverage (ONNX Runtime, OpenVINO, IPEX, llama.cpp)
- CPU benchmarking support
- ROCm (AMD GPU) support
- Training scenario benchmarking
- HuggingFace Hub integration for results sharing
- Hydra sweep support for hyperparameter grid searches
- Docker images per device type
- Model loading latency/memory/energy tracking
- numactl support for NUMA node control

**What we do that it doesn't**:
- LLM-specific streaming latency (TTFT, ITL with trimming, per-token timestamps)
- FLOPs estimation with multi-strategy fallback chain
- Baseline power measurement and adjustment
- Thermal throttle detection and warnings
- Precision-normalised cross-backend efficiency metrics (tokens/joule, TEI)
- Extended efficiency metrics (TPOT, batch efficiency, KV cache metrics)
- Power/thermal time-series sampling during inference
- SSOT config introspection system
- Experiment lifecycle management with state machine

**Relationship**: **Primary competitor**. Most similar tool to LLenergyMeasure. They have broader backend coverage; we have deeper LLM inference analysis (especially energy + streaming latency + efficiency normalisation).

**Integration opportunity**: Low -- different config systems (Hydra vs our YAML/Pydantic), but their backend abstraction pattern is worth studying.

---

### 1.2 AIPerf (NVIDIA) -- successor to GenAI-Perf

**Repository**: https://github.com/ai-dynamo/aiperf
**License**: Apache-2.0 | **Status**: Active (new, replacing GenAI-Perf)

**What it does**: Comprehensive benchmarking tool for generative AI models served through inference servers. Measures throughput and latency of LLM serving endpoints. This is NVIDIA's official replacement for GenAI-Perf.

**Key capabilities**:
- **Metrics**: Output token throughput, TTFT, time to second token, ITL, request throughput, request latency (p50/p90/p95/p99)
- **Load modes**: Concurrency, request-rate, request-rate with max concurrency, trace replay
- **Advanced features**:
  - Multiprocess support for scalability
  - Gradual ramping of load
  - Warmup phase configuration
  - Goodput measurement (requests meeting SLOs)
  - Time-sliced metrics for detecting degradation
  - HTTP trace metrics (DNS, TCP/TLS, TTFB)
  - **GPU telemetry via DCGM** (power, utilisation, memory, temperature)
  - Server-side Prometheus metrics collection
  - Pareto analysis and visualisation
  - Template endpoints for custom APIs
  - User-centric timing for KV cache benchmarking
  - Request cancellation testing
  - Multi-URL load balancing
- **UI**: Dashboard (TUI), simple (progress bars), or none
- **Output**: CSV, JSON, profile exports, PNG plots

**What it does that we don't**:
- Server/API endpoint benchmarking (we benchmark local inference only)
- Load generation (concurrent requests, request rate control)
- Goodput measurement with SLO thresholds
- Time-sliced metric analysis
- HTTP-layer metrics (DNS, TLS, TTFB at transport level)
- GPU telemetry via NVIDIA DCGM (more accurate than NVML for energy)
- Trace replay for deterministic benchmarking
- Request cancellation and timeout testing
- Multi-URL load balancing across GPU instances
- Arrival pattern control (Poisson, gamma, constant)
- Real-time TUI dashboard

**What we do that it doesn't**:
- Full energy measurement (Joules, kWh, CO2 emissions via CodeCarbon)
- Energy per token calculation
- Baseline-adjusted energy attribution
- FLOPs estimation
- Multi-backend local inference orchestration (not just hitting an API)
- Config-driven experiment management
- Precision-normalised efficiency comparisons
- Model loading as part of the measurement workflow

**Relationship**: **Complementary, partially overlapping**. AIPerf is a serving-layer benchmarking tool (tests a running server); we measure the inference engine itself. They focus on serving under load; we focus on per-inference efficiency measurement. Their DCGM telemetry for GPU power is interesting.

**Integration opportunity**: Medium -- we could use AIPerf to benchmark our backends when exposed as servers (e.g., vLLM in server mode). Their DCGM integration approach is worth studying.

---

### 1.3 GenAI-Perf (NVIDIA) -- DEPRECATED

**Repository**: https://github.com/triton-inference-server/perf_analyzer/tree/main/genai-perf
**Stars**: ~134 (parent repo) | **License**: BSD-3-Clause | **Status**: Being phased out, replaced by AIPerf

**What it does**: Command-line tool for measuring throughput and latency of LLM inference servers. Required a running inference server (Triton, TensorRT-LLM, vLLM, etc.).

**Note**: Superseded by AIPerf (see above). Listed for completeness as some documentation and guides still reference it.

---

### 1.4 LLMPerf (Anyscale/Ray)

**Repository**: https://github.com/ray-project/llmperf
**Stars**: ~1,088 | **License**: Apache-2.0 | **Status**: ARCHIVED

**What it does**: Evaluates performance of LLM API endpoints. Implements load testing and correctness testing for LLM APIs.

**Key capabilities**:
- Load testing with concurrent requests
- Correctness testing
- Metrics: Inter-token latency, generation throughput per request, aggregate throughput
- Supports: OpenAI-compatible APIs, Anthropic, TogetherAI, HuggingFace, Vertex AI, LiteLLM
- Uses Shakespeare sonnets as test prompts
- Token counting via LlamaTokenizer for consistency

**What it does that we don't**:
- API endpoint testing (cloud LLM services)
- Multi-provider support (OpenAI, Anthropic, Together, etc.)
- Concurrent request load testing

**What we do that it doesn't**:
- Everything related to local inference measurement
- Energy measurement
- FLOPs estimation
- Backend orchestration
- Config management

**Relationship**: **Different scope**. LLMPerf benchmarks cloud API endpoints; we benchmark local inference. Also now archived/unmaintained.

---

### 1.5 vLLM Built-in Benchmarks

**Repository**: https://github.com/vllm-project/vllm (benchmarks/ directory)
**Stars**: ~52,000+ | **Status**: Active

**What it does**: vLLM ships with benchmark scripts in `benchmarks/`:
- `benchmark_serving.py` -- serving throughput/latency benchmark
- `benchmark_throughput.py` -- offline throughput benchmark
- `benchmark_latency.py` -- latency benchmark
- `benchmark_prefix_caching.py` -- prefix caching performance

**Key metrics**: Throughput (tokens/s), latency (TTFT, ITL, E2E), request throughput, concurrent request handling.

**What it does that we don't**:
- Native vLLM-specific benchmarks (prefix caching, disaggregated prefill)
- Built-in serving workload generation
- Tight integration with vLLM internal metrics

**What we do that it doesn't**:
- Energy measurement
- Cross-backend comparison
- FLOPs estimation
- Statistical aggregation with late aggregation pattern
- Config-driven experiments

**Relationship**: **Complementary**. We orchestrate vLLM as a backend; their scripts benchmark vLLM's serving layer specifically. We could align our vLLM metrics with theirs for comparability.

---

### 1.6 TensorRT-LLM Benchmarks

**Repository**: https://github.com/NVIDIA/TensorRT-LLM
**Stars**: ~12,893 | **Status**: Active

**What it does**: TensorRT-LLM includes its own benchmarking tools:
- `benchmarks/cpp/gptManagerBenchmark` -- C++ benchmark for throughput/latency
- Python benchmarking scripts for various scenarios
- Inflight batching benchmarks

**What it does that we don't**:
- C++ level benchmarking (lower overhead)
- Inflight batching specific metrics
- TRT-LLM internal profiling

**What we do that it doesn't**: Same as vLLM comparison above.

**Relationship**: **Complementary**. We orchestrate TensorRT-LLM as a backend.

---

### 1.7 MLPerf Inference (MLCommons)

**Repository**: https://github.com/mlcommons/inference
**Stars**: ~1,530 | **License**: Apache-2.0 | **Status**: Active (v6.0 in progress, deadline Feb 2026)

**What it does**: Industry-standard benchmark suite for ML inference performance. Now includes LLM benchmarks: Llama2-70B, Llama3.1-405B, Llama3.1-8B, Mixtral-8x7B, DeepSeek-R1.

**Key characteristics**:
- **Standardised scenarios**: SingleStream, MultiStream, Offline, Server
- **Power measurement**: Uses SPEC PTD (Power and Temperature Daemon) for hardware-level power measurement
- **Strict methodology**: Formal submission process, audited results
- **Categories**: Edge and Datacenter
- **Submission-based**: Hardware vendors submit results

**What it does that we don't**:
- Hardware-level power measurement (SPEC PTD with PDUs)
- Standardised, auditable benchmark methodology
- Industry-accepted credibility
- Multi-hardware comparison (different vendors submit)

**What we do that it doesn't**:
- Accessible to individual researchers (MLPerf requires significant infrastructure)
- Quick iteration (minutes vs days)
- Arbitrary model support (not limited to benchmark models)
- Configuration exploration (vary batch size, precision, etc.)
- Software-level energy estimation for any setup

**Relationship**: **Reference standard**. MLPerf is the "gold standard" that we should cite for methodological best practices. We cannot compete on rigour, but we fill the gap for researchers who need quick, accessible measurements.

---

## 2. Energy and Carbon Measurement Tools

### 2.1 Zeus (ML.ENERGY, Carnegie Mellon)

**Repository**: https://github.com/ml-energy/zeus
**Stars**: ~332 | **License**: Apache-2.0 | **Status**: Active (updated 2026-02-05)
**PyPI**: `zeus-ml` (v0.11.0)

**What it does**: GPU energy measurement and optimisation library. Pure Python library, no CLI.

**Key capabilities**:
- Direct NVML access (~10ms overhead, lower than CodeCarbon)
- CPU + DRAM energy via Intel RAPL
- AMD GPU support (ROCm SMI)
- Apple Silicon support (powermetrics)
- NVIDIA Jetson support
- CO2 emissions calculation
- **Batch size optimiser** (finds energy-optimal batch size)
- **Power limit optimiser** (finds optimal GPU power cap)
- Pareto frontier analysis (energy vs performance)

**Architecture**:
```python
from zeus.monitor import ZeusMonitor
monitor = ZeusMonitor(gpu_indices=[0, 1])
monitor.begin_window("inference")
# ... run inference ...
result = monitor.end_window("inference")
print(f"Energy: {result.total_energy} J")
```

**What it does that we don't**:
- Direct NVML (lower overhead than CodeCarbon wrapper)
- AMD GPU energy measurement
- Apple Silicon energy measurement
- Intel RAPL for CPU/DRAM energy
- Batch size optimisation
- Power limit optimisation
- Broader hardware support

**What we do that it doesn't**:
- LLM-specific metrics (TTFT, ITL, streaming latency)
- FLOPs estimation
- Inference backend orchestration
- Config-driven experiment management
- Extended efficiency metrics (TEI, tokens/joule, etc.)

**Relationship**: **Integration candidate -- highest priority**. Zeus could replace CodeCarbon as our energy backend for more accurate, lower-overhead measurements. This was already identified in the earlier RESEARCH-LANDSCAPE.md.

---

### 2.2 CodeCarbon

**Repository**: https://github.com/mlco2/codecarbon
**Stars**: ~1,200+ | **License**: MIT | **Status**: Active (v3.2.2)

**What it does**: Tracks carbon emissions from computing. Currently used as LLenergyMeasure's energy backend.

**Key capabilities**:
- Software estimation of CPU, GPU, RAM energy
- CO2 emissions based on regional grid intensity
- Dashboard for visualisation
- Minimal setup required

**Known limitations in our context**:
- Wraps pynvml (adds overhead vs direct NVML)
- Reports energy in kWh (we need joules, must convert)
- `duration_sec=0.0` bug (we already work around)
- Less accurate than direct NVML for GPU power
- No AMD GPU or Apple Silicon support
- No RAPL integration on all platforms

**Relationship**: **Current dependency**. Should be retained as fallback but Zeus should be primary energy backend for serious benchmarking.

---

### 2.3 Scaphandre

**Repository**: https://github.com/hubblo-org/scaphandre
**Stars**: ~1,900 | **License**: Apache-2.0 | **Status**: Active

**What it does**: Rust-based energy consumption metrology agent. Measures power at bare-metal host level. Designed for infrastructure monitoring, not ML workloads specifically.

**Key capabilities**:
- Bare metal and VM power measurement
- VM power attribution (qemu/kvm)
- Prometheus exporter
- Pushgateway, Riemann, Warp10, JSON output
- Works on GNU/Linux and Windows
- RAPL-based sensor (Intel)
- Kubernetes compatible

**What it does that we don't**:
- System-level power monitoring (not process-specific)
- VM power attribution
- Prometheus integration for continuous monitoring
- Rust performance (near-zero overhead)

**What we do that it doesn't**:
- Everything ML/LLM specific
- Process-level energy attribution
- GPU energy measurement (Scaphandre is CPU/RAPL focused)

**Relationship**: **Infrastructure tool, not directly applicable**. Useful for datacentre-level monitoring but not for per-inference energy measurement. Could complement our measurements for total system power validation.

---

### 2.4 Kepler (Kubernetes-based Efficient Power Level Exporter)

**Repository**: https://github.com/sustainable-computing-io/kepler
**Stars**: ~1,458 | **License**: Apache-2.0 | **Status**: Active (major v0.10.0 rewrite)

**What it does**: Prometheus exporter measuring energy at container, pod, and node level in Kubernetes clusters.

**Key capabilities**:
- Container-level energy attribution
- RAPL/powercap framework
- Kubernetes native (DaemonSet)
- Prometheus metrics export
- eBPF-based (v0.9.x), now simpler proc/sys approach (v0.10.0)

**What it does that we don't**:
- Container-level energy attribution in K8s
- Continuous monitoring (not one-shot measurement)
- Kubernetes-native deployment

**What we do that it doesn't**:
- Everything LLM/ML specific
- GPU energy measurement (Kepler v0.10 dropped GPU support temporarily)

**Relationship**: **Infrastructure tool, not directly applicable**. Relevant only if LLenergyMeasure experiments run in Kubernetes (future HPC/cloud use case).

---

### 2.5 PowerJoular

**Repository**: https://github.com/joular/powerjoular
**Stars**: ~100 | **License**: GPL-3.0 | **Status**: Active

**What it does**: Cross-platform power monitoring (Linux, Windows, macOS, Raspberry Pi). Written in Ada.

**Key capabilities**:
- Per-process power monitoring
- Multiple platforms (Intel RAPL, ARM, AMD)
- CSV output for analysis
- Lightweight

**Relevance**: Low. GPL-3.0 license is incompatible with our Apache-2.0. Limited GPU support. Mainly useful for CPU-focused measurements.

---

### 2.6 pyJoules

**Repository**: https://pypi.org/project/pyJoules/ (v0.5.1)
**License**: MIT | **Status**: Low maintenance

**What it does**: Python library for measuring energy consumption using RAPL. Context manager API.

```python
from pyJoules.energy_meter import EnergyMeter
with EnergyMeter() as meter:
    # ... computation ...
    pass
print(meter.result)
```

**Relevance**: Low. Limited to RAPL (CPU only). Zeus provides better coverage with similar API.

---

### 2.7 Experiment Impact Tracker

**Repository**: https://github.com/Breakend/experiment-impact-tracker
**PyPI**: v0.1.8 | **License**: MIT | **Status**: Low maintenance

**What it does**: Tracks energy, carbon, and compute metrics for ML experiments. Academic tool.

**Key capabilities**:
- Power consumption tracking
- Carbon emissions calculation
- GPU utilisation logging
- Regional carbon intensity

**Relevance**: Low. Academic prototype, not actively maintained. Superseded by CodeCarbon and Zeus.

---

### 2.8 NVIDIA DCGM (Data Center GPU Manager)

**Repository**: https://github.com/NVIDIA/DCGM (or via `nvidia-dcgm` package)
**Status**: Active (enterprise product)

**What it does**: NVIDIA's enterprise GPU monitoring and management tool. Provides detailed GPU telemetry including power, temperature, utilisation, ECC errors, and more.

**Key capabilities**:
- GPU power measurement (more accurate than NVML `nvmlDeviceGetPowerUsage`)
- Per-GPU and per-MIG-instance metrics
- Prometheus exporter (`dcgm-exporter`)
- Health checks and diagnostics
- Historical data collection
- Multi-GPU and multi-node support
- Integration with AIPerf for GPU telemetry

**What it does that we don't**:
- Higher-accuracy power measurement (hardware sensors with better calibration)
- MIG-instance-level power (NVML reports parent GPU power)
- Health monitoring and diagnostics
- Enterprise-grade reliability

**What we do that it doesn't**: Everything ML/LLM specific.

**Relationship**: **Integration candidate**. DCGM could be an alternative power measurement source, especially for MIG instances where NVML is insufficient. AIPerf already integrates with DCGM.

---

### 2.9 Intel RAPL (Running Average Power Limit)

Not a standalone tool but the underlying hardware interface used by Zeus, Scaphandre, Kepler, pyJoules, and others for CPU/DRAM energy measurement on Intel processors.

**Access methods**:
- `/sys/class/powercap/intel-rapl/` (Linux sysfs)
- MSR (Model-Specific Registers) -- requires elevated privileges
- perf subsystem

**Relevance**: We currently have no RAPL integration. Zeus provides this. If we adopt Zeus as energy backend, we get RAPL for free.

---

### 2.10 Green Algorithms

**Repository**: https://github.com/GreenAlgorithms/green-algorithms-tool
**Stars**: ~119 | **License**: CC-BY-4.0 | **Status**: Active

**What it does**: Web calculator for estimating carbon footprint of computational tasks. Input: hardware type, runtime, location. Output: CO2 estimate.

**Relevance**: Very low. Manual web calculator, not programmatic. Useful as a reference for carbon intensity data.

---

## 3. Deployment Configuration and Optimisation Tools

### 3.1 HuggingFace Optimum

**Repository**: https://github.com/huggingface/optimum
**Stars**: ~3,291 | **License**: Apache-2.0 | **Status**: Active

**What it does**: Hardware acceleration toolkit for HuggingFace models. Bridges HF Transformers with hardware-specific optimisation backends.

**Key capabilities**:
- ONNX Runtime optimisation and export
- OpenVINO integration
- Intel Neural Compressor integration
- BetterTransformer (built into Transformers now)
- Quantisation workflows (GPTQ, AWQ integration)

**Relationship**: **Complementary**. Optimum is about applying optimisations; we measure the impact of those optimisations. Users apply quantisation via Optimum, then benchmark with us.

---

### 3.2 AutoGPTQ

**Repository**: https://github.com/AutoGPTQ/AutoGPTQ
**Stars**: ~5,026 | **License**: MIT | **Status**: ARCHIVED

**What it does**: Easy-to-use LLM quantisation based on GPTQ algorithm. 4-bit quantisation.

**Note**: Archived/deprecated. GPTQ is now integrated into Transformers and vLLM natively.

---

### 3.3 AutoAWQ

**Repository**: https://github.com/casper-hansen/AutoAWQ
**Stars**: ~2,313 | **License**: MIT | **Status**: ARCHIVED

**What it does**: AWQ (Activation-aware Weight Quantization) for 4-bit quantisation with 2x inference speedup.

**Note**: Archived. AWQ support now integrated into Transformers and vLLM.

**Related**: The original AWQ paper/code (mit-han-lab/llm-awq, ~3,439 stars, MLSys 2024 Best Paper) remains active as a research reference.

---

### 3.4 BitsAndBytes

**Repository**: https://github.com/bitsandbytes-foundation/bitsandbytes
**Status**: Active | We already depend on this.

**Relevance**: Already integrated as our quantisation backend for PyTorch. Our FLOPs estimation correctly handles BNB's compute-at-FP16 behaviour.

---

### 3.5 Flash Attention (Dao-AILab)

**Repository**: https://github.com/Dao-AILab/flash-attention
**Stars**: ~22,270 | **License**: BSD-3 | **Status**: Active

**What it does**: Fast and memory-efficient exact attention. IO-aware algorithm that reduces memory from O(N^2) to O(N) and speeds up attention 2-4x.

**Relevance**: **Indirect**. Flash Attention is a kernel-level optimisation that our backends use (vLLM uses Flash Attention, PyTorch via `attn_implementation="flash_attention_2"`). We should document how to enable it and measure its impact, but we don't need to integrate with it directly.

---

### 3.6 SGLang

**Repository**: https://github.com/sgl-project/sglang
**Stars**: ~23,559 | **License**: Apache-2.0 | **Status**: Very active

**What it does**: High-performance serving framework for LLMs and multimodal models. Competitor/alternative to vLLM with novel optimisations (RadixAttention, compressed FSM for structured output).

**Key capabilities**:
- RadixAttention for automatic KV cache reuse
- Compressed FSM for structured generation
- High throughput (competitive with or faster than vLLM in some benchmarks)
- Multi-modal model support
- OpenAI-compatible API

**Relationship**: **Potential future backend**. SGLang is emerging as a serious vLLM competitor. Adding it as a 4th backend would be a significant differentiation for LLenergyMeasure.

---

### 3.7 llama.cpp / llamafile

**llama.cpp**: https://github.com/ggerganov/llama.cpp (50,000+ stars)
**llamafile**: https://github.com/mozilla/llamafile

**What they do**: CPU/GPU inference for LLMs using quantised GGUF models. llama.cpp is the de facto standard for local inference. llamafile packages models as single-file executables.

**Relevance**: **Potential future backend**. Many researchers use llama.cpp for local inference. Adding GGUF/llama.cpp support would reach a large user base, especially those on consumer hardware. Optimum-Benchmark already supports llama.cpp via llama-cpp-python bindings.

---

### 3.8 MLC LLM

**Repository**: https://github.com/mlc-ai/mlc-llm
**Stars**: ~22,048 | **License**: Apache-2.0 | **Status**: Active

**What it does**: Universal LLM deployment engine using ML compilation (TVM-based). Supports diverse hardware: NVIDIA, AMD, Apple, WebGPU.

**Relevance**: Low for now. Interesting as a future backend candidate for non-NVIDIA hardware measurement.

---

### 3.9 exo

**Repository**: https://github.com/exo-explore/exo
**Stars**: ~41,486 | **License**: GPL-3.0 | **Status**: Very active

**What it does**: Run frontier AI locally. Distributed inference across heterogeneous devices (different GPU types, even phones).

**Relevance**: Low. Different use case (distributed across consumer devices). GPL license incompatible.

---

## 4. Academic Tools and Frameworks

### 4.1 lm-evaluation-harness (EleutherAI)

**Repository**: https://github.com/EleutherAI/lm-evaluation-harness
**Stars**: ~11,434 | **License**: MIT | **Status**: Very active

**What it does**: Standard framework for evaluating LLM quality (accuracy on benchmarks like MMLU, HellaSwag, etc.). Not efficiency-focused.

**Key design patterns worth studying**:
- Task registry for extensible benchmarks
- Model wrapper abstraction (supports HF, vLLM, GGUF, API models)
- Only 3 top-level CLI commands
- Clean library-first architecture (`lm_eval.simple_evaluate()`)

**Relationship**: **Complementary**. Measures quality; we measure efficiency. An integration that reports both quality degradation and efficiency gain from quantisation would be extremely valuable.

**Integration opportunity**: High. We could run lm-eval quality benchmarks alongside efficiency benchmarks to produce quality-adjusted efficiency metrics.

---

### 4.2 HELM (Stanford CRFM)

**Repository**: https://github.com/stanford-crfm/helm
**Stars**: ~2,676 | **License**: Apache-2.0 | **Status**: Active

**What it does**: Holistic Evaluation of Language Models. Framework for reproducible, transparent evaluation across many dimensions: accuracy, robustness, fairness, bias, toxicity, and efficiency.

**Key characteristics**:
- Standardised benchmark format
- Unified model interface (supports API and local models)
- **Includes efficiency metrics** (inference time, but not energy)
- Web UI for inspection
- Multiple leaderboards (Capabilities, Safety, VHELM, MedHELM, etc.)

**Efficiency metrics in HELM**: Runtime and inference time are tracked but energy is not measured. This is a gap they acknowledge.

**Relationship**: **Complementary**. HELM's efficiency dimension could be enhanced by LLenergyMeasure's energy data. Academic credibility of HELM is very high.

---

### 4.3 Key Academic Papers and Associated Tools

#### Energy and AI

| Paper | Year | Tool/Contribution | Relevance |
|-------|------|-------------------|-----------|
| "Energy and Policy Considerations for Deep Learning in NLP" (Strubell et al.) | 2019 | Foundational paper on ML energy costs | High -- motivational reference |
| "Carbontracker: Tracking and Predicting the Carbon Footprint of Training DNNs" | 2020 | Carbontracker tool | Low -- training-focused |
| "Measuring the Carbon Intensity of AI in Cloud Instances" (Dodge et al.) | 2022 | Cloud carbon measurement methodology | Medium -- methodology reference |
| "Zeus: Understanding and Optimizing GPU Energy" (Li et al.) | 2023 | Zeus framework | High -- energy backend candidate |
| "Power Hungry Processing: Watts Driving the Cost of AI Deployment?" (Luccioni et al.) | 2023 | ML.ENERGY/HF energy research | High -- validates our approach |
| "Towards Sustainable AI: Environmental Implications of LLMs" (various) | 2024-25 | Survey papers | High -- contextual awareness |

#### Efficient Inference

| Paper | Year | Tool/Contribution | Relevance |
|-------|------|-------------------|-----------|
| "FlashAttention" (Dao et al.) | 2022 | Flash Attention kernel | Medium -- optimisation we measure |
| "AWQ: Activation-aware Weight Quantization" (Lin et al.) | 2024 | AWQ quantisation | Medium -- quantisation method |
| "GPTQ: Accurate Post-Training Quantization" (Frantar et al.) | 2023 | GPTQ quantisation | Medium -- quantisation method |
| "Efficient Memory Management for LLM Serving with PagedAttention" (Kwon et al.) | 2023 | vLLM/PagedAttention | Medium -- backend technology |
| "StreamingLLM" (Xiao et al.) | 2024 | Efficient long-context | Low -- specialised optimisation |
| "SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Compression" | 2023 | Mixed sparsity + quantization | Low -- research method |

#### Benchmarking Methodology

| Paper | Year | Contribution | Relevance |
|-------|------|-------------|-----------|
| "MLPerf Inference Benchmark" (Reddi et al.) | 2019 | Standardised ML benchmarking | High -- methodology reference |
| "Holistic Evaluation of Language Models" (Liang et al.) | 2022 | HELM framework | High -- holistic evaluation |
| "Chatbot Arena: An Open Platform for Evaluating LLMs" | 2024 | Human evaluation methodology | Low -- different dimension |

---

## 5. Industry Leaderboards

### 5.1 HuggingFace Open LLM Leaderboard

**URL**: https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard
**Focus**: Model quality (accuracy on benchmarks)
**Trust model**: Runs benchmarks on HF infrastructure
**Metrics**: MMLU, ARC, HellaSwag, TruthfulQA, Winogrande, GSM8K
**Energy**: NOT measured

**Relevance**: Reference for how leaderboards work. Does not compete with us.

---

### 5.2 ML.ENERGY Leaderboard

**URL**: https://ml.energy/leaderboard
**Focus**: LLM energy efficiency
**Trust model**: All benchmarks run on their own hardware (A40 GPUs)
**Metrics**: Energy per request (J), throughput (tokens/s), latency, response quality
**Models**: Open-source LLMs served via various inference engines

**THIS IS THE CLOSEST LEADERBOARD TO OUR VISION**.

**Key characteristics**:
- Measures energy using Zeus library
- Standardised hardware (A40)
- Tests with chat workloads
- Reports energy per request alongside quality
- Does NOT accept user submissions
- Maintained by CMU team

**What it covers that we don't**:
- Centralised, standardised hardware for fair comparison
- Chat-workload specific benchmarking
- Quality-alongside-efficiency presentation

**What we cover that it doesn't**:
- Multi-backend comparison (they use a single serving engine)
- User-configurable experiments
- FLOPs estimation
- Configuration exploration (what if I change batch size?)
- Local measurement (researchers can run on their own hardware)

**Relationship**: **Key reference and potential partner**. The ML.ENERGY leaderboard validates the market for energy-focused LLM benchmarking. Our tool enables researchers to run similar measurements locally. Future web platform could complement their centralised leaderboard with user-contributed data.

---

### 5.3 HuggingFace LLM-Perf Leaderboard

**URL**: https://huggingface.co/spaces/optimum/llm-perf-leaderboard
**Focus**: LLM inference performance (latency, throughput, memory)
**Trust model**: Runs on HF infrastructure (A100, H100)
**Powered by**: Optimum-Benchmark (see 1.1)
**Metrics**: Latency (ms), throughput (tokens/s), memory (GB), energy (Wh -- via CodeCarbon)
**Energy**: YES -- measured via CodeCarbon in Optimum-Benchmark

**Key insight**: This leaderboard DOES include energy, powered by the same CodeCarbon backend we use. However, their energy measurements have the same limitations as ours (CodeCarbon accuracy).

**Relationship**: **Competitive**. Uses Optimum-Benchmark (our primary competitor) to generate results. We need to differentiate on measurement accuracy (Zeus > CodeCarbon), deeper metrics (TTFT, ITL, FLOPs), and config exploration.

---

### 5.4 Artificial Analysis

**URL**: https://artificialanalysis.ai
**Focus**: LLM API performance and pricing comparison
**Model**: Tests cloud API endpoints, not local inference
**Metrics**: TTFT, throughput, total response time, pricing per token

**Relevance**: Different scope (API benchmarking). However, their presentation and UX for comparing models is excellent reference material for our future web platform.

---

### 5.5 Chatbot Arena (LMSYS)

**URL**: https://chat.lmsys.org
**Repository**: https://github.com/lm-sys/FastChat (~39,404 stars)
**Focus**: Human evaluation of LLM quality via blind comparison
**Metrics**: Elo ratings based on human preferences

**Relevance**: Low for efficiency. Different dimension (quality, not efficiency).

---

### 5.6 MLPerf Results

**URL**: https://mlcommons.org/benchmarks/inference-datacenter/
**Focus**: Hardware vendor performance comparison
**Trust model**: Formal submission and audit process
**Metrics**: Throughput, latency, power (via SPEC PTD)
**Energy**: YES -- hardware-measured via PDUs

**Relevance**: Gold standard for hardware comparison. Not accessible for individual researchers.

---

### 5.7 AI Energy Score (HuggingFace)

**URL**: https://huggingface.co/spaces/AIEnergyScore/AI_Energy_Score
**Focus**: Energy efficiency ratings for AI models (like EU energy labels A-G)
**Status**: Early/experimental

**Concept**: Assign letter grades (A-G) to AI models based on energy efficiency, similar to appliance energy ratings. Targets policy makers and public communication.

**Relevance**: **Highly relevant to our vision**. This aligns with our planned web platform targeting policy makers. We could contribute measurement data to this initiative or adopt their rating methodology.

---

## 6. LLM Inference Serving Engines

These are not measurement tools but the engines we orchestrate. Listed for completeness of the competitive landscape.

| Engine | Stars | Key Features | Backend in LLenergyMeasure |
|--------|-------|-------------|---------------------------|
| vLLM | ~52K+ | PagedAttention, continuous batching | Yes |
| TensorRT-LLM | ~12.9K | NVIDIA optimised, FP8/INT4 | Yes |
| SGLang | ~23.6K | RadixAttention, structured output | No (candidate) |
| TGI (HuggingFace) | ~10.8K | Production serving, Rust core | No |
| llama.cpp | ~50K+ | CPU/GPU, GGUF quantisation | No (candidate) |
| DeepSpeed-MII | ~? | Microsoft, ZeRO-Inference | No |
| LoRAX | ~3.7K | Multi-LoRA serving | No |
| OpenLLM (BentoML) | ~12.1K | BentoML ecosystem | No |
| MLC LLM | ~22K | Universal deployment, TVM | No |

---

## 7. Gap Analysis Summary

### 7.1 What exists in the landscape that we lack

| Capability | Available in | Priority for us | Effort |
|-----------|-------------|-----------------|--------|
| Direct NVML energy (bypassing CodeCarbon) | Zeus | **P0** | Medium (Zeus integration) |
| AMD GPU energy measurement | Zeus | P2 | Low (comes with Zeus) |
| Apple Silicon energy | Zeus | P2 | Low (comes with Zeus) |
| CPU/DRAM energy via RAPL | Zeus, pyJoules | P1 | Low (comes with Zeus) |
| DCGM power telemetry | AIPerf, NVIDIA DCGM | P2 | Medium |
| API endpoint benchmarking | AIPerf, LLMPerf | P3 | High |
| Load generation (concurrent requests) | AIPerf, vLLM benchmarks | P2 | Medium |
| llama.cpp backend | Optimum-Benchmark | P2 | Medium |
| SGLang backend | None (novel) | P2 | Medium |
| Quality-alongside-efficiency (lm-eval) | None (novel combination) | **P1** | Medium |
| Goodput (SLO-based throughput) | AIPerf | P2 | Low |
| Time-sliced metric analysis | AIPerf | P3 | Low |
| HuggingFace Hub results sharing | Optimum-Benchmark | P3 | Low |
| CPU-only benchmarking | Optimum-Benchmark | P3 | Low |

### 7.2 What we have that nobody else combines

| Unique Combination | Nearest Alternatives |
|-------------------|---------------------|
| Energy + LLM streaming latency (TTFT/ITL) | Zeus (energy only) + AIPerf (latency only) |
| Multi-backend energy comparison | Optimum-Benchmark (energy but shallow) |
| FLOPs estimation with precision normalisation | None |
| Baseline-adjusted energy per inference | None |
| Thermal throttle detection with measurement invalidation | None |
| Config-driven experiment management with SSOT introspection | None |
| Late aggregation with statistically correct multi-process metrics | None |
| Precision-normalised efficiency index (TEI) | None |

### 7.3 Competitive positioning matrix

```
                    Energy Depth
                    ^
                    |
            Zeus    |  LLenergyMeasure
            (lib)   |  (framework)
                    |
                    |         ML.ENERGY
                    |         (leaderboard)
    ----------------+--------------------------------> LLM Specificity
                    |
        CodeCarbon  |     Optimum-Benchmark
        (generic)   |     (multi-backend perf)
                    |
                    |     AIPerf            lm-eval
                    |     (serving perf)    (quality)
                    |
```

LLenergyMeasure occupies the upper-right quadrant: deep energy measurement combined with LLM-specific inference metrics. No other single tool occupies this space.

---

## 8. Strategic Recommendations

### 8.1 Highest-priority integration: Zeus as energy backend

**Rationale**: Zeus provides more accurate GPU energy (direct NVML), broader hardware support (AMD, Apple Silicon), and CPU/DRAM energy via RAPL -- all areas where CodeCarbon is weaker.

**Implementation approach**:
1. Add Zeus as a second energy backend alongside CodeCarbon
2. Auto-detect available energy backends (prefer Zeus, fallback to CodeCarbon)
3. Expose energy backend selection in config

**Estimated effort**: 2-3 days for basic integration.

### 8.2 Quality-alongside-efficiency (lm-eval integration)

**Rationale**: The single most impactful differentiation. No tool currently combines quality degradation measurement with efficiency gains. This is the core value proposition for researchers evaluating quantisation or optimisation tradeoffs.

**Use case**: "Quantising Llama-3-8B to INT4 improves energy efficiency by 2.3x while losing 1.2% accuracy on MMLU."

**Implementation approach**:
1. Optional lm-eval dependency
2. Run quality benchmarks before/after efficiency measurement
3. Report quality-adjusted efficiency metrics

**Estimated effort**: 5-7 days for a minimal integration.

### 8.3 Expand backend coverage strategically

**Priority backends to add**:
1. **SGLang** -- 23K+ stars, growing fast, novel optimisations. Would be a unique offering (no other benchmarking tool has SGLang + energy).
2. **llama.cpp** -- Massive user base (50K+ stars), consumer hardware focus. Optimum-Benchmark already supports it.

**Not recommended**: TGI, DeepSpeed-MII, OpenLLM (lower priority, smaller differentiation).

### 8.4 Learn from AIPerf's metrics

AIPerf has introduced several metrics we should consider:
- **Goodput**: Throughput of requests meeting SLO thresholds
- **Time-sliced metrics**: Detect warm-up effects and degradation over time
- **GPU telemetry via DCGM**: Higher accuracy power for datacenter GPUs

These could be added incrementally without major architectural changes.

### 8.5 Positioning relative to leaderboards

- **ML.ENERGY leaderboard**: Position ourselves as the "local measurement tool" that enables researchers to reproduce ML.ENERGY-style benchmarks on their own hardware
- **HF LLM-Perf leaderboard**: Differentiate on measurement depth (Zeus > CodeCarbon, TTFT/ITL, FLOPs)
- **AI Energy Score**: Align with their rating methodology for our web platform

### 8.6 Do NOT try to compete with

- **MLPerf**: Hardware-measured power with PDUs. Different rigour level entirely.
- **Chatbot Arena**: Human evaluation. Different dimension.
- **lm-eval-harness on quality**: Integrate with it, don't rebuild it.
- **vLLM/TRT-LLM on serving benchmarks**: Use AIPerf for that, focus on per-inference measurement.

---

## Appendix A: Tool Summary Table

| Tool | Type | Stars | Energy | LLM Metrics | Backend Support | Maintained | License |
|------|------|-------|--------|-------------|-----------------|------------|---------|
| **LLenergyMeasure** | Framework | -- | Yes (CodeCarbon) | TTFT, ITL, FLOPs | PyTorch, vLLM, TRT | Yes | -- |
| Optimum-Benchmark | Benchmark | 329 | Yes (CodeCarbon) | Latency, throughput | 8+ backends | Yes | Apache-2.0 |
| AIPerf | Benchmark | New | DCGM GPU telemetry | TTFT, ITL, goodput | API endpoints | Yes | Apache-2.0 |
| Zeus | Library | 332 | Yes (NVML, RAPL) | No | N/A (monitoring) | Yes | Apache-2.0 |
| CodeCarbon | Library | ~1.2K | Yes (estimated) | No | N/A (monitoring) | Yes | MIT |
| lm-eval-harness | Evaluation | 11.4K | No | No (quality only) | HF, vLLM, GGUF | Yes | MIT |
| HELM | Evaluation | 2.7K | No | Runtime only | API + local | Yes | Apache-2.0 |
| MLPerf Inference | Benchmark | 1.5K | Yes (SPEC PTD) | Latency, throughput | Hardware vendor | Yes | Apache-2.0 |
| Scaphandre | Monitoring | 1.9K | Yes (RAPL) | No | N/A (system) | Yes | Apache-2.0 |
| Kepler | Monitoring | 1.5K | Yes (RAPL) | No | N/A (K8s) | Yes | Apache-2.0 |
| LLMPerf | Benchmark | 1.1K | No | ITL, throughput | API endpoints | **Archived** | Apache-2.0 |
| GenAI-Perf | Benchmark | 134 | No | TTFT, ITL | Triton/TRT/vLLM | **Deprecated** | BSD-3 |

## Appendix B: Energy Measurement Method Comparison

| Method | Source | Accuracy | Overhead | GPU | CPU | DRAM | Platforms |
|--------|--------|----------|----------|-----|-----|------|-----------|
| SPEC PTD (PDU) | Hardware | +/- 1% | None | Yes | Yes | Yes | Any |
| NVML direct (Zeus) | Software/HW | +/- 5% | ~10ms | NVIDIA | No | No | Linux, Windows |
| CodeCarbon | Software | +/- 10-15% | ~50ms | NVIDIA | Partial | Partial | Cross-platform |
| Intel RAPL | Hardware/SW | +/- 5% | ~1ms | No | Intel | Intel | Linux |
| DCGM | Software/HW | +/- 3-5% | ~10ms | NVIDIA | No | No | Linux |
| Estimation (params) | Model | +/- 30-50% | None | N/A | N/A | N/A | Any |

**Recommendation order**: SPEC PTD (if available) > DCGM > Zeus/NVML > RAPL (CPU) > CodeCarbon > Estimation

## Appendix C: Leaderboard Comparison

| Leaderboard | Metrics | Energy | Hardware | Trust Model | Public |
|-------------|---------|--------|----------|-------------|--------|
| ML.ENERGY | Energy, latency, quality | **Yes (Zeus)** | A40 | Self-run | Yes |
| HF LLM-Perf | Latency, throughput, memory, energy | **Yes (CodeCarbon)** | A100, H100 | Self-run | Yes |
| Open LLM Leaderboard | Quality only | No | HF infra | Self-run | Yes |
| Artificial Analysis | TTFT, throughput, pricing | No | API testing | Self-run | Yes |
| Chatbot Arena | Human preference (Elo) | No | N/A | Crowdsourced | Yes |
| MLPerf | Latency, throughput, power | **Yes (SPEC PTD)** | Submitted | Audited | Yes |
| AI Energy Score | Energy rating (A-G) | **Yes** | TBD | TBD | Yes |
