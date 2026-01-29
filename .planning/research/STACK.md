# Technology Stack

**Project:** LLenergyMeasure Milestone - Precision Enhancement
**Researched:** 2026-01-29
**Confidence:** HIGH

## Executive Summary

For the precision enhancement milestone, the stack focuses on three areas: (1) NVML-based GPU monitoring for environment metadata and time-series power sampling, (2) Docker orchestration for campaign execution, and (3) backend parameter introspection for systematic completeness audits. The recommended stack prioritises official, actively maintained libraries with proven stability.

**Key Decision:** Use `nvidia-ml-py` (official NVIDIA bindings) over deprecated `pynvml`, and `python-on-whales` over `docker-py` for campaign orchestration due to superior feature support and CLI parity.

---

## Core Monitoring Stack

### NVML GPU Monitoring

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **nvidia-ml-py** | 13.590.48 | GPU power, thermal, memory, clock monitoring | Official NVIDIA bindings, actively maintained, released Jan 2026. Exposes nvmlDeviceGetTotalEnergyConsumption (Volta+), thermal state, power limits, clock frequencies, utilisation rates. |

**Confidence:** HIGH - Official NVIDIA package, latest version verified Jan 2026.

**Why nvidia-ml-py:**
- **Official vs deprecated:** `pynvml` package is deprecated as of 2025. NVIDIA officially recommends `nvidia-ml-py`. PyTorch 25.09 containers emit deprecation warnings for `pynvml`.
- **API completeness:** As of v11.0.0, the NVML wrappers in `pynvml` converged with `nvidia-ml-py`, making migration straightforward.
- **Current release:** Version 13.590.48 (Jan 2026) includes all NVML 13.590 functions, including power/thermal monitoring APIs required for this milestone.

**Installation:**
```bash
pip install nvidia-ml-py>=13.590.48
```

**Key NVML Functions for This Milestone:**

| Function | Purpose | Use Case |
|----------|---------|----------|
| `nvmlDeviceGetTotalEnergyConsumption()` | Total energy in mJ (Volta+) | Baseline measurement, high precision power tracking |
| `nvmlDeviceGetPowerUsage()` | Instantaneous power draw (mW) | Time-series sampling for pre-Volta GPUs |
| `nvmlDeviceGetPowerManagementLimit()` | Power limit (mW) | Environment metadata capture |
| `nvmlDeviceGetTemperature()` | GPU core temperature (C) | Thermal state monitoring |
| `nvmlDeviceGetClockInfo()` | Current clock frequencies | Detect thermal throttling |
| `nvmlDeviceGetUtilizationRates()` | GPU/memory utilisation (%) | Existing usage in `gpu_utilisation.py` |
| `nvmlDeviceGetDriverVersion()` | Driver version string | Environment metadata |
| `nvmlDeviceGetCudaComputeCapability()` | Compute capability (major, minor) | Environment metadata (already used in `gpu_info.py`) |

**Existing Integration:**
The codebase already uses `nvidia-ml-py` (imported as `pynvml`) in:
- `core/gpu_utilisation.py` - Background sampling via `GPUUtilisationSampler`
- `core/gpu_info.py` - GPU detection, MIG mode detection, compute capability

**Migration Status:** ✅ Already using `nvidia-ml-py` (pyproject.toml line 17: `nvidia-ml-py = ">=12.0.0"`). Comments reference "pynvml" as import alias for compatibility.

---

### Time-Series Power Sampling Architecture

**Pattern:** Background thread sampling with timestamps (already implemented in `gpu_utilisation.py`).

**Recommendation:** Extend existing `GPUUtilisationSampler` to capture power metrics.

```python
# Existing pattern in gpu_utilisation.py
class GPUUtilisationSampler:
    def _sample_loop(self):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        while not self._stop_event.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self._samples.append((time.time(), util.gpu, util.memory))
            time.sleep(self._sample_interval)

# Proposed extension for power sampling
class PowerSampler(GPUUtilisationSampler):
    def _sample_loop(self):
        # ... same pattern but capture:
        # - nvmlDeviceGetPowerUsage() for time-series
        # - nvmlDeviceGetTemperature() for thermal
        # - nvmlDeviceGetClockInfo() for throttling detection
```

**Why extend existing pattern:**
- Already thread-safe, context manager pattern
- Graceful NVML unavailability handling
- Timestamp alignment with inference metrics

---

### Baseline Power Subtraction Methodology

**Research Finding:** No authoritative research paper found specifying "baseline idle subtraction" methodology for NVML measurements. However, best practices from ML.ENERGY and GPU profiling communities converge on:

1. **Measurement Method (Volta+):** Use `nvmlDeviceGetTotalEnergyConsumption()` before/after inference, subtract for interval energy. This is more accurate than integrating `nvmlDeviceGetPowerUsage()`.

2. **Baseline Establishment:**
   - Measure idle power over 30-60 seconds with GPU in idle state
   - Close all GPU-accelerated applications (browsers, overlays, etc.)
   - Use median of samples to account for power state transitions
   - Expected idle ranges: 30-50W (high-end), 10-20W (mid-range)

3. **Subtraction Approach:**
   - **Conservative:** Subtract per-measurement baseline (idle power × duration)
   - **Aggressive:** Subtract idle energy from total energy reading
   - **Recommended:** Conservative approach to avoid negative energy for low-intensity workloads

**Confidence:** MEDIUM - Community best practices, not official NVIDIA methodology.

**Sources:**
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/)
- [GPU Idle Power Benchmark Guide](https://www.ywian.com/blog/gpu-idle-power-benchmark-fix-it-guide)
- [Part-time Power Measurements: nvidia-smi's Lack of Attention (arXiv)](https://arxiv.org/html/2312.02741v2)

**Implementation Notes:**
- nvidia-smi sampling period is 15ms with distortion (arxiv paper 2312.02741v2)
- Direct NVML calls avoid nvidia-smi sampling issues
- MIG instances report parent GPU power (cannot isolate per-instance)

---

## Docker Orchestration Stack

### Campaign Container Execution

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **python-on-whales** | Latest (0.70+) | Docker Compose exec, container orchestration | CLI parity with docker/docker-compose, supports docker compose exec, simpler API, thread-safe, actively maintained (696 stars, MIT license). |

**Confidence:** HIGH - Official Docker blog endorsement, widespread adoption.

**Why python-on-whales:**

| Criterion | docker-py | python-on-whales | Winner |
|-----------|-----------|------------------|---------|
| **Architecture** | Re-implements Docker Engine API in Python | Wraps Docker CLI via subprocess | python-on-whales |
| **Feature parity** | Lags behind Docker CLI (no native Buildx) | 1-to-1 CLI mapping, immediate feature availability | python-on-whales |
| **API ergonomics** | Low-level, Engine API-focused | Pythonic, mirrors CLI structure | python-on-whales |
| **docker compose exec** | Limited support, requires manual container.exec_run() | Native `docker.compose.exec()` | python-on-whales |
| **Maintenance burden** | High (full API re-implementation) | Low (CLI wrapper) | python-on-whales |
| **Thread safety** | Stateful client, not guaranteed | Stateless, explicitly thread-safe | python-on-whales |
| **Dependencies** | Only Docker Engine API | Requires Docker/Podman CLI installed | docker-py |

**Decision Rationale:**
1. **docker compose exec support:** Campaign orchestrator needs to run commands inside running containers. `python-on-whales` provides `docker.compose.exec(service, command)` matching CLI behaviour.
2. **Future-proof:** CLI wrapper means new Docker features available immediately without library updates.
3. **Developer experience:** Code reads like CLI commands: `docker.compose.up(detach=True)` vs docker-py's verbose Engine API calls.
4. **Docker endorsement:** Featured in official Docker blog (2021), indicating NVIDIA/Docker ecosystem acceptance.

**When to use docker-py:** Headless environments without Docker CLI, pure API access required. NOT applicable for this project (Docker containers available in dev/prod).

**Alternative considered:** Direct `subprocess` calls. **Why not:** Error handling, output parsing, cross-platform compatibility burden. python-on-whales abstracts this.

**Installation:**
```bash
pip install python-on-whales>=0.70
```

**Usage Pattern for Campaign Orchestration:**
```python
from python_on_whales import docker

# Start backend container in detached mode
docker.compose.up(["pytorch"], detach=True)

# Execute experiment inside container
result = docker.compose.exec(
    service="pytorch",
    command=["lem", "experiment", "/app/configs/config.yaml"],
    tty=False,
    detach=False
)

# Collect results from container filesystem
docker.compose.cp("pytorch:/app/results", "./local_results")

# Stop container
docker.compose.down()
```

**Confidence:** HIGH - Matches project requirements, Docker ecosystem standard.

**Sources:**
- [Docker Official Blog: Python-on-whales](https://www.docker.com/blog/guest-post-calling-the-docker-cli-from-python-with-python-on-whales/)
- [python-on-whales GitHub](https://github.com/gabrieldemarmiesse/python-on-whales)
- [docker-py GitHub](https://github.com/docker/docker-py)

---

## Backend Parameter Introspection

### PyTorch / HuggingFace Transformers

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **transformers** | 5.0+ | `generate()` method parameters | Already dependency (pyproject.toml), official HF library, comprehensive parameter set. |

**Confidence:** HIGH - Official HuggingFace library, verified documentation.

**Key `generate()` Parameters (Partial - Full List ~50+ Parameters):**

**Length Control:**
- `max_length` (int) - Maximum total sequence length
- `max_new_tokens` (int) - Maximum tokens to generate (recommended over max_length)
- `min_length` (int) - Minimum total sequence length
- `min_new_tokens` (int) - Minimum tokens to generate

**Sampling Strategy:**
- `do_sample` (bool) - Enable sampling vs greedy decoding
- `temperature` (float) - Sampling temperature
- `top_k` (int) - Top-k sampling
- `top_p` (float) - Nucleus sampling
- `num_beams` (int) - Beam search width
- `early_stopping` (bool | str) - Beam search stopping condition

**Stopping Conditions:**
- `eos_token_id` (int | list[int]) - End-of-sequence token(s)
- `pad_token_id` (int) - Padding token
- `stop_strings` (str | list[str]) - String-based stopping
- `max_time` (float) - Maximum generation time (seconds)

**Output Control:**
- `return_dict_in_generate` (bool) - Return GenerateOutput vs tensor
- `output_scores` (bool) - Return token scores
- `output_attentions` (bool) - Return attention weights
- `output_hidden_states` (bool) - Return hidden states

**Advanced Features:**
- `assistant_model` (PreTrainedModel) - Speculative decoding
- `prompt_lookup_num_tokens` (int) - Prompt lookup decoding
- `streamer` (BaseStreamer) - Token streaming callback
- `repetition_penalty` (float) - Penalise repetition
- `no_repeat_ngram_size` (int) - Block n-gram repetition
- `encoder_repetition_penalty` (float) - Encoder repetition penalty

**Introspection Method:**
Use `GenerationConfig` class to enumerate all parameters:
```python
from transformers import GenerationConfig
config = GenerationConfig()
params = {k: v for k, v in config.to_dict().items()}
```

**Confidence:** HIGH - Official HuggingFace Transformers v5.0 documentation verified.

**Source:** [HuggingFace Transformers: Text Generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation)

---

### vLLM Backend

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **vllm** | 0.8.1+ | `LLM` class constructor parameters | Already dependency (pyproject.toml), official vLLM library. |

**Confidence:** HIGH - Official vLLM documentation verified.

**Key `LLM()` Constructor Parameters (25+ Parameters):**

**Model Loading:**
- `model` (str) - Model name/path **(required)**
- `tokenizer` (str | Path) - Tokenizer name/path
- `tokenizer_mode` (Literal['auto', 'slow']) - Tokenizer selection
- `trust_remote_code` (bool) - Allow remote code execution
- `revision` (str) - Model version (branch/tag/commit)
- `tokenizer_revision` (str) - Tokenizer version

**Parallelism:**
- `tensor_parallel_size` (int) - Number of GPUs for tensor parallelism
- `dtype` (str) - Data type (float32, float16, bfloat16, auto)
- `quantization` (str) - Quantization method (awq, gptq, fp8, etc.)

**Memory Management:**
- `gpu_memory_utilization` (float) - GPU memory ratio (0-1, default 0.9)
- `swap_space` (int) - CPU swap space (GiB)
- `cpu_offload_gb` (int) - CPU memory for weight offloading

**Performance Optimisation:**
- `enforce_eager` (bool) - Force eager execution (disable CUDA graphs)
- `max_seq_len_to_capture` (int) - Max sequence length for CUDA graphs
- `disable_custom_all_reduce` (bool) - Disable custom all-reduce
- `disable_async_output_proc` (bool) - Disable async output processing

**Generation Control:**
- `seed` (int) - Random seed for sampling
- `task` (str) - Task type (auto, generate, embedding, etc.)

**Advanced Configuration:**
- `compilation_config` (CompilationConfig) - Compilation optimisations
- `hf_overrides` (dict) - HuggingFace config modifications
- `mm_processor_kwargs` (dict) - Multimodal processor arguments
- `override_pooler_config` (PoolerConfig) - Custom pooler config
- `allowed_local_media_path` (str) - Directory access for local media

**Introspection Method:**
```python
from vllm import LLM
import inspect
sig = inspect.signature(LLM.__init__)
params = {k: v for k, v in sig.parameters.items() if k != 'self'}
```

**Confidence:** HIGH - vLLM v0.8.1 documentation verified.

**Source:** [vLLM LLM Class API](https://docs.vllm.ai/en/v0.8.1/api/offline_inference/llm.html)

---

### TensorRT-LLM Backend

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| **tensorrt-llm** | 0.12.0+ | `LLM` class configuration parameters | Already optional dependency (pyproject.toml), official NVIDIA library. |

**Confidence:** HIGH - Official NVIDIA TensorRT-LLM documentation verified.

**Key `LLM()` Constructor Parameters (60+ Parameters):**

**Core Model:**
- `model` (str | Path) - Model checkpoint path/HuggingFace name **(required)**
- `tokenizer` (str | Path | TokenizerBase) - Tokenizer
- `tokenizer_mode` (Literal['auto', 'slow']) - Tokenizer selection
- `dtype` (str) - Data type (auto, float16, bfloat16, etc.)
- `trust_remote_code` (bool) - Allow remote code
- `revision` (str) - Model version

**Parallelism (Comprehensive):**
- `tensor_parallel_size` (int, default 1) - Tensor parallelism
- `pipeline_parallel_size` (int, default 1) - Pipeline parallelism
- `context_parallel_size` (int, default 1) - Context parallelism
- `gpus_per_node` (int) - GPUs per node
- `moe_tensor_parallel_size` (int) - MoE tensor parallelism
- `moe_expert_parallel_size` (int) - MoE expert parallelism
- `moe_cluster_parallel_size` (int) - MoE cluster parallelism

**Resource Constraints:**
- `max_batch_size` (int) - Maximum batch size
- `max_input_len` (int) - Maximum input length
- `max_seq_len` (int) - Maximum sequence length
- `max_beam_width` (int) - Maximum beam width
- `max_num_tokens` (int, default 8192) - Maximum tokens

**Cache & Memory:**
- `kv_cache_config` (KvCacheConfig) - KV cache configuration
- `peft_cache_config` (PeftCacheConfig) - PEFT cache configuration
- `enable_chunked_prefill` (bool) - Chunked prefill for long sequences
- `garbage_collection_gen0_threshold` (int, default 20000) - GC threshold

**Advanced Features:**
- `enable_lora` (bool) - Enable LoRA adapters
- `lora_config` (LoraConfig) - LoRA configuration
- `guided_decoding_backend` (Literal['xgrammar', 'llguidance']) - Guided decoding
- `speculative_config` (SpeculativeConfig) - Speculative decoding
- `sparse_attention_config` (SparseAttentionConfig) - Sparse attention

**Performance & Optimisation:**
- `cuda_graph_config` (CudaGraphConfig) - CUDA graph optimisation
- `enable_autotuner` (bool, default True) - Enable autotuning
- `attn_backend` (str, default 'TRTLLM') - Attention backend
- `sampler_type` (str | SamplerType, default 'auto') - Sampler type
- `torch_compile_config` (TorchCompileConfig) - torch.compile settings

**Scheduling & Batching:**
- `scheduler_config` (SchedulerConfig) - Scheduler configuration
- `batch_wait_timeout_ms` (float) - Batch wait timeout (ms)
- `batch_wait_timeout_iters` (int) - Batch wait timeout (iterations)
- `batch_wait_max_tokens_ratio` (float) - Batch tokens ratio
- `stream_interval` (int, default 1) - Streaming interval

**Monitoring:**
- `return_perf_metrics` (bool) - Return performance metrics
- `enable_iter_perf_stats` (bool) - Iteration perf stats
- `enable_iter_req_stats` (bool) - Iteration request stats
- `print_iter_log` (bool) - Print iteration logs
- `otlp_traces_endpoint` (str) - OpenTelemetry traces endpoint

**Distributed:**
- `orchestrator_type` (Literal['rpc', 'ray']) - Distributed orchestrator
- `checkpoint_loader` (BaseCheckpointLoader) - Custom checkpoint loader
- `checkpoint_format` (str) - Checkpoint format
- `allreduce_strategy` (str) - AllReduce strategy

**Stability Labels:** Parameters marked as `stable`, `prototype`, `beta`, or `deprecated` in documentation.

**Introspection Method:**
```python
from tensorrt_llm import LLM
import inspect
sig = inspect.signature(LLM.__init__)
params = {k: v for k, v in sig.parameters.items()}
```

**Confidence:** HIGH - TensorRT-LLM documentation verified.

**Source:** [TensorRT-LLM API Reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html)

---

## Supporting Libraries

### Environment Metadata Capture

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **psutil** | 6.1+ | CPU, memory stats | Already used in `compute_metrics.py` |
| **platform** | stdlib | OS, Python version | Environment metadata |
| **torch.cuda** | via torch | CUDA version, device properties | Environment metadata (already used) |

**Installation:** psutil already dependency. `platform` is stdlib.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| **NVML Bindings** | nvidia-ml-py | pynvml | Deprecated, PyTorch emits warnings |
| **NVML Bindings** | nvidia-ml-py | py3nvml | Unmaintained, last update 2019 |
| **Docker SDK** | python-on-whales | docker-py | Missing docker compose exec, Buildx support |
| **Docker SDK** | python-on-whales | subprocess | Error handling burden, cross-platform issues |
| **Power Sampling** | Extend GPUUtilisationSampler | New thread implementation | Code duplication, inconsistent patterns |
| **Baseline Method** | Conservative subtraction | Aggressive subtraction | Avoids negative energy readings |

---

## Installation Guide

### Core Dependencies (Add to pyproject.toml)

```toml
[tool.poetry.dependencies]
# Existing: nvidia-ml-py = ">=12.0.0" → Bump to 13.590.48
nvidia-ml-py = ">=13.590.48"

# New: Docker orchestration
python-on-whales = ">=0.70.0"

# Existing: psutil, torch, transformers, vllm, tensorrt-llm already present
```

### Installation Commands

```bash
# Development environment
poetry install --with dev

# Add python-on-whales
poetry add python-on-whales

# Update nvidia-ml-py
poetry add nvidia-ml-py@latest

# Verify installation
python -c "import pynvml; pynvml.nvmlInit(); print(pynvml.nvmlSystemGetDriverVersion())"
python -c "from python_on_whales import docker; print(docker.version())"
```

---

## Integration Notes

### NVML Thread Safety
- `nvmlInit()` and `nvmlShutdown()` are NOT thread-safe
- Device handle queries ARE thread-safe after initialisation
- **Pattern:** Call `nvmlInit()` once per thread, `nvmlShutdown()` in cleanup

### Docker Container NVML Access
- Requires `--privileged` or `--gpus all` flag
- MIG instances report parent GPU power (limitation documented)
- CodeCarbon already handles NVML permission errors (see `energy_backends/codecarbon.py:102`)

### vLLM/TensorRT CUDA Management
- Backends with `cuda_management=BACKEND` prohibit `torch.cuda.*` calls before `initialize()`
- NVML operates independently of CUDA context
- **Safe:** NVML queries can occur before backend initialisation

---

## Version Compatibility Matrix

| Component | Minimum Version | Tested Version | Notes |
|-----------|----------------|----------------|-------|
| Python | 3.10 | 3.12 | Project constraint |
| nvidia-ml-py | 12.0.0 | 13.590.48 | Jan 2026 release |
| python-on-whales | 0.70.0 | Latest | Active maintenance |
| CUDA Driver | 12.x | 12.6 | For TensorRT-LLM |
| Docker | 20.10+ | 27.x | Compose V2 support |
| transformers | 4.49.0 | 5.0.0 | Already dependency |
| vllm | 0.6.0 | 0.8.1+ | Already dependency |
| tensorrt-llm | 0.12.0 | Latest | Already optional dependency |

---

## Sources

**NVML Monitoring:**
- [nvidia-ml-py PyPI](https://pypi.org/project/nvidia-ml-py/) - HIGH confidence
- [NVML API Reference (Jan 2026)](https://docs.nvidia.com/deploy/pdf/NVML_API_Reference_Guide.pdf) - HIGH confidence
- [pynvml deprecation issue](https://github.com/deepbeepmeep/Wan2GP/issues/925) - MEDIUM confidence

**Power Measurement Methodology:**
- [ML.ENERGY: Measuring GPU Energy Best Practices](https://ml.energy/blog/energy/measurement/measuring-gpu-energy-best-practices/) - MEDIUM confidence
- [GPU Idle Power Benchmark Guide](https://www.ywian.com/blog/gpu-idle-power-benchmark-fix-it-guide) - LOW confidence
- [Part-time Power Measurements (arXiv)](https://arxiv.org/html/2312.02741v2) - HIGH confidence

**Docker Orchestration:**
- [Docker Blog: Python-on-whales](https://www.docker.com/blog/guest-post-calling-the-docker-cli-from-python-with-python-on-whales/) - HIGH confidence
- [python-on-whales GitHub](https://github.com/gabrieldemarmiesse/python-on-whales) - HIGH confidence
- [docker-py GitHub](https://github.com/docker/docker-py) - HIGH confidence

**Backend APIs:**
- [HuggingFace Transformers: Text Generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation) - HIGH confidence
- [vLLM LLM Class API](https://docs.vllm.ai/en/v0.8.1/api/offline_inference/llm.html) - HIGH confidence
- [TensorRT-LLM API Reference](https://nvidia.github.io/TensorRT-LLM/llm-api/reference.html) - HIGH confidence

---

## Confidence Assessment

| Technology | Confidence | Rationale |
|------------|------------|-----------|
| nvidia-ml-py | HIGH | Official NVIDIA package, Jan 2026 release verified |
| python-on-whales | HIGH | Docker blog endorsement, widespread adoption, active maintenance |
| Baseline power methodology | MEDIUM | Community best practices, no official NVIDIA spec |
| vLLM parameter list | HIGH | Official documentation verified |
| TensorRT-LLM parameter list | HIGH | Official NVIDIA documentation verified |
| HuggingFace generate() | HIGH | Official documentation verified |

---

## Open Questions for Implementation

1. **Baseline power measurement:** Should idle measurement be per-experiment or per-session? (Recommendation: per-experiment for reproducibility)
2. **Power sampling frequency:** What interval balances overhead vs resolution? (Recommendation: 100ms based on existing `gpu_utilisation.py` pattern)
3. **MIG power isolation:** Accept parent GPU power as limitation or implement estimation heuristic? (Recommendation: document limitation, no estimation)
4. **Campaign result aggregation:** Store per-experiment results in container filesystem or stream to host? (Recommendation: volume mount for real-time access)
