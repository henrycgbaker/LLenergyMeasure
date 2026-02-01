# Inference Backends

LLenergyMeasure supports multiple inference backends, each optimised for different use cases. This guide covers backend selection, configuration mapping, and compatibility.

## Backend Overview

| Backend | Status | Use Case | Key Features |
|---------|--------|----------|--------------|
| `pytorch` | **Default** | Research, experimentation | HuggingFace Transformers + Accelerate |
| `vllm` | Available | Production, throughput testing | PagedAttention, continuous batching, native TP |
| `tensorrt` | Available | Enterprise, maximum efficiency | Compiled inference, TensorRT-LLM, FP8/INT8/INT4 |

## Checking Your Environment

Use `lem doctor` to verify your environment setup:

```bash
lem doctor
```

**Sample output:**

```
=== LLenergyMeasure Environment Check ===

✓ Python: 3.10.12
✓ CUDA available: True (12.4)
✓ GPU count: 2
  - GPU 0: NVIDIA A100-PCIE-40GB
  - GPU 1: NVIDIA A100-PCIE-40GB

Backends:
✓ pytorch: Available (included in base install)
✗ vllm: Not installed (pip install llenergymeasure[vllm])
✗ tensorrt: Not installed (pip install llenergymeasure[tensorrt])

Docker:
✓ Docker available: 24.0.7
✓ Docker Compose available: 2.21.0
✓ nvidia-container-toolkit: installed

=== Environment OK ===
```

This command checks:
- Python version and CUDA availability
- GPU detection and properties
- Backend availability (PyTorch, vLLM, TensorRT)
- Docker setup for multi-backend campaigns

## Choosing a Backend

```
                         ┌─────────────────────────────────┐
                         │   Which backend should I use?   │
                         └──────────────┬──────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
           Need maximum         Production            Research &
           efficiency?          serving?              flexibility?
                │                   │                     │
                ▼                   ▼                     ▼
          Use tensorrt          Use vllm             Use pytorch
         (compiled plans,    (continuous batching,  (HuggingFace,
          FP8/INT8/INT4,      PagedAttention)       full control)
          inflight batching)
```

**Use `pytorch` (default) when:**
- Experimenting with different models
- Need full control over decoder parameters
- Using BitsAndBytes 4-bit/8-bit quantization (PyTorch only)
- Research and development workflows

**Use `vllm` when:**
- Measuring production serving performance
- Need high throughput with continuous batching
- Using tensor parallelism for large models
- Comparing optimised vs naive inference

**Use `tensorrt` when:**
- Maximum inference efficiency is critical
- Using Hopper+ GPUs with FP8 quantization
- Need compiled, optimised execution plans
- Enterprise/production deployments requiring peak performance
- Benchmarking with INT8/INT4 quantization

## Feature Support Matrix

Quick reference for feature availability across backends.

### Core Capabilities

| Feature | PyTorch | vLLM | TensorRT | Notes |
|---------|:-------:|:----:|:--------:|-------|
| **Inference Modes** |
| Batch inference | ✓ | ✓ | ✓ | All support batched generation |
| Streaming inference | ✓ | ✓ | ✓ | TTFT/ITL metrics |
| Continuous batching | ✗ | ✓ | ✓ | vLLM/TRT manage batches internally |
| **Parallelism** |
| Tensor parallelism | ✓ | ✓ | ✓ | PyTorch via Accelerate |
| Pipeline parallelism | ✗ | ✓ | ✓ | vLLM (0.6+) and TensorRT |
| **Quantisation** |
| BitsAndBytes 4-bit | ✓ | ✗ | ✗ | PyTorch only (use vLLM AWQ/GPTQ, TRT INT4) |
| BitsAndBytes 8-bit | ✓ | ✗ | ✗ | PyTorch only |
| FP8 | ✗ | ✓ | ✓ | Hopper+ GPUs (sm_90+) |
| INT8 (smooth quant) | ✗ | ✗ | ✓ | Requires calibration |
| GPTQ | ✓ | ✓ | ✓ | Pre-quantised checkpoints |
| AWQ | ✓ | ✓ | ✓ | Pre-quantised checkpoints |
| **Optimisations** |
| KV caching | ✓ | ✓ | ✓ | All support by default |
| Prefix caching | ✗ | ✓ | ✓ | TensorRT via enable_kv_cache_reuse |
| PagedAttention | ✗ | ✓ | ✓ | Memory-efficient KV |
| CUDA graphs | ✓ | ✓ | ✓ | `torch_compile`, automatic |
| Flash Attention | ✓ | ✓ | ✓ | Different implementations |
| **Speculative Decoding** |
| Draft model | ✓ | ✓ | ✓ | Different APIs |
| N-gram speculation | ✗ | ✓ | ✗ | vLLM only |
| **Adapters** |
| LoRA (single) | ✓ | ✓ | ✗ | TRT requires merged weights |
| Multi-LoRA | ✗ | ✓ | ✗ | vLLM native support |

### Decoder Parameters

| Parameter | PyTorch | vLLM | TensorRT | Notes |
|-----------|:-------:|:----:|:--------:|-------|
| `temperature` | ✓ | ✓ | ✓ | Universal |
| `top_p` | ✓ | ✓ | ✓ | Nucleus sampling |
| `top_k` | ✓ | ✓ | ✓ | Universal (in decoder config, 0=off) |
| `min_p` | ✓ | ✓ | ✗ | Use `top_p`/`top_k` instead |
| `repetition_penalty` | ✓ | ✓ | ✓ | Universal |
| `no_repeat_ngram_size` | ✓ | ✗ | ✗ | PyTorch only |
| `do_sample` | ✓ | — | — | Implicit via temperature=0 |
| `beam_search` | ✓ | ✓ | ✓ | `num_beams > 1` |

### Batching Strategies

| Strategy | PyTorch | vLLM | TensorRT | Notes |
|----------|:-------:|:----:|:--------:|-------|
| `static` | ✓ | — | — | Fixed batch size |
| `dynamic` | ✓ | — | — | Token-aware batching |
| `sorted_static` | ✓ | — | — | Length-sorted static |
| `sorted_dynamic` | ✓ | — | — | Length-sorted dynamic |
| Continuous batching | — | ✓ | ✓ | Automatic, always-on |

**Legend:** ✓ = Supported, ✗ = Not supported, — = Not applicable (managed differently)

### Latency Measurement Modes

| Mode | PyTorch | vLLM | TensorRT | Description |
|------|:-------:|:----:|:--------:|-------------|
| `TRUE_STREAMING` | ✓ | ✓ | ✓* | Actual per-token timestamps |
| `PER_REQUEST_BATCH` | ✓ | — | — | Batch mode with per-request timing |
| `PROPORTIONAL_ESTIMATE` | ✓ | — | ✓* | Estimated ITL from total time |

*TensorRT v0.9+ supports true streaming; older versions use estimation.

### Runtime & Architecture

| Property | PyTorch | vLLM | TensorRT |
|----------|---------|------|----------|
| Launch mode | Accelerate | Direct | Direct |
| CUDA management | Orchestrator | Backend | Backend |
| Process model | Single/Multi | Multiprocess (spawn) | Multiprocess |
| Manages own batching | No | Yes | Yes |

**Implications:**
- **PyTorch**: Orchestration layer can call `torch.cuda.*` before backend init
- **vLLM/TensorRT**: Backend manages CUDA; no `torch.cuda.*` calls before `initialize()`

## Configuration

### Installing Backends

**PyTorch (included in base install):**
```bash
pip install -e .
```

**vLLM (optional):**
```bash
pip install -e ".[vllm]"
```

**TensorRT (optional):**
```bash
pip install -e ".[tensorrt]"
```

Check installation: `lem doctor`

### Selecting a Backend

```yaml
# In config.yaml
backend: vllm  # or 'pytorch' (default)
```

```bash
# Or via CLI
lem experiment config.yaml --backend vllm
```

### Backend-Specific Docker Images

```bash
# PyTorch backend (default)
docker compose run --rm pytorch lem experiment ...

# vLLM backend
docker compose run --rm vllm lem experiment ... --backend vllm
```

## Backend-Native Configuration Architecture

This tool uses a **backend-native configuration architecture** with two tiers:

- **Tier 1 (Universal)**: Parameters at the top level with identical semantics across all backends
- **Tier 2 (Backend-Native)**: Parameters in backend sections (`pytorch:`, `vllm:`, `tensorrt:`) using native API names

### Tier 1: Universal Parameters

These work identically across all backends:

| Parameter | Config Path | Description |
|-----------|-------------|-------------|
| `model_name` | `model_name` | HuggingFace model ID |
| `fp_precision` | `fp_precision` | `float16`, `bfloat16`, `float32` |
| `max_output_tokens` | `max_output_tokens` | Maximum generated tokens |
| `min_output_tokens` | `min_output_tokens` | Minimum generated tokens |
| `random_seed` | `random_seed` | Reproducibility seed |
| `temperature` | `decoder.temperature` | Sampling temperature |
| `top_k` | `decoder.top_k` | Top-k sampling (0 = disabled) |
| `top_p` | `decoder.top_p` | Nucleus sampling threshold |
| `repetition_penalty` | `decoder.repetition_penalty` | Repetition penalty |
| `do_sample` | `decoder.do_sample` | Enable sampling |
| `traffic_simulation` | `traffic_simulation.*` | MLPerf-style Poisson arrivals |

### Tier 2: Backend-Native Parameters

These parameters have different semantics per backend and use native API naming:

| Capability | PyTorch Location | vLLM Location | TensorRT Location |
|------------|------------------|---------------|-------------------|
| **Batching** | `pytorch.batch_size`, `pytorch.batching_strategy` | `vllm.max_num_seqs` (continuous) | `tensorrt.max_batch_size` (compile-time) |
| **Parallelism** | `pytorch.parallelism_strategy/degree` | `vllm.tensor_parallel_size/pipeline_parallel_size` | `tensorrt.tp_size/pp_size` |
| **Quantisation** | `pytorch.load_in_4bit/8bit` (BitsAndBytes) | `vllm.quantization` (awq, gptq, fp8) | `tensorrt.quantization` (fp8, int8_sq) |
| **Top-k sampling** | `decoder.top_k` (0=off) | `decoder.top_k` (0=off, converted to -1) | `decoder.top_k` (0=off) |
| **Min-p sampling** | `pytorch.min_p` | `vllm.min_p` | Not supported |
| **N-gram blocking** | `pytorch.no_repeat_ngram_size` | Not supported | Not supported |
| **Beam search** | `pytorch.beam_search.*` | Not supported | Limited support |

### Key Semantic Differences

**Batching:**
- **PyTorch**: Application-level batching via `pytorch.batch_size` and `pytorch.batching_strategy`
- **vLLM**: Continuous batching via `vllm.max_num_seqs` (backend manages internally)
- **TensorRT**: Compile-time max via `tensorrt.max_batch_size`, runtime uses inflight batching

**Top-k sampling:**
- **All backends**: `decoder.top_k=0` disables (universal parameter)
- **vLLM internally**: Converts 0 to -1 (vLLM's native disabled convention)
- **PyTorch/TensorRT**: Use 0 directly (native disabled convention)

**Quantisation:**
- **PyTorch**: BitsAndBytes (`pytorch.load_in_4bit`, `pytorch.load_in_8bit`)
- **vLLM**: Native methods (`vllm.quantization`: awq, gptq, fp8, marlin)
- **TensorRT**: Build-time methods (`tensorrt.quantization`: fp8, int8_sq, int4_awq)

## Backend-Specific Configuration

Each backend exposes its own configuration section for advanced optimisation. These are set via `vllm:` or `pytorch:` blocks in YAML.

### vLLM Configuration (`vllm:`)

vLLM uses **backend-native parameter names** that map directly to `vLLM LLM()` constructor and `SamplingParams`.

```yaml
backend: vllm
model_name: meta-llama/Llama-2-7b-hf

# Tier 1: Universal decoder params
decoder:
  temperature: 0.0
  do_sample: false
  top_k: 50                      # Top-k sampling (0=disabled) - UNIVERSAL
  top_p: 1.0
  repetition_penalty: 1.0

# Tier 2: vLLM-native params
vllm:
  # Memory & Concurrency
  max_num_seqs: 256              # Max concurrent sequences (1-1024)
  max_num_batched_tokens: null   # Max tokens per iteration (null=auto)
  gpu_memory_utilization: 0.9    # GPU memory fraction for KV cache (0.5-0.99)
  swap_space: 4.0                # CPU swap per GPU in GiB
  cpu_offload_gb: 0.0            # CPU offload for model weights

  # Parallelism (native vLLM params)
  tensor_parallel_size: 2        # Split layers across GPUs
  pipeline_parallel_size: 1      # Pipeline stages (for very large models)
  distributed_backend: mp        # mp (multiprocessing) or ray
  disable_custom_all_reduce: false

  # KV Cache
  enable_prefix_caching: true    # Reuse KV cache for repeated prefixes
  enable_chunked_prefill: true   # Chunk large prefills with decode batches
  kv_cache_dtype: fp8            # KV precision: auto, float16, bfloat16, fp8
  block_size: 16                 # KV block size: 8, 16, or 32

  # Context
  max_model_len: 8192            # Maximum context length
  max_seq_len_to_capture: 8192   # Max seq len for CUDA graphs

  # Execution
  enforce_eager: false           # Disable CUDA graphs (for debugging)

  # Attention
  attention:
    backend: auto                # auto, FLASH_ATTN, FLASHINFER, TORCH_SDPA
    flash_version: 2             # 2 or 3 (H100/Hopper)
    disable_sliding_window: false

  # Speculative Decoding
  speculative:
    model: "TinyLlama/TinyLlama-1.1B"
    num_tokens: 5
    method: ngram
    prompt_lookup_min: 1
    prompt_lookup_max: 4
    draft_tp_size: 1

  # LoRA
  lora:
    enabled: true
    max_loras: 4
    max_rank: 16
    extra_vocab_size: 256

  # Quantization (vLLM-native methods)
  quantization: awq              # gptq, awq, fp8, marlin, etc.
  load_format: safetensors       # auto, pt, safetensors, gguf

  # Decoder Extensions (vLLM-specific)
  # Note: top_k is now in universal decoder config above
  min_p: 0.0                     # Min probability sampling
  best_of: 3                     # Generate N, return best
  logprobs: 5                    # Return top-k logprobs (1-20)
  logit_bias: {123: -100}        # Per-token bias

  # Escape hatch
  extra: {}
```

#### Key vLLM Parameters

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `gpu_memory_utilization` | Higher = more KV cache = more concurrent reqs | 0.9 for single model, 0.8 with other GPU usage |
| `enable_prefix_caching` | 30-50% throughput boost for similar prompts | Enable for chat/RAG workloads |
| `enable_chunked_prefill` | Better latency for mixed workloads | Enable for production serving |
| `kv_cache_dtype: fp8` | ~50% memory savings | Use on Hopper/Ada GPUs |
| `max_num_seqs` | Throughput vs latency tradeoff | Higher for throughput, lower for latency |
| `enforce_eager` | Debugging, first-token latency | Disable for benchmarks (slower) |

### PyTorch Configuration (`pytorch:`)

PyTorch uses **backend-native parameter names** that map to HuggingFace `transformers.GenerationConfig` and `model.from_pretrained()`.

```yaml
backend: pytorch
model_name: meta-llama/Llama-2-7b-hf

# Tier 1: Universal decoder params
decoder:
  temperature: 0.0
  do_sample: false
  top_k: 50                      # Top-k sampling (0=disabled) - UNIVERSAL
  top_p: 1.0
  repetition_penalty: 1.0

# Tier 2: PyTorch-native params
pytorch:
  # Batching (application-level)
  batch_size: 1
  batching_strategy: static      # static | dynamic | sorted_static | sorted_dynamic
  max_tokens_per_batch: null     # For dynamic strategies

  # Parallelism
  parallelism_strategy: none     # none | tensor_parallel | data_parallel
  parallelism_degree: 1

  # Quantization (BitsAndBytes)
  load_in_4bit: false
  load_in_8bit: false
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: false

  # Attention Implementation
  attn_implementation: flash_attention_2  # sdpa, flash_attention_2, eager

  # Compilation
  torch_compile: reduce-overhead  # false, default, reduce-overhead, max-autotune
  torch_compile_backend: null     # inductor, cudagraphs, onnxrt, aot_eager

  # KV Caching
  use_cache: true
  cache_implementation: null      # dynamic | static | hybrid | sliding_window

  # Memory
  low_cpu_mem_usage: true
  max_memory:
    "0": "20GiB"
    cpu: "30GiB"

  # Decoder Extensions (PyTorch-specific)
  # Note: top_k is now in universal decoder config above
  min_p: 0.0                     # Min probability sampling
  no_repeat_ngram_size: 0        # N-gram blocking (PyTorch only)
  output_scores: false
  return_dict_in_generate: false

  # Beam Search (PyTorch-specific)
  beam_search:
    enabled: false
    num_beams: 1
    length_penalty: 1.0
    early_stopping: false
    no_repeat_ngram_size: 0

  # Speculative Decoding
  assisted_generation:
    model: "TinyLlama/TinyLlama-1.1B"
    num_tokens: 5

  # Legacy
  use_bettertransformer: false

  # Escape hatch
  extra: {}
```

#### Key PyTorch Parameters

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `attn_implementation` | `flash_attention_2` fastest, `eager` most compatible | flash_attention_2 on Ampere+ GPUs |
| `torch_compile` | 10-30% speedup after warmup | `reduce-overhead` for small batches |
| `use_cache` | 2-10x faster generation | Always enable unless OOM |
| `low_cpu_mem_usage` | Avoids CPU memory spike during load | Always enable |
| `assisted_generation` | 2-3x latency improvement | Use small draft model (same family) |

#### Attention Implementation Comparison

| Implementation | Speed | Compatibility | Requirements |
|----------------|-------|---------------|--------------|
| `flash_attention_2` | Fastest | Ampere+ GPUs only | `pip install flash-attn` |
| `sdpa` | Fast | PyTorch 2.0+ | Built-in |
| `eager` | Slowest | All hardware | None |

#### torch.compile Modes

| Mode | Compile Time | Runtime Speed | Use Case |
|------|--------------|---------------|----------|
| `false` | None | Baseline | Debugging, compatibility |
| `default` | Fast | Good | General use |
| `reduce-overhead` | Medium | Better | Small batches, benchmarks |
| `max-autotune` | Slow | Best | Production, long-running |

### TensorRT-LLM Configuration (`tensorrt:`)

TensorRT-LLM uses **backend-native parameter names** that map to `trtllm-build` and TensorRT-LLM runtime. Engines can be pre-compiled or built on-demand from HuggingFace checkpoints.

```yaml
backend: tensorrt
model_name: meta-llama/Llama-2-7b-hf

# Tier 1: Universal decoder params
decoder:
  temperature: 0.0
  do_sample: false
  top_k: 50                      # Top-k sampling (0=disabled) - UNIVERSAL
  top_p: 1.0
  repetition_penalty: 1.0

# Tier 2: TensorRT-native params
tensorrt:
  # Engine Source
  engine_path: null                # Pre-compiled engine path (optional)
  engine_cache_dir: null           # Cache dir (default: ~/.cache/lem/tensorrt-engines/)
  force_rebuild: false             # Force rebuild even if cached

  # Build Configuration (compile-time)
  max_batch_size: 8                # Maximum batch size for compiled engine (1-256)
  max_input_len: null              # Max input tokens (defaults to model's max)
  max_output_len: null             # Max output tokens (defaults to config.max_output_tokens)
  builder_opt_level: 3             # TensorRT optimization level (0-5, higher=slower build)
  strongly_typed: true             # Enable strong typing for FP8 (recommended)
  multiple_profiles: false         # Multiple TRT profiles for different input shapes

  # Parallelism (native TRT-LLM params)
  tp_size: 1                       # Tensor parallel size
  pp_size: 1                       # Pipeline parallel size

  # Quantization (flattened - native TRT-LLM)
  quantization: none               # none, fp8, int8_sq, int8_weight_only, int4_awq, int4_gptq
  calibration:                     # Required for int8_sq
    dataset: wikitext
    split: train
    num_samples: 512
    max_length: 2048

  # Runtime Options
  kv_cache_type: paged             # paged (memory efficient) or continuous
  enable_chunked_context: true     # Chunk long sequences
  max_num_tokens: null             # Max tokens per iteration (inflight batching)
  gpu_memory_utilization: 0.9      # GPU memory fraction for KV cache (0.5-0.99)
  enable_kv_cache_reuse: true      # Enable prefix caching

  # Decoder Extensions (TensorRT-specific)
  # Note: top_k is now in universal decoder config above
  # Note: min_p and no_repeat_ngram_size NOT supported by TensorRT-LLM

  # Speculative Decoding
  draft_model: null                # Draft model for speculation
  num_draft_tokens: 5              # Tokens to speculate per step (1-10)

  # Escape Hatches
  extra_build_args: {}
  extra_runtime_args: {}
```

#### TensorRT Quantization Methods

| Method | Precision | Requirements | Use Case |
|--------|-----------|--------------|----------|
| `none` | FP16/BF16 | — | Baseline accuracy |
| `fp8` | FP8 | Hopper+ (sm_90+) | Fast, minimal accuracy loss |
| `int8_sq` | INT8 | Calibration data | Best INT8 accuracy |
| `int8_weight_only` | INT8 weights | — | Faster load, FP16 compute |
| `int4_awq` | INT4 | Pre-quantized checkpoint | Maximum compression |
| `int4_gptq` | INT4 | Pre-quantized checkpoint | Alternative INT4 method |

#### Key TensorRT Parameters

| Parameter | Impact | Recommendation |
|-----------|--------|----------------|
| `builder_opt_level` | Higher = slower build, faster inference | 3 for dev, 5 for production |
| `max_batch_size` | Compile-time limit affects memory | Match expected workload |
| `kv_cache_type: paged` | Memory-efficient KV management | Always use paged |
| `enable_chunked_context` | Better long sequence handling | Enable for long contexts |
| `enable_kv_cache_reuse` | Prefix caching, avoids recomputation | Enable for repeated prefixes |
| `quantization.method: fp8` | ~50% memory savings, minimal accuracy loss | Use on Hopper+ GPUs |

#### TensorRT Limitations

| Limitation | Workaround |
|------------|------------|
| No BitsAndBytes quantization | Use tensorrt.quantization.method instead |
| No `no_repeat_ngram_size` | Use `repetition_penalty` |
| No `min_p` sampling | Use `top_p` or `top_k` |
| Engines tied to GPU architecture | Rebuild when changing GPU type |
| Long initial build time | Use engine caching (automatic) |

### Backend-Specific Presets

Built-in presets for common scenarios:

```bash
# vLLM presets
lem experiment --preset vllm-throughput --model <model>    # High throughput
lem experiment --preset vllm-speculative --model <model>   # Speculative decoding
lem experiment --preset vllm-memory-efficient --model <model>  # fp8 KV cache
lem experiment --preset vllm-low-latency --model <model>   # Low latency

# PyTorch presets
lem experiment --preset pytorch-optimized --model <model>  # Flash attn + compile
lem experiment --preset pytorch-speculative --model <model> # Assisted generation
lem experiment --preset pytorch-compatible --model <model>  # Maximum compatibility
```

| Preset | Backend | Key Settings |
|--------|---------|--------------|
| `vllm-throughput` | vLLM | max_num_seqs=512, chunked_prefill, prefix_caching |
| `vllm-speculative` | vLLM | ngram speculation, 5 tokens |
| `vllm-memory-efficient` | vLLM | fp8 KV cache, prefix_caching, 95% GPU util |
| `vllm-low-latency` | vLLM | max_num_seqs=32, enforce_eager |
| `pytorch-optimized` | PyTorch | flash_attention_2, torch_compile=reduce-overhead |
| `pytorch-speculative` | PyTorch | assisted_generation enabled |
| `pytorch-compatible` | PyTorch | eager attention, no compile |

## Performance Tuning Guide

### Throughput Optimisation

**vLLM:**
```yaml
vllm:
  max_num_seqs: 512              # More concurrent requests
  enable_chunked_prefill: true   # Better scheduling
  enable_prefix_caching: true    # KV reuse for similar prompts
  gpu_memory_utilization: 0.95   # Maximum KV cache
```

**PyTorch:**
```yaml
pytorch:
  attn_implementation: flash_attention_2
  torch_compile: reduce-overhead
  batch_size: 8
  batching_strategy: dynamic     # Token-aware batching
```

### Latency Optimisation

**vLLM:**
```yaml
vllm:
  max_num_seqs: 32               # Fewer concurrent for faster response
  enforce_eager: true            # Skip CUDA graph capture
  speculative:
    method: ngram
    num_tokens: 5
```

**PyTorch:**
```yaml
pytorch:
  attn_implementation: flash_attention_2
  batch_size: 1
  assisted_generation:
    model: "TinyLlama/TinyLlama-1.1B"
    num_tokens: 5
```

### Memory Optimisation

**vLLM:**
```yaml
vllm:
  kv_cache_dtype: fp8            # 50% KV memory reduction
  gpu_memory_utilization: 0.8    # Leave headroom
  cpu_offload_gb: 4.0            # Offload model weights
```

**PyTorch:**
```yaml
fp_precision: float16
pytorch:
  load_in_4bit: true             # BitsAndBytes 4-bit quantization
  low_cpu_mem_usage: true
  max_memory:
    "0": "20GiB"
```

## Streaming Latency Metrics (TTFT/ITL)

All backends support streaming latency measurement for Time to First Token (TTFT) and Inter-Token Latency (ITL) metrics.

### Configuration

```yaml
streaming: true                    # Enable streaming latency measurement
streaming_warmup_requests: 5       # Warmup requests (excluded from stats)
```

```bash
# Or via CLI
lem experiment config.yaml --streaming --streaming-warmup 5
```

### Metrics Collected

| Metric | Description |
|--------|-------------|
| `ttft_mean_ms` | Mean time to first token |
| `ttft_p50_ms` | Median TTFT |
| `ttft_p95_ms` | 95th percentile TTFT |
| `ttft_p99_ms` | 99th percentile TTFT |
| `itl_mean_ms` | Mean inter-token latency |
| `itl_p50_ms` | Median ITL |
| `itl_p95_ms` | 95th percentile ITL |
| `itl_p99_ms` | 99th percentile ITL |

### Backend Measurement Methods

| Backend | Method | Accuracy | Notes |
|---------|--------|----------|-------|
| `pytorch` | Streaming callbacks | High | True per-token timestamps |
| `vllm` | Streaming API | High | Native async iteration |
| `tensorrt` | Version-dependent | High/Medium | v0.9+: native streaming; older: estimated |

### Important Caveats

1. **Sequential Processing**: Streaming mode processes prompts one at a time to capture accurate per-token timing. `batch_size` is ignored when `streaming: true`.

2. **Warmup Exclusion**: The first N requests (default 5) are excluded from statistics to avoid cold-start bias. Ensure `num_input_prompts > streaming_warmup_requests`.

3. **Statistical Significance**: For reliable percentiles, use at least 30+ measurement samples (prompts after warmup).

4. **TensorRT Estimation**: TensorRT-LLM versions prior to v0.9 don't provide native streaming timestamps. ITL values are estimated using proportional distribution and marked with `measurement_method: proportional_estimate`.

5. **Energy Profile**: Streaming mode may affect energy consumption patterns differently than batch mode due to sequential processing.

### Example Configuration

```yaml
# Streaming latency benchmark
config_name: latency-benchmark
model_name: meta-llama/Llama-2-7b-hf

streaming: true
streaming_warmup_requests: 5
num_input_prompts: 100            # 95 measurement samples after warmup

max_output_tokens: 128
decoder:
  preset: deterministic           # Reproducible results
```

## LoRA Adapter Support

Load LoRA (Low-Rank Adaptation) adapters for fine-tuned model inference.

### Configuration

```yaml
# Top-level adapter field
model_name: meta-llama/Llama-2-7b-hf
adapter: your-org/your-lora-adapter    # HuggingFace Hub ID or local path
```

### Backend Support

| Backend | LoRA Support | Multi-LoRA | Notes |
|---------|--------------|------------|-------|
| `pytorch` | ✓ | Single | Via PEFT library |
| `vllm` | ✓ | Multiple | Native support with concurrent adapters |
| `tensorrt` | ✗ | — | LoRA must be merged before engine build |

### vLLM Multi-LoRA Configuration

vLLM supports serving multiple LoRA adapters concurrently:

```yaml
backend: vllm
model_name: meta-llama/Llama-2-7b-hf
adapter: your-org/default-adapter       # Default adapter

vllm:
  lora:
    enabled: true
    max_loras: 4                         # Max concurrent adapters
    max_rank: 16                         # Max LoRA rank supported
    extra_vocab_size: 256                # Extra vocab for adapters
```

### PyTorch LoRA Configuration

PyTorch uses the PEFT library for LoRA loading:

```yaml
backend: pytorch
model_name: meta-llama/Llama-2-7b-hf
adapter: your-org/your-lora-adapter     # Loaded via PEFT
```

### TensorRT with LoRA

TensorRT-LLM requires LoRA weights to be merged into the base model before engine compilation:

```bash
# 1. Merge LoRA weights (external step)
python merge_lora.py --base meta-llama/Llama-2-7b-hf --lora your-adapter --output merged-model/

# 2. Use merged model with TensorRT
backend: tensorrt
model_name: ./merged-model/
```

## Limitations

### vLLM Backend

| Limitation | Workaround |
|------------|------------|
| No BitsAndBytes quantization | Use pre-quantized AWQ/GPTQ models or FP8 |
| No `no_repeat_ngram_size` | Use `repetition_penalty` |
| `batch_size` is a hint | vLLM's continuous batching optimises automatically |
| Energy is aggregate only | Per-request energy not available |

### PyTorch Backend

| Limitation | Workaround |
|------------|------------|
| Static batching | Use `strategy: dynamic` for token-aware batching |
| Lower throughput | Use vLLM for production-grade throughput |

### TensorRT Backend

| Limitation | Workaround |
|------------|------------|
| No BitsAndBytes quantization | Use tensorrt.quantization.method instead |
| No `no_repeat_ngram_size` | Use `repetition_penalty` |
| No `min_p` sampling | Use `top_p` or `top_k` instead |
| Engines tied to GPU architecture | Rebuild when changing GPU type |
| Long initial build time | Use engine caching (automatic) |
| FP8 requires Hopper+ | Use INT8 on older GPUs |

## Cross-Backend Comparison

Comparing results across backends is scientifically valid when done correctly:

**Valid comparisons:**
- Energy per output token across backends
- Throughput (tokens/second) with same prompts
- End-to-end efficiency with identical workloads

**Important considerations:**
1. Use identical prompts and model weights
2. Compare normalised metrics (`energy_per_output_token_j`)
3. Document backend in results (automatic)
4. Mixed-backend aggregation is rejected by default

```bash
# Run same workload on both backends
lem experiment config.yaml --backend pytorch
lem experiment config.yaml --backend vllm

# Compare results
lem results show <pytorch_exp_id>
lem results show <vllm_exp_id>
```

## Architecture

### RuntimeCapabilities Protocol

Backends declare their runtime requirements via `RuntimeCapabilities`, enabling the orchestration layer to configure itself without hardcoded backend checks:

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI / Runner                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    launcher.py                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ _get_launch_mode(config)                                   │ │
│  │   └─► backend.get_runtime_capabilities()                   │ │
│  │       └─► Returns LaunchMode (DIRECT/ACCELERATE/TORCHRUN)  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                │                                 │
│         ┌──────────────────────┼──────────────────────┐         │
│         ▼                      ▼                      ▼         │
│   Direct Python         torchrun              accelerate        │
│   (vLLM, TRT)          (TP/PP)                (PyTorch)         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    context.py                                    │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ ExperimentContext.create(config)                           │ │
│  │   └─► capabilities.orchestrator_may_call_cuda              │ │
│  │       ├─► True:  Use Accelerator (initialises CUDA)        │ │
│  │       └─► False: Use MinimalAccelerator (no CUDA calls)    │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Backend Capabilities

| Backend | LaunchMode | CudaManagement | Tensor Parallel |
|---------|------------|----------------|-----------------|
| `pytorch` | ACCELERATE | ORCHESTRATOR | ✓ (via Accelerate) |
| `vllm` | DIRECT | BACKEND | ✓ (native) |
| `tensorrt` | DIRECT | BACKEND | ✓ (native) |

**Key concepts:**

- **LaunchMode.DIRECT**: Backend manages its own multiprocessing (e.g., vLLM uses spawn)
- **LaunchMode.ACCELERATE**: Use HuggingFace Accelerate for distributed setup
- **LaunchMode.TORCHRUN**: Use torchrun for tensor/pipeline parallelism
- **CudaManagement.BACKEND**: Backend initialises CUDA; orchestration must NOT call `torch.cuda.*`
- **CudaManagement.ORCHESTRATOR**: Orchestration layer may safely initialise CUDA

### Adding New Backends

To add a new backend, implement `InferenceBackend` protocol and declare capabilities:

```python
class MyBackend:
    def get_runtime_capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            launch_mode=LaunchMode.DIRECT,  # or ACCELERATE/TORCHRUN
            cuda_management=CudaManagement.BACKEND,  # or ORCHESTRATOR
            supports_tensor_parallel=True,
            supports_pipeline_parallel=False,
            manages_own_batching=True,
        )

    # ... implement other protocol methods
```

Register in `inference_backends/__init__.py`:

```python
_LAZY_BACKENDS["mybackend"] = "llenergymeasure.core.inference_backends.mybackend:MyBackend"
```

## Troubleshooting

### Backend Not Available

```
ERROR: Backend 'vllm' requires extra dependencies.
Install with: pip install lem[vllm]
```

**Solution:** Install the backend extras:
```bash
pip install lem[vllm]
# Or use Docker images which include dependencies
```

### Config Warnings

```
WARNING: decoder.no_repeat_ngram_size not supported by vLLM. Use repetition_penalty instead.
```

**Solution:** Adjust config for backend compatibility or ignore with `--ignore-backend-warnings`.

### Mixed Backend Aggregation

```
ERROR: Mixed backends detected: pytorch, vllm. Aggregating results from different backends produces statistically invalid comparisons.
```

**Solution:** Either:
- Ensure all processes use the same backend
- Use `--allow-mixed-backends` (not recommended)
