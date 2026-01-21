# Inference Backends

LLM Energy Measure supports multiple inference backends, each optimised for different use cases. This guide covers backend selection, configuration mapping, and compatibility.

## Backend Overview

| Backend | Status | Use Case | Key Features |
|---------|--------|----------|--------------|
| `pytorch` | **Default** | Research, experimentation | HuggingFace Transformers + Accelerate |
| `vllm` | Available | Production, throughput testing | PagedAttention, continuous batching, native TP |
| `tensorrt` | Available | Enterprise, maximum efficiency | Compiled inference, TensorRT-LLM, FP8/INT8/INT4 |

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
- Using BitsAndBytes 8-bit quantization
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
| Pipeline parallelism | ✗ | ✗ | ✓ | TensorRT only |
| **Quantisation** |
| BitsAndBytes 4-bit | ✓ | ✓ | ✗ | Use TRT's native INT4 instead |
| BitsAndBytes 8-bit | ✓ | ✗ | ✗ | PyTorch only |
| FP8 | ✗ | ✓ | ✓ | Hopper+ GPUs (sm_90+) |
| INT8 (smooth quant) | ✗ | ✗ | ✓ | Requires calibration |
| GPTQ | ✓ | ✓ | ✓ | Pre-quantised checkpoints |
| AWQ | ✓ | ✓ | ✓ | Pre-quantised checkpoints |
| **Optimisations** |
| KV caching | ✓ | ✓ | ✓ | All support by default |
| Prefix caching | ✗ | ✓ | ✗ | vLLM only |
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
| `top_k` | ✓ (0=off) | ✓ (-1=off) | ✓ | Disable value differs |
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

### Selecting a Backend

```yaml
# In config.yaml
backend: vllm  # or 'pytorch' (default)
```

```bash
# Or via CLI
llm-energy-measure experiment config.yaml --backend vllm
```

### Backend-Specific Docker Images

```bash
# PyTorch backend (default)
docker compose run --rm llm-energy-measure-app llm-energy-measure experiment ...

# vLLM backend
docker compose run --rm vllm llm-energy-measure experiment ... --backend vllm
```

## Parameter Compatibility Matrix

### Universal Parameters

These work identically across all backends:

| Parameter | Config Path | Description |
|-----------|-------------|-------------|
| `model_name` | `model_name` | HuggingFace model ID |
| `fp_precision` | `fp_precision` | `float16`, `bfloat16`, `float32` |
| `max_output_tokens` | `max_output_tokens` | Maximum generated tokens |
| `min_output_tokens` | `min_output_tokens` | Minimum generated tokens |
| `random_seed` | `random_seed` | Reproducibility seed |
| `temperature` | `decoder.temperature` | Sampling temperature |
| `top_p` | `decoder.top_p` | Nucleus sampling threshold |
| `repetition_penalty` | `decoder.repetition_penalty` | Repetition penalty |

### Backend-Specific Support

| Parameter | PyTorch | vLLM | Notes |
|-----------|:-------:|:----:|-------|
| `decoder.top_k` | ✓ (0=off) | ✓ (-1=off) | Semantic difference: PyTorch uses 0, vLLM uses -1 to disable |
| `decoder.min_p` | ✓ | ✓ | Both support min_p sampling |
| `decoder.no_repeat_ngram_size` | ✓ | ✗ | vLLM: use `repetition_penalty` instead |
| `decoder.do_sample` | ✓ | — | vLLM: controlled via `temperature=0` for greedy |
| `quantization.load_in_4bit` | ✓ | ✓ | BitsAndBytes 4-bit on both |
| `quantization.load_in_8bit` | ✓ | ✗ | vLLM does not support BNB 8-bit |
| `batching.batch_size` | ✓ (exact) | ✓ (hint) | vLLM: continuous batching makes this a hint |
| `batching.strategy` | ✓ | — | vLLM always uses continuous batching |
| `sharding.strategy` | ✓ | ✓ | Both support TP; PyTorch via Accelerate, vLLM native |
| `traffic_simulation` | ✓ | ✓ | MLPerf-style Poisson arrivals |

### Semantic Differences

**`batch_size`**
- **PyTorch**: Exact batch size for static batching
- **vLLM**: Hint for `max_num_seqs` (concurrent requests); vLLM manages batching internally

**`top_k`**
- **PyTorch**: `0` disables top-k sampling
- **vLLM**: `-1` disables top-k sampling (automatically converted)

**`do_sample`**
- **PyTorch**: Explicit flag to enable sampling
- **vLLM**: Implicit; `temperature=0` means greedy decoding

## Backend-Specific Configuration

Each backend exposes its own configuration section for advanced optimisation. These are set via `vllm:` or `pytorch:` blocks in YAML.

### vLLM Configuration (`vllm:`)

```yaml
backend: vllm
model_name: meta-llama/Llama-2-7b-hf

vllm:
  # Memory & Batching
  max_num_seqs: 256              # Max concurrent sequences (1-1024)
  max_num_batched_tokens: null   # Max tokens per iteration (null=auto)
  gpu_memory_utilization: 0.9    # GPU memory fraction for KV cache (0.5-0.99)
  swap_space: 4.0                # CPU swap per GPU in GiB
  cpu_offload_gb: 0.0            # CPU offload for model weights

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

  # Parallelism
  distributed_backend: mp        # mp (multiprocessing) or ray
  disable_custom_all_reduce: false

  # Attention
  attention:
    backend: auto                # auto, FLASH_ATTN, FLASHINFER, TORCH_SDPA
    flash_version: 2             # 2 or 3 (H100/Hopper)
    disable_sliding_window: false

  # Speculative Decoding
  speculative:
    model: "TinyLlama/TinyLlama-1.1B"  # Draft model
    num_tokens: 5                # Tokens to speculate (1-10)
    method: ngram                # ngram, eagle, eagle3, medusa, mlp
    ngram_min: 1
    ngram_max: 4
    draft_tp_size: 1

  # LoRA
  lora:
    enabled: true
    max_loras: 4
    max_rank: 16
    extra_vocab_size: 256

  # Quantization
  quantization_method: awq       # gptq, awq, fp8, marlin, etc.
  load_format: safetensors       # auto, pt, safetensors, gguf

  # Advanced Sampling
  best_of: 3                     # Generate N, return best
  use_beam_search: false
  length_penalty: 1.0
  logprobs: 5                    # Return top-k logprobs (1-20)
  logit_bias: {123: -100}        # Per-token bias

  # Escape hatch for experimental options
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

```yaml
backend: pytorch
model_name: meta-llama/Llama-2-7b-hf

pytorch:
  # Attention Implementation
  attn_implementation: flash_attention_2  # sdpa, flash_attention_2, eager

  # Compilation
  torch_compile: reduce-overhead  # false, default, reduce-overhead, max-autotune

  # Legacy (pre-PyTorch 2.0)
  use_bettertransformer: false

  # KV Caching
  use_cache: true                # Disable to reduce memory at cost of speed

  # Memory
  low_cpu_mem_usage: true        # Load directly to GPU
  max_memory:                    # Per-device limits
    "0": "20GiB"
    cpu: "30GiB"

  # Assisted Generation (Speculative Decoding)
  assisted_generation:
    model: "TinyLlama/TinyLlama-1.1B"  # Assistant model
    num_tokens: 5                # Tokens to speculate (1-10)

  # Beam Search
  num_beams: 1                   # 1=greedy/sampling, >1=beam search
  early_stopping: false
  length_penalty: 1.0

  # Output Configuration
  output_scores: false           # Return generation scores
  return_dict_in_generate: false

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

TensorRT-LLM provides compiled inference plans optimised for specific GPU configurations. Engines can be pre-compiled or built on-demand from HuggingFace checkpoints.

```yaml
backend: tensorrt
model_name: meta-llama/Llama-2-7b-hf

tensorrt:
  # Engine Source
  engine_path: null                # Pre-compiled engine path (optional)
  engine_cache_dir: null           # Cache dir (default: ~/.cache/llm-energy-measure/tensorrt-engines/)
  force_rebuild: false             # Force rebuild even if cached

  # Build Configuration (when compiling from HF checkpoint)
  max_batch_size: 8                # Maximum batch size for compiled engine (1-256)
  max_input_len: null              # Max input tokens (defaults to model's max)
  max_output_len: null             # Max output tokens (defaults to config.max_output_tokens)
  builder_opt_level: 3             # TensorRT optimization level (0-5, higher=slower build)
  strongly_typed: true             # Enable strong typing for FP8 (recommended)

  # Quantization
  quantization:
    method: none                   # none, fp8, int8_sq, int8_weight_only, int4_awq, int4_gptq
    calibration:                   # Required for int8_sq
      dataset: wikitext            # Calibration dataset
      split: train
      num_samples: 512             # Calibration samples (512-1024 typical)
      max_length: 2048

  # Tensor Parallelism
  tp_size: null                    # TP size (defaults to sharding.num_shards)
  pp_size: 1                       # Pipeline parallel size

  # Runtime Options
  kv_cache_type: paged             # paged (memory efficient) or continuous
  enable_chunked_context: true     # Chunk long sequences
  max_num_tokens: null             # Max tokens per iteration (inflight batching)
  gpu_memory_utilization: 0.9      # GPU memory fraction for KV cache (0.5-0.99)

  # Speculative Decoding
  draft_model: null                # Draft model for speculation
  num_draft_tokens: 5              # Tokens to speculate per step (1-10)

  # Escape Hatches
  extra_build_args: {}             # Additional trtllm-build kwargs
  extra_runtime_args: {}           # Additional runtime kwargs
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
llm-energy-measure experiment --preset vllm-throughput --model <model>    # High throughput
llm-energy-measure experiment --preset vllm-speculative --model <model>   # Speculative decoding
llm-energy-measure experiment --preset vllm-memory-efficient --model <model>  # fp8 KV cache
llm-energy-measure experiment --preset vllm-low-latency --model <model>   # Low latency

# PyTorch presets
llm-energy-measure experiment --preset pytorch-optimized --model <model>  # Flash attn + compile
llm-energy-measure experiment --preset pytorch-speculative --model <model> # Assisted generation
llm-energy-measure experiment --preset pytorch-compatible --model <model>  # Maximum compatibility
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
batching:
  batch_size: 8
  strategy: dynamic
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
  assisted_generation:
    num_tokens: 5
batching:
  batch_size: 1
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
quantization:
  load_in_4bit: true
pytorch:
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
llm-energy-measure experiment config.yaml --streaming --streaming-warmup 5
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
| No 8-bit BitsAndBytes | Use 4-bit or pre-quantized GPTQ/AWQ models |
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
llm-energy-measure experiment config.yaml --backend pytorch
llm-energy-measure experiment config.yaml --backend vllm

# Compare results
llm-energy-measure results show <pytorch_exp_id>
llm-energy-measure results show <vllm_exp_id>
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
_LAZY_BACKENDS["mybackend"] = "llm_energy_measure.core.inference_backends.mybackend:MyBackend"
```

## Troubleshooting

### Backend Not Available

```
ERROR: Backend 'vllm' requires extra dependencies.
Install with: pip install llm-energy-measure[vllm]
```

**Solution:** Install the backend extras:
```bash
pip install llm-energy-measure[vllm]
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
