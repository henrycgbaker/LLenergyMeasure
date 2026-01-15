# Inference Backends

LLM Energy Measure supports multiple inference backends, each optimised for different use cases. This guide covers backend selection, configuration mapping, and compatibility.

## Backend Overview

| Backend | Status | Use Case | Key Features |
|---------|--------|----------|--------------|
| `pytorch` | **Default** | Research, experimentation | HuggingFace Transformers + Accelerate |
| `vllm` | Available | Production, throughput testing | PagedAttention, continuous batching, native TP |
| `tensorrt` | Planned | Enterprise, maximum efficiency | Compiled inference, TensorRT-LLM |

## Choosing a Backend

```
                        ┌─────────────────────────────┐
                        │  Which backend should I use? │
                        └──────────────┬──────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │ Need production-grade throughput?   │
                    └──────────────────┬──────────────────┘
                               Yes     │     No
                    ┌──────────────────┴──────────────────┐
                    ▼                                      ▼
              Use vllm                              Use pytorch
         (continuous batching,                   (HuggingFace models,
          PagedAttention)                        full decoder control)
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
| No TTFT metric | TTFT requires streaming callbacks (not implemented) |

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
