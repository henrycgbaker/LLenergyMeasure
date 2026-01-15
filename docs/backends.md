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
