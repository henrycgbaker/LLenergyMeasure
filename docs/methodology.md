# LLM Efficiency Measurement Methodology

This document describes the methodology used by `lem` to measure LLM inference efficiency.

## Overview

The framework measures three primary dimensions of LLM inference:

1. **Throughput** - Tokens generated per second
2. **Energy Consumption** - Joules consumed per inference
3. **Computational Cost** - FLOPs required for inference

These metrics are combined to derive efficiency ratios that enable fair comparison across models, configurations, and hardware.

## Measurement Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Runner                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Energy    │  │  Inference  │  │   FLOPs     │          │
│  │   Backend   │  │   Engine    │  │  Estimator  │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                 │
│         ▼                ▼                ▼                 │
│  ┌─────────────────────────────────────────────────┐        │
│  │              Raw Process Results                │        │
│  │  (per-GPU: tokens, energy, FLOPs, timestamps)   │        │
│  └──────────────────────┬──────────────────────────┘        │
└─────────────────────────┼───────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │     Aggregation       │
              │  (sum energy, avg     │
              │   throughput)         │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Aggregated Result    │
              │  + Efficiency Metrics │
              └───────────────────────┘
```

## Metrics Collected

### Inference Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `total_tokens` | count | Total tokens generated (input + output) |
| `input_tokens` | count | Prompt/input token count |
| `output_tokens` | count | Generated output token count |
| `inference_time_sec` | seconds | Wall-clock inference duration |
| `tokens_per_second` | tok/s | Throughput rate |
| `latency_per_token_ms` | ms | Average per-token latency |

### Energy Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `total_energy_j` | Joules | Total energy consumed |
| `gpu_energy_j` | Joules | GPU energy consumption |
| `cpu_energy_j` | Joules | CPU energy consumption |
| `gpu_power_w` | Watts | Average GPU power draw |
| `cpu_power_w` | Watts | Average CPU power draw |
| `duration_sec` | seconds | Measurement window |

### Compute Metrics

| Metric | Unit | Description |
|--------|------|-------------|
| `flops_total` | FLOPs | Total floating-point operations |
| `flops_per_second` | FLOP/s | Computational throughput |
| `flops_method` | string | Estimation method used |
| `flops_confidence` | high/medium/low | Confidence in estimate |

## Energy Measurement

### Backend: CodeCarbon

The default energy backend uses [CodeCarbon](https://codecarbon.io/) which provides:

- **GPU Energy**: Via NVIDIA SMI power readings (integrated over time)
- **CPU Energy**: Via Intel RAPL (Running Average Power Limit) when available
- **Fallback**: TDP-based estimation when hardware counters unavailable

### Measurement Protocol

1. **Warmup Phase**: Initial inference runs to stabilize GPU frequencies
2. **Measurement Start**: Energy tracker initialized
3. **Inference Execution**: Model generates tokens
4. **Measurement Stop**: Energy tracker finalized
5. **Result Recording**: Metrics saved with timestamps

### Multi-GPU Considerations

For distributed inference across multiple GPUs:

- Each process records its own energy consumption
- GPU energy is summed across processes
- Temporal overlap is verified to ensure concurrent execution
- GPU attribution is checked to prevent double-counting

## FLOPs Estimation

FLOPs (Floating-Point Operations) provide a hardware-independent measure of computational cost.

### Estimation Strategies

The framework uses a 3-tier fallback strategy:

1. **CalFlops** (High Confidence)
   - Traces model execution to count actual operations
   - Most accurate for supported model architectures
   - Requires model execution

2. **Architecture-Based** (Medium Confidence)
   - Calculates FLOPs from model configuration
   - Uses known formulas for transformer architectures
   - `FLOPs = 2 × params × tokens` (forward pass approximation)

3. **Parameter Estimate** (Low Confidence)
   - Uses model parameter count as proxy
   - Least accurate but always available
   - Fallback when architecture unknown

### Quantization Handling

For quantized models (4-bit, 8-bit):
- FLOPs are calculated at the compute precision (typically FP16)
- Dequantization happens before matrix operations
- Energy savings come from reduced memory bandwidth, not fewer FLOPs

## Aggregation Methodology

### Multi-Process Results

When aggregating results from multiple GPU processes:

```python
# Energy: Sum across processes
total_energy = sum(process.energy for process in results)

# Throughput: Average across processes
avg_throughput = mean(process.tokens_per_second for process in results)

# Tokens: Sum across processes
total_tokens = sum(process.total_tokens for process in results)

# FLOPs: Sum across processes
total_flops = sum(process.flops_total for process in results)
```

### Validation Checks

Before aggregation, the system verifies:

1. **Temporal Overlap**: Processes ran concurrently (not sequentially)
2. **GPU Attribution**: Each GPU ID appears only once (no double-counting)

Warnings are generated if these checks fail.

## Efficiency Metrics

### Primary Efficiency Ratios

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Tokens per Joule | `total_tokens / total_energy_j` | Energy efficiency |
| Joules per Token | `total_energy_j / total_tokens` | Energy cost per token |
| Effective Throughput | `total_tokens / duration_sec` | System-wide throughput |

### Derived Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| FLOP/J | `total_flops / total_energy_j` | Computational efficiency |
| FLOP/token | `total_flops / total_tokens` | Model complexity |
| W per tok/s | `avg_power / throughput` | Power-performance ratio |

## Extended Efficiency Metrics

Beyond the primary metrics, the framework computes additional efficiency indicators when data is available. These use a **stable schema** - all fields are always present in results, but show N/A when not computable for a given configuration.

### TPOT (Time Per Output Token)

| Field | Unit | Description | Condition |
|-------|------|-------------|-----------|
| `tpot_ms` | ms | Mean inter-token latency | `streaming=True` |

TPOT equals the mean ITL (Inter-Token Latency) from streaming inference. It represents the average time to generate each output token after the first.

### Token Efficiency Index

| Field | Formula | Description |
|-------|---------|-------------|
| `token_efficiency_index` | `throughput × tokens_per_joule × precision_factor` | Composite efficiency score |

The Token Efficiency Index combines throughput and energy efficiency into a single comparable metric. The `precision_factor` adjusts for compute precision (higher precision = more efficient use of FLOPs).

### Memory Efficiency

| Field | Unit | Description | Condition |
|-------|------|-------------|-----------|
| `tokens_per_gb_vram` | tok/GB | Output tokens per GB VRAM used | Always |
| `model_memory_utilisation` | 0.0-1.0 | Model memory / total VRAM | Always |
| `kv_cache_memory_ratio` | 0.0-1.0 | KV cache / peak memory | vLLM only |

### GPU Utilisation

| Field | Unit | Description | Condition |
|-------|------|-------------|-----------|
| `sm_utilisation_mean` | % | Mean SM (streaming multiprocessor) utilisation | NVML available |
| `memory_bandwidth_utilisation` | % | Memory bandwidth utilisation | NVML available |

GPU utilisation is sampled in the background during inference using NVML (NVIDIA Management Library) via the `nvidia-ml-py` package. If NVML is unavailable (e.g., CUDA context conflicts with vLLM), these metrics show N/A.

> **Note**: The package is `nvidia-ml-py` on PyPI but imports as `pynvml` in Python code. The old `pynvml` package on PyPI is deprecated.

### Request Latency

| Field | Unit | Description | Condition |
|-------|------|-------------|-----------|
| `e2e_latency_mean_ms` | ms | Mean end-to-end request latency | Always |
| `e2e_latency_median_ms` | ms | Median E2E latency | Always |
| `e2e_latency_p95_ms` | ms | 95th percentile E2E latency | Always |
| `e2e_latency_p99_ms` | ms | 99th percentile E2E latency | Always |

Request latency measures the total time from request submission to completion, useful for understanding serving performance.

### Batch Efficiency

| Field | Unit | Description | Condition |
|-------|------|-------------|-----------|
| `effective_batch_size` | count | Average actual batch size | Static batching |
| `batch_utilisation` | 0.0-1.0 | Actual / configured batch size | Static batching |
| `padding_overhead` | 0.0-1.0 | Padding tokens / total tokens | Static batching |

These metrics are only relevant for backends using static batching (PyTorch, TensorRT). vLLM's continuous batching doesn't have a fixed batch size.

### KV Cache Efficiency (vLLM Only)

| Field | Unit | Description | Condition |
|-------|------|-------------|-----------|
| `kv_cache_hit_rate` | 0.0-1.0 | Prefix cache hit rate | vLLM + `enable_prefix_caching` |
| `kv_cache_blocks_used` | count | KV cache blocks used | vLLM |
| `kv_cache_blocks_total` | count | Total KV cache blocks | vLLM |

KV cache metrics are specific to vLLM's paged attention mechanism. The hit rate is only available when prefix caching is enabled.

### Late Aggregation

Extended metrics use **late aggregation** - raw samples are stored per-process and statistics are computed during aggregation:

```
Per-process:                     Aggregation:
├── per_request_latencies_ms     → Combine all latencies
├── gpu_utilisation_samples      → Combine all GPU samples
└── extended_metrics             → Recompute stats on combined data
```

This approach ensures statistically correct percentiles (not average-of-percentiles) when aggregating multi-GPU results.

## Experimental Controls

### Recommended Practices

1. **Fixed Prompts**: Use consistent prompt sets across experiments
2. **Token Limits**: Set explicit `max_output_tokens` for comparability
3. **Warmup**: Run 3-5 warmup iterations before measurement
4. **Multiple Runs**: Average across 3+ runs for statistical significance
5. **Temperature**: Use `temperature=0` for deterministic output

### Configuration Options

```yaml
# Example config with controls
config_name: controlled-experiment
model_name: meta-llama/Llama-2-7b-hf

max_input_tokens: 1024
max_output_tokens: 512

decoder:
  temperature: 0.0
  do_sample: false

num_processes: 2
gpus: [0, 1]
fp_precision: float16
```

## Limitations

### Energy Measurement

- **Software-based**: Uses OS/driver power readings, not hardware meters
- **Granularity**: Typically 100ms+ sampling intervals
- **Attribution**: Shared components (RAM, PSU) may not be attributed
- **Idle Power**: Background system load affects measurements

### FLOPs Estimation

- **Forward-only**: Does not account for attention caching benefits
- **Architecture-specific**: Some custom architectures may be inaccurate
- **Quantization**: May overestimate for heavily quantized models

### Multi-GPU

- **Communication Overhead**: Inter-GPU communication not separately measured
- **Load Imbalance**: Assumes balanced workload across GPUs

## References

- [CodeCarbon Documentation](https://mlco2.github.io/codecarbon/)
- [calflops: FLOPs calculation tool](https://github.com/MrYxJ/calculate-flops.pytorch)
- [NVIDIA SMI Power Monitoring](https://developer.nvidia.com/nvidia-system-management-interface)
- [Intel RAPL Documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)
