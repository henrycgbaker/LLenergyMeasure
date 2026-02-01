# Configuration Reference

> This file is auto-generated from Pydantic models. Do not edit manually.
> Generated: 2026-02-01

## Table of Contents

- [adapter](#adapter)
- [backend](#backend)
- [baseline](#baseline)
- [config_name](#config-name)
- [cycle_id](#cycle-id)
- [dataset](#dataset)
- [decode_token_to_text](#decode-token-to-text)
- [decoder](#decoder)
- [extra_metadata](#extra-metadata)
- [fp_precision](#fp-precision)
- [gpus](#gpus)
- [io](#io)
- [max_input_tokens](#max-input-tokens)
- [max_output_tokens](#max-output-tokens)
- [min_output_tokens](#min-output-tokens)
- [model_name](#model-name)
- [num_cycles](#num-cycles)
- [num_input_prompts](#num-input-prompts)
- [prompts](#prompts)
- [pytorch](#pytorch)
- [query_rate](#query-rate)
- [random_seed](#random-seed)
- [save_outputs](#save-outputs)
- [schedule](#schedule)
- [streaming](#streaming)
- [streaming_warmup_requests](#streaming-warmup-requests)
- [tensorrt](#tensorrt)
- [timeseries](#timeseries)
- [traffic_simulation](#traffic-simulation)
- [vllm](#vllm)
- [warmup](#warmup)

---

## adapter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapter` | str \| None | None | LoRA adapter: HuggingFace Hub ID or local path |

## backend

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | Literal['pytorch', 'tensorrt', 'vllm'] | 'pytorch' | Inference backend |

## baseline

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline` | BaselineConfig | PydanticUndefined | Baseline power measurement configuration |
| `baseline.enabled` | bool | True | Enable baseline power measurement |
| `baseline.required` | bool | False | Fail experiment if baseline measurement fails (false = warn and continue) |
| `baseline.duration_sec` | float | 30.0 | Baseline measurement duration in seconds |
| `baseline.cache_ttl_sec` | float | 3600.0 | Cache validity in seconds (default 1 hour) |
| `baseline.sample_interval_ms` | int | 100 | Sampling interval in milliseconds |

## config_name

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_name` | str | PydanticUndefined *(required)* | Unique config identifier |

## cycle_id

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cycle_id` | int \| None | None | Experiment cycle ID |

## dataset

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | llenergymeasure.config.models.DatasetConfig \| None | None | Simple dataset config. For advanced options, use 'prompts' instead. |

## decode_token_to_text

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decode_token_to_text` | bool | False | Decode tokens to text |

## decoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decoder` | DecoderConfig | PydanticUndefined | Universal decoder/generation configuration |
| `decoder.temperature` | float | 1.0 | Sampling temperature (0=greedy) |
| `decoder.do_sample` | bool | True | Enable sampling (ignored if temp=0) |
| `decoder.top_k` | int | 50 | Top-k sampling (0=disabled) |
| `decoder.top_p` | float | 1.0 | Top-p nucleus sampling (1.0=disabled) |
| `decoder.repetition_penalty` | float | 1.0 | Repetition penalty (1.0=no penalty) |
| `decoder.preset` | Literal['deterministic', 'standard', 'creative', 'factual'] \| None | None | Sampling preset (expands to preset values, overrides apply on top) |

## extra_metadata

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `extra_metadata` | dict[str, Any] | PydanticUndefined | Additional metadata |

## fp_precision

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `fp_precision` | Literal['float32', 'float16', 'bfloat16'] | 'float16' | Floating point precision |

## gpus

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gpus` | list[int] | PydanticUndefined | GPU indices to use |

## io

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `io` | IOConfig | PydanticUndefined | I/O paths configuration |
| `io.results_dir` | str \| None | None | Results output directory (overrides .env default) |

## max_input_tokens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_input_tokens` | int | 512 | Max input tokens |

## max_output_tokens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_output_tokens` | int | 128 | Max output tokens |

## min_output_tokens

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_output_tokens` | int | 0 | Min output tokens |

## model_name

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | PydanticUndefined *(required)* | HuggingFace model name/path |

## num_cycles

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_cycles` | int | 1 | Number of cycles for statistical robustness (1-10). With 1 cycle, confidence intervals and robustness metrics cannot be computed. Use >= 3 cycles for basic statistical validity, >= 5 for publication-grade results. |

## num_input_prompts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_input_prompts` | int | 1 | Number of prompts |

## prompts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | Annotated \| None | None | Advanced prompt source: file or huggingface dataset with full options |

## pytorch

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pytorch` | PyTorchConfig \| None | None | PyTorch-specific configuration (only used when backend=pytorch) |

## query_rate

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query_rate` | float | 1.0 | Query rate (queries/sec) |

## random_seed

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | int \| None | None | Random seed for reproducibility (None = non-deterministic) |

## save_outputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `save_outputs` | bool | False | Save generated outputs |

## schedule

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `schedule` | ScheduleConfig | PydanticUndefined | Schedule config for daemon mode |
| `schedule.enabled` | bool | False | Enable scheduled mode |
| `schedule.interval` | str \| None | None | Interval between runs (e.g., '6h', '30m', '1d') |
| `schedule.at` | str \| None | None | Specific time of day to run (e.g., '09:00', '14:30') |
| `schedule.days` | list[str] \| None | None | Days to run on (e.g., ['mon', 'wed', 'fri'] or ['weekdays']) |
| `schedule.total_duration` | str | '24h' | Total duration to run daemon (e.g., '24h', '7d') |

## streaming

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming` | bool | False | Enable streaming mode for TTFT/ITL latency measurement. |

## streaming_warmup_requests

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming_warmup_requests` | int | 5 | Warmup requests before streaming measurement (excluded from stats) |

## tensorrt

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensorrt` | TensorRTConfig \| None | None | TensorRT-LLM configuration (only used when backend=tensorrt) |

## timeseries

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `timeseries` | TimeSeriesConfig | PydanticUndefined | Time-series data collection configuration |
| `timeseries.enabled` | bool | False | Enable time-series data collection |
| `timeseries.save` | bool | False | Save time-series to separate file (--save-timeseries) |
| `timeseries.sample_interval_ms` | int | 100 | Sampling interval in ms (100ms = 10Hz, 1000ms = 1Hz) |

## traffic_simulation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `traffic_simulation` | TrafficSimulation | PydanticUndefined | MLPerf-style traffic simulation |
| `traffic_simulation.enabled` | bool | False | Enable traffic simulation |
| `traffic_simulation.mode` | Literal['constant', 'poisson'] | 'poisson' | Traffic arrival pattern (MLPerf terminology) |
| `traffic_simulation.target_qps` | float | 1.0 | Target queries per second (arrival rate λ) |
| `traffic_simulation.seed` | int \| None | None | Random seed for reproducible Poisson arrivals |

## vllm

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vllm` | VLLMConfig \| None | None | vLLM-specific configuration (only used when backend=vllm) |

## warmup

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `warmup` | WarmupConfig | PydanticUndefined | Warmup convergence configuration |
| `warmup.enabled` | bool | True | Enable warmup phase before inference |
| `warmup.convergence_detection` | bool | True | Use CV-based convergence detection (false = fixed iterations) |
| `warmup.cv_threshold` | float | 0.05 | Target CV threshold (default 5%) |
| `warmup.max_prompts` | int | 50 | Maximum warmup iterations (safety cap) |
| `warmup.window_size` | int | 5 | Rolling window size for CV calculation |
| `warmup.min_prompts` | int | 5 | Minimum warmup prompts before checking convergence |

---

## Built-in Presets

Presets provide convenient defaults for common use cases.

### benchmark

```yaml
_meta:
  description: "Formal benchmark measurements"
  use_case: "Reproducible benchmarks, paper results"
max_input_tokens: 2048
max_output_tokens: 512
fp_precision: "float16"
batching:
  batch_size: 1
decoder:
  preset: "deterministic"
```

### pytorch-compatible

```yaml
_meta:
  description: "PyTorch maximum compatibility"
  use_case: "Older GPUs, debugging, model issues"
backend: "pytorch"
max_input_tokens: 2048
max_output_tokens: 512
fp_precision: "float16"
decoder:
  preset: "deterministic"
pytorch:
  attn_implementation: "eager"
  torch_compile: false
```

### pytorch-optimized

```yaml
_meta:
  description: "PyTorch with Flash Attention + compile"
  use_case: "Best PyTorch performance (Ampere+ GPU)"
backend: "pytorch"
max_input_tokens: 2048
max_output_tokens: 512
fp_precision: "float16"
decoder:
  preset: "deterministic"
pytorch:
  attn_implementation: "flash_attention_2"
  torch_compile: "reduce-overhead"
```

### pytorch-speculative

```yaml
_meta:
  description: "PyTorch with assisted generation"
  use_case: "Speculative decoding for lower latency"
backend: "pytorch"
max_input_tokens: 2048
max_output_tokens: 256
fp_precision: "float16"
decoder:
  preset: "deterministic"
pytorch:
  attn_implementation: "sdpa"
  assisted_generation:
    num_tokens: 5
```

### quick-test

```yaml
_meta:
  description: "Fast validation runs"
  use_case: "Quick sanity checks, CI testing"
max_input_tokens: 64
max_output_tokens: 32
num_processes: 1
gpus: [0]
batching:
  batch_size: 1
decoder:
  preset: "deterministic"
```

### throughput

```yaml
_meta:
  description: "Throughput-optimised testing"
  use_case: "Maximum tokens/second measurement"
max_input_tokens: 512
max_output_tokens: 256
fp_precision: "float16"
batching:
  batch_size: 8
  dynamic_batching: true
decoder:
  preset: "deterministic"
```

### vllm-low-latency

```yaml
_meta:
  description: "vLLM optimised for TTFT"
  use_case: "Interactive chat, low first-token latency"
backend: "vllm"
max_input_tokens: 512
max_output_tokens: 128
fp_precision: "float16"
decoder:
  preset: "deterministic"
vllm:
  max_num_seqs: 32
  max_num_batched_tokens: 2048
  enforce_eager: true
```

### vllm-memory-efficient

```yaml
_meta:
  description: "vLLM with FP8 KV cache"
  use_case: "Large context, memory-constrained GPUs"
backend: "vllm"
max_input_tokens: 4096
max_output_tokens: 512
fp_precision: "float16"
decoder:
  preset: "deterministic"
vllm:
  kv_cache_dtype: "fp8"
  enable_prefix_caching: true
  gpu_memory_utilization: 0.95
```

### vllm-speculative

```yaml
_meta:
  description: "vLLM with speculative decoding"
  use_case: "Lower latency via n-gram speculation"
backend: "vllm"
max_input_tokens: 2048
max_output_tokens: 256
fp_precision: "float16"
decoder:
  preset: "deterministic"
vllm:
  speculative:
    method: "ngram"
    num_tokens: 5
    ngram_max: 4
```

### vllm-throughput

```yaml
_meta:
  description: "vLLM high-throughput serving"
  use_case: "Production serving, max tokens/second"
backend: "vllm"
max_input_tokens: 2048
max_output_tokens: 512
fp_precision: "float16"
decoder:
  preset: "deterministic"
vllm:
  max_num_seqs: 512
  enable_chunked_prefill: true
  enable_prefix_caching: true
```
