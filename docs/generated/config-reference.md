# Configuration Reference

> This file is auto-generated from Pydantic models. Do not edit manually.
> Generated: 2026-01-21 11:29:19

## Table of Contents

- [adapter](#adapter)
- [backend](#backend)
- [batching](#batching)
- [config_name](#config-name)
- [cycle_id](#cycle-id)
- [decode_token_to_text](#decode-token-to-text)
- [decoder](#decoder)
- [extra_metadata](#extra-metadata)
- [fp_precision](#fp-precision)
- [gpus](#gpus)
- [inference_type](#inference-type)
- [io](#io)
- [is_encoder_decoder](#is-encoder-decoder)
- [max_input_tokens](#max-input-tokens)
- [max_output_tokens](#max-output-tokens)
- [min_output_tokens](#min-output-tokens)
- [model_name](#model-name)
- [num_cycles](#num-cycles)
- [num_input_prompts](#num-input-prompts)
- [num_processes](#num-processes)
- [parallelism](#parallelism)
- [prompts](#prompts)
- [pytorch](#pytorch)
- [quantization](#quantization)
- [query_rate](#query-rate)
- [random_seed](#random-seed)
- [save_outputs](#save-outputs)
- [schedule](#schedule)
- [sharding](#sharding)
- [streaming](#streaming)
- [streaming_warmup_requests](#streaming-warmup-requests)
- [task_type](#task-type)
- [tensorrt](#tensorrt)
- [traffic_simulation](#traffic-simulation)
- [vllm](#vllm)

---

## adapter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `adapter` | str \| None | None | LoRA adapter: HuggingFace Hub ID or local path |

## backend

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | Literal['pytorch', 'tensorrt', 'vllm'] | 'pytorch' | Inference backend |

## batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `batching` | BatchingConfig | PydanticUndefined | Batching configuration |
| `batching.batch_size` | int | 1 | Max prompts per batch |
| `batching.strategy` | Literal['static', 'dynamic', 'sorted_static', 'sorted_dynamic'] | 'static' | Batching strategy (MLPerf terminology) |
| `batching.max_tokens_per_batch` | int \| None | None | Max tokens per batch (for dynamic strategies). Defaults to max_input_tokens. |
| `batching.dynamic_batching` | bool | False | [Deprecated] Use strategy='dynamic' instead. Kept for backwards compat. |

## config_name

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_name` | str | PydanticUndefined *(required)* | Unique config identifier |

## cycle_id

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cycle_id` | int \| None | None | Experiment cycle ID |

## decode_token_to_text

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decode_token_to_text` | bool | False | Decode tokens to text |

## decoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decoder` | DecoderConfig | PydanticUndefined | Decoder/generation configuration |
| `decoder.temperature` | float | 1.0 | Sampling temperature (0=greedy) |
| `decoder.do_sample` | bool | True | Enable sampling (ignored if temp=0) |
| `decoder.top_p` | float | 1.0 | Top-p nucleus sampling (1.0=disabled) |
| `decoder.top_k` | int | 50 | Top-k sampling (0=disabled) |
| `decoder.min_p` | float | 0.0 | Min probability relative to top token (0=disabled) |
| `decoder.repetition_penalty` | float | 1.0 | Repetition penalty (1.0=no penalty) |
| `decoder.no_repeat_ngram_size` | int | 0 | Prevent n-gram repetition (0=disabled) |
| `decoder.beam_search` | BeamSearchConfig | PydanticUndefined | Beam search configuration |
| `decoder.beam_search.enabled` | bool | False | Enable beam search (disables sampling) |
| `decoder.beam_search.num_beams` | int | 1 | Beam width (1=greedy, >1=beam search) |
| `decoder.beam_search.length_penalty` | float | 1.0 | Exponential length penalty (>1 favours longer, <1 favours shorter) |
| `decoder.beam_search.early_stopping` | bool | False | Stop when num_beams best sequences complete |
| `decoder.beam_search.no_repeat_ngram_size` | int | 0 | Prevent n-gram repetition within beam (0=disabled) |
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

## inference_type

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inference_type` | Literal['pure_generative', 'reasoning'] | 'pure_generative' | Inference type |

## io

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `io` | IOConfig | PydanticUndefined | I/O paths configuration |
| `io.results_dir` | str \| None | None | Results output directory (overrides .env default) |

## is_encoder_decoder

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_encoder_decoder` | bool | False | Is encoder-decoder model |

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
| `num_cycles` | int | 1 | Number of cycles for statistical robustness (1-10) |

## num_input_prompts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_input_prompts` | int | 1 | Number of prompts |

## num_processes

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_processes` | int | 1 | [Deprecated] Number of processes. Use parallelism.degree instead. |

## parallelism

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `parallelism` | ParallelismConfig | PydanticUndefined | Unified parallelism configuration for multi-GPU inference |
| `parallelism.strategy` | Literal['none', 'tensor_parallel', 'pipeline_parallel', 'data_parallel'] | 'none' | Parallelism strategy |
| `parallelism.degree` | int | 1 | Number of GPUs/workers for parallelism |
| `parallelism.tp_plan` | Literal['auto'] \| None | None | Tensor parallel plan ('auto' uses model's predefined config) |

## prompts

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompts` | Annotated \| None | None | Prompt source: file or huggingface dataset |

## pytorch

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pytorch` | PyTorchConfig \| None | None | PyTorch-specific configuration (only used when backend=pytorch) |

## quantization

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `quantization` | QuantizationConfig | PydanticUndefined | Quantization configuration |
| `quantization.quantization` | bool | False | Enable quantization |
| `quantization.load_in_4bit` | bool | False | Load in 4-bit (BNB) |
| `quantization.load_in_8bit` | bool | False | Load in 8-bit (BNB) |
| `quantization.bnb_4bit_compute_dtype` | str | 'float16' | Compute dtype for 4-bit |
| `quantization.bnb_4bit_quant_type` | str | 'nf4' | Quantization type (nf4, fp4) |
| `quantization.bnb_4bit_use_double_quant` | bool | False | Use double quantization |

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

## sharding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sharding` | ShardingConfig | PydanticUndefined | [Deprecated] Use parallelism instead. Legacy sharding configuration. |
| `sharding.strategy` | Literal['none', 'tensor_parallel', 'pipeline_parallel'] | 'none' | Sharding strategy |
| `sharding.num_shards` | int | 1 | Number of GPUs for parallelism |
| `sharding.tp_plan` | Literal['auto'] \| None | None | Tensor parallel plan ('auto' uses model's predefined config) |

## streaming

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming` | bool | False | Enable streaming mode for TTFT/ITL latency measurement. Also a testable parameter - streaming may affect energy profile. |

## streaming_warmup_requests

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `streaming_warmup_requests` | int | 5 | Warmup requests before streaming measurement (excluded from stats) |

## task_type

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_type` | Literal['text_generation', 'translation', 'summarisation'] | 'text_generation' | Task type |

## tensorrt

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tensorrt` | TensorRTConfig \| None | None | TensorRT-LLM configuration (only used when backend=tensorrt) |

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

---

## Built-in Presets

Presets provide convenient defaults for common use cases.

### benchmark

```yaml
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
