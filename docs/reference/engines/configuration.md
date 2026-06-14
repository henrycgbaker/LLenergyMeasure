---
title: Engine configuration
description: Per-engine YAML configuration fields - structure, types, defaults, validation rules.
---

# Engine configuration

This page documents the per-engine YAML configuration surface. Each
experiment selects exactly one engine via the top-level `engine:` field
and configures it through a same-named block (`transformers:`, `vllm:`,
`tensorrt:`). Each engine block has two sub-sections: `engine_params:`
(constructor / load-time arguments) and `sampling_params:` (decode-time
arguments).

The fields documented below are the ones declared on the engine's
generated Pydantic model in
[`src/llenergymeasure/engines/<engine>/config.py`](https://github.com/henrycgbaker/llenergymeasure/tree/main/src/llenergymeasure/engines)
(one `Config` per engine, each exposing `EngineParams` and
`SamplingParams`). These modules are regenerated from the curated
engine data, so they are never hand-edited. Unknown fields are forwarded
to the underlying engine via `extra="allow"`, so newer engine parameters
work without an `llenergymeasure` release.

Knobs that `llenergymeasure` implements itself (prompt-batching,
`torch.compile`, TF32, autocast) have no engine-native API, so they live
in a separate top-level `harness:` section rather than the engine block.
See [Harness orchestration knobs](#harness-orchestration-knobs-harness)
below.

For study-level controls (sweeps, runners, cycles, output) see
[study-config.md](../study-config.md). For the auto-generated parameter
inventories (full type tables straight from engine introspection) see
the per-engine schema pages in this section.

## Top-level shape

A single experiment YAML has a task block, a measurement block, the
engine selector, the matching engine block, and an optional `harness:`
section:

```yaml
task:
  model: gpt2
  dataset:
    source: aienergyscore
    n_prompts: 100
    order: interleaved
  max_input_tokens: 256
  max_output_tokens: 256
  random_seed: 42

engine: transformers

measurement:
  warmup:
    enabled: true
    n_warmup: 5
  baseline:
    enabled: true
    duration_seconds: 30.0
  energy_sampler: auto

transformers:
  engine_params:
    dtype: bfloat16
    attn_implementation: sdpa
  sampling_params:
    temperature: 0.0

harness:
  transformers:
    batch_size: 4

# Optional
sampling_preset: deterministic   # deterministic | standard | creative | factual
passthrough_kwargs:
  trust_remote_code: true
```

The engine block must match the `engine:` field; mixing
`engine: vllm` with a `transformers:` block is a configuration error
(`models.py`). When `engine:` is set without a matching block, the
engine's own defaults are used.

`dtype` lives **inside** the engine block's `engine_params:`
(`transformers.engine_params.dtype`, `vllm.engine_params.dtype`,
`tensorrt.engine_params.dtype`) because each engine accepts a different
subset of dtypes (`ssot.py`). There is no top-level `dtype:` field.

`runners:` and `images:` are study-level fields and not valid in a
single-experiment YAML; they belong on `StudyConfig` (`models.py`). See
[study-config.md](../study-config.md).

## Common fields (all engines)

These fields are declared on `ExperimentConfig` and apply identically
across engines.

### `task:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str (required) | - | HuggingFace model ID or local path |
| `dataset.source` | str | `aienergyscore` | Built-in dataset alias or `.jsonl` path |
| `dataset.n_prompts` | int >= 1 | `100` | Number of prompts to load or generate |
| `dataset.order` | `interleaved` \| `grouped` \| `shuffled` | `interleaved` | Prompt ordering strategy |
| `max_input_tokens` | int >= 1 \| null | `256` | Input truncation cap; null disables |
| `max_output_tokens` | int >= 1 \| null | `256` | Output token budget; null generates to EOS or context limit |
| `random_seed` | int | `42` | Per-experiment seed for inference RNG and dataset ordering |

### `measurement:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `warmup.enabled` | bool | `true` | Enable warmup phase before measurement |
| `warmup.n_warmup` | int >= 1 | `5` | Number of full-length warmup prompts |
| `warmup.thermal_floor_seconds` | float >= 30.0 | `60.0` | Minimum post-warmup wait for thermal stabilisation |
| `warmup.convergence_detection` | bool | `false` | Enable adaptive CV-based convergence (additive to `n_warmup`) |
| `warmup.cv_threshold` | float [0.01, 0.5] | `0.05` | CV target for convergence |
| `warmup.max_prompts` | int >= 5 | `20` | Safety cap for CV mode |
| `warmup.window_size` | int >= 3 | `3` | Sliding window size for CV calculation |
| `warmup.min_prompts` | int >= 1 | `5` | Minimum prompts before checking convergence |
| `baseline.enabled` | bool | `true` | Measure idle GPU power before experiments |
| `baseline.duration_seconds` | float [5.0, 120.0] | `30.0` | Baseline measurement window |
| `baseline.strategy` | `cached` \| `validated` \| `fresh` | `validated` | Caching strategy for the baseline measurement |
| `baseline.cache_ttl_seconds` | float >= 60.0 | `7200.0` | Cached baseline lifetime (cached/validated only) |
| `baseline.validation_interval` | int >= 1 | `5` | Re-validate every N experiments (validated only) |
| `baseline.drift_threshold` | float [0.01, 0.50] | `0.10` | Drift fraction that triggers re-measurement (validated only) |
| `energy_sampler` | `auto` \| `nvml` \| `zeus` \| `codecarbon` \| null | `auto` | Energy sampler; null disables energy measurement |
| `latency_profiling` | bool | `false` | Opt-in per-token latency profiling. Captures TTFT/ITL (transformers via a streamer forced to `batch_size=1`; vLLM via decode-average ITL). Overhead may perturb energy and latency, so profiled runs are tagged in `measurement_warnings` and energy is emitted as-is. Unsupported on tensorrt. |

### Top-level optional fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sampling_preset` | `deterministic` \| `standard` \| `creative` \| `factual` \| null | null | Merges preset values into the active engine's `sampling_params:` section at parse time. Explicit YAML values take precedence. |
| `passthrough_kwargs` | dict \| null | null | Extra kwargs forwarded to the engine; keys must not collide with `ExperimentConfig` fields |

## Harness orchestration knobs (`harness:`)

`harness:` holds the per-engine knobs `llenergymeasure` implements itself
because the engine exposes no native API for them: prompt-batching in the
runner loop, the PyTorch backend globals, and `torch.autocast` context
wrapping. They are declared in `src/llenergymeasure/config/harness.py`
(hand-written and tracked, **not** in the generated engine `Config`
classes). The shape mirrors the engine blocks (`harness.transformers:`,
`harness.vllm:`, `harness.tensorrt:`), but only transformers has a
residual today; vllm and tensorrt have native batching and precision APIs
and so accept no harness fields.

### `harness.transformers:`

All fields default to null meaning "use `llenergymeasure`'s own default
at execution". `model.generate()` has no `batch_size` kwarg, and the
compile / TF32 / autocast knobs drive PyTorch calls made around the engine
rather than engine-native config.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `batch_size` | int >= 1 | `1` | Prompts per forward pass in the runner loop |
| `torch_compile` | bool | `false` | Enable `torch.compile` on the loaded model |
| `torch_compile_mode` | str | `default` | `default` \| `reduce-overhead` \| `max-autotune`. Requires `torch_compile=true`. |
| `torch_compile_backend` | str | `inductor` | `torch.compile` backend. Requires `torch_compile=true`. |
| `allow_tf32` | bool \| null | null | Allow TF32 matmul on Ampere+ via `torch.backends` |
| `autocast_enabled` | bool | `false` | Wrap generation in `torch.autocast` mixed precision |
| `autocast_dtype` | `float16` \| `bfloat16` | `bfloat16` | AMP dtype (used when `autocast_enabled=true`) |

## Transformers engine (`transformers:`)

Loads a model via `AutoModelForCausalLM.from_pretrained()` and generates
with `model.generate()`. All fields default to `null` meaning "use the
engine's own default". Unknown fields under `transformers.engine_params:`
are forwarded to the underlying HuggingFace APIs.

### `transformers.engine_params:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | `float32` \| `float16` \| `bfloat16` | `bfloat16` | Model compute dtype |
| `attn_implementation` | `sdpa` \| `flash_attention_2` \| `flash_attention_3` \| `eager` | `sdpa` | Attention kernel |
| `load_in_4bit` | bool | `false` | BitsAndBytes 4-bit quantisation |
| `load_in_8bit` | bool | `false` | BitsAndBytes 8-bit quantisation (mutually exclusive with `load_in_4bit`) |
| `bnb_4bit_compute_dtype` | `float16` \| `bfloat16` \| `float32` | `float32` | Compute dtype for 4-bit. Requires `load_in_4bit=true`. |
| `bnb_4bit_quant_type` | `nf4` \| `fp4` | `nf4` | 4-bit quantisation type. Requires `load_in_4bit=true`. |
| `bnb_4bit_use_double_quant` | bool | `false` | Double quantisation. Requires `load_in_4bit=true`. |
| `use_cache` | bool | `true` | Enable KV cache during generation |
| `cache_implementation` | `static` \| `offloaded_static` \| `sliding_window` | dynamic | KV cache strategy; `static` enables CUDA graphs. Requires `use_cache` to be true or unset. |
| `num_beams` | int >= 1 | `1` | Beam search width (1 = greedy/sampling) |
| `early_stopping` | bool | `false` | Stop beam search when all beams hit EOS |
| `length_penalty` | float | `1.0` | Beam length penalty (>1 favours shorter, <1 longer) |
| `no_repeat_ngram_size` | int >= 0 | `0` | Prevent n-gram repetition (0 = disabled) |
| `prompt_lookup_num_tokens` | int >= 1 \| null | null | Prompt-lookup speculative decoding tokens |
| `device_map` | str | `auto` | Device placement strategy |
| `max_memory` | dict | null | Per-device memory limits, e.g. `{0: "10GiB", cpu: "50GiB"}` |
| `low_cpu_mem_usage` | bool | `false` | Load weights incrementally to reduce peak CPU RAM |
| `tp_plan` | `auto` \| null | null | Native HF tensor parallelism plan (HF >= 4.50). Mutually exclusive with `device_map`; requires `torchrun` launch. |
| `tp_size` | int >= 1 | WORLD_SIZE | Tensor parallel ranks. Used only when `tp_plan` is set. |

Prompt-batching, `torch.compile`, TF32 and autocast are
`llenergymeasure`-side orchestration and live under
[`harness.transformers:`](#harnesstransformers), not here.

### `transformers.sampling_params:`

Maps to `model.generate()` kwargs. Field names mirror HuggingFace's
native conventions (`top_k=0` for disabled, `do_sample` controls greedy
vs sampling).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float [0.0, 2.0] | HF default | Sampling temperature (0 = greedy) |
| `do_sample` | bool \| null | HF default | Enable sampling. Greedy is inferred from `temperature=0` when null. |
| `top_k` | int >= 0 | `50` | HF convention: 0 = disabled |
| `top_p` | float [0.0, 1.0] | `1.0` | Nucleus sampling threshold (1.0 = disabled) |
| `repetition_penalty` | float [0.1, 10.0] | `1.0` | Repetition penalty (1.0 = no penalty) |
| `min_p` | float [0.0, 1.0] | null | Minimum probability filter |
| `min_new_tokens` | int >= 1 | HF default | Minimum output tokens |

## vLLM engine (`vllm:`)

vLLM exposes a two-API surface (`vllm.LLM()` constructor and
`SamplingParams`); the `engine_params:` / `sampling_params:` split mirrors
that. Beam search uses a nested `engine_params.beam_search:` block,
mutually exclusive with `sampling_params:`.

### `vllm.engine_params:`

Loaded once at model initialisation. Maps to the `vllm.LLM()`
constructor; unknown fields are forwarded.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | `float16` \| `bfloat16` \| `auto` | `bfloat16` | Model dtype; `auto` infers from weights. `float32` is not supported. |
| `gpu_memory_utilization` | float [0.0, 1.0) | `0.9` | GPU memory fraction reserved for KV cache |
| `swap_space` | float >= 0.0 | `4` GiB | CPU swap for KV cache offload |
| `cpu_offload_gb` | float >= 0.0 | `0` | CPU RAM in GiB to offload model weights to |
| `block_size` | `8` \| `16` \| `32` | `16` | KV cache block size in tokens |
| `kv_cache_dtype` | `auto` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` | `auto` | KV cache storage dtype; `fp8` halves VRAM on Ampere+ |
| `enforce_eager` | bool | `false` | Disable CUDA graphs, always use eager mode |
| `enable_chunked_prefill` | bool | `false` | Chunk large prefills across scheduler iterations |
| `max_num_seqs` | int >= 1 | `256` | Max concurrent sequences per scheduler iteration |
| `max_num_batched_tokens` | int >= 1 | auto | Max tokens processed per scheduler iteration. Must be >= `max_model_len` when both are set. |
| `max_model_len` | int >= 1 | model default | Max sequence length (input + output) |
| `num_scheduler_steps` | int >= 1 | `1` | Multi-step scheduling: decode N steps before returning to scheduler |
| `tensor_parallel_size` | int >= 1 | `1` | Number of GPUs to shard the model across |
| `pipeline_parallel_size` | int >= 1 | `1` | Pipeline parallel stages |
| `distributed_executor_backend` | `mp` \| `ray` | `mp` | Multi-GPU executor backend |
| `enable_prefix_caching` | bool | `false` | Automatic prefix caching for shared prompt prefixes |
| `quantization` | `awq` \| `gptq` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` \| `marlin` \| `bitsandbytes` | null | Quantisation method (requires pre-quantised checkpoint) |
| `max_seq_len_to_capture` | int >= 1 | `8192` | Maximum sequence length eligible for CUDA graph capture |
| `speculative_config.*` | sub-config | null | Speculative decoding (see below) |
| `offload_group_size` | int >= 0 | `0` | Groups of layers for CPU offloading |
| `offload_num_in_group` | int >= 1 | `1` | Layers offloaded per group |
| `offload_prefetch_step` | int >= 0 | `1` | Prefetch steps ahead for CPU offload |
| `offload_params` | list[str] | null | Specific parameter names to offload |
| `disable_custom_all_reduce` | bool | `false` | Disable custom all-reduce for multi-GPU |
| `kv_cache_memory_bytes` | int >= 1 | null | Absolute KV cache size; mutually exclusive with `gpu_memory_utilization` |
| `compilation_config` | dict | null | Full passthrough to vLLM `CompilationConfig` (no validation) |
| `attention.*` | sub-config | null | Attention backend selection (see below) |
| `beam_search.*` | sub-config | null | `vllm.BeamSearchParams()` arguments (mutually exclusive with `sampling_params`) |

### `vllm.engine_params.speculative_config:` sub-section

Mirrors vLLM's native `speculative_config` shape; unknown fields are
forwarded.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | str | null | Draft model name or path |
| `num_speculative_tokens` | int >= 1 | null | Tokens to draft per speculative step |
| `method` | str | null | Speculative method (e.g. `draft_model`, `ngram`, `medusa`, `eagle`) |

### `vllm.engine_params.attention:` sub-section

Maps to vLLM's `AttentionConfig`. All fields default to null (vLLM
auto-selects). Unknown fields are forwarded.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | str | auto | Attention backend (`flash_attn`, `flashinfer`, ...) |
| `flash_attn_version` | int | auto | Flash attention version |
| `flash_attn_max_num_splits_for_cuda_graph` | int | auto | Max splits for CUDA graph with flash attention |
| `use_prefill_decode_attention` | bool | `true` | Use prefill-decode attention |
| `use_prefill_query_quantization` | bool | `false` | Quantise queries during prefill |
| `use_cudnn_prefill` | bool | `false` | Use cuDNN for prefill |
| `disable_flashinfer_prefill` | bool | `false` | Disable FlashInfer for prefill |
| `disable_flashinfer_q_quantization` | bool | `false` | Disable FlashInfer query quantisation |
| `use_trtllm_attention` | bool | `false` | Use TensorRT-LLM attention backend |
| `use_trtllm_ragged_deepseek_prefill` | bool | `false` | Use TRT-LLM ragged DeepSeek prefill |

### `vllm.sampling_params:` sub-section

Maps to `vllm.SamplingParams()`. `max_tokens` is intentionally absent at
the study-shared level; it is bridged from `task.max_output_tokens` at
execution time (it may still be set explicitly per experiment). `top_k`
follows vLLM's `-1` for disabled convention.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float [0.0, 2.0] | vLLM default | Sampling temperature (0 = greedy) |
| `top_k` | int | `-1` | -1 = disabled (vLLM convention) |
| `top_p` | float [0.0, 1.0] | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | float [0.1, 10.0] | `1.0` | Repetition penalty |
| `min_p` | float [0.0, 1.0] | null | Minimum probability filter |
| `min_tokens` | int >= 0 | `0` | Minimum output tokens before EOS allowed |
| `presence_penalty` | float [-2.0, 2.0] | `0.0` | Penalises tokens that appear at all |
| `frequency_penalty` | float [-2.0, 2.0] | `0.0` | Penalises tokens proportional to frequency |
| `ignore_eos` | bool | `false` | Continue generating past EOS (forces full `max_tokens` generation) |
| `n` | int >= 1 | `1` | Number of output sequences per prompt |

### `vllm.engine_params.beam_search:` sub-section

Mutually exclusive with `vllm.sampling_params:`. When set, the engine
uses `BeamSearchParams` instead of `SamplingParams`. `max_tokens` is
bridged from `task.max_output_tokens`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `beam_width` | int >= 1 | vLLM default | Number of beams |
| `length_penalty` | float | `1.0` | Length penalty (>1 favours shorter, <1 longer) |
| `early_stopping` | bool | `false` | Stop when `beam_width` complete sequences are found |

## TensorRT-LLM engine (`tensorrt:`)

TensorRT-LLM compiles a model into an optimised engine on first use;
subsequent runs reuse the cached engine. Compile-time fields are baked
into the engine and changing one invalidates the cache. The nested
sub-configs under `engine_params:` mirror TRT-LLM's own API split:
`quant_config`, `kv_cache_config`, `scheduler_config`.

> **Note on the internal `backend` field.** TRT-LLM's own
> `tensorrt.engine_params.backend` field selects the runtime mode within
> TRT-LLM (`trt`, `pytorch`, `_autodeploy`). It is distinct from the
> top-level `engine:` selector that picks transformers / vLLM /
> TensorRT-LLM. The field name preserves TRT-LLM's native vocabulary.

### `tensorrt.engine_params:` (compile-time parameters)

Changing any of these triggers a fresh engine build.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_batch_size` | int >= 1 | `8` | Maximum batch size the engine accepts |
| `tensor_parallel_size` | int >= 1 | `1` | Number of GPUs to shard across |
| `pipeline_parallel_size` | int >= 1 | `1` | Pipeline parallel stages |
| `max_input_len` | int >= 1 | `1024` | Maximum input sequence length |
| `max_seq_len` | int >= 1 | `2048` | Maximum total sequence length (input + output) |
| `max_num_tokens` | int >= 1 | auto | Maximum tokens the engine handles per iteration |
| `dtype` | `float16` \| `bfloat16` | auto | Model compute dtype. `float32` is not supported. |
| `fast_build` | bool | `false` | Reduced-optimisation build for faster compilation |
| `backend` | `trt` \| `pytorch` \| `_autodeploy` | TRT-LLM auto-picks | TRT-LLM's internal runtime selector. `trt` is the AOT-compiled engine (best steady-state); `pytorch` is TRT-LLM's eager runtime (no compile, supports newer architectures); `_autodeploy` is experimental. Respects `TLLM_USE_TRT_ENGINE`. |

### `tensorrt.engine_params.quant_config:` sub-section

Quantisation is applied at engine compile time; changing any field
here triggers a recompile. Uses TRT-LLM's native `QuantAlgo` enum names.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `quant_algo` | see below | null (no quantisation) | Quantisation algorithm |
| `kv_cache_quant_algo` | `FP8` \| `INT8` | null | KV cache quantisation algorithm |

Valid `quant_algo` values:

| Value | Description |
|-------|-------------|
| `FP8` | FP8 weight + activation quantisation. Requires SM >= 8.9 (Ada Lovelace or Hopper); not supported on A100 (SM 8.0). |
| `INT8` | INT8 smooth quantisation |
| `W4A16_AWQ` | 4-bit AWQ weights, FP16 activations |
| `W4A16_GPTQ` | 4-bit GPTQ weights, FP16 activations |
| `W8A16` | 8-bit weights, FP16 activations |
| `W8A16_GPTQ` | 8-bit GPTQ weights, FP16 activations |
| `W4A8_AWQ` | 4-bit AWQ weights, INT8 activations |
| `NO_QUANT` | Explicitly disable quantisation |

### `tensorrt.engine_params.kv_cache_config:` sub-section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_block_reuse` | bool | `false` | Enable KV cache block reuse across requests |
| `free_gpu_memory_fraction` | float [0.0, 1.0] | `0.9` | Fraction of free GPU memory to allocate for KV cache |
| `max_tokens` | int >= 1 | auto | Maximum total tokens in the KV cache |
| `host_cache_size` | int >= 0 | `0` | Host (CPU) cache size in bytes for KV cache offloading (0 = disabled) |

### `tensorrt.engine_params.scheduler_config:` sub-section

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `capacity_scheduling_policy` | `GUARANTEED_NO_EVICT` \| `MAX_UTILIZATION` \| `STATIC_BATCH` | `GUARANTEED_NO_EVICT` | Scheduling capacity policy |

Policy semantics:

- `GUARANTEED_NO_EVICT` - guarantees no request eviction; may reduce
  throughput.
- `MAX_UTILIZATION` - maximises GPU utilisation; may evict requests
  under memory pressure.
- `STATIC_BATCH` - fixed batch size; useful for reproducible
  benchmarking.

### `tensorrt.sampling_params:` sub-section

Maps to `tensorrt_llm.SamplingParams`. `top_k` uses TRT-LLM's
0-for-disabled convention (matches HuggingFace, not vLLM).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float [0.0, 2.0] | TRT default | Sampling temperature (0 = greedy) |
| `top_k` | int >= 0 | TRT default | TRT-LLM convention: 0 = disabled |
| `top_p` | float [0.0, 1.0] | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | float [0.1, 10.0] | `1.0` | Repetition penalty |
| `min_p` | float [0.0, 1.0] | null | Minimum probability filter |
| `min_tokens` | int >= 0 | `0` | Minimum output tokens before EOS allowed |
| `n` | int >= 1 | `1` | Number of output sequences per prompt |
| `ignore_eos` | bool | `false` | Continue generating past EOS |

## Validation rules

The Pydantic models enforce these rules at config-load time. The full
catalogue of invalid combinations, including engine-runtime invariants
mined from upstream libraries, is in
[invalid-combos.md](invalid-combos.md).

### Cross-engine

- The engine block must match `engine:` (`models.py`).
- `passthrough_kwargs` keys must not collide with `ExperimentConfig`
  field names (`models.py`).

### Transformers

- `load_in_4bit` and `load_in_8bit` are mutually exclusive.
- `torch_compile_mode` and `torch_compile_backend` require
  `harness.transformers.torch_compile=true` (`harness.py`).
- `bnb_4bit_*` fields require `load_in_4bit=true`.
- `cache_implementation` requires `use_cache` to be true or unset.
- `tp_plan` and `device_map` are mutually exclusive.
- `attn_implementation` `flash_attention_2`/`flash_attention_3` requires
  `dtype` `float16` or `bfloat16` (`models.py`).

### vLLM

- `kv_cache_memory_bytes` and `gpu_memory_utilization` are mutually
  exclusive.
- `max_num_batched_tokens` must be `>= max_model_len` when both are set.
- `beam_search` and `sampling_params` sections are mutually exclusive.
- `dtype: float32` is rejected by the field's `Literal` type.

### TensorRT-LLM

- `dtype: float32` is rejected by the field's `Literal` type.
- `quant_algo: FP8` requires SM >= 8.9 (Ada Lovelace or Hopper); not
  supported on A100 (SM 8.0). Hardware-side check; runtime error.

### Engine x dtype matrix

From `ssot.DTYPE_SUPPORT` (`ssot.py`):

| Engine | `float32` | `float16` | `bfloat16` |
|--------|-----------|-----------|------------|
| `transformers` | yes | yes | yes |
| `vllm` | no | yes | yes |
| `tensorrt` | no | yes | yes |

## Worked examples

### Minimal Transformers experiment

```yaml
task:
  model: gpt2
engine: transformers
```

All other fields fall back to defaults: `aienergyscore` dataset, 100
prompts, 256-token input/output caps, `bfloat16`, `sdpa` attention,
`auto` energy sampler.

### Transformers with quantisation and compilation

```yaml
task:
  model: meta-llama/Llama-2-7b-hf
  dataset:
    n_prompts: 50

engine: transformers

transformers:
  engine_params:
    dtype: bfloat16
    load_in_4bit: true
    bnb_4bit_compute_dtype: bfloat16
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true
    attn_implementation: flash_attention_2

harness:
  transformers:
    batch_size: 4
```

### vLLM with prefix caching and FP8 KV cache

```yaml
task:
  model: meta-llama/Llama-2-7b-hf
  max_input_tokens: 1024
  max_output_tokens: 256

engine: vllm

vllm:
  engine_params:
    dtype: bfloat16
    gpu_memory_utilization: 0.9
    enable_prefix_caching: true
    kv_cache_dtype: fp8
    block_size: 16
  sampling_params:
    temperature: 0.0
    n: 1
```

### vLLM with beam search

`vllm.engine_params.beam_search:` replaces `vllm.sampling_params:`; the
two are mutually exclusive.

```yaml
task:
  model: gpt2

engine: vllm

vllm:
  engine_params:
    enforce_eager: false
    beam_search:
      beam_width: 4
      length_penalty: 1.0
      early_stopping: false
```

### TensorRT-LLM with AWQ quantisation

```yaml
task:
  model: meta-llama/Llama-2-7b-hf

engine: tensorrt

tensorrt:
  engine_params:
    dtype: bfloat16
    max_batch_size: 8
    max_input_len: 1024
    max_seq_len: 2048
    tensor_parallel_size: 1
    quant_config:
      quant_algo: W4A16_AWQ
    kv_cache_config:
      free_gpu_memory_fraction: 0.9
      enable_block_reuse: true
    scheduler_config:
      capacity_scheduling_policy: GUARANTEED_NO_EVICT
```

### Sampling preset (preset values merged into the engine's sampling section)

```yaml
task:
  model: gpt2
engine: transformers
sampling_preset: deterministic   # sets temperature: 0.0 under transformers.sampling_params
```

`sampling_preset` is expanded at parse time into the active engine's
`sampling_params:` sub-section. Explicit YAML values take precedence over
preset values.

## See also

- [study-config.md](../study-config.md) - sweep grammar, runners,
  cycles, output configuration.
- [schema-transformers.md](schema-transformers.md),
  [schema-vllm.md](schema-vllm.md),
  [schema-tensorrt.md](schema-tensorrt.md) - auto-generated full
  parameter inventories straight from each engine's introspection.
- [curation-transformers.md](curation-transformers.md),
  [curation-vllm.md](curation-vllm.md),
  [curation-tensorrt.md](curation-tensorrt.md) - which engine
  parameters are explicitly modelled vs forwarded via `extra="allow"`.
- [invariants-transformers.md](invariants-transformers.md),
  [invariants-vllm.md](invariants-vllm.md),
  [invariants-tensorrt.md](invariants-tensorrt.md) - mined runtime
  invariants per engine.
- [invalid-combos.md](invalid-combos.md) - catalogue of rejected
  parameter combinations.
