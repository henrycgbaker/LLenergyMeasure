---
title: Engine configuration
description: Per-engine YAML configuration fields - structure, types, defaults, validation rules.
---

# Engine configuration

This page documents the per-engine YAML configuration surface. Each
experiment selects exactly one engine via the top-level `engine:` field
and configures it through a same-named block (`transformers:`, `vllm:`,
`tensorrt:`). The fields documented below are the ones declared on the
engine's Pydantic model in
[`src/llenergymeasure/config/engine_configs.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/config/engine_configs.py);
unknown fields are forwarded to the underlying engine via `extra="allow"`,
so newer engine parameters work without an `llenergymeasure` release.

For study-level controls (sweeps, runners, cycles, output) see
[study-config.md](../study-config.md). For the auto-generated parameter
inventories (full type tables straight from engine introspection) see
the per-engine schema pages in this section.

## Top-level shape

A single experiment YAML has three blocks plus the engine selector and
an optional engine-specific section:

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
  batch_size: 4
  dtype: bfloat16
  attn_implementation: sdpa

# Optional
sampling_preset: deterministic   # deterministic | standard | creative | factual
passthrough_kwargs:
  trust_remote_code: true
```

The engine-specific section must match the `engine:` field; mixing
`engine: vllm` with a `transformers:` section is a configuration error
(`models.py:441-464`). When `engine:` is set without a matching section,
the engine's own defaults are used.

`dtype` lives **inside** the engine section (`transformers.dtype`,
`vllm.dtype`, `tensorrt.dtype`) because each engine accepts a different
subset of dtypes (`ssot.py:156-160`). There is no top-level `dtype:`
field.

`runners:` and `images:` are study-level fields and not valid in a
single-experiment YAML; they belong on `StudyConfig`
(`models.py:740-759`). See [study-config.md](../study-config.md).

## Common fields (all engines)

These fields are declared on `ExperimentConfig` and apply identically
across engines.

### `task:`

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `model` | str (required) | - | HuggingFace model ID or local path | `models.py:265-270` |
| `dataset.source` | str | `aienergyscore` | Built-in dataset alias or `.jsonl` path | `models.py:198-203` |
| `dataset.n_prompts` | int >= 1 | `100` | Number of prompts to load or generate | `models.py:204-209` |
| `dataset.order` | `interleaved` \| `grouped` \| `shuffled` | `interleaved` | Prompt ordering strategy | `models.py:210-216` |
| `max_input_tokens` | int >= 1 \| null | `256` | Input truncation cap; null disables | `models.py:276-284` |
| `max_output_tokens` | int >= 1 \| null | `256` | Output token budget; null generates to EOS or context limit | `models.py:285-293` |
| `random_seed` | int | `42` | Per-experiment seed for inference RNG and dataset ordering | `models.py:294-297` |

### `measurement:`

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `warmup.enabled` | bool | `true` | Enable warmup phase before measurement | `models.py:79` |
| `warmup.n_warmup` | int >= 1 | `5` | Number of full-length warmup prompts | `models.py:81-85` |
| `warmup.thermal_floor_seconds` | float >= 30.0 | `60.0` | Minimum post-warmup wait for thermal stabilisation | `models.py:86-90` |
| `warmup.convergence_detection` | bool | `false` | Enable adaptive CV-based convergence (additive to `n_warmup`) | `models.py:93-96` |
| `warmup.cv_threshold` | float [0.01, 0.5] | `0.05` | CV target for convergence | `models.py:97-102` |
| `warmup.max_prompts` | int >= 5 | `20` | Safety cap for CV mode | `models.py:103-107` |
| `warmup.window_size` | int >= 3 | `3` | Sliding window size for CV calculation | `models.py:108-112` |
| `warmup.min_prompts` | int >= 1 | `5` | Minimum prompts before checking convergence | `models.py:113-117` |
| `baseline.enabled` | bool | `true` | Measure idle GPU power before experiments | `models.py:142` |
| `baseline.duration_seconds` | float [5.0, 120.0] | `30.0` | Baseline measurement window | `models.py:143-148` |
| `baseline.strategy` | `cached` \| `validated` \| `fresh` | `validated` | Caching strategy for the baseline measurement | `models.py:149-156` |
| `baseline.cache_ttl_seconds` | float >= 60.0 | `7200.0` | Cached baseline lifetime (cached/validated only) | `models.py:157-164` |
| `baseline.validation_interval` | int >= 1 | `5` | Re-validate every N experiments (validated only) | `models.py:165-171` |
| `baseline.drift_threshold` | float [0.01, 0.50] | `0.10` | Drift fraction that triggers re-measurement (validated only) | `models.py:172-180` |
| `energy_sampler` | `auto` \| `nvml` \| `zeus` \| `codecarbon` \| null | `auto` | Energy sampler; null disables energy measurement | `models.py:320-327` |

### Top-level optional fields

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `sampling_preset` | `deterministic` \| `standard` \| `creative` \| `factual` \| null | null | Merges preset values into the active engine's `sampling:` section at parse time. Explicit YAML values take precedence. | `models.py:367-375`, `ssot.py:27-32` |
| `passthrough_kwargs` | dict \| null | null | Extra kwargs forwarded to the engine; keys must not collide with `ExperimentConfig` fields | `models.py:395-399` |

## Transformers engine (`transformers:`)

Loads a model via `AutoModelForCausalLM.from_pretrained()` and generates
with `model.generate()`. All fields default to `null` meaning "use the
engine's own default". Unknown fields under `transformers:` are
forwarded to the underlying HuggingFace APIs (`engine_configs.py:94`).

### Parameters

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `batch_size` | int >= 1 | `1` | Prompts per forward pass | `engine_configs.py:100-104` |
| `dtype` | `float32` \| `float16` \| `bfloat16` | `bfloat16` | Model compute dtype | `engine_configs.py:110-113` |
| `attn_implementation` | `sdpa` \| `flash_attention_2` \| `flash_attention_3` \| `eager` | `sdpa` | Attention kernel | `engine_configs.py:119-124` |
| `torch_compile` | bool | `false` | Enable `torch.compile` | `engine_configs.py:130-133` |
| `torch_compile_mode` | str | `default` | `default` \| `reduce-overhead` \| `max-autotune`. Requires `torch_compile=true`. | `engine_configs.py:134-137` |
| `torch_compile_backend` | str | `inductor` | `torch.compile` backend. Requires `torch_compile=true`. | `engine_configs.py:138-141` |
| `load_in_4bit` | bool | `false` | BitsAndBytes 4-bit quantisation | `engine_configs.py:147-150` |
| `load_in_8bit` | bool | `false` | BitsAndBytes 8-bit quantisation (mutually exclusive with `load_in_4bit`) | `engine_configs.py:151-154` |
| `bnb_4bit_compute_dtype` | `float16` \| `bfloat16` \| `float32` | `float32` | Compute dtype for 4-bit. Requires `load_in_4bit=true`. | `engine_configs.py:155-158` |
| `bnb_4bit_quant_type` | `nf4` \| `fp4` | `nf4` | 4-bit quantisation type. Requires `load_in_4bit=true`. | `engine_configs.py:159-162` |
| `bnb_4bit_use_double_quant` | bool | `false` | Double quantisation. Requires `load_in_4bit=true`. | `engine_configs.py:163-166` |
| `use_cache` | bool | `true` | Enable KV cache during generation | `engine_configs.py:172-175` |
| `cache_implementation` | `static` \| `offloaded_static` \| `sliding_window` | dynamic | KV cache strategy; `static` enables CUDA graphs. Requires `use_cache` to be true or unset. | `engine_configs.py:176-179` |
| `num_beams` | int >= 1 | `1` | Beam search width (1 = greedy/sampling) | `engine_configs.py:185-189` |
| `early_stopping` | bool | `false` | Stop beam search when all beams hit EOS | `engine_configs.py:190-193` |
| `length_penalty` | float | `1.0` | Beam length penalty (>1 favours shorter, <1 longer) | `engine_configs.py:194-197` |
| `no_repeat_ngram_size` | int >= 0 | `0` | Prevent n-gram repetition (0 = disabled) | `engine_configs.py:203-207` |
| `prompt_lookup_num_tokens` | int >= 1 \| null | null | Prompt-lookup speculative decoding tokens | `engine_configs.py:213-217` |
| `device_map` | str | `auto` | Device placement strategy | `engine_configs.py:223-226` |
| `max_memory` | dict | null | Per-device memory limits, e.g. `{0: "10GiB", cpu: "50GiB"}` | `engine_configs.py:227-230` |
| `low_cpu_mem_usage` | bool | `false` | Load weights incrementally to reduce peak CPU RAM | `engine_configs.py:248-251` |
| `allow_tf32` | bool \| null | null | Allow TF32 matmul on Ampere+ | `engine_configs.py:236-239` |
| `autocast_enabled` | bool | `false` | Enable `torch.autocast` during generation | `engine_configs.py:240-243` |
| `autocast_dtype` | `float16` \| `bfloat16` | `bfloat16` | AMP dtype (used when `autocast_enabled=true`) | `engine_configs.py:244-247` |
| `tp_plan` | `auto` \| null | null | Native HF tensor parallelism plan (HF >= 4.50). Mutually exclusive with `device_map`; requires `torchrun` launch. | `engine_configs.py:257-264` |
| `tp_size` | int >= 1 | WORLD_SIZE | Tensor parallel ranks. Used only when `tp_plan` is set. | `engine_configs.py:265-273` |
| `sampling.*` | sub-config | null | `model.generate()` sampling kwargs (see below) | `engine_configs.py:279-285` |

### `transformers.sampling:` sub-section

Maps to `model.generate()` kwargs. Field names mirror HuggingFace's
native conventions (`top_k=0` for disabled, `do_sample` controls greedy
vs sampling) (`engine_configs.py:41-79`).

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `temperature` | float [0.0, 2.0] | HF default | Sampling temperature (0 = greedy) | `engine_configs.py:54-56` |
| `do_sample` | bool \| null | HF default | Enable sampling. Greedy is inferred from `temperature=0` when null. | `engine_configs.py:57-62` |
| `top_k` | int >= 0 | `50` | HF convention: 0 = disabled | `engine_configs.py:63-67` |
| `top_p` | float [0.0, 1.0] | `1.0` | Nucleus sampling threshold (1.0 = disabled) | `engine_configs.py:68-70` |
| `repetition_penalty` | float [0.1, 10.0] | `1.0` | Repetition penalty (1.0 = no penalty) | `engine_configs.py:71-73` |
| `min_p` | float [0.0, 1.0] | null | Minimum probability filter | `engine_configs.py:74-76` |
| `min_new_tokens` | int >= 1 | HF default | Minimum output tokens | `engine_configs.py:77-79` |

## vLLM engine (`vllm:`)

vLLM exposes a two-API surface (`vllm.LLM()` constructor and
`SamplingParams`); the configuration mirrors that split. Beam search
uses a separate `vllm.beam_search:` block, mutually exclusive with
`vllm.sampling:` (`engine_configs.py:842-850`).

### Top-level vLLM fields

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `dtype` | `float16` \| `bfloat16` \| `auto` | `bfloat16` | Model dtype; `auto` infers from weights. `float32` is not supported. | `engine_configs.py:817-823` |
| `engine.*` | sub-config | null | `vllm.LLM()` constructor arguments | `engine_configs.py:824-827` |
| `sampling.*` | sub-config | null | `vllm.SamplingParams()` arguments | `engine_configs.py:828-833` |
| `beam_search.*` | sub-config | null | `vllm.BeamSearchParams()` arguments (mutually exclusive with `sampling`) | `engine_configs.py:834-840` |

### `vllm.engine:` sub-section

Loaded once at model initialisation. Unknown fields are forwarded to
`vllm.LLM()` (`engine_configs.py:434`).

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `gpu_memory_utilization` | float [0.0, 1.0) | `0.9` | GPU memory fraction reserved for KV cache | `engine_configs.py:440-447` |
| `swap_space` | float >= 0.0 | `4` GiB | CPU swap for KV cache offload | `engine_configs.py:448-455` |
| `cpu_offload_gb` | float >= 0.0 | `0` | CPU RAM in GiB to offload model weights to | `engine_configs.py:456-463` |
| `block_size` | `8` \| `16` \| `32` | `16` | KV cache block size in tokens | `engine_configs.py:469-475` |
| `kv_cache_dtype` | `auto` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` | `auto` | KV cache storage dtype; `fp8` halves VRAM on Ampere+ | `engine_configs.py:476-482` |
| `enforce_eager` | bool | `false` | Disable CUDA graphs, always use eager mode | `engine_configs.py:488-494` |
| `enable_chunked_prefill` | bool | `false` | Chunk large prefills across scheduler iterations | `engine_configs.py:495-501` |
| `max_num_seqs` | int >= 1 | `256` | Max concurrent sequences per scheduler iteration | `engine_configs.py:507-514` |
| `max_num_batched_tokens` | int >= 1 | auto | Max tokens processed per scheduler iteration. Must be >= `max_model_len` when both are set. | `engine_configs.py:515-522` |
| `max_model_len` | int >= 1 | model default | Max sequence length (input + output) | `engine_configs.py:523-530` |
| `num_scheduler_steps` | int >= 1 | `1` | Multi-step scheduling: decode N steps before returning to scheduler | `engine_configs.py:531-535` |
| `tensor_parallel_size` | int >= 1 | `1` | Number of GPUs to shard the model across | `engine_configs.py:541-547` |
| `pipeline_parallel_size` | int >= 1 | `1` | Pipeline parallel stages | `engine_configs.py:548-552` |
| `distributed_executor_backend` | `mp` \| `ray` | `mp` | Multi-GPU executor backend | `engine_configs.py:553-556` |
| `enable_prefix_caching` | bool | `false` | Automatic prefix caching for shared prompt prefixes | `engine_configs.py:562-565` |
| `quantization` | `awq` \| `gptq` \| `fp8` \| `fp8_e5m2` \| `fp8_e4m3` \| `marlin` \| `bitsandbytes` | null | Quantisation method (requires pre-quantised checkpoint) | `engine_configs.py:566-571` |
| `max_seq_len_to_capture` | int >= 1 | `8192` | Maximum sequence length eligible for CUDA graph capture | `engine_configs.py:577-581` |
| `speculative_config.*` | sub-config | null | Speculative decoding (see below) | `engine_configs.py:587-593` |
| `offload_group_size` | int >= 0 | `0` | Groups of layers for CPU offloading | `engine_configs.py:599-603` |
| `offload_num_in_group` | int >= 1 | `1` | Layers offloaded per group | `engine_configs.py:604-608` |
| `offload_prefetch_step` | int >= 0 | `1` | Prefetch steps ahead for CPU offload | `engine_configs.py:609-613` |
| `offload_params` | list[str] | null | Specific parameter names to offload | `engine_configs.py:614-617` |
| `disable_custom_all_reduce` | bool | `false` | Disable custom all-reduce for multi-GPU | `engine_configs.py:623-626` |
| `kv_cache_memory_bytes` | int >= 1 | null | Absolute KV cache size; mutually exclusive with `gpu_memory_utilization` | `engine_configs.py:632-639` |
| `compilation_config` | dict | null | Full passthrough to vLLM `CompilationConfig` (no validation) | `engine_configs.py:645-651` |
| `attention.*` | sub-config | null | Attention backend selection (see below) | `engine_configs.py:657-660` |

### `vllm.engine.speculative_config:` sub-section

Mirrors vLLM's native `speculative_config` shape; unknown fields are
forwarded (`engine_configs.py:353`).

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `model` | str | null | Draft model name or path | `engine_configs.py:355-358` |
| `num_speculative_tokens` | int >= 1 | null | Tokens to draft per speculative step | `engine_configs.py:359-363` |
| `method` | str | null | Speculative method (e.g. `draft_model`, `ngram`, `medusa`, `eagle`) | `engine_configs.py:364-371` |

### `vllm.engine.attention:` sub-section

Maps to vLLM's `AttentionConfig`. All fields default to null (vLLM
auto-selects). Unknown fields are forwarded
(`engine_configs.py:382`).

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `backend` | str | auto | Attention backend (`flash_attn`, `flashinfer`, ...) | `engine_configs.py:384-387` |
| `flash_attn_version` | int | auto | Flash attention version | `engine_configs.py:388-391` |
| `flash_attn_max_num_splits_for_cuda_graph` | int | auto | Max splits for CUDA graph with flash attention | `engine_configs.py:392-395` |
| `use_prefill_decode_attention` | bool | `true` | Use prefill-decode attention | `engine_configs.py:396-399` |
| `use_prefill_query_quantization` | bool | `false` | Quantise queries during prefill | `engine_configs.py:400-403` |
| `use_cudnn_prefill` | bool | `false` | Use cuDNN for prefill | `engine_configs.py:404-407` |
| `disable_flashinfer_prefill` | bool | `false` | Disable FlashInfer for prefill | `engine_configs.py:408-411` |
| `disable_flashinfer_q_quantization` | bool | `false` | Disable FlashInfer query quantisation | `engine_configs.py:412-415` |
| `use_trtllm_attention` | bool | `false` | Use TensorRT-LLM attention backend | `engine_configs.py:416-419` |
| `use_trtllm_ragged_deepseek_prefill` | bool | `false` | Use TRT-LLM ragged DeepSeek prefill | `engine_configs.py:420-423` |

### `vllm.sampling:` sub-section

Maps to `vllm.SamplingParams()`. `max_tokens` is intentionally absent;
it is bridged from `task.max_output_tokens` at execution time
(`engine_configs.py:702-704`). `top_k` follows vLLM's `-1` for
disabled convention.

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `temperature` | float [0.0, 2.0] | vLLM default | Sampling temperature (0 = greedy) | `engine_configs.py:710-712` |
| `top_k` | int | `-1` | -1 = disabled (vLLM convention) | `engine_configs.py:713-716` |
| `top_p` | float [0.0, 1.0] | `1.0` | Nucleus sampling threshold | `engine_configs.py:717-719` |
| `repetition_penalty` | float [0.1, 10.0] | `1.0` | Repetition penalty | `engine_configs.py:720-722` |
| `min_p` | float [0.0, 1.0] | null | Minimum probability filter | `engine_configs.py:723-725` |
| `min_tokens` | int >= 0 | `0` | Minimum output tokens before EOS allowed | `engine_configs.py:728-732` |
| `presence_penalty` | float [-2.0, 2.0] | `0.0` | Penalises tokens that appear at all | `engine_configs.py:733-741` |
| `frequency_penalty` | float [-2.0, 2.0] | `0.0` | Penalises tokens proportional to frequency | `engine_configs.py:742-750` |
| `ignore_eos` | bool | `false` | Continue generating past EOS (forces full `max_tokens` generation) | `engine_configs.py:751-757` |
| `n` | int >= 1 | `1` | Number of output sequences per prompt | `engine_configs.py:758-762` |

### `vllm.beam_search:` sub-section

Mutually exclusive with `vllm.sampling:`. When set, the engine uses
`BeamSearchParams` instead of `SamplingParams`
(`engine_configs.py:842-850`). `max_tokens` is bridged from
`task.max_output_tokens`.

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `beam_width` | int >= 1 | vLLM default | Number of beams | `engine_configs.py:779-783` |
| `length_penalty` | float | `1.0` | Length penalty (>1 favours shorter, <1 longer) | `engine_configs.py:784-787` |
| `early_stopping` | bool | `false` | Stop when `beam_width` complete sequences are found | `engine_configs.py:788-791` |

## TensorRT-LLM engine (`tensorrt:`)

TensorRT-LLM compiles a model into an optimised engine on first use;
subsequent runs reuse the cached engine. Compile-time fields are baked
into the engine and changing one invalidates the cache. The nested
sub-configs mirror TRT-LLM's own API split: `quant_config`,
`kv_cache_config`, `scheduler_config`, `sampling`
(`engine_configs.py:991-995`).

> **Note on the internal `backend` field.** TRT-LLM's own
> `tensorrt.backend` field selects the runtime mode within TRT-LLM
> (`trt`, `pytorch`, `_autodeploy`). It is distinct from the top-level
> `engine:` selector that picks transformers / vLLM / TensorRT-LLM. The
> field name preserves TRT-LLM's native vocabulary
> (`engine_configs.py:1069-1079`).

### Compile-time parameters

Changing any of these triggers a fresh engine build.

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `max_batch_size` | int >= 1 | `8` | Maximum batch size the engine accepts | `engine_configs.py:1025-1029` |
| `tensor_parallel_size` | int >= 1 | `1` | Number of GPUs to shard across | `engine_configs.py:1030-1038` |
| `pipeline_parallel_size` | int >= 1 | `1` | Pipeline parallel stages | `engine_configs.py:1039-1043` |
| `max_input_len` | int >= 1 | `1024` | Maximum input sequence length | `engine_configs.py:1044-1048` |
| `max_seq_len` | int >= 1 | `2048` | Maximum total sequence length (input + output) | `engine_configs.py:1049-1053` |
| `max_num_tokens` | int >= 1 | auto | Maximum tokens the engine handles per iteration | `engine_configs.py:1054-1058` |
| `dtype` | `float16` \| `bfloat16` | auto | Model compute dtype. `float32` is not supported. | `engine_configs.py:1059-1064` |
| `fast_build` | bool | `false` | Reduced-optimisation build for faster compilation | `engine_configs.py:1065-1068` |
| `backend` | `trt` \| `pytorch` \| `_autodeploy` | TRT-LLM auto-picks | TRT-LLM's internal runtime selector. `trt` is the AOT-compiled engine (best steady-state); `pytorch` is TRT-LLM's eager runtime (no compile, supports newer architectures); `_autodeploy` is experimental. Respects `TLLM_USE_TRT_ENGINE`. | `engine_configs.py:1069-1079` |

### `tensorrt.quant_config:` sub-section

Quantisation is applied at engine compile time; changing any field
here triggers a recompile. Uses TRT-LLM's native `QuantAlgo` enum
names (`engine_configs.py:858-887`).

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `quant_algo` | see below | null (no quantisation) | Quantisation algorithm | `engine_configs.py:868-883` |
| `kv_cache_quant_algo` | `FP8` \| `INT8` | null | KV cache quantisation algorithm | `engine_configs.py:884-887` |

Valid `quant_algo` values:

| Value | Description | Source |
|-------|-------------|--------|
| `FP8` | FP8 weight + activation quantisation. Requires SM >= 8.9 (Ada Lovelace or Hopper); not supported on A100 (SM 8.0). | `engine_configs.py:870` |
| `INT8` | INT8 smooth quantisation | `engine_configs.py:871` |
| `W4A16_AWQ` | 4-bit AWQ weights, FP16 activations | `engine_configs.py:872` |
| `W4A16_GPTQ` | 4-bit GPTQ weights, FP16 activations | `engine_configs.py:873` |
| `W8A16` | 8-bit weights, FP16 activations | `engine_configs.py:874` |
| `W8A16_GPTQ` | 8-bit GPTQ weights, FP16 activations | `engine_configs.py:875` |
| `W4A8_AWQ` | 4-bit AWQ weights, INT8 activations | `engine_configs.py:876` |
| `NO_QUANT` | Explicitly disable quantisation | `engine_configs.py:877` |

### `tensorrt.kv_cache_config:` sub-section

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `enable_block_reuse` | bool | `false` | Enable KV cache block reuse across requests | `engine_configs.py:895-898` |
| `free_gpu_memory_fraction` | float [0.0, 1.0] | `0.9` | Fraction of free GPU memory to allocate for KV cache | `engine_configs.py:899-904` |
| `max_tokens` | int >= 1 | auto | Maximum total tokens in the KV cache | `engine_configs.py:905-909` |
| `host_cache_size` | int >= 0 | `0` | Host (CPU) cache size in bytes for KV cache offloading (0 = disabled) | `engine_configs.py:910-914` |

### `tensorrt.scheduler_config:` sub-section

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `capacity_scheduling_policy` | `GUARANTEED_NO_EVICT` \| `MAX_UTILIZATION` \| `STATIC_BATCH` | `GUARANTEED_NO_EVICT` | Scheduling capacity policy | `engine_configs.py:922-932` |

Policy semantics:

- `GUARANTEED_NO_EVICT` - guarantees no request eviction; may reduce
  throughput.
- `MAX_UTILIZATION` - maximises GPU utilisation; may evict requests
  under memory pressure.
- `STATIC_BATCH` - fixed batch size; useful for reproducible
  benchmarking.

### `tensorrt.sampling:` sub-section

Maps to `tensorrt_llm.SamplingParams`. `top_k` uses TRT-LLM's
0-for-disabled convention (matches HuggingFace, not vLLM)
(`engine_configs.py:935-981`).

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `temperature` | float [0.0, 2.0] | TRT default | Sampling temperature (0 = greedy) | `engine_configs.py:949-951` |
| `top_k` | int >= 0 | TRT default | TRT-LLM convention: 0 = disabled | `engine_configs.py:952-956` |
| `top_p` | float [0.0, 1.0] | `1.0` | Nucleus sampling threshold | `engine_configs.py:957-959` |
| `repetition_penalty` | float [0.1, 10.0] | `1.0` | Repetition penalty | `engine_configs.py:960-962` |
| `min_p` | float [0.0, 1.0] | null | Minimum probability filter | `engine_configs.py:963-965` |
| `min_tokens` | int >= 0 | `0` | Minimum output tokens before EOS allowed | `engine_configs.py:968-972` |
| `n` | int >= 1 | `1` | Number of output sequences per prompt | `engine_configs.py:973-977` |
| `ignore_eos` | bool | `false` | Continue generating past EOS | `engine_configs.py:978-981` |

## Validation rules

The Pydantic models enforce these rules at config-load time. The full
catalogue of invalid combinations, including engine-runtime invariants
mined from upstream libraries, is in
[invalid-combos.md](invalid-combos.md).

### Cross-engine

- The engine-specific section must match `engine:` (`models.py:441-464`).
- `passthrough_kwargs` keys must not collide with `ExperimentConfig`
  field names (`models.py:466-481`).

### Transformers

- `load_in_4bit` and `load_in_8bit` are mutually exclusive
  (`engine_configs.py:291-298`).
- `torch_compile_mode` and `torch_compile_backend` require
  `torch_compile=true` (`engine_configs.py:300-307`).
- `bnb_4bit_*` fields require `load_in_4bit=true`
  (`engine_configs.py:309-318`).
- `cache_implementation` requires `use_cache` to be true or unset
  (`engine_configs.py:320-327`).
- `tp_plan` and `device_map` are mutually exclusive
  (`engine_configs.py:329-337`).
- `attn_implementation` `flash_attention_2`/`flash_attention_3` requires
  `dtype` `float16` or `bfloat16` (`models.py:487-508`).

### vLLM

- `kv_cache_memory_bytes` and `gpu_memory_utilization` are mutually
  exclusive (`engine_configs.py:666-674`).
- `max_num_batched_tokens` must be `>= max_model_len` when both are
  set (`engine_configs.py:676-689`).
- `beam_search` and `sampling` sections are mutually exclusive
  (`engine_configs.py:842-850`).
- `dtype: float32` is rejected by the field's `Literal` type
  (`engine_configs.py:817-823`).

### TensorRT-LLM

- `dtype: float32` is rejected by the field's `Literal` type
  (`engine_configs.py:1059-1064`).
- `quant_algo: FP8` requires SM >= 8.9 (Ada Lovelace or Hopper); not
  supported on A100 (SM 8.0). Hardware-side check; runtime error.

### Engine x dtype matrix

From `ssot.DTYPE_SUPPORT` (`ssot.py:156-160`):

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
  dtype: bfloat16
  batch_size: 4
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true
  attn_implementation: flash_attention_2
```

### vLLM with prefix caching and FP8 KV cache

```yaml
task:
  model: meta-llama/Llama-2-7b-hf
  max_input_tokens: 1024
  max_output_tokens: 256

engine: vllm

vllm:
  dtype: bfloat16
  engine:
    gpu_memory_utilization: 0.9
    enable_prefix_caching: true
    kv_cache_dtype: fp8
    block_size: 16
  sampling:
    temperature: 0.0
    n: 1
```

### vLLM with beam search

`vllm.beam_search:` replaces `vllm.sampling:`; the two are mutually
exclusive.

```yaml
task:
  model: gpt2

engine: vllm

vllm:
  engine:
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
sampling_preset: deterministic   # sets temperature: 0.0 under transformers.sampling
```

`sampling_preset` is expanded at parse time into the active engine's
`sampling:` sub-section. Explicit YAML values take precedence over
preset values (`models.py:405-433`).

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
