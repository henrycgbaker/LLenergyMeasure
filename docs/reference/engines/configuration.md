---
title: Engine configuration
description: Per-engine YAML configuration fields - structure, types, defaults, validation rules.
---

# Engine configuration

This page documents the per-engine YAML configuration surface. Each
experiment selects exactly one engine via the top-level `engine:` field
and configures it through a same-named block (`transformers:`, `vllm:`,
`tensorrt:`).

Each engine block is a generated model at
`src/llenergymeasure/engines/<engine>/config.py`, regenerated from the
committed schema snapshot (its header reads `DO NOT EDIT`). The block has
exactly two sub-sections, `engine_params:` and `sampling_params:`, and both
sub-models set `extra="allow"`, so a parameter the current snapshot does not
model is still forwarded to the underlying engine. The tables below document
the fields the generated models name explicitly at the pinned versions
(transformers 5.7.0, vllm 0.19.1, tensorrt 1.2.1); for the exhaustive
introspected inventory see the per-engine schema pages, and for which
parameters are explicitly modelled vs forwarded see the curation pages (both
linked under [See also](#see-also)).

For study-level controls (sweeps, runners, images, cycles, output) see
[study-config.md](../study-config.md).

## Top-level shape

A single experiment YAML selects an engine and configures it through a
same-named block. The engine block nests `engine_params:` and
`sampling_params:` under it:

```yaml
task:
  model: Qwen/Qwen2.5-0.5B
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
    n_prompts: 5
  baseline:
    enabled: true
    duration_seconds: 30.0
  energy_sampler: auto

transformers:
  engine_params:
    dtype: bfloat16
    attn_implementation: sdpa
  sampling_params:
    temperature: 0.7

# transformers-only orchestration knobs (see Harness section)
harness:
  transformers:
    batch_size: 4

# Optional
sampling_preset: deterministic   # deterministic | standard | creative | factual
passthrough_kwargs:
  trust_remote_code: true
```

`ExperimentConfig` sets `extra="forbid"`, so an unknown **top-level** key is a
hard error, and so is a key placed directly on an engine block that is not
`engine_params` or `sampling_params` (the validator emits a migration hint;
`validate_engine_section_extras` in `models.py`). Parameters therefore live one level deeper than they did
in the pre-0.10 flat layout: `transformers.engine_params.dtype`, not
`transformers.dtype`.

The engine-specific section must match the `engine:` field; mixing
`engine: vllm` with a `transformers:` section is a configuration error
(`validate_engine_section_match` in `models.py`). When `engine:` is set
without a matching section, the engine's own defaults are used.

There is no top-level `dtype:` field. `dtype` lives inside each engine's
`engine_params` (`transformers.engine_params.dtype`,
`vllm.engine_params.dtype`, `tensorrt.engine_params.dtype`) because each engine
models it differently (see the per-engine tables).

`runners:`, `images:`, `sweep:`, and `experiments:` are study-level fields and
not valid in a single-experiment YAML; they belong on the study document
(`StudyConfig` in `models.py`). See
[study-config.md](../study-config.md).

## Common fields (all engines)

These fields are declared on `ExperimentConfig` and its sub-models and apply
identically across engines. Every one is `extra="forbid"`.

### `task:`

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `model` | str (required, min length 1) | - | HuggingFace model ID or local path | `TaskConfig` |
| `dataset.source` | str (min length 1) | `aienergyscore` | Built-in dataset alias or `.jsonl` path | `DatasetConfig` |
| `dataset.n_prompts` | int >= 1 | `100` | Number of prompts to load or generate | `DatasetConfig` |
| `dataset.order` | `interleaved` \| `grouped` \| `shuffled` | `interleaved` | Prompt ordering strategy | `DatasetConfig` |
| `max_input_tokens` | int >= 1 \| null | `256` | Input truncation cap; null disables | `TaskConfig` |
| `max_output_tokens` | int >= 1 \| null | `256` | Output token budget; null generates to EOS or context limit | `TaskConfig` |
| `random_seed` | int | `42` | Per-experiment seed for inference RNG and dataset ordering | `TaskConfig` |

### `measurement:`

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `warmup.enabled` | bool | `true` | Enable warmup phase before measurement | `WarmupConfig` |
| `warmup.n_prompts` | int >= 1 | `5` | Number of warmup prompts in fixed mode | `WarmupConfig` |
| `warmup.thermal_floor_seconds` | float >= 30.0 | `60.0` | Minimum post-warmup wait for thermal stabilisation | `WarmupConfig` |
| `warmup.convergence_detection` | bool | `false` | Enable adaptive CV-based convergence (replaces the fixed `n_prompts` count) | `WarmupConfig` |
| `warmup.cv_threshold` | float [0.01, 0.5] | `0.05` | CV target for convergence | `WarmupConfig` |
| `warmup.max_prompts` | int >= 5 | `20` | Safety cap for CV mode | `WarmupConfig` |
| `warmup.window_size` | int >= 3 | `3` | Sliding window size for CV calculation | `WarmupConfig` |
| `warmup.min_prompts` | int >= 1 | `5` | Minimum prompts before checking convergence | `WarmupConfig` |
| `baseline.enabled` | bool | `true` | Measure idle GPU power before experiments | `BaselineConfig` |
| `baseline.duration_seconds` | float [5.0, 120.0] | `30.0` | Baseline measurement window | `BaselineConfig` |
| `baseline.strategy` | `cached` \| `validated` \| `fresh` | `validated` | Caching strategy for the baseline measurement | `BaselineConfig` |
| `baseline.cache_ttl_seconds` | float >= 60.0 | `7200.0` | Cached baseline lifetime (cached/validated only) | `BaselineConfig` |
| `baseline.validation_interval` | int >= 1 | `5` | Re-validate every N experiments (validated only) | `BaselineConfig` |
| `baseline.drift_threshold` | float [0.01, 0.50] | `0.10` | Drift fraction that triggers re-measurement (validated only) | `BaselineConfig` |
| `energy_sampler` | `auto` \| `nvml` \| `zeus` \| `codecarbon` \| null | `auto` | Energy sampler; null disables energy measurement | `MeasurementConfig` |
| `latency_profiling` | bool | `false` | Opt-in per-token latency profiling (TTFT/ITL). Overhead may perturb energy and latency, so profiled runs are tagged in `measurement_warnings`. | `MeasurementConfig` |
| `measurement_methodology` | `total` \| `windowed` \| `steady_state` | `total` | How the measurement window is derived from the run | `MeasurementConfig` |
| `measurement_window` | [float, float] \| null | null | Required when `measurement_methodology=windowed` (`start >= 0`, `end > start`) | `MeasurementConfig` |
| `warmup_discard_fraction` | float [0.0, 1.0) | `0.1` | Fraction of the run discarded as warmup when deriving steady state | `MeasurementConfig` |
| `warmup_discard_seconds` | float >= 0.0 \| null | null | Absolute warmup discard window (overrides the fraction) | `MeasurementConfig` |
| `steady_state_auto_detect` | bool | `false` | Auto-detect the steady-state window | `MeasurementConfig` |

### Top-level optional fields

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `sampling_preset` | `deterministic` \| `standard` \| `creative` \| `factual` \| null | null | Merges preset values into the active engine's `sampling_params:` section at parse time. Explicit YAML values take precedence. | `ExperimentConfig` (`expand_sampling_preset`, `SAMPLING_PRESETS`) |
| `passthrough_kwargs` | dict \| null | null | Extra kwargs forwarded to the engine at execution time; keys must not collide with `ExperimentConfig` top-level field names | `ExperimentConfig` (`validate_passthrough_kwargs_no_collision`) |

## Harness (`harness:`) - transformers-only orchestration knobs

Some knobs are features llem implements itself in its own runner loop, because
the engine exposes no native API for them. They are not engine config, so they
live under `harness:` (hand-written at `config/harness.py`), not under the
generated engine block. Only transformers has a harness residual today; vllm
and tensorrt drive batching and precision through native engine APIs, so
`harness` carries no block for them.

```yaml
harness:
  transformers:
    batch_size: 4
    torch_compile: true
    torch_compile_mode: reduce-overhead
```

| Field | Type | Default | Description | Source |
|-------|------|---------|-------------|--------|
| `batch_size` | int >= 1 | null (-> 1) | Prompt-batching size for llem's runner loop (`model.generate()` has no batch_size kwarg) | `TransformersHarness` |
| `torch_compile` | bool | null (-> false) | Enable `torch.compile` on the loaded model | `TransformersHarness` |
| `torch_compile_mode` | str | null (-> `default`) | `default` \| `reduce-overhead` \| `max-autotune`. Requires `torch_compile=true`. | `TransformersHarness` |
| `torch_compile_backend` | str | null (-> `inductor`) | `torch.compile` backend. Requires `torch_compile=true`. | `TransformersHarness` |
| `allow_tf32` | bool | null | Allow TF32 matmul on Ampere+ via `torch.backends` | `TransformersHarness` |
| `autocast_enabled` | bool | null (-> false) | Wrap generation in `torch.autocast` | `TransformersHarness` |
| `autocast_dtype` | `float16` \| `bfloat16` | null (-> bfloat16 on Ampere) | AMP dtype (used when `autocast_enabled=true`) | `TransformersHarness` |

Naming `torch_compile_mode` or `torch_compile_backend` without
`torch_compile=true` is rejected (`validate_torch_compile_options` in
`harness.py`).

## Transformers engine (`transformers:`)

Loads a model via `AutoModelForCausalLM.from_pretrained()` and generates with
`model.generate()`. Every generated field is `X | None = None`: null means
"use the engine's own default". Unknown fields under `engine_params:` or
`sampling_params:` are forwarded to the underlying HuggingFace APIs
(`extra="allow"`).

### `transformers.engine_params:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | any \| null | null | Model compute dtype (e.g. `float32`, `float16`, `bfloat16`); untyped, forwarded as given |
| `attn_implementation` | any \| null | null | Attention kernel (e.g. `sdpa`, `flash_attention_2`, `flash_attention_3`, `eager`) |
| `load_in_4bit` | any \| null | null | BitsAndBytes 4-bit quantisation |
| `load_in_8bit` | any \| null | null | BitsAndBytes 8-bit quantisation |
| `bnb_4bit_compute_dtype` | any \| null | null | Compute dtype for 4-bit |
| `bnb_4bit_quant_type` | any \| null | null | 4-bit quantisation type (e.g. `nf4`, `fp4`) |
| `bnb_4bit_use_double_quant` | any \| null | null | Double quantisation |
| `use_cache` | bool \| null | null | Enable KV cache during generation |
| `cache_implementation` | str \| null | null | KV cache strategy (e.g. `static`, `offloaded_static`, `sliding_window`) |
| `num_beams` | int \| null | null | Beam search width (1 = greedy/sampling) |
| `early_stopping` | bool \| null | null | Stop beam search when all beams hit EOS |
| `length_penalty` | float \| null | null | Beam length penalty |
| `no_repeat_ngram_size` | int \| null | null | Prevent n-gram repetition |
| `prompt_lookup_num_tokens` | int \| null | null | Prompt-lookup speculative decoding tokens |
| `device_map` | any \| null | null | Device placement strategy (e.g. `auto`) |
| `max_memory` | any \| null | null | Per-device memory limits, e.g. `{0: "10GiB", cpu: "50GiB"}` |
| `low_cpu_mem_usage` | any \| null | null | Load weights incrementally to reduce peak CPU RAM |
| `tp_plan` | any \| null | null | Native HF tensor parallelism plan |
| `tp_size` | any \| null | null | Tensor parallel ranks (used with `tp_plan`) |

Source: `src/llenergymeasure/engines/transformers/config.py`.

### `transformers.sampling_params:`

Maps to `model.generate()` kwargs; field names mirror HuggingFace conventions.
All fields default to null (use the HF default). Untyped or freeform kwargs are
forwarded via `extra="allow"`.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float \| null | null | Sampling temperature (0 = greedy) |
| `do_sample` | bool \| null | null | Enable sampling |
| `top_k` | int \| null | null | HF convention: 0 = disabled |
| `top_p` | float \| null | null | Nucleus sampling threshold |
| `repetition_penalty` | float \| null | null | Repetition penalty (1.0 = no penalty) |
| `min_p` | float \| null | null | Minimum probability filter |
| `min_new_tokens` | int \| null | null | Minimum output tokens |

Source: `src/llenergymeasure/engines/transformers/config.py`.

## vLLM engine (`vllm:`)

vLLM exposes a two-API surface (`vllm.LLM()` constructor and `SamplingParams`);
the block mirrors that split across `engine_params:` and `sampling_params:`.
Unlike the flat pre-0.10 layout there is no separate top-level `vllm.dtype` or
`vllm.beam_search` block: `dtype` is a field under `engine_params`, and
`beam_search` / `attention` are freeform (`Any`-typed) dict fields under
`engine_params`.

### `vllm.engine_params:`

Constructor arguments for `vllm.LLM()`. Unlike transformers, several fields
carry defaults and bounds from the generated model.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dtype` | `auto` \| `half` \| `float16` \| `bfloat16` \| `float` \| `float32` \| null | `auto` | Model dtype; `auto` infers from weights |
| `gpu_memory_utilization` | float (0.0, 1.0] | `0.9` | GPU memory fraction reserved for KV cache |
| `cpu_offload_gb` | float >= 0.0 | `0` | CPU RAM in GiB to offload model weights to |
| `block_size` | int \| null | null | KV cache block size in tokens |
| `kv_cache_dtype` | `auto` \| `float16` \| `bfloat16` \| `fp8` \| `fp8_e4m3` \| `fp8_e5m2` \| `fp8_inc` \| `fp8_ds_mla` \| null | `auto` | KV cache storage dtype |
| `enforce_eager` | bool \| null | `false` | Disable CUDA graphs, always use eager mode |
| `enable_chunked_prefill` | bool \| null | null | Chunk large prefills across scheduler iterations |
| `max_num_seqs` | int >= 1 \| null | null | Max concurrent sequences per scheduler iteration |
| `max_num_batched_tokens` | int >= 1 \| null | null | Max tokens per scheduler iteration |
| `max_model_len` | int >= 1 \| null | null | Max sequence length (input + output) |
| `tensor_parallel_size` | int \| null | `1` | Number of GPUs to shard the model across |
| `pipeline_parallel_size` | int \| null | `1` | Pipeline parallel stages |
| `distributed_executor_backend` | any \| null | null | Multi-GPU executor backend (e.g. `mp`, `ray`) |
| `enable_prefix_caching` | bool \| null | null | Automatic prefix caching for shared prompt prefixes |
| `quantization` | any \| null | null | Quantisation method (requires a pre-quantised checkpoint) |
| `speculative_config` | sub-model \| null | null | Speculative decoding config (see the schema page) |
| `offload_group_size` | int >= 0 \| null | `0` | Groups of layers for CPU offloading |
| `offload_num_in_group` | int >= 1 \| null | `1` | Layers offloaded per group |
| `offload_prefetch_step` | int >= 0 \| null | `1` | Prefetch steps ahead for CPU offload |
| `offload_params` | any \| null | `[]` | Specific parameter names to offload |
| `disable_custom_all_reduce` | bool \| null | `false` | Disable custom all-reduce for multi-GPU |
| `kv_cache_memory_bytes` | int \| null | null | Absolute KV cache size |
| `compilation_config` | sub-model \| null | null | vLLM `CompilationConfig` passthrough (see the schema page) |
| `attention` | any (dict) \| null | null | Attention backend selection; freeform dict (e.g. `{backend: flash_attn}`) |
| `beam_search` | any (dict) \| null | null | Beam-search parameters; freeform dict (e.g. `{beam_width: 4, early_stopping: true}`) |

Source: `src/llenergymeasure/engines/vllm/config.py`. `speculative_config` and
`compilation_config` are typed sub-models with many fields; see
[schema-vllm.md](schema-vllm.md) for the full inventory.

### `vllm.sampling_params:`

Maps to `vllm.SamplingParams()`. `max_tokens` is intentionally absent; it is
bridged from `task.max_output_tokens` at execution time. `top_k` follows
vLLM's `0`-for-default convention.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float \| null | `1.0` | Sampling temperature (0 = greedy) |
| `top_k` | int \| null | `0` | Top-k cutoff |
| `top_p` | float \| null | `1.0` | Nucleus sampling threshold |
| `repetition_penalty` | float \| null | `1.0` | Repetition penalty |
| `min_p` | float \| null | `0.0` | Minimum probability filter |
| `min_tokens` | int \| null | `0` | Minimum output tokens before EOS allowed |
| `presence_penalty` | float \| null | `0.0` | Penalises tokens that appear at all |
| `frequency_penalty` | float \| null | `0.0` | Penalises tokens proportional to frequency |
| `ignore_eos` | bool \| null | `false` | Continue generating past EOS |
| `n` | int \| null | `1` | Number of output sequences per prompt |

Source: `src/llenergymeasure/engines/vllm/config.py`.

## TensorRT-LLM engine (`tensorrt:`)

TensorRT-LLM exposes two runtimes behind one engine, selected by `backend`:
`pytorch` (the default) runs the model through TensorRT-LLM's PyTorch runtime
with no ahead-of-time build, and `trt` compiles the model into an optimised
TensorRT engine on first use and reuses the cached engine afterwards.
`backend` is resolved by constructor class (`pytorch` -> `tensorrt_llm.LLM`,
`trt` -> `tensorrt_llm._tensorrt_engine.LLM`), never forwarded as a kwarg. A
handful of fields (`quant_config`, `fast_build`) exist only on the `trt`
backend; declaring them under `pytorch` is a config-load error. On the `trt`
backend, compile-time fields are baked into the engine and changing one keys to
a fresh build. The nested TRT-LLM sub-configs (`quant_config`,
`kv_cache_config`, `scheduler_config`) are freeform (`Any`-typed) dict fields
under `engine_params:` on the current pin, so they are written as whole dicts
(see the worked example).

### `tensorrt.engine_params:`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `max_batch_size` | int \| null | null | Maximum batch size the engine accepts (compile-time) |
| `tensor_parallel_size` | int \| null | `1` | Number of GPUs to shard across (compile-time) |
| `pipeline_parallel_size` | int \| null | `1` | Pipeline parallel stages (compile-time) |
| `max_input_len` | int \| null | null | Maximum input sequence length (compile-time) |
| `max_seq_len` | int \| null | null | Maximum total sequence length (compile-time) |
| `max_num_tokens` | int \| null | null | Maximum tokens the engine handles per iteration (compile-time) |
| `dtype` | str \| null | `auto` | Model compute dtype; untyped, forwarded as given |
| `backend` | `pytorch` \| `trt` \| null | `pytorch` | Runtime selector, resolved by constructor class (see intro); distinct from the top-level `engine:` selector. Any other value is a config-load error |
| `fast_build` | bool \| null | `false` | Reduced-optimisation build for faster compilation. **`trt` backend only** - rejected under `pytorch` |
| `quant_config` | any (dict) \| null | null | Quantisation config; freeform dict (e.g. `{quant_algo: W4A16_AWQ}`). **`trt` backend only** - rejected under `pytorch` |
| `kv_cache_config` | any (dict) \| null | null | KV cache config; freeform dict (e.g. `{enable_block_reuse: true, free_gpu_memory_fraction: 0.9}`) |
| `scheduler_config` | any (dict) \| null | null | Scheduler config; freeform dict (e.g. `{capacity_scheduling_policy: MAX_UTILIZATION}`) |

Source: `src/llenergymeasure/engines/tensorrt/config.py`.

### `tensorrt.sampling_params:`

Maps to `tensorrt_llm.SamplingParams`. `top_k` uses TRT-LLM's convention
(matches HuggingFace, not vLLM).

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `temperature` | float \| null | null | Sampling temperature (0 = greedy) |
| `top_k` | int \| null | null | Top-k cutoff |
| `top_p` | float \| null | null | Nucleus sampling threshold |
| `repetition_penalty` | float \| null | null | Repetition penalty |
| `min_p` | float \| null | null | Minimum probability filter |
| `min_tokens` | int \| null | null | Minimum output tokens before EOS allowed |
| `n` | int \| null | `1` | Number of output sequences per prompt |
| `ignore_eos` | bool \| null | `false` | Continue generating past EOS |

Source: `src/llenergymeasure/engines/tensorrt/config.py`.

## Validation rules

The Pydantic models enforce structural rules at config-load time, and the
engine's shipped rule corpus (`rules.yaml`) enforces the mined constraints. The
full catalogue of invalid combinations - derived from the live corpus plus the
cross-engine validators - is in [invalid-combos.md](invalid-combos.md).

### Cross-engine (structural, in `models.py`)

- The engine-specific section must match `engine:`
  (`validate_engine_section_match`).
- A key placed directly on an engine block (not under `engine_params` or
  `sampling_params`) is rejected with a migration hint
  (`validate_engine_section_extras`).
- `passthrough_kwargs` keys must not collide with `ExperimentConfig` top-level
  field names (`validate_passthrough_kwargs_no_collision`).

### Transformers

- With `attn_implementation` `flash_attention_2` or `flash_attention_3`,
  `dtype` must not be `float32` (`validate_transformers_flash_attn_dtype` in
  `models.py`; a missing dtype is treated as `bfloat16`).
- All further transformers constraints (beam-search divisibility,
  quantisation, caching) come from the shipped rule corpus; see
  [invalid-combos.md](invalid-combos.md).

### vLLM and TensorRT-LLM

- The vLLM/TensorRT constraints (mutually exclusive sections, cross-field
  bounds, dormant normalisations) come from each engine's shipped rule corpus.
  Note that the old hand-written `dtype: float32` rejection was dropped with
  the generated configs: vLLM's `dtype` now includes `float32` in its Literal,
  and the informational per-engine `dtypes` (on the `ssot.ENGINES` descriptor)
  drive pre-flight checks, not Pydantic parsing.

### Engine x dtype support (pre-flight)

From the per-engine `dtypes` on the `ssot.ENGINES` descriptor - an informational
pre-flight map, not a parse-time constraint:

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

All other fields fall back to defaults: `aienergyscore` dataset, 100 prompts,
256-token input/output caps, `auto` energy sampler, and the engine's own
dtype/attention defaults.

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
    torch_compile: false
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

`beam_search` is a freeform dict under `engine_params`; it is written as a
whole dict.

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
      early_stopping: true
  sampling_params:
    temperature: 0.0
```

### TensorRT-LLM with AWQ quantisation

The TRT-LLM sub-configs are freeform dicts on the current pin, written whole.

```yaml
task:
  model: meta-llama/Llama-2-7b-hf

engine: tensorrt

tensorrt:
  engine_params:
    backend: trt          # quant_config requires the compiled trt backend
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
`sampling_params:` sub-section via `setdefault`, so explicit YAML values take
precedence over preset values (`expand_sampling_preset` in `models.py`).

## See also

- [study-config.md](../study-config.md) - sweep grammar, runners, images,
  cycles, output configuration.
- [schema-transformers.md](schema-transformers.md),
  [schema-vllm.md](schema-vllm.md),
  [schema-tensorrt.md](schema-tensorrt.md) - auto-generated full
  parameter inventories straight from each engine's introspection.
- [curation-transformers.md](curation-transformers.md),
  [curation-vllm.md](curation-vllm.md),
  [curation-tensorrt.md](curation-tensorrt.md) - which engine
  parameters are explicitly modelled vs forwarded via `extra="allow"`.
- [invalid-combos.md](invalid-combos.md) - catalogue of rejected
  parameter combinations. The full verified rule set per engine lives at
  `src/llenergymeasure/engines/<engine>/rules.yaml`.
