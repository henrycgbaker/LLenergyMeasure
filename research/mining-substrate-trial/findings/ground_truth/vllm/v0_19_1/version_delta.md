# Version delta: vllm 0.7.3 -> 0.19.1

Explicit add / remove / rename of the config surface and invariants between the bump-pair
ground truths:

- `research/mining-substrate-trial/findings/ground_truth/vllm/v0_7_3/`
- `research/mining-substrate-trial/findings/ground_truth/vllm/v0_19_1/` (this directory)

## 0. The one structural fact

`vllm/config.py` (flat, ~3490 LOC) -> `vllm/config/` (subpackage, ~30 per-concern modules).
Every subconfig class moved to its own module and became a `@config`-decorated **pydantic
dataclass**. The public `vllm.config.X` re-exports still resolve (`config/__init__.py`), so
import-path-based code keeps working, but every source citation that pointed at `config.py:NNNN`
is now `config/<module>.py:NNN`.

## 1. Config classes: added / removed / renamed

### Added (new in v0.19.1)

| Class | Module | Purpose |
| --- | --- | --- |
| `AttentionConfig` | attention.py | Attention backend + flash-attn version selection (lifted out of env/ModelConfig). |
| `KernelConfig` | kernel.py | MoE backend + flashinfer autotune. |
| `OffloadConfig` (+ `UVAOffloadConfig`, `PrefetchOffloadConfig`) | offload.py | Absorbs v0.7.3 `cpu_offload_gb`/`swap_space`; adds prefetch offloading. |
| `EPLBConfig` | parallel.py | Expert-parallel load balancing. |
| `ECTransferConfig` | ec_transfer.py | Embedding-cache transfer (mirrors KVTransferConfig). |
| `KVEventsConfig` | kv_events.py | KV-cache event publishing (zmq). |
| `ProfilerConfig` | profiler.py | torch-profiler controls (was env-only in v0.7.3). |
| `ReasoningConfig` | reasoning.py | Reasoning start/end token handling. |
| `SpeechToTextConfig` | speech_to_text.py | ASR runner config. |
| `WeightTransferConfig` | weight_transfer.py | nccl/ipc weight transfer. |
| `DynamicShapesConfig`, `PassConfig` (nested) | compilation.py | Compile-pass + dynamic-shapes sub-config. |

### Renamed

| v0.7.3 | v0.19.1 | Notes |
| --- | --- | --- |
| `DecodingConfig` | `StructuredOutputsConfig` (structured_outputs.py) | `guided_decoding_backend` field -> `backend` (StructuredOutputsBackend Literal) + new `disable_any_whitespace` / `disable_additional_properties` / reasoning fields. |
| `GuidedDecodingParams` | `StructuredOutputsParams` (sampling_params.py) | + `structural_tag`, `disable_additional_properties`; now REQUIRES exactly one constraint mode. |
| `CompilationConfig.level` (int enum 0-3) | `CompilationConfig.mode` (CompilationMode IntEnum) + new `cudagraph_mode` (CUDAGraphMode) | compile vs cudagraph split into two enums. |
| Speculative `ngram_prompt_lookup_max/min` | `prompt_lookup_max/min` | + declarative Field(ge=1). |
| Speculative `speculative_model` (string, `[ngram]` sentinel) | `model` + `method` (SpeculativeMethod Literal) | sentinel-string dispatch -> explicit method enum. |

### Removed

| v0.7.3 | Fate |
| --- | --- |
| `TokenizerPoolConfig` | gone (ray tokenizer pool removed). |
| `PromptAdapterConfig` | gone (prompt-adapter feature removed; `enable_prompt_adapter`/`max_prompt_adapters`/`max_prompt_adapter_token` EngineArgs fields removed). |
| `MultiModalConfig.limit_per_prompt`-only stub | replaced by a 15-field MultiModalConfig. |
| Speculative `spec_decoding_acceptance_method` + `typical_acceptance_sampler_*` | replaced by `rejection_sample_method` Literal + `synthetic_acceptance_rate`. |
| `LoRAConfig.lora_extra_vocab_size`, `long_lora_scaling_factors` | removed. |
| `SchedulerConfig.num_scheduler_steps`, `num_lookahead_slots`, `multi_step_stream_outputs`, `delay_factor`, `send_delta_data` | removed (multi-step scheduling gone in V1). |
| `CacheConfig.swap_space` | removed (offload is OffloadConfig now). |

## 2. EngineArgs: 103 -> 185 fields

Net +82. The dominant additions are data-parallel / expert-parallel (`data_parallel_*`,
`enable_expert_parallel`, `enable_eplb`, `eplb_config`, `all2all_backend`, `ubatch_size`,
`dbo_*`), context-parallel (`*_context_parallel_size`, `dcp_*`, `cp_*`), mamba cache
(`mamba_*`), offload (`offload_*`, `cpu_offload_params`), kernel/attention (`attention_config`,
`kernel_config`, `moe_backend`), structured outputs / reasoning, kv-events / ec-transfer /
weight-transfer, and observability metrics (`kv_cache_metrics`, `cudagraph_metrics`,
`enable_mfu_metrics`). Removed: the speculative_* flat fields (now nested
`speculative_config: dict`), prompt-adapter fields, tokenizer-pool fields, rope_scaling/rope_theta.

The biggest *pattern* change: EngineArgs defaults are now class-attribute references
(`model: str = ModelConfig.model`, `get_field(SubConfig, "x")`) rather than literals. The
authoritative literal default lives in the subconfig field.

## 3. SamplingParams

- Removed: `best_of`, `_real_n`, `truncate_prompt_tokens`, `output_text_buffer_length` semantics
  (best_of split into the beam/n path). The v0.7.3 invariants
  `best_of_lt_ref_n`, `delta_output_kind_requires_best_of_eq_n` are **GONE**.
- Added fields: `flat_logprobs`, `skip_clone`, `extra_args`, `thinking_token_budget`,
  `repetition_detection` (nested RepetitionDetectionParams), `skip_reading_prefix_cache`,
  `_bad_words_token_ids`, `_eos_token_id`. `guided_decoding` -> `structured_outputs`.
- Behaviour changes:
  - `top_k` default `-1` -> `0`; disable-sentinel is now `0` (so `top_k=0` is VALID, was an error).
  - `repetition_penalty` band `(0, 2]` -> just `> 0` (upper bound 2 dropped).
  - `logprobs` / `prompt_logprobs` now also accept `-1`.
  - `n` gains an env-gated upper bound (`VLLM_MAX_N_SEQUENCES`, default 16384).
  - `stop_token_ids` gains a per-element int check.
  - Several raises now throw the new typed `VLLMValidationError` instead of bare `ValueError`.

## 4. Invariants: where they moved

| v0.7.3 invariant | v0.19.1 fate |
| --- | --- |
| MLA disables chunked_prefill + prefix_caching (VllmConfig, all platforms) | **MOVED + NARROWED** to `CpuPlatform.check_and_update_config` (cpu.py:362); non-GPU only. On CUDA, no longer fires. |
| LoRA disables torch.compile (VllmConfig) | **REMOVED**; v0.19.1 funnels compile-disable through `enforce_eager -> mode=NONE` (vllm.py:847). |
| cpu_offload disables torch.compile (VllmConfig) | **REMOVED** (cpu_offload moved to OffloadConfig; no compile interaction in config). |
| CacheConfig.gpu_memory_utilization > 1 raise | **CHANGED to declarative** `Field(gt=0, le=1)` (now also rejects <= 0). |
| LoRAConfig max_loras < 1 raise | **CHANGED to declarative** `Field(ge=1)`. |
| SpeculativeConfig num_speculative_tokens <= 0 raise | kept AND also declarative `Field(gt=0)`. |
| tokenizer_mode / cache_dtype / load_format enum raises | **CHANGED to declarative** Literal types. |
| rope_scaling / rope_theta DeprecationWarning | **REMOVED** (fields removed). |
| SchedulerConfig max_long_partial_prefills full-band check | narrowed to upper-bound only (lower via Field(ge=1)). |

### New invariant classes in v0.19.1

- **Declarative pydantic `Field(ge/gt/le)` constraints** - a NEW invariant source the mining
  substrate must parse (they raise `pydantic.ValidationError`, not a grep-able `raise`).
- **Per-platform `check_and_update_config`** silent overrides (CPU fp8-KV fallback, CUDA
  mm-prefix-lm, XPU graph disable) - a NEW per-platform mining surface.
- **EPLB / expert-parallel / elastic-EP / data-parallel** cross-field guards (parallel.py).
- **Suffix-decoding / draft-vocab-match** speculative guards (speculative.py).
- **StructuredOutputsConfig backend-compat** guards + the new "must specify exactly one constraint"
  rule on StructuredOutputsParams.
- **RepetitionDetectionParams** validation (new struct).

## 5. env surface: 84 -> 238

Net +154. Largest families: ROCm AITER (`VLLM_ROCM_USE_AITER_*`, ~20), FlashInfer MoE
(`VLLM_USE_FLASHINFER_MOE_*`, ~8), DeepGEMM (`VLLM_USE_DEEP_GEMM*`), data-parallel (`VLLM_DP_*`),
compile/AOT (`VLLM_USE_AOT_COMPILE`, `VLLM_FORCE_AOT_LOAD`, `VLLM_USE_STANDALONE_COMPILE`), MoRI-IO
/ NIXL / Mooncake connectors, media-fetch controls. Notable single additions: `VLLM_MAX_N_SEQUENCES`
(caps SamplingParams.n), `VLLM_BATCH_INVARIANT` (deterministic kernels), `VLLM_USE_V2_MODEL_RUNNER`.

Footgun note: the v0.7.3 stub-vs-lambda discrepancies (`VLLM_CUDA_MEM_ALIGN_KV_CACHE`,
`VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH` reading `VLLM_CONTIGUOUS_PA`) are GONE - those keys were
removed/reworked in v0.19.1. The env module no longer ships the parallel TYPE_CHECKING stub block
in the same divergent form (the lambdas are authoritative as before).
