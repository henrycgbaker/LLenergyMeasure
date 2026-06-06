# Pass B methodology - tensorrt-llm v1.0.0 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the tensorrt-llm config surface by TYPE, not by call path.
Starting from the public config classes, every reachable pydantic model and
dataclass was walked and, for each, every validity rule extracted. This pass is
the COMPLEMENT to Pass A (entry-point walk): its whole point is to catch rules an
entry-point walk misses - declarative Literal/enum field typing, validators on
base classes that only fire for specific subclasses, dispatch-only classes, and
standalone-model field validators not on the public construction path.

- Engine source: `/tmp/trt-llm-1.0.0/tensorrt_llm/` (`version.py` confirms
  `__version__ = "1.0.0"`).
- Pure source analysis, no GPU, no model download.

## Type tree enumerated (1.0.0)

Root config file `llmapi/llm_args.py` (2441 lines - notably SMALLER than the
1.2.1 file's 3416 lines) defines the bulk of the surface. Every class was located
via `grep -n "^class "` and read in full:

- Base: `StrictBaseModel` (extra="forbid").
- LlmArgs tree: `BaseLlmArgs` -> `TrtLlmArgs`, `TorchLlmArgs`.
- Nested sub-models (`StrictBaseModel`): `CudaGraphConfig`, `MoeConfig`,
  `AttentionDpConfig`, `CalibConfig`, `DecodingBaseConfig` (+ Medusa / Eagle /
  UserProvided / NGram / DraftTarget / MTP / Auto / Lookahead subclasses),
  `DynamicBatchConfig`, `SchedulerConfig`, `PeftCacheConfig`, `KvCacheConfig`,
  `ExtendedRuntimePerfKnobConfig`, `CacheTransceiverConfig`, `TorchCompileConfig`.
- `_ParallelConfig` (dataclass), `_ModelWrapper` (dataclass).
- Enums: `LoadFormat`, `BatchingType`, `CapacitySchedulerPolicy`,
  `ContextChunkingPolicy`, `_ModelFormatKind`.
- Out-of-file types walked: `PluginConfig` (`plugin/plugin.py`), `BuildConfig`
  (`builder.py`), `QuantConfig` + `QuantAlgo` + `KV_CACHE_QUANT_ALGO_LIST`
  (`models/modeling_utils.py`, `quantization/mode.py`), `LoraConfig`
  (`lora_manager.py`), `SamplingParams` + `GuidedDecodingParams`
  (`sampling_params.py`).

## Extraction per class

For each class the following were harvested: `Literal[...]` / `StrEnum` / `Enum`
typed fields -> membership rules; inline `Field(gt/ge/le/lt/...)` numeric
constraints; `@field_validator` / `@model_validator` bodies -> every
`raise`/`assert`/`logger.warning`; `__init__`/`__post_init__`/`model_post_init`
checks; `from_dict` dispatch tables -> `decode_dispatch`; `supports_backend()`
per-subclass overrides -> `backend_dispatch`. `outcome` derived from severity:
error/raise/assert -> `invalid`; warn -> `valid_with_warning`; catalogue
self-check -> `meta`.

## kwargs replayability discipline

`kwargs_positive` = the FIRING / invalid case (must trigger the rule);
`kwargs_negative` = the VALID case (must pass). Pairs were emitted ONLY where the
predicate is reachable at plain construction on a CPU host. For rules that are
GPU-gated (`validate_dtype` SM check, `PluginConfig.validate` SM-100,
`validate_kv_cache_dtype` and `validate_enable_build_cache` which sit behind
TrtLlmArgs' CUDA-querying `validate_dtype`), env-gated (greedy `best_of`),
filesystem-stateful (`get_model_format`, engine/ckpt loaders, moe load-balancer
file), deferred to a later lifecycle method (`SamplingParams._get_*_words`,
`QuantMode.from_quant_algo` via `quant_mode`), or expressed as a not-auto-invoked
method (see flagged entries), NO kwargs were fabricated; a `dormant_reason` is
recorded in `notes` and the gate skips it. The dispatch-only rule
(`DecodingBaseConfig.from_dict`) carries `replay_via: from_dict`.

CPU-replayable standalone models verified importable from `tensorrt_llm.llmapi`
(checked in `llmapi/__init__.py`): `MoeConfig`, `CalibConfig`, `SchedulerConfig`,
`CacheTransceiverConfig`, `CudaGraphConfig`, `TorchCompileConfig`,
`LookaheadDecodingConfig`, `SamplingParams`, `GuidedDecodingParams`, plus
`LoraConfig` (lora_manager) and `QuantConfig`.

## Result

- 66 total candidates.
- 65 folded (rule present at BOTH 1.0.0 and 1.2.1; every cited line re-verified
  against 1.0.0 source - line numbers re-derived, NOT copied from 1.2.1).
- 1 net-new declarative type-level constraint an entry-point/call-graph walk
  structurally misses (no explicit `raise` on any path):
  `torchLlmArgs_allreduce_strategy_literal` (8-member beta Optional[Literal]).
- 17 carry CPU-replayable kwargs pairs; 49 are dormant (GPU/env/filesystem/
  lifecycle/not-auto-invoked) and carry a `dormant_reason`.

## Class-hierarchy cases this pass caught that an entry-point walk misses

- Declarative `Literal`/`StrEnum` membership enforced by pydantic typing with no
  explicit raise anywhere: `MoeConfig.backend`, `CalibConfig.device`,
  `CacheTransceiverConfig.backend`, `SchedulerConfig.capacity_scheduler_policy` /
  `context_chunking_policy`, `BatchingType`, `TorchLlmArgs.allreduce_strategy`,
  the BaseLlmArgs Literals (`tokenizer_mode`, `load_format`,
  `guided_decoding_backend`).
- `LoraConfig.lora_ckpt_source` `assert` in `__post_init__` on a dataclass that
  the TRT-path entry-point walk does not route to.
- `QuantAlgo` StrEnum membership and the `kv_cache_quant_algo` allowlist, both
  living in `quantization/mode.py` and reached only via `QuantConfig.quant_mode`.
- Per-subclass `supports_backend()` overrides on the `DecodingBaseConfig` MRO
  (sibling-specific behaviour: Medusa/Lookahead vs NGram/MTP/Auto/DraftTarget).
- `DecodingBaseConfig.from_dict` string-dispatch registry (8 keys at 1.0.0).
- Standalone field validators on sub-models that an entry-point walk only sees if
  that exact field is populated on the public path: `CudaGraphConfig`,
  `TorchCompileConfig`, `LookaheadConfig.validate_positive_values`.

## 1.0.0-vs-1.2.1 API differences (re-derived, not assumed)

Rules PRESENT at 1.2.1 but ABSENT at 1.0.0 (correctly NOT emitted here):

1. PluginConfig is NOT pydantic at 1.0.0 - it is a metaclass-based class
   (`plugin/plugin.py:141`) with NO Literal-typed fields and NO `@field_validator`.
   ALL the 1.2.1 pass-B PluginConfig Literals (`dtype`, `gemm_plugin`,
   `gemm_swiglu_plugin`, `low_latency_gemm_plugin`,
   `low_latency_gemm_swiglu_plugin`, `gemm_allreduce_plugin`,
   `bert_attention_plugin` and the whole DefaultPluginDtype family) do not exist
   at 1.0.0. The only PluginConfig construction rule is the SM-100 killswitch in
   `validate()` (GPU/build-gated), which IS emitted (dormant).
2. KvCacheConfig has NO Python field_validators at 1.0.0: no
   `free_gpu_memory_fraction` range, no `max_gpu_total_bytes`, no
   `max_attention_window` list check, and `mamba_ssm_cache_dtype` does not exist;
   `dtype` is a plain `str` (no Literal). All NEW at 1.2.1 (C++-side at 1.0.0).
3. SamplingParams._validate has NO `top_p` / `top_k` / `temperature` range checks
   at 1.0.0 (NEW Python checks at 1.2.1). 1.0.0 only checks best_of>=n, the greedy
   best_of env gate, truncate_prompt_tokens>=1, and one-guide mutual exclusion.
4. CacheTransceiverConfig has NO `kv_transfer_timeout_ms` /
   `kv_transfer_sender_future_timeout_ms` Field(gt=0) at 1.0.0 (the two net-new
   inline Field(gt=0) of 1.2.1). 1.0.0 has only `backend` + `max_tokens_in_buffer`.
5. No `Nvfp4GemmConfig`, `MoeLoadBalancerConfig` (referenced via `_torch` but not a
   walked config class at 1.0.0 with a num_slots/ep_size rule), `RayPlacementConfig`,
   `BaseSparseAttentionConfig`/Rocket/DeepSeek/SkipSoftmax, `KvCacheConnectorConfig`
   at 1.0.0. No `orchestrator_type`, no ray_worker_extension_cls / ray_placement_config
   validators. No `stream_interval`/`batch_wait_*`/`attention_dp timeout/iters`-style
   1.2.1 additions beyond what is emitted (stream_interval and attention_dp ARE present).
6. No `validate_helix_tokens_per_block` (HELIX cp_type) at 1.0.0.
7. DecodingBaseConfig has NO `acceptance_window` / `acceptance_length_threshold` /
   `draft_len_schedule` validators at 1.0.0 (all NEW at 1.2.1). The from_dict
   registry has 8 keys (no `SaveState`); the SpeculativeConfig union has 8 members
   (no `SaveHiddenStates`).
8. MoeConfig.backend is a 6-member Literal at 1.0.0 (1.2.1 added `TRITON`).
   CacheTransceiverConfig.backend is 4-member (1.2.1 added `MOONCAKE`).
   allreduce_strategy is 8-member (1.2.1 added `NCCL_SYMMETRIC`). LoadFormat enum
   is {AUTO, DUMMY} (1.2.1 added `VISION_ONLY`). QuantAlgo is 21-member (1.2.1: 26).

Same-rule, DIFFERENT behaviour / location at 1.0.0:

- `validate_build_config_with_runtime_params` RAISES ValueError on
  max_batch_size/max_num_tokens > build_config at 1.0.0 (1.2.1 changed to a
  warn/clamp). It lives on `BaseLlmArgs` at 1.0.0 (1.2.1 split it onto TrtLlmArgs).
- `validate_kv_cache_dtype` uses a bare `assert` at 1.0.0 (1.2.1 raises ValueError).
- `LoraConfig` is in `lora_manager.py` with an `assert` __post_init__ enforcing
  `lora_ckpt_source` (1.2.1 moved it to `lora_helper.py` and made it a Literal).
- The Eagle/DraftTarget speculative messages differ ("Path to EAGLE3 weights must
  be specified." / "Path to draft model must be specified." at 1.0.0).
- `speculative_model_dir` is the field name at 1.0.0 (1.2.1: `speculative_model`).
- `validate_speculative_config` and the `_load_config_from_engine/_ckpt` loaders
  live on `BaseLlmArgs` at 1.0.0 (1.2.1 has a TrtLlmArgs copy of the spec validator).

## PoC / 1.2.1 entries flagged at 1.0.0

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage`: as at 1.2.1, the method
  (def 2258) is NOT `@model_validator`-decorated; the decorated after-validator
  stack ends at `validate_attention_dp_config` (2279). It does NOT fire
  automatically at construction. `pass_b_flag: possibly_invalid_not_auto_invoked`.
- `tensorrt_eagleDecodingConfig_validate_draft_model_required`: `EagleDecodingConfig.validate()`
  (424) is a PLAIN method, not a model_validator - it does NOT fire at
  EagleDecodingConfig construction. 1.0.0 also lacks the 1.2.1 construction-time
  `max_draft_len is required for Eagle` model_validator entirely.
  `pass_b_flag: possibly_invalid_not_auto_invoked`.
- `tensorrt_guidedDecodingParams_at_most_one_guide`: `GuidedDecodingParams` is a
  dataclass with NO `__post_init__`; `_validate` (36) fires only when reached via
  `SamplingParams.__post_init__` (320). Flagged; recorded as drive-through-SamplingParams.
  (Contrast: the 1.2.1 entry asserted `_validate invoked from __post_init__`, which
  is NOT how the 1.0.0 source reads.)

## Uncertain / caveats

- The three `pass_b_flag` entries are kept in the catalogue (they encode a real
  declared rule) but the runtime gate may classify them as non-firing-at-construction.
  Pass A (entry-point walk) is expected to confirm whether the public `LLM(...)`
  path invokes them downstream.
- Most BaseLlmArgs/TrtLlmArgs/TorchLlmArgs-level rules are marked dormant because
  full *LlmArgs construction at 1.0.0 unconditionally runs `validate_dtype` (CUDA
  query), `validate_and_init_tokenizer` (tokenizer fetch), and BuildConfig
  side-effects. A GPU+model fixture is required to replay them; the standalone
  sub-model and dataclass rules (17 of them) are the CPU-replayable subset.
- `BuildConfig` (builder.py:481) carries no construction-time field validators at
  1.0.0 (its checks are asserts inside `build()`); consistent with the 1.2.1
  blind-spot note. Left to the entry-point/call-graph pass.
