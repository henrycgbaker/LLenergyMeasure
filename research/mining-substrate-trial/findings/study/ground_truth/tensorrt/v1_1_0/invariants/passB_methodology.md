# Pass B methodology - tensorrt-llm v1.1.0 class-hierarchy / type-tree walk

## Strategy

Strategy B (Pass B) enumerates the tensorrt-llm config surface by TYPE, not by
call path. It is the complement to Pass A (entry-point walk): its job is to catch
validity rules that an entry-point walk structurally misses because they live on a
class never routed to by public construction, or are enforced declaratively
(pydantic typing) with no explicit `raise` on any path.

- Engine source: `/tmp/trt-llm-1.1.0/tensorrt_llm/` (`version.py` confirms
  `__version__ = "1.1.0"`).
- Pure source analysis. No GPU, no model download.

## Type tree enumerated (as it exists at 1.1.0)

Root config file `llmapi/llm_args.py` (2677 lines). Every class located via
`grep -n "^class "` and read in full:

- Base: `StrictBaseModel` (extra="forbid").
- LlmArgs tree: `BaseLlmArgs` -> `TrtLlmArgs`, `TorchLlmArgs`.
- Nested sub-models (StrictBaseModel): `CudaGraphConfig`, `MoeConfig`,
  `AttentionDpConfig`, `CalibConfig`, `DecodingBaseConfig`
  (+ Medusa / Eagle / UserProvided / NGram / DraftTarget / MTP / Auto /
  Lookahead subclasses), `KvCacheConnectorConfig`, `DynamicBatchConfig`,
  `SchedulerConfig`, `PeftCacheConfig`, `KvCacheConfig`,
  `ExtendedRuntimePerfKnobConfig`, `CacheTransceiverConfig`, `TorchCompileConfig`.
- Dataclass: `_ParallelConfig`, `_ModelWrapper`.
- Enums: `LoadFormat` (AUTO/DUMMY/VISION_ONLY), `SamplerType`, `BatchingType`,
  `CapacitySchedulerPolicy`, `ContextChunkingPolicy`, `_ModelFormatKind`.
- PybindMirror machinery: `PybindMirror`, `PybindMirrorMeta`,
  `PybindMirrorEnumMeta` (the scheduler/batching enums and several configs mirror
  C++ pybind fields).
- Out-of-file types walked: `PluginConfig` (`plugin/plugin.py`), `QuantConfig` +
  `QuantAlgo` + `KV_CACHE_QUANT_ALGO_LIST` (`quantization/mode.py`), `LoraConfig`
  (`lora_helper.py`), `SamplingParams` + `GuidedDecodingParams`
  (`sampling_params.py`).

## Extraction per class

For each class the following were harvested by reading the 1.1.0 source line:

1. `Literal[...]` / `StrEnum` / `Enum`-typed fields -> membership rules
   (`literal_in` / `strenum_in`).
2. Inline `Field(gt/ge/le/lt/multiple_of)` numeric constraints.
3. `@field_validator` / `@model_validator` bodies -> every `raise` / `assert` /
   `logger.warning`.
4. `__init__` / `__post_init__` / property-setter checks.
5. `from_dict` dispatch tables -> `decode_dispatch`.
6. `supports_backend()` per-subclass overrides -> `backend_dispatch`.

`outcome` derived from severity: error/raise/assert -> `invalid`; warn ->
`valid_with_warning`; catalogue self-check -> `meta`.

## kwargs replayability discipline

`kwargs_positive` (FIRING / invalid case) and `kwargs_negative` (VALID case) were
emitted ONLY where the predicate is reachable at plain CPU construction of a pure
pydantic/dataclass model (no CUDA, no model dir, no filesystem state). For rules
that are GPU-gated, env-gated, filesystem-stateful, fire at a later lifecycle
method, or fire on attribute assignment rather than construction, NO kwargs were
fabricated and a `dormant_reason` / `replay_via` note was recorded instead. A
wrong pair would make the downstream runtime gate reject an otherwise-valid GT
entry, so dormancy is preferred over fabrication.

Notable dormant classes:
- `BaseLlmArgs` and subclasses construct through `validate_dtype` (calls
  `torch.cuda.get_device_properties(0)`) and `validate_and_init_tokenizer`
  (tokenizer load), so even CPU-pure field rules on the LlmArgs tree are not
  safely CPU-replayable; recorded dormant.
- `PluginConfig` (1.1.0) is a plain `@dataclass(slots=True)` whose validity asserts
  fire in the property setter, on attribute ASSIGNMENT, not construction; recorded
  with `replay_via=setattr`.
- `QuantConfig.kv_cache_quant_algo` allowlist fires inside
  `QuantMode.from_quant_algo`, not at plain construction.
- `GuidedDecodingParams` has no `__post_init__` at 1.1.0; its `_validate` only runs
  via `SamplingParams._validate`, so the replay routes through
  `SamplingParams(guided_decoding=...)`.

Freely-CPU-replayable standalone models (carry kwargs pairs): `CudaGraphConfig`,
`MoeConfig`, `CalibConfig`, `LookaheadDecodingConfig`, `KvCacheConfig`,
`SchedulerConfig`, `CacheTransceiverConfig`, `TorchCompileConfig`,
`TorchLlmArgs.allreduce_strategy` field, `LoraConfig`, `SamplingParams`.

## Result

- 71 total invariants.
- 68 fold (the same rule is present in the mined 1.2.1 pass-B catalogue, here
  re-derived against 1.1.0 source with 1.1.0's own line numbers and messages).
- 3 net_new for this pass (declarative / setter-time rules an entry-point walk
  misses):
  - `torchLlmArgs_allreduce_strategy_literal` - 9-member pydantic Literal,
    status="beta", never explicitly raised.
  - `pluginConfig_default_dtype_allowlist_assert` - the DEFAULT_PLUGIN_DTYPE_OPTIONS
    allowlist applied to the whole str-plugin family via the setter assert.
  - `pluginConfig_restricted_dtype_allowlist_assert` - the per-field
    PLUGIN_DTYPE_OPTIONS_MAP restricted allowlists (gemm_plugin, gemm_swiglu_plugin,
    low_latency_gemm*, gemm_allreduce_plugin).
- 24 CPU-replayable (kwargs pairs), 47 dormant (GPU / filesystem / env / later
  lifecycle / setattr-time / LlmArgs-construction-gated).
- outcome split: 61 invalid, 9 valid_with_warning, 1 meta.

## Class-hierarchy cases caught that an entry-point walk would miss

1. `allreduce_strategy` Literal on `TorchLlmArgs` - pure typing, no raise.
2. PluginConfig dtype allowlists - 1.1.0 enforces these in a metaclass-generated
   property setter (`_make_plugin_property`), not in any function the public
   construction call graph touches; the allowlists (`DEFAULT_PLUGIN_DTYPE_OPTIONS`
   and `PLUGIN_DTYPE_OPTIONS_MAP`) are pure data, invisible to a `raise`-scanning
   walk.
3. Per-subclass `supports_backend()` overrides across the whole DecodingBaseConfig
   family - the predicate differs per subclass (Medusa/Lookahead exclude
   pytorch/_autodeploy; NGram/MTP/Auto require pytorch; DraftTarget requires
   pytorch ONLY at 1.1.0). An entry-point walk that hits only one routed subclass
   would not enumerate the sibling predicates.
4. `MoeConfig.backend`, `KvCacheConfig.mamba_ssm_cache_dtype`,
   `CalibConfig.device`, the scheduler/batching StrEnums - declarative Literal /
   StrEnum membership on nested sub-models, enforced by pydantic typing alone.
5. `CacheTransceiverConfig.backend` Literal on a deep nested optional config.
6. `LoraConfig.lora_ckpt_source` and `QuantAlgo` membership living in
   out-of-tree files (`lora_helper.py`, `quantization/mode.py`).

## Where 1.1.0 sits between 1.0.0 and 1.2.1

1.1.0 has ALREADY adopted (present, mined here): `MoeConfig.backend` Literal,
`KvCacheConfig.mamba_ssm_cache_dtype` Literal + `max_gpu_total_bytes` /
`max_attention_window` validators, `guided_decoding_backend` Literal, `LoadFormat`
with VISION_ONLY, both `TorchCompileConfig` validators, the attention_dp /
batch_wait / stream_interval validators, `allreduce_strategy` Literal,
`TrtLlmArgs.validate_kv_cache_dtype` (assert dtype=="auto"),
`validate_peft_cache_config` raise.

1.1.0 has NOT YET adopted (these are 1.2.1-new and are correctly ABSENT here):

- PluginConfig pydantic-isation (1.1.0 is a plain `@dataclass`; per-field pydantic
  Literals are 1.2.1-new, replaced here by the setter-assert allowlist entries).
- `SamplingParams` top_p / top_k / temperature range checks (1.1.0 `_validate` has
  only best_of<n, the greedy best_of>1 env gate, and truncate_prompt_tokens>=1).
- `CacheTransceiverConfig` timeout `Field(gt=0)` fields (kv_transfer_timeout_ms,
  kv_transfer_sender_future_timeout_ms).
- `KvCacheConfig.free_gpu_memory_fraction` validator.
- `LoraConfig.lora_ckpt_source` pydantic Literal (1.1.0 uses a dataclass
  `__post_init__` assert).
- `Nvfp4GemmConfig`, `MoeLoadBalancerConfig.setup` divisibility,
  `RayPlacementConfig`, `orchestrator_type` Literal,
  `BaseSparseAttentionConfig` family, `SaveHiddenStatesDecodingConfig`.
- `DecodingBaseConfig` acceptance_window / acceptance_length_threshold /
  draft_len_schedule validators (the from_dict registry is 8-key here vs 9-key at
  1.2.1, missing SaveState).
- HELIX `cp_config.tokens_per_block` check, `ray_worker_extension_cls` /
  `ray_placement_config` requires-ray checks.
- `NVFP4_AWQ` QuantAlgo member (24 members at 1.1.0 vs 26 at 1.2.1).
- EagleDecodingConfig model_validator (max_draft_len required, dynamic-tree
  checks) - 1.1.0 has only a plain `.validate()` checking speculative_model_dir.

## Key behaviour delta (not just additions)

`validate_build_config_with_runtime_params` is on `BaseLlmArgs` at 1.1.0 (not
`TrtLlmArgs` as in 1.2.1) and it RAISES `ValueError` when `max_batch_size` or
`max_num_tokens` exceed the corresponding `build_config` value - the 1.0.0-style
hard error. 1.2.1 demoted those two cases to warn/clamp. The `max_seq_len` /
`max_beam_width` / `max_input_len` cases are warn-only in both versions.
Similarly, the engine/checkpoint parallel-size consistency checks
(`_load_config_from_engine` / `_load_config_from_ckpt`) and
`validate_speculative_config` live on `BaseLlmArgs` at 1.1.0, whereas 1.2.1 split
them across `TrtLlmArgs` / `TorchLlmArgs`.

## Items flagged

- `torchLlmArgs_warn_on_unstable_feature_usage` (def line 2445): NOT decorated with
  `@model_validator` at 1.1.0 (same as the 1.2.1 finding). As written it is a plain
  method and does not fire automatically on construction. Recorded with
  `pass_b_flag: possibly_invalid_not_auto_invoked`.
- `torchLlmArgs_validate_batch_wait_timeout_ms_non_negative`: message text says
  "greater than 0" but the predicate is `< 0` (0 is accepted). Kept; kwargs reflect
  the actual `< 0` predicate (positive=-1, negative=0).

## Blind spots (what an entry-point / call-graph walk should catch that B misses)

A type-tree walk under-covers rules that live in the execution path rather than a
config class: `BuildConfig` build-time asserts in `builder.py`, cross-config rules
wired in `LLM.__init__` / `_build_model`, C++/pybind validation surfaced only at
the `_to_pybind` round-trip, and speculative-decoding side effects that fire during
model-engine construction. Those are deferred to Pass A.
