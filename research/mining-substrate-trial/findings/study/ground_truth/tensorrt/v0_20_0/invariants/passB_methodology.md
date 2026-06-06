# Pass B methodology - tensorrt-llm v0.20.0 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the tensorrt-llm config surface by TYPE, walking the class
hierarchy / MRO of every config/args/params model. It is the complement to a
Pass A entry-point walk: the goal is to catch what an entry-point walk misses
(validators on base/sibling classes, overridden validators, declarative
Literal/enum typing, dataclass-property asserts, helper-dataclass property
raises).

- Engine source: `/tmp/trt-llm-0.20.0/tensorrt_llm/`
  (`version.py` confirms `__version__ = "0.20.0"`).
- Pure source analysis, no GPU, no model download. Every cited line read directly
  from 0.20.0 source; the 0.21.0 passB catalogue was used as a schema/convention
  template only, with every rule re-derived from 0.20.0 source.

## Type tree enumerated (0.20.0)

Root config file `llmapi/llm_args.py` is 1457 lines (0.21.0 is 2072, 1.2.1 is
3416). Classes located via `grep -n "^class "` and read in full:

- LlmArgs tree: a SINGLE monolithic `LlmArgs(BaseModel)` with
  `model_config = {"arbitrary_types_allowed": True, "extra": "allow"}`. There is
  NO BaseLlmArgs / TrtLlmArgs / TorchLlmArgs split and NO `_AutoDeployLlmArgs`
  (0.21.0 introduced the split). This is the headline 0.20-vs-0.21 difference.
- Nested sub-models (plain `BaseModel`, extra allowed): `CalibConfig`,
  `DecodingBaseConfig` (+ Medusa / Eagle / MTP / Lookahead - only 4 subclasses),
  `DynamicBatchConfig`, `SchedulerConfig`, `PeftCacheConfig`, `KvCacheConfig`,
  `ExtendedRuntimePerfKnobConfig`, `CacheTransceiverConfig`.
- `PybindMirror` ABC + `PybindMirrorEnumMeta`; enums `BatchingType`,
  `CapacitySchedulerPolicy`, `ContextChunkingPolicy`.
- `_ParallelConfig` (@dataclass) and `_ModelWrapper` (@dataclass) helpers built by
  LlmArgs in model_post_init / _setup.
- Out-of-file types walked:
  - `PluginConfig` (`plugin/plugin.py`) - @dataclass(slots=True) with a metaclass
    (`PluginConfigMeta`) that wraps every `_field` as a property whose setter
    asserts the value against an allowlist (`DEFAULT_PLUGIN_DTYPE_OPTIONS` /
    `PLUGIN_DTYPE_OPTIONS_MAP`).
  - `BuildConfig` (`builder.py`) - plain @dataclass with NO `__post_init__`.
  - `QuantConfig` (`models/modeling_utils.py`) - plain @dataclass with NO
    `__post_init__`; `QuantAlgo` (20 members) + `KV_CACHE_QUANT_ALGO_LIST` +
    `QuantMode.from_quant_algo` (`quantization/mode.py`).
  - `LoraConfig` (`lora_manager.py`) - `DictConversion` @dataclass with a
    `__post_init__` assert. NOTE: there is NO `lora_helper.py` in 0.20.0.
  - `SamplingParams` + `GuidedDecodingParams` (`sampling_params.py`) - @dataclass.

## Extraction per class

For each class in the MRO the following were harvested:

1. `Literal[...]` / `StrEnum` / `Enum`-typed fields -> membership rules
   (`literal_in` / `strenum_in`).
2. Inline `Field(gt/ge/le/lt/...)` numeric constraints. (0.20.0 llm_args.py has
   NONE - all numeric bounds are inside validator bodies or method bodies.)
3. `@validator` (pydantic v1) bodies -> every `raise`. The ONLY pydantic validator
   in 0.20.0 llm_args.py is `LookaheadDecodingConfig.validate_positive_values`.
4. `model_post_init` / `__post_init__` / `_setup` / `_ensure_*` /
   `_maybe_update_config_for_consistency` method bodies -> every `raise` /
   `assert` / `logger.warning`. In 0.20.0 most LlmArgs-level checks live here, not
   in named validators (0.21.0 promoted many to @field_validator/@model_validator).
5. `from_dict` dispatch tables -> `decode_dispatch` (DecodingBaseConfig, 4 keys).
6. dataclass-property setter asserts (PluginConfig) -> allowlist_constant / type_is.
7. helper-dataclass property raises (`_ParallelConfig.world_size`).

`outcome` derived from severity: error/raise/assert -> `invalid`; warn ->
`valid_with_warning`; catalogue self-check -> `meta`.

## kwargs replayability discipline

`kwargs_positive` (FIRING / invalid) / `kwargs_negative` (VALID) pairs were emitted
ONLY where the predicate is reachable at plain construction on a CPU host with no
side effects: standalone sub-model field validators (`LookaheadDecodingConfig`,
`CalibConfig`, `SchedulerConfig` enums), the standalone dataclasses
(`QuantConfig`, `LoraConfig`, `SamplingParams`), the `DecodingBaseConfig.from_dict`
dispatch (`replay_via: from_dict`), the standalone enum call
(`BatchingType("...")`, `replay_via: enum_call`), and the `GuidedDecodingParams`
mutual-exclusion (`replay_via: via_sampling_params`, since `_validate` is invoked
only through `SamplingParams._validate`).

NO kwargs were fabricated for:

- Any rule that lives on `LlmArgs` - constructing it triggers `model_post_init`
  (tokenizer_factory, torch.cuda.get_device_properties, torch.cuda.device_count,
  parallel-config build) and, via `from_kwargs`, `_setup` (model-format inference
  on the filesystem, build-config init). These are marked
  `dormant_reason: full-LlmArgs construction side-effects`.
- GPU-gated rules (`bfloat16` SM<80 check, `gpus_per_node` default-fill,
  `PluginConfig.validate` SM-100).
- Filesystem-stateful rules (`get_model_format`, engine/ckpt loaders).
- Env-gated rules (greedy `best_of`).
- Lifecycle-only rules (`SamplingParams._get_bad_words` / `_get_stop_words`,
  `QuantConfig.kv_cache_quant_algo` which fires on `quant_mode` property access not
  construction).
- PluginConfig setter asserts: these fire on ATTRIBUTE ASSIGNMENT (every field is
  `init=False`), not at `PluginConfig()` construction, so they carry
  `replay_via: attribute_set` + `dormant_reason`.
- `_ParallelConfig.world_size` raises: property-access raises during LlmArgs init.

## Result

- 56 total invariants.
- 46 folded analogues (a 0.21.0 / 1.2.1 passB rule whose 0.20.0 form was
  re-derived and re-cited from 0.20.0 source).
- 10 net-new, all class-hierarchy / type-tree / method-body constraints an
  entry-point walk misses:
  - `llmArgs_setup_invalid_embedding_parallel_mode` (free-str field whose 3-value
    allowlist is enforced only by an if/elif/else raise in
    `_setup_embedding_parallel_mode`).
  - `llmArgs_ensure_lora_config_consistency_max_lora_rank_ignored_warns`
    (0.20.0-specific deprecation-shim warning for the flat max_lora_rank field that
    0.21.0 dropped).
  - `parallelConfig_world_size_auto_parallel_with_manual_tp_pp_cp` and
    `parallelConfig_world_size_gt_1_without_auto_parallel` (property-access raises
    in the `_ParallelConfig` helper dataclass).
  - five PluginConfig allowlist / type rules enforced by the metaclass property
    setter: `gemm_swiglu_plugin`, `low_latency_gemm_plugin`,
    `low_latency_gemm_swiglu_plugin`, `gemm_allreduce_plugin` (restricted
    allowlists in PLUGIN_DTYPE_OPTIONS_MAP), and `pluginConfig_bool_field_type_assert`
    (the isinstance(bool) assert applied to every bool plugin field).
  - The `default_dtype_allowlist_family` rule is the canonical representative of the
    whole DEFAULT_PLUGIN_DTYPE_OPTIONS-typed plugin-field family (net-new mechanism
    class).
- 13 invariants carry CPU-replayable kwargs; 43 are dormant (no kwargs).

## 0.20.0-vs-0.21.0 API differences caught

These are the cross-version deltas that make 0.20.0 NOT a copy of the 0.21.0
catalogue:

1. SINGLE monolithic `LlmArgs(BaseModel)` with `extra="allow"`. 0.21.0 split it
   into `BaseLlmArgs (extra=forbid) -> TrtLlmArgs / TorchLlmArgs -> _AutoDeployLlmArgs`.
   Every 0.21.0 BaseLlmArgs/TrtLlmArgs/TorchLlmArgs rule collapses onto `LlmArgs`
   here, and the 0.21.0 `_AutoDeployLlmArgs` rules (free_mem_ratio range,
   model_factory / mla_backend Literals) have NO analogue in 0.20.0 - the subclass
   does not exist.
2. 0.20.0 uses pydantic v1 `@validator`. The only one in llm_args.py is
   `LookaheadDecodingConfig.validate_positive_values`. 0.21.0 promoted many LlmArgs
   checks to `@field_validator` / `@model_validator`
   (validate_dtype, validate_gpus_per_node, validate_build_config_with_runtime_params,
   validate_model_format_misc, validate_lora_config_consistency,
   validate_speculative_config, validate_enable_build_cache, convert_load_format,
   validate_cuda_graph_*, validate_moe_load_balancer). In 0.20.0 the equivalent
   logic lives inline in `model_post_init`, `_setup`,
   `_ensure_lora_config_consistency`, `_maybe_update_config_for_consistency`, and
   `_setup_embedding_parallel_mode` - method bodies, not validators.
3. The speculative `max_draft_len > 0` asserts live in `LlmArgs._setup`
   (lines 1193 Medusa, 1200 Eagle), NOT in a validator. The
   "Speculative config type not recognized" raise is the else-branch of the
   `_setup` isinstance ladder (line 1232). 0.21.0 routes all of these through one
   `validate_speculative_config` validator.
4. `validate_build_config_with_runtime_params` does not exist. The build-config
   reconcile is a `logger.warning` in `_maybe_update_config_for_consistency`
   (loops max_input_len / max_seq_len / max_beam_width) and `_setup`
   (max_batch_size / max_num_tokens conflict warnings). 0.20.0 WARNS only; 0.21.0
   RAISES on max_batch_size / max_num_tokens; 1.2.1 clamps and warns.
5. `DecodingBaseConfig.from_dict` dispatches only 4 keys (MTP/Medusa/Eagle/Lookahead);
   0.21.0 added NGram/DraftTarget (6), 1.2.1 -> 9. The speculative_config Union has
   4 members. DecodingBaseConfig has none of the later acceptance_window /
   acceptance_length_threshold / draft_len_schedule validators.
6. `QuantAlgo` has 20 members (0.21.0: 21, adds W4A8_MXFP4_FP8; 1.2.1: 26).
   `QuantConfig` is a plain @dataclass with NO `__post_init__`; the
   kv_cache_quant_algo allowlist assert fires lazily via the `quant_mode`
   cached_property, not at construction.
7. `LoraConfig` is a `DictConversion` @dataclass in `lora_manager.py` with a
   `__post_init__` assert on `lora_ckpt_source in ['hf','nemo']` (AssertionError).
   Identical to 0.21.0; 1.2.1 made it a pydantic Literal in a new `lora_helper.py`.
8. `KvCacheConfig` / `CacheTransceiverConfig` / `SamplingParams` (top_p / top_k /
   temperature) range checks are all 0.21.0+/1.2.1+ and ABSENT here, exactly as in
   0.21.0. `SamplingParams._validate` checks only best_of>=n (guarded by
   best_of>1), greedy best_of, and truncate_prompt_tokens>=1. CacheTransceiverConfig
   has only `max_num_tokens` (no backend Literal, no Field(gt=0) timeouts).
9. `PluginConfig` is a `@dataclass(slots=True)` enforcing dtype allowlists via
   property-setter `assert` (metaclass mechanism), identical to 0.21.0. 1.2.1
   migrated these to pydantic Literal fields. predicate_kind differs
   (allowlist_constant / type_is vs literal_in) and the firing point is
   attribute-set, not construction.
10. There is NO `LoadFormat` StrEnum and NO `convert_load_format` override;
    `load_format` is a plain Literal['auto','dummy'] on the one LlmArgs class.
11. There is NO CudaGraphConfig, MoeConfig, Nvfp4GemmConfig, RayPlacementConfig,
    AttentionDpConfig, sparse-attention config family, MoeLoadBalancerConfig,
    or TorchCompileConfig in 0.20.0 (these arrive in 0.21.0+/1.2.1+). The
    `embedding_parallel_mode` field is a free str validated only by a method-body
    raise (net-new pass-B finding).

## Class-hierarchy / type-tree cases an entry-point walk would miss (this pass's point)

- PluginConfig metaclass property-setter allowlist asserts - enforced by a
  generated property, never an explicit raise in any call path; a `raise`-scanning
  walk misses every one.
- The `embedding_parallel_mode` 3-value allowlist - the field declaration is a
  plain `str`; the constraint exists only in the `_setup_embedding_parallel_mode`
  if/elif/else body.
- `_ParallelConfig.world_size` getter raises - property-access raises in a helper
  @dataclass that LlmArgs builds in model_post_init; not a field constraint.
- Standalone enum membership reachable only via `SchedulerConfig` /
  `update_llm_args_with_extra_dict` from_dict dispatch
  (BatchingType / CapacitySchedulerPolicy / ContextChunkingPolicy).
- `LoraConfig` / `QuantConfig` dataclass asserts (in lora_manager.py /
  quantization/mode.py) that the entry-point walk reaches only indirectly.

## Blind spots (what a Pass A entry-point walk should catch that B under-covers)

1. `BuildConfig` carries NO field validators and NO construction-time numeric
   constraints; all its checks are `assert`s inside `build()` / weight-load methods
   and `update_kv_cache_type`. Call-graph invariants invisible to a type walk.
2. Cross-config wiring in `LLM.__init__` / `_build_model` /
   `_setup` (build_config field clobbering, plugin_config mutation) that no single
   config class owns. The `_setup` body in 0.20.0 carries many of these
   (build_config defaulting, nccl_plugin nulling, lora_plugin enable, prompt-adapter
   sizing) which are state mutations rather than validity predicates.
3. C++/pybind validation in `bindings.executor` surfaced only at the `_to_pybind`
   round-trip (KvCacheConfig / PeftCacheConfig numeric bounds live there in 0.20.0).
4. Speculative-decoding side effects that fire during model-engine construction.
5. `_validate_kv_cache_config` (streaming-LLM gated raises at llm_args.py:1361) -
   reachable only via the streaming-LLM call path, not at config construction.

## Uncertain / flagged

- All `LlmArgs` rules are recorded with kwargs omitted because LlmArgs
  construction in 0.20.0 has hard side effects (model_post_init runs
  tokenizer_factory + torch.cuda queries unconditionally; _setup infers model
  format from the filesystem). A Pass A fixture with a stub model dir +
  skip_tokenizer_init may replay some; left to the gate. They are GT contributors
  regardless.
- `setup_model_type_must_be_str_or_path`: pydantic coerces the Union[str, Path]
  field before _setup, so the explicit assert is only reachable for non-coercible
  inputs; the effective predicate is largely subsumed by field typing.
- PluginConfig dtype/allowlist asserts fire on attribute SET, not construction
  (every field is `init=False`). Marked `replay_via: attribute_set` + dormant so
  the gate does not expect a construction-time failure.
- The speculative asserts and the embedding_parallel_mode / build-config reconcile
  warnings fire in `_setup` / `_maybe_update_config_for_consistency`, which run via
  `LlmArgs.from_kwargs(...)._setup()`, NOT at plain `LlmArgs(**kwargs)`
  construction. The gate must route through `from_kwargs` (with full side effects)
  to reach them, hence dormant.
