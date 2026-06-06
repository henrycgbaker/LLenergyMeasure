# Pass B methodology - tensorrt-llm v0.21.0 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the tensorrt-llm config surface by TYPE, walking the class
hierarchy / MRO of every config/args/params model. It is the complement to a
Pass A entry-point walk: the goal is to catch what an entry-point walk misses
(validators on base/sibling classes, overridden validators, subclasses the public
path does not route to, declarative Literal/enum typing, dataclass-property
asserts).

- Engine source: `/tmp/trt-llm-0.21.0/tensorrt_llm/`
  (`version.py` confirms `__version__ = "0.21.0"`).
- Pure source analysis, no GPU, no model download. Every cited line read directly
  from 0.21.0 source; nothing copied from the 1.2.1 catalogue without re-deriving.

## Type tree enumerated (0.21.0)

Root config file `llmapi/llm_args.py` is 2072 lines (1.2.1 is 3416). Classes
located via `grep -n "^class "` and read in full:

- LlmArgs tree: `BaseLlmArgs` (model_config extra=forbid) -> `TrtLlmArgs`,
  `TorchLlmArgs` -> `_AutoDeployLlmArgs`. `LlmArgs = TrtLlmArgs`.
- Nested sub-models (plain `BaseModel`, extra allowed): `CalibConfig`,
  `DecodingBaseConfig` (+ Medusa/Eagle/NGram/DraftTarget/MTP/Lookahead),
  `DynamicBatchConfig`, `SchedulerConfig`, `PeftCacheConfig`, `KvCacheConfig`,
  `ExtendedRuntimePerfKnobConfig`, `CacheTransceiverConfig`, `TorchCompileConfig`.
- `PybindMirror` ABC + `PybindMirrorEnumMeta`; enums `BatchingType`,
  `CapacitySchedulerPolicy`, `ContextChunkingPolicy`, `LoadFormat`.
- `_ParallelConfig` (@dataclass) and `_ModelWrapper` (@dataclass) helpers.
- Out-of-file types walked:
  - `PluginConfig` (`plugin/plugin.py`) - @dataclass(slots=True) with a metaclass
    (`PluginConfigMeta`) that wraps every `_field` as a property whose setter
    asserts the value against an allowlist.
  - `BuildConfig` (`builder.py`) - plain @dataclass.
  - `QuantConfig` (`models/modeling_utils.py`) - plain @dataclass; `QuantAlgo` +
    `KV_CACHE_QUANT_ALGO_LIST` + `QuantMode.from_quant_algo` (`quantization/mode.py`).
  - `LoraConfig` (`lora_manager.py`) - `DictConversion` @dataclass with
    `__post_init__` assert. NOTE: there is NO `lora_helper.py` in 0.21.0.
  - `SamplingParams` + `GuidedDecodingParams` (`sampling_params.py`) - @dataclass.

## Extraction per class

For each class in the MRO the following were harvested:

1. `Literal[...]` / `StrEnum` / `Enum`-typed fields -> membership rules
   (`literal_in` / `strenum_in`).
2. Inline `Field(gt/ge/le/lt/...)` numeric constraints. (0.21.0 llm_args.py has
   NONE - all numeric bounds are inside validator bodies.)
3. `@field_validator` / `@model_validator` bodies -> every `raise` / `logger.warning`.
4. `__init__` / `__post_init__` / `model_post_init` checks (LookaheadDecodingConfig
   `_check_fields`, LoraConfig, _AutoDeployLlmArgs model_post_init, SamplingParams).
5. `from_dict` dispatch tables -> `decode_dispatch` (DecodingBaseConfig).
6. dataclass-property setter asserts (PluginConfig) -> allowlist_constant / type_is.

`outcome` derived from severity: error/raise/assert -> `invalid`; warn ->
`valid_with_warning`; catalogue self-check -> `meta`.

## kwargs replayability discipline

`kwargs_positive` (FIRING / invalid) / `kwargs_negative` (VALID) pairs were emitted
ONLY where the predicate is reachable at plain construction on a CPU host with no
side effects: standalone sub-model field validators (`LookaheadDecodingConfig`,
`CalibConfig`, `SchedulerConfig` enums), the standalone dataclasses
(`QuantConfig`, `LoraConfig`, `SamplingParams`), the `DecodingBaseConfig.from_dict`
dispatch (`replay_via: from_dict`), the standalone enum calls
(`BatchingType("...")`, `replay_via: enum_call`), and the
`GuidedDecodingParams` mutual-exclusion (`replay_via: via_sampling_params`, since
`_validate` is invoked only through `SamplingParams._validate`).

NO kwargs were fabricated for:
- Any rule that lives on `BaseLlmArgs` / `TrtLlmArgs` / `TorchLlmArgs` /
  `_AutoDeployLlmArgs` - constructing those triggers tokenizer init, model-format
  inference (filesystem), parallel-config build, and build-config init. These are
  marked `dormant_reason: full-LlmArgs construction side-effects`.
- GPU-gated rules (`validate_dtype`, `validate_gpus_per_node`,
  `PluginConfig.validate` SM-100).
- Filesystem-stateful rules (`get_model_format`, engine/ckpt loaders,
  `moe_load_balancer`).
- Env-gated rules (greedy `best_of`).
- Lifecycle-only rules (`SamplingParams._get_bad_words` / `_get_stop_words`,
  `QuantConfig.kv_cache_quant_algo` which fires on `quant_mode` property access not
  construction).
- PluginConfig setter asserts: these fire on ATTRIBUTE ASSIGNMENT (every field is
  `init=False`), not at `PluginConfig()` construction, so they carry
  `replay_via: attribute_set` + `dormant_reason` (the gate may replay by
  constructing then setting the attribute, but no construction kwargs exist).

## Result

- 64 total invariants.
- 54 folded analogues (a 1.2.1 passB rule whose 0.21.0 form was re-derived and
  re-cited from 0.21.0 source).
- 10 net-new, all class-hierarchy / type-tree constraints an entry-point walk
  misses:
  - `baseLlmArgs_validate_gpus_per_node_default_warn` (before-validator default-fill).
  - `autoDeployLlmArgs_free_mem_ratio_range`, `autoDeployLlmArgs_model_factory_literal`,
    `autoDeployLlmArgs_mla_backend_literal` (rules on `_AutoDeployLlmArgs`, a
    TorchLlmArgs subclass reachable only via backend='_autodeploy' - the public
    LLM(...) walk never routes to it).
  - five PluginConfig allowlist rules enforced by the metaclass property setter:
    `gemm_swiglu_plugin`, `low_latency_gemm_plugin`,
    `low_latency_gemm_swiglu_plugin`, `gemm_allreduce_plugin` (restricted
    allowlists in PLUGIN_DTYPE_OPTIONS_MAP), and `pluginConfig_bool_field_type_assert`
    (the isinstance(bool) assert applied to every bool plugin field).
  - The `default_dtype_allowlist_family` rule is the canonical representative of the
    whole DEFAULT_PLUGIN_DTYPE_OPTIONS-typed plugin-field family (folded; same
    mechanism class as 1.2.1's representative).
- 14 invariants carry CPU-replayable kwargs; 49 are dormant (no kwargs).

## 0.21.0-vs-later (1.2.1) API differences caught

These are the cross-version deltas that make 0.21.0 NOT a copy of the 1.2.1
catalogue:

1. PluginConfig is a `@dataclass(slots=True)` enforcing dtype allowlists via
   property-setter `assert` (a metaclass mechanism). 1.2.1 migrated all of these
   to pydantic `Literal` fields. predicate_kind differs (allowlist_constant /
   type_is vs literal_in) and the firing point is attribute-set, not construction.
2. `KvCacheConfig` has NO Python field validators in 0.21.0
   (free_gpu_memory_fraction, max_gpu_total_bytes, max_attention_window,
   mamba_ssm_cache_dtype, dtype are all 1.2.1 additions; in 0.21.0 the bounds are
   C++-side only). None of those 1.2.1 KvCacheConfig invariants exist here.
3. `CacheTransceiverConfig` has only `max_num_tokens` - no `backend` Literal and no
   `kv_transfer_timeout_ms` / `kv_transfer_sender_future_timeout_ms` Field(gt=0).
   The two 1.2.1 net-new CacheTransceiver Field(gt=0) rules do not exist here.
4. `SamplingParams._validate` checks only best_of>=n, greedy best_of, and
   truncate_prompt_tokens>=1. The 1.2.1 top_p/top_k/temperature range checks are
   absent (C++-only in 0.21.0).
5. `DecodingBaseConfig.from_dict` dispatches 6 keys (MTP/Medusa/Eagle/Lookahead/
   NGram/DraftTarget). 1.2.1 added SaveState/UserProvided/AUTO (9). DecodingBaseConfig
   also has none of the 1.2.1 acceptance_window / acceptance_length_threshold /
   draft_len_schedule field validators.
6. QuantAlgo has 21 members (1.2.1: 26). `QuantConfig` is a plain @dataclass with no
   `__post_init__`; the kv_cache_quant_algo allowlist assert fires lazily via the
   `quant_mode` cached_property, not at construction.
7. `LoraConfig` is a `DictConversion` @dataclass in `lora_manager.py` with a
   `__post_init__` assert on `lora_ckpt_source in ['hf','nemo']` (AssertionError).
   1.2.1 made it a pydantic Literal in a new `lora_helper.py`.
8. The speculative-config asserts live in ONE `validate_speculative_config` on
   `BaseLlmArgs` (1.2.1 split them across TrtLlmArgs/TorchLlmArgs with per-class
   `supports_backend`). 0.21.0 has no sparse-attention config family,
   no Nvfp4GemmConfig, no MoeConfig, no RayPlacementConfig, no AttentionDpConfig,
   no CudaGraphConfig (the cuda-graph knob is a flat TorchLlmArgs int field).
9. `validate_build_config_with_runtime_params` RAISES ValueError on
   max_batch_size / max_num_tokens > build_config (1.2.1 clamps and warns).
10. `LoadFormat` enum has only AUTO, DUMMY (1.2.1 added VISION_ONLY).
11. `_load_config_from_engine` / `_load_config_from_ckpt` / `validate_model_format_misc`
    live on `BaseLlmArgs` in 0.21.0 (1.2.1 moved them to `TrtLlmArgs`).
12. The whole `model_config = StrictBaseModel` extra=forbid pattern of 1.2.1 is, in
    0.21.0, only on `BaseLlmArgs`; nested sub-models (KvCacheConfig, CalibConfig,
    SchedulerConfig, etc.) are plain BaseModel and ACCEPT extra fields, so an
    "unknown field" rejection that exists in 1.2.1 does NOT fire for those
    sub-models in 0.21.0.

## Class-hierarchy cases an entry-point walk would miss (this pass's whole point)

- `TorchLlmArgs.load_format` OVERRIDES the base `BaseLlmArgs.load_format` Literal
  with a different field type (Union[str, LoadFormat]) and a converting validator -
  a parent-field-overridden-in-child case.
- `_AutoDeployLlmArgs` (free_mem_ratio range, model_factory / mla_backend Literals,
  model_post_init default mutation) - a subclass the public LLM(...) walk does not
  route to.
- PluginConfig metaclass property-setter allowlist asserts - enforced by a
  generated property, never an explicit raise in any call path; a `raise`-scanning
  walk misses every one.
- Standalone enum membership reachable only via `SchedulerConfig` /
  `update_llm_args_with_extra_dict` from_dict dispatch
  (BatchingType/Capacity/Chunking).

## Blind spots (what a Pass A entry-point walk should catch that B under-covers)

1. `BuildConfig` carries NO field validators and NO construction-time numeric
   constraints; all its checks are `assert`s inside `build()` / weight-load methods
   (builder.py: encoder input==seq len equality line 891, EAGLE max_batch_size<=512
   line 1293, max_draft_len<=256 line 1294, SmoothQuant SM>=100 ban line 1178).
   Call-graph invariants invisible to a type walk.
2. Cross-config wiring in `LLM.__init__` / `_build_model` /
   `update_llm_args_with_extra_dict` (build_config field clobbering) that no single
   config class owns.
3. `_ParallelConfig` world_size getter/setter raises ("manually TP and PP are not
   supported in auto parallel mode", "world_size > 1 is only supported in auto
   parallel mode") fire at property access during LLM init, not at construction.
4. C++/pybind validation in `bindings.executor` surfaced only at the `_to_pybind`
   round-trip (KvCacheConfig / PeftCacheConfig numeric bounds live there in 0.21.0).
5. Speculative-decoding side effects that fire during model-engine construction.

## Uncertain / flagged

- All `BaseLlmArgs`-family rules are recorded with kwargs omitted because full
  LlmArgs construction in 0.21.0 has hard side effects (tokenizer factory, model
  format inference). A Pass A fixture with a stub model dir + skip_tokenizer_init
  may be able to replay some; left to the gate. They are GT contributors regardless.
- `validate_model` (str/Path type): pydantic coerces the Union[str, Path] field
  before the validator, so the explicit `raise` is only reachable for
  non-coercible inputs; the effective predicate is largely subsumed by field typing.
- PluginConfig dtype/allowlist asserts fire on attribute SET, not construction
  (every field is `init=False`). Marked `replay_via: attribute_set` + dormant so
  the gate does not expect a construction-time failure.
