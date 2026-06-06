# Pass A - entry-point / call-graph walk methodology (tensorrt-llm 1.1.0)

Engine source: `/tmp/trt-llm-1.1.0/tensorrt_llm/` (confirmed `__version__ = "1.1.0"`
via `version.py`).

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of the
two-pass bake-off. Goal: maximise recall of construction-time validation invariants
reachable from public, user-facing entry points. There is NO PoC ground truth for
1.1.0, so every entry is derived directly from 1.1.0 source. `provenance` marks
whether the same predicate also exists in the already-mined 1.2.1 sibling Pass A
(`fold`) or is genuinely 1.1.0-local in shape (`net_new`).

## Traversal (what I walked)

Starting roots (public surface a benchmark harness actually constructs):

1. `tensorrt_llm.LLM(...)`. In 1.1.0 `tensorrt_llm.LLM == _TorchLLM` (llm.py:1052),
   whose `__init__` calls `_validate_args_for_torch_backend(kwargs)` (llm.py:1031)
   before `BaseLLM.__init__` (llm.py:108) -> `llm_args_cls.from_kwargs(...)`
   (llm.py:156). The TRT path is `tensorrt_llm._tensorrt_engine.LLM` -> `_TrtLLM` ->
   `TrtLlmArgs`.
2. `from_kwargs` (llm_args.py:1445) -> `_check_consistency` -> pydantic construction
   of `TorchLlmArgs` / `TrtLlmArgs` (both subclass `BaseLlmArgs`), firing the full
   validator chain: every `@field_validator`, `@model_validator(mode="after")`,
   `Field(...)` Literal, plus `_check_consistency`.
3. Nested config fields reached transitively: `KvCacheConfig`, `CudaGraphConfig`,
   `TorchCompileConfig`, `MoeConfig`, `AttentionDpConfig`, `SchedulerConfig`
   (Capacity/ContextChunking), `CacheTransceiverConfig`, `CalibConfig`,
   `PeftCacheConfig`, `LoraConfig` (-> `lora_helper.py`), `QuantConfig`
   (-> `quantization/mode.py` + `models/modeling_utils.py`), `PluginConfig`
   (-> `plugin/plugin.py`), the `*DecodingConfig` speculative family
   (-> `DecodingBaseConfig.from_dict` dispatch + `validate_speculative_config` per-
   class asserts + `supports_backend`).
4. `SamplingParams(...)` -> `__post_init__` -> `_validate` (sampling_params.py:291)
   -> `GuidedDecodingParams._validate` (sampling_params.py:32). Also the deferred
   `_get_bad_words`/`_get_stop_words` setup guards.
5. TRT-only deferred config loaders reached from the args build:
   `_load_config_from_engine`, `_load_config_from_ckpt`, `get_model_format`.

## Method

- Enumerated every validator via grep for `@field_validator`, `@model_validator`,
  `def validate_`, `def _validate`, `__post_init__`, then read each in context to
  classify predicate + outcome + replayability.
- Enumerated every `Literal[...]` field and every `Field(...)` numeric constraint
  (none impose `ge/gt/le/lt` directly on headline numeric knobs at 1.1.0; the bounds
  live in validator functions, all captured).
- Read the non-pydantic config layers explicitly (PluginConfig, LoraConfig,
  QuantConfig) because at 1.1.0 these are NOT pydantic and their validation shape
  (assert in property setter / `__post_init__` / no validation at all) is materially
  different from 1.2.1.

## Where 1.1.0 sits in the 1.0.0 -> 1.2.1 window (verified against 1.1.0 source)

- **PluginConfig**: still a non-pydantic `@dataclass(slots=True)` (plugin.py:140)
  with `PluginConfigMeta`-generated property setters that `assert` against
  `DEFAULT_PLUGIN_DTYPE_OPTIONS` / `PLUGIN_DTYPE_OPTIONS_MAP`, and a `dtype` setter
  that asserts not-`auto`/not-`None`. This is the PRE-pydantic stage; 1.2.1 migrates
  to pydantic Literals + `validate_dtype_not_auto`. Captured as assert/allowlist
  entries with `replay_via` set-attribute notes (not bare-constructor kwargs).
- **SamplingParams._validate**: has only `best_of >= n`, the `best_of>1` greedy-env
  guard, and `truncate_prompt_tokens >= 1`. There are NO top_p/top_k/temperature
  range checks at 1.1.0 - those are added by 1.2.1. Confirmed by reading
  sampling_params.py:291-325.
- **CacheTransceiverConfig**: `backend` Literal has 4 members
  (`DEFAULT/UCX/NIXL/MPI`), NO `MOONCAKE`; and there is NO `timeout`/
  `kv_transfer_timeout_ms` `Field(gt=0)`. 1.2.1 adds MOONCAKE and timeout gt=0
  fields. (llm_args.py:1150)
- **validate_build_config_with_runtime_params**: at 1.1.0 this RAISES `ValueError`
  when runtime `max_batch_size`/`max_num_tokens` exceed the build_config values, and
  only WARNS for `max_seq_len`/`max_beam_width`/`max_input_len` mismatches
  (llm_args.py:1650). The overshoot raise was later softened to warn-only by 1.2.1.
  This is the window's raise-vs-warn divergence; recorded as two `net_new` error
  entries (the raises) plus three `fold` warn entries.
- **LoraConfig**: `@dataclass(DictConversion)` with a `__post_init__` assert on
  `lora_ckpt_source in {hf, nemo}` (lora_helper.py:94) - NOT a pydantic Literal yet
  (1.2.1 is pydantic). DOES fire at construction (CPU-replayable).
- **QuantConfig**: plain `@dataclass` (modeling_utils.py:128). `quant_algo` is typed
  `Optional[QuantAlgo]` but a plain dataclass does NOT coerce/validate a bad string
  at construction; the StrEnum membership and `KV_CACHE_QUANT_ALGO_LIST` allowlist
  only fire when `QuantAlgo(value)` is coerced or `QuantMode.from_quant_algo` is
  called. Recorded as dormant/replay_via. 1.1.0 `QuantAlgo` also lacks `NVFP4_AWQ`.

## Net-new (4 entries, 1.1.0-local shape)

1. `validate_build_config_with_runtime_params` max_batch_size overshoot RAISE
   (llm_args.py:1659) - error in 1.1.0, warn in 1.2.1.
2. `validate_build_config_with_runtime_params` max_num_tokens overshoot RAISE
   (llm_args.py:1664) - same.
3. `EagleDecodingConfig` `speculative_model_dir`-required assert inside
   `validate_speculative_config` ("Path to EAGLE3 weights must be specified.",
   llm_args.py:1748) - distinct from the 1.2.1 `validate_eagle_config` raise.
4. `EagleDecodingConfig.validate()` "Draft model must be provided for EAGLE"
   (llm_args.py:459) - a plain `validate()` method, NOT a pydantic model_validator,
   so it is dormant for bare-constructor replay; no `validate_eagle_config` /
   `max_draft_len is required` raise exists in 1.1.0.

## Folded (76 entries)

Predicates that the 1.2.1 Pass A also catalogued and that I independently re-derived
in 1.1.0 source, with citations re-resolved to the 1.1.0 line numbers. Notable shape
differences vs 1.2.1 noted inline in the YAML (e.g. `validate_kv_cache_dtype` is an
assert in 1.1.0; `warn_on_unstable_feature_usage` is NOT a model_validator in 1.1.0;
the build-config override validators live on `BaseLlmArgs` in 1.1.0, not split onto
`TrtLlmArgs`).

## Replayability notes for the downstream gate

- CPU-replayable (24 entries, no `dormant_reason`): the nested-config pure-pydantic /
  dataclass validators - `KvCacheConfig`, `CudaGraphConfig`, `TorchCompileConfig`,
  `MoeConfig`, `CalibConfig`, `SchedulerConfig` enums, `CacheTransceiverConfig`,
  `LookaheadDecodingConfig`, `LoraConfig.__post_init__`, `SamplingParams`,
  `GuidedDecodingParams`, plus the `DecodingBaseConfig.from_dict` dispatch
  (`replay_via: from_dict`). 20 of these carry explicit kwargs_positive/negative;
  the rest annotate `replay_via`.
- DORMANT (56 entries): everything that needs a GPU/SM/engine/ckpt/model dir, plus
  ALL args-model field-level entries. CRITICAL: `BaseLlmArgs.validate_dtype`
  (llm_args.py:1491) and `validate_gpus_per_node` (llm_args.py:1503) call
  `torch.cuda.get_device_properties(0)` / `torch.cuda.device_count()`
  unconditionally at model construction, so ANY `TorchLlmArgs`/`TrtLlmArgs`
  field-level kwargs replay requires a CUDA host. The gate runs `--gpus all`, so the
  Literals/validators still fire there; they are marked `dormant_reason` so a
  CPU-only host skips them. SM-100 plugin, engine/ckpt parallel-size mismatches,
  `get_model_format` config.json, and the bad/stop-words setup guards are genuinely
  host/artefact-dependent.
- `best_of_gt_1_greedy` is env-sensitive (`TLLM_ALLOW_N_GREEDY_DECODING` must be
  unset to fire).

## Blind spots (what a class-hierarchy walk should catch that this pass did not)

1. Validators on base/sibling classes never routed to from the public ctor
   (`MedusaDecodingConfig`, `UserProvidedDecodingConfig` internals; abstract-base
   `DecodingBaseConfig.validate`/`_check_fields` which are no-ops here).
2. Inherited pybind / C++ mirror constraints on `PybindMirror` subclasses
   (`KvCacheConfig`, `SchedulerConfig`, `PeftCacheConfig`, `CacheTransceiverConfig`,
   `LookaheadDecodingConfig`) enforced C++-side, invisible to a Python call-graph
   walk.
3. The full `BuildConfig` + non-pydantic `PluginConfig` deep tree (the default
   PyTorch LLM does not build a TRT engine, so only the surface PluginConfig setters
   were walked).
4. `@field_validator` defined on a parent and overridden in a child (MRO override
   resolution).
5. Standalone enums reachable only via `from_dict(string)` paths on classes not
   routed to.
