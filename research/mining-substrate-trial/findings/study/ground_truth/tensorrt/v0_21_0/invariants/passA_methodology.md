# Pass A - entry-point / call-graph walk methodology (tensorrt-llm 0.21.0)

Engine source: `/tmp/trt-llm-0.21.0/tensorrt_llm/` (confirmed
`__version__ = "0.21.0"` via `version.py`).

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of
a two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal: maximise
recall of construction-time validation invariants reachable from public,
user-facing entry points, re-derived from the 0.21.0 source (line numbers and
API shape differ from the 1.2.1 template - everything here is derived from the
0.21.0 files actually read, not copied from 1.2.1).

## Traversal (what I walked)

Starting roots (public surface a benchmark harness actually constructs):

1. `tensorrt_llm.LLM(...)` -> `LLM(_TrtLLM)` -> `BaseLLM.__init__`
   (llm.py:103). The kwarg dispatch (llm.py:124-131) chooses `llm_args_cls`:
   `backend=='pytorch'` -> `TorchLlmArgs`, `'_autodeploy'` ->
   `_AutoDeployLlmArgs`, else (the DEFAULT) -> `TrtLlmArgs`. The generic
   unknown-kwarg gate fires at llm.py:139. Then
   `llm_args_cls.from_kwargs(...)` (llm.py:143) -> `BaseLlmArgs._check_consistency`
   (llm_args.py:1032) -> pydantic `cls(**kwargs)`.
2. The full pydantic validator chain on `BaseLlmArgs` / `TrtLlmArgs` /
   `TorchLlmArgs` / `_AutoDeployLlmArgs`: every `@field_validator`,
   `@model_validator(mode="after")`, every `Literal[...]` field, plus
   `_check_consistency`'s ExecutorConfig-superset assert.
3. Nested config fields reached transitively: `KvCacheConfig`, `CalibConfig`,
   `SchedulerConfig` (+`CapacitySchedulerPolicy`/`ContextChunkingPolicy`/
   `DynamicBatchConfig`), `PeftCacheConfig`, `BatchingType`, `QuantConfig`
   (-> `quantization/mode.py` + `models/modeling_utils.py`), `LoraConfig`
   (-> `lora_manager.py`), `PluginConfig` (-> `plugin/plugin.py`),
   `BuildCacheConfig`/`BuildCache` (-> `build_cache.py`), the `*DecodingConfig`
   speculative family (-> `DecodingBaseConfig.from_dict` dispatch +
   per-subclass asserts inside `validate_speculative_config`), and
   `_ParallelConfig` / `_ModelWrapper` / `get_model_format` /
   `_load_config_from_engine` / `_load_config_from_ckpt`.
4. `SamplingParams(...)` -> `__post_init__` -> `_validate` (sampling_params.py)
   -> `GuidedDecodingParams._validate`. Plus the deferred
   `_get_bad_words`/`_get_stop_words` setup guards.

## Method

- Enumerated every validator via grep for `@field_validator`,
  `@model_validator`, `def validate_`, `def _validate`, `__post_init__`, and
  every `raise`/`assert`/`logger.warning` reachable from those defs, then READ
  each in context (opened the file at the cited line and confirmed predicate +
  outcome + message + replayability).
- Enumerated every `Literal[...]` field and every `Field(...)` numeric
  constraint. As at 1.2.1, NO `ge/gt/le/lt` Field constraints are imposed
  directly on the headline numeric knobs at 0.21.0; the bounds live in
  validator functions (all captured).
- Folded the 0.21.0 PoC ground truth
  (`research/mining-substrate-trial/findings/ground_truth/tensorrt/v0_21_0/invariants_ground_truth.yaml`,
  75 entries): re-resolved every citation against the live 0.21.0 source read
  here. The PoC GT used a different checkout path
  (`/tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt_llm`) but the SAME version, so
  its line numbers are very close; I re-cited each to the validator/Field/def
  DEFINITION line (stable qualname anchor) and noted the inner raise line in
  `notes`.

## Net-new vs PoC GT (3 entries)

1. `tensorrt_BaseLLM_rejects_unknown_kwarg` - the generic unknown-kwarg gate in
   `BaseLLM.__init__` (llm.py:139). First gate in the entry-point call graph;
   the PoC GT started at the args model and never saw it. NOTE this is the
   0.21.0 analogue of 1.2.1's TRT-specific-reject, but the 0.21.0 message is the
   GENERIC "got invalid argument" (no TRT-vs-Torch kwarg partitioning exists at
   0.21.0).
2. `tensorrt_baseLlmArgs_extra_forbid_unknown_field` - `BaseLlmArgs.model_config`
   sets `extra='forbid'` (llm_args.py:767), so the pydantic constructor itself
   rejects unknown fields independent of the llm.py pre-check.
3. `tensorrt_trtLlmArgs_validate_auto_parallel_world_size_conflict` - the
   reachable args-model entry (`TrtLlmArgs.validate_auto_parallel`,
   llm_args.py:1562) into the `_ParallelConfig.world_size` guard. The PoC GT
   recorded the predicate only at the internal `_ParallelConfig` level.

## Corrections / re-statements vs the PoC GT (re-derived from 0.21.0 source)

- `tensorrt_pluginConfig_*` - the PoC GT cited `PluginConfigMeta._make_plugin_property`.
  In 0.21.0 the property factory is a MODULE-LEVEL function `_make_plugin_property`
  (plugin.py:95); the asserts live in its inner `bind().prop` setter (lines 111,
  117, 120). Re-cited to the actual 0.21.0 qualname.
- NGram asserts - the PoC GT split `prompt_lookup_num_tokens > 0` and
  `max_matching_ngram_size > 0` into two entries. In 0.21.0 source these are a
  SINGLE `assert` statement (llm_args.py:1323), recorded here as one invariant.
- `validate_dtype` - re-cited to the def line (llm_args.py:1050; raise at 1057).
  This validator is the load-bearing CPU-replayability ceiling (see below).
- `get_model_format`, `BuildCache`, `LoraConfig`, `modeling_utils` quant asserts -
  all re-cited to def lines; PoC line numbers were off by 0-1 (def-vs-raise).

No PoC entry was found to be outright WRONG; every predicate re-derived matched.

## Runtime replayability notes for the downstream gate

- CPU-REPLAYABLE (13 entries with `kwargs_positive`/`kwargs_negative` or
  `replay_via`): the standalone nested configs that do NOT touch CUDA at
  construction - `LookaheadDecodingConfig`, `CalibConfig`, `SchedulerConfig`
  (capacity/context-chunking enums), `BatchingType`, `LoraConfig`,
  `SamplingParams` (best_of/n, truncate_prompt_tokens, greedy-block),
  `GuidedDecodingParams` (via `._validate()`), and `DecodingBaseConfig.from_dict`.
- DORMANT (63 entries with `dormant_reason`): three buckets -
  1. **Args-model-bound** - EVERY `BaseLlmArgs` subclass ctor
     (`TorchLlmArgs`/`TrtLlmArgs`/`_AutoDeployLlmArgs`) runs the `validate_dtype`
     field_validator (llm_args.py:1050) which calls
     `torch.cuda.get_device_properties(0)` UNCONDITIONALLY. On a CPU-only host
     this raises before any other validator, so all args-model-level invariants
     (literals, range checks, lora/build-config warns, speculative asserts,
     ExecutorConfig consistency) are dormant for pure kwargs replay. They are
     live only in a real GPU container.
  2. **Host/dir-dependent** - SM-gated dtype/plugin checks (need a specific CUDA
     SM), engine/ckpt parallel-size mismatches and `get_model_format` (need a
     real model/engine/ckpt directory).
  3. **Method-call-time, not construction-time** - `QuantConfig` quant-algo
     allowlists fire inside `_get_modelopt_*` (QuantConfig is a plain dataclass
     with no construction validator); `SamplingParams._get_bad_words`/
     `_get_stop_words` fire post-construction without a prior tokenizer `_setup`;
     `BuildCache.__init__`'s max_records check is on `BuildCache`, not
     `BuildCacheConfig`.
- `GuidedDecodingParams` nuance: it is a `@dataclass(slots=True, kw_only=True)`
  with NO `__post_init__` at 0.21.0, so `_validate` is NOT auto-called by the
  constructor; the gate must call `._validate()` after construction (or go via
  `SamplingParams(guided_decoding=...)`). Flagged `replay_via`.
- `best_of_gt_1_greedy` is env-sensitive; replay requires
  `TLLM_ALLOW_N_GREEDY_DECODING` unset.

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. **Validators on base/sibling classes never reached from the public ctor.**
   I only walk what `LLM(...)` / `SamplingParams(...)` construction actually
   touches. Abstract bases (`DecodingBaseConfig`, `PybindMirror`) and subclasses
   I confirmed the dispatch for but did not route to per-field (e.g.
   `MTPDecodingConfig` has no construction validator at 0.21.0, but a hierarchy
   walk would confirm that systematically).
2. **Inherited pybind / C++ mirror constraints.** Many configs subclass
   `PybindMirror` (KvCacheConfig, SchedulerConfig, PeftCacheConfig,
   LookaheadDecodingConfig, ...). Constraints enforced on the C++ side or in the
   mirrored pybind `__init__` (`_KvCacheConfig`, `_SchedulerConfig`, etc.) are
   invisible to a Python call-graph walk and out of source-scope here. An
   MRO-enumerating hierarchy walk would surface the mirror contract.
3. **`BuildConfig` + nested `PluginConfig` deep tree.** I followed the
   PluginConfig validators the PoC GT named (the property-setter asserts and
   `validate()`), but did not exhaustively walk every `BuildConfig` subfield,
   since reaching them requires the CUDA-bound args model first.
4. **`@field_validator` defined on a parent and overridden in a child.**
   `load_format` is the live example - `BaseLlmArgs` declares it as a
   `Literal['auto','dummy']` while `TorchLlmArgs` overrides it with
   `Union[str, LoadFormat]` + a before-validator. My grep-by-def approach caught
   both, but the MRO-ordered hierarchy walk is the right tool for override
   resolution in general.
5. **Enum classes with members not surfaced as a field Literal.** I caught
   field-level Literals/StrEnums (QuantAlgo, BatchingType, the scheduler
   policies); standalone enums only reachable via `from_dict(string)` paths on
   classes I did not route to may be missed.
