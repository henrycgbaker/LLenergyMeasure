# Pass A - entry-point / call-graph walk methodology (tensorrt-llm 1.2.1)

Engine source: `/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm/` (confirmed
`__version__ = "1.2.1"` via `version.py`).

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of
a two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal: maximise
recall of construction-time validation invariants reachable from public,
user-facing entry points.

## Traversal (what I walked)

Starting roots (public surface a benchmark harness actually constructs):

1. `tensorrt_llm.LLM(...)` -> `_TorchLLM.__init__` ->
   `_validate_args_for_torch_backend(kwargs)` (llm.py:1167) ->
   `BaseLLM.__init__` (llm.py:122) -> `llm_args_cls.from_kwargs(...)`
   (llm.py:175). The default public `LLM` is PyTorch-backed (`_TorchLLM`);
   the TRT path is `tensorrt_llm._tensorrt_engine.LLM` -> `_TrtLLM` ->
   `TrtLlmArgs`.
2. `from_kwargs` -> pydantic model construction of
   `TorchLlmArgs` / `TrtLlmArgs` (both subclass `BaseLlmArgs`), which fires the
   full validator chain: every `@field_validator`, `@model_validator(mode=
   "after")`, `Field(...)` Literal/constraint, plus `_check_consistency`.
3. Nested config fields reached transitively from the args models:
   `KvCacheConfig`, `CudaGraphConfig`, `TorchCompileConfig`, `MoeConfig` +
   `MoeLoadBalancerConfig`, `SchedulerConfig` (Capacity/ContextChunking),
   `CacheTransceiverConfig`, `CalibConfig`, `QuantConfig`
   (-> `quantization/mode.py`), `LoraConfig` (-> `lora_helper.py`),
   `PluginConfig` (-> `plugin/plugin.py`), `Nvfp4GemmConfig`,
   `RayPlacementConfig`, `AttentionDpConfig`, the `*DecodingConfig`
   speculative family (-> `DecodingBaseConfig.from_dict` dispatch +
   per-subclass `validate_*` + `supports_backend`), and the
   `BaseSparseAttentionConfig.from_dict` dispatch.
4. `SamplingParams(...)` -> `__post_init__` -> `_validate` (sampling_params.py)
   -> `GuidedDecodingParams._validate`. Also the deferred
   `_get_bad_words`/`_get_stop_words` setup guards.
5. TRT-only deferred config loaders reached from `TrtLlmArgs` build:
   `_load_config_from_engine`, `_load_config_from_ckpt`, `get_model_format`.

## Method

- Enumerated every validator via grep for `@field_validator`,
  `@model_validator`, `def validate_`, `def _validate`, `__post_init__`,
  and every `raise`/`assert`/`logger.warning` reachable from the above defs
  (193 raise/assert/validator hits in `llm_args.py` alone), then read each in
  context to classify predicate + outcome + replayability.
- Enumerated every `Literal[...]` field (grep) and cross-checked each against
  the PoC GT catalogue; enumerated `Field(...)` numeric constraints
  (`ge/gt/le/lt`) - none are imposed directly on the headline numeric knobs
  at v1.2.1 (the bounds live in validator functions, all captured).
- Folded the PoC ground truth: re-resolved every citation against the live
  v1.2.1 source. For folded entries I cite the validator/Field DEFINITION line
  (stable qualname anchor); the PoC GT sometimes cited the inner raise-site
  line. Both resolve to the same qualname; line drift is cosmetic.

## Coverage

- All TorchLlmArgs + BaseLlmArgs + TrtLlmArgs validators (error / warn /
  normalisation).
- All nested-config Literals/enums and range/allowlist validators reachable
  from construction.
- Full SamplingParams + GuidedDecodingParams predicate set.
- Speculative-decoding backend dispatch + per-subclass asserts, decoding-type
  registry dispatch, sparse-attention dispatch.

## Net-new vs PoC GT (7 entries)

1. `LLM` (PyTorch) rejects TRT-only kwargs - `_validate_args_for_torch_backend`
   (llm.py:1182). First gate in the entry-point call graph; PoC GT started at
   LlmArgs and never saw it.
2. `TorchLlmArgs.allreduce_strategy` Literal (llm_args.py:2905). PoC GT
   catalogued every other llm_args Literal but missed this one.
3. `validate_and_init_tokenizer` custom_tokenizer-vs-tokenizer-object conflict
   (llm_args.py:2265).
4. `BaseSparseAttentionConfig.from_dict` missing-`algorithm` raise
   (llm_args.py:212) - distinct message from the unrecognised-value dispatch.
5. `validate_checkpoint_format` dual-spec warn + normalisation
   (llm_args.py:3093).
6. `sync_quant_config_with_kv_cache_config_dtype` warn on unknown dtype
   (llm_args.py:3191).
7. `validate_misc` default-fill normalisation (llm_args.py:3031) - low value,
   included for catalogue completeness; downstream gate may drop it.

## PoC GT entries I believe are mis-stated (not invalid, but worth a flag)

- `tensorrt_baseLlmArgs_validate_dtype_bfloat16_sm_lt_80`: PoC cites line 2204;
  the validator def is `validate_dtype` at line 2199 (raise is inside). qualname
  correct, line off by a few. Re-cited to the def line in Pass A.
- `guidedDecodingParams_at_most_one_guide`: PoC message_template truncates the
  real message ("...but got {num_guides}."). Corrected in Pass A.
- Several PoC citations point at the inner raise/assert line rather than the
  validator def; harmless but inconsistent. Pass A normalises to def lines.

No PoC entry was found to be outright WRONG (every predicate I re-derived
matched). All folded with provenance: fold.

## Runtime replayability notes for the downstream gate

- `kwargs_positive`/`kwargs_negative` for `*DecodingConfig`, `KvCacheConfig`,
  `CudaGraphConfig`, `TorchCompileConfig`, `MoeConfig`, `LookaheadDecodingConfig`,
  `Nvfp4GemmConfig`, `RayPlacementConfig`, `CalibConfig`, `LoraConfig`,
  `SamplingParams`, `GuidedDecodingParams` are constructor-replayable on a
  CPU-only host (pure pydantic / dataclass validation, no CUDA, no model dir).
- `BaseSparseAttentionConfig` and `DecodingBaseConfig` dispatch entries replay
  via the `from_dict` classmethod (`replay_via` annotated), not the bare
  constructor.
- Entries marked `dormant_reason` need a GPU/SM/engine-dir/ckpt-dir/model-dir
  and CANNOT be replayed in this source-only environment: SM-gated dtype/plugin
  checks, engine/ckpt parallel-size mismatches, `get_model_format` config.json,
  the custom_tokenizer object conflict, and the SamplingParams
  `_get_bad_words`/`_get_stop_words` setup guards (fire post-construction).
- `best_of_gt_1_greedy` and the unstable-feature warning are env/field-status
  sensitive; replay requires `TLLM_ALLOW_N_GREEDY_DECODING` unset.

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. **Validators on base/sibling classes never reached from the public ctor.**
   I only walk what `LLM(...)` / `SamplingParams(...)` construction actually
   touches. Abstract bases (`DecodingBaseConfig`, `BaseSparseAttentionConfig`,
   `Pybind*` mirrors) may carry validators that only fire for subclasses I did
   not route to (e.g. `MedusaDecodingConfig`, `UserProvidedDecodingConfig`
   internals - I confirmed the dispatch but not each subclass's own
   `@model_validator`).
2. **Inherited pybind / C++ mirror constraints.** Many configs subclass
   `PybindMirror`; constraints enforced on the C++ side (or in a mirrored
   pybind `__init__`) are invisible to a Python call-graph walk and out of
   source-scope here. A hierarchy walk that enumerates the MRO would surface
   the mirror contract.
3. **`build_config` / `BuildConfig` + `PluginConfig` deep tree.** I followed
   PluginConfig validators the PoC GT named but did not exhaustively walk every
   BuildConfig nested validator, since the default PyTorch LLM does not build a
   TRT engine. A hierarchy walk over `BuildConfig` subfields would be more
   complete on the TRT side.
4. **`@field_validator` defined on a parent and overridden in a child.** My
   grep-by-def approach can miss a validator that a subclass re-declares with a
   different predicate; the MRO-ordered hierarchy walk is the right tool for
   override resolution.
5. **Enum classes with members not surfaced as a field Literal.** I caught
   field-level Literals/StrEnums; standalone enums only reachable via
   `from_dict(string)` paths on classes I did not route to may be missed.
