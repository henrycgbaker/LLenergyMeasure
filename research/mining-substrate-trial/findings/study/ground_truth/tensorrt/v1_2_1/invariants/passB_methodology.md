# Pass B methodology - tensorrt-llm v1.2.1 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the tensorrt-llm config surface by TYPE, not by call path.
Starting from the public config classes, every reachable pydantic model and
dataclass was walked and, for each, every validity rule extracted.

- Engine source: `/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm/`
  (`version.py` confirms `__version__ = "1.2.1"`).
- Pure source analysis, no GPU, no model download.

## Type tree enumerated

Root config file `llmapi/llm_args.py` (3416 lines) defines the bulk of the
surface. Every class was located via `grep -n "^class "` and read in full:

- Base: `StrictBaseModel` (extra="forbid").
- LlmArgs tree: `BaseLlmArgs` -> `TrtLlmArgs`, `TorchLlmArgs`.
- Nested sub-models (all `StrictBaseModel`): `CudaGraphConfig`,
  `GuidedDecodingConfig`, `BaseSparseAttentionConfig` (+ `Rocket` / `DeepSeek` /
  `SkipSoftmax` subclasses), `MoeLoadBalancerConfig`, `MoeConfig`,
  `Nvfp4GemmConfig`, `AttentionDpConfig`, `_ParallelConfig`, `CalibConfig`,
  `DecodingBaseConfig` (+ Medusa/Eagle/SaveHiddenStates/UserProvided/NGram/
  DraftTarget/MTP/Auto/Lookahead subclasses), `KvCacheConnectorConfig`,
  `RayPlacementConfig`, `DynamicBatchConfig`, `SchedulerConfig`,
  `PeftCacheConfig`, `KvCacheConfig`, `ExtendedRuntimePerfKnobConfig`,
  `CacheTransceiverConfig`, `TorchCompileConfig`.
- Enums: `LoadFormat`, `SamplerType`, `BatchingType`,
  `CapacitySchedulerPolicy`, `ContextChunkingPolicy`,
  `GuidedDecodingConfig.GuidedDecodingBackend`.
- Out-of-file types walked: `PluginConfig` (`plugin/plugin.py`), `BuildConfig`
  (`builder.py`), `QuantConfig` + `QuantAlgo` + `KV_CACHE_QUANT_ALGO_LIST`
  (`models/modeling_utils.py`, `quantization/mode.py`), `LoraConfig`
  (`lora_helper.py`), `SamplingParams` + `GuidedDecodingParams`
  (`sampling_params.py`).

## Extraction per class

For each class the following were harvested:

1. `Literal[...]` / `StrEnum` / `Enum`-typed fields -> membership rules
   (`predicate_kind: literal_in` / `strenum_in`).
2. Inline `Field(gt/ge/le/lt/multiple_of/...)` numeric constraints.
   (`grep -nE "gt=|ge=|lt=|le=|multiple_of="`).
3. `@field_validator` / `@model_validator` bodies -> every `raise` / `logger.warning`.
4. `__init__` / `__post_init__` / `model_post_init` checks.
5. `from_dict` dispatch tables -> `decode_dispatch`.
6. `supports_backend()` per-subclass overrides -> `backend_dispatch`.

`outcome` derived from severity: error/raise -> `invalid`; warn -> `valid_with_warning`;
catalogue self-check -> `meta`.

## kwargs replayability discipline

`kwargs_positive` (should trigger) / `kwargs_negative` (should not) were emitted
ONLY where the predicate is reachable at plain construction on a CPU host:
single-field declarative constraints and standalone-model field validators. For
rules that are GPU-gated (`validate_dtype` SM check, `PluginConfig.validate`
SM-100), env-gated (greedy `best_of`), filesystem-stateful (`get_model_format`,
moe load-balancer file), or fire only at a later lifecycle method
(`MoeLoadBalancerConfig.setup`, `SamplingParams._get_*_words`), NO kwargs were
fabricated - a wrong pair would make the gate reject an otherwise-valid entry.
All config classes carrying replay pairs are importable from
`tensorrt_llm.llmapi` (verified in `llmapi/__init__.py`).

## Result

- 100 total candidates.
- 92 folded from the PoC ground truth (all reproduced; every cited line
  re-verified against source).
- 1 of those folded with a correction (`guidedDecodingParams_at_most_one_guide`:
  PoC cited line 32 = class def; the raise is at line 37, message includes
  `but got {num_guides}`).
- 8 net-new, all declarative type-level constraints an entry-point/call-graph
  walk structurally misses (no explicit `raise` on any path):
  - `torchLlmArgs_allreduce_strategy_literal` (9-member beta Literal).
  - `cacheTransceiverConfig_kv_transfer_timeout_ms_gt_0` and
    `..._sender_future_timeout_ms_gt_0` (the only inline `Field(gt=0)` in the
    llm_args.py tree).
  - five `PluginConfig` Literal dtype fields:
    `gemm_swiglu_plugin`, `low_latency_gemm_plugin`,
    `low_latency_gemm_swiglu_plugin`, `gemm_allreduce_plugin`, and
    `bert_attention_plugin` (canonical representative of the whole
    `DefaultPluginDtype`-typed plugin-field family).

## PoC entries flagged

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (PoC, cited line 3227):
  in the v1.2.1 source the `warn_on_unstable_feature_usage` method (def at line
  3215) is NOT decorated with `@model_validator` - the decorated validator stack
  ends at `validate_helix_tokens_per_block` (line 3197). As written it is a plain
  method and does not fire automatically at construction. Flagged
  `pass_b_flag: possibly_invalid_not_auto_invoked`. (It may still be invoked
  elsewhere via the entry-point path - deferred to pass A to confirm.)
- `tensorrt_torchLlmArgs_validate_batch_wait_timeout_ms_non_negative`: message
  text says "greater than 0" but the predicate is `< 0` (0 is accepted). Kept;
  kwargs reflect the actual `< 0` predicate (positive=-1, negative=0).

No PoC entry was found to be outright wrong in its predicate; the catalogue is
solid. The deltas are a citation-line fix and one decorator-status flag.

## Blind spots (what an entry-point / call-graph walk should catch that B missed)

A type-tree walk under-covers rules that live in the execution path rather than
in a config class:

1. `BuildConfig` carries NO field_validators and NO construction-time numeric
   constraints; all its checks are `assert`s inside `build()` / weight-load
   methods (`builder.py` lines 833, 890, 1234-1235: encoder len equality, EAGLE
   max_batch_size <= 512, max_draft_len <= 256, SmoothQuant SM>=100 ban). These
   are call-graph invariants, invisible to a pure type walk.
2. Cross-config rules wired in `LLM.__init__` / `_build_model` /
   `update_llm_args_with_extra_dict` (e.g. build_config field clobbering at
   llm_args.py:3355) that no single config class owns.
3. C++/pybind-side validation in `bindings.executor` that the Python `_to_pybind`
   round-trip surfaces only at runtime.
4. Speculative-decoding side effects that fire during model engine construction
   (PyTorchModelEngine mutating `MTPDecodingConfig`), not at config build.
5. Any validity rule expressed as a runtime guard in executor / worker code
   reachable only by following the call graph from the public `LLM(...)` entry.
