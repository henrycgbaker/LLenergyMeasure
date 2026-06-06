# Pass A - entry-point / call-graph walk methodology (tensorrt-llm 1.0.0)

Engine source: `/tmp/trt-llm-1.0.0/tensorrt_llm/` (confirmed `__version__ = "1.0.0"`
via `version.py`).

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of
a two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal: maximise
recall of construction-time validation invariants reachable from public,
user-facing entry points.

There is NO PoC ground truth for 1.0.0
(`research/mining-substrate-trial/findings/ground_truth/tensorrt/v1_0_0/` does
not exist), so EVERY entry is `provenance: net_new`, derived directly from the
1.0.0 source. Nothing was copied from the 1.2.1 template; line numbers, message
strings, enum-member lists and several APIs differ and were all re-derived.

## Traversal (what I walked)

Starting roots (public surface a benchmark harness actually constructs):

1. `tensorrt_llm.LLM(...)` -> `LLM` is `_TorchLLM` (llm.py:1111) ->
   `_TorchLLM.__init__` (llm.py:923) pops `backend` (defaults to "pytorch") and
   calls `_validate_args_for_torch_backend(kwargs)` (llm.py:1090) ->
   `BaseLLM.__init__` (llm.py:109) which (a) selects `llm_args_cls`
   (TorchLlmArgs / AutoDeployLlmArgs / TrtLlmArgs by backend), (b) rejects
   unknown kwargs (llm.py:146), (c) calls `llm_args_cls.from_kwargs(...)`
   (llm.py:152). The TRT path is `tensorrt_llm._tensorrt_engine.LLM` ->
   `_TrtLLM` -> `TrtLlmArgs`.
2. `from_kwargs` (llm_args.py:1329) -> `_check_consistency` ->
   pydantic construction of `TorchLlmArgs` / `TrtLlmArgs` (both subclass
   `BaseLlmArgs`), firing the validator chain: every `@field_validator`,
   `@model_validator(mode="after")`, `Field(...)` Literal, plus the speculative
   dispatch and the deferred engine/ckpt loaders.
3. Nested config fields reached transitively: `KvCacheConfig`, `CudaGraphConfig`,
   `TorchCompileConfig`, `MoeConfig`, `AttentionDpConfig`, `SchedulerConfig`
   (Capacity / ContextChunking), `CacheTransceiverConfig`, `CalibConfig`,
   `PeftCacheConfig`, `LoraConfig` (-> `lora_manager.py`), `QuantConfig`
   (-> `models/modeling_utils.py` + `quantization/mode.py`), `PluginConfig`
   (-> `plugin/plugin.py`, via TRT `BuildConfig`), `MoeLoadBalancerConfig`
   (-> `_torch/model_config.py`), and the `*DecodingConfig` speculative family
   (`DecodingBaseConfig.from_dict` dispatch + per-subclass `supports_backend` /
   `validate`).
4. `SamplingParams(...)` -> `__post_init__` -> `_validate` (sampling_params.py)
   -> `GuidedDecodingParams._validate`. Also the deferred
   `_get_bad_words` / `_get_stop_words` setup guards.
5. TRT-only deferred loaders reached from `TrtLlmArgs` build:
   `_load_config_from_engine`, `_load_config_from_ckpt`, `get_model_format`.

## Method

- Enumerated every validator via grep for `@field_validator`,
  `@model_validator`, `def validate_`, `def _validate`, `def __post_init__`,
  `def from_dict`, then read each in context to classify predicate + outcome +
  replayability. Enumerated every `Literal[...]` field and every `Field(...)`
  with `ge/gt/le/lt` (1.0.0 imposes NO numeric `Field` constraints on the
  headline knobs; all numeric bounds live inside validator functions).
- For each `raise` / `assert` / `logger.warning` reachable from a construction
  path I recorded severity (error / warning / normalisation), outcome
  (invalid / warn / normalise), the exact message string, and the def-line
  citation (stable qualname anchor).

## The CUDA-at-construction wrinkle (replay-environment note)

1.0.0 `BaseLlmArgs.validate_dtype` (llm_args.py:1373) calls
`torch.cuda.get_device_properties(0)` UNCONDITIONALLY for every construction,
and `validate_gpus_per_node` (llm_args.py:1383) calls
`torch.cuda.device_count()`. Consequently ANY `TorchLlmArgs` / `TrtLlmArgs`
construction needs a live CUDA device even to PASS. The downstream gate replays
in a real 1.0.0 container with `--gpus all`, so kwargs on the args models ARE
replayable there but are NOT replayable on a CPU-only host. Every args-model
invariant therefore carries a `dormant_reason` flagging the CUDA dependency
even though its predicate is a plain pydantic check. Standalone nested configs
(CudaGraphConfig, MoeConfig, CalibConfig, LookaheadDecodingConfig,
TorchCompileConfig, SchedulerConfig, CacheTransceiverConfig, the
`*DecodingConfig` family via `from_dict`, SamplingParams, GuidedDecodingParams,
LoraConfig) construct without CUDA and carry concrete `kwargs_positive` /
`kwargs_negative`.

## Coverage

78 invariants, all `net_new`. By severity: 66 error, 11 warning, 1 normalisation.
By replayability: 18 carry CPU-constructible `kwargs_positive`/`kwargs_negative`
on a standalone nested config (replayable anywhere); 60 are `dormant`
(CUDA-at-construction, engine/ckpt/model dir, on-disk file, SM-gated, or
fires-after-construction).

## Notable 1.0.0-vs-1.2.1 API differences (verified against 1.0.0 source)

1. **No PoC GT and a much smaller validator surface.** 1.0.0 `llm_args.py` is
   2441 lines vs 1.2.1's much larger file. Many 1.2.1 validators DO NOT EXIST in
   1.0.0: no `validate_batch_wait_timeout_ms/_iters`, no
   `validate_batch_wait_max_tokens_ratio`, no `ray_worker_extension_cls` /
   `ray_placement_config` / `orchestrator_type`, no `validate_helix_tokens_per_block`,
   no `validate_misc`, no sparse-attention family
   (`BaseSparseAttentionConfig`), no `Nvfp4GemmConfig`, no `RayPlacementConfig`,
   no `warn_on_unstable_feature_usage` model_validator (the method exists at
   TorchLlmArgs but is NOT decorated `@model_validator` in 1.0.0, so it does not
   fire at construction - omitted).
2. **`validate_build_config_with_runtime_params` RAISES in 1.0.0** for
   `max_batch_size` / `max_num_tokens` exceeding `build_config` (error), whereas
   1.2.1 warns. The seq_len / beam_width / input_len mismatches still warn.
3. **QuantConfig is a plain dataclass in 1.0.0** (`models/modeling_utils.py:128`),
   NOT pydantic. It has no `__post_init__`, so `QuantConfig(quant_algo="garbage")`
   does NOT raise at construction. The `QUANT_ALGO_LIST` / `KV_CACHE_QUANT_ALGO_LIST`
   membership asserts fire only lazily via `QuantMode.from_quant_algo`
   (`quantization/mode.py:331`). Recorded as dormant (not construction-time).
   QuantAlgo has 24 members in 1.0.0 (no `W4A8_NVFP4_FP8`, `W4A8_MXFP4_MXFP8`,
   `W4A16_MXFP4`, `NVFP4_AWQ`).
4. **LoraConfig is a plain dataclass** (`lora_manager.py:236`) and its
   `lora_ckpt_source in ['hf','nemo']` check is an `assert` in `__post_init__`
   (message `"lora_ckpt_source must be one of 'hf' or 'nemo', got {...}"`), not a
   pydantic Literal. This one IS CPU-replayable.
5. **PluginConfig is a slots dataclass with a metaclass** (`plugin/plugin.py`),
   NOT pydantic. dtype / gemm_plugin allowlist checks are `assert`s on attribute
   ASSIGNMENT (property setters), reached only on the TRT `BuildConfig` path.
   `PluginConfig.validate()` is SM-100 (Blackwell) gated. All dormant.
6. **KvCacheConfig has NO Python validators in 1.0.0.** The
   `free_gpu_memory_fraction` range, `max_gpu_total_bytes`, `max_attention_window`
   and `mamba_ssm_cache_dtype` Literal validators present in 1.2.1 are absent;
   `dtype` is a plain `str` field (no Literal). No KvCacheConfig invariants are
   emitted for 1.0.0.
7. **SamplingParams `_validate` is leaner in 1.0.0.** It has NO `top_p` range,
   `top_k >= 0`, or `temperature >= 0` checks (added later). Only `best_of >= n`,
   greedy multi-return (env-gated by `TLLM_ALLOW_N_GREEDY_DECODING`), and
   `truncate_prompt_tokens >= 1`, plus the guided-decoding delegation.
8. **Enum-member list drift.** MoeConfig.backend has 6 members (no `TRITON`).
   CacheTransceiverConfig.backend has 4 (no `MOONCAKE`). TorchLlmArgs.allreduce_strategy
   has 8 (no `NCCL_SYMMETRIC`). TorchLlmArgs LoadFormat enum has 2 (no
   `VISION_ONLY`). DecodingBaseConfig.from_dict dispatch has 8 keys (no
   `SaveHiddenStates`/`SaveState`). CapacitySchedulerPolicy / ContextChunkingPolicy
   / BatchingType match 1.2.1.
9. **EagleDecodingConfig has no field-validator in 1.0.0.** The Eagle-specific
   check is `validate()` (line 422, `"Draft model must be provided for EAGLE"`),
   which is invoked by the framework rather than at `__init__`. There is no
   `validate_eagle_config` (with the dynamic-tree / eagle_choices logic) - that
   is a later addition. Recorded with `replay_via: EagleDecodingConfig.validate`.
10. **MoeLoadBalancerConfig** (`_torch/model_config.py:26`) is a plain dataclass;
    `setup()` only asserts `num_slots is not None` - there is NO
    num_slots-divisible-by-ep_size check (added later). Dormant (fires at setup).

## Runtime replayability notes for the downstream gate

- The 18 CPU-replayable entries (CudaGraphConfig, TorchCompileConfig, MoeConfig,
  CalibConfig, SchedulerConfig enums, CacheTransceiverConfig, LoraConfig,
  LookaheadDecodingConfig field validators, SamplingParams, GuidedDecodingParams,
  DecodingBaseConfig.from_dict dispatch) construct without CUDA. `from_dict`
  and `validate()` entries carry `replay_via`; the GuidedDecodingParams entry
  replays through `SamplingParams(guided_decoding=GuidedDecodingParams(...))`.
- All 60 dormant entries record a `dormant_reason`. The dominant reason is the
  CUDA-at-construction wrinkle on the args models; the gate, running in a
  `--gpus all` 1.0.0 container, can attempt those with the kwargs supplied in
  `notes` (e.g. `stream_interval=0`, `peft_cache_config.lora_prefetch_dir='x'`).
  Genuinely un-replayable-anywhere reasons: engine/ckpt/model dir, on-disk yaml
  file, SM-100 host, lazy-not-construction (QuantConfig, MoeLoadBalancer), and
  post-construction call sites (SamplingParams `_get_bad_words`/`_get_stop_words`).

## Blind spots (what a class-hierarchy walk should catch that this pass did not)

1. Validators on base/sibling classes never reached from the public ctor (e.g.
   per-subclass speculative `validate()` methods I confirmed by dispatch but did
   not exhaustively route through, MedusaDecodingConfig internals).
2. Inherited pybind / C++ mirror constraints (`PybindMirror` subclasses):
   constraints enforced C++-side are invisible to a Python call-graph walk.
3. The `BuildConfig` deep tree on the TRT side (the default Torch LLM does not
   build a TRT engine), including nested PluginConfig fields beyond dtype/gemm.
4. `@field_validator` overridden in a child (e.g. load_format on Base vs Torch)
   - resolved here by reading both, but an MRO walk is the systematic tool.
5. Standalone enums reachable only via `from_dict(string)` paths on classes not
   routed to from the public surface.
