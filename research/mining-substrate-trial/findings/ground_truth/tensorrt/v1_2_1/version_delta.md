# Version delta: tensorrt-llm 0.21.0 -> 1.2.1 ground truth

PRIMARY BUMP-PAIR DELIVERABLE. Compares this v1.2.1 source-walked ground truth
against the v0.21.0 ground truth at
`research/mining-substrate-trial/findings/ground_truth/tensorrt/v0_21_0/`.

This is a MAJOR-VERSION jump (0.x -> 1.x). The structural churn is large; this
file enumerates it so a substrate bake-off can measure recall across the bump.

## Headline counts

| Surface                      | v0.21.0 GT | v1.2.1 GT | Delta |
|------------------------------|-----------:|----------:|------:|
| BaseLlmArgs fields           |         51 |        51 |     0 |
| TrtLlmArgs-only fields       |         10 |        13 |    +3 |
| TorchLlmArgs-only fields     |         21 |        34 |   +13 |
| SamplingParams fields        |         47 |        48 |    +1 |
| GuidedDecodingParams fields  |          5 |         5 |     0 |
| additional_model_output_params |        2 |         0 |    -2 |
| Subconfig fields (expanded)  |        177 |       232 |   +55 |
| Subconfig classes            |         18 |        35 |   +17 |
| engine_envs (TLLM_*/TRTLLM_*)|         44 |        55 |   +11 |
| TOTAL schema entries         |        357 |       438 |   +81 |
| Invariants                   |         75 |        92 |   +17 |

BaseLlmArgs count is coincidentally flat (51 -> 51) but the membership shifted:
five fields moved OUT to subclasses (enable_prompt_adapter, max_prompt_adapter_token,
batching_type, normalize_log_probs -> TrtLlmArgs; garbage_collection_gen0_threshold
-> TorchLlmArgs) while ~nine NEW fields moved in (custom_tokenizer, enable_lm_head_tp_in_adp,
pp_partition, fail_fast_on_attention_window_too_large, sparse_attention_config,
otlp_traces_endpoint, orchestrator_type, env_overrides, return_perf_metrics). The
v0.21 deprecated LoRA scalars (max_lora_rank/max_loras/max_cpu_loras) and the
decoding-config knobs were dropped.

## Structural changes (the headline of a major bump)

### 1. LlmArgs alias flipped: TrtLlmArgs -> TorchLlmArgs

v0.21: `LlmArgs = TrtLlmArgs` (llm_args.py:1594). v1.2.1: `LlmArgs = TorchLlmArgs`
(llmapi/llm_args.py:3410). The DEFAULT backend a bare `LLM(model=...)` caller gets
changed from AOT TRT-engine-build to eager PyTorch. This is the single most
behaviourally-significant delta for a benchmark caller: same construction call,
different engine.

### 2. Inheritance flattened

v0.21: BaseLlmArgs(shared) then TrtLlmArgs / TorchLlmArgs, with LlmArgs aliasing.
v1.2.1: BaseLlmArgs is the explicit base; TrtLlmArgs(BaseLlmArgs) and
TorchLlmArgs(BaseLlmArgs) both subclass it directly.

### 3. Pydantic migration of three dataclass/metaclass configs

| Class        | v0.21 kind                       | v1.2.1 kind          |
|--------------|----------------------------------|----------------------|
| BuildConfig  | stdlib @dataclass                | pydantic BaseModel   |
| PluginConfig | @dataclass(slots) + PluginConfigMeta metaclass | pydantic BaseModel |
| LoraConfig   | @dataclass(DictConversion)       | pydantic BaseModel (moved lora_manager.py -> lora_helper.py) |

PluginConfig is the big one. v0.21's `PluginConfigMeta` synthesised public
properties from `_`-prefixed storage fields - invisible to class-body AST mining.
v1.2.1 declares plain pydantic fields, so the surface is now ordinarily mineable.
The v0.21 GT flagged PluginConfig as the hardest substrate target; v1.2.1 makes
it tractable.

### 4. Field nesting on TorchLlmArgs

- v0.21 flat `use_cuda_graph` / `cuda_graph_batch_sizes` / `cuda_graph_max_batch_size`
  / `cuda_graph_padding_enabled` -> v1.2.1 nested `CudaGraphConfig`
  (batch_sizes / max_batch_size / enable_padding).
- v0.21 flat `moe_max_num_tokens` / `moe_load_balancer` / `moe_backend` ->
  v1.2.1 nested `MoeConfig` (backend / max_num_tokens / load_balancer /
  disable_finalize_fusion / use_low_precision_moe_combine).
- v0.21 `mixed_sampler` + `enable_trtllm_sampler` booleans -> v1.2.1 single
  `sampler_type` enum (SamplerType: TRTLLMSampler / TorchSampler / auto).

## New subconfig classes (no v0.21 analogue)

CudaGraphConfig, GuidedDecodingConfig, MoeConfig, MoeLoadBalancerConfig
(promoted from torch-internal), Nvfp4GemmConfig, AttentionDpConfig,
RayPlacementConfig, KvCacheConnectorConfig, and the sparse-attention tree
(BaseSparseAttentionConfig + RocketSparseAttentionConfig +
DeepSeekSparseAttentionConfig + SkipSoftmaxAttentionConfig). 17 net new
subconfig classes.

## Speculative-decoding tree: 6 -> 9 classes

| Class                          | v0.21 | v1.2.1 | Note |
|--------------------------------|:-----:|:------:|------|
| MedusaDecodingConfig           |  yes  |  yes   | unchanged fields |
| EagleDecodingConfig            |  yes  |  yes   | 8 -> 11 fields; Eagle3 in-place (no Eagle3Config class) |
| LookaheadDecodingConfig        |  yes  |  yes   | Pybind-mirrored; unchanged |
| NGramDecodingConfig            |  yes  |  yes   | prompt_lookup_num_tokens REMOVED; max_matching_ngram_size default 4->0 |
| DraftTargetDecodingConfig      |  yes  |  yes   | pytorch_weights_path REMOVED (use speculative_model) |
| MTPDecodingConfig              |  yes  |  yes   | 4 -> 8 fields (use_mtp_vanilla, mtp_eagle_one_model, thinking tokens) |
| SaveHiddenStatesDecodingConfig |   -   |  NEW   | forces bs=1, no overlap, no cuda graph |
| UserProvidedDecodingConfig     |   -   |  NEW   | caller supplies Drafter |
| AutoDecodingConfig             |   -   |  NEW   | heuristic draft-model-free selection |

DecodingBaseConfig grew 2 -> 11 fields (max_total_draft_tokens, draft_len_schedule,
max_concurrency, acceptance_window, acceptance_length_threshold, load_format,
allow_advanced_sampling). Dispatch moved from parent isinstance-replacement to
per-subclass `supports_backend()`.

## Subconfig field-level deltas

- **KvCacheConfig**: 13 -> 18. NEW: attention_dp_events_gather_period_ms, use_uvm,
  max_gpu_total_bytes, dtype (pytorch-only), mamba_ssm_cache_dtype, tokens_per_block.
  free_gpu_memory_fraction default null -> 0.9. NEW Python validators for
  free_gpu_memory_fraction (0..1), max_gpu_total_bytes (>=0), max_attention_window
  (non-empty positive-int list).
- **CacheTransceiverConfig**: 1 -> 4. NEW backend enum (DEFAULT/UCX/NIXL/MOONCAKE/MPI),
  kv_transfer_timeout_ms, kv_transfer_sender_future_timeout_ms. v0.21 max_num_tokens
  renamed max_tokens_in_buffer. The backend enum SUPERSEDES the v0.21
  TRTLLM_USE_{UCX,NIXL,MPI}_KVCACHE env-var selection.
- **TorchCompileConfig**: 4 -> 6. NEW capture_num_tokens, max_num_streams.
- **QuantConfig**: 9 -> 10 (NEW mamba_ssm_cache_dtype). QuantAlgo enum 20 -> 26
  members (NEW W4A8_NVFP4_FP8, W4A8_MXFP4_MXFP8, W4A16_MXFP4, NVFP4_AWQ;
  W4A8_QSERVE_PER_GROUP/CHANNEL retained).
- **BuildConfig**: 28 -> 27. auto_parallel_config REMOVED (auto-parallel deprecated).
- **LoraConfig**: 7 -> 8 (NEW swap_gate_up_proj_lora_b_weight). max_loras/max_cpu_loras
  default 4 -> None.
- **PluginConfig**: 43 -> 43 user-facing (explicitly_disable_gemm_plugin demoted to
  PrivateAttr). gemm_plugin Literal gained 'nvfp4'. SM-killswitch retained (5 plugins
  on SM 100 Blackwell).
- **PeftCacheConfig / SchedulerConfig / DynamicBatchConfig / ExtendedRuntimePerfKnobConfig /
  CalibConfig**: field sets unchanged; some defaults made explicit.

## SamplingParams deltas

- NEW field: `prompt_ignore_length` (sampling_params.py:124).
- `additional_model_outputs` type changed List[AdditionalModelOutput] -> List[str];
  the `AdditionalModelOutput` class was REMOVED (the v0.21
  additional_model_output_params namespace is dropped, -2 entries).
- NEW Python range validators in `_validate`: top_p in [0,1], top_k >= 0,
  temperature >= 0 (these were C++-only at v0.21). This is a high-value mining
  signal: three invariants that were invisible to a v0.21 Python source-walk are
  now Python-visible.

## Env-var churn (44 -> 55, source-walk visible)

Persisting (sample): TLLM_ALLOW_N_GREEDY_DECODING, TLLM_LLMAPI_BUILD_CACHE,
TLLM_OVERRIDE_LAYER_NUM (file moved model_engine.py -> model_loader.py),
TLLM_LLM_ENABLE_DEBUG, TLLM_LLM_ENABLE_TRACER, TLLM_NVTX_DEBUG,
TLLM_SPAWN_PROXY_PROCESS*, TRTLLM_ENABLE_PDL (file moved custom_ops ->
flashinfer_utils), TRTLLM_FORCE_MNNVL_AR, TRTLLM_ENABLE_ATTENTION_NVFP4_OUTPUT,
TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED, TRTLLM_CAN_USE_DEEP_EP,
TRTLLM_MOE_POST_QUANT_ALLTOALLV, TRTLLM_DISABLE_KV_CACHE_TRANSFER_OVERLAP,
TRTLLM_DISABLE_UNIFIED_CONVERTER, TRTLLM_DISAGG_BENCHMARK_GEN_ONLY.

Newly source-walk-visible at v1.2.1 (sample, all flagged NEW in schema):
TLLM_AUTOTUNER_CACHE_PATH, TLLM_DISABLE_MPI, TLLM_INCREMENTAL_DETOKENIZATION_BACKEND,
TLLM_LLMAPI_ZMQ_DEBUG/PAIR, TLLM_NUMA_AWARE_WORKER_AFFINITY, TLLM_RAY_FORCE_LOCAL_CLUSTER,
TLLM_VIDEO_PRUNING_RATIO, TRTLLM_LOAD_KV_SCALES, TRTLLM_FORCE_ALLTOALL_METHOD,
TRTLLM_FORCE_COMM_METHOD, TRTLLM_MOE_A2A_WORKSPACE_MB, TRTLLM_WINDOW_SIZE_SHARES,
plus a cluster of GC / stack-dump / Ray / server controls
(TRTLLM_{WORKER,SERVER,DISAGG_SERVER}_DISABLE_GC, TRTLLM_*_PRINT_STACKS_PERIOD,
TRTLLM_RAY_*).

Dropped vs v0.21 source-walk (env var read context no longer matches the
`os.environ`/`os.getenv` grep, or feature removed): TLLM_DISABLE_FP4_ALLGATHER,
TRTLLM_USE_NIXL_KVCACHE (folded into CacheTransceiverConfig.backend),
TRTLLM_MOE_DISABLE_ALLTOALLV, and the TRTLLM_DG_* DeepGEMM JIT family
(TRTLLM_DG_CACHE_DIR / _JIT_DEBUG / _JIT_DUMP_CUBIN / _JIT_USE_NVCC /
_NVCC_COMPILER) - these were v0.21 low-confidence "grep hit, behaviour not
extracted" entries; at v1.2.1 the deep_gemm code no longer reads them through a
plain `os.environ`/`os.getenv` call that the grep catches (the deep_gemm JIT
config moved). Treat the DG family as a known low-confidence delta, not a
confirmed removal of capability.

## Mining-difficulty deltas (for the bake-off)

- **PluginConfig got EASIER.** The metaclass is gone; fields are now plain
  pydantic. The single hardest v0.21 substrate target is now ordinary.
- **More surface is Python-validated.** SamplingParams top_p/top_k/temperature,
  KvCacheConfig free_gpu_memory_fraction/max_gpu_total_bytes, and the
  CacheTransceiverConfig backend enum moved validation from C++ into Python -
  all newly recoverable by a Python source-walk substrate.
- **Speculative semantics got SIMPLER to model.** No more instance-replacement;
  each config self-declares supports_backend(). A substrate emitting
  positive/negative kwargs no longer needs to model the v0.21 "replace with a
  different class" trap.
- **More nesting.** CudaGraphConfig / MoeConfig nesting means a substrate that
  only walks top-level TorchLlmArgs fields will miss the nested knobs; recursion
  into pydantic sub-models is required.

## Open / low-confidence at this pin

- LookaheadDecodingConfig defaults (max_window_size / max_ngram_size /
  max_verification_set_size) still resolve from C++
  `_LookaheadDecodingConfig.get_default_*()` at class-load; recorded as
  `default_resolved_from_cpp`. Unchanged from v0.21.
- The TRTLLM_DG_* DeepGEMM JIT env family (see above) - low confidence on
  removal vs relocation.
- C++ pybind boundary excluded by explicit scope call (see methodology.md).
