# Delta: tensorrt-llm 0.21.0 ground-truth vs LLEM baseline

PRIMARY DELIVERABLE. Compares the new source-walked ground truth
(`schema_ground_truth.json` + `invariants_ground_truth.yaml`) against the
existing LLEM mining outputs:

- baseline schema: `engine_versions/tensorrt/v0_21_0/outputs/schema.discovered.json`
- baseline invariants (proposed): `engine_versions/tensorrt/v0_21_0/outputs/invariants.proposed.yaml`
- baseline invariants (validated): `engine_versions/tensorrt/v0_21_0/outputs/invariants.validated.yaml`
- baseline curated llem surface: `engine_versions/tensorrt/v0_21_0/outputs/curated.yaml`

## Headline counts

| Surface                                        | Baseline | Ground truth | Delta  |
|------------------------------------------------|----------|--------------|--------|
| Schema fields (BaseLlmArgs path only)          |       60 |           51 |     -9 |
| Schema fields (TrtLlmArgs-only knobs)          |        9 |           10 |     +1 |
| Schema fields (TorchLlmArgs-only knobs)        |        0 |           21 |    +21 |
| Schema fields (sampling_params)                |       47 |           47 |      0 |
| Schema fields (guided_decoding_params)         |        0 |            5 |     +5 |
| Schema fields (additional_model_output_params) |        0 |            2 |     +2 |
| Schema fields (subconfigs expanded)            |        0 |          177 |   +177 |
| Schema fields (engine_envs / TLLM_*)           |        0 |           44 |    +44 |
| TOTAL schema entries                           |      107 |          357 |   +250 |
| Invariants                                     |       35 |           75 |    +40 |

The `-9` on BaseLlmArgs reflects baseline conflating BaseLlmArgs +
TrtLlmArgs into a single flat `engine_params` namespace. The ground truth
splits them - all 9 'missing' fields appear elsewhere (in engine_params_trt
or engine_params_torch). No information is lost in the split; it is a
fidelity gain.

## Top headline delta - what was MISSED

### 1. PluginConfig (43 fields, 4 invariants)

Status in baseline: zero coverage. `engine_versions/.../schema.discovered.json:536-541`
lists `build_config` as a single opaque `{type: object, default: null}` entry,
with the explicit limitation note (line 10-16) 'BuildConfig is not a Pydantic
model; appears as Optional[object] in the schema'. Inside BuildConfig there
is `plugin_config: PluginConfig` (builder.py:505), and PluginConfig in turn
exposes 43 metaclass-generated properties that EACH have a typed setter
with assertion-based validation.

Why it matters: PluginConfig is the engine-side control surface for the
TRT plugins that govern attention path, GEMM kernel selection, FP8 / FP4
quant fusion, paged-KV cache, fused MLP, and pipeline-parallel reduce
scatter. A benchmark caller comparing two TRT-LLM engines is in practice
comparing two PluginConfig settings. Examples that fall out of any caller's
plausible reach today:

- `gemm_plugin`, `gpt_attention_plugin`, `nccl_plugin` - core perf knobs
  (plugin/plugin.py:166, 173, 215).
- `use_paged_context_fmha`, `use_fp8_context_fmha`, `fuse_fp4_quant` -
  decisive for FP8/FP4 engines (plugin/plugin.py:376, 383, 390).
- `paged_kv_cache`, `tokens_per_block` - decisive for KV-cache memory
  modelling (plugin/plugin.py:328, 369).
- The `cli_plugin_args` list (plugin/plugin.py:588-619) curates 28 of these
  43 as `trtllm-build` CLI flags - those 28 are the canonical 'public'
  surface and were entirely missed.

Invariants added:
- `tensorrt_pluginConfig_dtype_not_auto_or_none` (plugin/plugin.py:120)
- `tensorrt_pluginConfig_bool_field_must_be_bool` (plugin/plugin.py:112)
- `tensorrt_pluginConfig_str_plugin_in_DEFAULT_PLUGIN_DTYPE_OPTIONS` (plugin/plugin.py:117)
- `tensorrt_pluginConfig_validate_unsupported_on_sm100` (plugin/plugin.py:513)

### 2. BuildConfig field expansion (27 fields)

Status in baseline: zero internal coverage (single opaque `object` entry).

Ground truth catalogues all 27 BuildConfig dataclass fields
(builder.py:478-511): `max_input_len`, `max_seq_len`, `opt_batch_size`,
`max_batch_size` (default 2048!), `max_beam_width`, `max_num_tokens`
(default 8192), `opt_num_tokens`, `max_prompt_embedding_table_size`,
`kv_cache_type`, `gather_context_logits`, `gather_generation_logits`,
`strongly_typed`, `force_num_profiles`, `profiling_verbosity`,
`enable_debug_output`, `max_draft_len`, `speculative_decoding_mode`,
`use_refit`, `input_timing_cache`, `output_timing_cache`, `lora_config`,
`auto_parallel_config`, `weight_sparsity`, `weight_streaming`,
`plugin_config`, `use_strip_plan`, `max_encoder_input_len`, `dry_run`,
`visualize_network`, `monitor_memory`, `use_mrope`.

Why it matters: BuildConfig defines the COMPILED engine's geometry. Once an
engine is built, max_batch_size etc. are baked in; BaseLlmArgs.max_batch_size
in the runtime config can only DOWN-shift, not up. The baseline's
`tensorrt_warns_max_batch_size_set_True_build_config_with_runtime_params`
invariant references this interaction without having any of the
build-config fields catalogued, so the constraint is unverifiable from
the catalogue alone.

### 3. Speculative decoding config tree (5 of 6 subclasses missed)

Status in baseline: only LookaheadDecodingConfig partially covered (the 3
positive-value invariants in invariants.validated.yaml). The other five
speculative configs are entirely absent.

Ground truth catalogues:
- `MedusaDecodingConfig` (llmapi/llm_args.py:223) - medusa_choices, num_medusa_heads
- `EagleDecodingConfig` (llmapi/llm_args.py:234) - 8 fields incl. eagle_choices, greedy_sampling, posterior_threshold, eagle3_one_model
- `NGramDecodingConfig` (llmapi/llm_args.py:252) - 5 fields incl. prompt_lookup_num_tokens, max_matching_ngram_size
- `DraftTargetDecodingConfig` (llmapi/llm_args.py:286) - pytorch_weights_path
- `MTPDecodingConfig` (llmapi/llm_args.py:296) - num_nextn_predict_layers, use_relaxed_acceptance_for_thinking, relaxed_topk, relaxed_delta (DeepSeek-style multi-token prediction)
- `DecodingBaseConfig` (llmapi/llm_args.py:196) - max_draft_len, speculative_model (inherited by all)

Invariants added:
- `tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch` (llmapi/llm_args.py:215) - validates the `decoding_type` string against the 6-class registry
- `tensorrt_medusaDecodingConfig_max_draft_len_positive_when_routed` (line 1291)
- `tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed` (line 1298)
- `tensorrt_ngramDecodingConfig_prompt_lookup_num_tokens_positive_when_routed` (line 1323)
- `tensorrt_ngramDecodingConfig_max_matching_ngram_size_positive_when_routed` (line 1323)
- `tensorrt_ngramDecodingConfig_backend_must_be_torch_when_routed` (line 1322)
- `tensorrt_draftTargetDecodingConfig_backend_must_be_pytorch_when_routed` (line 1337)
- `tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed` (line 1338)

Why it matters: speculative decoding is one of the four headline TRT-LLM
optimisation knobs (the others being quant, KV cache reuse, and the
attention plugin). The baseline catalogue exposes none of it past the
Lookahead positive-int check.

### 4. CalibConfig (6 of 7 fields missed)

Status in baseline: catalogued only `calib_config` as an opaque `{type:
object, default: null}` entry (`schema.discovered.json:510-523`), plus
the `calib_config.device` enum invariant.

Ground truth catalogues the dataclass-defined fields:
- `device: Literal['cuda','cpu']` (llmapi/llm_args.py:148)
- `calib_dataset: str = 'cnn_dailymail'` (line 151)
- `calib_batches: int = 512` (line 154)
- `calib_batch_size: int = 1` (line 157)
- `calib_max_seq_length: int = 512` (line 159)
- `random_seed: int = 1234` (line 162)
- `tokenizer_max_seq_length: int = 2048` (line 164)

Why it matters: calibration controls dataset selection and sample count
for FP8/INT8/AWQ quantization. Reproducing a quant'd benchmark requires
all of these. The baseline schema has `calib_dataset` and `calib_batches`
totally invisible.

### 5. TLLM_*/TRTLLM_* env var control surface (44 entries)

Status in baseline: zero coverage. The static miner only reads class-body
AST.

Ground truth enumerates every TLLM_* / TRTLLM_* env var that the Python
source reads via `os.environ.get(...)` / `os.getenv(...)`. Highlights:

- `TLLM_LOG_LEVEL` (logger.py:38) - sets the package logger root level.
- `TLLM_ALLOW_N_GREEDY_DECODING` (sampling_params.py:295) - directly
  influences SamplingParams._validate behaviour (best_of > 1 under greedy).
  Already referenced obliquely in the baseline invariant
  `tensorrt_warns_lora_config_set_True_lora_config_consistency` via its
  cross-field validator, but the env var itself is invisible.
- `TLLM_LLMAPI_BUILD_CACHE` + `TLLM_LLMAPI_BUILD_CACHE_ROOT`
  (build_cache.py:24, 26) - control the BuildCacheConfig default
  resolution.
- `TLLM_OVERRIDE_LAYER_NUM` (_torch/pyexecutor/model_engine.py:175) -
  forcibly overrides the model layer count. Critical for test isolation
  + small-model debugging; invisible to any caller relying on
  schema.discovered.json.
- `TRTLLM_USE_UCX_KVCACHE` / `TRTLLM_USE_NIXL_KVCACHE` /
  `TRTLLM_USE_MPI_KVCACHE` (_torch/pyexecutor/kv_cache_transceiver.py:35-39)
  - first-match-wins KV cache transport selector. Disagg-serve auto-sets
  the MPI variant.
- `TRTLLM_ENABLE_PDL` (_torch/custom_ops/flashinfer_custom_ops.py:13) -
  FlashInfer Programmatic Dependent Launch. Force-set by the low-latency
  bench script.
- `TRTLLM_DEEPSEEK_EAGER_FUSION_DISABLED` /
  `TRTLLM_QWEN3_EAGER_FUSION_DISABLED` - per-architecture eager-fusion
  killswitches. Invisible to baseline.

Why it matters: env vars are HOW production deployments tune TRT-LLM
without re-building the engine. A benchmark catalogue that doesn't
enumerate them is structurally missing the ~30% of the control surface
that is reachable at runtime.

### 6. TorchLlmArgs (21 fields, 4 invariants) - entire backend missing

Status in baseline: zero coverage. The static miner picked up
`_AutoDeployLlmArgs` (a TorchLlmArgs subclass) but treated TorchLlmArgs
itself as not a thing.

Ground truth catalogues all 21 TorchLlmArgs fields
(llmapi/llm_args.py:1624-1818), including:
- `use_cuda_graph`, `cuda_graph_batch_sizes`, `cuda_graph_max_batch_size`,
  `cuda_graph_padding_enabled` - CUDA-graph capture knobs.
- `disable_overlap_scheduler` - PyT backend scheduler control.
- `moe_max_num_tokens`, `moe_load_balancer`, `moe_backend` - MoE knobs.
- `attn_backend` (default 'TRTLLM', also 'FLASHINFER',
  'TritonWithFlattenedInputs').
- `mixed_sampler`, `enable_trtllm_sampler`, `kv_cache_dtype`.
- `torch_compile_config: Optional[TorchCompileConfig]` - which further
  expands to 4 TorchCompileConfig fields.
- `load_format: Union[str, LoadFormat]` with its own `convert_load_format`
  before-validator (line 1730).

Invariants added: cuda_graph_max_batch_size ge 0, cuda_graph_config dual
specification, moe_load_balancer file_exists, moe_load_balancer parse,
load_format enum.

Why it matters: TorchLlmArgs is the eager-PyTorch backend (backend ==
'pytorch'). 0.21 is the version where PyTorch backend is becoming the
default migration path away from the TRT engine-build path; missing it
means missing the future TRT-LLM API surface entirely.

### 7. LoRA full subconfig (7 fields, 1 invariant)

Status in baseline: catalogued `lora_config: Optional[LoraConfig]` as
opaque object; one invariant
(`tensorrt_warns_lora_config_set_True_lora_config_consistency`).

Ground truth catalogues the LoraConfig dataclass fully
(lora_manager.py:138-156): lora_dir, lora_ckpt_source, max_lora_rank,
lora_target_modules, trtllm_modules_to_hf_modules, max_loras,
max_cpu_loras. Plus the lora_ckpt_source enum invariant
(lora_manager.py:148, asserts in {hf, nemo}).

### 8. SchedulerConfig sub-tree (DynamicBatchConfig)

Status in baseline: catalogued `scheduler_config` as opaque object.

Ground truth: SchedulerConfig has 3 fields (capacity_scheduler_policy,
context_chunking_policy, dynamic_batch_config). DynamicBatchConfig
itself has 3 required fields - none of them catalogued previously.
Capacity policy is a 3-member StrEnum; context chunking is a 2-member
StrEnum.

### 9. SamplingParams.GuidedDecodingParams (5 fields, 1 invariant)

Status in baseline: catalogued `guided_decoding` as opaque object inside
sampling_params; no expansion. Ground truth catalogues GuidedDecodingParams
(sampling_params.py:13-36) as 5 mutually-exclusive guide selectors (json,
regex, grammar, json_object, structural_tag) plus the at-most-one-truthy
invariant in `GuidedDecodingParams._validate` (line 35).

### 10. Internal consistency invariant (ExecutorConfig mirror)

Status in baseline: not catalogued. Ground truth surfaces
`BaseLlmArgs._check_consistency` (llmapi/llm_args.py:1032-1048) which
asserts `assert executor_config_attrs.issubset(llm_args_attr)`. This is
THE invariant that guarantees BaseLlmArgs stays a superset of
`tensorrt_llm.bindings.executor.ExecutorConfig`. Catalogued as
`tensorrt_baseLlmArgs_check_consistency_with_ExecutorConfig`. It is a
meta-invariant - it fires at the consistency check, not at user input
validation - but it bounds the catalogue's completeness claim, so it
matters for the substrate.

## Mid-tier additions

### Mapping-from-engine consistency (_load_config_from_engine / _load_config_from_ckpt)

Six invariants catalogued (llmapi/llm_args.py:1422-1468) that fire when the
user's tp_size / pp_size / cp_size disagrees with the loaded engine /
checkpoint's mapping. Baseline misses all six.

### _ParallelConfig invariants (4)

Catalogued _ParallelConfig's auto-parallel constraints (lines 89, 99, 105,
121). These are inner-class invariants that the baseline AST miner skips
because `_ParallelConfig` starts with `_`.

### _ModelWrapper.model required (line 727)

Catalogued. Baseline missed it.

### get_model_format dispatch invariants (lines 2043, 2061)

Catalogued. These fire when BaseLlmArgs sees a local-dir model but cannot
parse `config.json`.

### PluginConfig.validate (SM-100 unsupported plugins)

Catalogued at plugin/plugin.py:513. Affects 5 plugin fields on Blackwell.
Baseline misses entirely.

### LayerQuantConfig _get_modelopt_qformat assertion (line 343)

Catalogued. Baseline misses entirely.

### QuantConfig._get_modelopt_qformat assertion (line 208) and per-algo
allowlists (lines 210, 221)

Catalogued. Baseline catalogues QuantAlgo enum members but not the
modelopt-specific subsets.

### CapacitySchedulerPolicy enum (3 members), ContextChunkingPolicy enum
(2 members)

Catalogued. Baseline catalogues BatchingType but not these two sibling
enums, which control the same scheduler.

### setup_required-before-_get_*_words invariants in SamplingParams

Two invariants at sampling_params.py:363 and 378: RuntimeError fires if the
user calls `_get_bad_words` / `_get_stop_words` on a SamplingParams whose
`bad` / `stop` field was set but `setup(tokenizer)` was not called.
Baseline misses both.

## False positives in baseline (predicates not actually firing as written)

None confirmed in this scan. The baseline's 35 invariants are all
genuine validator outputs - the issue is scope (missing 40 more), not
correctness.

One soft observation: the baseline's
`tensorrt_warns_lora_config_set_True_lora_config_consistency` is grouped
under a single id but the validator emits 4 distinct warnings at lines
1376 / 1379 / 1382 / 1387 / 1394 / 1401 (5 in total, but two emit the
same message template). The baseline's `conflict_note` field captures
this collision. Ground truth splits them into 6 separate invariants for
unambiguous matching during scoring.

## Notes for the bake-off

- The 44-entry env-var namespace is the single largest miss. It is also
  the cheapest to recover at substrate level: a single `grep` pattern
  + a tiny LLM extraction prompt suffices, and the result is high-signal
  (each env var is a real control plane).
- BuildConfig + PluginConfig expansion together account for 70 of the
  250 schema additions and 5 of the 40 new invariants. Recovery requires
  the substrate to be able to walk `@dataclass` field declarations and
  metaclass-generated property assertions - the latter is HARD for the
  current source-driven static miner because there is no class-body
  AST that names the public properties; the metaclass synthesises them
  at class-creation time from the `_<name>` prefix convention.
- The speculative-config tree is medium-signal but has unusual semantics
  (the BaseLlmArgs.validate_speculative_config validator REPLACES the
  config instance with a different class at line 1314-1354 depending on
  backend). Any substrate that scores invariant fidelity by emitting
  positive/negative test kwargs will need to know about the replacement
  semantics; otherwise the `kwargs_positive` examples will fire under the
  wrong class.

## Open questions flagged for review

- LookaheadDecodingConfig defaults (`max_window_size`, `max_ngram_size`,
  `max_verification_set_size`) come from `_LookaheadDecodingConfig.get_default_lookahead_decoding_*()`
  (llmapi/llm_args.py:561-569). These are C++-side defaults; we do not
  know them without importing the .so. Catalogued as
  `default_resolved_from_cpp: true` with the call-site as the default
  string. Reviewer: confirm we don't need to hand-resolve these at this
  pin.

- `TRTLLM_DISABLE_UNIFIED_CONVERTER` and the `TRTLLM_DG_*` env vars hit
  the grep but their full read-context wasn't extracted in this scan
  (sole-hit reads; behaviour comment-only). Catalogued with low-detail
  source flag.

- The .so binding stub at `tensorrt_llm/bindings/__init__.pyi` exposes
  19 C++ classes (CudaStream, DataType, GptJsonConfig, GptModelVariant,
  IpcNvlsHandle, KVCacheType, KvCacheConfig, LayerType, LlmRequestState,
  LoraModule, LoraModuleType, MemoryCounters, ModelConfig, MpiComm,
  PeftCacheManagerConfig, QuantMode, SamplingConfig,
  TrtGptModelOptionalParams, WorldConfig). Of these, only
  `KvCacheConfig` and `PeftCacheManagerConfig` have Python BaseModel
  mirrors that BaseLlmArgs exposes; the rest are accessible only by a
  caller who imports `tensorrt_llm.bindings` directly. Treating them as
  out-of-scope for the LLM-API config catalogue is defensible; reviewer
  should confirm.
