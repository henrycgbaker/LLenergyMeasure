# Delta vs LLEM baseline - vllm 0.7.3

Baseline = `engine_versions/vllm/v0_7_3/outputs/{schema.discovered.json, invariants.validated.yaml, invariants.proposed.yaml}` as of 2026-05-24.
Ground truth = `research/mining-substrate-trial/findings/ground_truth/vllm/v0_7_3/{schema_ground_truth.json, invariants_ground_truth.yaml}` as of 2026-06-05.

## Headline numbers

| Surface | Baseline | Ground truth | Net additions |
| --- | --- | --- | --- |
| Engine params (EngineArgs) | 96 fields | 103 fields | +7 net (mostly tightened types / enum constraints; see "Engine-params shape upgrades" below; baseline had a few "untyped/Any" entries the catalogue resolves) |
| SamplingParams | 31 entries (12 properly typed + 19 "discovery emitted 'unknown'") | 29 fully-typed entries | +19 untyped -> typed; -2 net (the 19 "unknown" entries resolved; two private fields not enumerated) |
| BeamSearchParams | 0 | 6 | +6 (entire struct missing from baseline) |
| PoolingParams | 0 | 1 (flagged out-of-scope) | +1 |
| GuidedDecodingParams | 0 | 7 | +7 (entire struct missing) |
| Subconfig classes (full enumeration) | 0 explicit field listings | 18 classes / 125 fields | +125 (baseline only references subconfig classes by name in EngineArgs.compilation_config / kv_transfer_config / override_pooler_config) |
| **vllm.envs env vars** | **0** | **87** | **+87** (entire runtime control surface) |
| **Invariants** | **26** | **86** | **+60** |

## Primary gap: `vllm.envs` (+87 entries)

The baseline schema enumerates `engine_params` and `sampling_params` only. The `vllm.envs.environment_variables` dict (`vllm/envs.py:120-596`) defines **87** environment variables that change runtime behaviour, several of them in ways that affect benchmark correctness or fairness. The baseline misses all of them.

Concrete high-leverage examples:

- **`VLLM_USE_V1`** (`vllm/envs.py:503`) - switches to the V1 engine code path. Changes compilation defaults (CompilationConfig overrides at `vllm/config.py:3295`), scheduling, output processing chunk size, and which attention splitting ops are used (`vllm/config.py:2988`). A benchmark run with `VLLM_USE_V1=1` is a fundamentally different engine and the baseline catalogue gives the caller no way to record this.
- **`VLLM_ATTENTION_BACKEND`** (`vllm/envs.py:312`) - forces FLASH_ATTN / FLASHINFER / XFORMERS / TORCH_SDPA / ROCM_FLASH. Read at `vllm/config.py:336` and changes which kernel the attention layer dispatches to. Energy-per-token differs across backends.
- **`VLLM_ALLOW_LONG_MAX_MODEL_LEN`** (`vllm/envs.py:445`) - gates a `raise ValueError` in `_get_and_verify_max_len` (`vllm/config.py:2571,2576`). With it set, `max_model_len` > model's config.json is allowed; without it, the engine refuses to start. This is the textbook env-var-gated invariant.
- **`VLLM_USE_RAY_SPMD_WORKER`** (`vllm/envs.py:364`) - read at `vllm/config.py:691` to silently disable async output processing. Behavioural-change-on-env-var without any field-level signal.
- **`VLLM_MLA_DISABLE`** / **`VLLM_MLA_PERFORM_MATRIX_ABSORPTION`** / **`VLLM_MLA_DISABLE_REQUANTIZATION`** (`vllm/envs.py:538,546,555`) - control MLA attention behaviour for deepseek_v2/v3 models. Read at `vllm/config.py:999` (ModelConfig.use_mla property). Changes memory footprint, FLOPs, and numerical precision.
- **`VLLM_TORCH_PROFILER_DIR`** (`vllm/envs.py:473`) - enables the torch profiler. Has measurable overhead. A benchmark that sets this accidentally will report artificially slow latencies.
- **`VLLM_LOGITS_PROCESSOR_THREADS`** (`vllm/envs.py:295`) - thread-pool size for logits processors. Affects throughput and CPU contention.
- **`VLLM_CPU_KVCACHE_SPACE`**, **`VLLM_CPU_OMP_THREADS_BIND`** (`vllm/envs.py:331,336`) - CPU backend KV cache and thread binding. Energy-per-token sensitivity is high.

Discrepancies surfaced while cataloguing:

- **`VLLM_CUDA_MEM_ALIGN_KV_CACHE`** - the lambda key (`vllm/envs.py:582`) and the TYPE_CHECKING stub (`vllm/envs.py:87` claims `VLLM_MLA_CUDA_MEM_ALIGN_KV_CACHE`) disagree. The dict key is authoritative because that's what `envs.__getattr__` exposes (`vllm/envs.py:601`). A consumer who introspects the stub will look up the wrong name.
- **`VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH`** - the attribute name and the env-var-name-actually-read by the lambda differ. The attr is `VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH` but the lambda calls `os.environ.get("VLLM_CONTIGUOUS_PA", ...)` at `vllm/envs.py:593-595`. Setting `VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH=true` in your shell has no effect; you must set `VLLM_CONTIGUOUS_PA`. This is a pure footgun that the static-discovery substrate has been blind to.

## Invariants: +60 additions vs 26 baseline

The baseline's 26 invariants cover the most obvious enum / range checks on SamplingParams and one or two checks per subconfig. The 60 additions break down as follows.

### Cross-field / cross-config / cross-subconfig invariants (entirely missed)

These are the highest-value class because they require the miner to understand multi-field relationships. Baseline catches **none** of them.

| ID | Severity | What it checks | Citation |
| --- | --- | --- | --- |
| `vllm_samplingparams_raises_stop_requires_detokenize` | error | `stop` non-empty requires `detokenize=True` | sampling_params.py:404 |
| `vllm_samplingparams_raises_delta_output_kind_requires_best_of_eq_n` | error | `output_kind=DELTA` requires `best_of==_real_n` | sampling_params.py:408 |
| `vllm_samplingparams_raises_greedy_requires_n_eq_1` | error | `temperature<1e-5` requires `n==1` (greedy implies single sample) | sampling_params.py:412 |
| `vllm_cacheconfig_raises_prefix_caching_with_sliding_window` | error (NotImplementedError) | prefix caching incompatible with sliding window | config.py:1108 |
| `vllm_loraconfig_raises_cpu_offload_with_lora` | error | LoRA incompatible with `cpu_offload_gb>0` | config.py:2205 |
| `vllm_parallelconfig_raises_nsight_without_ray` | error | `ray_workers_use_nsight=True` requires use_ray | config.py:1418 |
| `vllm_schedulerconfig_raises_partial_prefills_requires_chunked_prefill` | error | `max_num_partial_prefills>1` requires chunked_prefill | config.py:1608 |
| `vllm_schedulerconfig_raises_long_prefill_token_threshold_gt_ref_max_model_len` | error | cross-field range check (only fires under partial-prefill mode) | config.py:1612 |
| `vllm_schedulerconfig_raises_max_long_partial_prefills_out_of_range` | error | `1 <= max_long_partial_prefills <= max_num_partial_prefills` | config.py:1618 |
| `vllm_modelconfig_raises_attention_head_div_tensor_parallel` | error | num_attention_heads must be divisible by TP size | config.py:716 |
| `vllm_modelconfig_raises_pipeline_parallel_unsupported_model` | error (NotImplementedError) | PP>1 requires `SupportsPP` | config.py:726 |
| `vllm_modelconfig_raises_limit_mm_per_prompt_for_non_multimodal` | error | mm-limit set on non-mm model | config.py:431 |
| `vllm_kvtransferconfig_raises_connector_without_role` | error | kv_connector set requires kv_role set | config.py:2745 |
| `vllm_vllmconfig_warning_mla_disables_chunked_prefill_and_prefix_cache` | warning | MLA models silently disable chunked_prefill AND prefix_caching at the VllmConfig level | config.py:3328 |
| `vllm_vllmconfig_warning_cpu_offload_disables_compile` | warning | cpu_offload silently disables torch.compile | config.py:3311 |
| `vllm_vllmconfig_warning_lora_disables_compile` | warning | LoRA silently disables torch.compile | config.py:3319 |
| `vllm_modelconfig_warning_async_output_proc_disabled_with_pipeline_parallel` | warning | PP>1 silently disables async output proc | config.py:675 |
| `vllm_modelconfig_warning_async_output_proc_disabled_with_spec_decode` | warning | spec-decode silently disables async output proc | config.py:705 |

**Why this matters:** silent-normalisation invariants are exactly what energy-benchmarking-as-a-service needs to surface. A caller who sets `enable_prefix_caching=true` and runs a deepseek_v2 model will have prefix_caching silently turned off and never know. The benchmark will report numbers as if prefix_caching was active.

### Env-var-gated invariants (entirely missed)

- `vllm_modelconfig_raises_max_model_len_exceeds_derived` - gated by `VLLM_ALLOW_LONG_MAX_MODEL_LEN`. config.py:2571.
- `vllm_modelconfig_warning_async_output_proc_disabled_with_ray_spmd` - gated by `VLLM_USE_RAY_SPMD_WORKER`. config.py:691.

### Platform-gated invariants (entirely missed)

- `vllm_modelconfig_raises_sleep_mode_non_cuda` - `enable_sleep_mode` requires CUDA. config.py:301.
- `vllm_parallelconfig_warning_rocm_disables_custom_all_reduce` - ROCm forces `disable_custom_all_reduce=True`. config.py:1413.

### Silent-normalisation invariants on ModelConfig (warning+clamp variety)

- `vllm_modelconfig_warning_cuda_graph_disabled_for_mllama` - model_type=='mllama' silently turns `enforce_eager=True`. config.py:640.
- `vllm_modelconfig_warning_cuda_graph_disabled_for_bnb_8bit` - bnb 8bit silently turns `enforce_eager=True`. config.py:658.
- `vllm_modelconfig_warning_quantization_not_optimized` - performance warning for non-optimised quant methods. config.py:628.
- `vllm_samplingparams_warning_temperature_clamped_to_min` - `0<temp<1e-2` clamped to 1e-2 with warning. sampling_params.py:303.

### Enum / range constraints baseline omitted

Several "obvious" enum checks that baseline missed:

- `vllm_samplingparams_raises_presence_penalty_out_of_range` (sampling_params.py:358) - `[-2, 2]`
- `vllm_samplingparams_raises_frequency_penalty_out_of_range` (sampling_params.py:361) - `[-2, 2]`
- `vllm_samplingparams_raises_repetition_penalty_out_of_range` (sampling_params.py:364) - `(0, 2]`
- `vllm_samplingparams_raises_top_p_out_of_range` (sampling_params.py:370) - `(0, 1]`
- `vllm_samplingparams_raises_top_k_value_out_of_range` (sampling_params.py:372) - `top_k < -1 or top_k == 0` (ValueError, not the TypeError baseline catches)
- `vllm_samplingparams_raises_min_p_out_of_range` (sampling_params.py:378) - `[0, 1]`
- `vllm_samplingparams_raises_stop_contains_empty_string` (sampling_params.py:402) - empty-string-in-stop rejected
- `vllm_loraconfig_raises_max_lora_rank_not_in_allowed_set` (config.py:2186) - `{8,16,32,64,128,256}`
- `vllm_loraconfig_raises_lora_extra_vocab_size_not_in_allowed_set` (config.py:2190) - `{0,256,512}`
- `vllm_modelconfig_raises_unknown_tokenizer_mode` (config.py:471) - `{auto,slow,mistral,custom}`
- `vllm_modelconfig_raises_unknown_quantization_method` (config.py:620) - must be in QUANTIZATION_METHODS
- `vllm_modelconfig_raises_unknown_dtype` (config.py:2429) - must be in `{half,float16,float,float32,bfloat16}`
- `vllm_cacheconfig_raises_unknown_cache_dtype` (config.py:1092) - `{auto, fp8, fp8_e4m3, fp8_e5m2}`
- `vllm_decodingconfig_raises_unknown_guided_decoding_backend` (config.py:2635) - `{outlines, lm-format-enforcer, xgrammar}`
- `vllm_kvtransferconfig_raises_unknown_kv_role` (config.py:2737) - `{kv_producer, kv_consumer, kv_both}`
- `vllm_speculativeconfig_raises_unknown_acceptance_method` (config.py:2112) - `{rejection_sampler, typical_acceptance_sampler}`
- `vllm_tokenizerpoolconfig_raises_unknown_pool_type` (config.py:1166) - `{'ray'} ∪ subclasses(BaseTokenizerGroup)`

### SpeculativeConfig: entire class missing from baseline (+10)

Baseline has zero invariants on SpeculativeConfig despite the class having ~10 enforced constraints. Examples:
- `num_speculative_tokens <= 0` raises (config.py:2098)
- `typical_acceptance_sampler_posterior_threshold < 0` or `..._alpha < 0` raises (config.py:2120)
- `ngram_prompt_lookup_max < 1` raises (config.py:1823)
- `ngram_prompt_lookup_min > ngram_prompt_lookup_max` raises (config.py:1827)
- `speculative_max_model_len > draft_max_model_len` raises (config.py:1959)
- `speculative_draft_tensor_parallel_size not in {1, target_tp}` raises (config.py:1997)
- `num_speculative_tokens > draft_hf_config.n_predict` raises (config.py:1884)
- `num_speculative_tokens is None` without `n_predict` raises (config.py:1909)

### Deprecation warnings (entirely missed)

Baseline severity vocab has `deprecated` but no entries use it.
- `vllm_modelconfig_deprecation_rope_scaling` (config.py:274) - DeprecationWarning emitted via `warnings.warn`.
- `vllm_modelconfig_deprecation_rope_theta` (config.py:280) - DeprecationWarning emitted via `warnings.warn`.

### AssertionError invariants

Rare in vllm config code; the one example baseline misses:
- `vllm_compilationconfig_assertion_custom_ops_none_and_all` (config.py:2985) - `assert count_none + count_all <= 1` on CompilationConfig.custom_ops. Raises AssertionError, not ValueError.

## 3 concrete high-value additions (review-priority picks)

1. **vllm.envs as a first-class catalogued surface (87 entries).** Without it the catalogue cannot answer "what state was this benchmark run in?" beyond CLI flags. `VLLM_USE_V1` alone changes engine behaviour enough to invalidate cross-run comparisons.
2. **Silent-normalisation invariants at the VllmConfig level (e.g. MLA disables prefix_caching, LoRA disables torch.compile, cpu_offload disables torch.compile).** Caller sets a flag, vLLM silently overrides it, benchmark records the declared not observed value. These are exactly the failure modes a benchmark-as-a-service must surface.
3. **Env-var-gated invariants (e.g. `VLLM_ALLOW_LONG_MAX_MODEL_LEN` gates the `max_model_len > derived_max_model_len` raise).** These are invisible to a pure-field static miner that doesn't model `envs.X` reads as inputs to predicates.

## Engine-params shape upgrades

Even where the baseline lists a field, several entries have richer constraints than the baseline carries:

- `tokenizer_mode`: baseline = string, ground truth = enum `{auto,slow,mistral,custom}` (validator at config.py:471).
- `load_format`: baseline = string, ground truth = enum of 11 values from `LoadFormat` enum (config.py:1204).
- `dtype`: baseline = string, ground truth = enum of `{auto,half,float16,float,float32,bfloat16}` (config.py:2360).
- `kv_cache_dtype`: baseline = string, ground truth = enum `{auto,fp8,fp8_e4m3,fp8_e5m2}` (config.py:1095).
- `guided_decoding_backend`: baseline = string, ground truth = enum `{outlines,lm-format-enforcer,xgrammar}` (config.py:2633).
- `model_impl`: baseline = string, ground truth = enum `{auto,vllm,transformers}` (config.py:87).
- `spec_decoding_acceptance_method`: baseline = string, ground truth = enum `{rejection_sampler,typical_acceptance_sampler}`.
- `max_lora_rank`: baseline = integer, ground truth = enum `{8,16,32,64,128,256}`.
- `lora_extra_vocab_size`: baseline = integer, ground truth = enum `{0,256,512}`.
- `device`: baseline = string, ground truth = enum-hint `{auto,cuda,rocm,neuron,cpu,openvino,tpu,hpu,xpu}`.
- `rope_scaling` and `rope_theta` marked `deprecated: true` with citation to the `warnings.warn` calls.

These tighter types are how a downstream config builder can reject invalid inputs at edit time instead of at engine construction time.

## What still gates exhaustiveness

- Quantization sub-config tree (AWQ / GPTQ / FP8 / Marlin / bitsandbytes / compressed-tensors / ...) - one class per method under `vllm/model_executor/layers/quantization/`. Not enumerated here; treat the `quantization` field as opaque pending a follow-up pass.
- Per-platform validation surface (`vllm/platforms/{cuda,rocm,neuron,cpu,openvino,tpu,hpu,xpu}.py`) - several invariants reference `current_platform.is_*()` checks but the platforms have their own `check_and_update_config` methods that were not walked.
- Multimodal processor kwargs (`mm_processor_kwargs` field) - schema is per-model and lives in `vllm/multimodal/`. Not enumerated.
- The msgspec.Meta annotations on `truncate_prompt_tokens` (ge=1) are captured as a `also_enforced_at` note, but msgspec's other validators on other fields (if any get added in future versions) would need a separate sweep.
