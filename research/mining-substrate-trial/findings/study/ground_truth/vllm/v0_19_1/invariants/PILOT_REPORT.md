# Pilot GT report - vllm 0.19.1 invariants (union + gate)

Round 0: union the 4 GT sources (mech, passA, passB, poc) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside vllm/vllm-openai:v0.19.1, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 82 | 82 | 82 | 6 |
| passB | 64 | 64 | 64 | 64 |
| mech | 111 | 105 | 111 | 100 |
| poc | 79 | 79 | 79 | 3 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **249**
- Tolerant keys (coarser, leaf+bucket): 166; of which **64** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **90**
- Probed candidates (native_type present, kwargs authored or synthesised): **336** (confirmed=127, failed=52, skipped=62, infra_error=95)
- Confirmations by probe provenance: **6 synthesised** by the gate, 121 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 90
- failed: 35
- infra_error: 63
- skipped: 61

## GT-growth vs PoC N=1 GT

PoC GT contributed **79** constraints. The gate-confirmed union grows GT by **54** confirmed constraints the PoC GT lacked:

- backend [membership] = [auto,guidance,lm-format-enforcer,outlines,xgrammar]
- backend [membership] = [ipc,nccl]
- collect_detailed_traces [presence] = {require=otlp_traces_endpoint is set,when=collect_detailed_traces}
- compile_cache_save_format [membership] = [binary,unpacked]
- data_parallel_backend [membership] = [mp,ray]
- dcp_comm_backend [membership] = [a2a,ag_rs]
- disable_additional_properties [presence] = {require=backend == guidance,when=disable_additional_properties}
- disable_any_whitespace [presence] = {require=backend in {xgrammar, guidance},when=disable_any_whitespace}
- ec_role [presence] = {require=ec_role is not None,when=ec_connector is not None}
- expert_placement_strategy [membership] = [linear,round_robin]
- flash_attn_version [membership] = [2,3,4]
- frequency_penalty [numeric] = {ge=-2,le=2}
- gpu_memory_utilization [numeric] = {gt=0,le=1}
- kv_load_failure_policy [membership] = [fail,recompute]
- kv_offloading_backend [membership] = [lmcache,native]
- kv_role [presence] = {require=kv_role is not None,when=kv_connector is not None}
- log_balancedness_interval [numeric] = 0
- log_balancedness_interval [presence] = {require=log_balancedness_interval > 0,when=log_balancedness}
- lora_dtype [membership] = [auto,bfloat16,float16]
- mamba_block_size [numeric] = {allow_none=True,gt=0}
- mamba_cache_dtype [membership] = [auto,float16,float32]
- mamba_cache_mode [membership] = [align,all,none]
- max_cpu_loras [presence] = {require=max_cpu_loras is None or max_cpu_loras >= max_loras}
- max_lora_rank [membership] = [1,128,16,256,32,320,512,64,8]
- max_loras [numeric] = {ge=1}
- max_tokens [numeric] = {allow_none=True,ge=1}
- min_count [presence] = {require=min_count >= 2,when=max_pattern_size > 0}
- min_p [numeric] = {ge=0,le=1}
- min_tokens [numeric] = {ge=0}
- mm_encoder_attn_backend [presence] = {else=AttentionBackendEnum[value.upper()],reject=XFORMERS}
- mm_encoder_tp_mode [membership] = [data,weights]
- mm_processor_cache_type [membership] = [lru,shm]
- n [numeric] = {ge=1}
- n [presence] = n > envs.VLLM_MAX_N_SEQUENCES (default 16384)
- n [presence] = {require=n == 1,when=temperature < _SAMPLING_EPS (greedy)}
- num_redundant_experts [numeric] = {ge=0}
- offload_backend [membership] = [auto,prefetch,uva]
- policy [membership] = [default]
- policy [membership] = [fcfs,priority]
- prefix_caching_hash_algo [membership] = [sha256,sha256_cbor,xxhash,xxhash_cbor]
- ... (+14 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| vllm_samplingparams__validate_spec_decode_min_p_gt | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__validate_structured_outputs_structured_outputs_not_equal | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_frequency_penalty_le | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_logprobs_not_equal | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_max_tokens_gt | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_max_tokens_lt | mech | vllm.SamplingParams | error |
| vllm_samplingparams__verify_args_min_p_le | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_n_gt | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_n_lt | mech | vllm.SamplingParams | error |
| vllm_samplingparams__verify_args_presence_penalty_le | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_prompt_logprobs_not_equal | mech | vllm.SamplingParams | no_op |
| vllm_samplingparams__verify_args_top_k_lt | mech | vllm.SamplingParams | no_op |
| vllm_loraconfig_normalise_max_cpu_loras_unset | passA | vllm.config.LoRAConfig | dormant_silent |
| vllm_modelconfig_raises_deprecated_quantization_method | passA | vllm.config.ModelConfig | error |
| vllm_modelconfig_raises_pipeline_parallel_unsupported_model | passA | vllm.config.ModelConfig | error |
| vllm_modelconfig_raises_sleep_mode_unsupported_platform | passA | vllm.config.ModelConfig | warn |
| vllm_modelconfig_raises_unknown_dtype | passA | vllm.config.ModelConfig | error |
| vllm_modelconfig_warning_cuda_graph_disabled_for_bnb_8bit | passA | vllm.config.ModelConfig | error |
| vllm_observabilityconfig_raises_otlp_without_opentelemetry | passA | vllm.config.ObservabilityConfig | no_op |
| vllm_offloadconfig_raises_num_in_group_gt_group_size | passA | vllm.config.OffloadConfig | error |
| vllm_samplingparams_normalise_seed_eq_neg1 | passA | vllm.SamplingParams | dormant_silent |
| vllm_speculativeconfig_raises_draft_max_model_len_override | passA | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_missing_num_speculative_tokens | passA | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_num_speculative_tokens_le_0 | passA | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_tensor_parallel_size_set | passA | vllm.config.SpeculativeConfig | error |
| vllm_vllmconfig_raises_torch_shm_requires_spawn | passA | vllm.config.VllmConfig | error |
| vllm_vllmconfig_warning_cudagraph_mode_incompatible_with_compile_mode | passA | vllm.config.VllmConfig | error |
| vllm_vllmconfig_warning_sequence_parallelism_requires_tp_gt_1 | passA | vllm.config.VllmConfig | error |
| vllm_attentionconfig_backend_enum_membership | passB | vllm.config.AttentionConfig | error |
| vllm_cacheconfig_cache_dtype_literal | passB | vllm.config.CacheConfig | error |
| vllm_modelconfig_dtype_literal | passB | vllm.config.ModelConfig | error |
| vllm_modelconfig_tokenizer_mode_literal | passB | vllm.config.ModelConfig | no_op |
| vllm_poolingparams_output_kind_final_only | passB | vllm.PoolingParams | error |
| vllm_schedulerconfig_batched_tokens_ge_max_num_seqs | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_max_long_le_max_partial_prefills | passB | vllm.config.SchedulerConfig | error |
| vllm_speculativeconfig_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_rejection_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_loraconfig_dormant_max_cpu_loras_unset | poc | vllm.config.LoRAConfig | dormant_silent |
| vllm_modelconfig_raises_deprecated_quantization_method | poc | vllm.config.ModelConfig | error |
| vllm_modelconfig_raises_pipeline_parallel_unsupported_model | poc | vllm.config.ModelConfig | error |
| vllm_modelconfig_raises_sleep_mode_unsupported_platform | poc | vllm.config.ModelConfig | no_op |
| vllm_modelconfig_raises_unknown_dtype | poc | vllm.config.ModelConfig | error |
| vllm_modelconfig_warning_cuda_graph_disabled_for_bnb_8bit | poc | vllm.config.ModelConfig | error |
| vllm_offloadconfig_raises_num_in_group_gt_group_size | poc | vllm.config.OffloadConfig | error |
| vllm_samplingparams_dormant_seed_eq_neg1 | poc | vllm.SamplingParams | dormant_silent |
| vllm_speculativeconfig_raises_draft_max_model_len_override | poc | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_missing_num_speculative_tokens | poc | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_num_speculative_tokens_le_0 | poc | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_tensor_parallel_size_set | poc | vllm.config.SpeculativeConfig | error |
| vllm_vllmconfig_raises_torch_shm_requires_spawn | poc | vllm.config.VllmConfig | error |
| vllm_vllmconfig_warning_cudagraph_mode_incompatible_with_compile_mode | poc | vllm.config.VllmConfig | error |
| vllm_vllmconfig_warning_sequence_parallelism_requires_tp_gt_1 | poc | vllm.config.VllmConfig | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `vllm_audiodummyoptions___declarative___length_gt` (mech): module vllm has no attribute AudioDummyOptions
- `vllm_cacheconfig___declarative___gpu_memory_utilization_gt` (mech): module vllm has no attribute CacheConfig
- `vllm_cacheconfig___declarative___mamba_block_size_gt` (mech): module vllm has no attribute CacheConfig
- `vllm_eplbconfig___declarative___num_redundant_experts_ge` (mech): module vllm has no attribute EPLBConfig
- `vllm_loraconfig___declarative___max_loras_ge` (mech): module vllm has no attribute LoRAConfig
- `vllm_modelconfig___declarative___max_model_len_ge` (mech): module vllm has no attribute ModelConfig
- `vllm_multimodalconfig___declarative___mm_processor_cache_gb_ge` (mech): module vllm has no attribute MultiModalConfig
- `vllm_multimodalconfig___declarative___mm_shm_cache_max_object_size_mb_ge` (mech): module vllm has no attribute MultiModalConfig
- `vllm_multimodalconfig___declarative___video_pruning_rate_ge` (mech): module vllm has no attribute MultiModalConfig
- `vllm_observabilityconfig___declarative___kv_cache_metrics_sample_gt` (mech): module vllm has no attribute ObservabilityConfig
- `vllm_prefetchoffloadconfig___declarative___offload_group_size_ge` (mech): module vllm has no attribute PrefetchOffloadConfig
- `vllm_prefetchoffloadconfig___declarative___offload_num_in_group_ge` (mech): module vllm has no attribute PrefetchOffloadConfig
- `vllm_prefetchoffloadconfig___declarative___offload_prefetch_step_ge` (mech): module vllm has no attribute PrefetchOffloadConfig
- `vllm_profilerconfig___declarative___active_iterations_ge` (mech): module vllm has no attribute ProfilerConfig
- `vllm_profilerconfig___declarative___delay_iterations_ge` (mech): module vllm has no attribute ProfilerConfig
- `vllm_profilerconfig___declarative___max_iterations_ge` (mech): module vllm has no attribute ProfilerConfig
- `vllm_profilerconfig___declarative___wait_iterations_ge` (mech): module vllm has no attribute ProfilerConfig
- `vllm_profilerconfig___declarative___warmup_iterations_ge` (mech): module vllm has no attribute ProfilerConfig
- `vllm_repetitiondetectionparams___post_init___max_pattern_size_gt` (mech): module vllm has no attribute RepetitionDetectionParams
- `vllm_repetitiondetectionparams___post_init___max_pattern_size_lt` (mech): module vllm has no attribute RepetitionDetectionParams
- `vllm_schedulerconfig___declarative___max_long_partial_prefills_ge` (mech): module vllm has no attribute SchedulerConfig
- `vllm_schedulerconfig___declarative___max_num_batched_tokens_ge` (mech): module vllm has no attribute SchedulerConfig
- `vllm_schedulerconfig___declarative___max_num_partial_prefills_ge` (mech): module vllm has no attribute SchedulerConfig
- `vllm_schedulerconfig___declarative___max_num_seqs_ge` (mech): module vllm has no attribute SchedulerConfig
- `vllm_schedulerconfig___declarative___stream_interval_ge` (mech): module vllm has no attribute SchedulerConfig
- `vllm_speculativeconfig___declarative___draft_tensor_parallel_size_ge` (mech): module vllm has no attribute SpeculativeConfig
- `vllm_speculativeconfig___declarative___num_speculative_tokens_gt` (mech): module vllm has no attribute SpeculativeConfig
- `vllm_speculativeconfig___declarative___prompt_lookup_max_ge` (mech): module vllm has no attribute SpeculativeConfig
- `vllm_speculativeconfig___declarative___prompt_lookup_min_ge` (mech): module vllm has no attribute SpeculativeConfig
- `vllm_structuredoutputsparams___post_init___count_gt` (mech): module vllm has no attribute StructuredOutputsParams
- ... (+65 more)
