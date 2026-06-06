# Pilot GT report - vllm 0.21.0 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside vllm/vllm-openai:v0.21.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 122 | 122 | 122 | 105 |
| passB | 108 | 108 | 108 | 91 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **213**
- Tolerant keys (coarser, leaf+bucket): 154; of which **48** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **132**
- Probed candidates (native_type present, kwargs authored or synthesised): **230** (confirmed=145, failed=49, skipped=15, infra_error=21)
- Confirmations by probe provenance: **3 synthesised** by the gate, 142 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 132
- failed: 45
- infra_error: 21
- skipped: 15

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **132** confirmed constraints the PoC GT lacked:

- _api_process_rank [presence] = _api_process_rank >= _api_process_count
- active_iterations [numeric] = 1
- active_iterations [numeric] = {ge=1}
- all2all_backend [membership] = [allgather_reducescatter,deepep_high_throughput,deepep_low_latency,flashinfer_all2allv,flashinfer_nvlink_one_sided,flashinfer_nvlink_two_sided,mori,naive,nixl_ep,pplx]
- backend [membership] = [auto,guidance,lm-format-enforcer,outlines,xgrammar]
- backend [membership] = [ipc,nccl]
- cache_dtype [membership] = [auto,bfloat16,float16,fp8,fp8_ds_mla,fp8_e4m3,fp8_e5m2,fp8_inc,fp8_per_token_head,int8_per_token_head,nvfp4,turboquant_3bit_nc,turboquant_4bit_nc,turboquant_k3v4_nc,turboquant_k8v4]
- collect_detailed_traces [presence] = collect_detailed_traces set AND not otlp_traces_endpoint
- collect_detailed_traces [presence] = {require=otlp_traces_endpoint is set,when=collect_detailed_traces}
- communicator [membership] = [nixl,pynccl,torch_gloo,torch_nccl]
- compile_cache_save_format [membership] = [binary,unpacked]
- cpu_offload_gb [numeric] = {ge=0}
- custom_ops [presence] = custom_ops.count('none') + custom_ops.count('all') > 1
- custom_ops [presence] = op not in {all, none} AND op[0] not in {+, -}
- data_parallel_backend [membership] = [mp,ray]
- data_parallel_external_lb [presence] = data_parallel_size <= 1 AND data_parallel_external_lb
- data_parallel_size_local [numeric] = @data_parallel_size
- dcp_comm_backend [membership] = [a2a,ag_rs]
- detokenize [presence] = {require=detokenize is True,when=stop is non-empty}
- disable_additional_properties [presence] = disable_additional_properties AND backend != 'guidance'
- disable_additional_properties [presence] = {require=backend == guidance,when=disable_additional_properties}
- disable_any_whitespace [presence] = disable_any_whitespace AND backend not in {xgrammar, guidance}
- disable_any_whitespace [presence] = {require=backend in {xgrammar, guidance},when=disable_any_whitespace}
- ec_role [membership] = @get_args(ECRole)
- ec_role [presence] = ec_connector is not None AND ec_role is None
- ec_role [presence] = {require=ec_role is not None,when=ec_connector is not None}
- encoder_cudagraph_max_vision_items_per_batch [presence] = cudagraph_mm_encoder AND encoder_cudagraph_max_vision_items_per_batch < 0
- expert_placement_strategy [membership] = [linear,round_robin]
- flash_attn_version [membership] = [2,3,4]
- frequency_penalty [membership] = [-2,2]
- frequency_penalty [numeric] = {ge=-2,le=2}
- gpu_memory_utilization [membership] = (0, 1]
- gpu_memory_utilization [numeric] = {gt=0,le=1}
- kv_cache_metrics_sample [membership] = (0, 1]
- kv_cache_metrics_sample [numeric] = {gt=0,le=1}
- kv_load_failure_policy [membership] = [fail,recompute]
- kv_offloading_backend [membership] = [lmcache,native]
- kv_role [membership] = [,kv_both,kv_consumer,kv_producer]
- kv_role [membership] = [kv_both,kv_consumer,kv_producer]
- kv_role [presence] = kv_connector is not None AND kv_role is None
- ... (+92 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| vllm_cacheconfig_warning_calculate_kv_scales_deprecated | passA | vllm.config.CacheConfig | dormant_announced |
| vllm_compilationconfig_raises_invalid_mode_string | passA | vllm.config.CompilationConfig | error |
| vllm_deviceconfig_field_literal_device | passA | vllm.config.DeviceConfig | error |
| vllm_loraconfig_normalise_max_cpu_loras_unset | passA | vllm.config.LoRAConfig | dormant_silent |
| vllm_mambaconfig_raises_unknown_backend | passA | vllm.config.MambaConfig | error |
| vllm_multimodalconfig_raises_fp8_scale_path_without_fp8_dtype | passA | vllm.config.MultiModalConfig | error |
| vllm_observabilityconfig_raises_otlp_without_opentelemetry | passA | vllm.config.ObservabilityConfig | no_op |
| vllm_offloadconfig_raises_num_in_group_gt_group_size | passA | vllm.config.OffloadConfig | error |
| vllm_offloadconfig_raises_prefetch_step_lt_1 | passA | vllm.config.OffloadConfig | error |
| vllm_parallelconfig_raises_dcp_a2a_requires_dcp_gt_1 | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_num_redundant_experts_without_eplb | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_cpus_bad_syntax | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_nodes_empty | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_nodes_negative | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_warning_removed_all2all_backend_normalised | passA | vllm.config.ParallelConfig | dormant_silent |
| vllm_poolerconfig_field_literal_pooling_type | passA | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_raises_logit_scale_zero | passA | vllm.config.PoolerConfig | error |
| vllm_poolingparams_assert_output_kind_final_only | passA | vllm.PoolingParams | error |
| vllm_samplingparams_normalise_seed_eq_neg1 | passA | vllm.SamplingParams | dormant_silent |
| vllm_samplingparams_raises_logprob_token_ids_too_long | passA | vllm.SamplingParams | no_op |
| vllm_schedulerconfig_raises_long_prefill_token_threshold_gt_ref_max_model_len | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_raises_max_long_partial_prefills_gt_max_num_partial_prefills | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_raises_partial_prefills_requires_chunked_prefill | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_warning_batched_tokens_exceeds_seqs_times_model_len | passA | vllm.config.SchedulerConfig | dormant_announced |
| vllm_speculativeconfig_raises_num_speculative_tokens_le_0 | passA | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_raises_tensor_parallel_size_set | passA | vllm.config.SpeculativeConfig | error |
| vllm_attentionconfig_backend_enum_membership | passB | vllm.config.AttentionConfig | error |
| vllm_cacheconfig_cache_dtype_literal | passB | vllm.config.CacheConfig | error |
| vllm_cacheconfig_calculate_kv_scales_deprecation_warns | passB | vllm.config.CacheConfig | dormant_announced |
| vllm_deviceconfig_device_literal | passB | vllm.config.DeviceConfig | error |
| vllm_mambaconfig_backend_enum_membership | passB | vllm.config.MambaConfig | error |
| vllm_modelconfig_dtype_literal | passB | vllm.config.ModelConfig | error |
| vllm_modelconfig_tokenizer_mode_literal | passB | vllm.config.ModelConfig | no_op |
| vllm_multimodalconfig_fp8_scale_path_requires_fp8_dtype | passB | vllm.config.MultiModalConfig | error |
| vllm_onlinequantizationconfigargs_scheme_enum_membership | passB | vllm.config.quantization.OnlineQuantizationConfigArgs | error |
| vllm_parallelconfig_numa_bind_cpus_syntax | passB | vllm.config.ParallelConfig | error |
| vllm_poolerconfig_logit_bias_and_logit_mean_conflict | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_logit_scale_and_logit_sigma_conflict | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_logit_scale_not_zero | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_pooling_type_and_seq_pooling_type_conflict | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_pooling_type_membership | passB | vllm.config.PoolerConfig | error |
| vllm_poolingparams_output_kind_final_only | passB | vllm.PoolingParams | error |
| vllm_schedulerconfig_batched_tokens_ge_max_num_seqs | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_max_long_le_max_partial_prefills | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_partial_prefills_require_chunked | passB | vllm.config.SchedulerConfig | error |
| vllm_speculativeconfig_draft_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_rejection_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `vllm_eplbconfig_field_constraints_positive` (passA): 1 validation error for EPLBConfig
window_size
  Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
    For further information vi
- `vllm_eplbconfig_raises_async_requires_default_policy` (passA): 1 validation error for EPLBConfig
policy
  Input should be 'default' [type=literal_error, input_value='__not_default__', input_type=str]
    For further informa
- `vllm_llm_raises_single_process_data_parallel` (passA): LLM.__init__() missing 1 required positional argument: 'model'
- `vllm_loadconfig_field_constraints_prefetch_ge_1` (passA): 1 validation error for LoadConfig
safetensors_prefetch_num_threads
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_t
- `vllm_mambaconfig_raises_stochastic_rounding_requires_cuda` (passA): 1 validation error for MambaConfig
  Value error, Stochastic rounding for Mamba cache with triton backend requires compute capability 10.0 (data center Blackwel
- `vllm_parallelconfig_raises_elastic_ep_without_eplb` (passA): 1 validation error for ParallelConfig
  Value error, Elastic EP is only supported with enable_eplb=True. [type=value_error, input_value=ArgsKwargs((), {'enable_
- `vllm_parallelconfig_raises_enable_eplb_requires_cuda` (passA): 1 validation error for ParallelConfig
  Value error, EPLB requires tensor_parallel_size or data_parallel_size to be greater than 1, but got TP=1,DP=1. [type=val
- `vllm_parallelconfig_raises_numa_bind_fields_require_numa_bind` (passA): 1 validation error for ParallelConfig
  Value error, numa_bind_nodes and numa_bind_cpus require numa_bind=True. [type=value_error, input_value=ArgsKwargs((), {'
- `vllm_parallelconfig_raises_ray_nsight_without_ray` (passA): 1 validation error for ParallelConfig
  Value error, Unable to use nsight profiling unless workers run with Ray. [type=value_error, input_value=ArgsKwargs((), {
- `vllm_parallelconfig_raises_tp_not_divisible_by_dcp` (passA): 1 validation error for ParallelConfig
  Value error, tp_size=3 must be divisible bydcp_size=2. [type=value_error, input_value=ArgsKwargs((), {'tensor_p...text_p
- `vllm_poolerconfig_raises_both_logit_bias_and_logit_mean` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `logit_bias` and `logit_mean`. `logit_bias` is deprecated, use `logit_mean` instead. [type=va
- `vllm_poolerconfig_raises_both_logit_scale_and_logit_sigma` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `logit_scale` and `logit_sigma`. `logit_scale` is deprecated, use `logit_sigma` instead. [typ
- `vllm_poolerconfig_raises_both_pooling_type_and_seq_pooling_type` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `pooling_type` and `seq_pooling_type` [type=value_error, input_value=ArgsKwargs((), {'pooling
- `vllm_poolerconfig_raises_both_pooling_type_and_tok_pooling_type` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `pooling_type` and `tok_pooling_type` [type=value_error, input_value=ArgsKwargs((), {'pooling
- `vllm_repetitiondetectionparams_raises_pattern_sizes_invalid` (passA): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_schedulerconfig_field_constraints_ge_1` (passA): 1 validation error for SchedulerConfig
max_num_seqs
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]
    Fo
- `vllm_speculativeconfig_raises_synthetic_rates_without_synthetic_method` (passA): 1 validation error for SpeculativeConfig
  Value error, num_speculative_tokens must be provided with speculative model unless the draft model config contains an
- `vllm_structuredoutputsparams_raises_multiple_modes_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
- `vllm_structuredoutputsparams_raises_no_mode_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You must use one kind of structured outputs constraint but none are specified: {'json': None, 'reg
- `vllm_repetitiondetectionparams_pattern_sizes_valid` (passB): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_structuredoutputsparams_exactly_one_constraint` (passB): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
