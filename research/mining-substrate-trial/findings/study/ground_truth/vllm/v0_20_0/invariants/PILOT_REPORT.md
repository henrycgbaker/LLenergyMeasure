# Pilot GT report - vllm 0.20.0 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside vllm/vllm-openai:v0.20.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 100 | 100 | 100 | 97 |
| passB | 92 | 92 | 92 | 89 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **189**
- Tolerant keys (coarser, leaf+bucket): 145; of which **36** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **119**
- Probed candidates (native_type present, kwargs authored or synthesised): **192** (confirmed=121, failed=36, skipped=5, infra_error=30)
- Confirmations by probe provenance: **1 synthesised** by the gate, 120 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 119
- failed: 35
- infra_error: 30
- skipped: 5

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **119** confirmed constraints the PoC GT lacked:

- active_iterations [numeric] = {ge=1}
- backend [membership] = [auto,guidance,lm-format-enforcer,outlines,xgrammar]
- backend [membership] = [ipc,nccl]
- cache_dtype [membership] = @CacheDType (auto/float16/bfloat16/fp8/fp8_e4m3/fp8_e5m2/fp8_inc/fp8_ds_mla/turboquant_*/int8_per_token_head/fp8_per_token_head/nvfp4)
- collect_detailed_traces [presence] = collect_detailed_traces set AND not otlp_traces_endpoint
- collect_detailed_traces [presence] = {require=otlp_traces_endpoint is set,when=collect_detailed_traces}
- communicator [membership] = [nixl,pynccl,torch_gloo,torch_nccl]
- compile_cache_save_format [membership] = [binary,unpacked]
- count [numeric] = {ge=0}
- cpu_offload_gb [numeric] = {ge=0}
- custom_ops [presence] = custom_ops.count('none') + custom_ops.count('all') > 1
- custom_ops [presence] = each op must be 'all','none','+op' or '-op'
- data_parallel_backend [membership] = [mp,ray]
- data_parallel_external_lb [presence] = data_parallel_size <= 1 AND data_parallel_external_lb
- data_parallel_size_local [numeric] = @data_parallel_size
- dcp_comm_backend [membership] = [a2a,ag_rs]
- disable_additional_properties [presence] = disable_additional_properties AND backend != 'guidance'
- disable_additional_properties [presence] = {require=backend == guidance,when=disable_additional_properties}
- disable_any_whitespace [presence] = disable_any_whitespace AND backend not in {xgrammar, guidance}
- disable_any_whitespace [presence] = {require=backend in {xgrammar, guidance},when=disable_any_whitespace}
- ec_role [membership] = @get_args(ECRole)
- ec_role [presence] = ec_connector is not None AND ec_role is None
- ec_role [presence] = {require=ec_role is not None,when=ec_connector is not None}
- enable_eplb [presence] = enable_eplb AND not current_platform.is_cuda_alike()
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
- kv_role [presence] = kv_connector is not None AND kv_role is None
- kv_role [presence] = {require=kv_role is not None,when=kv_connector is not None}
- log_balancedness_interval [numeric] = 0
- log_balancedness_interval [numeric] = {gt=0}
- logit_bias [presence] = {require=not (logit_bias is not None and logit_mean is not None)}
- ... (+79 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| vllm_cacheconfig_warning_calculate_kv_scales_deprecated | passA | vllm.config.CacheConfig | dormant_announced |
| vllm_compilationconfig_raises_invalid_mode | passA | vllm.config.CompilationConfig | error |
| vllm_compilationconfig_raises_vllm_compile_invalid_backend | passA | vllm.config.CompilationConfig | error |
| vllm_deviceconfig_field_constraint_device_literal | passA | vllm.config.DeviceConfig | error |
| vllm_deviceconfig_raises_failed_to_infer_device | passA | vllm.config.DeviceConfig | dormant_silent |
| vllm_loraconfig_normalise_max_cpu_loras_unset | passA | vllm.config.LoRAConfig | dormant_silent |
| vllm_mambaconfig_raises_unknown_backend | passA | vllm.config.MambaConfig | error |
| vllm_modelconfig_raises_pipeline_parallel_unsupported_model | passA | vllm.config.ModelConfig | error |
| vllm_modelconfig_raises_sleep_mode_unsupported_platform | passA | vllm.config.ModelConfig | warn |
| vllm_modelconfig_raises_unknown_dtype | passA | vllm.config.ModelConfig | error |
| vllm_observabilityconfig_raises_otlp_without_opentelemetry | passA | vllm.config.ObservabilityConfig | no_op |
| vllm_offloadconfig_raises_num_in_group_gt_group_size | passA | vllm.config.OffloadConfig | error |
| vllm_offloadconfig_raises_prefetch_step_lt_1 | passA | vllm.config.OffloadConfig | error |
| vllm_parallelconfig_raises_dcp_a2a_requires_dcp_gt_1 | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_num_redundant_experts_without_eplb | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_cpus_bad_syntax | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_nodes_empty | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_nodes_negative | passA | vllm.config.ParallelConfig | error |
| vllm_samplingparams_normalise_seed_eq_neg1 | passA | vllm.SamplingParams | dormant_silent |
| vllm_vllmconfig_raises_torch_shm_requires_spawn | passA | vllm.config.VllmConfig | error |
| vllm_vllmconfig_warning_cudagraph_mode_incompatible_with_compile_mode | passA | vllm.config.VllmConfig | error |
| vllm_vllmconfig_warning_sequence_parallelism_requires_tp_gt_1 | passA | vllm.config.VllmConfig | error |
| vllm_attentionconfig_backend_enum_membership | passB | vllm.config.AttentionConfig | error |
| vllm_cacheconfig_cache_dtype_literal | passB | vllm.config.CacheConfig | error |
| vllm_cacheconfig_calculate_kv_scales_deprecation_warn | passB | vllm.config.CacheConfig | dormant_announced |
| vllm_mambaconfig_backend_enum_membership | passB | vllm.config.MambaConfig | error |
| vllm_onlinequantizationconfigargs_scheme_enum_membership | passB | vllm.config.quantization.OnlineQuantizationConfigArgs | error |
| vllm_parallelconfig_all2all_backend_literal | passB | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_numa_bind_cpus_syntax | passB | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_numa_bind_nodes_valid | passB | vllm.config.ParallelConfig | error |
| vllm_poolerconfig_pooling_type_excludes_seq_tok | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_pooling_type_must_be_known | passB | vllm.config.PoolerConfig | error |
| vllm_poolingparams_output_kind_final_only | passB | vllm.PoolingParams | error |
| vllm_schedulerconfig_batched_tokens_ge_max_num_seqs | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_max_long_le_max_partial_prefills | passB | vllm.config.SchedulerConfig | error |
| vllm_speculativeconfig_rejection_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `vllm_cpuplatform_raises_chunked_prefill_fp8_kv_cache` (passA): CpuPlatform() takes no arguments
- `vllm_cpuplatform_silent_fp8_kv_cache_fallback` (passA): CpuPlatform() takes no arguments
- `vllm_cpuplatform_silent_mla_disables_chunked_prefill` (passA): CpuPlatform() takes no arguments
- `vllm_eplbconfig_field_constraints_gt_0` (passA): 1 validation error for EPLBConfig
window_size
  Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
    For further information vi
- `vllm_eplbconfig_raises_async_requires_default_policy` (passA): 1 validation error for EPLBConfig
policy
  Input should be 'default' [type=literal_error, input_value='__not_default__', input_type=str]
    For further informa
- `vllm_loraconfig_raises_dual_stream_requires_cuda` (passA): 1 validation error for LoRAConfig
env_VLLM_LORA_ENABLE_DUAL_STREAM
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value='1', input_type=
- `vllm_mambaconfig_raises_stochastic_rounding_requires_cuda` (passA): 1 validation error for MambaConfig
  Value error, Stochastic rounding for Mamba cache with triton backend requires compute capability 10.0 (data center Blackwel
- `vllm_modelconfig_raises_attention_head_div_tensor_parallel` (passA): 1 validation error for ModelConfig
tensor_parallel_size
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value=3, input_type=int]
    For 
- `vllm_modelconfig_raises_max_model_len_exceeds_derived` (passA): 1 validation error for ModelConfig
env_VLLM_ALLOW_LONG_MAX_MODEL_LEN
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value='0', input_typ
- `vllm_modelconfig_warning_override_attention_dtype_non_rocm` (passA): 1 validation error for ModelConfig
platform
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value='cuda', input_type=str]
    For further
- `vllm_parallelconfig_raises_elastic_ep_without_eplb` (passA): 1 validation error for ParallelConfig
  Value error, Elastic EP is only supported with enable_eplb=True. [type=value_error, input_value=ArgsKwargs((), {'enable_
- `vllm_parallelconfig_raises_eplb_requires_expert_parallel` (passA): 1 validation error for ParallelConfig
  Value error, enable_expert_parallel must be True to use EPLB. [type=value_error, input_value=ArgsKwargs((), {'enable_e..
- `vllm_parallelconfig_raises_nsight_without_ray` (passA): 1 validation error for ParallelConfig
  Value error, Unable to use nsight profiling unless workers run with Ray. [type=value_error, input_value=ArgsKwargs((), {
- `vllm_parallelconfig_raises_tp_not_divisible_by_dcp` (passA): 1 validation error for ParallelConfig
  Value error, tp_size=2 must be divisible bydcp_size=3. [type=value_error, input_value=ArgsKwargs((), {'tensor_p...text_p
- `vllm_poolerconfig_raises_both_logit_bias_and_logit_mean` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `logit_bias` and `logit_mean`. `logit_bias` is deprecated, use `logit_mean` instead. [type=va
- `vllm_poolerconfig_raises_both_pooling_type_and_seq_pooling_type` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `pooling_type` and `seq_pooling_type` [type=value_error, input_value=ArgsKwargs((), {'pooling
- `vllm_poolerconfig_raises_both_pooling_type_and_tok_pooling_type` (passA): 1 validation error for PoolerConfig
tok_pooling_type
  Input should be 'ALL' or 'STEP' [type=literal_error, input_value='MEAN', input_type=str]
    For further 
- `vllm_repetitiondetectionparams_raises_pattern_sizes_invalid` (passA): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_schedulerconfig_field_constraints_ge_1` (passA): 3 validation errors for SchedulerConfig
max_model_len
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_seqs': 0}), input_type=ArgsKwargs]
  
- `vllm_schedulerconfig_raises_long_prefill_token_threshold_gt_ref_max_model_len` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...chunked_prefill': True}), inpu
- `vllm_schedulerconfig_raises_max_long_partial_prefills_gt_max_num_partial_prefills` (passA): 2 validation errors for SchedulerConfig
max_model_len
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...g_partial_prefills': 3}), input_ty
- `vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_model_len` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...hunked_prefill': False}), inpu
- `vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs` (passA): 2 validation errors for SchedulerConfig
max_model_len
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...: 1, 'max_num_seqs': 2}), input_ty
- `vllm_schedulerconfig_raises_partial_prefills_requires_chunked_prefill` (passA): 2 validation errors for SchedulerConfig
max_model_len
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...hunked_prefill': False}), input_ty
- `vllm_schedulerconfig_warning_batched_tokens_exceeds_seqs_times_model_len` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...1, 'max_model_len': 10}), inpu
- `vllm_structuredoutputsparams_raises_multiple_modes_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
- `vllm_structuredoutputsparams_raises_no_mode_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You must use one kind of structured outputs constraint but none are specified: {'json': None, 'reg
- `vllm_vllmconfig_warning_enforce_eager_disables_compile_and_cudagraph` (passA): 1 validation error for VllmConfig
enforce_eager
  Unexpected keyword argument [type=unexpected_keyword_argument, input_value=True, input_type=bool]
    For furt
- `vllm_repetitiondetectionparams_pattern_sizes_valid` (passB): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_structuredoutputsparams_exactly_one_constraint` (passB): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
