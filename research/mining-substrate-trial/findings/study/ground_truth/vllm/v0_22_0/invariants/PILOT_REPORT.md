# Pilot GT report - vllm 0.22.0 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside vllm/vllm-openai:v0.22.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 109 | 109 | 109 | 105 |
| passB | 99 | 99 | 99 | 95 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **204**
- Tolerant keys (coarser, leaf+bucket): 159; of which **37** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **118**
- Probed candidates (native_type present, kwargs authored or synthesised): **208** (confirmed=121, failed=34, skipped=30, infra_error=23)
- Confirmations by probe provenance: **3 synthesised** by the gate, 118 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 118
- failed: 33
- infra_error: 23
- skipped: 30

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **118** confirmed constraints the PoC GT lacked:

- active_iterations [numeric] = {ge=1}
- all2all_backend [membership] = [allgather_reducescatter,deepep_high_throughput,deepep_low_latency,flashinfer_all2allv,flashinfer_nvlink_one_sided,flashinfer_nvlink_two_sided,mori,naive,nixl_ep,pplx]
- backend [membership] = [auto,guidance,lm-format-enforcer,outlines,xgrammar]
- bad_words [presence] = any(not bw for bw in bad_words)
- bad_words [presence] = {require=no element of bad_words is an empty string}
- cache_dtype [membership] = [auto,bfloat16,float16,fp8,fp8_ds_mla,fp8_e4m3,fp8_e5m2,fp8_inc,fp8_per_token_head,int8_per_token_head,nvfp4,turboquant_3bit_nc,turboquant_4bit_nc,turboquant_k3v4_nc,turboquant_k8v4]
- collect_detailed_traces [membership] = [all,model,worker]
- collect_detailed_traces [presence] = collect_detailed_traces set AND not otlp_traces_endpoint
- communicator [membership] = [nixl,pynccl,torch_gloo,torch_nccl]
- compile_cache_save_format [membership] = [binary,unpacked]
- cpu_offload_gb [numeric] = 0
- cpu_offload_gb [numeric] = {ge=0}
- custom_ops [presence] = custom_ops.count('none') + custom_ops.count('all') > 1
- data_parallel_backend [membership] = [mp,ray]
- data_parallel_external_lb [presence] = data_parallel_size <= 1 AND data_parallel_external_lb
- data_parallel_size_local [numeric] = @data_parallel_size
- dcp_comm_backend [membership] = [a2a,ag_rs]
- dcp_comm_backend [presence] = dcp_comm_backend == 'a2a' AND decode_context_parallel_size <= 1
- disable_additional_properties [presence] = disable_additional_properties AND backend != 'guidance'
- disable_additional_properties [presence] = {require=backend == guidance,when=disable_additional_properties}
- disable_any_whitespace [presence] = disable_any_whitespace AND backend not in {xgrammar, guidance}
- disable_any_whitespace [presence] = {require=backend in {xgrammar, guidance},when=disable_any_whitespace}
- ec_role [membership] = [,ec_both,ec_consumer,ec_producer]
- ec_role [presence] = ec_connector is not None AND ec_role is None
- ec_role [presence] = {require=ec_role is not None,when=ec_connector is not None}
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
- linear_backend [membership] = [aiter,auto,conch,cutlass,deep_gemm,emulation,exllama,fbgemm,flashinfer_cudnn,flashinfer_cutlass,flashinfer_trtllm,machete,marlin,torch,triton]
- log_balancedness_interval [numeric] = 0
- ... (+78 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| vllm_compilationconfig_raises_invalid_mode_string | passA | vllm.config.CompilationConfig | error |
| vllm_deviceconfig_field_constraint_device_literal | passA | vllm.config.DeviceConfig | error |
| vllm_loraconfig_normalise_max_cpu_loras_unset | passA | vllm.config.LoRAConfig | dormant_silent |
| vllm_modelconfig_raises_deprecated_quantization_method | passA | vllm.config.ModelConfig | error |
| vllm_modelconfig_raises_sleep_mode_unsupported_platform | passA | vllm.config.ModelConfig | dormant_announced |
| vllm_observabilityconfig_raises_otlp_without_opentelemetry | passA | vllm.config.ObservabilityConfig | no_op |
| vllm_offloadconfig_raises_num_in_group_gt_group_size | passA | vllm.config.OffloadConfig | error |
| vllm_offloadconfig_raises_prefetch_step_lt_1 | passA | vllm.config.OffloadConfig | error |
| vllm_parallelconfig_raises_num_redundant_experts_without_eplb | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_cpus_bad_syntax | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_nodes_empty | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_raises_numa_bind_nodes_negative | passA | vllm.config.ParallelConfig | error |
| vllm_parallelconfig_warning_removed_all2all_backend_normalised | passA | vllm.config.ParallelConfig | dormant_announced |
| vllm_poolerconfig_raises_logit_scale_zero | passA | vllm.config.PoolerConfig | error |
| vllm_samplingparams_normalise_seed_eq_neg1 | passA | vllm.SamplingParams | dormant_silent |
| vllm_speculativeconfig_field_constraint_num_speculative_tokens_gt_0 | passA | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_field_constraints_ge_1 | passA | vllm.config.SpeculativeConfig | error |
| vllm_attentionconfig_backend_enum_membership | passB | vllm.config.AttentionConfig | error |
| vllm_cacheconfig_cache_dtype_literal | passB | vllm.config.CacheConfig | error |
| vllm_mambaconfig_backend_enum_membership | passB | vllm.config.MambaConfig | error |
| vllm_modelconfig_dtype_literal | passB | vllm.config.ModelConfig | error |
| vllm_modelconfig_tokenizer_mode_literal | passB | vllm.config.ModelConfig | no_op |
| vllm_poolerconfig_logit_bias_mean_conflict | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_pooling_type_literal | passB | vllm.config.PoolerConfig | error |
| vllm_poolerconfig_pooling_type_mutual_exclusion | passB | vllm.config.PoolerConfig | error |
| vllm_poolingparams_output_kind_final_only | passB | vllm.PoolingParams | error |
| vllm_quantizationconfigargs_online_shorthand_spec | passB | vllm.config.quantization.QuantizationConfigArgs | error |
| vllm_quantspec_weight_key_allowlist | passB | vllm.config.quantization.QuantSpec | error |
| vllm_schedulerconfig_batched_tokens_ge_max_num_seqs | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_max_long_le_max_partial_prefills | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_policy_literal | passB | vllm.config.SchedulerConfig | error |
| vllm_speculativeconfig_draft_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_rejection_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `vllm_eplbconfig_field_constraints` (passA): 1 validation error for EPLBConfig
window_size
  Input should be greater than 0 [type=greater_than, input_value=0, input_type=int]
    For further information vi
- `vllm_loadconfig_field_constraints_safetensors_prefetch_ge_1` (passA): 1 validation error for LoadConfig
safetensors_prefetch_num_threads
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_t
- `vllm_parallelconfig_raises_elastic_ep_without_eplb` (passA): 1 validation error for ParallelConfig
  Value error, Elastic EP is only supported with enable_eplb=True. [type=value_error, input_value=ArgsKwargs((), {'enable_
- `vllm_parallelconfig_raises_nsight_without_ray` (passA): 1 validation error for ParallelConfig
  Value error, Unable to use nsight profiling unless workers run with Ray. [type=value_error, input_value=ArgsKwargs((), {
- `vllm_parallelconfig_raises_numa_bind_targets_require_flag` (passA): 1 validation error for ParallelConfig
  Value error, numa_bind_nodes and numa_bind_cpus require numa_bind=True. [type=value_error, input_value=ArgsKwargs((), {'
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
- `vllm_prefetchoffloadconfig_field_constraints` (passA): 1 validation error for PrefetchOffloadConfig
offload_num_in_group
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_ty
- `vllm_repetitiondetectionparams_raises_pattern_sizes_invalid` (passA): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_schedulerconfig_field_constraints_ge_1` (passA): 3 validation errors for SchedulerConfig
max_model_len
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_seqs': 0}), input_type=ArgsKwargs]
  
- `vllm_schedulerconfig_raises_long_prefill_token_threshold_gt_ref_max_model_len` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...chunked_prefill': True}), inpu
- `vllm_schedulerconfig_raises_max_long_partial_prefills_gt_max_num_partial_prefills` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_... 'max_model_len': 2048}), inpu
- `vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_model_len` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...hunked_prefill': False}), inpu
- `vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_... 2, 'max_model_len': 1}), inpu
- `vllm_schedulerconfig_raises_partial_prefills_requires_chunked_prefill` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_... 'max_model_len': 2048}), inpu
- `vllm_schedulerconfig_warning_batched_tokens_exceeds_seqs_times_model_len` (passA): 1 validation error for SchedulerConfig
is_encoder_decoder
  Field required [type=missing, input_value=ArgsKwargs((), {'max_num_...1, 'max_model_len': 10}), inpu
- `vllm_structuredoutputsparams_raises_multiple_modes_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
- `vllm_structuredoutputsparams_raises_no_mode_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You must use one kind of structured outputs constraint but none are specified: {'json': None, 'reg
- `vllm_repetitiondetectionparams_pattern_sizes_valid` (passB): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_structuredoutputsparams_exactly_one_constraint` (passB): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
