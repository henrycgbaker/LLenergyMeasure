# Pilot GT report - vllm 0.18.1 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside vllm/vllm-openai:v0.18.1, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 76 | 76 | 76 | 76 |
| passB | 69 | 69 | 69 | 69 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **145**
- Tolerant keys (coarser, leaf+bucket): 111; of which **32** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **94**
- Probed candidates (native_type present, kwargs authored or synthesised): **145** (confirmed=94, failed=23, skipped=15, infra_error=13)
- Confirmations by probe provenance: **2 synthesised** by the gate, 92 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 94
- failed: 23
- infra_error: 13
- skipped: 15

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **94** confirmed constraints the PoC GT lacked:

- _api_process_rank [presence] = @_api_process_count
- active_iterations [numeric] = {ge=1}
- backend [membership] = [auto,guidance,lm-format-enforcer,outlines,xgrammar]
- backend [membership] = [ipc,nccl]
- cache_dtype [membership] = [auto, fp8, fp8_e4m3, fp8_e5m2, fp8_inc, fp8_ds_mla]
- collect_detailed_traces [presence] = collect_detailed_traces set AND not otlp_traces_endpoint
- collect_detailed_traces [presence] = {require=otlp_traces_endpoint is set,when=collect_detailed_traces}
- compile_cache_save_format [membership] = [binary, unpacked]
- compile_cache_save_format [membership] = [binary,unpacked]
- custom_ops [presence] = custom_ops.count("none") + custom_ops.count("all") <= 1
- data_parallel_backend [membership] = [mp,ray]
- data_parallel_external_lb [presence] = data_parallel_size <= 1 AND data_parallel_external_lb is True
- data_parallel_size_local [numeric] = @data_parallel_size
- dcp_comm_backend [membership] = [a2a,ag_rs]
- disable_additional_properties [presence] = disable_additional_properties AND backend != guidance
- disable_additional_properties [presence] = {require=backend == guidance,when=disable_additional_properties}
- disable_any_whitespace [presence] = disable_any_whitespace AND backend not in (xgrammar, guidance)
- disable_any_whitespace [presence] = {require=backend in {xgrammar, guidance},when=disable_any_whitespace}
- ec_connector [presence] = ec_connector set AND ec_role is None
- ec_role [membership] = get_args(ECRole)
- ec_role [presence] = {require=ec_role is not None,when=ec_connector is not None}
- expert_placement_strategy [membership] = [linear,round_robin]
- flash_attn_version [membership] = [2,3,4]
- frequency_penalty [numeric] = [-2.0, 2.0]
- frequency_penalty [numeric] = {ge=-2,le=2}
- gpu_memory_utilization [numeric] = {gt=0,le=1}
- gpu_memory_utilization [presence] = gt 0, le 1
- kv_cache_metrics_sample [numeric] = {gt=0,le=1}
- kv_connector [presence] = kv_connector set AND kv_role is None
- kv_load_failure_policy [membership] = [fail,recompute]
- kv_offloading_backend [membership] = [lmcache,native]
- kv_role [membership] = get_args(KVRole)
- kv_role [presence] = {require=kv_role is not None,when=kv_connector is not None}
- log_balancedness_interval [presence] = log_balancedness is True AND log_balancedness_interval <= 0
- log_balancedness_interval [presence] = {require=log_balancedness_interval > 0,when=log_balancedness}
- logprobs [numeric] = 0
- lora_dtype [membership] = [auto,bfloat16,float16]
- mamba_block_size [numeric] = {allow_none=True,gt=0}
- mamba_block_size [presence] = 0
- mamba_cache_dtype [membership] = [auto,float16,float32]
- ... (+54 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| vllm_compilationconfig_raises_invalid_mode | passA | vllm.config.CompilationConfig | error |
| vllm_loraconfig_normalise_max_cpu_loras_unset | passA | vllm.config.LoRAConfig | dormant_silent |
| vllm_observabilityconfig_raises_otlp_without_opentelemetry | passA | vllm.config.ObservabilityConfig | no_op |
| vllm_parallelconfig_normalise_pplx_backend_removed | passA | vllm.config.ParallelConfig | dormant_silent |
| vllm_parallelconfig_raises_dcp_a2a_requires_dcp_gt_1 | passA | vllm.config.ParallelConfig | error |
| vllm_samplingparams_normalise_seed_eq_neg1 | passA | vllm.SamplingParams | dormant_silent |
| vllm_schedulerconfig_raises_long_prefill_token_threshold_gt_ref_max_model_len | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_raises_max_long_partial_prefills_gt_max_num_partial_prefills | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_raises_partial_prefills_requires_chunked_prefill | passA | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_warning_batched_tokens_exceeds_seqs_times_model_len | passA | vllm.config.SchedulerConfig | dormant_announced |
| vllm_speculativeconfig_raises_num_speculative_tokens_le_0 | passA | vllm.config.SpeculativeConfig | error |
| vllm_attentionconfig_backend_enum_membership | passB | vllm.config.AttentionConfig | error |
| vllm_cacheconfig_cache_dtype_literal | passB | vllm.config.CacheConfig | error |
| vllm_deviceconfig_device_literal_skipvalidation | passB | vllm.config.DeviceConfig | error |
| vllm_poolingparams_output_kind_final_only | passB | vllm.PoolingParams | error |
| vllm_schedulerconfig_batched_tokens_ge_max_num_seqs | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_max_long_le_max_partial_prefills | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_partial_prefills_require_chunked | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_policy_literal | passB | vllm.config.SchedulerConfig | error |
| vllm_schedulerconfig_runner_type_literal | passB | vllm.config.SchedulerConfig | error |
| vllm_speculativeconfig_method_literal | passB | vllm.config.SpeculativeConfig | error |
| vllm_speculativeconfig_rejection_sample_method_literal | passB | vllm.config.SpeculativeConfig | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `vllm_compilationconfig_raises_invalid_custom_op_syntax` (passA): 1 validation error for CompilationConfig
  Value error, Invalid syntax 'rotary_embedding' for custom op, must be 'all', 'none', '+op' or '-op' (where 'op' is th
- `vllm_eplbconfig_raises_async_requires_default_policy` (passA): 1 validation error for EPLBConfig
policy
  Input should be 'default' [type=literal_error, input_value='__not_default__', input_type=str]
    For further informa
- `vllm_parallelconfig_raises_elastic_ep_without_eplb` (passA): 1 validation error for ParallelConfig
  Value error, Elastic EP is only supported with enable_eplb=True. [type=value_error, input_value=ArgsKwargs((), {'enable_
- `vllm_parallelconfig_raises_nsight_without_ray` (passA): 1 validation error for ParallelConfig
  Value error, Unable to use nsight profiling unless workers run with Ray. [type=value_error, input_value=ArgsKwargs((), {
- `vllm_parallelconfig_raises_tp_not_divisible_by_dcp` (passA): 1 validation error for ParallelConfig
  Value error, tp_size=2 must be divisible bydcp_size=3. [type=value_error, input_value=ArgsKwargs((), {'tensor_p...text_p
- `vllm_poolerconfig_raises_both_pooling_type_and_seq_pooling_type` (passA): 1 validation error for PoolerConfig
  Value error, Cannot set both `pooling_type` and `seq_pooling_type` [type=value_error, input_value=ArgsKwargs((), {'pooling
- `vllm_poolerconfig_raises_both_pooling_type_and_tok_pooling_type` (passA): 1 validation error for PoolerConfig
tok_pooling_type
  Input should be 'ALL' or 'STEP' [type=literal_error, input_value='MEAN', input_type=str]
    For further 
- `vllm_repetitiondetectionparams_raises_pattern_sizes_invalid` (passA): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_schedulerconfig_field_constraints_ge_1` (passA): 1 validation error for SchedulerConfig
max_num_seqs
  Input should be greater than or equal to 1 [type=greater_than_equal, input_value=0, input_type=int]
    Fo
- `vllm_structuredoutputsparams_raises_multiple_modes_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': No
- `vllm_structuredoutputsparams_raises_no_mode_set` (passA): 1 validation error for StructuredOutputsParams
  Value error, You must use one kind of structured outputs constraint but none are specified: {'json': None, 'reg
- `vllm_repetitiondetectionparams_pattern_sizes_valid` (passB): 1 validation error for RepetitionDetectionParams
  Value error, max_pattern_size, min_pattern_size must be >=0, with min_pattern_size <= max_pattern_size. Set b
- `vllm_structuredoutputsparams_exactly_one_constraint` (passB): 1 validation error for StructuredOutputsParams
  Value error, You can only use one kind of structured outputs constraint but multiple are specified: {'json': '{
