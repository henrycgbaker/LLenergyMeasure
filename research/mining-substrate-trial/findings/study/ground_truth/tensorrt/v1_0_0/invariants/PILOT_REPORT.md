# Pilot GT report - tensorrt 1.0.0 invariants (union + gate)

Round 0: union the 3 GT sources (mech, passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:1.0.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 78 | 75 | 78 | 18 |
| passB | 66 | 63 | 66 | 6 |
| mech | 43 | 43 | 35 | 42 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **123**
- Tolerant keys (coarser, leaf+bucket): 93; of which **22** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **21**
- Probed candidates (native_type present, kwargs authored or synthesised): **179** (confirmed=37, failed=53, skipped=75, infra_error=14)
- Confirmations by probe provenance: **2 synthesised** by the gate, 35 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 21
- failed: 37
- infra_error: 14
- skipped: 43
- unreachable: 8

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **21** confirmed constraints the PoC GT lacked:

- allreduce_strategy [membership] = [AUTO,LOWPRECISION,MINLATENCY,MNNVL,NCCL,ONESHOT,TWOSHOT,UB]
- backend [membership] = [CUTEDSL,CUTLASS,DEEPGEMM,TRTLLM,VANILLA,WIDEEP]
- backend [membership] = [DEFAULT,MPI,NIXL,UCX]
- best_of [presence] = {best_of_gt_1_and_greedy_and_env_unset=True}
- capacity_scheduler_policy [membership] = [GUARANTEED_NO_EVICT,MAX_UTILIZATION,STATIC_BATCH]
- context_chunking_policy [membership] = [EQUAL_PROGRESS,FIRST_COME_FIRST_SERVED]
- device [membership] = [cpu,cuda]
- device [presence] = [cpu,cuda]
- guided_decoding_backend [membership] = [llguidance,xgrammar]
- lora_ckpt_source [membership] = [hf,nemo]
- max_batch_size [numeric] = {ge=0}
- max_ngram_size [numeric] = 0
- max_ngram_size [numeric] = {gt=0}
- max_num_streams [numeric] = {ge=1}
- max_verification_set_size [numeric] = 0
- max_verification_set_size [numeric] = {gt=0}
- max_window_size [numeric] = 0
- max_window_size [numeric] = {gt=0}
- model [type] = [Path,str]
- tokenizer_mode [membership] = [auto,slow]
- truncate_prompt_tokens [numeric] = {ge=1}

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| tensorrt__autodeployllmargs_mla_backend_in_1_values | mech | tensorrt_llm._AutoDeployLlmArgs | error |
| tensorrt__autodeployllmargs_model_factory_in_2_values | mech | tensorrt_llm._AutoDeployLlmArgs | error |
| tensorrt_autodeploy_free_mem_ratio_out_of_range | mech | tensorrt_llm._AutoDeployLlmArgs | error |
| tensorrt_basellmargs_load_format_in_2_values | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_basellmargs_tokenizer_mode_in_2_values | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_batching_type_in_2_values | mech | tensorrt_llm.TrtLlmArgs | error |
| tensorrt_build_cache_max_records_ge_1 | mech | tensorrt_llm.llmapi.build_cache.BuildCache | error |
| tensorrt_quant_config_kv_cache_quant_algo_in_allowlist | mech | tensorrt_llm.QuantConfig | no_op |
| tensorrt_quant_config_quant_algo_in_allowlist | mech | tensorrt_llm.QuantConfig | no_op |
| tensorrt_raises_cuda_graph_max_batch_size_lt_0_cuda_graph_max_batch_size | mech | tensorrt_llm.TorchLlmArgs | error |
| tensorrt_raises_cuda_graph_max_batch_size_ne_0_cuda_graph_config | mech | tensorrt_llm.TorchLlmArgs | error |
| tensorrt_raises_dtype_eq_bfloat16_dtype | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_raises_enable_build_cache_not_type_buildcacheconfig_enable_build_cache | mech | tensorrt_llm.TrtLlmArgs | error |
| tensorrt_raises_max_batch_size_set_True_build_config_with_runtime_params | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_raises_max_num_tokens_set_True_build_config_with_runtime_params | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_raises_model_not_type_model | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_raises_moe_load_balancer_type_str_moe_load_balancer | mech | tensorrt_llm.TorchLlmArgs | error |
| tensorrt_raises_speculative_config_set_True_speculative_config | mech | tensorrt_llm.BaseLlmArgs | error |
| tensorrt_torch_llm_load_format_invalid | mech | tensorrt_llm.TorchLlmArgs | error |
| tensorrt_autoDecodingConfig_backend_must_be_torch_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_baseLlmArgs_load_format_literal | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_batchingType_enum | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch | passA | tensorrt.DecodingBaseConfig | error |
| tensorrt_draftTargetDecodingConfig_backend_len_and_model_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_speculative_model_dir_required_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_validate_speculative_model_dir_required | passA | tensorrt.EagleDecodingConfig | no_op |
| tensorrt_guidedDecodingParams_at_most_one_guide | passA | tensorrt.GuidedDecodingParams | error |
| tensorrt_lookaheadDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_medusaDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_mtpDecodingConfig_num_nextn_predict_layers_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_ngramDecodingConfig_backend_and_lengths_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_pluginConfig_gemm_plugin_allowlist | passA | tensorrt.PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passA | tensorrt.QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passA | tensorrt.SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_trtLlmArgs_validate_kv_cache_dtype_must_be_auto | passA | tensorrt.TrtLlmArgs | warn |
| tensorrt_userProvidedDecodingConfig_backend_must_be_torch_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_baseLlmArgs_load_format_literal | passB | tensorrt_llm.llmapi.TrtLlmArgs | error |
| tensorrt_batchingType_enum | passB | tensorrt_llm.llmapi.llm_args.BatchingType | error |
| tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_speculative_model_required_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_speculative_model_required_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_validate_draft_model_required | passB | tensorrt_llm.llmapi.EagleDecodingConfig | error |
| tensorrt_mtpDecodingConfig_num_nextn_predict_layers_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_ngramDecodingConfig_max_draft_len_and_matching_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passB | tensorrt_llm.llmapi.QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passB | tensorrt_llm.llmapi.QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passB | tensorrt_llm.llmapi.SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_trtLlmArgs_validate_kv_cache_dtype_must_be_auto | passB | tensorrt_llm.llmapi.TrtLlmArgs | warn |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passB, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)

## Infra errors (could not run in container)

- `tensorrt_warns_backend_in_lora_config_consistency` (mech): 2 validation errors for BaseLlmArgs
enable_lora
  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='x', input_type=str
- `tensorrt_warns_build_config_set_True_model_format_misc` (mech): 1 validation error for BaseLlmArgs
build_config
  Extra inputs are not permitted [type=extra_forbidden, input_value='x', input_type=str]
    For further informa
- `tensorrt_warns_lora_config_set_True_lora_config_consistency` (mech): 1 validation error for BaseLlmArgs
lora_config
  Input should be a dictionary or an instance of LoraConfig [type=dataclass_type, input_value='x', input_type=str
- `tensorrt_warns_max_batch_size_set_True_set_runtime_knobs_from_build_config` (mech): 2 validation errors for BaseLlmArgs
max_batch_size
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', i
- `tensorrt_warns_max_beam_width_set_True_build_config_with_runtime_params` (mech): 1 validation error for BaseLlmArgs
max_beam_width
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', in
- `tensorrt_warns_max_beam_width_set_True_set_runtime_knobs_from_build_config` (mech): 2 validation errors for BaseLlmArgs
max_beam_width
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', i
- `tensorrt_warns_max_input_len_set_True_build_config_with_runtime_params` (mech): 1 validation error for BaseLlmArgs
max_input_len
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', inp
- `tensorrt_warns_max_input_len_set_True_set_runtime_knobs_from_build_config` (mech): 2 validation errors for BaseLlmArgs
max_input_len
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', in
- `tensorrt_warns_max_lora_rank_set_True_lora_config_consistency` (mech): 2 validation errors for BaseLlmArgs
lora_config
  Input should be a dictionary or an instance of LoraConfig [type=dataclass_type, input_value='x', input_type=st
- `tensorrt_warns_max_num_tokens_set_True_set_runtime_knobs_from_build_config` (mech): 2 validation errors for BaseLlmArgs
max_num_tokens
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', i
- `tensorrt_warns_max_seq_len_set_True_build_config_with_runtime_params` (mech): 1 validation error for BaseLlmArgs
max_seq_len
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', input
- `tensorrt_warns_max_seq_len_set_True_set_runtime_knobs_from_build_config` (mech): 2 validation errors for BaseLlmArgs
max_seq_len
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', inpu
- `tensorrt_LLM_pytorch_rejects_trt_specific_kwargs` (passA): The following arguments are specific to TensorRT backend and cannot be used with PyTorch backend: ['enable_build_cache'].
Please use 'from tensorrt_llm._tensorr
- `tensorrt_LLM_rejects_unknown_kwarg` (passA): LLM got invalid argument: __not_a_real_arg__
