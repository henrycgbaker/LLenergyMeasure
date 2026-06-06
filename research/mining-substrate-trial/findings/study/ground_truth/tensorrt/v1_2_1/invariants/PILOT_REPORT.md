# Pilot GT report - tensorrt 1.2.1 invariants (union + gate)

Round 0: union the 5 GT sources (mech, passA, passB, poc, prod) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:1.2.1, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 99 | 97 | 99 | 7 |
| passB | 100 | 98 | 100 | 2 |
| mech | 110 | 107 | 110 | 96 |
| poc | 92 | 90 | 0 | 0 |
| prod | 44 | 42 | 44 | 16 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **228**
- Tolerant keys (coarser, leaf+bucket): 157; of which **55** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **74**
- Probed candidates (native_type present, kwargs authored or synthesised): **353** (confirmed=124, failed=59, skipped=160, infra_error=10)
- Confirmations by probe provenance: **25 synthesised** by the gate, 99 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 74
- failed: 37
- infra_error: 2
- skipped: 115

## GT-growth vs PoC N=1 GT

PoC GT contributed **90** constraints. The gate-confirmed union grows GT by **37** confirmed constraints the PoC GT lacked:

- acceptance_length_threshold [numeric] = 0
- acceptance_window [numeric] = 0
- allreduce_strategy [membership] = [AUTO,LOWPRECISION,MINLATENCY,MNNVL,NCCL,NCCL_SYMMETRIC,ONESHOT,TWOSHOT,UB]
- batch_wait_max_tokens_ratio [numeric] = 0
- batch_wait_timeout_iters [numeric] = 0
- batch_wait_timeout_ms [numeric] = 0
- bert_attention_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- fp8_rowwise_gemm_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- gemm_allreduce_plugin [membership] = [,bfloat16,float16]
- gemm_swiglu_plugin [membership] = [,fp8]
- gpt_attention_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- identity_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- kv_transfer_sender_future_timeout_ms [numeric] = {gt=0}
- kv_transfer_timeout_ms [numeric] = {gt=0}
- layernorm_quantization_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- lora_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- low_latency_gemm_plugin [membership] = [,fp8]
- low_latency_gemm_swiglu_plugin [membership] = [,fp8]
- mamba_conv1d_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- max_batch_size [numeric] = 0
- max_gpu_total_bytes [numeric] = 0
- max_ngram_size [numeric] = 0
- max_verification_set_size [numeric] = 0
- max_window_size [numeric] = 0
- moe_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- nccl_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- per_worker_gpu_share [numeric] = 0
- qserve_gemm_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- rmsnorm_quantization_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- smooth_quant_gemm_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- speculative_config [presence] = True
- stream_interval [numeric] = 0
- temperature [numeric] = 0
- top_k [numeric] = 0
- top_p [numeric] = 0
- weight_only_groupwise_quant_matmul_plugin [membership] = [,auto,bfloat16,float16,float32,int32]
- weight_only_quant_matmul_plugin [membership] = [,auto,bfloat16,float16,float32,int32]

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| tensorrt_basellmargs_validate_dtype_dtype_lt | mech | tensorrt.BaseLlmArgs | error |
| tensorrt_basellmargs_validate_runtime_args_max_batch_size_gt | mech | tensorrt.BaseLlmArgs | no_op |
| tensorrt_cachetransceiverconfig___declarative___kv_transfer_sender_future_timeout_ms_gt | mech | tensorrt.CacheTransceiverConfig | no_op |
| tensorrt_cachetransceiverconfig___declarative___kv_transfer_timeout_ms_gt | mech | tensorrt.CacheTransceiverConfig | no_op |
| tensorrt_decodingbaseconfig_validate_draft_len_schedule_and_sort_draft_len_schedule_gt | mech | tensorrt.DecodingBaseConfig | error |
| tensorrt_decodingbaseconfig_validate_draft_len_schedule_and_sort_draft_len_schedule_lt | mech | tensorrt.DecodingBaseConfig | error |
| tensorrt_decodingbaseconfig_validate_draft_len_schedule_and_sort_draft_len_schedule_not_equal | mech | tensorrt.DecodingBaseConfig | error |
| tensorrt_guideddecodingparams__validate_num_guides_gt | mech | tensorrt.GuidedDecodingParams | error |
| tensorrt_kvcacheconfig_validate_free_gpu_memory_fraction_free_gpu_memory_fraction_le | mech | tensorrt.KvCacheConfig | no_op |
| tensorrt_kvcacheconfig_validate_max_attention_window_max_attention_window_le | mech | tensorrt.KvCacheConfig | error |
| tensorrt_rayplacementconfig_validate_ray_placement_has_pgs_not_equal | mech | tensorrt.RayPlacementConfig | error |
| tensorrt_rayplacementconfig_validate_ray_placement_placement_groups_not_equal | mech | tensorrt.RayPlacementConfig | error |
| tensorrt_samplingparams__validate_best_of_gt | mech | tensorrt.SamplingParams | no_op |
| tensorrt_samplingparams__validate_best_of_lt | mech | tensorrt.SamplingParams | error |
| tensorrt_samplingparams__validate_truncate_prompt_tokens_lt | mech | tensorrt.SamplingParams | error |
| tensorrt_torchcompileconfig_validate_torch_compile_max_num_streams_max_num_streams_lt | mech | tensorrt.TorchCompileConfig | error |
| tensorrt_torchllmargs_validate_attention_dp_config_config_lt | mech | tensorrt.TorchLlmArgs | error |
| tensorrt_torchllmargs_validate_cuda_graph_config_config_not_equal | mech | tensorrt.TorchLlmArgs | error |
| tensorrt_torchllmargs_validate_ray_placement_config_ray_placement_config_not_equal | mech | tensorrt.TorchLlmArgs | error |
| tensorrt_torchllmargs_validate_ray_worker_extension_cls_ray_worker_extension_cls_not_equal | mech | tensorrt.TorchLlmArgs | error |
| tensorrt_trtllmargs_validate_build_config_with_runtime_params_max_beam_width_not_equal | mech | tensorrt.TrtLlmArgs | no_op |
| tensorrt_trtllmargs_validate_build_config_with_runtime_params_max_input_len_not_equal | mech | tensorrt.TrtLlmArgs | no_op |
| tensorrt_trtllmargs_validate_build_config_with_runtime_params_max_num_tokens_gt | mech | tensorrt.TrtLlmArgs | no_op |
| tensorrt_trtllmargs_validate_build_config_with_runtime_params_max_seq_len_not_equal | mech | tensorrt.TrtLlmArgs | no_op |
| tensorrt_trtllmargs_validate_speculative_config_speculative_config_gt | mech | tensorrt.TrtLlmArgs | error |
| tensorrt_baseSparseAttentionConfig_from_dict_algorithm_dispatch | passA | tensorrt.BaseSparseAttentionConfig | error |
| tensorrt_baseSparseAttentionConfig_from_dict_algorithm_required | passA | tensorrt.BaseSparseAttentionConfig | no_op |
| tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_speculative_model_required_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_speculative_model_required_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_mtpDecodingConfig_num_nextn_predict_layers_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_ngramDecodingConfig_max_draft_len_and_matching_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passA | tensorrt.QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passA | tensorrt.SamplingParams | error |
| tensorrt_saveHiddenStatesDecodingConfig_backend_must_be_pytorch_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_torchLlmArgs_load_format_enum | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_trtLlmArgs_validate_kv_cache_dtype_must_be_auto | passA | tensorrt.TrtLlmArgs | no_op |
| tensorrt_baseLlmArgs_guided_decoding_backend_literal | passB | BaseLlmArgs | no_op |
| tensorrt_baseLlmArgs_orchestrator_type_literal | passB | BaseLlmArgs | no_op |
| tensorrt_baseLlmArgs_tokenizer_mode_literal | passB | BaseLlmArgs | no_op |
| tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed | passB | TorchLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_speculative_model_required_when_routed | passB | TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passB | TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_speculative_model_required_when_routed | passB | TorchLlmArgs | error |
| tensorrt_guidedDecodingParams_at_most_one_guide | passB | GuidedDecodingParams | no_op |
| tensorrt_mtpDecodingConfig_num_nextn_predict_layers_positive_when_routed | passB | TorchLlmArgs | error |
| tensorrt_ngramDecodingConfig_max_draft_len_and_matching_positive_when_routed | passB | TorchLlmArgs | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passB | QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passB | QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passB | SamplingParams | error |
| tensorrt_saveHiddenStatesDecodingConfig_backend_must_be_pytorch_when_routed | passB | TorchLlmArgs | error |
| tensorrt_torchLlmArgs_load_format_enum | passB | TorchLlmArgs | error |
| tensorrt_trtLlmArgs_validate_kv_cache_dtype_must_be_auto | passB | TrtLlmArgs | no_op |
| tensorrt_capacity_scheduler_policy_in_3_values | prod | tensorrt_llm.TrtLlmArgs | error |
| tensorrt_context_chunking_policy_in_2_values | prod | tensorrt_llm.TrtLlmArgs | error |
| tensorrt_raises_dtype_eq_bfloat16_dtype | prod | tensorrt_llm.BaseLlmArgs | no_op |
| tensorrt_raises_enable_build_cache_not_type_buildcacheconfig_enable_build_cache | prod | tensorrt_llm.TrtLlmArgs | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passA, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passB, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=poc, gateable=False, verdict=ungated, observed=n/a)

## Infra errors (could not run in container)

- `tensorrt_LLM_pytorch_rejects_trt_specific_kwargs` (passA): The following arguments are specific to TensorRT backend and cannot be used with PyTorch backend: ['enable_build_cache'].
Please use 'from tensorrt_llm._tensorr
- `tensorrt_warns_backend_in_lora_config_consistency` (prod): 2 validation errors for BaseLlmArgs
enable_lora
  Input should be a valid boolean, unable to interpret input [type=bool_parsing, input_value='x', input_type=str
- `tensorrt_warns_build_config_set_True_model_format_misc` (prod): 1 validation error for TrtLlmArgs
build_config
  Input should be a valid dictionary or instance of BuildConfig [type=model_type, input_value='x', input_type=str
- `tensorrt_warns_lora_config_set_True_lora_config_consistency` (prod): 1 validation error for BaseLlmArgs
lora_config
  Input should be a valid dictionary or instance of LoraConfig [type=model_type, input_value='x', input_type=str]
- `tensorrt_warns_lora_config_set_True_lora_config_consistency__2` (prod): 1 validation error for BaseLlmArgs
lora_config
  Input should be a valid dictionary or instance of LoraConfig [type=model_type, input_value='x', input_type=str]
- `tensorrt_warns_max_batch_size_set_True_build_config_with_runtime_params` (prod): 1 validation error for TrtLlmArgs
max_batch_size
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', inp
- `tensorrt_warns_max_beam_width_set_True_build_config_with_runtime_params` (prod): 1 validation error for TrtLlmArgs
max_beam_width
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', inp
- `tensorrt_warns_max_input_len_set_True_build_config_with_runtime_params` (prod): 1 validation error for TrtLlmArgs
max_input_len
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', inpu
- `tensorrt_warns_max_num_tokens_set_True_build_config_with_runtime_params` (prod): 1 validation error for TrtLlmArgs
max_num_tokens
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', inp
- `tensorrt_warns_max_seq_len_set_True_build_config_with_runtime_params` (prod): 1 validation error for TrtLlmArgs
max_seq_len
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='x', input_
