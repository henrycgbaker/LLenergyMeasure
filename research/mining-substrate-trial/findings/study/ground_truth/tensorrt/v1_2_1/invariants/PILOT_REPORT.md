# Pilot GT report - tensorrt 1.2.1 invariants (union + gate)

Round 0 pilot: union the 4 GT sources by tolerant identity (leaf_native_field, coarse_predicate_bucket), runtime-gate every kwargs-bearing candidate in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:1.2.1, keep gate-confirmed as GT.

## Per-source candidate counts

| source | raw candidates | tolerant keys | gateable | unique tolerant keys |
|---|---|---|---|---|
| passA | 99 | 86 | 99 | 4 |
| passB | 100 | 88 | 100 | 5 |
| mech | 110 | 95 | 110 | 51 |
| poc | 92 | 80 | 0 | 0 |

## Union + gate

- Union size (distinct tolerant identities across 4 sources): **144**
- Gate-confirmed tolerant identities: **46**
- Probed candidates (native_type present, kwargs authored or synthesised): **309** (confirmed=93, failed=56, skipped=160, infra_error=0)
- Confirmations by probe provenance: **25 synthesised** by the gate, 68 from hand-authored kwargs

Group status breakdown (per tolerant identity):

- confirmed: 46
- failed: 25
- skipped: 73

## GT-growth vs PoC N=1 GT

PoC GT contributed **80** tolerant identities. The gate-confirmed union grows GT by **8** confirmed identities the PoC GT lacked:

- allreduce_strategy [membership]
- bert_attention_plugin [membership]
- gemm_allreduce_plugin [membership]
- gemm_swiglu_plugin [membership]
- kv_transfer_sender_future_timeout_ms [numeric]
- kv_transfer_timeout_ms [numeric]
- low_latency_gemm_plugin [membership]
- low_latency_gemm_swiglu_plugin [membership]

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
| tensorrt_LLM_pytorch_rejects_trt_specific_kwargs | passA | tensorrt.LLM | error |
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

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passA, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passB, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=poc, gateable=False, verdict=ungated, observed=n/a)
