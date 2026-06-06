# Pilot GT report - tensorrt 0.21.0 invariants (union + gate)

Round 0: union the 4 GT sources (mech, passA, passB, poc) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:0.21.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 76 | 74 | 76 | 6 |
| passB | 64 | 62 | 64 | 30 |
| mech | 56 | 55 | 56 | 55 |
| poc | 75 | 73 | 0 | 5 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **164**
- Tolerant keys (coarser, leaf+bucket): 109; of which **44** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **18**
- Probed candidates (native_type present, kwargs authored or synthesised): **196** (confirmed=26, failed=45, skipped=116, infra_error=9)
- Confirmations by probe provenance: **8 synthesised** by the gate, 18 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 18
- failed: 31
- infra_error: 9
- skipped: 101
- unreachable: 5

## GT-growth vs PoC N=1 GT

PoC GT contributed **73** constraints. The gate-confirmed union grows GT by **6** confirmed constraints the PoC GT lacked:

- BatchingType [membership] = [INFLIGHT,STATIC]
- lora_ckpt_source [membership] = [hf,nemo]
- max_ngram_size [numeric] = 0
- max_verification_set_size [numeric] = 0
- max_window_size [numeric] = 0
- truncate_prompt_tokens [numeric] = {ge=1}

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| tensorrt_basellmargs_validate_dtype_dtype_lt | mech | tensorrt.BaseLlmArgs | error |
| tensorrt_basellmargs_validate_speculative_config_speculative_config_gt | mech | tensorrt.BaseLlmArgs | error |
| tensorrt_guideddecodingparams__validate_num_guides_gt | mech | tensorrt.GuidedDecodingParams | error |
| tensorrt_samplingparams__validate_best_of_gt | mech | tensorrt.SamplingParams | no_op |
| tensorrt_samplingparams__validate_best_of_lt | mech | tensorrt.SamplingParams | error |
| tensorrt_samplingparams__validate_truncate_prompt_tokens_lt | mech | tensorrt.SamplingParams | error |
| tensorrt_torchllmargs_validate_cuda_graph_config_cuda_graph_batch_sizes_not_equal | mech | tensorrt.TorchLlmArgs | error |
| tensorrt_torchllmargs_validate_cuda_graph_max_batch_size_cuda_graph_max_batch_size_lt | mech | tensorrt.TorchLlmArgs | error |
| tensorrt_baseLlmArgs_load_format_literal | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch | passA | tensorrt.DecodingBaseConfig | no_op |
| tensorrt_draftTargetDecodingConfig_backend_must_be_pytorch_when_routed | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_guidedDecodingParams_only_one_guide | passA | tensorrt.GuidedDecodingParams | no_op |
| tensorrt_medusaDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_ngramDecodingConfig_backend_must_be_torch_when_routed | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_ngramDecodingConfig_prompt_lookup_and_matching_positive_when_routed | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_pluginConfig_dtype_not_auto_or_none | passA | tensorrt.PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_KV_CACHE_QUANT_ALGO_LIST | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_modelopt_kv_cache_dtype_kv_cache_quant_algo_in_modelopt_map | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_modelopt_qformat_no_MIXED_PRECISION_assert | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_modelopt_qformat_quant_algo_in_modelopt_map | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passA | tensorrt.QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passA | tensorrt.SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_baseLlmArgs_load_format_literal | passB | BaseLlmArgs | error |
| tensorrt_baseLlmArgs_tokenizer_mode_literal | passB | BaseLlmArgs | error |
| tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch | passB | DecodingBaseConfig | no_op |
| tensorrt_draftTargetDecodingConfig_backend_must_be_pytorch_when_routed | passB | BaseLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed | passB | BaseLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passB | BaseLlmArgs | error |
| tensorrt_guidedDecodingParams_at_most_one_guide | passB | GuidedDecodingParams | no_op |
| tensorrt_medusaDecodingConfig_max_draft_len_positive_when_routed | passB | BaseLlmArgs | error |
| tensorrt_ngramDecodingConfig_backend_must_be_pytorch_when_routed | passB | BaseLlmArgs | error |
| tensorrt_ngramDecodingConfig_lookup_and_matching_positive_when_routed | passB | BaseLlmArgs | error |
| tensorrt_pluginConfig_default_dtype_allowlist_family | passB | PluginConfig | error |
| tensorrt_pluginConfig_gemm_allreduce_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_gemm_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_gemm_swiglu_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_low_latency_gemm_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_low_latency_gemm_swiglu_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passB | QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passB | QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passB | SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passB | TorchLlmArgs | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `tensorrt_basellmargs_validate_build_config_with_runtime_params_max_batch_size_gt` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_basellmargs_validate_build_config_with_runtime_params_max_beam_width_not_equal` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_basellmargs_validate_build_config_with_runtime_params_max_input_len_not_equal` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_basellmargs_validate_build_config_with_runtime_params_max_num_tokens_gt` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_basellmargs_validate_build_config_with_runtime_params_max_seq_len_not_equal` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_basellmargs_validate_lora_config_consistency_max_cpu_loras_not_equal` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_basellmargs_validate_lora_config_consistency_max_loras_not_equal` (mech): 'BaseLlmArgs' object has no attribute 'build_config'
- `tensorrt_batchingType_enum` (passA): '__not_a_batching__' is not a valid BatchingType
- `tensorrt_layerQuantConfig_modelopt_qformat_must_be_MIXED_PRECISION` (passA): TRT-LLM native_type 'tensorrt.LayerQuantConfig' (class 'LayerQuantConfig') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi
