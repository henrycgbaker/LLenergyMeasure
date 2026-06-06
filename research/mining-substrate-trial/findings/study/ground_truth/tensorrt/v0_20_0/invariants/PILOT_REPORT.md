# Pilot GT report - tensorrt 0.20.0 invariants (union + gate)

Round 0: union the 3 GT sources (mech, passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:0.20.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 45 | 45 | 45 | 12 |
| passB | 56 | 55 | 56 | 22 |
| mech | 38 | 38 | 35 | 37 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **104**
- Tolerant keys (coarser, leaf+bucket): 80; of which **19** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **14**
- Probed candidates (native_type present, kwargs authored or synthesised): **136** (confirmed=22, failed=44, skipped=58, infra_error=12)
- Confirmations by probe provenance: **0 synthesised** by the gate, 22 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 14
- failed: 35
- infra_error: 12
- skipped: 40
- unreachable: 3

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **14** confirmed constraints the PoC GT lacked:

- BatchingType [membership] = [INFLIGHT,STATIC]
- best_of [presence] = {best_of_gt_1_and_greedy_and_env_unset=True}
- capacity_scheduler_policy [membership] = [GUARANTEED_NO_EVICT,MAX_UTILIZATION,STATIC_BATCH]
- context_chunking_policy [membership] = [EQUAL_PROGRESS,FIRST_COME_FIRST_SERVED]
- device [membership] = [cpu,cuda]
- device [presence] = [cpu,cuda]
- lora_ckpt_source [membership] = [hf,nemo]
- max_ngram_size [numeric] = 0
- max_ngram_size [numeric] = {gt=0}
- max_verification_set_size [numeric] = 0
- max_verification_set_size [numeric] = {gt=0}
- max_window_size [numeric] = 0
- max_window_size [numeric] = {gt=0}
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
| tensorrt_batchingType_enum | passA | tensorrt.LlmArgs | error |
| tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch | passA | tensorrt.DecodingBaseConfig | no_op |
| tensorrt_llmArgs_embedding_parallel_mode_dispatch | passA | tensorrt.LlmArgs | warn |
| tensorrt_llmArgs_load_format_literal | passA | tensorrt.LlmArgs | error |
| tensorrt_llmArgs_model_must_be_str_or_path | passA | tensorrt.LlmArgs | error |
| tensorrt_llmArgs_tokenizer_mode_literal | passA | tensorrt.LlmArgs | error |
| tensorrt_pluginConfig_plugin_dtype_options_allowlist | passA | tensorrt.PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_allowlist_lazy | passA | tensorrt.QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passA | tensorrt.SamplingParams | error |
| tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch | passB | DecodingBaseConfig | no_op |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passB | LlmArgs | warn |
| tensorrt_guidedDecodingParams_at_most_one_guide | passB | GuidedDecodingParams | no_op |
| tensorrt_llmArgs_load_format_literal | passB | LlmArgs | error |
| tensorrt_llmArgs_setup_invalid_embedding_parallel_mode | passB | LlmArgs | warn |
| tensorrt_llmArgs_tokenizer_mode_literal | passB | LlmArgs | error |
| tensorrt_medusaDecodingConfig_max_draft_len_positive_when_routed | passB | LlmArgs | warn |
| tensorrt_pluginConfig_default_dtype_allowlist_family | passB | PluginConfig | error |
| tensorrt_pluginConfig_gemm_allreduce_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_gemm_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_gemm_swiglu_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_low_latency_gemm_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_pluginConfig_low_latency_gemm_swiglu_plugin_allowlist | passB | PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passB | QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passB | QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passB | SamplingParams | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `tensorrt_warns_backend_in_lora_config_consistency` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_build_config_set_True_model_format_misc` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_lora_config_set_True_lora_config_consistency` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_batch_size_set_True_set_runtime_knobs_from_build_config` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_beam_width_set_True_build_config_with_runtime_params` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_beam_width_set_True_set_runtime_knobs_from_build_config` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_input_len_set_True_build_config_with_runtime_params` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_input_len_set_True_set_runtime_knobs_from_build_config` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_lora_rank_set_True_lora_config_consistency` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_num_tokens_set_True_set_runtime_knobs_from_build_config` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_seq_len_set_True_build_config_with_runtime_params` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
- `tensorrt_warns_max_seq_len_set_True_set_runtime_knobs_from_build_config` (mech): TRT-LLM native_type 'tensorrt_llm.BaseLlmArgs' (class 'BaseLlmArgs') not resolvable in any of ('tensorrt_llm', 'tensorrt_llm.llmapi', 'tensorrt_llm.llmapi.llm_a
