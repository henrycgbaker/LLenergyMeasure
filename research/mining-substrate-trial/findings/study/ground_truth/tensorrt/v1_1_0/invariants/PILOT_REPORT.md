# Pilot GT report - tensorrt 1.1.0 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:1.1.0, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 80 | 77 | 80 | 14 |
| passB | 71 | 70 | 71 | 7 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **84**
- Tolerant keys (coarser, leaf+bucket): 71; of which **11** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **24**
- Probed candidates (native_type present, kwargs authored or synthesised): **151** (confirmed=45, failed=36, skipped=69, infra_error=1)
- Confirmations by probe provenance: **0 synthesised** by the gate, 45 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 24
- failed: 19
- infra_error: 1
- skipped: 40

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **24** confirmed constraints the PoC GT lacked:

- allreduce_strategy [membership] = [AUTO,LOWPRECISION,MINLATENCY,MNNVL,NCCL,NCCL_SYMMETRIC,ONESHOT,TWOSHOT,UB]
- backend [membership] = [CUTEDSL,CUTLASS,DEEPGEMM,TRITON,TRTLLM,VANILLA,WIDEEP]
- backend [membership] = [DEFAULT,MPI,NIXL,UCX]
- batch_wait_max_tokens_ratio [numeric] = {ge=0,le=1}
- batch_wait_timeout_iters [numeric] = {ge=0}
- batch_wait_timeout_ms [numeric] = {ge=0}
- capacity_scheduler_policy [membership] = [GUARANTEED_NO_EVICT,MAX_UTILIZATION,STATIC_BATCH]
- capture_num_tokens [numeric] = {applies_to=each_element,gt=0}
- context_chunking_policy [membership] = [EQUAL_PROGRESS,FIRST_COME_FIRST_SERVED]
- device [membership] = [cpu,cuda]
- guided_decoding_backend [membership] = [llguidance,xgrammar]
- lora_ckpt_source [membership] = [hf,nemo]
- mamba_ssm_cache_dtype [membership] = [auto,bfloat16,float16,float32]
- max_attention_window [numeric] = {applies_to=each_element,gt=0,must_be_nonempty=True}
- max_batch_size [numeric] = {ge=0}
- max_gpu_total_bytes [numeric] = {ge=0}
- max_ngram_size [numeric] = {gt=0}
- max_num_streams [numeric] = {ge=1}
- max_verification_set_size [numeric] = {gt=0}
- max_window_size [numeric] = {gt=0}
- model [type] = [Path,str]
- stream_interval [numeric] = {gt=0}
- tokenizer_mode [membership] = [auto,slow]
- truncate_prompt_tokens [numeric] = {ge=1}

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| tensorrt_baseLlmArgs_load_format_literal | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_batchingType_enum | passA | tensorrt.TrtLlmArgs | error |
| tensorrt_decodingBaseConfig_from_dict_decoding_type_dispatch | passA | tensorrt.DecodingBaseConfig | error |
| tensorrt_draftTargetDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_speculative_model_dir_required_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_speculative_model_dir_required_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_lookaheadDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_medusaDecodingConfig_max_draft_len_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_mtpDecodingConfig_num_nextn_predict_layers_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_ngramDecodingConfig_max_draft_len_and_matching_positive_when_routed | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_pluginConfig_dtype_field_allowlist | passA | tensorrt.PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passA | tensorrt.QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passA | tensorrt.QuantAlgo | error |
| tensorrt_samplingParams_best_of_ge_n | passA | tensorrt.SamplingParams | error |
| tensorrt_samplingParams_best_of_gt_1_greedy_requires_env | passA | tensorrt.SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passA | tensorrt.TorchLlmArgs | error |
| tensorrt_trtLlmArgs_validate_kv_cache_dtype_must_be_auto | passA | tensorrt.TrtLlmArgs | warn |
| tensorrt_baseLlmArgs_guided_decoding_backend_literal | passB | tensorrt_llm.llmapi.BaseLlmArgs | error |
| tensorrt_baseLlmArgs_load_format_literal | passB | tensorrt_llm.llmapi.BaseLlmArgs | error |
| tensorrt_baseLlmArgs_tokenizer_mode_literal | passB | tensorrt_llm.llmapi.BaseLlmArgs | error |
| tensorrt_batchingType_enum | passB | tensorrt_llm.llmapi.TrtLlmArgs | error |
| tensorrt_draftTargetDecodingConfig_backend_and_max_draft_len_and_model_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_max_draft_len_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_speculative_model_dir_required_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_eagleDecodingConfig_validate_speculative_model_required_when_routed | passB | tensorrt_llm.llmapi.llm_args.EagleDecodingConfig | error |
| tensorrt_guidedDecodingParams_at_most_one_guide | passB | tensorrt_llm.sampling_params.GuidedDecodingParams | error |
| tensorrt_mtpDecodingConfig_num_nextn_predict_layers_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_ngramDecodingConfig_max_draft_len_and_matching_positive_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_pluginConfig_default_dtype_allowlist_assert | passB | tensorrt_llm.plugin.PluginConfig | error |
| tensorrt_quantConfig_kv_cache_quant_algo_in_allowlist | passB | tensorrt_llm.quantization.QuantConfig | no_op |
| tensorrt_quantConfig_quant_algo_in_QuantAlgo_enum | passB | tensorrt_llm.quantization.QuantConfig | no_op |
| tensorrt_samplingParams_best_of_ge_n | passB | tensorrt_llm.sampling_params.SamplingParams | error |
| tensorrt_torchLlmArgs_load_format_enum | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |
| tensorrt_trtLlmArgs_validate_kv_cache_dtype_must_be_auto | passB | tensorrt_llm.llmapi.TrtLlmArgs | warn |
| tensorrt_userProvided_and_auto_backend_must_be_pytorch_when_routed | passB | tensorrt_llm.llmapi.TorchLlmArgs | error |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passA, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)
- `tensorrt_torchLlmArgs_warn_on_unstable_feature_usage` (source=passB, gateable=True, verdict=skipped, observed=skipped_unsynthesizable)

## Infra errors (could not run in container)

- `tensorrt_LLM_pytorch_rejects_trt_specific_kwargs` (passA): The following arguments are specific to TensorRT backend and cannot be used with PyTorch backend: ['enable_build_cache'].
Please use 'from tensorrt_llm._tensorr
