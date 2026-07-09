# Invalid Parameter Combinations

> Auto-generated. Do not edit by hand: run
> `python scripts/generate_invalid_combos_doc.py` (or `make docs-all`).

This document lists parameter combinations that fail validation or run
differently than declared. The error rules are enforced at config load
time with a clear error message; the dormant rules are accepted but
silently normalised by the engine. Both are derived from the live rule
corpus (`src/llenergymeasure/engines/<engine>/rules.yaml`) plus the
cross-engine `ExperimentConfig` validators, so this page cannot drift
from what actually fires at runtime.

## Config Validation Errors

These combinations are rejected at config load time with a clear error
message. Rows citing a rule id come from that engine's shipped rule
corpus; the rest are `ExperimentConfig` pydantic validators.

| Engine | Invalid Combination | Reason | Resolution |
|---------|---------------------|--------|------------|
| all | `engine section mismatch` | The engine section must match the engine field (validate_engine_section_match). | Ensure the transformers:/vllm:/tensorrt: section matches the engine: field. |
| all | `passthrough_kwargs key collision` | passthrough_kwargs keys must not collide with ExperimentConfig fields (validate_passthrough_kwargs_no_collision). | Set the named field directly instead of via passthrough_kwargs. |
| all | `unknown field on the engine section wrapper` | A key placed directly on the engine section (not under engine_params/sampling_params) is never forwarded to the engine (validate_engine_section_extras). | Move the key under <engine>.engine_params or <engine>.sampling_params. |
| transformers | `attn_implementation in [flash_attention_2, flash_attention_3] and dtype=float32` | attn_implementation='flash_attention_2'/'flash_attention_3' requires dtype='float16' or dtype='bfloat16'; FlashAttention does not support float32 computation (validate_transformers_flash_attn_dtype). | Set transformers.engine_params.dtype to float16 or bfloat16. |
| tensorrt | `max_input_len < 1` | max_input_len must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_engineparams_raises_max_input_len_lt_1. |
| tensorrt | `max_num_tokens < 1` | max_num_tokens must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_engineparams_raises_max_num_tokens_lt_1. |
| tensorrt | `max_seq_len < 1` | max_seq_len must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_engineparams_raises_max_seq_len_lt_1. |
| tensorrt | `pipeline_parallel_size < 1` | pipeline_parallel_size must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_engineparams_raises_pipeline_parallel_size_lt_1. |
| tensorrt | `tensor_parallel_size < 1` | tensor_parallel_size must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_engineparams_raises_tensor_parallel_size_lt_1. |
| tensorrt | `max_batch_size < 0` | engine_params.max_batch_size must be non-negative, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_raises_max_batch_size_lt_0. |
| tensorrt | `min_p > 1.0` | min_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_min_p_gt_1p0. |
| tensorrt | `min_p < 0.0` | min_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_min_p_lt_0p0. |
| tensorrt | `min_tokens < 0` | min_tokens must be >= 0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_min_tokens_lt_0. |
| tensorrt | `n < 1` | n must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_n_lt_1. |
| tensorrt | `repetition_penalty <= 0.0` | repetition_penalty must be >= 0.0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_repetition_penalty_le_0p0. |
| tensorrt | `temperature < 0.0` | temperature must be >= 0.0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_temperature_lt_0p0. |
| tensorrt | `top_k < 0` | top_k must be >= 0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_top_k_lt_0. |
| tensorrt | `top_p > 1.0` | top_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_top_p_gt_1p0. |
| tensorrt | `top_p < 0.0` | top_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule tensorrt_samplingparams_raises_top_p_lt_0p0. |
| transformers | `load_in_4bit=True and load_in_8bit=True` | load_in_4bit and load_in_8bit are both True, but only one can be used at the same time | Adjust the field(s) so the condition no longer holds; see rule transformers_bnb_load_in_4bit_xor_load_in_8bit. |
| transformers | `cache_implementation is set and cache_implementation not in [static, offloaded_static, sliding_window, hybrid, hybrid_chunked, offloaded_hybrid, offloaded_hybrid_chunked, dynamic, dynamic_full, offloaded, quantized, paged]` | Invalid `cache_implementation` (nonsense). Choose one of: ('static', 'offloaded_static', 'sliding_window', 'hybrid', 'hybrid_chunked', 'offloaded_hybrid', 'offloaded_hybrid_chunked', 'dynamic', 'dynamic_full', 'offloaded', 'quantized', 'paged') | Adjust the field(s) so the condition no longer holds; see rule transformers_cache_choice_cache_implementation_not_in_allowlist. |
| transformers | `compile_config > 0` | You provided `compile_config` as an instance of <class 'int'>, but it must be an instance of `CompileConfig`. | Adjust the field(s) so the condition no longer holds; see rule transformers_compile_config_type_compile_config_exceeds_zero. |
| transformers | `compile_config is set and type(compile_config) is not CompileConfig` | You provided `compile_config` as an instance of <class 'str'>, but it must be an instance of `CompileConfig`. | Adjust the field(s) so the condition no longer holds; see rule transformers_compile_config_type_compile_config_type_not_in_CompileConfig. |
| transformers | `early_stopping is set and early_stopping not in [False, True, never]` | `early_stopping` must be a boolean or 'never', but is sometimes. | Adjust the field(s) so the condition no longer holds; see rule transformers_early_stopping_type_early_stopping_not_in_allowlist. |
| transformers | `early_stopping is set and type(early_stopping) is not bool, int, str` | `early_stopping` must be a boolean or 'never', but is 1.5. | Adjust the field(s) so the condition no longer holds; see rule transformers_early_stopping_type_early_stopping_type_not_in_bool_or_int_or_str. |
| transformers | `no_repeat_ngram_size < 0` | no_repeat_ngram_size must be >= 0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_engineparams_raises_no_repeat_ngram_size_lt_0. |
| transformers | `num_beams < 1` | num_beams must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_engineparams_raises_num_beams_lt_1. |
| transformers | `prompt_lookup_num_tokens < 1` | prompt_lookup_num_tokens must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_engineparams_raises_prompt_lookup_num_tokens_lt_1. |
| transformers | `num_beams=1 and num_return_sequences > 1 and do_sample=False` | Greedy methods (do_sample != True) without beam search do not support `num_return_sequences` different than 1 (got 2). | Adjust the field(s) so the condition no longer holds; see rule transformers_num_return_vs_beams_do_sample_eq_false_and_num_beams_eq_1. |
| transformers | `num_beams < @transformers.sampling_params.num_return_sequences` | `num_return_sequences` (4) has to be smaller or equal to `num_beams` (2). | Adjust the field(s) so the condition no longer holds; see rule transformers_num_return_vs_beams_num_beams_lt_num_return_sequences. |
| transformers | `num_return_sequences > @transformers.engine_params.num_beams` | `num_return_sequences` has to be smaller or equal to `num_beams`. | Adjust the field(s) so the condition no longer holds; see rule transformers_num_return_vs_beams_num_return_sequences_gt_num_beams. |
| transformers | `max_new_tokens <= 0` | `max_new_tokens` must be greater than 0, but is -1. | Adjust the field(s) so the condition no longer holds; see rule transformers_output_token_ids_max_new_tokens_le_zero. |
| transformers | `type(bnb_4bit_quant_type) is not str` | bnb_4bit_quant_type must be a string | Adjust the field(s) so the condition no longer holds; see rule transformers_raises_bnb_4bit_quant_type_not_type_str. |
| transformers | `type(bnb_4bit_use_double_quant) is not bool` | bnb_4bit_use_double_quant must be a boolean | Adjust the field(s) so the condition no longer holds; see rule transformers_raises_bnb_4bit_use_double_quant_not_type_bool. |
| transformers | `type(compile_config) is not CompileConfig and compile_config is set` | You provided `compile_config` as an instance of {type(self.compile_config)}, but it must be an instance of `CompileConfig`. | Adjust the field(s) so the condition no longer holds; see rule transformers_raises_compile_config_not_type_compileconfig. |
| transformers | `early_stopping not in [None, True, False, never]` | `early_stopping` must be a boolean or 'never', but is {early_stopping}. | Adjust the field(s) so the condition no longer holds; see rule transformers_raises_early_stopping_not_in_set. |
| transformers | `type(load_in_4bit) is not bool` | load_in_4bit must be a boolean | Adjust the field(s) so the condition no longer holds; see rule transformers_raises_load_in_4bit_not_type_bool. |
| transformers | `type(load_in_8bit) is not bool` | load_in_8bit must be a boolean | Adjust the field(s) so the condition no longer holds; see rule transformers_raises_load_in_8bit_not_type_bool. |
| transformers | `min_new_tokens < 1` | min_new_tokens must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_min_new_tokens_lt_1. |
| transformers | `min_p > 1.0` | min_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_min_p_gt_1p0. |
| transformers | `min_p < 0.0` | min_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_min_p_lt_0p0. |
| transformers | `repetition_penalty <= 0.0` | repetition_penalty must be >= 0.0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_repetition_penalty_le_0p0. |
| transformers | `temperature < 0.0` | temperature must be >= 0.0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_temperature_lt_0p0. |
| transformers | `top_k < 0` | top_k must be >= 0, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_top_k_lt_0. |
| transformers | `top_p > 1.0` | top_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_top_p_gt_1p0. |
| transformers | `top_p < 0.0` | top_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule transformers_samplingparams_raises_top_p_lt_0p0. |
| transformers | `watermarking_config > 0` | 'int' object has no attribute 'validate' | Adjust the field(s) so the condition no longer holds; see rule transformers_watermarking_type_watermarking_config_exceeds_zero. |
| transformers | `watermarking_config is set and type(watermarking_config) is not WatermarkingConfig` | 'int' object has no attribute 'validate' | Adjust the field(s) so the condition no longer holds; see rule transformers_watermarking_type_watermarking_config_type_not_in_WatermarkingConfig. |
| vllm | `kv_cache_memory_bytes < 1` | kv_cache_memory_bytes must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_engineparams_raises_kv_cache_memory_bytes_lt_1. |
| vllm | `pipeline_parallel_size < 1` | pipeline_parallel_size must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_engineparams_raises_pipeline_parallel_size_lt_1. |
| vllm | `tensor_parallel_size < 1` | tensor_parallel_size must be >= 1, got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_engineparams_raises_tensor_parallel_size_lt_1. |
| vllm | `data_parallel_size <= 1 and data_parallel_external_lb is set` | data_parallel_external_lb can only be set when data_parallel_size > 1 | Adjust the field(s) so the condition no longer holds; see rule vllm_parallelconfig_raises_data_parallel_external_lb_set_true. |
| vllm | `data_parallel_size_local > @data_parallel_size` | data_parallel_size_local ({data_parallel_size_local}) must be <= data_parallel_size ({data_parallel_size}) | Adjust the field(s) so the condition no longer holds; see rule vllm_parallelconfig_raises_data_parallel_size_local_gt_ref_data_parallel_size. |
| vllm | `frequency_penalty > 2.0` | frequency_penalty must be in [-2.0..2.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_frequency_penalty_gt_2p0. |
| vllm | `frequency_penalty < -2.0` | frequency_penalty must be in [-2.0..2.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_frequency_penalty_lt_neg2p0. |
| vllm | `min_p > 1.0` | min_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_min_p_gt_1p0. |
| vllm | `min_p < 0.0` | min_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_min_p_lt_0p0. |
| vllm | `max_tokens is set and min_tokens > @max_tokens` | min_tokens must be less than or equal to max_tokens={max_tokens}, got {min_tokens}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_min_tokens_gt_ref_max_tokens. |
| vllm | `min_tokens < 0` | min_tokens must be greater than or equal to 0, got {min_tokens}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_min_tokens_lt_0. |
| vllm | `n < 1` | n must be at least 1, got {n}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_n_lt_1. |
| vllm | `type(n) is not int` | n must be an int, but is of type {type(self.n)} | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_n_not_type_int. |
| vllm | `presence_penalty > 2.0` | presence_penalty must be in [-2.0..2.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_presence_penalty_gt_2p0. |
| vllm | `presence_penalty < -2.0` | presence_penalty must be in [-2.0..2.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_presence_penalty_lt_neg2p0. |
| vllm | `repetition_penalty <= 0.0` | repetition_penalty must be greater than zero, got {repetition_penalty}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_repetition_penalty_le_0p0. |
| vllm | `temperature < 0.0` | temperature must be non-negative, got {temperature}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_temperature_lt_0p0. |
| vllm | `top_k < -1` | top_k must be 0 (disable), or at least 1, got {top_k}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_top_k_lt_neg1. |
| vllm | `type(top_k) is not int` | top_k must be an integer, got {type(self.top_k).__name__} | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_top_k_not_type_int. |
| vllm | `top_p > 1.0` | top_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_top_p_gt_1p0. |
| vllm | `top_p <= 0.0` | top_p must be in [0.0..1.0], got {declared_value}. | Adjust the field(s) so the condition no longer holds; see rule vllm_samplingparams_raises_top_p_le_0p0. |
| vllm | `max_num_partial_prefills > 1 and long_prefill_token_threshold > @max_model_len` | long_prefill_token_threshold ({long_prefill_token_threshold}) cannot be greater than the max_model_len ({max_model_len}). | Adjust the field(s) so the condition no longer holds; see rule vllm_schedulerconfig_raises_long_prefill_token_threshold_gt_ref_max_model_len. |
| vllm | `max_long_partial_prefills > @max_num_partial_prefills` | self.max_long_partial_prefills={max_long_partial_prefills} must be less than or equal to self.max_num_partial_prefills={max_num_partial_prefills}. | Adjust the field(s) so the condition no longer holds; see rule vllm_schedulerconfig_raises_max_long_partial_prefills_gt_ref_max_num_partial_prefills. |
| vllm | `max_num_batched_tokens < @max_model_len and enable_chunked_prefill is unset` | max_num_batched_tokens ({max_num_batched_tokens}) is smaller than max_model_len ({max_model_len}). This effectively limits the maximum sequence length to max_num_batched_tokens and makes vLLM reject longer sequences. Please increase max_num_batched_tokens or decrease max_model_len. | Adjust the field(s) so the condition no longer holds; see rule vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_model_len. |
| vllm | `max_num_batched_tokens < @max_num_seqs` | max_num_batched_tokens ({max_num_batched_tokens}) must be greater than or equal to max_num_seqs ({max_num_seqs}). | Adjust the field(s) so the condition no longer holds; see rule vllm_schedulerconfig_raises_max_num_batched_tokens_lt_ref_max_num_seqs. |

## Dormant Parameters

These combinations pass validation, but the engine silently normalises
or ignores the declared field: the declared value is not the effective
value. The study planner deduplicates configs that differ only in a
dormant field, so the GPU runs such a cell once. `Normalised fields`
names the paths the engine drives back to their default.

| Engine | Combination | Effect | Normalised fields |
|---------|-------------|--------|-------------------|
| transformers | `use_cache=False` | You have not set `use_cache` to `True`, but cache_implementation is set to static.cache_implementation will have no effect. | - |
| transformers | `early_stopping is set` | Enforced by rule transformers_dormant_early_stopping_set_true. | - |
| transformers | `epsilon_cutoff != 0.0 and epsilon_cutoff is set` | Enforced by rule transformers_dormant_epsilon_cutoff_ne_0_0. | - |
| transformers | `eta_cutoff != 0.0 and eta_cutoff is set` | Enforced by rule transformers_dormant_eta_cutoff_ne_0_0. | - |
| transformers | `do_sample=False and epsilon_cutoff is set` | `do_sample` is not set to `True`. However, `epsilon_cutoff` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `epsilon_cutoff`. | - |
| transformers | `do_sample=False and eta_cutoff is set` | `do_sample` is not set to `True`. However, `eta_cutoff` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `eta_cutoff`. | - |
| transformers | `do_sample=False and min_p is set` | `do_sample` is not set to `True`. However, `min_p` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `min_p`. | - |
| transformers | `do_sample=False and temperature is set` | `do_sample` is not set to `True`. However, `temperature` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`. | - |
| transformers | `do_sample=False and top_h is set` | `do_sample` is not set to `True`. However, `top_h` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_h`. | - |
| transformers | `do_sample=False and top_k is set` | `do_sample` is not set to `True`. However, `top_k` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_k`. | - |
| transformers | `do_sample=False and top_p is set` | `do_sample` is not set to `True`. However, `top_p` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`. | - |
| transformers | `do_sample=False and typical_p is set` | `do_sample` is not set to `True`. However, `typical_p` is set to `{declared_value}` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `typical_p`. | - |
| transformers | `pad_token_id < 0` | `pad_token_id` should be positive but got -1. This will cause errors when batch generating, if there is padding. Please set `pad_token_id` explicitly as `model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation | - |
| transformers | `return_dict_in_generate=False and output_attentions is set` | `return_dict_in_generate` is NOT set to `True`, but `output_attentions` is. When `return_dict_in_generate` is not `True`, `output_attentions` is ignored. | - |
| transformers | `return_dict_in_generate=False and output_hidden_states is set` | `return_dict_in_generate` is NOT set to `True`, but `output_hidden_states` is. When `return_dict_in_generate` is not `True`, `output_hidden_states` is ignored. | - |
| transformers | `return_dict_in_generate=False and output_logits is set` | `return_dict_in_generate` is NOT set to `True`, but `output_logits` is. When `return_dict_in_generate` is not `True`, `output_logits` is ignored. | - |
| transformers | `return_dict_in_generate=False and output_scores is set` | `return_dict_in_generate` is NOT set to `True`, but `output_scores` is. When `return_dict_in_generate` is not `True`, `output_scores` is ignored. | - |
| transformers | `pad_token_id is set and pad_token_id not in [0, 50256]` | `pad_token_id` should be positive but got -1. This will cause errors when batch generating, if there is padding. Please set `pad_token_id` explicitly as `model.generation_config.pad_token_id=PAD_TOKEN_ID` to avoid errors in generation | - |
| transformers | `num_beams=1 and early_stopping is set` | `num_beams` is set to 1. However, `early_stopping` is set to `{declared_value}` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `early_stopping`. | - |
| transformers | `num_beams=1 and length_penalty is set` | `num_beams` is set to 1. However, `length_penalty` is set to `{declared_value}` -- this flag is only used in beam-based generation modes. You should set `num_beams>1` or unset `length_penalty`. | - |
| vllm | `all2all_backend in [pplx, naive]` | Enforced by rule vllm_parallelconfig_dormant_all2all_backend_in. | all2all_backend |
| vllm | `distributed_executor_backend=external_launcher and data_parallel_rank is set` | Enforced by rule vllm_parallelconfig_dormant_data_parallel_rank_set_true. | data_parallel_rank |
| vllm | `seed=-1` | Enforced by rule vllm_samplingparams_dormant_seed_eq_neg1. | seed |

## Runtime Limitations

These combinations pass config validation but may fail at runtime
due to hardware, model, or package requirements.

| Engine | Parameter | Limitation | Resolution |
|---------|-----------|------------|------------|
| transformers | `transformers.engine_params.attn_implementation=flash_attention_2` | flash-attn requires Ampere+ GPU (SM80+); fails on older architectures | Use attn_implementation='sdpa' on pre-Ampere GPUs |
| transformers | `transformers.engine_params.attn_implementation=flash_attention_3` | FA3 requires the flash_attn_3 package (built from flash-attn hopper/ directory) and Ampere+ GPU (SM80+). The Docker PyTorch image includes it pre-built | Install flash_attn_3 from source, or use the Docker runner |
| vllm | `vllm.engine_params.kv_cache_dtype=fp8` | FP8 KV cache requires Hopper (H100) or newer GPU | Use kv_cache_dtype='auto' for automatic selection |
| vllm | `vllm.engine_params.attention.backend=flashinfer` | FlashInfer requires JIT compilation on first use | Leave attention.backend unset (auto) or use 'flash_attn' |
| vllm | `vllm.engine_params.quantization=awq/gptq` | Requires a pre-quantized model checkpoint | Use a quantized model (e.g., TheBloke/*-AWQ) or omit |
| tensorrt | `tensorrt.engine_params.quant_config.quant_algo=FP8` | FP8 requires SM >= 8.9 (Ada Lovelace or Hopper). A100 (SM80) raises ConfigurationError - no silent emulation or fallback | Use INT8, W4A16_AWQ, W4A16_GPTQ, or W8A16 on A100 |
| tensorrt | `tensorrt.engine_params.quant_config.quant_algo=INT8` | INT8 quantisation requires a calibrated checkpoint; uncalibrated weights degrade accuracy | Use a pre-quantised checkpoint or a weight-only algo (W4A16_AWQ, W4A16_GPTQ, W8A16) |

## Engine Capability Matrix

| Feature | Transformers | vLLM | TensorRT |
|---------|---------|------|----------|
| Tensor Parallel | Yes | Yes | Yes |
| Data Parallel | No | No | No |
| BitsAndBytes (4-bit) | Yes | No | No |
| BitsAndBytes (8-bit) | Yes | No | No |
| Native Quantization | No | AWQ/GPTQ/FP8 | INT8/W4A16_AWQ/W4A16_GPTQ/FP8 |
| float32 precision | Yes | No | No |
| float16 precision | Yes | Yes | Yes |
| bfloat16 precision | Yes | Yes | Yes |
| Prefix Caching | No | Yes | No |
| torch.compile | Yes | No | No |
| Beam Search | Yes | Yes | No |
| Speculative Decoding | Yes | Yes | No |
| Static KV Cache | Yes | No | No |

**Notes:**
- vLLM supports 4-bit via AWQ/GPTQ quantized models, not bitsandbytes
- TensorRT-LLM is optimised for FP16/BF16/INT8, not FP32

## Recommended Configurations by Use Case

### Memory-Constrained (Consumer GPU)
```yaml
engine: transformers
transformers:
  load_in_4bit: true
  bnb_4bit_quant_type: nf4
```

### High Throughput (Production)
```yaml
engine: vllm
vllm:
  engine:
    gpu_memory_utilization: 0.9
    enable_prefix_caching: true
```

### Maximum Performance (Ampere+)
```yaml
engine: tensorrt
tensorrt:
  dtype: float16
  quant_config:
    quant_algo: FP8  # Hopper only
```
