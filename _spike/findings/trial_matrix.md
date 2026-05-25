# Empirical trial matrix

_generated at 2026-05-25T05:15:51.227561+00:00_

_score files aggregated: 11_

## Per-cell matrix

| strategy | engine | version | bump | schema_recall | schema_prec | inv_recall | inv_prec | sev_acc | wall_s | energy_wh | failure_modes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | tensorrt | v0_21_0 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | transformers | v4_57_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | vllm | v0_7_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| b | tensorrt | v0_21_0 | active | 56.1% | 46.5% | 0.0% | 0.0% | 0.0% | 1372.2 | 66.44 | none;silent |
| b | transformers | v4_57_3 | active | 83.0% | 93.9% | 56.4% | 43.1% | 77.3% | 1649.2 | 81.31 | none |
| b | vllm | v0_7_3 | active | 97.0% | 85.1% | 38.5% | 15.2% | 100.0% | 1414.3 | 67.93 | none |
| b_8b | transformers | v4_57_3 | active | 85.7% | 93.2% | 35.7% | 16.1% | 100.0% | 412.6 | 4.93 | none |
| c | transformers | v4_57_3 | active | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | key_absent |
| d-ab | tensorrt | v0_21_0 | active | 100.0% | 100.0% | 100.0% | 79.5% | 100.0% | 207.5 | 10.94 | none |
| d-ab | transformers | v4_57_3 | active | 100.0% | 100.0% | 100.0% | 93.3% | 100.0% | 20.1 | 0.84 | none |
| d-ab | vllm | v0_7_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 433.6 | 19.38 | none |

## Per-strategy aggregates

| strategy | cells | schema_recall_mean | schema_recall_median | inv_recall_mean | inv_recall_median | wall_mean_s | energy_mean_wh | crashes |
|---|---|---|---|---|---|---|---|---|
| a | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | 0 |
| b | 3 | 78.7% | 83.0% | 31.6% | 38.5% | 1478.5 | 71.89 | 0 |
| b_8b | 1 | 85.7% | 85.7% | 35.7% | 35.7% | 412.6 | 4.93 | 0 |
| c | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | 0 |
| d-ab | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 220.4 | 10.39 | 0 |

## Per-engine aggregates

| engine | cells | schema_recall_mean | inv_recall_mean | wall_mean_s |
|---|---|---|---|---|
| tensorrt | 3 | 85.4% | 66.7% | 526.6 |
| transformers | 5 | 73.8% | 58.4% | 416.4 |
| vllm | 3 | 99.0% | 79.5% | 616.0 |

## Per-bump-distance aggregates

| bump | cells | schema_recall_mean | inv_recall_mean | pass_through_mean |
|---|---|---|---|---|
| active | 11 | 83.8% | 66.4% | - |

## Adjacent observations (deduped per strategy)

### strategy a

- strategy_a: reusing canonical engine_versions outputs for a/tensorrt/v0_21_0
- strategy_a: reusing canonical engine_versions outputs for a/transformers/v4_57_3
- strategy_a: reusing canonical engine_versions outputs for a/vllm/v0_7_3

### strategy b

- chunk 'base_llm_args_validators_bottom' pass2 flag (non-applied): id='tensorrt_llm_enable_lora_ignored_when_lora_config_provided_for_pytorch_backend' reason="Source has a more specific condition `self.enable_lora and self.lora_config is not None and self.backend in ['pytorch', '_autodeploy']` which is not fully captured by the invariant." fix='correct_predicate:exact'
- chunk 'base_llm_args_validators_bottom' pass2 flag (non-applied): id='tensorrt_llm_both_lora_dir_and_lora_target_modules_empty' reason='Source has a more specific condition `len(self.lora_config.lora_dir) == 0 and len(self.lora_config.lora_target_modules) == 0` which is not fully captured by the invariant.' fix='correct_predicate:exact'
- multipass summary: pass2_dropped=0, pass3_added=18, total_invariants=39
- strategy_b: engine='tensorrt', schema_chunks=7, invariants_chunks=7, schema_wall=791.1s, invariants_wall=580.5s, multipass=True
- chunk 'bitsandbytes_config_invariants' pass2 flag (non-applied): id='transformers_bnb_4bit_compute_dtype_not_string_or_torch_dtype' reason='Source allows bnb_4bit_compute_dtype to be a string or torch.dtype, but invariant only checks for not being a string.' fix='correct_predicate:type_is_not_str_or_torch_dtype'
- chunk 'validate_section_01_1.1._Decoding_attributes' pass2 flag (non-applied): id='transformers_pad_token_id_lt_zero' reason='Source raises minor issue for pad_token_id < 0, but invariant severity is warning.' fix='correct_severity:error'
- invariants chunk 'validate_section_01_1.1._Decoding_attributes' pass3_extend: extraction failed; modes=['parse_failure_after_retries']
- multipass summary: pass2_dropped=0, pass3_added=23, total_invariants=51
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=517.3s, invariants_wall=1125.5s
- invariants chunk 'model_config_verify_quantization' pass2_verify: extraction failed; modes=['parse_failure_after_retries']; pass1 unchanged
- invariants chunk 'cache_config_invariants' pass2_verify: extraction failed; modes=['parse_failure_after_retries']; pass1 unchanged
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_num_partial_prefills_lt_1' reason='Source checks for `max_num_partial_prefills < 1` but allows it to be equal to 1.' fix='correct_predicate:not_equal_or_less_than'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_long_partial_prefills_lt_1_or_gt_max_num_partial_prefills' reason='Source checks for `max_long_partial_prefills < 1 or max_long_partial_prefills > max_num_partial_prefills` but allows it to be equal to max_num_partial_prefills.' fix='correct_predicate:not_equal_or_less_than_and_not_greater_than'
- chunk 'scheduler_config_invariants' pass2 flag (non-applied): id='vllm_max_num_partial_prefills_gt_1_without_chunked_prefill_enabled' reason='Source checks for `max_num_partial_prefills > 1 and not chunked_prefill_enabled` but allows it to be equal to 1 without chunked_prefill_enabled.' fix='correct_predicate:not_equal_or_greater_than_and_not_equal'
- chunk 'parallel_config_invariants' pass2 flag (non-applied): id='vllm_tpu_backend_not_ray_for_distributed_inference' reason='Source sets distributed_executor_backend to "ray" when current_platform.device_type is "tpu" and world_size > 1, but does not raise an error for other backends.' fix='correct_predicate:not_equal'
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_pool_type_not_in_allowlist' reason='Source allows pool_type to be a type, not just "ray".' fix='correct_predicate:not_in_or_isinstance'
- chunk 'lora_prompt_adapter_invariants' pass2 flag (non-applied): id='vllm_extra_config_not_dict' reason='Source checks for isinstance(self.extra_config, dict), not just type.' fix='correct_predicate:type_is'
- multipass summary: pass2_dropped=1, pass3_added=18, total_invariants=66
- strategy_b: engine='vllm', schema_chunks=7, invariants_chunks=10, schema_wall=516.6s, invariants_wall=896.8s, multipass=True

### strategy b_8b

- invariants chunk 'validate_section_03_1.3._Performance_attributes': extraction failed; modes=['parse_failure_after_retries']
- invariants chunk 'validate_section_08_2.4._check_num_return_sequences': parsed but yielded 0 unique invariants
- strategy_b: schema_chunks=5, invariants_chunks=14, schema_wall=228.3s, invariants_wall=178.3s

### strategy c

- cell crashed: KeyAbsentError: ANTHROPIC_API_KEY not set; strategy (c) cells are skipped.

### strategy d-ab

- strategy_d_ab on tensorrt: extension=8, flagged_spurious=1, merged_total=43, elapsed=207.1s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=13.9s
- hybrid (d-ab) for vllm: extraction failed; modes=['parse_failure_after_retries']
- strategy_d_ab on vllm: extension=0, flagged_spurious=0, merged_total=26, elapsed=433.2s
