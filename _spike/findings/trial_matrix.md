# Empirical trial matrix

_generated at 2026-05-25T06:04:54.323871+00:00_

_score files aggregated: 19_

## Per-cell matrix

| strategy | engine | version | bump | schema_recall | schema_prec | inv_recall | inv_prec | sev_acc | wall_s | energy_wh | failure_modes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | tensorrt | v0_21_0 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | transformers | v4_55_4 | v-2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 6.7 | 0.08 | detectable |
| a | transformers | v4_56_2 | v-1 | 100.0% | 100.0% | 48.7% | 54.3% | 94.7% | 7.0 | 0.08 | none |
| a | transformers | v4_57_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| a | transformers | v4_57_6 | v+1 | 100.0% | 100.0% | 43.6% | 60.7% | 100.0% | 2.9 | 0.03 | none |
| a | transformers | v5_9_0 | v+major | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.8 | 0.01 | detectable |
| a | vllm | v0_7_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0 | 0.00 | none |
| b | tensorrt | v0_21_0 | active | 56.1% | 46.5% | 0.0% | 0.0% | 0.0% | 1372.2 | 66.44 | none;silent |
| b | transformers | v4_57_3 | active | 83.0% | 93.9% | 56.4% | 43.1% | 77.3% | 1649.2 | 81.31 | none |
| b | vllm | v0_7_3 | active | 97.0% | 85.1% | 38.5% | 15.2% | 100.0% | 1414.3 | 67.93 | none |
| b_8b | transformers | v4_57_3 | active | 85.7% | 93.2% | 35.7% | 16.1% | 100.0% | 412.6 | 4.93 | none |
| c | transformers | v4_57_3 | active | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | key_absent |
| d-ab | tensorrt | v0_21_0 | active | 100.0% | 100.0% | 100.0% | 79.5% | 100.0% | 207.5 | 10.94 | none |
| d-ab | transformers | v4_55_4 | v-2 | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 455.9 | 21.24 | none |
| d-ab | transformers | v4_56_2 | v-1 | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 473.7 | 22.25 | none |
| d-ab | transformers | v4_57_3 | active | 100.0% | 100.0% | 100.0% | 93.3% | 100.0% | 20.1 | 0.84 | none |
| d-ab | transformers | v4_57_6 | v+1 | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 490.5 | 23.32 | none |
| d-ab | transformers | v5_9_0 | v+major | 100.0% | 100.0% | 100.0% | 95.1% | 100.0% | 507.7 | 24.29 | none |
| d-ab | vllm | v0_7_3 | active | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 433.6 | 19.38 | none |

## Per-strategy aggregates

| strategy | cells | schema_recall_mean | schema_recall_median | inv_recall_mean | inv_recall_median | wall_mean_s | energy_mean_wh | crashes |
|---|---|---|---|---|---|---|---|---|
| a | 7 | 71.4% | 100.0% | 56.0% | 48.7% | 2.5 | 0.03 | 0 |
| b | 3 | 78.7% | 83.0% | 31.6% | 38.5% | 1478.5 | 71.89 | 0 |
| b_8b | 1 | 85.7% | 85.7% | 35.7% | 35.7% | 412.6 | 4.93 | 0 |
| c | 1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.00 | 0 |
| d-ab | 7 | 100.0% | 100.0% | 100.0% | 100.0% | 369.9 | 17.47 | 0 |

## Per-engine aggregates

| engine | cells | schema_recall_mean | inv_recall_mean | wall_mean_s |
|---|---|---|---|---|
| tensorrt | 3 | 85.4% | 66.7% | 526.6 |
| transformers | 13 | 74.5% | 60.3% | 309.8 |
| vllm | 3 | 99.0% | 79.5% | 616.0 |

## Per-bump-distance aggregates

| bump | cells | schema_recall_mean | inv_recall_mean | pass_through_mean |
|---|---|---|---|---|
| v-2 | 2 | 50.0% | 50.0% | - |
| v-1 | 2 | 100.0% | 74.4% | - |
| active | 11 | 83.8% | 66.4% | - |
| v+1 | 2 | 100.0% | 71.8% | - |
| v+major | 2 | 50.0% | 50.0% | - |

## Adjacent observations (deduped per strategy)

### strategy a

- strategy_a: reusing canonical engine_versions outputs for a/tensorrt/v0_21_0
- strategy_a bumped: miner subprocess returncode=1; stderr_tail=Traceback (most recent call last):
  File "<string>", line 7, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py", line 1462, in walk_transformers
    import transformers.generation.configuration_utils as gen_mod  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/__init__.py", line 27, in <module>
    from . import dependency_versions_check
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/dependency_versions_check.py", line 57, in <module>
    require_version_core(deps[pkg])
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/utils/versions.py", line 117, in require_version_core
    return require_version(requirement, hint)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/utils/versions.py", line 111, in require_version
    _compare_versions(op, got_ver, want_ver, requirement, pkg, hint)
  File "/tmp/trial_transformers_v4_55_4_venv/src/transformers/utils/versions.py", line 44, in _compare_versions
    raise ImportError(
ImportError: tokenizers>=0.21,<0.22 is required for a normal functioning of this module, but found tokenizers==0.22.2.
Try: `pip install transformers -U` or `pip install -e '.[dev]'` if you're working with git main

- strategy_a bumped: failure_mode=miner_runtime_error
- strategy_a bumped: subprocess stdout=wrote 38 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/transformers/v4_56_2/invariants.proposed.yaml
- strategy_a: reusing canonical engine_versions outputs for a/transformers/v4_57_3
- strategy_a bumped: subprocess stdout=wrote 28 candidates to /home/h.baker@hertie-school.lan/workspace/llenergymeasure/_spike/findings/trial_runs/a/transformers/v4_57_6/invariants.proposed.yaml
- strategy_a bumped: miner subprocess returncode=1; stderr_tail=Traceback (most recent call last):
  File "<string>", line 7, in <module>
  File "/home/h.baker@hertie-school.lan/workspace/llenergymeasure/engine_versions/transformers/v4_57_3/producers/static_invariant_miner.py", line 1462, in walk_transformers
    import transformers.generation.configuration_utils as gen_mod  # type: ignore
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/__init__.py", line 30, in <module>
    from . import dependency_versions_check
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/dependency_versions_check.py", line 16, in <module>
    from .utils.versions import require_version, require_version_core
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/utils/__init__.py", line 76, in <module>
    from .hub import (
  File "/tmp/trial_transformers_v5_9_0_venv/src/transformers/utils/hub.py", line 29, in <module>
    from huggingface_hub import (
ImportError: cannot import name 'is_offline_mode' from 'huggingface_hub' (/home/h.baker@hertie-school.lan/miniforge3/lib/python3.12/site-packages/huggingface_hub/__init__.py)

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
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=455.4s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=473.2s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=13.9s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=490.0s
- strategy_d_ab: extension=2, flagged_spurious=2, merged_total=43, elapsed=507.2s
- hybrid (d-ab) for vllm: extraction failed; modes=['parse_failure_after_retries']
- strategy_d_ab on vllm: extension=0, flagged_spurious=0, merged_total=26, elapsed=433.2s
