# Pilot GT report - tensorrt 0.21.0 invariants (union + gate)

Round 0: union the 2 GT sources (mech, poc) by tolerant identity (leaf_native_field, coarse_predicate_bucket), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside nvcr.io/nvidia/tensorrt-llm/release:0.21.0, keep gate-confirmed as GT.

## Per-source candidate counts

| source | raw candidates | tolerant keys | gateable | unique tolerant keys |
|---|---|---|---|---|
| mech | 56 | 48 | 56 | 26 |
| poc | 75 | 63 | 0 | 41 |

## Union + gate

- Union size (distinct tolerant identities across 4 sources): **89**
- Gate-confirmed tolerant identities: **3**
- Probed candidates (native_type present, kwargs authored or synthesised): **56** (confirmed=3, failed=8, skipped=38, infra_error=7)
- Confirmations by probe provenance: **3 synthesised** by the gate, 0 from hand-authored kwargs

Group status breakdown (per tolerant identity):

- confirmed: 3
- failed: 7
- infra_error: 7
- skipped: 31
- unreachable: 41

## GT-growth vs PoC N=1 GT

PoC GT contributed **63** tolerant identities. The gate-confirmed union grows GT by **0** confirmed identities the PoC GT lacked:


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
