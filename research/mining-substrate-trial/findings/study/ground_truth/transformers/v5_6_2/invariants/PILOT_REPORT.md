# Pilot GT report - transformers 5.6.2 invariants (union + gate)

Round 0: union the 4 GT sources (mech, passA, passB, poc) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside transformers container, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 78 | 76 | 78 | 22 |
| passB | 126 | 124 | 126 | 16 |
| mech | 92 | 92 | 92 | 92 |
| poc | 118 | 116 | 0 | 9 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **247**
- Tolerant keys (coarser, leaf+bucket): 141; of which **70** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **83**
- Probed candidates (native_type present, kwargs authored or synthesised): **296** (confirmed=124, failed=16, skipped=142, infra_error=14)
- Confirmations by probe provenance: **4 synthesised** by the gate, 120 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 83
- failed: 16
- infra_error: 11
- skipped: 128
- unreachable: 9

## GT-growth vs PoC N=1 GT

PoC GT contributed **116** constraints. The gate-confirmed union grows GT by **21** confirmed constraints the PoC GT lacked:

- act_group_aware [presence] = {effect=act_group_aware silently forced to False,when={act_group_aware=True,desc_act=True}}
- backend [membership] = {note=value must be in AwqBackend members}
- backward_dtype [membership] = {in=[bf16,mxfp4,mxfp8],note=v5 widened from bf16-only}
- backward_dtype [membership] = {in=[bf16,mxfp4,mxfp8]}
- cache_implementation [membership] = {in=[dynamic,dynamic_full,hybrid,hybrid_chunked,offloaded,offloaded_hybrid,offloaded_hybrid_chunked,offloaded_static,paged,quantized,sliding_window,static],null_allowed=True}
- cache_implementation [membership] = {note=ALL_CACHE_IMPLEMENTATIONS + ('paged',); null allowed,null_allowed=True}
- early_stopping [presence] = {requires=early_stopping is False or None,when={num_beams=1}}
- group_size [numeric] = 0
- hadamard_size [membership] = 0
- in_group_size [type] = {fields=[in_group_size,nbits_per_codebook,num_codebooks,out_group_size],isinstance=int}
- length_penalty [presence] = {requires=length_penalty == 1.0 or None,when={num_beams=1}}
- max_new_tokens [numeric] = 0
- num_labels [presence] = {requires=len(id2label) == num_labels when both passed}
- output_attentions [presence] = {requires=_attn_implementation in {eager, None},when={output_attentions=True}}
- output_logits [presence] = {requires=output_logits is not True,when={return_dict_in_generate=False}}
- output_scores [presence] = {requires=output flag is not True,when={return_dict_in_generate=False}}
- pad_token_id [numeric] = 0
- temperature [presence] = {requires=temperature == 1.0 or None,when={do_sample=False}}
- top_h [presence] = {requires=top_h is None,when={do_sample=False}}
- top_k [presence] = {requires=top_k == 50 or None,when={do_sample=False}}
- top_p [presence] = {requires=top_p == 1.0 or None,when={do_sample=False}}

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| transformers_autoroundconfig_post_init_group_size_not_equal | mech | transformers.AutoRoundConfig | no_op |
| transformers_fpquantconfig_post_init_backward_dtype_not_equal | mech | transformers.FPQuantConfig | error |
| transformers_generationconfig_validate_minor_issues_gt | mech | transformers.GenerationConfig | no_op |
| transformers_generationconfig_validate_num_beams_gt | mech | transformers.GenerationConfig | no_op |
| transformers_generationconfig_validate_num_return_sequences_gt | mech | transformers.GenerationConfig | no_op |
| transformers_spqrconfig_post_init_beta1_not_equal | mech | transformers.SpQRConfig | error |
| transformers_spqrconfig_post_init_beta2_not_equal | mech | transformers.SpQRConfig | error |
| transformers_spqrconfig_post_init_bits_not_equal | mech | transformers.SpQRConfig | error |
| transformers_watermarkingconfig_validate_context_width_ge | mech | transformers.WatermarkingConfig | no_op |
| transformers_watermarkingconfig_validate_greenlist_ratio_le | mech | transformers.WatermarkingConfig | no_op |
| transformers_awq_llm_awq_backend_normalisation | passA | transformers.AwqConfig | no_op |
| transformers_torchao_requires_torchao_installed | passA | transformers.TorchAoConfig | error |
| awq_llm_awq_backend_normalisation | passB | transformers.AwqConfig | no_op |
| watermarking_context_width_positive | passB | transformers.WatermarkingConfig | no_op |
| watermarking_greenlist_ratio_in_unit_interval | passB | transformers.WatermarkingConfig | no_op |
| watermarking_seeding_scheme_allowlist | passB | transformers.WatermarkingConfig | no_op |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `transformers_finegrainedfp8config_post_init_weight_block_size_le` (mech): object of type 'int' has no len()
- `transformers_finegrainedfp8config_post_init_weight_block_size_not_equal` (mech): object of type 'int' has no len()
- `transformers_gptqconfig_post_init_damp_percent_lt` (mech): GPTQConfig.__init__() missing 1 required positional argument: 'bits'
- `transformers_synthidtextwatermarkingconfig_validate_sampling_table_size_gt` (mech): 
SynthIDTextWatermarkingConfig requires the PyTorch library but it was not found in your environment. Check out the instructions on the
installation page: https
- `transformers_finegrainedfp8_activation_scheme_allowlist` (passA): Activation scheme nonsense not supported
- `transformers_quanto_activations_allowlist` (passA): Only support weights in [None, 'int8', 'float8'] but found int4
- `transformers_sinq_nbits_int_coercible` (passA): invalid literal for int() with base 10: 'abc'
- `aqlm_size_fields_int_type` (passB): in_group_size must be a float
- `finegrainedfp8_activation_scheme_allowlist` (passB): Activation scheme nonsense not supported
- `fpquant_nonbf16_backward_requires_mxfp4_forward` (passB): Only 'mxfp4' forward is compatible with non-bf16 backwards for now.
- `quanto_activations_allowlist` (passB): Only support weights in [None, 'int8', 'float8'] but found int4
- `sinq_nbits_int_coercible` (passB): invalid literal for int() with base 10: 'abc'
- `synthid_sampling_table_size_max` (passB): 
SynthIDTextWatermarkingConfig requires the PyTorch library but it was not found in your environment. Check out the instructions on the
installation page: https
- `vptq_layer_is_indice_packed_must_be_true` (passB): 'VptqLayerConfig' object has no attribute 'quant_method'
