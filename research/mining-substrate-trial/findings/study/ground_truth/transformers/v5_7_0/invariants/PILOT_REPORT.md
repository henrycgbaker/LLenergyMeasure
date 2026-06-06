# Pilot GT report - transformers 5.7.0 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside transformers container, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 50 | 50 | 50 | 47 |
| passB | 59 | 57 | 59 | 54 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **104**
- Tolerant keys (coarser, leaf+bucket): 83; of which **18** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **74**
- Probed candidates (native_type present, kwargs authored or synthesised): **109** (confirmed=76, failed=5, skipped=23, infra_error=5)
- Confirmations by probe provenance: **0 synthesised** by the gate, 76 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 74
- failed: 5
- infra_error: 5
- skipped: 20

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **74** confirmed constraints the PoC GT lacked:

- act_group_aware [presence] = {force_false_when=desc_act is True}
- assistant_model [presence] = {forbidden_attr=assistant_model}
- backend [membership] = [auto,auto_trainable,autoawq,exllama_v1,exllama_v2,gemm,gemm_triton,gemv,gemv_fast,machete,marlin,torch_awq,torch_fused_awq]
- backward_dtype [membership] = [bf16,mxfp4,mxfp8]
- bits [membership] = [2,3,4,8]
- bits [membership] = [2,3,4]
- bits [membership] = [2,4,8]
- bits [presence] = {must_be=3}
- cache_config [presence] = {expects=cache_config is None,when={use_cache=False}}
- cache_implementation [membership] = {context=ALL_CACHE_IMPLEMENTATIONS + ('paged',)}
- cache_implementation [membership] = {in=[dynamic,dynamic_full,hybrid,hybrid_chunked,offloaded,offloaded_hybrid,offloaded_hybrid_chunked,offloaded_static,paged,quantized,sliding_window,static],null_allowed=True}
- cache_implementation [presence] = {expects=cache_implementation is None,when={use_cache=False}}
- cache_implementation [presence] = {is_not_none=True,when=use_cache is False}
- compile_config [type] = {isinstance=CompileConfig,null_allowed=True}
- damp_percent [numeric] = {gt=0,lt=1}
- dataset [membership] = [c4,c4-new,wikitext2]
- early_stopping [membership] = {in=[,False,True,never]}
- early_stopping [presence] = {expects=early_stopping is False or None,when={num_beams=1}}
- enable_proxy_error [presence] = {must_be=False}
- epsilon_cutoff [presence] = {expects=epsilon_cutoff == 0.0 or None,when={do_sample=False}}
- eta_cutoff [presence] = {expects=eta_cutoff == 0.0 or None,when={do_sample=False}}
- format [membership] = [gemm,gemv,gemv_fast,llm-awq]
- forward_dtype [membership] = [mxfp4,nvfp4]
- group_size [membership] = [128,256,64]
- group_size [numeric] = {gt=0,or_eq=-1}
- group_size [numeric] = {gt=0}
- group_size [numeric] = {multiple_of=8}
- hadamard_size [numeric] = {divisible_by=group_size}
- id2label [presence] = {requires=len(id2label) == num_labels,when=id2label is not None and num_labels passed}
- in_group_size [type] = [int]
- layer_types [membership] = {each_in=ALLOWED_LAYER_TYPES}
- length_penalty [presence] = {expects=length_penalty == 1.0 or None,when={num_beams=1}}
- length_penalty [presence] = {ne=1,when=num_beams in (None, 1)}
- linear_class [membership] = [autobitlinear,bitlinear]
- logits_processor [presence] = {forbidden_attr=logits_processor}
- max_new_tokens [numeric] = {gt=0,null_allowed=True}
- max_new_tokens [numeric] = {gt=0}
- method [membership] = [asinq,sinq]
- min_p [presence] = {expects=min_p is None,when={do_sample=False}}
- negative_prompt_attention_mask [presence] = {forbidden_attr=negative_prompt_attention_mask}
- ... (+34 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| transformers_generationConfig_early_stopping_valid | passB | transformers.GenerationConfig | error |
| transformers_generationConfig_extra_output_flag_without_return_dict_warn | passB | transformers.GenerationConfig | dormant_announced |
| transformers_watermarkingConfig_context_width_ge_1 | passB | transformers.WatermarkingConfig | no_op |
| transformers_watermarkingConfig_greenlist_ratio_range | passB | transformers.WatermarkingConfig | no_op |
| transformers_watermarkingConfig_seeding_scheme_in | passB | transformers.WatermarkingConfig | no_op |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `transformers_bitsAndBytesConfig_quant_storage_dtype_allowlist` (passB): name 'torch' is not defined
- `transformers_fineGrainedFP8Config_activation_scheme_allowlist` (passB): Activation scheme __bad__ not supported
- `transformers_generationConfig_no_generate_only_arguments` (passB): Argument `logits_processor` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.
- `transformers_hqqConfig_requires_hqq_package` (passB): A valid HQQ version (>=0.2.1) is not available. Please follow the instructions to install it: `https://github.com/mobiusml/hqq/`.
- `transformers_quantoConfig_activations_allowlist` (passB): Only support weights in [None, 'int8', 'float8'] but found int4
