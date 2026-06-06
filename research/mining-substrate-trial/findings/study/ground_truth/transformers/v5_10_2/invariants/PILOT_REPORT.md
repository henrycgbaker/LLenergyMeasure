# Pilot GT report - transformers 5.10.2 invariants (union + gate)

Round 0: union the 2 GT sources (passA, passB) at the CONSTRAINT grain (leaf_native_field, coarse_predicate_bucket, canonical_predicate_value), runtime-gate every candidate (kwargs authored or synthesised) in the production validator inside transformers container, keep gate-confirmed per constraint as GT.

## Per-source candidate counts

| source | raw candidates | constraints | gateable | unique constraints |
|---|---|---|---|---|
| passA | 28 | 28 | 28 | 28 |
| passB | 95 | 92 | 95 | 92 |

## Union + gate

- Union size (distinct CONSTRAINTS across sources): **120**
- Tolerant keys (coarser, leaf+bucket): 93; of which **23** held >1 distinct constraint (would have over-collapsed under the old leaf+bucket identity).
- Gate-confirmed constraints: **85**
- Probed candidates (native_type present, kwargs authored or synthesised): **123** (confirmed=87, failed=9, skipped=21, infra_error=6)
- Confirmations by probe provenance: **0 synthesised** by the gate, 87 from hand-authored kwargs

Group status breakdown (per constraint):

- confirmed: 85
- failed: 9
- infra_error: 6
- skipped: 20

## GT-growth vs PoC N=1 GT

PoC GT contributed **0** constraints. The gate-confirmed union grows GT by **85** confirmed constraints the PoC GT lacked:

- act_group_aware [presence] = {effect=act_group_aware silently forced False,when={act_group_aware=True,desc_act=True}}
- assistant_model [presence] = {forbidden_attr=assistant_model}
- backend [membership] = {in=[auto,auto_trainable,autoawq,exllama_v1,exllama_v2,gemm,gemm_triton,gemv,gemv_fast,machete,marlin,torch_awq,torch_fused_awq]}
- backward_dtype [membership] = {in=[bf16,mxfp4,mxfp8]}
- beta1 [exact] = {equals=16}
- beta1 [type] = {isinstance=int}
- beta2 [exact] = {equals=16}
- beta2 [type] = {isinstance=int}
- bits [exact] = {equals=3}
- bits [membership] = {in=[2,3,4,8]}
- bits [membership] = {in=[2,3,4]}
- bits [membership] = {in=[2,4,8]}
- bits [type] = {isinstance=int}
- cache_config [presence] = {requires=cache_config is None,when={use_cache=False}}
- cache_implementation [membership] = {in=[dynamic,dynamic_full,hybrid,hybrid_chunked,offloaded,offloaded_hybrid,offloaded_hybrid_chunked,offloaded_static,paged,quantized,sliding_window,static],null_allowed=True}
- cache_implementation [membership] = {must_be_in=ALL_CACHE_IMPLEMENTATIONS + ('paged',)}
- cache_implementation [presence] = {cache_arg_set_while_use_cache_false=[cache_config,cache_implementation]}
- cache_implementation [presence] = {requires=cache_implementation is None,when={use_cache=False}}
- compile_config [type] = [CompileConfig]
- compile_config [type] = {isinstance=CompileConfig,null_allowed=True}
- damp_percent [numeric] = {exclusiveMaximum=1,exclusiveMinimum=0}
- dataset [membership] = {also_allows=list,in=[c4,c4-new,wikitext2],null_allowed=True}
- early_stopping [membership] = [,False,True,never]
- early_stopping [membership] = {in=[,False,True,never]}
- early_stopping [presence] = {requires=early_stopping is False (or None),when={num_beams=1}}
- embed_dim [presence] = {requires=head_dim * num_heads == embed_dim,when=head_dim and num_heads and embed_dim all present}
- enable_proxy_error [exact] = {equals=False}
- epsilon_cutoff [presence] = {requires=epsilon_cutoff == 0.0 (or None),when={do_sample=False}}
- eta_cutoff [presence] = {requires=eta_cutoff == 0.0 (or None),when={do_sample=False}}
- format [membership] = {in=[gemm,gemv,gemv_fast,llm-awq]}
- forward_dtype [membership] = {in=[mxfp4,nvfp4]}
- forward_method [membership] = {in=[abs_max,quest],when={forward_dtype=mxfp4}}
- forward_method [membership] = {in=[abs_max],when={forward_dtype=nvfp4}}
- group_size [membership] = {in=[128,256,64]}
- group_size [numeric] = {exclusiveMinimum=0}
- group_size [numeric] = {gt=0,or_equals=-1}
- group_size [numeric] = {requires=group_size % 8 == 0}
- hadamard_group_size [membership] = {in=[128,16,32,64],when={forward_dtype=nvfp4}}
- hadamard_group_size [membership] = {in=[128,32,64],when={forward_dtype=mxfp4}}
- hadamard_size [presence] = {requires=hadamard_size % group_size == 0}
- ... (+45 more)

## Gate REJECTIONS (kwargs-bearing candidates that ran and were not confirmed)

| id | source | native_type | observed_outcome |
|---|---|---|---|
| transformers_bitsAndBytesConfig_4bit_8bit_mutually_exclusive | passA | transformers.BitsAndBytesConfig | error |
| transformers_watermarkingConfig_context_width_ge_1 | passA | transformers.GenerationConfig | error |
| transformers_watermarkingConfig_greenlist_ratio_range | passA | transformers.GenerationConfig | error |
| transformers_watermarkingConfig_seeding_scheme_allowlist | passA | transformers.GenerationConfig | error |
| transformers_awqConfig_llm_awq_backend_normalisation | passB | transformers.AwqConfig | no_op |
| transformers_pretrainedConfig_num_labels_id2label_length_mismatch_warns | passB | transformers.PreTrainedConfig | dormant_announced |
| transformers_watermarkingConfig_context_width_positive | passB | transformers.WatermarkingConfig | no_op |
| transformers_watermarkingConfig_greenlist_ratio_in_unit_interval | passB | transformers.WatermarkingConfig | no_op |
| transformers_watermarkingConfig_seeding_scheme_allowlist | passB | transformers.WatermarkingConfig | no_op |

### warn_on_unstable_feature_usage flag (passB flagged as possibly-invalid)

_No candidate id matched warn_on_unstable_feature_usage._

## Infra errors (could not run in container)

- `transformers_bitsAndBytesConfig_quant_storage_allowlist` (passA): name 'torch' is not defined
- `transformers_generationConfig_no_generate_only_arguments` (passA): Argument `logits_processor` is not a valid argument of `GenerationConfig`. It should be passed to `generate()` (or a pipeline) directly.
- `transformers_fineGrainedFP8Config_activation_scheme_allowlist` (passB): Activation scheme __nonsense__ not supported
- `transformers_fpquantConfig_nonbf16_backward_requires_mxfp4_forward` (passB): Only 'mxfp4' forward is compatible with non-bf16 backwards for now.
- `transformers_quantoConfig_activations_allowlist` (passB): Only support weights in [None, 'int8', 'float8'] but found int4
- `transformers_sinqConfig_nbits_int_coercible` (passB): invalid literal for int() with base 10: 'abc'
