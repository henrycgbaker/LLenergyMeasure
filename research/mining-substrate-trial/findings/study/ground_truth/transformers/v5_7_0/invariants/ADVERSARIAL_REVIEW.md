# Adversarial GT review - transformers 5.7.0 invariants

Reviewer: adversarial GT auditor (refute-not-rubber-stamp).
Source under audit: /tmp/tfvenv-5.7.0/lib/python3.12/site-packages/transformers
GT under audit: PILOT_GT.yaml `confirmed` list (n_confirmed = 74).
Citations resolved by id via passA_entrypoint.yaml / passB_classtree.yaml.
No mech-sourced and NO PoC-folded entries in this cell (sources: passA=36, passB=38).

## Scope of verification

Confirmed list = 74 (> 60). Verified ALL 74 entries against source (full, not
sampled). The sample therefore spans every native_type (GPTQConfig, GenerationConfig,
AwqConfig, FPQuantConfig, HiggsConfig, MetalConfig, SpQRConfig, VptqConfig, SinqConfig,
PretrainedConfig, AqlmConfig, BitNetQuantConfig, FineGrainedFP8Config, QuantoConfig,
EetqConfig), every predicate_kind (presence_conflict, required, strenum_in,
allowlist_constant, cross_field, enum, type_check, range, type_is, numeric_range,
literal_in), and both observed_outcome classes (error=48, dormant_announced=26).

Sampling scope: FULL (74 / 74 confirmed entries verified against source lines).

## Headline counts by class

- Total reviewed: 74
- REAL: 74
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

## Non-REAL entries

NONE.

## Construction-time provenance verification (the key FALSE-CONFIRM risk)

Two families could plausibly be FALSE-CONFIRMs if they were not really enforced at
construction; both were confirmed genuine:

1. New PretrainedConfig `validate_*` methods and the `problem_type` Literal.
   PreTrainedConfig is decorated `@strict(accept_kwargs=True)`
   (configuration_utils.py:119) from huggingface_hub.dataclasses. The @strict decorator
   (huggingface_hub/dataclasses.py) wraps __init__ as `init_with_validate` (lines
   268-276) which calls `cls.validate(self)` AFTER init; validate() runs every public
   `validate_*` method. So validate_output_attentions (line 435), validate_layer_type
   (lines 474, 476), validate_architecture, and validate_token_ids are AUTO-INVOKED at
   construction - they are NOT lazy/method-call-time. The Literal on problem_type (line
   239) is enforced by @strict's `_validate_literal` (dataclasses.py:512) which raises
   TypeError for an out-of-set value. The gate's `error` outcome for problem_type and
   the cross_field validators is therefore legitimate construction-time behaviour, not
   a gate artifact. Entries verified: 28, 30, 41, 42, 43, 49, 57, 58 -> all REAL.

2. GenerationConfig greedy/beam-only-parameter and use-cache warnings.
   v5.7.0 added a provenance gate `_should_warn(outer, inner, user_set_attributes)`
   (configuration_utils.py:65) on the do_sample/num_beams/use_cache families. __init__
   sets user_set_attributes = set(kwargs.keys()) (line 373) and calls
   validate(user_set_attributes=...) (line 491). The confirmed kwargs_positive for every
   such entry passes BOTH the conditioning attr (do_sample / num_beams / use_cache /
   return_dict_in_generate) AND the inner flag, so _should_warn returns True and the
   minor_issues warning fires; kwargs_negative flips the conditioning attr to suppress
   it. The dormant_announced (warn, not raise) outcome is therefore correct and genuine.
   Entries verified: 8, 11, 12, 17, 19, 20, 31, 32, 38, 48, 50, 51, 52, 63, 64, 65, 66,
   67, 68, 70 -> all REAL.

## Outcome-class verification (warn / normalise vs invalid)

All 26 dormant_announced entries map to a `minor_issues[...] = ...` (logger warn) or a
`logger.warning(...)` line, never a raise:
- GenerationConfig minor_issues family (lines 678-783): pad_token_id<0, temperature,
  top_p, min_p, top_h, typical_p, top_k, epsilon_cutoff, eta_cutoff, early_stopping
  (single beam), length_penalty (single beam), cache_implementation/cache_config (use
  cache false), output flags (return_dict false) -> warn. Correct.
- GPTQConfig act_group_aware auto-disable (line 762 `self.act_group_aware = False` +
  logger.warning 763) -> normalise/announced. Correct.
- SinqConfig group_size % 8 (logger.warning line 2000) -> warn. Correct.
- PretrainedConfig id2label/num_labels mismatch (logger.warning line 261) -> warn. Correct.
All 48 error entries map to a `raise ValueError/TypeError` line. No warn-vs-raise
mismatch found.

## Citation accuracy (notably better than v5.6.2)

Every cited (file:line) lands directly on the relevant rule / raise / warning /
annotation line in this venv. Quantization-config citations spot-checked at lines 742,
744, 746, 750, 762, 850, 853, 906, 1016, 1055, 1091, 1355, 1357, 1359, 1361, 1438,
1441, 1447, 1595, 1597, 1666, 1716, 1825, 1827, 1998, 2000 - all exact. No citation
drift of the kind seen in the 5.6.2 passA quant block.

## Allowlist / bound spot-checks (all match source)

GPTQ bits [2,3,4,8] (742); GPTQ group_size >0 or -1 (744); GPTQ damp 0<x<1 (746); GPTQ
dataset [wikitext2,c4,c4-new] (750); AWQ backend full AwqBackend list (853); AWQ format
[gemm,gemv,gemv_fast,llm-awq] (850); Higgs bits [2,3,4] (1355), p [1,2] (1357),
group_size [64,128,256] (1359), hadamard % group_size (1361); Metal bits [2,4,8] (1825),
group_size >0 (1827); SpQR bits==3 (1666); FPQuant forward_dtype [mxfp4,nvfp4] (1438),
backward_dtype [bf16,mxfp8,mxfp4] (1441), transform_init [hadamard,identity,gsr] (1447);
BitNet linear_class [bitlinear,autobitlinear] (1595), quantization_mode [online,offline]
(1597); Sinq method [sinq,asinq] (1998), group_size %8 warn (2000); Quanto weights
[float8,int8,int4,int2] (1055); Eetq weights [int8] (1091); FineGrainedFP8
weight_block_size len==2 & each>0 (1716); Vptq enable_proxy_error==False (1016); Aqlm
in_group_size isinstance int (906). All confirmed.

## Systemic issues

- Duplication across passA/passB (recall artifact, not an error): several constraints
  appear twice with differing predicate_kind labels for the same source rule, e.g.
  num_return_sequences le num_beams (entries 44 `range` + 46 `cross_field`), greedy
  num_return_sequences (45 `presence_conflict` + 47 `cross_field`), pad_token_id (54
  `numeric_range` + 55 `range`), id2label/num_labels mismatch (28 `cross_field` + 43
  `presence_conflict`), single_label num_labels (42 `presence_conflict` + 58
  `cross_field`). Both members of each pair were verified REAL against the same source
  line; the divergent predicate_kind labels are cosmetic taxonomy differences between
  passes, not contradictions.

## Overall trustworthiness verdict

TRUSTWORTHY. 74 / 74 confirmed entries verified REAL against source (fraction verified
REAL = 1.00, full verification not sampled). Every predicate, allowlist, numeric bound,
type/Literal check, cross-field condition, and outcome class matches the cited source
behaviour. The two highest-risk families (new @strict PretrainedConfig validators, and
the _should_warn-gated GenerationConfig warnings) were independently confirmed to be
genuine construction-time invariants, not gate artifacts. Citations are accurate. No
entry is mis-stated, false-confirmed, or fabricated; nothing needs removal.
