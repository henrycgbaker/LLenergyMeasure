# Adversarial GT review - transformers 5.8.1 invariants

Reviewer: adversarial GT auditor (refute-not-rubber-stamp).
Source under audit: /tmp/tfvenv-5.8.1/lib/python3.12/site-packages/transformers
GT under audit: PILOT_GT.yaml `confirmed` list (n_confirmed = 84).
Citations resolved by id via passA_entrypoint.yaml / passB_classtree.yaml.
No mech-sourced and NO PoC-folded entries in this cell (sources: passA=16, passB=68).

## Scope of verification

Confirmed list = 84 (> 60). Verified ALL 84 entries against source (full, not
sampled). The sample therefore spans every native_type (GPTQConfig, GenerationConfig,
AwqConfig, FPQuantConfig, SpQRConfig, HiggsConfig, MetalConfig, PreTrainedConfig,
VptqConfig, SinqConfig, AqlmConfig, BitNetQuantConfig, FineGrainedFP8Config,
QuantoConfig, EetqConfig), every predicate_kind (normalisation, required, enum,
identity, type_check, cross_field, allowlist_constant, mutual_exclusion_soft, type_is,
numeric_range, literal_in, range, presence_conflict, mutual_exclusion), and both
observed_outcome classes (error=59, dormant_announced=25).

Sampling scope: FULL (84 / 84 confirmed entries verified against source lines).

## Headline counts by class

- Total reviewed: 84
- REAL: 84
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

## Non-REAL entries

NONE.

## Highest-risk families - special scrutiny (all confirmed genuine)

1. New predicate_kind `mutual_exclusion_soft` (entries 16, 43, 63, 74) and
   `mutual_exclusion` (entry 56). These are FAMILY-GRAIN restatements of the
   GenerationConfig.validate conflict blocks, not per-flag entries:
   - 74 sampling_flag_in_greedy (cited line 667 `if self.do_sample is not True:`):
     covers the whole do_sample-not-True minor_issues block (lines 667-716). kpos
     do_sample=False + temperature=0.7 -> minor_issues warn. dormant_announced. REAL.
   - 43 beam_flag_in_single_beam (cited 721): covers the num_beams<=1 block (720-741).
     kpos num_beams=1 + length_penalty=2.0 -> warn. REAL.
   - 16 use_cache_false_conflict (cited 766): covers the use_cache=False cache-arg block
     (762-774). kpos use_cache=False + cache_implementation=static -> warn. REAL.
   - 63 return_dict_in_generate_conflict (cited 780): covers the return_dict-not-True
     output-flag block (777-783). REAL.
   - 56 num_return_sequences_greedy_no_beam (cited 747): line 744-750, greedy without
     beam + num_return_sequences>1 -> raise. error. REAL.
   Each maps to a real construction-time check; predicate text accurately summarises the
   block; outcome (warn for soft, error for hard) matches. These coexist with per-flag
   cross_field duplicates (e.g. 24/44/51/73/77 etc.), all also verified REAL.

2. PreTrainedConfig @strict validators + Literal. PreTrainedConfig is decorated
   `@strict(accept_kwargs=True)` (configuration_utils.py:121). The hub @strict decorator
   (huggingface_hub/dataclasses.py) wraps __init__ as init_with_validate (line 273) which
   calls cls.validate(self) after init, running every public validate_* method. So
   validate_output_attentions (line 437), validate_architecture (line 451), and
   validate_layer_type (lines 476, 478) are AUTO-INVOKED at construction - genuine
   construction-time invariants, not lazy. The problem_type Literal (line 241) is
   enforced by @strict _validate_literal (dataclasses.py:512). Entries 25, 41, 42, 52,
   53, 59, 68 -> all REAL.
   - Entry 25 embed_dim divisible (head_dim*num_heads == embed_dim): kpos head_dim=4,
     num_heads=3, embed_dim=16 -> 12 != 16 raises; kneg 4*4==16 passes. @strict
     accept_kwargs sets these as attrs so hasattr() is True and the check runs. REAL.

3. PreTrainedConfig __post_init__ checks: id2label/num_labels mismatch (line 263
   logger.warning -> dormant_announced, entry 54), single_label requires num_labels>1
   (line 271 raise -> error, entries 69, 70). Both verified against source. REAL.

## Outcome-class verification (warn / normalise vs invalid)

All 25 dormant_announced entries map to a logger.warning / minor_issues assignment, none
to a raise:
- GPTQConfig act_group_aware auto-disable (line 763 logger.warning) -> normalisation. Correct.
- GenerationConfig minor_issues family (lines 637, 678-783): pad_token_id<0, the
  greedy-only and beam-only flags, use_cache-false cache args, return_dict-false output
  flags, and the 4 family-grain soft entries -> warn. Correct.
- SinqConfig group_size %8 (logger.warning line 2000) -> warn. Correct.
- PreTrainedConfig id2label/num_labels mismatch (logger.warning line 263) -> warn. Correct.
All 59 error entries map to a raise ValueError/TypeError. No warn-vs-raise mismatch found.

## Citation accuracy

Every cited (file:line) lands directly on the relevant rule / raise / warning /
condition / annotation line in this venv. Quantization-config citations spot-checked at
lines 742, 744, 746, 750, 763, 850, 853, 906, 917, 1016, 1054, 1091, 1355, 1357, 1359,
1361, 1425, 1429, 1432, 1436, 1438, 1441, 1447, 1595, 1597, 1659, 1660, 1662, 1666,
1668, 1670, 1672, 1714, 1716, 1824, 1826, 1998, 2000 - all exact. configuration_utils.py
citations (GenerationConfig validate + PreTrainedConfig validators / __post_init__) all
exact. No citation drift of the kind seen in 5.6.2 passA quant block.

## Allowlist / bound spot-checks (all match source)

GPTQ bits [2,3,4,8] (742); GPTQ group_size >0 or -1 (744); GPTQ damp 0<x<1 (746); GPTQ
dataset [wikitext2,c4,c4-new] (750); AWQ backend (853) / format (850); SpQR bits==3
(1666), beta1==16 (1668), beta2==16 (1670), bits/beta1/beta2 isinstance int (1659/1660/
1662), shapes dict (1672); Higgs bits [2,3,4] (1355), p [1,2] (1357), group_size
[64,128,256] (1359), hadamard % group_size (1361); Metal bits [2,4,8] (1824), group_size
>0 (1826); FPQuant forward_dtype [mxfp4,nvfp4] (1438), backward_dtype [bf16,mxfp8,mxfp4]
(1441), mxfp4 forward_method [abs_max,quest] (1425), nvfp4 forward_method abs_max (1432),
mxfp4 hadamard [32,64,128] (1429), nvfp4 hadamard [16,32,64,128] (1436), transform_init
[hadamard,identity,gsr] (1447); BitNet linear_class (1595) / quantization_mode (1597);
Sinq method [sinq,asinq] (1998), group_size %8 warn (2000); Quanto weights
[float8,int8,int4,int2] (1054); Eetq weights [int8] (1091); FineGrainedFP8
weight_block_size len==2 (1714) & each>0 (1716); Vptq enable_proxy_error==False (1016);
Aqlm in_group_size isinstance int (906), linear_weights_not_to_quantize list (917);
PreTrainedConfig layer_types ALLOWED_LAYER_TYPES (13-value tuple, exact match to source
lines 62-76); GenerationConfig cache_implementation ALL_CACHE_IMPLEMENTATIONS+(paged,)
(646); early_stopping {None,True,False,"never"} (632); problem_type Literal (241). All
confirmed.

## Systemic issues

- Duplication across passes and grains (recall artifact, not an error): many constraints
  appear twice or three times with differing predicate_kind labels for the SAME source
  rule, e.g. cache_implementation (14 enum + 15 allowlist_constant), compile_config (18
  type_is + 19 type_check), early_stopping (22 literal_in + 23 enum), max_new_tokens (48
  numeric_range + 49 range), pad_token_id (66 range + 67 numeric_range),
  num_return_sequences le num_beams (55 range + 57 cross_field), greedy num_return (56
  mutual_exclusion + 58 cross_field), layer_types allowlist (41 enum + 42
  allowlist_constant), layer_types count (52 range + 53 cross_field), problem_type
  single-label (69 presence_conflict + 70 cross_field), plus the 4 family-grain
  mutual_exclusion_soft entries that overlap the per-flag cross_field warns. Every member
  of every pair/group was verified REAL against the same source line; divergent labels
  are cosmetic taxonomy differences, not contradictions. This inflates n_confirmed but
  does not harm correctness.

## Overall trustworthiness verdict

TRUSTWORTHY. 84 / 84 confirmed entries verified REAL against source (fraction verified
REAL = 1.00, full verification not sampled). Every predicate, allowlist, numeric bound,
type/Literal check, cross-field condition, family-grain conflict summary, and outcome
class matches the cited source behaviour. The highest-risk constructs (the new
mutual_exclusion[_soft] family entries, the @strict PreTrainedConfig validators incl.
validate_architecture and the problem_type Literal) were independently confirmed to be
genuine construction-time invariants, not gate artifacts. Citations are accurate. No
entry is mis-stated, false-confirmed, or fabricated; nothing needs removal.
