# Adversarial GT review - tensorrt 0.20.0 invariants

Reviewer: adversarial GT auditor (refute-first).
Engine: tensorrt   Version: 0.20.0
Source: /tmp/trt-llm-0.20.0/tensorrt_llm
GT under review: PILOT_GT.yaml `confirmed` list (n_confirmed = 14)
Scope: FULL verification of every confirmed entry (no sampling; list is small).

## Method

For each confirmed entry I resolved the citation (passA_entrypoint.yaml /
passB_classtree.yaml) to its file+line in SOURCE, opened that source region, and
checked predicate_kind, predicate_value (allowlist/bound), native_field, severity
and observed_outcome against the actual code. I also checked for FALSE-CONFIRM:
that the positive kwargs fire the CLAIMED rule (not a missing-required-field
TypeError / unrelated error) and that the rule is genuinely construction-time
(auto-run via __post_init__ / pydantic field validation / @validator / enum call),
not a lazy/deferred method-call.

Source facts established (load-bearing):
- llm_args.py:11  `from pydantic import BaseModel, Field, validator` (v1-shim
  validator under pydantic v2 -> runs at field validation).
- llm_args.py:138 `CalibConfig.device: Literal['cuda','cpu']` (only validated
  CalibConfig field).
- llm_args.py:354-356 `BatchingType(StrEnum)` = {STATIC, INFLIGHT}.
- llm_args.py:363-366 `CapacitySchedulerPolicy(StrEnum)` = {MAX_UTILIZATION,
  GUARANTEED_NO_EVICT, STATIC_BATCH}.
- llm_args.py:373-376 `ContextChunkingPolicy(StrEnum)` = {FIRST_COME_FIRST_SERVED,
  EQUAL_PROGRESS}.
- llm_args.py:409-415 SchedulerConfig (pydantic BaseModel) carries
  capacity_scheduler_policy (CapacitySchedulerPolicy, required-ish via enum) and
  context_chunking_policy (Optional[ContextChunkingPolicy]).
- llm_args.py:515-519 LookaheadDecodingConfig.validate_positive_values @validator
  over (max_window_size, max_ngram_size, max_verification_set_size): `if v <= 0:
  raise ValueError("Value must be positive, got {v}")`. __init__ also calls
  _check_fields().
- lora_manager.py:144-156 LoraConfig(DictConversion) @dataclass; __post_init__
  asserts `lora_ckpt_source in ['hf','nemo']`.
- sampling_params.py:267-301 SamplingParams.__post_init__ -> _validate:
  - 284-287 best_of>1 and best_of<n -> raise (beam best_of>=n).
  - 289-296 best_of>1 and _greedy_decoding and not env TLLM_ALLOW_N_GREEDY_DECODING
    -> raise.
  - 298-301 truncate_prompt_tokens is not None and <1 -> raise (>=1).
  - 311-315 _greedy_decoding := not use_beam_search and top_k in (None,1) and
    top_p in (None,0.0).
  - All SamplingParams fields (195-254) have defaults -> no missing-arg TypeError
    risk for the probed kwargs.

## Per-entry verdicts

1. tensorrt_batchingType_enum (source passB; cite llm_args.py:354) - REAL.
   StrEnum {STATIC,INFLIGHT} matches predicate_value [STATIC,INFLIGHT]. Replay
   BatchingType("__bad__") raises ValueError; negative "INFLIGHT" passes. error/
   invalid correct. Fires for the right reason (enum membership).

2. tensorrt_samplingParams_best_of_gt_1_greedy_requires_env (passA; cite
   sampling_params.py:291) - REAL. kwargs_positive {best_of:2, top_k:1}: best_of=2>1,
   n defaults 1 so 284 (best_of<n) is False; top_k==1 -> _greedy_decoding True;
   env unset -> 289-296 raises. kwargs_negative {best_of:2, top_k:50}: greedy
   False -> passes. predicate presence_conflict
   {best_of_gt_1_and_greedy_and_env_unset:true}, error/invalid. Correct, fires for
   the claimed rule (not the best_of>=n rule, which is a separate REJECTED entry).

3. tensorrt_capacitySchedulerPolicy_enum (passA; cite llm_args.py:362) - REAL.
   Allowlist [MAX_UTILIZATION,GUARANTEED_NO_EVICT,STATIC_BATCH] matches source enum
   exactly. SchedulerConfig(capacity_scheduler_policy="__not_a_policy__") raises a
   pydantic ValidationError; "GUARANTEED_NO_EVICT" passes. strenum_in, error/invalid.

4. tensorrt_contextChunkingPolicy_enum (passA; cite llm_args.py:372) - REAL.
   Allowlist [FIRST_COME_FIRST_SERVED,EQUAL_PROGRESS] matches source enum exactly.
   Field is Optional but a non-member string still fails enum coercion -> raises.
   "EQUAL_PROGRESS" passes. strenum_in, error/invalid.

5. tensorrt_calibConfig_device_literal (passA; cite llm_args.py:138) - REAL.
   Literal['cuda','cpu']; predicate literal_in [cuda,cpu] matches. CalibConfig is a
   pydantic BaseModel; CalibConfig(device="__not_a_device__") raises ValidationError,
   "cpu" passes. error/invalid. CPU-constructible.

6. tensorrt_calibconfig_device_in_2_values (mech; same site llm_args.py:138) - REAL.
   Static-miner duplicate of #5 expressed as `in:[cuda,cpu]`; match-fields
   semantics equivalent to the Literal. positive probe (invalid placeholder) fires
   the same pydantic Literal validator, negative "cuda" passes. error/invalid.
   Redundant with #5 but not wrong (see systemic note on duplication).

7. tensorrt_loraConfig_lora_ckpt_source_assert (passA; cite lora_manager.py:154)
   - REAL. __post_init__ assert `lora_ckpt_source in ['hf','nemo']` matches
   literal_in [hf,nemo]. LoraConfig(lora_ckpt_source="__not_a_source__") raises
   AssertionError at construction; "hf" passes. error/invalid. Auto-run via
   dataclass __post_init__ (construction-time, not lazy).

8. tensorrt_raises_max_ngram_size_le_0_positive_values (mech; site
   llm_args.py:515-518) - REAL. Static-miner form of the @validator `v<=0` rule;
   match `max_ngram_size <= 0`, kwargs 0 (fire) / 1 (pass). Same underlying
   validator as #9. error/invalid.

9. tensorrt_lookaheadDecodingConfig_max_ngram_size_positive (passA; cite
   llm_args.py:516) - REAL. @validator covers max_ngram_size; v<=0 -> ValueError.
   predicate range {gt:0} matches `if v<=0 raise`. positive 0 fires, negative 1
   passes. error/invalid. Auto-run at pydantic field validation.

10. tensorrt_raises_max_verification_set_size_le_0_positive_values (mech;
    llm_args.py:515-518) - REAL. Static-miner duplicate of #11; same @validator.

11. tensorrt_lookaheadDecodingConfig_max_verification_set_size_positive (passA;
    cite llm_args.py:516) - REAL. @validator covers max_verification_set_size; same
    v<=0 rule. range {gt:0}, 0 fires / 1 passes. error/invalid.

12. tensorrt_raises_max_window_size_le_0_positive_values (mech; llm_args.py:515-518)
    - REAL. Static-miner duplicate of #13; same @validator.

13. tensorrt_lookaheadDecodingConfig_max_window_size_positive (passA; cite
    llm_args.py:516) - REAL. @validator covers max_window_size; same v<=0 rule.
    range {gt:0}, 0 fires / 1 passes. error/invalid.

14. tensorrt_samplingParams_truncate_prompt_tokens_ge_1 (passA; cite
    sampling_params.py:298) - REAL. `truncate_prompt_tokens is not None and <1 ->
    raise`. predicate range {ge:1} matches. positive 0 fires, negative 1 passes.
    error/invalid. Auto-run via __post_init__ -> _validate.

## Counts by class

- Total reviewed: 14 (FULL, no sampling)
- REAL: 14
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

## Non-REAL entries

None.

## Systemic observations (not defects in the confirmed GT)

- DUPLICATION: 5 of the 14 confirmed entries are static-miner (`source: mech`)
  re-statements of the same source rule already captured by a passA entry, at the
  identical source line:
    * #6 (mech device) == #5 (passA device), llm_args.py:138
    * #8 (mech max_ngram) == #9 (passA), llm_args.py:515
    * #10 (mech max_verification) == #11 (passA), llm_args.py:515
    * #12 (mech max_window) == #13 (passA), llm_args.py:515
  These mech entries carry native_field=null / predicate_kind=null (the miner does
  not resolve the leaf) and survive as distinct constraints because the identity
  key uses the canonical_predicate_value (e.g. "0" vs "{gt=0}"). They are not
  wrong, but the confirmed count of 14 over-states the number of DISTINCT source
  rules: the underlying distinct rules number ~9 (device-literal, lora_ckpt_source,
  3x lookahead-positive, truncate>=1, best_of-greedy-env, BatchingType,
  CapacitySchedulerPolicy, ContextChunkingPolicy). The lookahead positivity is a
  single @validator over 3 fields, counted as 6 entries (3 mech + 3 passA).
  Recommendation: collapse the mech/passA pairs at the leaf grain in synthesis, or
  the recall metric will inflate against a single upstream check.

- CITATION LINE DRIFT: passA/passB cite line numbers are off by 1-2 from the
  installed source (e.g. BatchingType cited 353/354 vs source 354; @validator cited
  516 vs def at 515). Within tolerance; every citation lands inside the correct
  construct and was independently confirmed. No FABRICATION.

- No FALSE-CONFIRM risk realised: every positive-kwarg config (CalibConfig,
  SchedulerConfig, LookaheadDecodingConfig, LoraConfig, SamplingParams,
  BatchingType) is constructible with the probed kwargs alone (all other fields
  default), so each positive fired the named validator rather than a missing-arg
  TypeError or an import/CUDA error. The CUDA-gated LlmArgs._setup / model_post_init
  rules were correctly EXCLUDED from confirmed (they sit in unverified/failed).

## Overall trustworthiness verdict

TRUSTWORTHY. All 14 confirmed entries are REAL: predicate kind, allowlist/bound,
native field, severity and outcome each match the cited source line, and every
positive probe fires the claimed construction-time validation for the right reason.
Fraction verified REAL: 14/14 = 1.00 (full verification, no sampling).

The only caveat is presentational: ~5 confirmed entries are duplicate
constraint-key restatements of the same source rule (mech-vs-passA), so 14 is an
inflated count of distinct upstream checks (~9 distinct rules). This does not make
any confirmed entry false; it is a de-duplication concern for downstream recall
accounting.
