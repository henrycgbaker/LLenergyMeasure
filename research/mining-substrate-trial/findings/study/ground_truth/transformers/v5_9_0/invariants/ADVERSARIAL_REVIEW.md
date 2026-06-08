# Adversarial GT review - transformers 5.9.0 invariants

Reviewer: adversarial GT auditor (refute-first).
SOURCE: /tmp/tfvenv-5.9.0/lib/python3.12/site-packages/transformers
GT under review: PILOT_GT.yaml `confirmed` list (n_confirmed = 116).
Citation resolution: by id via passA_entrypoint.yaml / passB_classtree.yaml; `mech`-source
entries carry no passA/passB citation (native_field/predicate_kind/predicate_value all null)
and were verified directly against the source rule their kwargs+match encode, plus live
in-process construction probes (the gate's own model: GenerationConfig constructed in-process,
validate() runs inside __init__ at generation/configuration_utils.py:491).

## Methodology

- Read the three load-bearing source files at every cited line:
  - generation/configuration_utils.py GenerationConfig.validate (lines 612-825) and __init__ (369-491)
  - configuration_utils.py PreTrainedConfig validators (436-482), __post_init__ (243-275),
    output_attentions setter (355-365), ALLOWED_LAYER_TYPES (62-76)
  - utils/quantization_config.py every cited post_init / __init__ raise + the AwqFormat/AwqBackend enums
- Cross-checked every allowlist against the actual enum/constant members in source.
- Ran live GenerationConfig / GPTQConfig / PreTrainedConfig constructions in the 5.9.0 venv to
  confirm the OUTCOME CLASS (raise vs the logger.warning_once "minor issue" advisory that the GT
  records as `dormant_announced`). The torch-free venv matches the gate's source host.

## Sampling scope

Because the sample surfaced a non-real entry, per the thoroughness override this was expanded to
a FULL verification of all 116 confirmed entries. Coverage:
- ALL 15 config classes present (GPTQConfig, GenerationConfig, AwqConfig, FPQuantConfig, SpQRConfig,
  HiggsConfig, MetalConfig, PreTrainedConfig, VptqConfig, SinqConfig, AqlmConfig, BitNetQuantConfig,
  FineGrainedFP8Config, QuantoConfig, EetqConfig).
- BOTH outcome classes: all 73 `error` and all 43 `dormant_announced` (warn/normalise) entries.
- All three sources: passB (68), passA (16), mech (32).
- Every quant-config raise verified against its exact source line; every GenerationConfig.validate
  rule verified against lines 632-801; every PreTrainedConfig validator verified against 436-482/263-271.

## Counts by class

- Total reviewed: 116
- REAL: 115
- MIS-STATED: 0
- FALSE-CONFIRM: 1
- FABRICATED: 0

## Non-REAL entries

### transformers_watermarking_type_watermarking_config_type_not_in_WatermarkingConfig  [FALSE-CONFIRM]
- source: mech; native_type GenerationConfig; kwargs_positive {watermarking_config: 42}; observed_outcome error.
- Claim (per id / constraint_key): a construction-time TYPE check rejecting watermarking_config when
  it is not a WatermarkingConfig instance.
- Source reality: there is NO isinstance/type guard on watermarking_config. __init__
  (generation/configuration_utils.py:418-420) only converts a dict to WatermarkingConfig; a non-dict
  (here int 42) is stored verbatim. validate() (line 658-659) does
  `if self.watermarking_config is not None: self.watermarking_config.validate()`.
- Live probe: GenerationConfig(watermarking_config=42) raises
  `AttributeError: 'int' object has no attribute 'validate'` -- NOT a ValueError from a type check.
- Verdict: the positive "fired" for an UNRELATED reason (an incidental AttributeError from calling
  .validate() on a non-config object), not the claimed type validation. Gate artifact (FALSE-CONFIRM
  mode 2). Net effect (rejecting 42 at construction) happens to be real, but the recorded invariant
  mischaracterises the mechanism and there is no actual type-allowlist rule in the source.

## Systemic observations (not defects)

1. `dormant_announced` is the correct, source-faithful label for the GenerationConfig "minor issue"
   bucket: validate() collects sampling-only / beam-only / cache-when-use_cache-false /
   output-flag-without-return_dict conflicts and pad_token_id<0 into `minor_issues`, then under the
   default (non-strict) path emits a single `logger.warning_once(...)` (line 824) and construction
   SUCCEEDS. Verified live: each such kwargs pair logs the advisory and does not raise. The GPTQConfig
   act_group_aware case is a true silent normalise (forced False + logger.warning, line 761-763).
   All 43 dormant_announced entries are genuine warn/normalise invariants.
2. passA/passB cited line numbers occasionally point at the enclosing `if`/block rather than the exact
   raise (e.g. cache_implementation: passB line 648 vs raise at 647; compile_config: 653 vs if at 652).
   These are correct resolutions, not mis-statements.
3. mech entries duplicate passA/passB rules at a different (mechanical) grain; they carry no citation
   but every one verified against the corresponding source rule and live behaviour -- except [111] above.
4. Entries with kwargs that need extra (model-subclass) attributes (embed_dim divisibility,
   output_attentions-requires-eager) DID confirm at construction via PreTrainedConfig's
   @strict(accept_kwargs=True) decorator, which runs validate_* automatically. Legitimate.

## Overall trustworthiness verdict

TRUSTWORTHY. 115/116 (99.1%) confirmed REAL against source + live construction. The single defect is
a mech-grain FALSE-CONFIRM (watermarking_config type) whose positive fires via an incidental
AttributeError rather than the claimed type validation; the underlying GenerationConfig.validate /
PreTrainedConfig / quantization_config rules are all faithfully stated with correct predicate, field,
allowlist/bound, and outcome class. No fabrications, no mis-stated allowlists or bounds.
Fraction verified REAL: 115/116 (full verification, not sampled).
