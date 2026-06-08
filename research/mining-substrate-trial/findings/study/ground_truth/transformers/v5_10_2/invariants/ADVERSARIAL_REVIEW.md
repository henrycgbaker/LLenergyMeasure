# Adversarial GT review - transformers 5.10.2 invariants

Reviewer: adversarial GT auditor (refute-first).
SOURCE: /tmp/tfvenv-5.10.2/lib/python3.12/site-packages/transformers
GT under review: PILOT_GT.yaml `confirmed` list (n_confirmed = 85).
Sources present: passB (69), passA (16). NO mech entries in this cell.
Citation resolution: by id via passA_entrypoint.yaml / passB_classtree.yaml; every cited line
opened in the 5.10.2 source.

## Methodology

- Read every cited source line in the three load-bearing files at their 5.10.2 line numbers:
  - generation/configuration_utils.py GenerationConfig.validate (611-824) + __init__ validate call (490)
  - configuration_utils.py PreTrainedConfig validators (436-484), __post_init__ (243-275),
    output_attentions setter (355-365), ALLOWED_LAYER_TYPES (62-76)
  - utils/quantization_config.py every cited post_init/__init__ raise + AwqFormat/AwqBackend enums
- Verified every allowlist against the actual enum/constant members in source.
- Ran live in-process construction of GenerationConfig + all 14 quant classes + PreTrainedConfig in
  the 5.10.2 venv (confirmed torch-unavailable, matching the gate's no-torch model) to verify the
  OUTCOME CLASS: hard raise vs the logger.warning_once "minor issue" advisory the GT records as
  `dormant_announced`, vs silent normalise.

## Sampling scope

FULL verification of all 85 confirmed entries (no sampling shortfall; the entire space was probed):
- ALL 16 config classes: GPTQConfig, GenerationConfig, AwqConfig, FPQuantConfig, SpQRConfig,
  AutoRoundConfig, HiggsConfig, MetalConfig, PreTrainedConfig, VptqConfig, SinqConfig, AqlmConfig,
  BitNetQuantConfig, FineGrainedFP8Config, QuantoConfig, EetqConfig.
- BOTH outcome classes: all 60 `error` and all 25 `dormant_announced` (warn/normalise).
- Both sources: passA (16), passB (69).
- Every GenerationConfig.validate rule verified against lines 631-799; every PreTrainedConfig
  validator against 436-484 / 263-271; every quant raise against its exact 5.10.2 line.

## Counts by class

- Total reviewed: 85
- REAL: 85
- MIS-STATED: 0
- FALSE-CONFIRM: 0
- FABRICATED: 0

## Non-REAL entries

None.

## Spot-check of new / version-drifted entries (all REAL)

- AutoRoundConfig.bits [2,3,4,8] (L240-241) and group_size {gt:0 or -1} (L242-243): not in 5.9.0
  confirmed; live RAISE ValueError confirmed.
- FineGrainedFP8Config.scale_fmt ('float','ue8m0') (L1683-1684): NEW field at 5.10.2; live RAISE
  ValueError confirmed. The GT predicate {in: [float, ue8m0]} matches the source tuple exactly.
- validate_layer_type now loops over ["layer_types","mlp_layer_types"] (L474); the layer_types
  allowlist + count-match entries still fire as recorded (StrictDataclassClassValidationError).

## Systemic observations (not defects)

1. `dormant_announced` is the source-faithful label for the GenerationConfig minor_issues bucket
   (sampling-only / beam-only / cache-when-use_cache-false / output-flag-without-return_dict /
   pad_token_id<0). validate() under the default non-strict path emits one logger.warning_once
   (L823) and construction SUCCEEDS; verified live for all 13 such entries. GPTQConfig act_group_aware
   (L761-763) and SinqConfig gs%8 (L1967-1969) are genuine warn/normalise.
2. PreTrainedConfig validate_* raises surface as StrictDataclassClassValidationError (and
   problem_type as StrictDataclassFieldValidationError) via the @strict(accept_kwargs=True)
   decorator at construction; passA/passB explicitly state the gate treats the wrapped raise as the
   FIRE. The __post_init__ single_label/num_labels raise is a bare ValueError. All observed live.
3. passA/passB cited lines occasionally point at the enclosing `if` rather than the exact raise
   (off by 1, e.g. cache_implementation L646 vs raise L646/647). Correct resolutions, not mis-statements.
4. GenerationConfig.validate logic is byte-identical to 5.9.0; only line numbers shifted by ~1-2.
   The quant configs retain identical predicates with re-resolved 5.10.2 line numbers; every allowlist
   (AwqBackend 13 members, AwqFormat 4, ALLOWED_LAYER_TYPES 13) matches source exactly.

## Overall trustworthiness verdict

FULLY TRUSTWORTHY. 85/85 (100%) confirmed REAL against source + live construction. Every predicate,
field, allowlist/bound, severity, and outcome class matches the cited 5.10.2 source line. No
fabrications, no mis-stated bounds or allowlists, no false-confirms. (Notably, the 5.9.0
watermarking_config mech false-confirm does not recur here - this cell carries no mech-source entries.)
Fraction verified REAL: 85/85 (full verification, not sampled).
