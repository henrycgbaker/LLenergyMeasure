# Pass B methodology - transformers v5.10.2 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the transformers config surface by TYPE, not by the public
generate/load call path. Starting from the config class tree, every reachable
config class was read in full and, for each, every validity rule extracted. This
is the COMPLEMENT to a Pass A entry-point walk: it deliberately catches checks
that an entry-point walk does not reach as standalone raises.

- Engine source: `/tmp/tfvenv-5.10.2/lib/python3.12/site-packages/transformers/`
  (`__init__.py` confirms `__version__ = "5.10.2"`).
- Pure source analysis plus in-process construction replay. The gate venv has
  NO torch (verified: `import transformers` prints "PyTorch was not found").

## Type tree enumerated

1. `PreTrainedConfig` base (`configuration_utils.py`). PreTrainedConfig is
   `@strict`-decorated, so its registered validators run at plain construction:
   `validate_output_attentions`, `validate_architecture`, `validate_token_ids`,
   `validate_layer_type` (raising `StrictDataclassClassValidationError`), plus the
   inline `problem_type`/`num_labels` raise and the `id2label`/`num_labels`
   advisory in `__init__`, and the two `save_pretrained` gates (file-path and
   generation-params-on-config). These are the headline Strategy-B gain: an
   entry-point generate/load walk does not surface them as standalone construction
   raises.
2. `GenerationConfig.validate()` (`generation/configuration_utils.py`): individual
   attribute gates, the greedy-strip and beam-strip "minor issue" warnings, the
   cache/output-flag advisories, the 8-entry generate-only kwarg rejection tuple,
   the strict-mode aggregation, the `save_pretrained` strict gate, and the
   `from_model_config` return-dict normalisation.
3. The watermarking config family: `BaseWatermarkingConfig` ->
   `WatermarkingConfig` (3 checks) / `SynthIDTextWatermarkingConfig` (1 check),
   each with its own `validate()`.
4. `CompileConfig` and `ContinuousBatchingConfig` (`generation/configuration_utils.py`):
   `CompileConfig` is a plain dataclass with NO validators (confirmed, none
   emitted). `ContinuousBatchingConfig.__post_init__` has one warning gated on
   `WORLD_SIZE > 1` (env-dependent, dormant, not emitted as a replayable pair).
5. The full `QuantizationConfigMixin` subclass tree (`utils/quantization_config.py`,
   24 classes): every `__init__`/`post_init` raise, assert, normalisation, warning.
   Classes with no construction-time validity rules (`FbgemmFp8Config`,
   `QuarkConfig`, `Mxfp4Config`, `FourOverSixConfig`) were read and excluded.

## Extraction per class

For each class the following were harvested and read in source context:
1. `raise` / `assert` statements (`grep -nE "raise |assert "`).
2. `logger.warning(_once)` / `logger.info` advisories.
3. Silent normalisations (a field reassigned to a corrected value).
4. Enum/Literal membership and numeric-range checks inside `post_init`/`__init__`.
5. Overridden validators (a child redefining a parent's check with a different
   predicate).

`outcome` derived from severity: raise/error -> `invalid`; warn/advisory -> `warn`;
silent value rewrite -> `normalise`.

## transformers idiom: validates LATER and SOFTER

Unlike tensorrt (which mostly raises at pydantic construction), transformers:
- Defers most GenerationConfig combination checks to a `.validate()` method.
- Records combination problems as "minor issues" that only WARN by default and
  raise only under `validate(strict=True)` or `save_pretrained`. These are
  recorded with their TRUE default outcome (`warn`), not assumed to raise. (Direct
  `.validate()` calls leave `user_set_attributes=None`, so the provenance filter
  `_should_warn` treats all set attributes as user-set and the warnings fire.)
- Uses silent clamps/normalisations (`AwqConfig` llm-awq backend rewrite,
  `GPTQConfig` act_group_aware auto-disable, `GenerationConfig.from_model_config`
  return-dict, `CompressedTensorsConfig` run_compressed auto-disable).

## kwargs replayability discipline

`kwargs_positive` (should trigger) / `kwargs_negative` (should not) were emitted
ONLY where the rule is reachable at plain construction in the torch-free gate venv,
and EVERY emitted pair was replayed in-process to confirm the declared outcome:
- invalid -> positive raises, negative is accepted;
- warn -> positive emits a transformers logger warning, negative is silent;
- normalise -> positive is accepted (with a warning/rewrite), negative is silent.

80 of the 95 invariants carry replayed kwargs pairs; all 80 passed the in-process
replay (0 failures). The 15 dormant entries (carrying `dormant_reason`, no kwargs)
are:
- torch-gated: `BitsAndBytesConfig` post_init type checks and its mutual-exclusion
  NEGATIVE path (the compute_dtype default is `torch.float32`, verified NameError);
  `SynthIDTextWatermarkingConfig` (`requires_backends` PyTorch at construction,
  verified ImportError).
- external-package-gated: `HqqConfig` (hqq >= 0.2.1, verified ImportError),
  `TorchAoConfig` (torchao, verified ValueError), `GPTQConfig.modules_in_block_to_quantize`
  (optimum >= 1.15.0), `CompressedTensorsConfig.run_compressed` normalisation
  (compressed-tensors >= 0.15.0, verified ImportError at __init__).
- nested-object-gated: `VptqLayerConfig.is_indice_packed` (built only inside
  `VptqConfig.config_for_layers`), `GenerationConfig.watermarking_config`
  delegation (needs a watermarking instance).
- lifecycle-gated: `GenerationConfig.save_pretrained` (file + strict gates),
  `PreTrainedConfig.save_pretrained` (file gate + generation-params gate),
  `from_model_config` normalisation (classmethod needs a model_config),
  `validate(strict=True)` aggregation (needs the strict gate variant),
  `PreTrainedConfig.validate_token_ids` (needs a populated `text_config.vocab_size`).

All classes carrying replay pairs are importable from the `transformers`
top-level namespace (verified). `native_type` is the dotted importable path; the
gate special-cases `transformers.GenerationConfig` (construct then `.validate()`)
and `transformers.BitsAndBytesConfig`, constructs every other class directly.

## Result

- 95 total invariants, all `net_new` (no prior PoC ground truth exists for
  v5_10_2 under `findings/ground_truth/transformers/`).
- outcome split: 71 invalid, 20 warn, 4 normalise.
- 80 CPU-replayable (verified pairs, 0 failures), 15 dormant.

### Strategy-B-specific gains over an entry-point walk

The `PreTrainedConfig` base-class `@strict` validators
(`validate_output_attentions`, `validate_architecture`, `validate_layer_type`
x2, plus the `problem_type` and `id2label` checks in `__init__`) are pure
class-tree invariants: they fire at bare config construction and are invisible to
a walk that only follows `generate()` / `from_pretrained()`. The `AwqConfig`
post_init OVERRIDE (it is a `GPTQConfig` subclass but replaces post_init entirely,
so the inherited bits/group_size/damp checks do NOT run at AwqConfig construction
- only format/backend) is a textbook overridden-validator case a call-path walk
would mis-attribute.

## Version deltas observed vs the v5.8.1 reference (sibling pass-B GT)

- `PreTrainedConfig.validate_layer_type` now loops over BOTH `layer_types` and
  `mlp_layer_types` (configuration_utils.py:474), and the count-mismatch message
  was reworded to embed the field name
  (`must be equal to the number of \`{layer_types}\``).
- `PreTrainedConfig.save_pretrained` gained a save-time gate (line 514) refusing
  to persist generation parameters on the model config.
- `FineGrainedFP8Config` gained a `scale_fmt` allowlist check (line 1684:
  `scale_fmt in ('float', 'ue8m0')`) that v5.8.1 lacked.
- `SinqConfig.post_init` was restructured: nbits/group_size/tiling_mode are now
  int/str-coerced first, and the TypeError message text changed from
  "must be a float" to "must be convertible to an int" (the non-coercible failure
  path raises a ValueError at the `int()` call before the isinstance check).
- Line numbers across `generation/configuration_utils.py`,
  `configuration_utils.py` and `utils/quantization_config.py` all shifted; every
  citation line here was re-resolved against the installed 5.10.2 source.

## YAML safety

`scope_notes` is a literal block scalar; strings containing colon-space, leading
backticks, brackets or other special characters are quoted. The output was loaded
with `yaml.safe_load` (parses clean) and every replayable pair was executed in
the 5.10.2 interpreter (80/80 passed).
