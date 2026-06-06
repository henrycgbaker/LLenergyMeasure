# Pass B methodology - transformers v5.8.1 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the transformers config surface by TYPE, not by the public
generate/load call path. Starting from the config class tree, every reachable
config class was read in full and, for each, every validity rule extracted. This
is the COMPLEMENT to a Pass A entry-point walk: it deliberately catches checks
that an entry-point walk does not reach as standalone raises.

- Engine source: `/tmp/tfvenv-5.8.1/lib/python3.12/site-packages/transformers/`
  (`__init__.py` confirms `__version__ = "5.8.1"`).
- Pure source analysis plus in-process construction replay. The gate venv has
  NO torch (verified: `import transformers` prints "PyTorch was not found").

## Type tree enumerated

1. `PreTrainedConfig` base (`configuration_utils.py`). PreTrainedConfig is
   `@strict`-decorated, so its registered validators run at plain construction:
   `validate_output_attentions`, `validate_architecture`, `validate_token_ids`,
   `validate_layer_type` (raising `StrictDataclassClassValidationError`), plus the
   inline `problem_type`/`num_labels` raise and the `id2label`/`num_labels`
   advisory in `__init__`. These are the headline Strategy-B gain: an entry-point
   generate/load walk does not surface them as standalone construction raises.
2. `GenerationConfig.validate()` (`generation/configuration_utils.py`): individual
   attribute gates, the greedy-strip and beam-strip "minor issue" warnings, the
   cache/output-flag advisories, the 8-entry generate-only kwarg rejection tuple,
   the strict-mode aggregation, the `save_pretrained` strict gate, and the
   `from_model_config` return-dict normalisation.
3. The watermarking config family: `BaseWatermarkingConfig` ->
   `WatermarkingConfig` (3 checks) / `SynthIDTextWatermarkingConfig` (1 check),
   each with its own `validate()`.
4. `CompileConfig` and `ContinuousBatchingConfig` (`generation/configuration_utils.py`):
   plain dataclasses, NO validators - confirmed by full read, none emitted.
5. The full `QuantizationConfigMixin` subclass tree (`utils/quantization_config.py`,
   28 classes): every `__init__`/`post_init` raise, assert, normalisation, warning.
   Classes with no validity rules (`CompressedTensorsConfig`, `FbgemmFp8Config`,
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
  recorded with their TRUE default outcome (`warn`), not assumed to raise.
- Uses silent clamps/normalisations (e.g. `AwqConfig` llm-awq backend rewrite,
  `GPTQConfig` act_group_aware auto-disable, `HqqConfig` axis=None->1).

## kwargs replayability discipline

`kwargs_positive` (should trigger) / `kwargs_negative` (should not) were emitted
ONLY where the rule is reachable at plain construction in the torch-free gate venv,
and EVERY emitted pair was replayed in-process to confirm the declared outcome:
- invalid -> positive raises, negative is accepted;
- warn -> positive emits a transformers logger warning, negative is silent;
- normalise -> positive is accepted (with a warning/rewrite), negative is silent.

79 of the 93 invariants carry replayed kwargs pairs; all 79 passed the in-process
replay (0 failures). The 14 dormant entries (carrying `dormant_reason`, no kwargs)
are:
- torch-gated: `BitsAndBytesConfig` post_init type checks and its mutual-exclusion
  NEGATIVE path (the compute_dtype default is `torch.float32`);
  `SynthIDTextWatermarkingConfig` (`requires_backends` PyTorch at construction).
- external-package-gated: `HqqConfig` (hqq >= 0.2.1), `TorchAoConfig` (torchao),
  `GPTQConfig.modules_in_block_to_quantize` (optimum >= 1.15.0).
- nested-object-gated: `VptqLayerConfig.is_indice_packed` (built only inside
  `VptqConfig.config_for_layers`), `GenerationConfig.watermarking_config`
  delegation (needs a watermarking instance).
- lifecycle-gated: `GenerationConfig.save_pretrained` (file/strict gates),
  `PreTrainedConfig.save_pretrained` (file gate), `from_model_config`
  normalisation (classmethod needs a model_config), `validate(strict=True)`
  aggregation (needs the strict gate variant), `PreTrainedConfig.validate_token_ids`
  (needs a populated `text_config.vocab_size`).

All classes carrying replay pairs are importable from the `transformers`
top-level namespace (verified). `native_type` is the dotted importable path; the
gate special-cases `transformers.GenerationConfig` (construct then `.validate()`)
and `transformers.BitsAndBytesConfig`, constructs every other class directly.

## Result

- 93 total invariants, all `net_new` (no prior PoC ground truth exists for
  v5_8_1; only v4_57_3 and v5_6_2 are on disk, and neither is folded here).
- outcome split: 70 invalid, 20 warn, 3 normalise.
- 79 CPU-replayable (verified pairs), 14 dormant.

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

## Version deltas observed vs the v5.6.2 reference GT

- `SpQRConfig` gained `beta1`/`beta2` `isinstance(int)` TypeError checks
  (lines 1660/1662) that v5.6.2 lacked.
- New quant classes present at 5.8.1: `FourOverSixConfig` (no validators).
- GenerationConfig minor-issue message text changed: "is set not to set `True`"
  (v5.6.2) -> "is not set to `True`" (v5.8.1).
- Line numbers across `generation/configuration_utils.py`,
  `configuration_utils.py` and `utils/quantization_config.py` all shifted; every
  citation line here was re-resolved against the installed 5.8.1 source.

## YAML safety

`scope_notes` is a literal block scalar; strings containing colon-space, leading
backticks, brackets or other special characters are quoted. The output was loaded
with `yaml.safe_load` (parses clean) and every replayable pair was executed in
the 5.8.1 interpreter.
