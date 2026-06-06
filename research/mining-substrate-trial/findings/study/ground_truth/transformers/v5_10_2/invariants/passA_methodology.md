# Pass A - entry-point / call-graph walk methodology (transformers 5.10.2)

Engine source: `/tmp/tfvenv-5.10.2/lib/python3.12/site-packages/transformers/`
(confirmed `__version__ = "5.10.2"`).

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of
a two-pass bake-off; a sibling pass (`passB_classtree.yaml`) does a
class-hierarchy walk. Goal: maximise recall of construction-time validation
invariants reachable from public, user-facing entry points that a benchmark
harness actually constructs. transformers is a third engine alongside
tensorrt-llm and vllm; this pass ignores their APIs.

## Traversal (what I walked)

Starting roots (public surface a generation/load harness constructs):

1. `transformers.GenerationConfig(...)` -> `__init__`
   (generation/configuration_utils.py:368) which `kwargs.pop(...)`s each known
   field, setattrs the rest, then calls
   `self.validate(user_set_attributes=...)` at line 490. `validate()`
   (line 611) holds every range / allowlist / mutual-exclusion / sampling-vs-
   greedy / beam-only check, plus the `watermarking_config.validate()` dispatch
   (line 658) and the `generate`-only-argument rejection loop (line 795).
2. `WatermarkingConfig.validate()` (line 1420), reached transitively because
   `__init__` converts a dict `watermarking_config` to a `WatermarkingConfig`
   (line 418). Replayable through `GenerationConfig(watermarking_config={...})`.
3. `transformers.PreTrainedConfig(...)` (alias `PretrainedConfig`,
   configuration_utils.py:1352). At 5.10.2 it is a huggingface_hub `@strict`
   dataclass (`@strict(accept_kwargs=True)`, line 121). Construction runs
   `init_with_validate` (huggingface_hub/dataclasses.py:273): the wrapped
   `__init__` assigns each field via `@strict`'s `__setattr__` (line 139 - runs
   the per-field TYPE validator, enforcing the `problem_type` Literal), then
   `__post_init__` (line 243), then every `validate_*` class validator
   (`validate.py` loop, line 247).
4. `transformers.BitsAndBytesConfig(...)` -> `__init__`
   (utils/quantization_config.py:439) + `post_init()` (line 520).

## Method

- Read `GenerationConfig.validate()` in full and enumerated every `raise`,
  every `minor_issues[...] = ...` (the warn bucket), and the strict-mode
  promotion at line 814. Classified each by true outcome at the source line:
  invalid (raise) | warn (logger.warning_once via minor_issues) | normalise.
- Read `PreTrainedConfig.__post_init__`, every `@property`/setter, and every
  `validate_*` method (`validate_output_attentions`, `validate_architecture`,
  `validate_token_ids`, `validate_layer_type`). Resolved the `@strict`
  machinery in `huggingface_hub/dataclasses.py` to confirm WHICH wrapper class
  each raise surfaces through (field-type -> `StrictDataclassFieldValidationError`,
  validate_* -> `StrictDataclassClassValidationError`, `__post_init__` ->
  bare `ValueError`).
- Read `BitsAndBytesConfig.__init__` + `post_init` and every load-time `raise`.
- EMPIRICALLY VERIFIED every replayable predicate in this exact venv: I
  constructed each native type with the recorded `kwargs_positive` (confirmed it
  FIRES: raises for invalid, emits the minor-issues warning for warn) and with
  `kwargs_negative` (confirmed accepted). All 12 GenerationConfig +
  WatermarkingConfig raises, all 5 GenerationConfig warns, the PreTrainedConfig
  problem_type/layer_types/num_labels checks, and all bnb checks were verified.

## Idiom note (transformers vs tensorrt/vllm)

transformers validates LATER and SOFTER. Sampling-only flags set while
`do_sample` is not True, beam-only flags set while `num_beams<=1`, negative
`pad_token_id`, `use_cache=False` cache conflicts, and
`return_dict_in_generate` conflicts are all WARN (minor_issues ->
`logger.warning_once`, line 823), NOT raise - unless `validate(strict=True)`
(only invoked by `save_pretrained`, line 854), which promotes the whole bucket
to a single raise. The minor-issue warns are gated by `_should_warn`
(line 64): they fire only for user-set flags, never for values inherited from a
model's `generation_config.json`. The gate passes flags explicitly, so the
warns fire.

## Replay model for the downstream gate

- GenerationConfig: the gate calls `.validate()` after construction; validation
  ALSO runs inside `__init__` (line 490), so plain construction already fires
  the rule. All 12 raises + 5 warns + 3 watermarking raises are
  construct-replayable on a CPU-only host (no torch needed for GenerationConfig).
- PreTrainedConfig: validators run at construction. A `validate_*` raise is
  WRAPPED in `StrictDataclassClassValidationError` (`__cause__` = the quoted
  `ValueError`); a Literal/field-type violation is wrapped in
  `StrictDataclassFieldValidationError` (`__cause__` a `TypeError`); the
  `__post_init__` raises (`problem_type` single_label, `num_labels`/`id2label`
  warn) are bare. The gate must treat the wrapped raise as the FIRE.
- BitsAndBytesConfig: the task gate environment is declared CPU-only with NO
  torch. `quantization_config.py` imports torch at module load and `__init__`
  assigns `torch.float32`/`torch.uint8` (lines 468/477) before the
  quant_storage/compute_dtype checks; `post_init` type-checks against
  `torch.dtype`. Only the `load_in_4bit`/`load_in_8bit` conflict (line 455)
  raises BEFORE any torch use and is replayable in a no-torch gate. The other
  three bnb entries are recorded `dormant` for the gate.

## Source host vs gate environment (a real 5.10.2 difference)

The 5.8.1 GT marked all bnb checks dormant because its source host had no torch.
This 5.10.2 source host HAS torch (2.5.1+cpu, resolved from the project .venv),
so I re-verified that every bnb check fires exactly as declared
(`bnb_4bit_quant_storage='__bad__'` -> ValueError; `bnb_4bit_compute_dtype=123`
-> ValueError; `llm_int8_threshold='x'` -> TypeError; `load_in_4bit='yes'` ->
TypeError). The `dormant_reason` flags are kept anyway because they track the
GATE's stated no-torch contract (the env that replays), not the source host.

## Net-new vs prior GT

No PoC/prior GT exists for transformers v5_10_2 (the task specified all entries
`net_new`). Every entry was independently re-derived against the 5.10.2 source
and re-cited to that version's lines; nothing was copied from another
version/engine without re-resolving.

## Drift caught vs the sibling 5.8.1 GT (re-derived, not folded)

- Line drift throughout `GenerationConfig.validate()` (e.g. early_stopping raise
  632 -> 631; max_new_tokens 635 -> 633; validate() entry 612 -> 611;
  `__init__` validate call 491 -> 490; minor-issues warn block 824 -> 823).
- `validate_layer_type` now loops over BOTH `layer_types` and `mlp_layer_types`
  (configuration_utils.py:474) and the count-mismatch message interpolates the
  field name: "must be equal to the number of \`layer_types\`" (was the static
  "number of layer types"). Re-cited and re-worded the message_template.
- `ALLOWED_LAYER_TYPES` grew new members (compressed_sparse_attention,
  heavily_compressed_attention, conv, ...); the predicate (must-all-be-in the
  constant) is unchanged, value list noted.
- bnb checks: replayability flipped (torch present on host) - re-verified but
  kept dormant for the gate's no-torch contract, as above.

## Coverage

- Full `GenerationConfig.validate()`: 8 hard raises (early_stopping,
  max_new_tokens, cache_implementation, compile_config, num_return_sequences x2,
  generate-only-args, strict-mode promotion) + 5 minor-issue warns
  (pad_token_id, sampling-in-greedy, beam-in-single-beam, use_cache conflict,
  return_dict_in_generate conflict).
- `WatermarkingConfig.validate()`: 3 raises (seeding_scheme, greenlist_ratio,
  context_width).
- `PreTrainedConfig`: problem_type semantic raise + problem_type Literal +
  layer_types allowlist + layer_types count + validate_architecture head_dim
  (dormant) + output_attentions-vs-eager (dormant) + token_ids out-of-vocab warn
  (dormant) + num_labels/id2label warn.
- `BitsAndBytesConfig`: load conflict (replayable) + quant_storage allowlist +
  compute_dtype type + post_init type checks (all dormant for the gate).

## Blind spots (what the class-hierarchy walk should catch that this pass did not)

1. Validators on model-subclass configs never reached from the public
   GenerationConfig/PreTrainedConfig/BitsAndBytesConfig constructors. A bare
   `PreTrainedConfig` skips checks that need subclass-only fields
   (`vocab_size` for token-id validation, `head_dim`/`num_heads`/`embed_dim` for
   architecture validation). The hierarchy walk over concrete `*Config`
   subclasses would surface those.
2. `SynthIDTextWatermarkingConfig.validate()` (sampling_table_size < 2**24,
   line 1530) is a sibling watermarking class not reachable through the dict ->
   `WatermarkingConfig` conversion path (that path only builds the plain
   `WatermarkingConfig`); a hierarchy walk routing to it would add one more
   raise.
3. Other quantization configs (`GPTQConfig`, `AwqConfig`, etc. in
   quantization_config.py - GPTQ `post_init` bits/group_size/damp_percent
   raises at lines 742-771) are constructed only when a user passes that
   `quantization_config`; not on the default generate/load path walked here.
4. `@strict` per-field TYPE validators on every other PreTrainedConfig field
   (dtype, is_encoder_decoder, etc.) fire on bad types but were not enumerated
   individually - only `problem_type` (the one with a non-trivial Literal) was
   recorded. The hierarchy walk that enumerates fields would catalogue the rest.
5. Deprecated module-level `layer_type_validation` (line 1355) duplicates the
   validator logic with a different message; only reachable if user code calls
   it directly, out of the construction call graph.
