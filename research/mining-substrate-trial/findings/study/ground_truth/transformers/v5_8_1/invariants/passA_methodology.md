# Pass A - entry-point / call-graph walk methodology (transformers 5.8.1)

Engine source: `/tmp/tfvenv-5.8.1/lib/python3.12/site-packages/transformers/`
(confirmed `__version__ = "5.8.1"` in `transformers/__init__.py`). This venv has
NO torch installed (import emits "PyTorch was not found"); all replay checks
below were run with the venv interpreter `/tmp/tfvenv-5.8.1/bin/python` under
that no-torch constraint, matching the downstream CPU-only / no-torch gate.

Output: `passA_entrypoint.yaml`. This is the entry-point/call-graph half of a
two-pass bake-off (a sibling pass does a class-hierarchy walk). transformers is
a third engine alongside tensorrt-llm and vllm; their APIs were ignored.

## Traversal (what I walked)

Public surface a benchmark harness actually constructs and validates:

1. `transformers.GenerationConfig(...)` -> `__init__`
   (`generation/configuration_utils.py:369`). `__init__` snapshots
   `user_set_attributes`, pops every known generation knob, sets unknown keys
   via `setattr`, converts a dict `watermarking_config` into a
   `WatermarkingConfig`, and at line 491 calls
   `self.validate(user_set_attributes=...)`. So plain construction already runs
   validation; the gate's "construct then call `.validate()`" path is consistent
   with the in-`__init__` call.
2. `GenerationConfig.validate()` (line 612) - the load-bearing surface. Read in
   full: the per-attribute raises (early_stopping, max_new_tokens,
   cache_implementation, compile_config), the `num_return_sequences` mutual
   exclusions, the `generate`-only-argument rejection loop, the watermarking
   dispatch, and the `minor_issues` bucket (sampling-only flags in greedy,
   beam-only flags in single-beam, pad_token_id<0, use_cache=False conflicts,
   return_dict_in_generate conflicts) that WARNS by default and RAISES under
   `strict=True`.
3. `WatermarkingConfig.validate()` (line 1421), reached transitively from
   `GenerationConfig.validate()` line 659. Replayable through
   `GenerationConfig(watermarking_config={...})`.
4. `transformers.BitsAndBytesConfig(...)` -> `__init__`
   (`utils/quantization_config.py:439`) + `post_init()` (line 520) + the
   `load_in_4bit` / `load_in_8bit` property setters.
5. `transformers.PreTrainedConfig(...)` (alias `PretrainedConfig`). At 5.8.1
   this is a `huggingface_hub.dataclasses.@strict` dataclass. `@strict` collects
   every `validate_*` method and runs them at construction
   (`huggingface_hub/dataclasses.py:276` `init_with_validate` ->
   `validate(self)` -> each `validate_*`). The reachable validators are
   `validate_output_attentions`, `validate_architecture`, `validate_token_ids`,
   `validate_layer_type`. `__post_init__` (line 243) carries additional raises
   (problem_type/num_labels) and warns (num_labels vs id2label).

## Method

- Grepped `generation/configuration_utils.py`, `configuration_utils.py`, and
  `utils/quantization_config.py` for `raise`, `assert`, `logger.warning`,
  `warning_once`, `warnings.warn`, `def validate`, `def __init__`,
  `__post_init__`, and `@property` setters; read each hit in source context and
  classified outcome as invalid (raise) | warn (logger.warning) | normalise
  (silent clamp).
- Confirmed the `@strict` validation contract by reading
  `huggingface_hub/dataclasses.py` (validators are auto-run at init; raises are
  wrapped in `StrictDataclassClassValidationError` / field validation errors,
  with the quoted `ValueError` as `__cause__`).
- VERIFIED replayability by constructing each replayable type in the venv
  interpreter and observing the declared FIRE: `early_stopping=1.5`,
  `max_new_tokens=0`, `cache_implementation` bad, `num_return_sequences` greedy
  and >num_beams, `logits_processor=[]`, watermarking bad seeding_scheme,
  `problem_type=single_label_classification`+`num_labels=1`, bad/short
  `layer_types`, and the sampling-flag-in-greedy WARN (observed
  `logger.warning_once` output). BitsAndBytesConfig confirmed: the 4bit/8bit
  conflict raises pre-torch; every other bnb check NameErrors without torch.

## Idiom note (transformers vs tensorrt/vllm)

transformers validates LATER and SOFTER. A large share of GenerationConfig
checks WARN (collected into `minor_issues`, emitted once via
`logger.warning_once` at line 824) rather than raise. The same conflict that
WARNS under default `validate(strict=False)` RAISES under
`validate(strict=True)` (line 816) - `save_pretrained()` uses strict mode. The
recorded `outcome` is the DEFAULT-path behaviour the gate observes (warn),
except for the dedicated strict-mode entry. No pure silent-normalisation
(clamp) invariant was found on the entry-point surface at 5.8.1: the
`_get_default_generation_params` defaults are a static helper, not applied as a
constructor clamp; `BitsAndBytesConfig` None->torch.uint8/float32 defaulting is
torch-gated and was classified under the dormant bnb entries rather than as a
replayable normalise.

## Coverage

- Full `GenerationConfig.validate()` raise + warn set (16 entries on
  `transformers.GenerationConfig`, including the watermarking dispatch).
- `PreTrainedConfig` `@strict` validators + `__post_init__` raises/warns (8
  entries).
- `BitsAndBytesConfig` `__init__` / `post_init` / setter checks (4 entries, all
  torch-gated except the 4bit/8bit conflict's positive raise).

## CPU-replayable vs dormant (for the downstream gate)

Replayable (20): all GenerationConfig and WatermarkingConfig raises and warns,
the two PreTrainedConfig `__post_init__`/`validate_layer_type` raises and the
problem_type Literal and num_labels-vs-id2label warn. These construct CPU-only,
no torch, no model dir.

Dormant (8) and why:
- `generationConfig_strict_minor_issue_raises` - needs `validate(strict=True)`,
  which the construct-then-validate gate path does not pass.
- `pretrainedConfig_validate_architecture_head_dim`,
  `pretrainedConfig_output_attentions_requires_eager`,
  `pretrainedConfig_validate_token_ids_out_of_vocab_warns` - need
  model-subclass-specific attrs (head_dim/num_heads/embed_dim; a non-eager
  attn_implementation set before output_attentions; a text_config carrying
  vocab_size) that a bare `PreTrainedConfig` does not have.
- All non-conflict `BitsAndBytesConfig` checks - `quantization_config.py`
  assigns `torch.float32` / `torch.uint8` early in `__init__` (lines 467-487),
  which NameErrors on this no-torch host before the allowlist / type / post_init
  checks are reached. The 4bit/8bit conflict raise (line 456) fires pre-torch
  and is real, but its `kwargs_negative` also needs torch, so that entry is
  flagged dormant for the no-torch host.

## Replay gotcha for the gate

PreTrainedConfig validator raises are WRAPPED by huggingface_hub: the visible
exception type is `StrictDataclassClassValidationError` (or the field-validation
variant for the `problem_type` Literal), not a bare `ValueError`. The quoted
`ValueError` text is the `__cause__`. The gate must treat the wrapped raise as
the FIRE. `GenerationConfig` and `WatermarkingConfig` raises are bare
`ValueError`s; the `problem_type`+`num_labels` raise from
`PreTrainedConfig.__post_init__` is also a bare `ValueError` (raised inside
`__post_init__`, before/outside the field-level strict wrapping).

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. Per-model `PreTrainedConfig` subclass `__init__` / `validate_*` overrides
   (every model config subclasses PreTrainedConfig and may add field checks);
   the entry-point walk only routes through the base class.
2. `SynthIDTextWatermarkingConfig.validate()` (line 1526, sampling_table_size <
   2**24) - reachable only when a SynthID config is supplied; the default
   harness path constructs the plain `WatermarkingConfig`.
3. `CompileConfig` / `ContinuousBatchingConfig` deep validators and the
   `decide_use_cuda_graphs` warnings (lines 1781-1833) - torch/CUDA gated, out
   of scope for the no-torch host.
4. Other quantization configs (GPTQConfig, AwqConfig, etc.) and the
   `QuantizationConfigMixin` base - not on the default generate/load path.
5. Field-level `@strict` type enforcement across all PreTrainedConfig typed
   fields (only `problem_type`'s Literal was surfaced explicitly); an MRO walk
   would enumerate every annotated field's strict constraint.
