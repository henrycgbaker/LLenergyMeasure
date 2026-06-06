# Pass A - entry-point / call-graph walk methodology (transformers 5.7.0)

Engine source: `/tmp/tfvenv-5.7.0/lib/python3.12/site-packages/transformers`
(confirmed `__version__ = "5.7.0"` in `transformers/__init__.py`).

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of
a two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal: maximise
recall of construction-time validation invariants reachable from the public,
user-facing surface a benchmark harness actually constructs. transformers is a
third engine alongside tensorrt-llm and vllm; their APIs are out of scope here.

## Traversal (what I walked)

Public roots a generation harness constructs, followed to every
construction-time validation site:

1. `transformers.GenerationConfig(...)` -> `__init__`
   (`generation/configuration_utils.py:491` calls `self.validate(...)`) ->
   `validate(strict=False)` (line 612). This is the load-bearing surface:
   - 1.1 individual-attribute raises: `early_stopping` allowlist (633),
     `max_new_tokens > 0` (635), `cache_implementation` allowlist incl. `paged`
     (647), `compile_config` isinstance (653).
   - `pad_token_id >= 0` (636) is a warn, not a raise.
   - 1.4 delegated `watermarking_config.validate()` (659).
   - 2.1 sampling-only-flag-when-greedy strips (`do_sample is not True`):
     temperature/top_p/min_p/top_h/typical_p/top_k/epsilon_cutoff/eta_cutoff
     (678-716) - all WARN via `minor_issues`.
   - 2.2 beam-only-flag-when-single-beam strips: early_stopping/length_penalty
     (731-739) - WARN.
   - 2.4 `num_return_sequences` raises (greedy>1 at 747; > num_beams at 756).
   - 2.5 cache-args-when-`use_cache=False` (770-772) - WARN over the 2-tuple
     `(cache_implementation, cache_config)`.
   - 2.6 extra_output_flags without `return_dict_in_generate` (777-780) - WARN
     over the 4-tuple `(output_attentions, output_hidden_states, output_scores,
     output_logits)`.
   - 3. generate-only-kwarg rejection (786-800) - raises over the 8-tuple
     `(logits_processor, stopping_criteria, prefix_allowed_tokens_fn,
     synced_gpus, assistant_model, streamer, negative_prompt_ids,
     negative_prompt_attention_mask)`.
   - strict-mode promotion (816): with `strict=True`, every minor_issue is
     raised; `save_pretrained` (855/862) forces strict validate + a
     file-not-dir AssertionError.
   - `from_model_config` (1262): silent normalisation of
     `return_dict_in_generate -> True` when an output flag is set.
2. `transformers.BitsAndBytesConfig(...)` (`utils/quantization_config.py:439`)
   - the downstream gate special-cases this class. `__init__` mutual-exclusion
     (455), quant_storage allowlist (480); `post_init` type checks (520-548,
     all `TypeError`).
3. `transformers.PretrainedConfig` (alias of `PreTrainedConfig`,
   `configuration_utils.py:121`). In v5 this is a `@strict @dataclass`:
   `__post_init__` (241) carries the `problem_type`/`num_labels` raise (269) and
   the `id2label` warn (261); the huggingface_hub `@strict` validators
   `validate_output_attentions` (435), `validate_layer_type` (469/474/476),
   `validate_architecture`, `validate_token_ids` fire on construction. The
   kwargs-wrapping `__init__` (83) rejects positional args (91).

## Method

- Grepped `generation/configuration_utils.py`, `utils/quantization_config.py`,
  and `configuration_utils.py` for every `raise`, `assert`,
  `logger.warning(_once)`, `def validate`, `def post_init`, `def __post_init__`,
  and property setter, then READ each in source context to classify
  predicate_kind + outcome.
- Classified outcome per source, NOT by assumption: `invalid` = raise reached on
  the default path; `warn` = routed to `minor_issues` and logged (the gate calls
  `validate(strict=False)`, so these construct successfully and warn); `normalise`
  = silent mutation.
- Empirically replayed every non-dormant entry against the installed 5.7.0
  source with `/tmp/tfvenv-5.7.0/bin/python3` (no torch): confirmed each
  raise-class `kwargs_positive` raises and `kwargs_negative` is accepted, and
  that warn-class positives construct + emit `logger.warning_once`.

## Idiom note (transformers vs tensorrt/vllm)

transformers validates LATER and SOFTER. The single biggest difference from the
tensorrt idiom: most GenerationConfig combination checks WARN (via the
`minor_issues` dict + `logger.warning_once` at line 824) rather than raise. They
only become errors under `strict=True` (save_pretrained, or explicit
`validate(strict=True)`), which the default gate path does not exercise. 18 of
50 entries are warn-class; only 31 raise; 1 is a silent normalisation.

## Replayability (gate env has NO torch)

- CPU-replayable (36): all GenerationConfig raise + warn entries, and the
  PretrainedConfig `problem_type` / `output_attentions` / `layer_types`
  (allowlist + count) / `id2label` entries. GenerationConfig is replayed via
  `.validate()` after construction (the gate already special-cases it);
  PretrainedConfig is constructed directly (the `@strict` validators fire on
  construction).
- Dormant (14):
  - `strict`-mode promotion + both `save_pretrained` entries: need a
    `strict=True` / save call the gate does not make.
  - `from_model_config` normalisation: needs a `model_config` argument, not
    direct construction.
  - All WatermarkingConfig / SynthIDTextWatermarkingConfig entries:
    `WatermarkingConfig.validate()` only runs when nested in
    `GenerationConfig(watermarking_config=<instance>)`, which cannot be
    expressed as a plain-value kwarg; `SynthIDTextWatermarkingConfig` is
    additionally a torch-gated dummy object (import fails without torch).
  - All BitsAndBytesConfig entries: `__init__` assigns `torch.float32` /
    `torch.uint8` (lines 468/477) and checks `torch.dtype`, so even the
    mutual-exclusion positive (which raises before the torch line) has no
    constructible `kwargs_negative` without torch.
  - PretrainedConfig positional-args rejection: positional args are not
    expressible in a kwargs-only gate.

## Net-new vs prior GT

Per the task framing, this pass is all `provenance: net_new`: no prior-version
PoC ground-truth file was folded into this output. The independent walk was
informed by the structure of the v5.6.2 hand-curated GT (same major surface),
but every citation, predicate, and outcome was re-derived and empirically
re-confirmed against the installed 5.7.0 source. 5.7.0 is structurally identical
to 5.6.2 on this surface; line numbers shifted slightly and the greedy/beam
warning wording is now "`do_sample` is not set to `True`" (was differently
phrased earlier).

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. Per-model `PreTrainedConfig` subclass validators (each `configuration_*.py`
   model config may add its own `__post_init__` / `@strict` checks) - I only
   walked the base `PreTrainedConfig`, which is what a generic harness
   constructs.
2. The full QuantizationConfigMixin family beyond BitsAndBytesConfig (GPTQ, Awq,
   Aqlm, Vptq, Quanto, Eetq, Hqq, AutoRound, Higgs, FPQuant, TorchAo, BitNet,
   etc.) - the entry-point gate only special-cases BitsAndBytesConfig; the
   others are reachable only via `from_pretrained(quantization_config=...)` and
   most need torch. A hierarchy walk over `quantization_config.py` would
   enumerate all 20+ `post_init` validators.
3. `PreTrainedModel.from_pretrained` pre-flight gates (device_map,
   tensor-parallel, dtype) - all require torch + a model dir and are out of
   scope for a CPU/source-only gate.
4. huggingface_hub `@strict` validators that fire only for fields a subclass
   declares (e.g. `validate_architecture` needs head_dim/num_heads/embed_dim) -
   not reachable from a bare `PretrainedConfig()`.
5. Logits-processor / SamplingParams-equivalent runtime checks live in
   `generation/utils.py` (generate-time), not construction-time, so they are
   outside this pass by design.
