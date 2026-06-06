# Pass B methodology - transformers v5.7.0 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the transformers config surface by TYPE, not by call path.
Starting from the public config classes, every reachable construction-time
validation rule was extracted and classified.

- Engine source: `/tmp/tfvenv-5.7.0/lib/python3.12/site-packages/transformers/`
  (`__init__.py` confirms `__version__ = "5.7.0"`).
- Pure source analysis plus in-venv replay. The venv has NO torch installed
  (verified), so torch-touching construction paths are dormant here.
- No prior ground truth for v5_7_0 (5.6.2 has one, this version does not), so all
  59 invariants are `net_new`.

## Type tree enumerated

1. `GenerationConfig` (`generation/configuration_utils.py`): the full
   `validate()` body (lines 612-825) plus the nested validatable members it
   owns - `WatermarkingConfig` / `SynthIDTextWatermarkingConfig`
   (`BaseWatermarkingConfig` subclasses, their own `validate()`), and the
   `CompileConfig` type check.
2. `PretrainedConfig` (`configuration_utils.py`): the `@strict`-dataclass base of
   the entire model-config tree. `__init__` construction checks plus the
   `@strict`-powered `validate_*` methods.
3. The quantization config family (`utils/quantization_config.py`):
   `QuantizationConfigMixin` base + every `*Config` subclass located by
   `grep -nE "^class .*Config"` and read in full - AutoRound, Hqq,
   BitsAndBytes, GPTQ, Awq (GPTQ subclass), Aqlm, VptqLayer, Vptq, Quanto, Eetq,
   CompressedTensors, FbgemmFp8, Higgs, FPQuant, TorchAo, BitNet, SpQR,
   FineGrainedFP8, Quark, Mxfp4, Metal, FourOverSix, Sinq.

## Extraction per class

For each class the following were harvested and read in source context:

1. Every `raise` (`ValueError` / `TypeError` / `AssertionError`) in
   `__init__` / `post_init` / `validate`.
2. Every `logger.warning(_once)` / accumulated-minor-issue warning.
3. Every silent normalisation (a field re-assigned to a coerced value with no
   raise) - classified `outcome: normalise`.
4. `Literal[...]` / `StrEnum` (`AwqFormat`, `AwqBackend`) / `Enum`-typed fields
   whose invalid value is rejected by type (`PretrainedConfig.problem_type`).

`outcome` derived from observed behaviour: a `raise` -> `invalid`; a warning that
still constructs -> `warn`; a silent field mutation -> `normalise`.

## transformers idiom (validates later and softer)

Unlike a pydantic engine, transformers collects "minor issues" in
`GenerationConfig.validate()` into a dict and, with the default `strict=False`,
emits ONE accumulated `logger.warning_once` (line 824) rather than raising. Only
a hard subset raises unconditionally (early_stopping, max_new_tokens,
cache_implementation, num_return_sequences combinations, compile_config type,
generate-only-arg rejection). The minor-issue family (sampling-only flags while
greedy, beam-only flags while single-beam, extra-output flags without
return_dict, cache args while use_cache=False, negative pad_token_id) is recorded
as `outcome: warn`. Quant configs, by contrast, raise eagerly from `post_init`.
Outcomes were recorded as observed in the venv, never assumed.

## kwargs replayability discipline

Every emitted `kwargs_positive`/`kwargs_negative` pair was REPLAYED in this venv:
the positive triggered (raised for `invalid`, emitted the warning for `warn`); the
negative constructed clean. 48 of 59 invariants carry replayable pairs. For
GenerationConfig the gate calls `.validate()`; for the standalone configs the
predicate fires inside the constructor (`post_init`/`__init__`) or, for the
Watermarking configs, inside `.validate()` which the replay invokes.

11 invariants are DORMANT (carry a `dormant_reason`, no kwargs) because the
predicate cannot fire on this CPU-no-torch host without first hitting a blocker:

- `BitsAndBytesConfig` (two entries): the load_in_4bit/8bit mutual-exclusion
  raises BEFORE torch, but any non-raising negative touches the
  `torch.float32`/`torch.uint8` defaults -> `NameError` (no torch). No clean
  negative is constructible, so no pair is fabricated.
- `SynthIDTextWatermarkingConfig`: instantiation requires torch.
- `CompressedTensorsConfig`: `__init__` requires the `compressed_tensors` package.
- `HqqConfig`: `__init__` requires the `hqq` package.
- `TorchAoConfig`: `post_init` requires the `torchao` package.
- `GPTQConfig.modules_in_block_to_quantize`: reads `optimum` package metadata
  (optimum not installed).
- `GenerationConfig.compile_config` type check: both arms need object fixtures
  (a real `CompileConfig` is torch-adjacent).
- `PretrainedConfig.validate_architecture` / `validate_layer_type`:
  `@strict`-powered validators NOT auto-invoked at plain base-class construction.
- `VptqLayerConfig.is_indice_packed`: the class is not importable from the
  `transformers` top-level (constructed only internally by `VptqConfig`), so it
  has no dotted `transformers.<Class>` path for the gate.

## Class-tree cases caught (Strategy-B gain over an entry-point walk)

- The entire quantization config class tree: 20+ `*Config` classes, most never
  touched by a `generate`/`load` entry-point. Each carries a `post_init`
  allowlist/range raise (GPTQ bits/group_size/damp/dataset, Quanto/Eetq weights,
  Higgs bits/p/group_size/hadamard, FPQuant dtype allowlists, SpQR bits/beta,
  FineGrainedFP8 activation/weight_block, Metal bits/group_size, Sinq method,
  AutoRound bits, BitNet linear/mode, Aqlm field types, Vptq enable_proxy).
- The OVERRIDE case: `AwqConfig` subclasses `GPTQConfig` but replaces `post_init`
  with an AwqFormat/AwqBackend allowlist - a different predicate fired only for
  the child, exactly the kind a base-class call-path walk misses.
- The declarative `Literal`-typed `@strict`-dataclass field
  `PretrainedConfig.problem_type` (rejected by type at field-set, no explicit
  `raise` in any call path).
- The GPTQ `desc_act` -> `act_group_aware=False` silent normalisation
  (`outcome: normalise`) and the CompressedTensors `run_compressed` normalisation.

## Blind spots (what an entry-point / call-graph walk should catch that B missed)

1. Runtime validation in `generate()` / generation-loop code (length checks,
   model-dependent cache validation) that `validate()` explicitly defers
   (docstring lines 617-618).
2. Cross-config wiring done in `PreTrainedModel.from_pretrained` /
   `AutoQuantizationConfig` dispatch - the quant-method-to-class routing and
   per-quantizer `validate_environment` checks live in `quantizers/`, not in the
   config classes.
3. The `@strict` validation harness invocation site: `validate_architecture` /
   `validate_token_ids` / `validate_layer_type` fire only when a model config
   subclass opts into strict validation, reachable only by following the load
   call graph.
4. torch / package-gated predicates (BitsAndBytes dtype, SynthID, compressed_
   tensors, hqq, torchao, optimum) that need a fuller fixture or a torch-bearing
   gate host to replay.
