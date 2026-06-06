# Pass B methodology - transformers v5.6.2 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the transformers config surface by TYPE, not by call path.
Starting from the public config class trees, every reachable validator was read in
source context and classified. This is the COMPLEMENT to Pass A (which walks only
what the public generate/load construction touches).

- Engine source: `/tmp/tfvenv-5.6.2/lib/python3.12/site-packages/transformers/`
  (`__init__.py` confirms `__version__ = "5.6.2"`).
- Pure source analysis plus CPU-only construction smoke-tests. The venv has NO
  torch installed (`[transformers] PyTorch was not found ...`), which is the
  decisive fact for replayability classification below.

## Type tree enumerated

1. `generation/configuration_utils.py`:
   - `GenerationConfig` (+ its `validate()` body, `save_pretrained`,
     `from_model_config`). Crucially, `GenerationConfig.__init__` calls
     `self.validate()` (line 469), so every `validate()` field rule fires at plain
     construction, not only when `.validate()` is invoked explicitly.
   - `BaseWatermarkingConfig` -> `WatermarkingConfig`,
     `SynthIDTextWatermarkingConfig` (each defines its own `validate()`; the parent
     is abstract). These do NOT validate in `__init__`; their `validate()` is
     reached when a `GenerationConfig` carrying the instance is constructed.
   - `CompileConfig`, `ContinuousBatchingConfig` (no construction-time raises).
2. `utils/quantization_config.py` - the full `QuantizationConfigMixin` tree
   (located via `grep -n "^class "`): `AutoRoundConfig`, `HqqConfig`,
   `BitsAndBytesConfig`, `GPTQConfig`, `AwqConfig` (now a `GPTQConfig` subclass),
   `AqlmConfig`, `VptqLayerConfig`, `VptqConfig`, `QuantoConfig`, `EetqConfig`,
   `CompressedTensorsConfig`, `FbgemmFp8Config`, `HiggsConfig`, `FPQuantConfig`,
   `TorchAoConfig`, `BitNetQuantConfig`, `SpQRConfig`, `FineGrainedFP8Config`,
   `QuarkConfig`, `Mxfp4Config`, `MetalConfig`, `FourOverSixConfig`, `SinqConfig`.
   For each: `__init__` and `post_init` bodies read for every `raise` / `assert` /
   `logger.warning` / silent normalisation; the `AwqFormat` / `AwqBackend` /
   `ExllamaVersion` enums read for membership lists.
3. `configuration_utils.py` - the `PreTrainedConfig` base. v5 turned this into a
   strict-dataclass base; its synthesised `__init__` (positional-arg ban,
   missing-required-field check), its property-setter validators
   (`output_attentions` vs eager attn), and its `@strict`-powered `validate_*`
   hooks (`validate_output_attentions`, `validate_architecture`,
   `validate_token_ids`, `validate_layer_type`) fire for EVERY config subclass.
   None of these are in the entry-point/quant-only PoC GT; they are the principal
   Strategy-B gain for transformers.

## Extraction and the transformers idiom

transformers validates LATER and SOFTER than tensorrt-llm. Many rules are
`logger.warning_once` advisories ("minor_issues") rather than raises. The
`outcome` taxonomy reflects the TRUE behaviour observed in source and confirmed by
CPU smoke-test:

- `invalid`  = a `raise` / `assert` (hard error).
- `warn`     = a `logger.warning(_once)` advisory; construction succeeds. (The PoC
  GT labelled these `dormant`; here they are `warn` because they DO fire on a
  CPU-only host, just without raising.)
- `normalise` = a silent clamp / coercion (e.g. GPTQ `act_group_aware` -> False,
  Awq `llm-awq` backend coercion, `from_*` dtype BC folds).

## kwargs replayability discipline (empirically grounded)

`kwargs_positive` (SHOULD trigger) / `kwargs_negative` (should NOT) were emitted
only where the predicate is reachable at plain construction on this TORCHLESS CPU
host. The smoke-test surfaced three classes of non-replayable rule, all marked
`dormant_reason` with NO kwargs (a fabricated pair would make the gate reject an
otherwise-valid entry):

- `BitsAndBytesConfig`: `__init__` sets the default
  `bnb_4bit_compute_dtype=torch.float32` (line 468), dereferencing torch. The
  mutual-exclusion raise (line 456) fires before that, but the NEGATIVE side and
  every `post_init` type-check hit `NameError: name 'torch' is not defined`. ALL
  10 BnB entries are therefore dormant `requires torch`.
- `HqqConfig`: `__init__` requires the `hqq` package (>=0.2.1); raises `ImportError`
  before the axis checks. Both Hqq entries dormant.
- `TorchAoConfig`: `post_init` requires `torchao`; the requires-torchao raise and a
  new string-rejection raise shadow the `quant_type` type-check. Both dormant
  (outcome is env-coupled).
- `gptq_modules_in_block_requires_optimum_115`: outcome depends on the installed
  `optimum` version - dormant.
- The `from_pretrained` / `accelerate` / `tensor_parallel` family (19 entries):
  call-graph rules that no config class owns and that need torch + a real load.
  Folded for catalogue completeness but all dormant - these are Pass A territory,
  not a config type-tree.

Empirically verified to RAISE/WARN cleanly CPU-only (full pos+neg pairs emitted):
GenerationConfig errors and warn-advisories; the watermarking tree (via a
GenerationConfig wrapper); and the construction-clean quant classes - GPTQ,
AutoRound, Quanto, Eetq, Higgs, FPQuant, SpQR, Metal, FineGrainedFP8, Sinq,
BitNet, Aqlm, Vptq. The net-new `PreTrainedConfig.output_attentions`-vs-eager rule
raises `StrictDataclassClassValidationError` on `LlamaConfig`, CPU-only - a clean
Strategy-B catch.

## Result

- 126 total invariants.
- 118 folded from the PoC ground truth (1:1; every cited line re-resolved against
  the installed 5.6.2 wheel).
- 8 net-new, all `PreTrainedConfig`-base / config-tree rules a generate/load
  entry-point walk structurally misses:
  - `output_attentions`-requires-eager (property setter + `@strict` validator).
  - `validate_layer_type` ALLOWED_LAYER_TYPES membership.
  - `validate_layer_type` num_hidden_layers == len(layer_types).
  - `validate_architecture` head_dim*num_heads == embed_dim.
  - `validate_token_ids` special-token in-vocab (warn).
  - num_labels vs id2label length mismatch (warn, `__post_init__`).
  - positional-args ban on the strict-dataclass `__init__`.
  - `CompressedTensorsConfig` run_compressed auto-disable (normalise) - a quant
    class entirely absent from the PoC GT.

## Citation re-resolution vs the PoC GT

The PoC GT was curated against a near-identical wheel; its lines were off by 0-9
lines from the installed 5.6.2 wheel. Every line was re-located by its message text
and corrected, e.g.:

- BitsAndBytesConfig: PoC 455/524/527/530/533/535/538/541/544/547 ->
  456/525/528/531/534/536/539/542/545/548.
- GPTQConfig: 741/743/745/749/761/768 -> 742/744/746/751/762/772.
- AwqConfig: 845/849/852 -> 846/850/853.
- save_pretrained dir-check: PoC 802 (the `if`) -> 803 (the `raise AssertionError`).
- from_model_config: PoC 1197 (the `if`-block) -> 1203 (the silent set).
- AutoRound 240/242 -> 241/243; Higgs 1352/1354/1356/1358 -> 1355/1357/1359/1361;
  FPQuant 1546-1564 block -> 1425-1447; SpQR 1648-1662 -> 1659-1672;
  FineGrainedFP8 1704/1714/1716 -> 1712/1714/1716; Metal 1824/1826 -> 1825/1827;
  Sinq 1991/1997/1999 -> 1992/1998/2001; BitNet 1568/1570 -> 1595/1597;
  VptqLayer 975 / Vptq 1016 / Quanto 1055/1057 / Eetq 1091 / Aqlm 906/917 - all
  re-verified.

No PoC predicate was found to be wrong. The deltas are citation-line corrections
and the severity re-mapping (PoC `dormant` warn-advisories -> `warn`).

## Blind spots (what an entry-point / call-graph walk should catch that B missed)

A type-tree walk under-covers rules that live in the execution path rather than a
config class:

1. The `PreTrainedModel.from_pretrained` pre-flight gates and the
   `integrations/accelerate.py` + `integrations/tensor_parallel.py` helpers
   (folded here as dormant only for completeness) - Pass A owns them.
2. Runtime guards inside `generate()` length / logits processing.
3. C++/accelerate-side device-map dispatch surfaced only at load time.
4. Cross-config wiring done in `PreTrainedModel.__init__` / weight load that no
   single config class owns.
