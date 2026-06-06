# Pass A - entry-point / call-graph walk methodology (transformers 5.6.2)

Engine source: `/tmp/tfvenv-5.6.2/lib/python3.12/site-packages/transformers/`
(confirmed `__version__ = "5.6.2"` via `__init__.py`). The venv has NO torch
installed - it is the CPU-only, no-torch host the downstream gate replays in.

Output: `passA_entrypoint.yaml`. This pass is the entry-point/call-graph half of
a two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal: maximise
recall of construction-time validation invariants reachable from the public,
user-facing surface a benchmark harness actually touches. transformers is a
third engine alongside tensorrt-llm and vllm; their APIs are ignored.

## Traversal (what I walked)

Starting roots (public surface a benchmark harness constructs):

1. `transformers.GenerationConfig(...)`. Its `__init__` (configuration_utils.py:469)
   calls `self.validate()` at construction, so the entire `validate()` body
   (configuration_utils.py:590-766) is on the construction call graph. I read
   every `raise` and every `minor_issues[...]` accumulation arm and classified
   each by its true outcome.
2. The quantization config family, all importable as `transformers.<Class>`:
   BitsAndBytesConfig, GPTQConfig, AwqConfig (now a GPTQConfig subclass),
   AqlmConfig, VptqConfig, QuantoConfig, EetqConfig, HqqConfig, AutoRoundConfig,
   HiggsConfig, FPQuantConfig, TorchAoConfig, BitNetQuantConfig, SpQRConfig,
   FineGrainedFP8Config, MetalConfig (new in v5), SinqConfig (new in v5). Each
   runs `__init__`/`post_init` validation at construction.
3. The `WatermarkingConfig` / `SynthIDTextWatermarkingConfig` `.validate()`
   methods reached transitively from `GenerationConfig.validate()` (line 632:
   `self.watermarking_config.validate()`).
4. The model-loading pre-flight gates: `PreTrainedModel.from_pretrained`
   (modeling_utils.py) plus the relocated-in-v5 `check_and_set_device_map`
   (integrations/accelerate.py) and `initialize_tensor_parallelism`
   (integrations/tensor_parallel.py).

## Method

- Enumerated every `raise`, `logger.warning(_once)`, and silent normalisation in
  `GenerationConfig.validate`, `WatermarkingConfig.validate`,
  `SynthIDTextWatermarkingConfig.validate`, `save_pretrained`,
  `from_model_config`, and each quant-config `__init__`/`post_init`, by reading
  the source in context (not grep alone).
- Folded the PoC ground truth
  (`findings/ground_truth/transformers/v5_6_2/invariants_ground_truth.yaml`,
  schema 2.0.0) and re-resolved every citation against this venv source. The
  PoC GT was already cited against 5.6.2; line numbers matched within +/-2 in
  every case I checked and were re-cited to the exact line in this source.
- Empirically verified replayability: I constructed each native_type in the
  5.6.2 venv (no torch) and confirmed positive-trigger kwargs raise the declared
  exception and negative kwargs are accepted. GenerationConfig hard-raise and
  warn paths both behave as declared; every replayable quant-config trigger
  raised ValueError/TypeError as cited.

## transformers idiom (validates later and softer)

`GenerationConfig.validate()` splits into hard raises and a `minor_issues` dict.
The hard raises (early_stopping allowlist, max_new_tokens>0, cache_implementation
allowlist, compile_config type, the num_return_sequences cross-checks, and the
generate-only-kwarg rejection loop) fire unconditionally -> outcome=invalid. The
do_sample/num_beams greedy/beam strips, the use_cache-false cache args, and the
output-flag-without-return-dict checks only append to `minor_issues`, which at
default `validate(strict=False)` emits a single `logger.warning_once` ->
outcome=warn. They are promoted to a raised ValueError only under
`validate(strict=True)`, which `save_pretrained` forces. Outcome is recorded per
the actual default-construction behaviour, not assumed to be a raise.

## Net-new vs PoC GT (3 entries)

1. `GenerationConfig.__init__` calls `validate()` (configuration_utils.py:469) -
   the meta-fact that construction == validation; not an explicit PoC entry.
2. `BitsAndBytesConfig.__init__` references `torch.float32` unconditionally
   (quantization_config.py:468) when `bnb_4bit_compute_dtype` is None (the
   default). In a no-torch env this raises NameError before any field check, so
   EVERY BitsAndBytesConfig invariant is dormant for the gate. The PoC GT listed
   the BnB type checks as replayable; at 5.6.2 on this host they are not.
3. `load_in_4bit`/`load_in_8bit` are now `@property` setters; the
   load_in_4bit/8bit mutual-exclusion fires from the `load_in_8bit` setter
   (quantization_config.py:517) in addition to the old `__init__` site
   (line 456). The PoC GT cited only the old site.

## Folded entries restated for the Pass A schema

The PoC GT used schema 2.0.0 (`source`/`native_field`/`severity`). I translated
to the cross-engine Pass A schema: `native_type` is now a dotted importable path
(`transformers.GenerationConfig`, `transformers.GPTQConfig`, ...), `citation`
replaces `source`, and `outcome` (invalid|warn|normalise) is added alongside
`severity`. The PoC GT collapsed the 8-entry greedy strip, 2-entry beam strip,
4-entry output-flag, and 8-entry generate-kwarg families into one entry each in
some places and split them in others; I keep one or two representative entries
per family (same line/qualname, same message shape) and name the full member set
in `notes`, to keep the catalogue replayable without 30 near-duplicate rows.

## Replayability for the downstream gate

- CPU-no-torch replayable (56 entries): all GenerationConfig validate() raises
  and warn-path checks (verified construct + fire), and every quant-config check
  whose class constructs without torch (GPTQ, Awq, Aqlm, Vptq, Quanto, Eetq,
  AutoRound, Higgs, FPQuant, BitNet, SpQR, FineGrainedFP8, Metal, Sinq) plus the
  TorchAoConfig "torchao not installed" guard (it fires before torch is needed).
- DORMANT (22 entries):
  - BitsAndBytesConfig (torch.float32 default in __init__) - whole class.
  - HqqConfig (requires the hqq package >= 0.2.1).
  - WatermarkingConfig / SynthIDTextWatermarkingConfig validate() - the gate
    bare-constructs these and does not auto-call .validate(); the GenerationConfig
    arm needs a real BaseWatermarkingConfig object, not plain kwargs.
  - GenerationConfig.save_pretrained (needs a save_directory) and
    from_model_config (needs a model_config object / classmethod entry).
  - All from_pretrained / device_map / tensor-parallel gates (need torch,
    accelerate, a model identifier, or a live device/distributed context).

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. Quant-config validators reachable only via subclasses I did not route to
   (e.g. quantizer-side checks in `transformers/quantizers/`), or `post_init`
   arms gated on packages not installed here.
2. `PreTrainedConfig` base `__init__` cross-field checks and per-model config
   subclass validators - the harness loads a concrete model config, but a pure
   call-graph walk from GenerationConfig/quant-config does not traverse the
   ~450 model config classes; an MRO/hierarchy walk would.
3. Validators that fire only under `validate(strict=True)` semantics beyond the
   save_pretrained path, or in `from_dict`/`from_pretrained` JSON round-trips.
4. Logits-processor / sampling-time guards (e.g. WatermarkLogitsProcessor
   construction) that fire at generate runtime, not config construction.
