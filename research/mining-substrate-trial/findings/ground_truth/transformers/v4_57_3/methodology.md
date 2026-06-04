# Ground-truth construction methodology - transformers v4.57.3

## Scope of "ground truth"

The mining substrate produces two stable artefacts per engine-version:

1. **schema** - the engine's user-facing parameter surface (names, types,
   defaults).
2. **invariants** - the predicates the engine enforces over those parameters
   (raises / dormant warnings / silent normalisations).

This document captures what was used as source of truth and what was
intentionally left out. The companion `delta.md` itemises additions on top
of the existing LLEM-mined baseline (`engine_versions/transformers/v4_57_3/outputs/`).

## Source corpus

All citations point at the v4.57.3 install resolved by `venv_setup.py` at
`/tmp/trial_transformers_v4_57_3_venv/src/transformers/`. The installed
`__version__` string was confirmed to be `4.57.3` before any enumeration began.

Primary files walked end-to-end:

- `transformers/generation/configuration_utils.py` (1478 LOC) - `GenerationConfig`,
  `WatermarkingConfig`, `SynthIDTextWatermarkingConfig`, `CompileConfig`,
  `BaseWatermarkingConfig`. All `__init__` kwarg pops + the full
  `validate(strict=False)` body.
- `transformers/utils/quantization_config.py` (2110 LOC) - every
  `QuantizationConfigMixin` subclass: 19 classes total, including
  `BitsAndBytesConfig`, `GPTQConfig`, `AwqConfig`, `AqlmConfig`, `VptqConfig`,
  `QuantoConfig`, `EetqConfig`, `HqqConfig`, `AutoRoundConfig`,
  `CompressedTensorsConfig`, `FbgemmFp8Config`, `HiggsConfig`, `FPQuantConfig`,
  `TorchAoConfig`, `BitNetQuantConfig`, `SpQRConfig`, `FineGrainedFP8Config`,
  `QuarkConfig`, `Mxfp4Config`.
- `transformers/modeling_utils.py` (6165 LOC) - `PreTrainedModel.from_pretrained`
  signature + all kwarg pops + all pre-flight gates from line 4384-4970.
- `transformers/utils/hub.py` (1162 LOC) - env-var read points and the
  `_is_offline_mode` import-time binding gotcha.
- `transformers/utils/logging.py` - `TRANSFORMERS_VERBOSITY`,
  `TRANSFORMERS_NO_ADVISORY_WARNINGS`.
- `transformers/cache_utils.py` - the full cache-class taxonomy
  (`DynamicCache`, `StaticCache`, `OffloadedCache`, `HybridCache`,
  `SlidingWindowCache`, etc.).
- `transformers/pipelines/__init__.py` - `pipeline()` factory and its
  17 keyword params.

External upstream:

- `huggingface_hub.constants` - inspected via runtime introspection
  (`python3 -c "import huggingface_hub.constants as c; ..."`) on the
  same venv. All HF_HUB_* env vars enumerated from there.

Cross-referenced online:

- https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/generation/configuration_utils.py
  (verification of `GenerationConfig.__init__` field list + `validate` predicates)
- https://huggingface.co/docs/transformers/v4.57.3/en/main_classes/text_generation
  (cross-check of public field names; tip blocks referenced for default
  semantics).
- https://huggingface.co/docs/huggingface_hub/main/en/package_reference/environment_variables
  (env-var canonical list).

## Method

1. Read the existing baseline (`schema.discovered.json`,
   `invariants.validated.yaml`, `invariants.proposed.yaml`,
   `curated.yaml`) to understand the LLEM scoring envelope and what was
   already mined.
2. Walked each entry-point top-down, capturing every field name with
   default, type annotation (or runtime resolution rule when annotation is
   absent), and source line.
3. For invariants: every `if ... : raise` / `minor_issues[...] = ...` /
   silent re-assignment was lifted as a discrete invariant with a positive
   and negative kwargs example.
4. Quantization configs: walked `__init__` then `post_init` for each of the
   19 classes. Type checks, enum allowlists, mutual exclusions, and
   peer-package version gates each became distinct invariants.
5. Env vars: enumerated from `os.getenv` / `os.environ` reads in
   `transformers.utils.hub` and `transformers.utils.logging`, plus the
   pass-through set from `huggingface_hub.constants` that transformers
   honours implicitly.
6. Cross-validated against the public HF docs and v4.57.3 GitHub tag
   for spot-checks on `GenerationConfig` defaults.

## Confidence per section

| Section | Confidence | Notes |
| --- | --- | --- |
| `GenerationConfig` fields | High | Direct line-by-line read of `__init__`; matches docstring. |
| `GenerationConfig.validate` invariants | High | Direct read of `validate()`; every `raise`/`minor_issues` captured. |
| `WatermarkingConfig` / `SynthIDTextWatermarkingConfig` | High | Small, fully read. |
| `from_pretrained` kwargs | High for documented kwargs; medium for `**kwargs` catch-all | Documented and via `kwargs.pop` enumeration. Some kwargs (e.g. `state_dict`) are docstring-only; surfaced via the docstring walker would miss the `_from_pipeline` / `_from_auto` internal pops. |
| Quantization configs | High for `BitsAndBytesConfig`, `GPTQConfig`, `AwqConfig`, `QuantoConfig`, `EetqConfig`, `HqqConfig`, `AqlmConfig`, `VptqConfig`, `HiggsConfig`, `FPQuantConfig`, `SpQRConfig`, `FineGrainedFP8Config`, `BitNetQuantConfig`, `AutoRoundConfig`, `Mxfp4Config`, `FbgemmFp8Config` | All read line-by-line. |
| Quantization configs | Medium for `TorchAoConfig`, `CompressedTensorsConfig`, `QuarkConfig` | Partial peer-package dependency surface; some predicates depend on dynamic introspection of the installed torchao / compressed-tensors / quark version. |
| Env vars | High for transformers-owned; medium for huggingface_hub pass-through | Transformers reads via `huggingface_hub.constants`, so the upstream set is normative; the `HF_HUB_OFFLINE` import-time-binding gotcha is independently confirmed. |
| Pipeline kwargs | High | Signature lifted from `__init__.py` line 637. |
| `device_map` semantics | High | The four named strings (`auto`, `balanced`, `balanced_low_0`, `sequential`) come directly from the type-coercion gate at line 4786 of `modeling_utils.py`. Numerical-int negative-value gate at 4796. |
| `attn_implementation` enum | Medium | The static set (`eager`, `sdpa`, `flash_attention_2`, `flash_attention_3`) is documented; the dynamic registry `ALL_ATTENTION_FUNCTIONS` admits any registered key plus HF kernel-hub repo specs (`<namespace>/<repo>[@rev][:kernel]`). Full enumeration would require runtime introspection of the kernel registry. |
| Cache classes taxonomy | High | All class definitions enumerated. The `cache_implementation` string enum is decoupled from the class names (lines 46-57 of `configuration_utils.py`). |
| Cross-package peer configs (`bitsandbytes` runtime, `accelerate` `device_map` semantics) | Medium | Transformers' side is high-confidence; the actual `bitsandbytes` / `accelerate` SOTA surface is out of scope here and would be its own ground-truth doc. |

## Out of scope (intentional)

- TensorFlow / Flax loading paths (`from_tf` / `from_flax`): the kwargs
  are surfaced but their downstream invariants are not exhaustively walked
  (LLEM target is PyTorch causal-LM inference).
- `Trainer` / training loop config (`TrainingArguments`, `Trainer.train`).
- Tokenizer / processor / image-processor `from_pretrained` chains.
- Per-model `PretrainedConfig` subclasses (one per architecture; ~150+
  classes). The LLEM substrate aims at the generation surface, not per-model
  hyperparameters.
- Continuous-batching subsystem (`transformers/generation/continuous_batching/`):
  surfaced briefly but its scheduler config is research-grade and changes
  rapidly.
