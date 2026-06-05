# Ground-truth construction methodology - transformers v5.6.2

## Scope of "ground truth"

The mining substrate produces two stable artefacts per engine-version:

1. **schema** - the engine's user-facing parameter surface (names, types,
   defaults).
2. **invariants** - the predicates the engine enforces over those parameters
   (raises / dormant warnings / silent normalisations).

This document captures what was used as source of truth and what was
intentionally left out. The companion `delta.md` itemises additions on top
of the LLEM-mined baseline; `version_delta.md` itemises the diff against the
v4.57.3 ground truth (the bump-pair predecessor).

## Source corpus

All citations point at the v5.6.2 install resolved by `venv_setup.py` at
`/tmp/trial_transformers_v5_6_2_venv/src/transformers/`. The wheel
(`transformers-5.6.2-py3-none-any.whl`) was downloaded source-only; the
installed `__version__` string was confirmed to be `5.6.2`
(`src/transformers/__init__.py`) before enumeration began.

Primary files walked end-to-end:

- `transformers/generation/configuration_utils.py` (1834 LOC) -
  `GenerationConfig` (all kwarg pops + `_get_default_generation_params` +
  full `validate(strict=False)` body + `save_pretrained` + `from_model_config`
  + `update`), `WatermarkingConfig`, `SynthIDTextWatermarkingConfig`,
  `CompileConfig`, `ContinuousBatchingConfig` (new), `BaseWatermarkingConfig`.
- `transformers/utils/quantization_config.py` (2002 LOC) - every
  `QuantizationConfigMixin` subclass: 22 classes, including the 3 v5-new ones
  (`MetalConfig`, `FourOverSixConfig`, `SinqConfig`). All `__init__` + `post_init`.
- `transformers/modeling_utils.py` (4998 LOC) - `PreTrainedModel.from_pretrained`
  signature (line 3730) + all kwarg pops + pre-flight gates (4000-4128).
- `transformers/integrations/accelerate.py` - `check_and_set_device_map`
  (device_map gates relocated here in v5).
- `transformers/integrations/tensor_parallel.py` - `initialize_tensor_parallelism`
  (tp_plan / tp_size gates relocated here in v5).
- `transformers/utils/hub.py` (949 LOC) - env-var read points; confirmed the
  removal of the transformers-local offline binding (now imports
  `is_offline_mode` from `huggingface_hub` directly, line 40).
- `transformers/utils/logging.py` - `TRANSFORMERS_VERBOSITY`,
  `TRANSFORMERS_NO_ADVISORY_WARNINGS`, `CI` (new read).
- `transformers/cache_utils.py` (1574 LOC) - the rearchitected
  layer-mixin cache taxonomy (4 top-level Cache classes + 11 layer classes;
  `SlidingWindowCache` is a deprecated alias of `StaticCache`).
- `transformers/pipelines/__init__.py` - `pipeline()` factory (real impl at
  line 440; lines 383-431 are typed `@overload` stubs).

Cross-referenced online:

- https://github.com/huggingface/transformers/releases/tag/v5.0.0 - confirmed
  the v5 breaking-change headlines: generation params no longer on model
  config; cache class consolidation (SlidingWindowCache/HybridCache/
  OffloadedCache/SinkCache/MambaCache removed); from_pretrained removed
  load_in_8bit/load_in_4bit/use_auth_token/from_tf/from_flax/low_cpu_mem_usage;
  pipeline framework/image handling refactor.

## Method

1. Read the LLEM baseline producers (no `outputs/` exist for v5.6.2 yet; the
   substrate has not been run on this version) and the v4.57.3 ground-truth
   corpus to establish the starting point and the canonical envelope shapes.
2. Walked each entry-point top-down, capturing every field name with
   default, type annotation, and source line.
3. For invariants: every `if ... : raise` / `minor_issues[...] = ...` /
   silent re-assignment was lifted as a discrete invariant with positive and
   negative kwargs examples.
4. Quantization configs: walked `__init__` then `post_init` for all 22
   classes. Type checks, enum allowlists, mutual exclusions, identity gates,
   and peer-package version gates each became distinct invariants.
5. Explicit version delta: every field/invariant was diffed against the
   v4.57.3 GT (added / removed / renamed / semantics-changed).

## Confidence per section

| Section | Confidence | Notes |
| --- | --- | --- |
| `GenerationConfig` fields | High | Direct line-by-line read of `__init__` + `_get_default_generation_params`. |
| `GenerationConfig.validate` invariants | High | Direct read of `validate()`; every `raise`/`minor_issues` captured. |
| `_get_default_generation_params` lazy-default semantics | High | Read in full; the None-in-init / effective-default split is explicit at lines 551-588. |
| `WatermarkingConfig` / `SynthIDTextWatermarkingConfig` / `CompileConfig` | High | Small, fully read; unchanged from v4.57.3. |
| `ContinuousBatchingConfig` | High for field list; Medium for cross-field validators | The 19 fields are dataclass-explicit. Its deprecated-arg accounting lives in `account_for_cb_deprecated_arguments` and the continuous_batching subsystem, which is research-grade and not exhaustively walked. |
| `from_pretrained` kwargs | High for explicit + popped kwargs; Medium for `**kwargs` catch-all | Signature + pops enumerated. The dead-kwargs sweep (line 4008) is explicit. |
| `from_pretrained` / device_map / tp gates | High | Gates read line-by-line in modeling_utils + integrations/accelerate.py + integrations/tensor_parallel.py. |
| Quantization configs | High for all 22 classes' `__init__`/`post_init` read line-by-line | `CompressedTensorsConfig`, `TorchAoConfig`, `QuarkConfig` retain Medium confidence on the peer-package dynamic surface (compressed_tensors / torchao / quark introspection). |
| Env vars (transformers-owned) | High | Enumerated from `os.getenv`/`os.environ` reads in hub.py + logging.py. |
| Env vars (huggingface_hub pass-through) | Medium | The pass-through set is carried from the v4.57.3 GT. huggingface_hub was NOT installed in this source-only venv, so the upstream constants list could not be re-introspected at v5.6.2's pinned hf_hub. The set is version-pinned to huggingface_hub, not transformers, and transformers v5.6.2 still imports it; the risk is a hf_hub minor bump adding/removing a constant. |
| Pipeline kwargs | High | Real impl signature lifted from line 440. |
| Cache taxonomy | High | All class definitions enumerated; the layer-mixin refactor and the `SlidingWindowCache = StaticCache` alias are explicit. |

## Out of scope (intentional)

- TensorFlow / Flax loading paths: removed entirely in v5 (`from_tf`/`from_flax`
  are now dead kwargs).
- `Trainer` / `TrainingArguments`.
- Tokenizer / processor / image-processor `from_pretrained` chains.
- Per-model `PreTrainedConfig` subclasses (one per architecture).
- The continuous-batching scheduler subsystem internals (only the
  `ContinuousBatchingConfig` surface is captured).
- Cross-package peer config SOTA (`bitsandbytes`, `accelerate`,
  `compressed_tensors`, `torchao`): transformers' side is high-confidence;
  the peer packages' own surfaces are separate ground-truth docs.
