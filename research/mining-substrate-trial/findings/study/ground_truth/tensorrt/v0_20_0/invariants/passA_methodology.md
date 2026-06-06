# Pass A - entry-point / call-graph walk methodology (tensorrt-llm 0.20.0)

Engine source: `/tmp/trt-llm-0.20.0/tensorrt_llm/` (confirmed
`__version__ = "0.20.0"` via `version.py`).

Output: `passA_entrypoint.yaml`. This is the entry-point/call-graph half of a
two-pass bake-off; the sibling `passB_classtree.yaml` does a class-hierarchy
walk. Goal: maximise recall of construction-time validation invariants reachable
from public, user-facing entry points.

There is no PoC ground truth for 0.20.0, so every entry is `provenance: net_new`.
The schema, top-level keys, `predicate_kind` taxonomy and field conventions are
matched exactly to the 1.2.1 `passA_entrypoint.yaml` (schema/conventions only,
not content).

## How 0.20.0 differs from the 1.2.1 reference structure

These structural facts drove the walk and the replay/dormant classification:

1. **Single `LlmArgs`.** No Torch/Trt split, no `_TorchLLM`, no
   `_validate_args_for_torch_backend` kwargs-rejection gate. The public
   `tensorrt_llm.LLM(...)` calls `LlmArgs.from_kwargs` directly (llm.py:115).
2. **pydantic.** The import is the v1 style (`from pydantic import validator`)
   but the installed pydantic is v2 (2.12.x) and `LlmArgs` uses v2 API
   (`model_post_init`, `model_config` dict, `model_fields`, `model_dump`). So the
   `@validator` on `LookaheadDecodingConfig` is the v2 compatibility shim, and
   field-level `Literal` validators run at `cls(**kwargs)`. There is NO
   `@field_validator` / `@model_validator` / `def validate_` anywhere in
   `llm_args.py`; the only `@validator` is `LookaheadDecodingConfig`.
3. **Plain dataclasses for SamplingParams / QuantConfig / LoraConfig.** No
   `top_p` / `top_k` / `temperature` range checks exist at 0.20 - those
   validators were added later. `SamplingParams._validate` only enforces
   `best_of >= n`, the `best_of>1` greedy env guard, and `truncate_prompt_tokens
   >= 1`, plus dispatch to `GuidedDecodingParams._validate`.
4. **PluginConfig is a metaclass + property-setter class.** `PluginConfigMeta`
   wraps every `_field` (all `init=False`) as a property whose setter asserts
   (dtype-not-auto, dtype-options allowlist, bool type). The asserts fire on
   attribute assignment, NOT at the constructor - so they are replayable only via
   `PluginConfig().<field> = value`, flagged with a `dormant_reason` for plain
   constructor replay.
5. **QuantConfig allowlists are lazy.** `QuantConfig` is a plain `@dataclass`
   with no `__post_init__`; `quant_algo` / `kv_cache_quant_algo` accept any value
   at construction. The allowlist `assert`s live in `QuantMode.from_quant_algo`
   (mode.py:319-320) and only fire when the `.quant_mode` / `.layer_quant_mode`
   `@cached_property` is accessed. Marked dormant (lazy).
6. **`model_post_init` is unconditionally CUDA-bound.** It calls
   `torch.cuda.get_device_properties(0)` (llm_args.py:969) before any guard, so
   the entire `LlmArgs.from_kwargs` path requires a GPU. The gate runs
   `--gpus all`, which exercises this path; the SM<80 bfloat16 guard and the
   `gpus_per_node` default-fill are reachable there but host-SM-dependent and
   flagged dormant.

## Traversal (what I walked)

Starting roots (the public surface a benchmark harness constructs):

1. `tensorrt_llm.LLM(...)` -> `LLM.__init__` (llm.py:97) ->
   `LlmArgs.from_kwargs(...)` (llm.py:115) ->
   `LlmArgs._maybe_update_config_for_consistency` (the ExecutorConfig subset
   assert + the build_config override warns) -> `cls(**kwargs)` (pydantic field
   validation: `tokenizer_mode`, `load_format` Literals, `model` type) ->
   `model_post_init` (SM<80 bfloat16 guard, `gpus_per_node` fill) -> `_setup`
   (embedding_parallel_mode dispatch, enable_build_cache type, speculative_config
   dispatch, build_config conflict warns, lora consistency warns,
   engine/ckpt loaders).
2. Nested config fields reachable from `LlmArgs`: `CalibConfig` (device Literal),
   `SchedulerConfig` (Capacity/ContextChunking StrEnums), `BatchingType`,
   `KvCacheConfig` (no validators at 0.20 - all plain Fields),
   `PeftCacheConfig`, `CacheTransceiverConfig`, `LoraConfig`
   (`__post_init__` assert in lora_manager.py), `QuantConfig` (lazy allowlist),
   `PluginConfig` (setter asserts), `BuildConfig` (no `__post_init__`
   validation; all checks are build-time), and the four `*DecodingConfig`
   speculative classes (`DecodingBaseConfig.from_dict` dispatch +
   `LookaheadDecodingConfig` `@validator`).
3. `SamplingParams(...)` -> `__post_init__` -> `_validate` (sampling_params.py)
   -> `GuidedDecodingParams._validate`. Plus the deferred
   `_get_bad_words` / `_get_stop_words` setup guards.
4. Deferred config loaders reached from `_setup`: `_load_config_from_engine`,
   `_load_config_from_ckpt`, `get_model_format` (all dir-gated, dormant).

## Method

- Enumerated every `@validator` / `def validate_` / `__post_init__` /
  `raise` / `assert` / `logger.warning` / `Literal[...]` / StrEnum reachable from
  the roots above and READ each in context to classify predicate, severity,
  outcome and replayability. `llm_args.py`: 1 `@validator`, 2 field `Literal`s
  (plus `CalibConfig.device`), 11 `logger.warning` sites, and the full
  `raise`/`assert` set in `model_post_init` / `_setup` / `_load_config_*` /
  `get_model_format`.
- Cross-checked the plugin/quant/lora/sampling modules: `plugin/plugin.py`
  (setter asserts + SM-100 `validate`), `quantization/mode.py`
  (`from_quant_algo` lazy asserts), `lora_manager.py` (`LoraConfig.__post_init__`),
  `sampling_params.py` (`_validate`, guided-decoding dispatch, setup guards).
- Confirmed `BuildConfig` (builder.py:475) and `KvCacheConfig` (llm_args.py:543)
  carry NO construction-time validators at 0.20 - their checks are all build-time
  or absent - so they contribute no entry-point invariants.

## Coverage

- All `LlmArgs` field Literals + the `model_post_init` / `_setup` /
  `_maybe_update_config_for_consistency` raise/warn set (error / warning /
  normalisation).
- All nested-config Literals/StrEnums and the single `@validator`
  (`LookaheadDecodingConfig`) reachable from construction.
- `DecodingBaseConfig.from_dict` decoding-type dispatch (4 types at 0.20).
- Full `SamplingParams` + `GuidedDecodingParams` predicate set as it exists at
  0.20 (best_of/greedy-env/truncate + at-most-one-guide; NO top_p/top_k/temp).
- `PluginConfig` setter asserts + SM-100 `validate`; `QuantConfig` lazy
  allowlists; `LoraConfig` source assert.

## Runtime replayability notes for the downstream gate

- `kwargs_positive` / `kwargs_negative` are constructor-replayable on a host with
  CUDA visible (the gate runs `--gpus all`) for: `CalibConfig`,
  `SchedulerConfig`, `LookaheadDecodingConfig`, `LoraConfig`, `SamplingParams`,
  and the `LlmArgs` field Literals (`tokenizer_mode`, `load_format`, `model`
  type) - the Literal validators fire at `cls(**kwargs)` before
  `model_post_init`.
- `BatchingType` replays via the bare `BatchingType("...")` StrEnum;
  `DecodingBaseConfig` replays via its `from_dict` classmethod (`replay_via`
  annotated), not the bare constructor.
- `PluginConfig` entries replay via attribute assignment
  (`PluginConfig().<field> = value`), since all fields are `init=False`; flagged
  dormant for plain-constructor replay with the exact `replay_via` recipe.
- `QuantConfig` allowlist entries are dormant (lazy): they assert only on
  `.quant_mode` / `.layer_quant_mode` access, never at `QuantConfig(...)`.
- Entries marked `dormant_reason` cannot be replayed by plain CPU construction:
  the SM-gated dtype/plugin guards, engine/ckpt parallel-size mismatches,
  `get_model_format` config.json, the `_setup`-only checks (whole path needs CUDA
  via `model_post_init`), the lazy QuantConfig asserts, the PluginConfig
  attribute-set asserts, and the SamplingParams `_get_bad_words` /
  `_get_stop_words` post-construction setup guards.
- `best_of_gt_1_greedy` is env-sensitive; replay requires
  `TLLM_ALLOW_N_GREEDY_DECODING` unset.

## Blind spots (what the class-hierarchy walk should catch)

1. **Validators on base/sibling classes never reached from the public ctor.**
   I only walk what `LLM(...)` / `SamplingParams(...)` construction touches.
   `MedusaDecodingConfig` / `MTPDecodingConfig` / `EagleDecodingConfig` carry no
   own `@validator` at 0.20 (confirmed), but abstract bases and pybind mirrors
   may carry constraints a hierarchy walk over the MRO would surface.
2. **Inherited pybind / C++ mirror constraints.** Many configs subclass
   `PybindMirror`; constraints enforced on the C++ side (or in a mirrored pybind
   ctor) are invisible to a Python call-graph walk and out of source scope here.
3. **`BuildConfig` / `PluginConfig` deep tree.** I followed the PluginConfig
   setters and the BuildConfig override warns, but did not exhaustively walk
   every build-time `assert` in `builder.py` (those fire at engine build, not
   config construction, so they are out of Pass A scope by definition).
4. **Lazy / deferred asserts.** The QuantConfig `from_quant_algo` allowlist and
   the PluginConfig `validate()` SM guard fire after construction; a hierarchy
   walk that enumerates property/`cached_property` accessors would catalogue
   these more systematically.
