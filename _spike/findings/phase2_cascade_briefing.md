# Phase 2 cascade briefing (read this first)

You are a sonnet subagent doing mechanical field-path migration after a
larger architectural refactor. **Do not re-derive the design.** Use this
briefing as the contract. Phase 1 already shipped at commit `c25541f2`;
the architecture is locked.

## What just changed in phase 1

The hand-written engine config classes (`TransformersConfig`,
`VLLMConfig`, `TensorRTConfig` in `src/llenergymeasure/config/engine_configs.py`)
have been replaced by generated Pydantic classes in
`src/llenergymeasure/engines/<engine>/config.py`. Field SHAPE changed
from FLAT to NESTED:

- **Old**: `cfg.transformers.dtype`, `cfg.transformers.batch_size`,
  `cfg.transformers.sampling.temperature`
- **New**: `cfg.transformers.engine_params.dtype` (engine knowledge),
  `cfg.harness.transformers.batch_size` (llem orchestration; moved),
  `cfg.transformers.sampling_params.temperature` (engine knowledge)

A new `HarnessConfig` lives at `src/llenergymeasure/config/harness.py`
and holds llem-side orchestration (fields with no engine-native API).

The phase 1 commit changed:
- `models.py` (drop engine_v2, add harness, switch types)
- `engines/{transformers,vllm,tensorrt}/plugin.py` (path renames)
- `engines/{transformers,vllm,tensorrt}/config.py` (generated; regenerated with overlay)
- `engine_versions/transformers/v4_57_3/outputs/overlay.yaml` (new)
- `scripts/engine_producers/regen_engine_configs.py` (overlay support)

DO NOT TOUCH any of the above files. They are phase 1 deliverables.

## The mapping table

### Transformers

| Old (flat) | New (nested) | Category |
|---|---|---|
| `cfg.transformers.dtype` | `cfg.transformers.engine_params.dtype` | engine |
| `cfg.transformers.attn_implementation` | `cfg.transformers.engine_params.attn_implementation` | engine |
| `cfg.transformers.device_map` | `cfg.transformers.engine_params.device_map` | engine |
| `cfg.transformers.max_memory` | `cfg.transformers.engine_params.max_memory` | engine |
| `cfg.transformers.tp_plan` | `cfg.transformers.engine_params.tp_plan` | engine |
| `cfg.transformers.tp_size` | `cfg.transformers.engine_params.tp_size` | engine (now int, was str) |
| `cfg.transformers.load_in_4bit` | `cfg.transformers.engine_params.load_in_4bit` | engine |
| `cfg.transformers.load_in_8bit` | `cfg.transformers.engine_params.load_in_8bit` | engine |
| `cfg.transformers.bnb_4bit_compute_dtype` | `cfg.transformers.engine_params.bnb_4bit_compute_dtype` | engine |
| `cfg.transformers.bnb_4bit_quant_type` | `cfg.transformers.engine_params.bnb_4bit_quant_type` | engine |
| `cfg.transformers.bnb_4bit_use_double_quant` | `cfg.transformers.engine_params.bnb_4bit_use_double_quant` | engine |
| `cfg.transformers.use_cache` | `cfg.transformers.sampling_params.use_cache` | engine (sampling) |
| `cfg.transformers.cache_implementation` | `cfg.transformers.sampling_params.cache_implementation` | engine (sampling) |
| `cfg.transformers.num_beams` | `cfg.transformers.sampling_params.num_beams` | engine (sampling) |
| `cfg.transformers.early_stopping` | `cfg.transformers.sampling_params.early_stopping` | engine (sampling) |
| `cfg.transformers.length_penalty` | `cfg.transformers.sampling_params.length_penalty` | engine (sampling) |
| `cfg.transformers.no_repeat_ngram_size` | `cfg.transformers.sampling_params.no_repeat_ngram_size` | engine (sampling) |
| `cfg.transformers.prompt_lookup_num_tokens` | `cfg.transformers.sampling_params.prompt_lookup_num_tokens` | engine (sampling) |
| `cfg.transformers.sampling.temperature` | `cfg.transformers.sampling_params.temperature` | engine (sampling) |
| `cfg.transformers.sampling.top_k` | `cfg.transformers.sampling_params.top_k` | engine (sampling) |
| `cfg.transformers.sampling.top_p` | `cfg.transformers.sampling_params.top_p` | engine (sampling) |
| `cfg.transformers.sampling.<X>` (any sampling field) | `cfg.transformers.sampling_params.<X>` | engine (sampling) |
| `cfg.transformers.batch_size` | `cfg.harness.transformers.batch_size` | LLEM |
| `cfg.transformers.allow_tf32` | `cfg.harness.transformers.allow_tf32` | LLEM |
| `cfg.transformers.autocast_enabled` | `cfg.harness.transformers.autocast_enabled` | LLEM |
| `cfg.transformers.autocast_dtype` | `cfg.harness.transformers.autocast_dtype` | LLEM |
| `cfg.transformers.torch_compile` | DROPPED (replaced by `cfg.transformers.sampling_params.compile_config` dict) | engine native |
| `cfg.transformers.torch_compile_mode` | `cfg.transformers.sampling_params.compile_config.mode` (inside the dict) | engine native |
| `cfg.transformers.torch_compile_backend` | `cfg.transformers.sampling_params.compile_config.backend` | engine native |
| `cfg.transformers.low_cpu_mem_usage` | DROPPED entirely - HF 4.57.3 discards the kwarg as a no-op | n/a |

### vLLM

| Old | New | Notes |
|---|---|---|
| `cfg.vllm.dtype` | `cfg.vllm.engine_params.dtype` | engine |
| `cfg.vllm.engine.<field>` (was nested VLLMEngineConfig) | `cfg.vllm.engine_params.<field>` (flat) | engine; the .engine sub-class is GONE |
| `cfg.vllm.sampling.<field>` | `cfg.vllm.sampling_params.<field>` | engine sampling |
| `cfg.vllm.beam_search` | DROPPED - Move 1 walker gap; use `cfg.vllm.sampling_params` extras for now | parked |
| `cfg.vllm.engine.attention.<X>` | DROPPED - Move 1 walker gap; use `cfg.vllm.engine_params` extras for now | parked |
| `cfg.vllm.engine.speculative_config` | DROPPED - Move 1 walker gap; use `cfg.vllm.engine_params` extras | parked |

### TensorRT

| Old | New | Notes |
|---|---|---|
| `cfg.tensorrt.tensor_parallel_size` | `cfg.tensorrt.engine_params.tensor_parallel_size` | engine |
| `cfg.tensorrt.pipeline_parallel_size` | `cfg.tensorrt.engine_params.pipeline_parallel_size` | engine |
| `cfg.tensorrt.max_batch_size` | `cfg.tensorrt.engine_params.max_batch_size` | engine |
| `cfg.tensorrt.max_input_len` | `cfg.tensorrt.engine_params.max_input_len` | engine |
| `cfg.tensorrt.max_seq_len` | `cfg.tensorrt.engine_params.max_seq_len` | engine |
| `cfg.tensorrt.max_num_tokens` | `cfg.tensorrt.engine_params.max_num_tokens` | engine |
| `cfg.tensorrt.dtype` | `cfg.tensorrt.engine_params.dtype` | engine |
| `cfg.tensorrt.fast_build` | `cfg.tensorrt.engine_params.fast_build` | engine |
| `cfg.tensorrt.backend` | `cfg.tensorrt.engine_params.backend` | engine |
| `cfg.tensorrt.quant_config.<X>` | DROPPED - Move 1 walker gap; use `cfg.tensorrt.engine_params.quant_config` as dict (via extras) | parked |
| `cfg.tensorrt.kv_cache_config.<X>` | DROPPED - same | parked |
| `cfg.tensorrt.scheduler_config.<X>` | DROPPED - same | parked |
| `cfg.tensorrt.sampling.<X>` | `cfg.tensorrt.sampling_params.<X>` | engine sampling |

### Imports

| Old | New |
|---|---|
| `from llenergymeasure.config.engine_configs import TransformersConfig` | `from llenergymeasure.engines.transformers.config import Config as TransformersConfig` (or just use the new path) |
| `from llenergymeasure.config.engine_configs import VLLMConfig` | `from llenergymeasure.engines.vllm.config import Config as VLLMConfig` |
| `from llenergymeasure.config.engine_configs import TensorRTConfig` | `from llenergymeasure.engines.tensorrt.config import Config as TensorRTConfig` |
| `from llenergymeasure.config.engine_configs import VLLMEngineConfig` | DROPPED - flatten into engine_params construction |
| `from llenergymeasure.config.engine_configs import TransformersSamplingConfig` | DROPPED - sampling fields land in sampling_params |
| `from llenergymeasure.config.engine_configs import TensorRTQuantConfig` | DROPPED - quant_config is a dict now (parked) |
| ditto other sub-config imports | DROPPED |

### YAML/dict construction

Old:
```python
cfg = ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    engine="transformers",
    transformers=TransformersConfig(dtype="bfloat16", batch_size=4),
)
```

New:
```python
cfg = ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    engine="transformers",
    transformers={"engine_params": {"dtype": "bfloat16"}},
    harness={"transformers": {"batch_size": 4}},
)
```

Or using the dict form (recommended for tests; avoids needing to import the new Config class):
```python
cfg = ExperimentConfig(
    task=TaskConfig(model="gpt2"),
    engine="transformers",
    transformers={
        "engine_params": {"dtype": "bfloat16"},
        "sampling_params": {"temperature": 0.7},
    },
    harness={"transformers": {"batch_size": 4}},
)
```

## Common gotchas

1. **HarnessConfig is per-engine.** `cfg.harness` is `HarnessConfig` which
   has sub-fields `transformers`, `vllm`, `tensorrt`. Always go through
   the engine name: `cfg.harness.transformers.batch_size` (not
   `cfg.harness.batch_size`).

2. **Generated classes have `extra='allow'` on engine_params and
   sampling_params.** Unknown fields land in `.model_extra`. Tests that
   used to assert "field X rejected" may now need to assert presence
   in `model_extra` instead.

3. **`temperature` default is now 1.0 (mined from HF), not None.** Old
   `TransformersSamplingConfig.temperature` was `None` by default; new
   `SamplingParams.temperature` is `1.0`. Tests that checked
   `cfg.transformers.sampling.temperature is None` need updating.

4. **Some narrow Literals are gone.** E.g. `dtype: Literal["float32",
   "float16", "bfloat16"]` is now `dtype: str | None`. Tests asserting
   `dtype="half"` was rejected need to flip to assert it's accepted.

5. **Overlay narrowings ADD constraints.** `temperature: minimum: 0.0`
   means tests passing `temperature=-1.0` now correctly fail with
   ValidationError. Tests asserting acceptance of negative temperature
   need updating (or removing - they were testing wrong behaviour).

6. **`torch_compile` boolean is gone.** Old `cfg.transformers.torch_compile = True`
   doesn't exist. Use `cfg.transformers.sampling_params.compile_config = {"mode":
   "reduce-overhead", ...}` instead. The plugin sets HF's compile_config
   on `model.generation_config`.

7. **`cfg.vllm.engine.X` access pattern is gone.** The nested
   `VLLMEngineConfig` doesn't exist anymore. Direct attribute access
   would crash; use `cfg.vllm.engine_params.X`.

## What to do

1. Read this briefing fully.
2. For each test file in your assigned bucket:
   - Run the test alone: `uv run python -m pytest <test_file> -q --no-header --tb=short`
   - Look at failures. Identify the migration pattern (import? construction shape? field access?).
   - Apply the mapping table.
   - Re-run to verify the file is green (or document remaining issues).
3. For production code files in your bucket:
   - Same approach: identify field accesses, apply mapping, verify by running adjacent tests.
4. **Do NOT commit.** Leave changes staged-but-uncommitted (or just on disk; the orchestrator will review).
5. Run `uv run python -m pytest <your_assigned_dirs> -q --no-header --tb=no | tail -5` at the end to record final state.
6. Report back: which files you touched, which tests pass/fail in your bucket, any patterns that don't fit the mapping table.

## What NOT to touch

- `src/llenergymeasure/config/models.py` (already updated)
- `src/llenergymeasure/config/harness.py` (already created)
- `src/llenergymeasure/engines/<e>/plugin.py` (already updated)
- `src/llenergymeasure/engines/<e>/config.py` (generated; would be overwritten)
- `engine_versions/<e>/v<safe>/outputs/overlay.yaml` (data, not code)
- `scripts/engine_producers/regen_engine_configs.py` (already updated)
- `tests/integration/test_codegen_tracer_bullet.py` (already updated; 30/30 green)
- `src/llenergymeasure/config/engine_configs.py` - the old hand-written class
  file. **Leave it in place for now**; phase 3 deletes it once all
  consumers have migrated. You will be REMOVING imports of its classes
  from your assigned files, but the file itself stays on disk.

## Where to ask questions

If you encounter a pattern that doesn't fit the mapping table, STOP and
add it to `_spike/findings/phase2_cascade_briefing.md` (this file) under
a new "## Patterns not in original briefing" section. Don't guess. The
orchestrator will pick up your additions and decide.

## Patterns not in original briefing

### 1. Invariant corpus paths use old schema

The engine-invariants YAML corpus (`engines/*/invariants.*.yaml`) stores
`match_fields` keys using the old flat schema paths:
`transformers.sampling.temperature`, `transformers.sampling.do_sample`, etc.

These paths fail to resolve against the new nested schema.

**Fix applied:** Added `_translate_invariant()` and `_translate_corpus_path()`
helpers in `study/library_resolution.py` that remap `<engine>.sampling.<field>`
→ `<engine>.sampling_params.<field>` when translating corpus invariants before
use in `_apply_invariants_fixpoint`. The invariant YAML files are NOT modified
(they're data files; a future phase should update them properly).

### 2. `build_resolved_view` in `study/hashing.py` needs updating

The old code did `dump.pop("sampling", None)` to separate sampling params from
engine params. With the new schema, sampling lives under `sampling_params` and
engine config lives under `engine_params`. Both need to be extracted.

**Fix applied:** Updated `build_resolved_view` to pop `sampling_params` (with
fallback to `sampling`), and also pop `engine_params` and flatten it into
`observed_engine_params`.

### 3. `conftest.py` dtype routing needs updating

The `make_config` helper in `tests/conftest.py` routed a top-level `dtype`
kwarg directly into the engine config dict (e.g. `{"dtype": "float16"}`).
With the new schema, dtype lives under `engine_params.dtype`.

**Fix applied:** Updated `make_config` to route `dtype` into
`{"engine_params": {"dtype": dtype}}`.

### 4. Downstream dtype consumers need `engine_params.dtype` access

`cli/_vram.py` and `cli/_display.py` both did `getattr(engine_section, "dtype", None)`.
With the new schema, dtype lives under `engine_params`.

**Fix applied:** Both files now check `engine_section.engine_params.dtype` first,
with fallback to `engine_section.dtype`.

### 5. `study/library_resolution.py` `_canonical_excerpt` used `.sampling`

The `_canonical_excerpt` function accessed `section.sampling` (old sub-config).
New schema has `section.sampling_params`.

**Fix applied:** Updated to use `sampling_params`, with `model_dump` path for
Pydantic models and dict fallback.

### 6. `harness/preflight.py` `engine_path` access

The `_check_tensorrt_checkpoint_compat` function accessed `getattr(trt, "engine_path", None)`.
With new schema, `engine_path` lands in `engine_params.engine_path` (model_extra).

**Fix applied:** Updated to check `trt.engine_params.engine_path` first.

### 7. Pre-existing failures in `test_sweep_groups.py`

`tests/unit/study/test_sweep_groups.py::TestCombinatorialWarnings::test_large_study_info_log`
was already failing before phase 2. The test uses old sweep paths like
`transformers.dtype`, `transformers.attn_implementation`, `transformers.torch_compile`
that should be `transformers.engine_params.dtype` etc. The grid/sweep expansion
routes these to `model_extra` now rather than real schema fields, so validation
no longer rejects the `flash_attention_2 + float32` combination.
Not fixed in agent C scope - needs sweep path translation in the grid layer.

`tests/unit/study/test_study_grid.py::TestExpandGridSweep::test_multi_engine_scoped_sweep`
also pre-existing: accesses `c.vllm.engine.max_num_seqs` (old `VLLMEngineConfig` sub-field).

## Patterns not in original briefing

### P1: Old engine class objects passed directly as ExperimentConfig field values

Tests that did `ExperimentConfig(transformers=TransformersConfig(...))` need
to use dict form: `ExperimentConfig(transformers={"engine_params": {...}})`.
The new `transformers` field type is `engines.transformers.config.Config`,
not `engine_configs.TransformersConfig`. Pydantic rejects the old class
instance with a type-mismatch error.

### P2: Invariant match_field paths for nested shape

Test stub invariants used paths like `transformers.attn_implementation`.
These now need to be `transformers.engine_params.attn_implementation` and
`transformers.sampling.temperature` -> `transformers.sampling_params.temperature`.
The `resolve_field_path` function in the invariants loader resolves via
`model_fields`; fields that are not typed model fields are skipped, returning
`None` for the predicate and causing false-negative no-match.

### P3: dict sub-object access for model_extra values

When a flat YAML key like `kv_cache_config: {enable_block_reuse: true}` is
passed at the top level of the tensorrt dict (not nested under `engine_params`),
it lands in `model_extra` as a raw dict. Accessing `.enable_block_reuse`
then fails because dicts don't have attribute access. Fix: use
`c.tensorrt.kv_cache_config["enable_block_reuse"]` for model_extra dict values.

### P4: No Literal rejection for TensorRT quant_config.quant_algo in sweep

The old `TensorRTQuantConfig.quant_algo` was `Literal[...]`. The new generated
`engine_params.quant_config` is `Any | None`. Tests asserting that invalid
quant_algo values are rejected at config parse time (e.g. via sweep expansion)
need to flip: all string values are now accepted; validation is deferred to
the engine at runtime.

### P5: conftest.make_config already updated to route dtype to engine_params

`tests/conftest.py`'s `make_config(dtype=...)` already routes dtype into
`{"engine_params": {"dtype": dtype}}`. Assertions on `config.transformers.dtype`
need to become `config.transformers.engine_params.dtype`. The `Config` class
has `extra='allow'`, so Pydantic's `model_extra` attribute access works for
genuinely flat top-level keys, but NOT for nested sub-keys placed into
`engine_params` - those are accessed via the typed sub-model.
