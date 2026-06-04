# Ground-truth methodology: tensorrt-llm 0.21.0

Established 2026-06-05. Source-walk against the v0.21.0 wheel from pypi.nvidia.com,
unpacked at `/tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt_llm/`.

## Pin

- pip spec: `tensorrt-llm==0.21.0` (extra index `https://pypi.nvidia.com`,
  cp312/linux_x86_64 wheel, 3932.9 MB).
- venv setup: `uv run python research/mining-substrate-trial/scripts/venv_setup.py
  --engine tensorrt --version-slug v0_21_0` (see scripts/venv_setup.py:78).
- The wheel is a binary distribution: the Python source we walk is shipped under
  `tensorrt_llm/` directly; the C++ runtime lives in
  `tensorrt_llm/bindings.cpython-312-x86_64-linux-gnu.so` (6.4 MB) plus a
  hand-written stub `tensorrt_llm/bindings/__init__.pyi` (803 lines).

## Why source-walk, not import-walk

`tensorrt_llm` cannot be imported on a CPU-only host (the .so requires CUDA
12.6+). All catalogue entries are sourced from reading the Python files; this
matches the operational convention already adopted by the version-pinned static
miner under
`engine_versions/tensorrt/v0_21_0/producers/static_invariant_miner.py:11-14`
(quoted: 'The TRT-LLM library at 0.21.0 (CUDA 12.6.x compatibility) cannot be
imported on the host (no CUDA). The miner therefore reads the extracted source
tree and never imports the installed library.').

## Coverage scope

In scope (caller-touchable Python config surface):

| Class                              | File                                | Status                  |
|------------------------------------|-------------------------------------|-------------------------|
| BaseLlmArgs                        | llmapi/llm_args.py:761              | full                    |
| TrtLlmArgs                         | llmapi/llm_args.py:1482             | full                    |
| TorchLlmArgs                       | llmapi/llm_args.py:1624             | full                    |
| _AutoDeployLlmArgs                 | llmapi/llm_args.py:1886             | full                    |
| SamplingParams                     | sampling_params.py:124              | full                    |
| GuidedDecodingParams               | sampling_params.py:13               | full                    |
| AdditionalModelOutput              | sampling_params.py:111              | full                    |
| CalibConfig                        | llmapi/llm_args.py:144              | full                    |
| KvCacheConfig                      | llmapi/llm_args.py:601              | full                    |
| SchedulerConfig                    | llmapi/llm_args.py:465              | full                    |
| DynamicBatchConfig                 | llmapi/llm_args.py:439              | full                    |
| PeftCacheConfig                    | llmapi/llm_args.py:487              | full                    |
| ExtendedRuntimePerfKnobConfig      | llmapi/llm_args.py:677              | full                    |
| CacheTransceiverConfig             | llmapi/llm_args.py:708              | full                    |
| LookaheadDecodingConfig            | llmapi/llm_args.py:554              | full                    |
| MedusaDecodingConfig               | llmapi/llm_args.py:223              | full                    |
| EagleDecodingConfig                | llmapi/llm_args.py:234              | full                    |
| NGramDecodingConfig                | llmapi/llm_args.py:252              | full                    |
| DraftTargetDecodingConfig          | llmapi/llm_args.py:286              | full                    |
| MTPDecodingConfig                  | llmapi/llm_args.py:296              | full                    |
| DecodingBaseConfig                 | llmapi/llm_args.py:196              | full                    |
| QuantConfig                        | models/modeling_utils.py:121        | full                    |
| LayerQuantConfig                   | models/modeling_utils.py:263        | full (invariants only)  |
| QuantAlgo                          | quantization/mode.py:23             | full                    |
| LoraConfig                         | lora_manager.py:138                 | full                    |
| BuildConfig                        | builder.py:478                      | full                    |
| BuildCacheConfig / BuildCache      | llmapi/build_cache.py:31,68         | full                    |
| PluginConfig                       | plugin/plugin.py:140                | full (44 fields)        |
| TorchCompileConfig                 | llmapi/llm_args.py:1603             | full                    |
| _ParallelConfig                    | llmapi/llm_args.py:63               | invariants only         |
| _ModelWrapper                      | llmapi/llm_args.py:721              | invariants only         |
| LogitsProcessor / BatchedLogitsProcessor | sampling_params.py:48,80      | abstract; pass-through  |

In scope (env-var control surface):

- All `TLLM_*` and `TRTLLM_*` env vars that the Python source reads via
  `os.environ.get(...)` / `os.getenv(...)`. Enumerated by `grep -rE
  "T(RT)?LLM_[A-Z_0-9]+" /tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt_llm/`
  (54 unique names). Catalogued in `schema_ground_truth.json` under
  `engine_envs`.

Out of scope (flagged for future iteration):

- C++ pybind module fields not surfaced via a Python BaseModel mirror. The
  pybind classes used by TRT-LLM Python (e.g. `tllme.SamplingConfig`,
  `tllme.OutputConfig`, `tllme.GuidedDecodingParams`, `tllme.ExecutorConfig`,
  `tllme.SchedulerConfig`, `tllme.KvCacheConfig`, `tllme.PeftCacheConfig`,
  `tllme.ExtendedRuntimePerfKnobConfig`, `tllme.CacheTransceiverConfig`,
  `tllme.LookaheadDecodingConfig`) all have Python `@PybindMirror.mirror_pybind_fields`
  decorators whose runtime check (llmapi/llm_args.py:309-397) asserts the
  Python class mirrors the C++ class field-for-field. We treat the Python
  mirror as authoritative at this pin. The 0.21.0 source-walk surfaces no
  Python-side ExecutorConfig knob that BaseLlmArgs does not itself catalogue
  (sanity-checked via `BaseLlmArgs._check_consistency` at
  llmapi/llm_args.py:1032).
  Risk: a C++-only field added after 0.21.0 without a Python mirror would be
  invisible to this catalogue. Mitigation: the `_check_consistency` assertion
  would itself break on the upgrade, surfacing the regression.

- Env vars read only inside the C++ .so (e.g. legacy
  `TLLM_USE_TRT_ENGINE`, `TLLM_OP_AVAILABLE`, `TLLM_FP4_OP_AVAILABLE`,
  `TLLM_CHECK`, `TLLM_THROW`, `TLLM_GEN`, `TLLM_CKPT`, `TLLM_ENGINE` -
  these grep-hit the source as namespace strings but no Python `os.environ`
  read references them). Listed as 'cpp_runtime_only' in the env-var
  enumeration; not catalogued as user-facing controls.

- `ExecutorConfig` constructed manually (i.e. without going through
  BaseLlmArgs). In the LLM-API code path (the documented entry point for
  benchmark callers) BaseLlmArgs is the constructor. Direct
  `tllme.ExecutorConfig(...)` construction is reachable but undocumented
  and out of scope for this catalogue.

- `_AutoDeployLlmArgs` is a TorchLlmArgs subclass for the `_autodeploy`
  backend (model_factory='AutoModelForCausalLM' by default); we catalogue
  its invariants but DO NOT expand its field surface separately - it
  inherits everything from TorchLlmArgs and adds 8 fields (model_factory,
  model_kwargs, mla_backend, skip_loading_weights, free_mem_ratio,
  simple_shard_only, attn_page_size, checkpoint_device).

## TRT-LLM-specific quirks

### LlmArgs split (0.21-specific)

The 0.21 release SPLIT the monolithic `LlmArgs` class into
`BaseLlmArgs` (shared) + `TrtLlmArgs` (AOT engine-build) + `TorchLlmArgs`
(PyTorch backend). `LlmArgs = TrtLlmArgs` alias at llmapi/llm_args.py:1594
preserves the old import path. Pre-0.21 callers picking `LlmArgs` land in
TrtLlmArgs; new callers wanting the PyTorch backend instantiate TorchLlmArgs
directly (or set `backend='pytorch'` on the LLM constructor and let the
factory route).

### PybindMirror pattern

Seven Python BaseModels (`KvCacheConfig`, `SchedulerConfig`, `PeftCacheConfig`,
`ExtendedRuntimePerfKnobConfig`, `CacheTransceiverConfig`, `DynamicBatchConfig`,
`LookaheadDecodingConfig`) are decorated with
`@PybindMirror.mirror_pybind_fields(<C++ class>)`. The decorator runs at
class-creation time (llmapi/llm_args.py:344-352) and raises
ValueError('Field {field} is not mirrored in Python class {cls_name} from
C++ class {pybind_class_name}') if any C++ field lacks a Python counterpart.
This means the Python class is provably a superset of the C++ field surface
AT THIS PIN - it cannot be a strict subset.

### BuildConfig is NOT a Pydantic model

BuildConfig (builder.py:478) is a stdlib `@dataclass`. Pydantic schema
introspection cannot recurse into it; the baseline
`engine_versions/tensorrt/v0_21_0/outputs/schema.discovered.json` flags it as
'BuildConfig is not a Pydantic model; appears as Optional[object] in the
schema' (line 10-16). This catalogue manually enumerates all 27 BuildConfig
fields under `subconfigs.BuildConfig`.

### PluginConfig fields are private with property accessors

PluginConfig (plugin/plugin.py:140) uses a custom metaclass
`PluginConfigMeta` (plugin/plugin.py:130-138) that takes every annotated
attribute of the form `_<name>: T = field(...)` and synthesises a property
named `<name>` with a typed setter. The setter for `bool` fields asserts
isinstance(value, bool); the setter for `str`/`Optional[str]` fields
asserts value in `DEFAULT_PLUGIN_DTYPE_OPTIONS` (or a per-field override in
`PLUGIN_DTYPE_OPTIONS_MAP`). This is the strictest in-Python validation
in the entire TRT-LLM Python surface and is invisible to baseline AST mining
(it lives in metaclass-generated code).

### SamplingParams has no Pydantic model

SamplingParams (sampling_params.py:124) is a `@dataclass(slots=True,
kw_only=True)`. The baseline schema discovery (`schema.discovered.json`)
notes: 'SamplingParams is a dataclass; no per-field descriptions' (line 18-22).
We pull the field descriptions from the class docstring (lines 137-191) and
attach them as `_note` on the catalogue entries where useful.

### GenerationConfig: does NOT exist in TRT-LLM

Confirmed via `grep -rn "^class GenerationConfig" /tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt_llm/`
which returns no hits. The HuggingFace-style generation surface is split
between SamplingParams (request-time decoding knobs) and BuildConfig
(engine-build-time max_draft_len / max_beam_width / max_seq_len). This is
called out explicitly in `schema_ground_truth.json` scope_notes; treat as
a documentation distinction, not a missing surface.

## Citation convention

- `file` is the path RELATIVE to the package root (`tensorrt_llm/`).
- `line` is the 1-based line number AS DISPLAYED BY the Read tool against
  `/tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt_llm/`. Re-running Read on the
  same source dir should yield the same line numbers; if a re-pin produces
  drift, that itself is a useful signal for the substrate bake-off.
- `qualname` follows Python dot-attribute syntax, including nested classes
  and validator method names (e.g.
  `BaseLlmArgs.validate_build_config_with_runtime_params`).

## Cross-references

- nvidia/TensorRT-LLM github tag `v0.21.0` (commit referenced as 0.21.0
  release; release-notes URL is `docs/source/release-notes.md` in the
  upstream repo).
- nvidia docs: docs.nvidia.com/tensorrt-llm (versioned docs for 0.21).
- Local baseline outputs:
  - `engine_versions/tensorrt/v0_21_0/outputs/schema.discovered.json`
  - `engine_versions/tensorrt/v0_21_0/outputs/invariants.proposed.yaml`
  - `engine_versions/tensorrt/v0_21_0/outputs/invariants.validated.yaml`
  - `engine_versions/tensorrt/v0_21_0/outputs/curated.yaml`
- Comparison deliverable: see `delta.md` in the same directory.

## Reviewer protocol

For each catalogue entry the reviewer should be able to:
1. Open `/tmp/trial_tensorrt_v0_21_0_venv/src/tensorrt_llm/<file>` at
   `<line>`.
2. Confirm the field annotation / validator body / docstring matches the
   `type`, `default`, `enum`, `constraint`, and `message_template`
   recorded in the catalogue.
3. For invariants: cross-check the kwargs_positive / kwargs_negative
   examples against the validator body to confirm the predicate fires
   in the expected direction.

A field that fails this check is a catalogue bug; file it as a delta entry.
