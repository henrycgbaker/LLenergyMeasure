# Ground-truth methodology: tensorrt-llm 1.2.1

Established 2026-06-05. Source-walk against the v1.2.1 wheel from pypi.nvidia.com,
unpacked at `/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm/`.

## Pin

- pip spec: `tensorrt-llm==1.2.1` (extra index `https://pypi.nvidia.com`,
  cp312/linux_x86_64 wheel).
- venv setup: `uv run python research/mining-substrate-trial/scripts/venv_setup.py
  --engine tensorrt --version-slug v1_2_1`.
- version confirmed: `tensorrt_llm/version.py` -> `__version__ = "1.2.1"`.
- Binary distribution: Python source shipped under `tensorrt_llm/` directly;
  C++ runtime in `tensorrt_llm/bindings/` (.so + stub `bindings/__init__.pyi`,
  923 lines, 18 classes).

## Why source-walk, not import-walk

`tensorrt_llm` cannot be imported on a CPU-only host (the .so requires CUDA).
All catalogue entries are read from the Python source. Same operational
convention as the v0.21.0 ground truth and the version-pinned static miner.

## MAJOR-VERSION structural change (v0.x -> v1.x)

1.x is a major-version jump. The class tree was reorganised. Key facts the
catalogue depends on:

- **LlmArgs alias flipped.** v0.21: `LlmArgs = TrtLlmArgs` (TRT engine-build
  default). v1.2.1: `LlmArgs = TorchLlmArgs` (llmapi/llm_args.py:3410). The
  PyTorch backend is now the default.
- **Inheritance flattened.** v0.21 split a monolith into BaseLlmArgs +
  TrtLlmArgs + TorchLlmArgs where LlmArgs aliased one of them. v1.2.1 keeps
  BaseLlmArgs (llmapi/llm_args.py:1885) as the explicit common base and both
  `TrtLlmArgs(BaseLlmArgs)` (line 2358) and `TorchLlmArgs(BaseLlmArgs)`
  (line 2768) subclass it directly.
- **Pydantic migration.** BuildConfig (builder.py:453), PluginConfig
  (plugin/plugin.py:88) and LoraConfig (lora_helper.py:82) were stdlib
  `@dataclass` / metaclass at v0.21; they are pydantic `BaseModel` at v1.2.1.
  PluginConfig in particular dropped the `PluginConfigMeta` metaclass entirely
  (the single biggest mining-difficulty win - see version_delta.md).
- **Field nesting.** v0.21 flat TorchLlmArgs `cuda_graph_*` and `moe_*` fields
  are now nested into `CudaGraphConfig` and `MoeConfig`.
- **Speculative tree expanded** from 6 to 9 concrete config classes, and
  dispatch moved from isinstance-replacement to per-subclass `supports_backend()`.

## Coverage scope

In scope (caller-touchable Python config surface):

| Class                          | File                          | Status |
|--------------------------------|-------------------------------|--------|
| BaseLlmArgs                    | llmapi/llm_args.py:1885       | full   |
| TrtLlmArgs                     | llmapi/llm_args.py:2358       | full   |
| TorchLlmArgs                   | llmapi/llm_args.py:2768       | full   |
| SamplingParams                 | sampling_params.py:113        | full   |
| GuidedDecodingParams           | sampling_params.py:14         | full   |
| CudaGraphConfig                | llmapi/llm_args.py:107        | full   |
| GuidedDecodingConfig           | llmapi/llm_args.py:167        | full   |
| MoeConfig / MoeLoadBalancerConfig | llmapi/llm_args.py:443,340 | full   |
| Nvfp4GemmConfig                | llmapi/llm_args.py:480        | full   |
| AttentionDpConfig              | llmapi/llm_args.py:508        | full   |
| RayPlacementConfig             | llmapi/llm_args.py:1194       | full   |
| KvCacheConnectorConfig         | llmapi/llm_args.py:817        | full   |
| Sparse-attn tree (3 + base)    | llmapi/llm_args.py:187-337    | full   |
| CalibConfig                    | llmapi/llm_args.py:593        | full   |
| KvCacheConfig                  | llmapi/llm_args.py:1627       | full   |
| SchedulerConfig / DynamicBatchConfig | llmapi/llm_args.py:1465,1439 | full |
| PeftCacheConfig                | llmapi/llm_args.py:1487       | full   |
| ExtendedRuntimePerfKnobConfig  | llmapi/llm_args.py:1775       | full   |
| CacheTransceiverConfig         | llmapi/llm_args.py:1806       | full   |
| DecodingBaseConfig + 9 subclasses | llmapi/llm_args.py:645-1191 | full   |
| QuantConfig / QuantAlgo        | models/modeling_utils.py:131, quantization/mode.py:23 | full |
| LoraConfig                     | lora_helper.py:82             | full   |
| BuildConfig                    | builder.py:453                | full (27 fields) |
| BuildCacheConfig               | llmapi/build_cache.py:31      | full   |
| PluginConfig                   | plugin/plugin.py:88           | full (43 fields) |
| TorchCompileConfig             | llmapi/llm_args.py:2714       | full   |
| _ParallelConfig / _ModelWrapper | llmapi/llm_args.py:525,1845  | invariants only |

In scope (env-var control surface): all `TLLM_*` / `TRTLLM_*` env vars read
via `os.environ`/`os.getenv` in the Python tree. 55 unique. Catalogued under
`engine_envs` with first-hit source location.

Out of scope (same call as v0.21 - see scope note below):

- The C++ pybind classes in `tensorrt_llm.bindings` (.so + 923-line stub).
- C++-only env vars (read inside the .so, never named in a Python
  `os.environ` read).

## C++ pybind boundary - EXPLICIT scope call for v1.2.1

**Decision: EXCLUDE the raw `tensorrt_llm.bindings` classes; the Python
`@PybindMirror`-decorated BaseModels supersede them.**

Rationale (carried forward and re-verified at this pin):

- Every C++ runtime-config class an LLM-API caller reaches has a Python
  `@PybindMirror.mirror_pybind_fields(<C++ class>)` BaseModel mirror:
  DynamicBatchConfig, SchedulerConfig, PeftCacheConfig, LookaheadDecodingConfig,
  KvCacheConfig, ExtendedRuntimePerfKnobConfig, CacheTransceiverConfig, plus
  three mirror-decorated enums (BatchingType, CapacitySchedulerPolicy,
  ContextChunkingPolicy).
- The decorator (llmapi/llm_args.py:1270-1298) asserts at class-creation time
  that every C++ field has a Python counterpart, so the Python class is
  provably a field-superset of the C++ class at this pin.
- Raw `tensorrt_llm.bindings.executor.ExecutorConfig` is constructed internally
  from BaseLlmArgs, not user-constructed; `BaseLlmArgs._check_consistency`
  (line 2179-2195) asserts the ExecutorConfig attr set (minus max_beam_width)
  is a subset of BaseLlmArgs.model_fields.

Risk / mitigation unchanged: a C++-only field added without a Python mirror
would be invisible here, but the decorator assertion (or the
`_check_consistency` assertion) would break on the upgrade and surface the gap.

## TRT-LLM-specific quirks at v1.2.1

### PluginConfig is now a plain pydantic BaseModel

No more `PluginConfigMeta`. Fields are plain pydantic fields with
`model_config = ConfigDict(validate_assignment=True, extra="ignore")`. The
'auto' resolution moved to `__getattribute__` (plugin/plugin.py:297-308):
reading a field whose stored value is 'auto' returns `self.dtype`. The
enable/disable string coercion is a wildcard before-validator
(plugin/plugin.py:317-329). `dtype` cannot be 'auto' (validator line 310).

### Speculative dispatch via supports_backend()

v0.21's `validate_speculative_config` did isinstance dispatch and REPLACED the
config instance with a `torch.speculative.*` class. v1.2.1 each `*DecodingConfig`
implements `supports_backend(backend)`; `TorchLlmArgs.validate_speculative_config`
(line 3039) and `TrtLlmArgs.validate_speculative_config` (line 2503) call it and
raise on mismatch, then run per-class asserts. No instance replacement.

### Eagle3 folded into EagleDecodingConfig

There is NO separate `Eagle3Config` class. Eagle3 is selected in-place via
`eagle3_one_model` / `eagle3_layers_to_capture` / `eagle3_model_arch`
(llmapi/llm_args.py:862-866) and the `spec_dec_mode` cached property
(line 967-973). The v0.21 "replace with torch.speculative.Eagle3Config"
mechanism is gone.

### GenerationConfig still does not exist

`grep '^class GenerationConfig'` under `tensorrt_llm/` returns nothing. Same as
v0.21: request-time decoding via SamplingParams, build-time via BuildConfig.

### Pydantic 'status' tag

Many fields carry `json_schema_extra={'status': 'beta'|'prototype'|'deprecated'}`.
`TorchLlmArgs.warn_on_unstable_feature_usage` (line 3215) warns on use of any
set beta/prototype field. Recorded per-field as `status` in the schema.

## Citation convention

- `file` is RELATIVE to the package root (`tensorrt_llm/`).
- `line` is the 1-based line number as displayed by the Read tool against
  `/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm/`.
- `qualname` follows Python dot-attribute syntax including nested classes and
  validator method names.

## Cross-references

- nvidia/TensorRT-LLM github tag `v1.2.1`.
- nvidia docs: docs.nvidia.com/tensorrt-llm (1.2 versioned docs).
- Prior ground truth (bump-pair):
  `research/mining-substrate-trial/findings/ground_truth/tensorrt/v0_21_0/`.
- Version delta deliverable: `version_delta.md` (v0.21.0 -> v1.2.1).
- Baseline delta deliverable: `delta.md` (vs LLEM baseline mining outputs).

## Reviewer protocol

For each catalogue entry, open the cited file at the cited line in
`/tmp/trial_tensorrt_v1_2_1_venv/src/tensorrt_llm/<file>` and confirm the
annotation / validator body / docstring matches the recorded type, default,
enum, constraint and message_template. A field that fails is a catalogue bug;
file it as a delta entry.
