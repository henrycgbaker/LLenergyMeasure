# Ground-truth methodology - vllm 0.19.1

## Scope

The deliverable answers two questions for the LLEM mining-substrate trial:

1. **What is the complete, type-aware schema** of vllm 0.19.1's caller-touchable
   configuration surface (engine, sampling, beam-search, pooling, structured-outputs,
   every subconfig class in the new `vllm/config/*` subpackage, AND the `vllm.envs`
   env-var control surface)?
2. **What are the citation-bearing invariants** vllm 0.19.1 enforces at config / params
   construction time (raises, warnings, silent normalisations, deprecations)?

The headline structural fact is that **vllm 0.19.1 replaced the flat `vllm/config.py`
(~3490 LOC) with a per-concern SUBPACKAGE `vllm/config/*`**. Every subconfig now lives in
its own module (`model.py`, `cache.py`, `parallel.py`, `scheduler.py`, `speculative.py`,
`compilation.py`, `kv_transfer.py`, `structured_outputs.py`, ...) and is a
`@config`-decorated pydantic dataclass (the `@config` decorator wraps
`pydantic.dataclasses.dataclass`). This is the layout the v0.7.3 ground-truth methodology
predicted would land "post-0.10.x".

## Sources

### Primary source (authoritative)

Pinned vllm 0.19.1 source tree, installed via
`research/mining-substrate-trial/scripts/venv_setup.py --engine vllm --version-slug v0_19_1`.
Tree root:

    /tmp/trial_vllm_v0_19_1_venv/src/vllm

The setup is source-only (no Python interpreter, no GPU). Confirmed version `0.19.1` via
`vllm-0.19.1.dist-info` and `_version.py`.

Files walked (line counts at scan time):

| File | LOC | Role |
| --- | --- | --- |
| `vllm/engine/arg_utils.py` | 2348 | `EngineArgs` @dataclass (lines 372-638); thin overlay over subconfigs. `AsyncEngineArgs` (line 2249) out of scope. |
| `vllm/sampling_params.py` | 922 | `SamplingParams` msgspec.Struct + `__post_init__`/`_verify_args`/`_verify_greedy_sampling`; `BeamSearchParams`; `StructuredOutputsParams` (renamed from GuidedDecodingParams); `RepetitionDetectionParams` (new). |
| `vllm/config/*.py` | ~30 modules | The subpackage. 29 in-scope config classes enumerated. Key modules: model.py, cache.py, parallel.py, scheduler.py, speculative.py, compilation.py, kv_transfer.py, structured_outputs.py, observability.py, offload.py, multimodal.py, lora.py, vllm.py. |
| `vllm/config/__init__.py` | 134 | Re-export surface; the public `vllm.config.X` names still resolve. |
| `vllm/envs.py` | 1858 | `environment_variables` dict. **238 keys** (was 84). |
| `vllm/pooling_params.py` | 233 | `PoolingParams` @dataclass (grew 1 -> 10 fields). |
| `vllm/platforms/{cpu,cuda,rocm,xpu,tpu,interface}.py` | - | Per-platform `check_and_update_config` (the platform-conditional invariant surface). |

### Secondary cross-reference

- **GitHub tag `v0.19.1`** of `vllm-project/vllm` is the canonical release the venv pins to.
- docs.vllm.ai was not consulted as authoritative; in-source docstrings and the dataclass
  field declarations are sufficient and avoid documentation-vs-code drift.

## Method

### Schema (field) enumeration

Because the surface is ~3x larger than v0.7.3 (185 EngineArgs fields, 29 subconfig classes,
396 subconfig fields, 238 env vars), field enumeration was done by **AST extraction** (Python
`ast`, stdlib only, run with the system interpreter against the on-disk source) rather than by
hand, to guarantee 1:1 coverage and exact line numbers. For each class the extractor records,
per field: the verbatim `type_annotation` (ast.unparse of the annotation) and the verbatim
`default_expr` (ast.unparse of the RHS, preserving `Field(...)`, `get_field(...)`, `field(...)`
wrappers so declarative `ge`/`gt`/`le` constraints and `init=False` markers stay visible).

The EngineArgs overlay pattern is preserved faithfully: a field like `model: str = ModelConfig.model`
keeps `default_expr = "ModelConfig.model"`; the authoritative literal default lives in the
referenced subconfig field, captured in the `subconfigs` namespace. This is the most honest
representation of how v0.19.1 actually wires defaults.

For env vars, the keys of `environment_variables` were extracted by walking the dict literal
(238 keys). Per-key behaviour summaries were NOT transcribed for the full set (CI-affordability:
the cost-frontier framing says the substrate must stay cheap on every upstream bump); the
high-leverage entries are annotated in `delta.md` / `version_delta.md`.

### Invariant enumeration

1. Ran an AST sweep for every `raise`, `assert`, and `logger.warning/info/warning_once` /
   `warnings.warn` across `vllm/config/*`, `sampling_params.py`, `pooling_params.py`, and the
   per-platform modules. Result: ~450 hit sites (vs ~60 enumerated for v0.7.3).
2. Filtered to the **LLEM scope** matching the v0.7.3 GT: caller-touchable,
   config-construction-time, energy/fairness-relevant. 79 invariants enumerated in the catalogue.
   Excluded the long tail (compile-internal asserts, distributed-init races, kernel-dispatch
   `NotImplementedError`, `get_*` accessor asserts).
3. For each in-scope invariant: identified predicate kind, emission channel, severity, minimal
   `kwargs_positive`/`kwargs_negative`, and `silent_normalisation: {field, declared, observed}`
   for the dormant / silent-override cases.
4. Tagged each with `in_baseline` (present-by-predicate in the v0.7.3 GT) and a `delta_v073`
   note: UNCHANGED / CHANGED / ADDED / REMOVED / MOVED / RENAMED.

### Field-coverage cross-check

EngineArgs field count (185) cross-checked via AST against the `@dataclass` body. Env-var count
(238) cross-checked by walking the `environment_variables` dict-literal keys. Subconfig class
inventory (29 in-scope) cross-checked against `config/__init__.py`'s `__all__`.

## Confidence per section

| Section | Confidence | Rationale |
| --- | --- | --- |
| `engine_params` (EngineArgs) | High | AST-extracted from arg_utils.py:372-638; every field's annotation + default_expr verbatim. |
| `sampling_params` | High | msgspec fields explicit; validators are the entirety of `_verify_args` (427-519) + `__post_init__` (373) + `_verify_greedy_sampling` (521). |
| `beam_search_params` | High | 6 fields, unchanged shape from v0.7.3. |
| `pooling_params` | High (out-of-LLEM-scope) | 10 fields enumerated; flagged out of text-gen scope. |
| `structured_outputs_params` | High | Renamed struct; 10 fields + 2 post_init raises read directly. |
| `subconfigs` | High for field shapes | All 29 in-scope classes AST-extracted. MEDIUM where a field's *runtime* default is computed (e.g. `block_size=None` then resolved to 16/128 by platform) - the declared default is captured, the resolution is noted in invariants. |
| `engine_envs` | High for the key set (238 enumerated, citation-bound) | MEDIUM on per-key behaviour: only high-leverage entries annotated; the bulk are key+line only by design (cost-frontier). |
| Invariants - SamplingParams | High | Every raise in `_verify_args` traced; exception-type changes (VLLMValidationError) and predicate changes (top_k, repetition_penalty, logprobs -1) verified against source. |
| Invariants - subconfig classes | High where the raise is a literal predicate; MEDIUM for cross-config (`verify_with_parallel_config`) and the declarative `Field(ge=...)` constraints (which raise pydantic.ValidationError at construction, not a custom message). |
| Invariants - platform `check_and_update_config` | MEDIUM | Predicates correctly identified and cited, but exercising them requires real platform context; kwargs encode `platform:`/`env_*` placeholders a downstream runner must inject. The MLA relocation (VllmConfig -> CpuPlatform, non-GPU-only) is HIGH confidence (read both the v0.7.3 and v0.19.1 sites). |

## Out-of-scope / known limitations

- **`AsyncEngineArgs`** (arg_utils.py:2249) - out of scope (synchronous engine focus).
- **Quantization sub-config tree** - `vllm/model_executor/layers/quantization/` has ~25 method
  modules (awq, gptq, fp8, fp4, mxfp4, mxfp8, modelopt, compressed_tensors, quark, ...). The
  `QuantizationMethods` Literal (24 entries) and `DEPRECATED_QUANTIZATION_METHODS` (5) are
  captured; per-quantizer config-class fields are NOT enumerated (treat `quantization` as an enum
  + opaque per-method config, as in v0.7.3).
- **The full ~200-site invariant tail** under VllmConfig.__post_init__ (now ~600 LOC, the single
  biggest growth area), compilation.py, and the platform modules is summarised, not exhaustively
  transcribed. The catalogue captures the caller-touchable, config-time subset.
- **Per-platform validation surface beyond `check_and_update_config`** (attention-backend dispatch
  raises, dtype-support checks) is referenced but not enumerated per platform.
- **InitVar shims on ModelConfig** (`limit_mm_per_prompt`, `mm_*`) forward into MultiModalConfig;
  captured as fields but their forwarding logic in `__post_init__` is summarised.

## Reviewer audit-path

For any catalogue entry: open `/tmp/trial_vllm_v0_19_1_venv/src/vllm/<source.file>` at the cited
line and confirm the field declaration / `raise` / `logger.warning` matches. Note that `config.py`
no longer exists - all config citations are under `config/<module>.py`. If a citation does not
resolve cleanly, treat that entry as low-confidence and revalidate.
