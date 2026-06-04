# Ground-truth methodology - vllm 0.7.3

## Scope

The deliverable answers two questions for the LLEM mining-substrate trial:

1. **What is the complete, documented, type-aware schema** of vllm 0.7.3's caller-touchable configuration surface (engine, sampling, beam-search, pooling, guided-decoding, every subconfig class, AND the `vllm.envs` env-var control surface)?
2. **What are the complete, citation-bearing invariants** that vllm 0.7.3 enforces at config / params construction time (raises, warnings, silent normalisations, deprecations)?

Coverage target: every public-shape field and every constraint a benchmark caller can plausibly trip.

## Sources

### Primary source (authoritative)

Pinned vllm 0.7.3 source tree, installed via `research/mining-substrate-trial/scripts/venv_setup.py --engine vllm --version-slug v0_7_3`. Tree root:

    /tmp/trial_vllm_v0_7_3_venv/src/vllm

Files walked exhaustively (line counts at scan time):

| File | LOC | Role |
| --- | --- | --- |
| `vllm/engine/arg_utils.py` | 1425 | `EngineArgs` dataclass (lines 91-211); `AsyncEngineArgs` is out of scope. |
| `vllm/sampling_params.py` | 508 | `SamplingParams` msgspec.Struct + `__post_init__` + `_verify_args` + `_verify_greedy_sampling`; `BeamSearchParams` msgspec.Struct; `GuidedDecodingParams` @dataclass + `__post_init__`. |
| `vllm/config.py` | 3489 | Flat module pre-refactor. All subconfig classes plus `_get_and_verify_dtype` / `_get_and_verify_max_len`. |
| `vllm/envs.py` | 609 | `environment_variables: Dict[str, Callable[[], Any]]` (lines 120-596). 84 keys. The TYPE_CHECKING stub block at the top (lines 7-92) is a typing aid only; the lambdas in `environment_variables` are authoritative. |
| `vllm/pooling_params.py` | 25 | `PoolingParams` msgspec.Struct (placeholder in 0.7.3). |
| `vllm/beam_search.py` | 73 | `BeamSearchSequence`, `BeamSearchOutput`, scoring helpers - downstream of `BeamSearchParams`. |

### Secondary cross-reference

- **GitHub tag `v0.7.3`** of `vllm-project/vllm` was used as a tiebreaker for any line numbers that might differ between the on-disk venv and the canonical release. The venv was set up by `venv_setup.py` to install vllm==0.7.3, so the source matches the upstream tag.
- **docs.vllm.ai (0.7.x docs)** were not consulted as authoritative; docstrings inside the source are sufficient and avoid documentation-vs-code drift.

## Method

### Schema (field) enumeration

1. Read existing baseline (`engine_versions/vllm/v0_7_3/outputs/schema.discovered.json` and `invariants.{proposed,validated}.yaml`).
2. Walked every `class` declaration in the six source files. For each class, enumerated:
   - dataclass / msgspec.Struct / pydantic.BaseModel fields with their type annotation and literal default,
   - `__post_init__` normalisations (e.g. `None -> []`, `str -> [str]`, `int -> 1 if True`),
   - `_verify_*` methods (constraint enforcers - feed into invariants),
   - cross-field constraints from `verify_with_*_config` methods,
   - private/internal fields (`_real_n`, `world_size`, `num_gpu_blocks`, `chunked_prefill_enabled`, etc.) flagged with `internal: true` / `scope: subconfig_only`.
3. For `EngineArgs` vs subconfig field overlap, both surfaces are recorded:
   - `engine_params` namespace captures CLI-touchable surface (every `EngineArgs` field),
   - `subconfigs` namespace adds the subconfig-only fields that have no `EngineArgs` counterpart (e.g. `ParallelConfig.world_size`, `SchedulerConfig.send_delta_data`, `ParallelConfig.sd_worker_cls`, `LoRAConfig.bias_enabled`, etc.) plus any default that differs between the two layers (e.g. `SchedulerConfig.max_num_seqs` default 128 but `EngineArgs.max_num_seqs` default None).
4. For env vars, enumerated every key of `environment_variables` (lines 126-595 in `vllm/envs.py`). Each entry's `behaviour` field is summarised from the docstring-comment above the lambda. Where the env-var key in the `environment_variables` dict and the TYPE_CHECKING stub disagree (`VLLM_CUDA_MEM_ALIGN_KV_CACHE` vs `VLLM_MLA_CUDA_MEM_ALIGN_KV_CACHE`), the dict key is treated as authoritative because that's what `envs.__getattr__` actually exposes (line 601). Same for `VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH` whose lambda reads `VLLM_CONTIGUOUS_PA` from os.environ - a discrepancy flagged inline.
5. Every entry carries `source: {file, line, qualname?}` with the line number resolved against the Read tool's output of the venv source.

### Invariant enumeration

1. Started with the 26 baseline invariants (validated.yaml) - inherited verbatim with `in_baseline: true`.
2. Walked every `raise`, `logger.warning`, `logger.info`, `logger.warning_once`, `warnings.warn`, `assert` in the six source files. For each:
   - Identified the predicate (lt / le / gt / ge / eq / ne / in / not_in / type_check / mutually_exclusive / cross_field / cross_subconfig / env_var_combo / platform_combo).
   - Identified emission channel and severity.
   - Identified what the user-touchable input is (e.g. `temperature=0.001` triggers the silent clamp; `max_lora_rank=15` triggers the enum rejection).
   - Constructed minimal `kwargs_positive` (triggers) / `kwargs_negative` (does not trigger) pairs.
3. Recorded `silent_normalisation: {field, declared, observed}` for all dormant / silent-override cases.
4. For cross-config invariants (e.g. `ModelConfig.verify_with_parallel_config` rejects non-divisible head/TP count), recorded both the predicate and the fact that it lives in a `verify_with_*` method that fires only after both subconfigs exist.
5. Marked every NEW invariant (not in baseline) with `in_baseline: false` and a `delta_note` explaining what it adds.

### Field-coverage cross-check

After the schema was written, ran a small Python check against `/tmp/trial_vllm_v0_7_3_venv/src/vllm/envs.py` confirming the 87 env vars in the catalogue exhaust the keys of `environment_variables` (the regex count of 84 missed three single-quoted keys: `VLLM_HOST_IP`, `VLLM_PORT`, `VLLM_RPC_BASE_PATH` - all three are in the catalogue).

EngineArgs field set was cross-checked against `grep -nE '^    [a-z_]+: ' vllm/engine/arg_utils.py` (103 fields when excluding the `AsyncEngineArgs.disable_log_requests` override and `out_dict` local), and matches the catalogue 1:1.

## Confidence per section

| Section | Confidence | Rationale |
| --- | --- | --- |
| `engine_params` (EngineArgs) | High | Mechanically enumerated from `arg_utils.py` lines 91-211; types and defaults read directly from source; cross-checked with baseline (96 fields overlap; 7 added types / enum constraints the baseline missed). |
| `sampling_params` | High | msgspec.Struct fields are explicit (lines 171-211); validators are the entirety of `__post_init__` (287-350) and `_verify_args` (352-410). |
| `beam_search_params` | High | 6 fields, msgspec.Struct, no validators beyond msgspec's required-field check. |
| `pooling_params` | High (but flagged out-of-LLEM-scope) | Single field, placeholder in 0.7.3 per source self-description. |
| `guided_decoding_params` | High | 7 fields, one constraint (mutual exclusion of constraint modes) at line 67. |
| `subconfigs` | High | Every class in `vllm/config.py` enumerated; constructor params / dataclass fields directly listed. Fields like `quant_config` and `world_size` flagged as internal / computed. |
| `engine_envs` | High | All 87 env-var entries citation-bound to specific lines; defaults read from the lambda. Two known discrepancies (lambda key vs TYPE_CHECKING stub for `VLLM_CUDA_MEM_ALIGN_KV_CACHE`; attr name vs env-var name for `VLLM_USE_HPU_CONTIGUOUS_CACHE_FETCH`) flagged inline. |
| Invariants - SamplingParams | High | 23 invariants vs baseline's 12; the 11 additions are mechanically read from `_verify_args` lines 358-410 plus `__post_init__` line 303 (warn+clamp) and `_verify_greedy_sampling` line 412. |
| Invariants - subconfig classes | High for ModelConfig/CacheConfig/LoRAConfig/SchedulerConfig/SpeculativeConfig (every `raise` traced); Medium for cross-config / cross-runtime invariants (e.g. `current_platform.is_cuda()` gating on `enable_sleep_mode`, `current_platform.is_rocm()` gating on `disable_custom_all_reduce`) - these are correctly identified in source but their kwargs_positive / kwargs_negative pairs encode platform context as `env_<NAME>` placeholders that a downstream runner must inject. |
| Invariants - env-var-gated | Medium | Invariants like `VLLM_ALLOW_LONG_MAX_MODEL_LEN` gating `_get_and_verify_max_len` raises were identified and cited, but exercising them requires real model loads (HuggingFace download). The static-mining substrate can detect the predicate but cannot validate it without an integration harness. |

## Out-of-scope / known limitations

- **`AsyncEngineArgs`** (subclass of `EngineArgs` at line 1396+ in `arg_utils.py`) adds `disable_log_requests: bool = False` - intentionally omitted because LLEM targets the synchronous engine surface.
- **`QuantizationConfig`** (referenced by `VllmConfig.quant_config`) is defined under `vllm/model_executor/layers/quantization/` and has per-method subclasses (AWQ, GPTQ, FP8, ...). Enumerating each subclass's fields is out of scope for this exercise - the catalogue captures the top-level reference and flags `quant_config` as `internal: true`.
- **`PoolerConfig.from_json`** parses JSON from CLI - not a config field per se, classmethod.
- **CompilationConfig private attrs** (`max_capture_size`, `local_cache_dir`, `bs_to_padded_graph_size`, `enabled_custom_ops`, `disabled_custom_ops`, `traced_files`, `compilation_time`, `static_forward_context`) are PrivateAttr / computed-not-input; listed conceptually under the class but not as configurable fields.
- **Platform classes** (`vllm.platforms.*`) gate several invariants (rocm, neuron, hpu, openvino, tpu). The invariant catalogue references `current_platform.is_*` checks but does not enumerate each platform's own validation surface.

## Reviewer audit-path

For any catalogue entry, the reviewer should be able to:

1. Open `/tmp/trial_vllm_v0_7_3_venv/src/vllm/<source.file>` at the cited line.
2. See the field declaration / `raise` / `logger.warning` exactly as cited.
3. Verify the predicate, default, type, and message_template match.

If a citation does not resolve cleanly, that catalogue entry should be treated as low-confidence and revalidated.
