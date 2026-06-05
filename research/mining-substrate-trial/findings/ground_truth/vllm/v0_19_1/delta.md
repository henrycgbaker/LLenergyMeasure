# Delta vs LLEM baseline - vllm 0.19.1

There is **no `engine_versions/vllm/v0_19_1/outputs/` baseline** on disk (only the
`producers/` mining scripts: `schema_introspector.py`, `static_invariant_miner.py`). So
"baseline" here means *what the existing static-mining producers would emit for v0.19.1*,
contrasted against this ground truth. The structural conclusions transfer directly from the
v0.7.3 exercise: the producers capture `EngineArgs` + `SamplingParams` shapes and a handful of
enum/range invariants, and miss the entire subconfig tree, the env surface, and all cross-field /
silent-normalisation invariants.

## Headline numbers (ground truth)

| Surface | Ground truth v0.19.1 | v0.7.3 GT (ref) | Note |
| --- | --- | --- | --- |
| Engine params (EngineArgs) | **185 fields** | 103 | +82; now a thin overlay over subconfigs |
| SamplingParams | **32 fields** | 29 | best_of/_real_n/truncate removed; flat_logprobs/extra_args/repetition_detection/thinking_token_budget added |
| BeamSearchParams | 6 | 6 | unchanged |
| PoolingParams | 10 | 1 | grew from placeholder |
| StructuredOutputsParams (was GuidedDecodingParams) | 10 | 7 | renamed + structural_tag/disable_additional_properties |
| RepetitionDetectionParams | 5 | 0 | new struct |
| Subconfig classes (full enumeration) | **29 classes / 396 fields** | 18 classes / 125 fields | subpackage layout |
| **vllm.envs env vars** | **238** | 84 | ~2.8x growth |
| **Invariants (LLEM-scope catalogue)** | **79 enumerated** (of ~200 total raise/warn sites) | 86 enumerated | scope-matched subset |

## Primary gap: `vllm.envs` (+238 entries)

The static producers enumerate `engine_params` and `sampling_params` only. The
`environment_variables` dict (`vllm/envs.py:489-1666`) now defines **238** runtime-behaviour
env vars - nearly triple v0.7.3. A static field miner sees none of them. High-leverage examples
still present and still invisible:

- **`VLLM_MLA_DISABLE`** (envs.py:1076) - read by `ModelConfig.use_mla` (model.py:1581); flips
  the MLA path that, on non-GPU platforms, silently disables chunked prefill + prefix caching.
- **`VLLM_ALLOW_LONG_MAX_MODEL_LEN`** (envs.py:861) - gates the `max_model_len > derived` raise
  (`model.py:2178`). The textbook env-var-gated invariant, unchanged.
- **`VLLM_MAX_N_SEQUENCES`** (envs.py:884, default 16384) - **NEW** upper bound on
  `SamplingParams.n`; a benchmark that bumps `n` past it gets a hard error gated purely by env.
- **`VLLM_ATTENTION_BACKEND`** equivalent is now resolved per-platform; **`VLLM_USE_DEEP_GEMM`**,
  the `VLLM_ROCM_USE_AITER_*` family (~20 vars), `VLLM_USE_FLASHINFER_MOE_*` (~8 vars), and
  `VLLM_TORCH_PROFILER_DIR`-style profiler controls all change energy-per-token and are absent
  from any static field catalogue.
- **`VLLM_BATCH_INVARIANT`** (envs.py:503) - NEW; forces deterministic batch-invariant kernels,
  a direct energy/throughput tradeoff a benchmark MUST record.

## Invariants the static producers miss (same classes as v0.7.3, larger)

1. **Silent-normalisation invariants** - the highest-value class. v0.19.1 has *more* of them and
   has *relocated* the marquee one: MLA-disables-chunked-prefill moved from `VllmConfig.__post_init__`
   (all platforms) to `CpuPlatform.check_and_update_config` (non-GPU only). New silent overrides:
   `enforce_eager` -> compile mode NONE (vllm.py:847); CPU fp8-KV-cache -> auto (cpu.py:186);
   CUDA mm-prefix-lm -> disable_chunked_mm_input (cuda.py:199); XPU no-graph -> cudagraph NONE
   (xpu.py:180). A caller who sets these flags and never reads the log records the *declared*, not
   the *observed*, config.
2. **Declarative pydantic constraints** - many v0.7.3 imperative raises are now `Field(ge=, gt=, le=)`
   on the dataclass field (e.g. `gpu_memory_utilization` gt=0/le=1, `max_loras` ge=1,
   `num_speculative_tokens` gt=0). These raise `pydantic.ValidationError` at construction. A miner
   that only scans for `raise` statements will MISS them - it must also parse `Field(...)` kwargs.
   This is a NEW mining requirement introduced by the subpackage refactor.
3. **Cross-config / cross-subconfig invariants** - unchanged class, now spread across modules:
   EPLB-requires-expert-parallel (parallel.py:370), elastic-EP-requires-EPLB (parallel.py:644),
   draft/target vocab-size match (speculative.py:835), detailed-traces-requires-otlp (observability.py:149).
4. **Per-platform invariants** (task item 8) - entirely invisible to a field miner; CPU/CUDA/XPU
   each have a `check_and_update_config` with platform-conditional silent overrides and raises.

## 3 concrete high-value findings for the substrate

1. **The subpackage refactor changed the mining substrate's job.** v0.7.3 needed one walk of
   `config.py`; v0.19.1 needs a walk of ~30 `config/*` modules PLUS the platform modules PLUS a
   `Field(...)` constraint parser (declarative pydantic validators are now a first-class invariant
   source). A miner that greps `config.py` finds nothing.
2. **`vllm.envs` tripled (84 -> 238).** The env surface is the fastest-growing part of vLLM's config
   and the part a static field miner is structurally blind to. Cost-frontier implication: even a
   key-only env catalogue is high-value and cheap (one dict walk).
3. **Silent normalisations relocated to platform code.** The MLA / chunked-prefill override that
   v0.7.3 surfaced at the VllmConfig level is now non-GPU-only and lives in `platforms/cpu.py`. A
   substrate that only inspects `vllm/config/*` will report it as removed when it actually moved -
   the platform modules are now mandatory mining input.
