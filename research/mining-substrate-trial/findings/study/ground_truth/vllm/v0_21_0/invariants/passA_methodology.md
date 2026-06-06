# Pass A - entry-point / call-graph walk methodology (vLLM 0.21.0)

Engine source: `/tmp/vllm-0.21.0/vllm/` (confirmed `__version__ = '0.21.0'` via
`vllm/_version.py`).

Output: `passA_entrypoint.yaml`. This is the entry-point/call-graph half of a
two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal: maximise
recall of construction-time validation invariants reachable from the public,
user-facing entry points a benchmark harness constructs.

## Provenance

There is NO PoC ground truth for v0_21_0 (under
`findings/ground_truth/vllm/` only `v0_19_1` and `v0_7_3` exist). Nothing was
folded; every invariant is `provenance: net_new`, derived independently from the
0.21.0 source. The existing `v0_19_1` study cell was skimmed once for vLLM field
shapes and predicate-kind vocabulary, then this version's content was re-derived
line-by-line.

## Traversal (what I walked)

Public roots a harness actually constructs:

1. `vllm.LLM(...)` (`vllm/entrypoints/llm.py`) -> the single-process
   data-parallel guard (llm.py:333) -> `EngineArgs` -> `create_engine_config`
   -> the per-aspect `vllm.config.*` dataclasses.
2. Every leaf `vllm.config.*` dataclass reached from the engine config:
   CacheConfig, LoRAConfig, SchedulerConfig, PoolerConfig,
   StructuredOutputsConfig, KVTransferConfig, ECTransferConfig, OffloadConfig
   (+ nested UVA/Prefetch), ObservabilityConfig, MultiModalConfig, EPLBConfig,
   ParallelConfig, CompilationConfig, ProfilerConfig, KVEventsConfig,
   AttentionConfig, KernelConfig, LoadConfig, DeviceConfig, MambaConfig,
   ModelConfig, SpeculativeConfig.
3. `vllm.SamplingParams(...)` -> `__post_init__` -> `_verify_args` /
   `_verify_greedy_sampling`; nested `StructuredOutputsParams.__post_init__` and
   `RepetitionDetectionParams.__post_init__`; the model-config-dependent
   `verify()` / `_validate_logprobs` (dormant).
4. `vllm.PoolingParams(...)` -> `__post_init__` (output_kind assert); `verify()`
   chain is model-config-dependent (dormant).

## Method

- Counted validation sites per file
  (`raise`/`assert`/`@field_validator`/`@model_validator`/`__post_init__`/
  `logger.warning`/`Field(...)`/`Literal[`) across `config/*.py`,
  `sampling_params.py`, `pooling_params.py`, `engine/arg_utils.py`, then read
  each candidate in source context to classify predicate + outcome +
  replayability. Every cited line was read at this version (no line numbers
  inherited from another version).
- Enumerated declarative `Field(ge/gt/le/lt)` constraints and `Literal[...]` /
  StrEnum fields and emitted each as its own invariant (they raise
  `pydantic.ValidationError` at the bare constructor).
- For every cross-field / `@model_validator(mode="after")` check, recorded
  `kwargs_positive` (fires) and `kwargs_negative` (passes), choosing values that
  isolate the target raise from earlier raises in the same validator (e.g.
  SchedulerConfig uses `enable_chunked_prefill=true` + high
  `max_num_batched_tokens` to reach later branches; SamplingParams `top_k=1.0`
  isolates the TypeError from the `< -1` ValueError).

## CPU-replayability model

- Most `vllm.config.*` leaf dataclasses and SamplingParams / PoolingParams /
  StructuredOutputsParams / RepetitionDetectionParams construct CPU-only (pure
  pydantic / dataclass validation, no CUDA, no model dir). These carry
  `kwargs_positive` / `kwargs_negative`.
- `dormant_reason` is attached when a check needs a resolved ModelConfig (HF
  config load -> model dir), a CUDA/ROCm/XPU platform, a distributed world, a
  live tokenizer, or a real file on disk. These cannot replay on a source-only
  CPU host.
- `severity: dormant` + `outcome: normalise` marks the two silent
  normalisations (SamplingParams seed=-1 -> None; LoRAConfig max_cpu_loras=None
  -> max_loras): no raise/warn, observable only via a post-construction
  attribute.

## Version-specific findings vs the v0_19_1 cell

1. **SchedulerConfig cross-checks are now CPU-replayable.** `max_model_len` and
   `is_encoder_decoder` became `InitVar`s consumed by `__post_init__`, which
   calls `verify_max_model_len(max_model_len)`. In v0_19_1 these were dormant
   ("verify_max_model_len takes max_model_len as an argument"); at 0.21.0 you
   pass `max_model_len=` as a constructor kwarg and the full family of
   cross-checks fires at construction.
2. **PoolerConfig grew.** New `pooling_type` / `seq_pooling_type` /
   `tok_pooling_type` Literal fields and new `__post_init__` raises:
   `logit_bias`/`logit_mean` conflict, `logit_scale`/`logit_sigma` conflict,
   `logit_scale==0`, `logit_sigma==0`, plus the two pooling-type conflicts. (The
   `seq_pooling_type` vs `tok_pooling_type` confusion from earlier versions is
   resolved by the three-field split.)
3. **CacheConfig.gpu_memory_utilization** default moved 0.9 -> 0.92; the `gt=0,
   le=1` constraint is unchanged. New `CacheDType`, `PrefixCachingHashAlgo`,
   `MambaDType`, `KVOffloadingBackend` Literals.
4. **New configs walked:** MambaConfig (string-parsed backend enum + CUDA/
   Blackwell-gated stochastic rounding), ProfilerConfig (profiler Literal +
   torch_profiler_dir cross-checks + ge constraints), OffloadConfig
   (restructured into UVA/Prefetch sub-configs with a num_in_group/group_size
   and prefetch_step>=1 cross-check), AttentionConfig (flash_attn_version
   Literal[2,3,4] + cuDNN-prefill-removed raise), KernelConfig (moe_backend
   Literal), KVEventsConfig (publisher Literal), ObservabilityConfig version-
   parse validator + kv_cache_metrics_sample range.
5. **SamplingParams** gained `logprob_token_ids` with a `MAX_LOGPROB_TOKEN_IDS=
   128` cap, enforced in `verify()/_validate_logprobs` (model-config-dependent ->
   dormant). The `_verify_args` numeric/type/cross-field set is otherwise stable
   (line numbers shifted ~19 down).
6. **Entry-point gate:** `LLM(data_parallel_size>1)` single-process raise
   (llm.py:333) is the first user-facing gate; dormant for pure kwargs (LLM
   needs a model) but a real construction-time raise.

## Coverage

- Full SamplingParams `_verify_args` + `_verify_greedy_sampling` +
  `__post_init__` predicate set; StructuredOutputsParams +
  RepetitionDetectionParams.
- All CPU-constructible leaf config dataclasses' Field constraints, Literals/
  StrEnums, `@field_validator`s and `@model_validator(mode="after")` cross-field
  checks.
- ParallelConfig + EPLBConfig field/cross-field checks (platform-gated EPLB and
  ray-nsight flagged dormant).
- ModelConfig / SpeculativeConfig headline raises captured as dormant (both
  load/resolve a model config from a model dir).

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. **Validators on base/sibling classes not on the public construction path.**
   I only walk what `LLM(...)` / `SamplingParams(...)` / `PoolingParams(...)`
   construction touches, plus each leaf config's own constructor. Abstract bases
   or alternative config entry points (e.g. per-connector KV/EC subclasses,
   tensorizer LoadConfig branch) may carry validators I did not route to.
2. **create_engine_config cross-config wiring.** Many `arg_utils.py` raises fire
   only after ModelConfig resolves (model dir) and wires CacheConfig /
   SchedulerConfig / ParallelConfig together; I captured the leaf-level checks
   but not the full `create_engine_config` cross-config matrix (all
   model-dir-dependent -> dormant anyway).
3. **Platform `check_and_update_config` mutation chains.** CpuPlatform /
   CudaPlatformBase / XPUPlatform `check_and_update_config` apply warn+normalise
   and a few raises on a fully-built VllmConfig; these need a resolved config
   and a specific platform and were not enumerated here (all dormant). A
   hierarchy walk over the platform registry would surface them.
4. **Field-validator overrides across the MRO.** A subclass that re-declares a
   parent's validator with a different predicate can be missed by a per-file
   def scan; MRO-ordered traversal is the right tool.
5. **C-side / msgspec constraints.** SamplingParams / PoolingParams are
   `msgspec.Struct`s; any constraint enforced in the msgspec decode path (vs the
   Python `__post_init__`) is invisible to a call-graph walk.
