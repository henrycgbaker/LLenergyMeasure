# Pass A - entry-point / call-graph walk methodology (vLLM 0.22.0)

Engine source: `/tmp/vllm-0.22.0/vllm/`. This pass is the entry-point /
call-graph half of the two-pass bake-off; a sibling pass does a class-hierarchy
walk. Goal: maximise recall of construction-time validation invariants reachable
from the public, user-facing entry points a benchmark harness constructs.

Output: `passA_entrypoint.yaml` (109 invariants).

## Traversal (what I walked)

Starting roots (public surface):

1. `vllm.LLM(...)` (vllm/entrypoints/llm.py) -> `EngineArgs` /
   `create_engine_config` (vllm/engine/arg_utils.py) -> the per-aspect configs
   in `vllm/config/*.py`: `ModelConfig`, `CacheConfig`, `ParallelConfig`
   (+ nested `EPLBConfig`), `SchedulerConfig`, `DeviceConfig`, `LoadConfig`,
   `LoRAConfig`, `SpeculativeConfig`, `StructuredOutputsConfig`,
   `CompilationConfig`, `KVTransferConfig`, `ECTransferConfig`,
   `ObservabilityConfig`, `OffloadConfig` (+ nested `UVAOffloadConfig` /
   `PrefetchOffloadConfig`), `PoolerConfig`, and `VllmConfig` cross-config
   `__post_init__`.
2. `vllm.SamplingParams(...)` (vllm/sampling_params.py) -> `__post_init__` ->
   `_verify_args` / `_verify_greedy_sampling`; plus the
   `StructuredOutputsParams` and `RepetitionDetectionParams` dataclass
   `__post_init__` guards.
3. The platform hook `CpuPlatform.check_and_update_config`
   (vllm/platforms/cpu.py) reached at the end of config resolution.

## Method

- Read each config module in full and enumerated every `__post_init__`,
  `@field_validator`, `@model_validator(mode="after")`, declarative
  `Field(ge/gt/le/lt)` / `Literal[...]`, explicit `raise`, `assert`, and
  `logger.warning` / `warning_once` / `warnings.warn` reachable from
  construction. Each predicate, outcome (invalid / warn / normalise) and
  replayability classification was verified against the actual v0.22.0 source
  line.
- `SamplingParams` was read end-to-end; every `_verify_args` predicate was
  re-derived with its current line number (line numbers shifted vs earlier
  versions; the v0.22.0 _verify_args added a `bad_words` empty-string guard).
- `native_type` is always a dotted importable path the gate constructs. All of
  `EPLBConfig`, `PoolerConfig`, `OffloadConfig`, `UVAOffloadConfig`,
  `PrefetchOffloadConfig`, `StructuredOutputsConfig`, `KVTransferConfig`,
  `ECTransferConfig`, `ObservabilityConfig`, `DeviceConfig`, `LoadConfig`,
  `SpeculativeConfig`, `CompilationConfig` are re-exported from `vllm.config`
  (confirmed in `vllm/config/__init__.py`), so `vllm.config.<Class>` is valid.
  `StructuredOutputsParams` / `RepetitionDetectionParams` live in
  `vllm.sampling_params`.

## Provenance

No PoC ground truth exists for v0.22.0 at
`research/mining-substrate-trial/findings/ground_truth/vllm/v0_22_0/` (only
`v0_7_3` and `v0_19_1` exist there). Per the Pass A instructions, every entry is
therefore `provenance: net_new`, re-derived from this version's source. The
v0_19_1 study cell was skimmed for vLLM field SHAPES only (id naming,
predicate_kind vocabulary, kwargs layout); no content was copied without
re-deriving the predicate and re-resolving the citation against v0.22.0.

## Coverage

- Full `SamplingParams` predicate set (n, penalties, temperature, top_p, top_k,
  min_p, max/min_tokens, logprobs, prompt_logprobs, stop / stop_token_ids /
  bad_words, greedy n==1, seed normalisation) + `StructuredOutputsParams`
  mutual-exclusion + `RepetitionDetectionParams`.
- All declarative `Field` numeric/Literal constraints on `CacheConfig`,
  `LoRAConfig`, `SchedulerConfig`, `EPLBConfig`, `ObservabilityConfig`,
  `LoadConfig`, `UVAOffloadConfig`, `PrefetchOffloadConfig`,
  `SpeculativeConfig`, `DeviceConfig`, `StructuredOutputsConfig`.
- All `ParallelConfig` cross-field validators (`_validate_parallel_config`,
  `__post_init__`, `_verify_args`) + the new `numa_bind_nodes` /
  `numa_bind_cpus` field validators.
- `PoolerConfig.__post_init__` deprecation/mutual-exclusion raises (out of
  LLEM text-gen scope; enumerated for catalogue completeness).
- `KVTransferConfig` / `ECTransferConfig` role guards;
  `StructuredOutputsConfig`, `ObservabilityConfig`, `OffloadConfig`,
  `CompilationConfig` cross-field validators.
- `ModelConfig` / `VllmConfig` / `CpuPlatform` headline checks captured as
  dormant (need a model dir / GPU / resolved sub-config), citations re-resolved
  to v0.22.0 lines.

## CPU-replayable vs dormant (for the downstream gate)

79 CPU-replayable, 30 dormant.

- CPU-replayable: all `SamplingParams` / `StructuredOutputsParams` /
  `RepetitionDetectionParams` predicates, and every per-config-object validator
  that constructs from bare kwargs (or a nested dict) without a model dir,
  CUDA/ROCm/XPU device, distributed world, or optional package. `SchedulerConfig`
  cross-checks replay via `SchedulerConfig.default_factory(...)` (annotated
  `replay_via`), which supplies the `max_model_len` / `is_encoder_decoder`
  InitVars; `OffloadConfig` / `ParallelConfig.num_redundant_experts` replay via
  nested dicts.
- Dormant (`dormant_reason` set): every `ModelConfig` check (`__post_init__`
  loads the HF config), every `VllmConfig.__post_init__` cross-config warn/raise
  (needs a resolved ModelConfig), the `CpuPlatform.check_and_update_config`
  normalisation (needs a built VllmConfig + MLA model), `SpeculativeConfig`
  deep validators (resolve a draft ModelConfig), platform-gated
  `ParallelConfig` branches (EPLB CUDA-gate, world-size-vs-GPU, ray-only paths),
  the `VLLM_LORA_ENABLE_DUAL_STREAM` raise (env-gated, no kwarg), and
  `otlp_traces_endpoint` (depends on whether opentelemetry is importable).
- Two `EPLBConfig` validator raises (`use_async` + non-default policy;
  `log_balancedness_interval <= 0`) are marked dormant because the field-level
  `Literal["default"]` / `Field(gt=0)` reject the bad value BEFORE the
  `@model_validator` runs, so the validator raise is unreachable through the
  normal kwargs path. Same reasoning for `PoolerConfig` unknown-pooling-type
  (the Literal union pre-empts the `NotImplementedError`).

## Version-specific notes vs the v0.19.1 shape

- `CacheConfig.gpu_memory_utilization` default changed to 0.92 (still
  `Field(gt=0, le=1)`).
- `SamplingParams._verify_args` added a `bad_words` empty-string guard
  (line 546); line numbers across `_verify_args` shifted.
- `LoRAConfig` gained `max_lora_rank` `Literal[MaxLoRARanks]` and an env-gated
  `VLLM_LORA_ENABLE_DUAL_STREAM` CUDA-only raise.
- `ParallelConfig` was heavily refactored: new `numa_bind_nodes` /
  `numa_bind_cpus` field validators, `all2all_backend` pplx/naive removal-warn,
  DCP divisibility and `dcp_comm_backend='a2a'` checks.
- `OffloadConfig` was restructured into nested `UVAOffloadConfig` /
  `PrefetchOffloadConfig`; the num-in-group / group-size cross-check now lives
  under `prefetch`.
- `ObservabilityConfig` gained `kv_cache_metrics_sample` `Field(gt=0, le=1)` and
  a `show_hidden_metrics_for_version` PEP440-parse validator.
- `PoolerConfig.__post_init__` gained logit_bias/logit_mean and
  logit_scale/logit_sigma deprecation/exclusion raises.
- The CPU-platform MLA handling is now `logger.info` + normalisation (was a
  warning in earlier shapes); the earlier CPU FP8-KV-cache RuntimeError is not
  present in this form at v0.22.0, so it was not fabricated.

## Blind spots (what a class-hierarchy walk should catch that this did not)

1. Validators on config classes never reached from the default `LLM(...)` /
   `SamplingParams(...)` text-gen path (e.g. pooling/embedding-only configs were
   enumerated opportunistically but not exhaustively MRO-walked).
2. Constraints enforced only after a real `ModelConfig` is resolved - every
   ModelConfig/VllmConfig/CpuPlatform predicate here is dormant because the
   source host has no model dir; their exact firing order under a real model is
   not verified.
3. Pydantic field validators defined on a parent `@config` base and overridden
   in a child: the read-by-module approach can miss an override that changes a
   predicate.
4. Deep `SpeculativeConfig` per-method logic (eagle / medusa / mtp / draft) that
   only fires once a draft ModelConfig is built - confirmed dispatch shape but
   not each per-method raise.
