# Pass B methodology - vLLM v0.20.0 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the vLLM config surface by TYPE, not by call path. Starting
from the public config + params classes, every reachable pydantic dataclass and
msgspec.Struct was walked and, for each, every construction-time validity rule
extracted. This is the COMPLEMENT to Pass A (which walks public-construction
entry points).

- Engine source: `/tmp/vllm-0.20.0/vllm/` (`version.py` / `_version.py` confirm
  `__version__ == '0.20.0'`).
- Pure source analysis, no GPU, no model download.

## Type tree enumerated

All `vllm/config/*.py` modules (modular config package) were read in full, plus
`vllm/sampling_params.py` and `vllm/pooling_params.py`. Classes covered:

- `cache.py`: `CacheConfig` (CacheDType / MambaDType / MambaCacheMode /
  PrefixCachingHashAlgo / KVOffloadingBackend Literals, Field ranges,
  calculate_kv_scales deprecation warn).
- `lora.py`: `LoRAConfig` (MaxLoRARanks / LoRADType Literals, max_loras Field,
  max_cpu_loras model_validator).
- `scheduler.py`: `SchedulerConfig` (RunnerType / SchedulerPolicy Literals,
  stream_interval Field NEW at 0.20.0, verify_max_model_len cross-field raises,
  InitVars max_model_len + is_encoder_decoder).
- `structured_outputs.py`: `StructuredOutputsConfig` (backend Literal +
  disable_any_whitespace / disable_additional_properties backend conflicts).
- `kv_transfer.py` / `ec_transfer.py`: connector-requires-role conflicts;
  kv_load_failure_policy Literal; dead get_args raises (kv_role now Literal-typed
  at 0.20.0, ec_role already Literal-typed).
- `kv_events.py`: publisher Literal default-None.
- `weight_transfer.py`: backend Literal (the file's ONLY invariant).
- `attention.py`: flash_attn_version Literal[2,3,4]; backend AttentionBackendEnum
  membership (KeyError on unknown).
- `multimodal.py`: MMCacheType / MMEncoderTPMode / MMTensorIPC (NEW) Literals;
  video_pruning_rate / mm_shm_cache_max_object_size_mb (NEW) Field ranges; shm
  cache-size-only-for-shm model_validator (NEW); XFORMERS-removed backend dispatch;
  Video/Image/Audio/Base DummyOptions Field(gt=0/ge=0) sub-configs (NEW).
- `offload.py` (NEW file): OffloadConfig backend Literal + num_in_group <=
  group_size + prefetch_step >= 1 model_validator; UVAOffloadConfig /
  PrefetchOffloadConfig Field ranges.
- `parallel.py`: EPLBConfig (policy single-member Literal, communicator Literal NEW,
  window_size / step_interval / log_balancedness_interval Field gt=0,
  num_redundant_experts ge=0); ParallelConfig (data_parallel_backend /
  expert_placement_strategy / dcp_comm_backend / all2all_backend NEW Literals,
  numa_bind_nodes / numa_bind_cpus field_validators NEW at 0.20.0).
- `profiler.py`: ProfilerKind Literal; torch_profiler_dir <-> profiler conflicts;
  active_iterations Field(ge=1).
- `observability.py`: show_hidden_metrics version-parse; collect_detailed_traces
  requires endpoint; kv_cache_metrics_sample Field range NEW; otlp requires
  opentelemetry (env-dependent, dormant).
- `mamba.py` (NEW file): MambaBackendEnum membership (custom metaclass __getitem__
  raises); stochastic-rounding GPU gate (dormant).
- `kernel.py` (NEW file): MoEBackend 11-member Literal.
- `quantization.py` (NEW file): OnlineQuantizationConfigArgs scheme Enum coercion.
- `pooler.py` (NEW file): logit_bias/logit_mean + logit_scale/logit_sigma
  exclusions, logit_sigma!=0, pooling_type vs seq/tok exclusion + membership.
- `speculative.py`: method / rejection_sample_method Literals (dormant: post_init
  resolves draft model).
- `model.py`: TokenizerMode and sibling Literals (dormant: needs real model dir).
- `sampling_params.py`: SamplingParams._verify_args (called from __post_init__,
  line 423) full numeric/range battery + 0.20.0 additions (min_tokens<=max_tokens,
  logprobs/prompt_logprobs non-negative-or-(-1), stop_token_ids ints, stop
  no-empty-string, stop-requires-detokenize, n<=VLLM_MAX_N_SEQUENCES env-gated);
  _verify_greedy_sampling; StructuredOutputsParams / RepetitionDetectionParams
  param dataclasses.
- `pooling_params.py`: PoolingParams output_kind FINAL_ONLY assert.

## Extraction per class

For each class the following were harvested:

1. `Literal[...]` / `Enum`-typed fields -> membership rules (`literal_in` /
   `strenum_in`).
2. Inline `Field(gt/ge/le/lt)` numeric constraints -> `range`.
3. `@field_validator` / `@model_validator` bodies -> every `raise` /
   `logger.warning`.
4. `__post_init__` (model_post_init) checks.
5. before-validator `Enum[value.upper()]` membership (KeyError/ValueError).

`outcome` derived from severity: error/raise -> `invalid`; warn -> `warn`;
catalogue self-check -> `meta`. native_type is the dotted importable path of the
owning type so the gate can construct it directly.

## kwargs replayability discipline

`kwargs_positive` (should fire) / `kwargs_negative` (should pass) were emitted
ONLY where the rule is reachable at plain CPU-only construction. All replay-paired
config classes are importable from `vllm.config`; SamplingParams / PoolingParams
from `vllm`; param dataclasses from `vllm.sampling_params`.

NO kwargs were fabricated for rules that are:
- GPU/platform-gated (`MambaConfig` stochastic rounding, `ModelConfig` Literals
  needing a real model dir).
- env-gated (`SamplingParams.n <= VLLM_MAX_N_SEQUENCES`, observability OTLP
  opentelemetry-import dependency).
- only reachable after draft-model resolution (`SpeculativeConfig` method /
  rejection_sample_method, whose __post_init__ resolves a draft model).

These carry `dormant_reason` instead.

## CPU-construction safety notes

- `@config` is `pydantic.dataclasses.dataclass` with `extra="forbid"`, so Literal /
  Field constraints raise pydantic `ValidationError` and `__post_init__` runs as
  `model_post_init`.
- `SchedulerConfig` requires the InitVars `max_model_len` and `is_encoder_decoder`;
  both are supplied in every SchedulerConfig replay pair.
- `ParallelConfig.__post_init__` runs distributed/env setup, but for the default
  DP=1 path it takes the env-var fallback branch and sets
  `distributed_executor_backend="uni"` with no GPU calls or port allocation, so the
  numa field_validators and the DP-backend Literals are CPU-replayable.
- `OffloadConfig` num_in_group <= group_size: the negative pair uses
  `offload_backend="auto"` (validator body not entered); the positive pair uses
  `offload_backend="prefetch"`, which with the sub-config defaults
  (num_in_group=1 > group_size=0) fires the raise.

## Result

- 92 total invariants, all `provenance: net_new` (no PoC ground truth exists for
  v0_20_0 under findings/study/ground_truth/vllm/v0_20_0 or the legacy
  findings/ground_truth/vllm/v0_20_0).
- 85 CPU-replayable (kwargs_positive + kwargs_negative pair).
- 7 dormant (lora_dtype torch.dtype-union, observability OTLP, speculative method +
  rejection_sample_method, model tokenizer_mode, mamba stochastic-rounding,
  sampling n<=VLLM_MAX_N_SEQUENCES).
- 1 warn-outcome (cache calculate_kv_scales deprecation).

## Class-tree cases an entry-point walk would miss

1. Declarative Literal/Enum field membership with NO coded raise on any path:
   cache dtypes (15-member CacheDType), MoEBackend (kernel.py), All2AllBackend
   (parallel.py), MMTensorIPC, MambaBackendEnum, AttentionBackendEnum,
   weight_transfer backend (the file's only rule), EPLBConfig communicator,
   OnlineQuantScheme.
2. Inline `Field(gt/ge/le/lt)` constraints scattered across the new modular
   sub-configs (offload UVA/Prefetch, multimodal DummyOptions, EPLB window/step,
   observability kv_cache_metrics_sample, scheduler stream_interval).
3. Dead-raise observations the type view surfaces:
   - `EPLBConfig.policy` is single-member Literal['default'], so the
     `_validate_eplb_config` async-policy raise is dead via direct construction;
     and `log_balancedness_interval` Field(gt=0) shadows the coded interval raise.
   - `kv_role` became `KVRole | None` Literal at 0.20.0 (was plain str at 0.19.1),
     so the manual `get_args(KVRole)` membership raise is now dead, mirroring
     `ec_role`.
4. PoolerConfig deprecated-alias mutual exclusions and zero-division guards that
   live only in `__post_init__`.

## Deltas vs vLLM 0.19.1 (Pass B)

- New config files mined: `pooler.py`, `mamba.py`, `kernel.py`, `quantization.py`,
  `offload.py` sub-configs, `multimodal.py` DummyOptions.
- `CacheDType` grew 8 -> 15 members.
- `SchedulerConfig.stream_interval`, `ObservabilityConfig.kv_cache_metrics_sample`,
  `MultiModalConfig.mm_tensor_ipc` + `mm_shm_cache_max_object_size_mb` +
  `_validate_multimodal_config`, `ParallelConfig.all2all_backend` + numa
  validators, EPLBConfig window/step/communicator are all NEW at 0.20.0.
- `kv_role` Literal-typing flips its get_args raise from reachable (0.19.1) to dead
  (0.20.0).
- `SamplingParams._verify_args` gained min_tokens<=max_tokens, logprobs /
  prompt_logprobs non-negative-or-(-1), stop_token_ids-int, stop-no-empty-string,
  stop-requires-detokenize, and the env-gated n<=VLLM_MAX_N_SEQUENCES upper bound.

## Blind spots (deferred to Pass A / a richer fixture)

A type-tree walk under-covers rules in the execution path rather than in a config
class: VllmConfig-level cross-config wiring in `__post_init__`, ModelConfig
HF-download-dependent Literals, SpeculativeConfig draft-model resolution, and any
runtime guard reachable only from the public `LLM(...)` entry point.
