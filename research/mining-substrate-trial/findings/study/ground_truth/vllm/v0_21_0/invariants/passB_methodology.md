# Pass B methodology - vLLM 0.21.0 class-hierarchy / type-tree walk

## Strategy

Strategy B enumerates the vLLM config + params surface by TYPE, not by call
path. Starting from the public config classes exported by `vllm.config`, every
reachable `@config` pydantic dataclass and every dataclass/msgspec params struct
was walked and, for each, every construction-time validity rule extracted. This
is the COMPLEMENT to Pass A (which walks only public-construction entry points);
the gain here is the declarative type-level constraints (Literal/enum membership,
inline `Field(ge/gt/le/lt)`) that pydantic enforces and that never appear as an
explicit `raise` on any call path.

- Engine source: `/tmp/vllm-0.21.0/vllm/`
  (`vllm/_version.py` line 21 confirms `__version__ = version = '0.21.0'`).
- Pure source analysis, no GPU, no model download.

## Type tree enumerated

The modular `vllm/config/` package was walked file by file. Every `@config`
class was located via `grep -n "^class "` / the `@config` decorator and read in
full:

- `attention.py`: AttentionConfig
- `cache.py`: CacheConfig (+ CacheDType / MambaDType / MambaCacheMode /
  PrefixCachingHashAlgo / KVOffloadingBackend Literals)
- `compilation.py`: CompilationConfig (+ CompilationMode IntEnum, CUDAGraphMode,
  PassConfig)
- `device.py`: DeviceConfig
- `ec_transfer.py`: ECTransferConfig (+ ECRole Literal)
- `kernel.py`: KernelConfig, IrOpPriorityConfig (+ MoEBackend Literal) - NEW at 0.21.0
- `kv_events.py`: KVEventsConfig
- `kv_transfer.py`: KVTransferConfig (+ KVRole Literal)
- `load.py`: LoadConfig (load_format is `str | LoadFormats`, lowercased; no
  membership Literal -> no membership invariant emitted)
- `lora.py`: LoRAConfig (+ MaxLoRARanks / LoRADType Literals)
- `mamba.py`: MambaConfig (+ MambaBackendEnum) - NEW at 0.21.0
- `model.py`: ModelConfig (+ TokenizerMode / ModelDType / LogprobsMode Literals)
- `multimodal.py`: MultiModalConfig (+ MMCacheType / MMEncoderTPMode /
  MMTensorIPC Literals, BaseDummyOptions / Video/Image/AudioDummyOptions)
- `observability.py`: ObservabilityConfig
- `offload.py`: OffloadConfig, UVAOffloadConfig, PrefetchOffloadConfig - NEW at 0.21.0
- `parallel.py`: ParallelConfig, EPLBConfig (+ DataParallelBackend /
  ExpertPlacementStrategy / DCPCommBackend / All2AllBackend /
  EPLBCommunicatorBackend / EPLBPolicyOption Literals)
- `pooler.py`: PoolerConfig (+ SequencePoolingType / TokenPoolingType Literals)
- `profiler.py`: ProfilerConfig (+ ProfilerKind Literal)
- `quantization.py`: OnlineQuantizationConfigArgs (+ OnlineQuantScheme Enum) - NEW at 0.21.0
- `reasoning.py`: ReasoningConfig (only validity rule is the tokenizer-gated
  token-id raise -> dormant, model-dir dependent)
- `scheduler.py`: SchedulerConfig (+ RunnerType / SchedulerPolicy Literals)
- `speculative.py`: SpeculativeConfig (+ SpeculativeMethod / RejectionSampleMethod
  / DraftSampleMethod Literals)
- `structured_outputs.py`: StructuredOutputsConfig (+ StructuredOutputsBackend Literal)
- `weight_transfer.py`: WeightTransferConfig

Params surface:

- `sampling_params.py`: SamplingParams (msgspec.Struct, `_verify_args` in
  `__post_init__`), StructuredOutputsParams, RepetitionDetectionParams,
  RequestOutputKind.
- `pooling_params.py`: PoolingParams (msgspec.Struct), LateInteractionParams.

## Extraction per class

For each class the following were harvested:

1. `Literal[...]` / `Enum`-typed fields -> membership rules
   (`predicate_kind: literal_in` / `strenum_in`).
2. Inline `Field(gt/ge/le/lt)` numeric constraints (`predicate_kind: range`).
3. `@field_validator` / `@model_validator` bodies -> every `raise` /
   `logger.warning`.
4. `__init__` / `__post_init__` checks (asserts, raises, hand-rolled
   `get_args(...)` membership).
5. before-validators parsing strings into enums via `Enum[value.upper()]` ->
   `strenum_in` / `backend_dispatch` (KeyError/ValueError membership with no
   coded raise).

`outcome` derived from severity: error/raise -> `invalid`; `logger.warning` ->
`warn`; silent value coercion the gate must not treat as a rejection ->
`normalise` (none emitted this version).

## kwargs replayability discipline

`native_type` is a DOTTED IMPORTABLE PATH so the gate can construct the owner
directly. Almost all live in `vllm.config` (verified against
`vllm/config/__init__.py` `__all__`); `OnlineQuantizationConfigArgs` is not
re-exported there and is cited at its module path
`vllm.config.quantization.OnlineQuantizationConfigArgs`.

`kwargs_positive` (should fire) / `kwargs_negative` (should not) were emitted
ONLY where the predicate is reachable at plain construction on a CPU-only host:

- Single-field declarative Literal / Field constraints on standalone configs.
- Standalone model/field validators (PoolerConfig, ProfilerConfig,
  StructuredOutputsConfig, OffloadConfig leaf fields, EPLBConfig,
  ObservabilityConfig, KV/EC transfer post_init conflicts, SamplingParams,
  StructuredOutputsParams, RepetitionDetectionParams, PoolingParams).
- SchedulerConfig validators ARE replayable because the two InitVars
  (`max_model_len`, `is_encoder_decoder`) are supplied explicitly in each pair.
- ParallelConfig literal fields ARE replayable because default `ParallelConfig()`
  (DP=1) falls into the env-var branch of `__post_init__` (parallel.py:811),
  which is CPU-safe (same observation as the 0.19.1 pass-B walk).

No kwargs were fabricated where a rule is:

- model-dir dependent (ModelConfig.* - `__post_init__` downloads the HF config),
- draft-model dependent (SpeculativeConfig.* - `__post_init__` resolves a draft
  model and needs `target_model_config`),
- platform/GPU-gated (`current_platform.is_cuda_alike()` for enable_eplb,
  ParallelConfig dcp/a2a entangled with distributed post_init,
  DeviceConfig.__post_init__ platform probe, MambaConfig stochastic-rounding),
- env-dependent (ObservabilityConfig otlp -> opentelemetry import),
- a nested-sub-config cross-field rule a flat kwargs pair cannot express
  (OffloadConfig prefetch.* validator),
- entangled with CompilationConfig.__post_init__ side effects.

These carry a `dormant_reason` and no kwargs rather than a fragile pair that
would make the gate reject an otherwise-valid entry.

## Result

- 108 total invariants, all `provenance: net_new` (no PoC ground_truth fold file
  exists for vllm v0_21_0; the gate folds it as net_new).
- 92 CPU-replayable (carry a kwargs_positive / kwargs_negative pair).
- 16 dormant (carry a dormant_reason, no kwargs).

### Strategy-B-specific gains (rules an entry-point walk structurally misses)

The bulk of the catalogue is declarative type-level membership / range
constraints with no coded raise: cache_dtype (+ the 0.21.0 turboquant_*/nvfp4
additions), prefix_caching_hash_algo, MaxLoRARanks integer Literal, scheduler
policy/runner_type, structured-outputs backend, kv/ec/weight-transfer backends,
attention/mamba backend enums (KeyError/ValueError membership), moe_backend,
offload backend, parallel data_parallel_backend / expert_placement_strategy /
dcp_comm_backend / all2all_backend / eplb communicator/policy, compilation
mode / compile_cache_save_format, multimodal cache-type / tp-mode / tensor-ipc /
encoder-attn-dtype, profiler kind, online-quant scheme enum.

Two dead-raise observations the type walk surfaces:

- `EPLBConfig.policy` is a single-member `Literal["default"]`, so the
  `_validate_eplb_config` "Async EPLB is only supported with the default policy"
  raise (parallel.py:101) is unreachable via direct construction.
- `ECTransferConfig.ec_role` IS Literal-typed, so its hand-rolled
  `not in get_args(ECRole)` raise (ec_transfer.py:82) is dead (Literal rejects
  first); only the connector-without-role conflict is reachable. Contrast
  `KVTransferConfig.kv_role`, which is a plain `str | None`, so its hand-rolled
  membership check IS live and emitted separately.

Also: `EPLBConfig.log_balancedness_interval` has `Field(gt=0)` AND a redundant
model_validator raise - the Field constraint makes the validator raise dead.

## Deltas vs vLLM 0.19.1 (re-derived, not copied)

- `cache_dtype` Literal grew (added turboquant_k8v4/4bit_nc/k3v4_nc/3bit_nc,
  int8_per_token_head, fp8_per_token_head, nvfp4).
- `CacheConfig.mamba_cache_mode` default flipped `all` -> `none`.
- `RejectionSampleMethod` value-set changed: `strict/probabilistic/synthetic`
  -> `standard/synthetic`; `DraftSampleMethod` (greedy/gumbel) is new.
- `TokenizerMode` gained `deepseek_v4`, `fastokens`.
- New config trees: `kernel.py` (KernelConfig, MoEBackend),
  `offload.py` (OffloadConfig + UVA/Prefetch sub-configs),
  `quantization.py` (OnlineQuantizationConfigArgs + OnlineQuantScheme),
  `mamba.py` (MambaConfig + MambaBackendEnum).
- New SamplingParams `_verify_args` checks: `min_tokens <= max_tokens`,
  logprobs/prompt_logprobs `>= 0 or -1`, stop_token_ids integers, stop no empty
  string, stop requires detokenize. `temperature` now raises
  `VLLMValidationError` (a ValueError subclass).
- New numeric Field constraints: `SchedulerConfig.stream_interval` (ge=1),
  `ObservabilityConfig.kv_cache_metrics_sample` (gt=0, le=1),
  `MultiModalConfig.mm_encoder_fp8_scale_save_margin` (gt=0).

## Blind spots (what an entry-point / call-graph walk should catch that B missed)

A type-tree walk under-covers rules that live in the execution path rather than
in a standalone config class:

1. ModelConfig declarative Literals (TokenizerMode, ModelDType, ConvertType,
   RunnerOption, ModelImpl, LogprobsMode) - real type constraints, but
   `__post_init__` loads the HF config, so only an entry-point fixture with a
   real (or stubbed) model dir can exercise them.
2. SpeculativeConfig Literals (method / rejection_sample_method /
   draft_sample_method) and `num_speculative_tokens` Field(gt=0) - entangled
   with the draft-model-resolving `__post_init__`.
3. CompilationConfig mode / compile_cache_save_format membership and the
   custom_ops 'none'/'all' assert - reachable only once the heavy
   `__post_init__` (inductor config, torch-version probes) is satisfied, i.e.
   via the VllmConfig fixture an entry-point walk builds.
4. Cross-config VllmConfig-level rules (`_validate_v2_model_runner`,
   `validate_block_size`, `validate_nvfp4_kv_cache_with_mla`,
   `validate_mamba_block_size`, observability sub-checks) wired in
   `VllmConfig.__post_init__` / `@model_validator` that no single leaf config
   owns.
5. Platform/distributed-gated rules (ParallelConfig enable_eplb CUDA check,
   dcp/a2a, MambaConfig stochastic rounding, DeviceConfig platform probe) that
   need a GPU/distributed fixture.
6. The opentelemetry-import-dependent otlp endpoint check (environment-gated).
