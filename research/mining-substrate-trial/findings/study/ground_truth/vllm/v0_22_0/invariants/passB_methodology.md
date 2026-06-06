# Pass B methodology - vLLM v0.22.0 class-hierarchy / type-tree walk

## Strategy

Strategy B (the complement to Pass A's entry-point walk) enumerates the vLLM
config surface by TYPE, not by call path. Starting from the public config
classes and the standalone params types, every reachable pydantic `@config`
dataclass and plain dataclass was walked and, for each, every validity rule
extracted and read in source context.

- Engine source: `/tmp/vllm-0.22.0/vllm/` (`_version.py` confirms
  `__version__ = "0.22.0"`).
- Pure source analysis. The host vLLM install is incomplete (missing `cbor2`
  and carrying a torch whose `_inductor` API does not match this vLLM build),
  so full empirical replay is not possible here; that is by design - the gate
  replays each pair inside a real vLLM 0.22.0 CPU container. The handful of
  configs that DID import cleanly on the host (CacheConfig, AttentionConfig,
  DynamicShapesConfig) were exercised and behaved exactly as the source reading
  predicted, anchoring confidence in the pure-source derivations.

## Type tree enumerated

Config package `vllm/config/` (modular). Every class located via
`grep -nE "^class "` across `config/*.py` and read:

- CacheConfig, AttentionConfig, LoRAConfig, SchedulerConfig,
  StructuredOutputsConfig, KVTransferConfig, ECTransferConfig, KVEventsConfig,
  WeightTransferConfig, MultiModalConfig (+ BaseDummyOptions /
  Video/Image/AudioDummyOptions), OffloadConfig (+ UVAOffloadConfig /
  PrefetchOffloadConfig), ParallelConfig, EPLBConfig, ProfilerConfig,
  ObservabilityConfig, KernelConfig (+ IrOpPriorityConfig), MambaConfig,
  PoolerConfig, QuantizationConfigArgs (+ QuantSpec), CompilationConfig
  (+ DynamicShapesConfig / PassConfig), ModelConfig, SpeculativeConfig,
  LoadConfig, DeviceConfig, ReasoningConfig, SpeechToTextConfig.
- Params: `vllm.SamplingParams`, `StructuredOutputsParams`,
  `RepetitionDetectionParams`, `vllm.PoolingParams`.

## Extraction per class

For each class the following were harvested:

1. `Literal[...]` / str-Enum-typed fields -> membership rules
   (`predicate_kind: literal_in` / `strenum_in`).
2. Inline `Field(gt/ge/le/lt)` numeric constraints (`predicate_kind: range`).
3. `@field_validator` / `@model_validator` bodies -> every `raise` /
   `logger.warning` / propagated parse exception.
4. `__post_init__` checks (SchedulerConfig, KV/ECTransferConfig, PoolerConfig,
   KVEventsConfig, params types).
5. before-validators that coerce a string into an enum via `Enum[value.upper()]`
   (AttentionConfig backend / mla_prefill_backend, MambaConfig backend) ->
   `strenum_in`.
6. custom-schema allowlists (quantization `_coerce_quant_key`) ->
   `allowlist_constant`.

`outcome` derived from severity: error/raise -> `invalid`; warn ->
`valid_with_warning`. (No warn-only invariants survived into this catalogue as
standalone entries; warn paths were noted in-line.) `native_type` is a dotted
importable path so the gate can construct it directly.

## kwargs replayability discipline

`kwargs_positive` (should FIRE) / `kwargs_negative` (should PASS) emitted ONLY
where the predicate is reachable at plain construction on a CPU host:
single-field declarative Literal/Field constraints and standalone-model field /
model validators / post_inits. For rules that are GPU-gated, model-dir-gated,
distributed/post-init-entangled, env-gated, filesystem-stateful, or fire only at
a later lifecycle method, NO kwargs were fabricated - a `dormant_reason` records
why and the gate skips replay rather than rejecting an otherwise-valid entry.

## Result

- 99 total invariants, all `provenance: net_new`. No PoC ground truth exists for
  v0_22_0 (no `.../ground_truth/vllm/v0_22_0/invariants_ground_truth.yaml` and no
  `invariants_ground_truth.yaml` under the study tree), so nothing was folded;
  every entry was re-derived from THIS version's source.
- 85 CPU-replayable (carry a kwargs pair); 14 dormant (carry a dormant_reason).
- predicate_kind mix: literal_in 35, range 26, presence_conflict 25,
  strenum_in 6, allowlist_constant 3, type_is 2, backend_dispatch 1,
  mutual_exclusion_soft 1.

## Strategy-B-specific gain over an entry-point walk

The declarative type-level constraints that carry NO explicit `raise` on any
call path (pydantic rejects them on type/Field validation alone), which a
call-graph walk structurally misses:

- `KernelConfig.moe_backend` (14-member) and `KernelConfig.linear_backend`
  (15-member) Literals - an entire NEW config file at 0.22.0.
- `MambaConfig.backend` (NEW file) MambaBackendEnum membership.
- `ParallelConfig.all2all_backend` (NEW 10-member Literal),
  `EPLBConfig.communicator` (NEW Literal).
- `MultiModalConfig.mm_tensor_ipc` and `mm_encoder_attn_dtype` (NEW Literals),
  `AttentionConfig.mla_prefill_backend` (NEW enum-membership validator).
- `DynamicShapesConfig.type` (NEW str-enum), the only compilation-tree membership
  that is standalone-CPU-constructable (CompilationConfig itself is dormant).
- `QuantSpec.weight` / `QuantizationConfigArgs.linear` allowlist (NEW file),
  enforced by a custom pydantic plain-validator, not a declarative raise.

## Deltas vs the 0.19.1 vLLM Pass-B catalogue (re-derived, not copied)

- `CacheConfig.cache_dtype` Literal grew to 15 members (turboquant_* family,
  int8/fp8_per_token_head, nvfp4); `mamba_cache_dtype` gained `bfloat16`.
- `WeightTransferConfig.backend` is NO LONGER a Literal at 0.22.0 - it is a plain
  `str` validated at engine-creation against WeightTransferEngineFactory. The
  0.19.1 backend-Literal invariant is gone; recorded as a dormant note for
  completeness.
- `SchedulerConfig` gained inline `Field(ge=1)` on max_num_batched_tokens /
  max_num_seqs / max_num_partial_prefills / max_long_partial_prefills /
  stream_interval; the verify-time `batched >= max_model_len when not chunked`
  raise is now captured.
- `OffloadConfig` was restructured into `UVAOffloadConfig` /
  `PrefetchOffloadConfig` sub-configs with their own Field(ge) constraints and a
  cross-field model_validator.
- `SpeculativeConfig.rejection_sample_method` Literal changed from
  `[strict, probabilistic, synthetic]` to `[standard, synthetic]`; `method`
  Literal member set grew; new `draft_sample_method` Literal.
- `ModelConfig.tokenizer_mode` gained `deepseek_v4`.
- New params checks at 0.22.0: SamplingParams `min_tokens <= max_tokens`,
  `logprobs`/`prompt_logprobs` non-negative-or-(-1), `stop` no-empty-string,
  `stop` requires detokenize, `bad_words` no-empty-string.
- New PoolerConfig affine-calibration checks (logit_sigma != 0, logit_bias /
  logit_mean and logit_scale / logit_sigma conflicts).

## Dead-raise observations (Strategy-B value-set consequences)

- `KVTransferConfig.kv_role` and `ECTransferConfig.ec_role` are Literal-typed, so
  the manual `not in get_args(...)` raises in their `__post_init__` are DEAD via
  direct construction (the Literal rejects first). Only the
  connector-without-role conflict half is reachable.
- `EPLBConfig.policy` is a single-member `Literal["default"]`, so the
  `Async EPLB only supported with the default policy` raise is unreachable via
  direct construction; likewise the `log_balancedness_interval <= 0` raise is
  dead behind the `Field(gt=0)` on that field.

## Blind spots (what an entry-point / call-graph walk should catch that B misses)

1. Rules wired into `VllmConfig.__post_init__` / `EngineArgs.create_engine_config`
   that no single config class owns (cross-config consistency, derived defaults).
2. `ModelConfig` / `SpeculativeConfig` / `CompilationConfig` / `ParallelConfig`
   declarative Literals whose `__post_init__` does model download, draft-model
   resolution, torch-compile derivation, or distributed/env setup - recorded here
   as dormant, but their firing on the real construction path is a Pass-A job.
3. Registry-validated string fields (LoadConfig.load_format,
   WeightTransferConfig.backend, distributed_executor_backend) whose membership
   is enforced only at engine-creation lookup, not at config construction.
4. Runtime guards in worker/executor code reachable only by following the call
   graph from the public `LLM(...)` / `AsyncLLM` entry.
