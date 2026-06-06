# Pass B methodology - vLLM 0.18.1 (class-hierarchy / type-tree walk)

## What this pass is

Pass B is the COMPLEMENT to Pass A (the public-construction / entry-point walk).
Where Pass A walks the call paths reachable from public construction, Pass B walks
the class / type tree of every config and params model in vLLM 0.18.1 and extracts
construction-time validity rules that an entry-point walk structurally misses -
above all the DECLARATIVE membership rules (pydantic `Literal[...]` fields and
`AttentionBackendEnum[...]` lookups) that are enforced by typing and never appear
as a coded `raise` on any call path.

A construction-time invariant = a rule that makes a config / params object VALID
or INVALID at construction: a `Field(ge/gt/le/lt)` constraint, a `Literal` /
enum-membership field, a `@field_validator` / `@model_validator` body, a
`__post_init__` / `_verify_args` check, or any reachable `raise` / `assert`.

## Source

- Engine source: `/tmp/vllm-0.18.1/vllm` (vLLM 0.18.1, one minor below 0.19.1).
- Confirmed the same modular `vllm/config/*.py` package layout as 0.19.1.

## Walk procedure

1. Enumerated every config dataclass in `vllm/config/*.py` (attention, cache,
   compilation, device, ec_transfer, kv_events, kv_transfer, lora, multimodal,
   observability, offload, parallel, profiler, scheduler, speculative,
   structured_outputs, weight_transfer) plus `vllm/sampling_params.py` and
   `vllm/pooling_params.py`.
2. For each class, by TYPE: read every field annotation for a `Literal` or enum
   type; every `Field(...)` for ge/gt/le/lt bounds; every `@field_validator` /
   `@model_validator` body; every `__post_init__` / `_verify_args` /
   `verify_max_model_len`; every `raise` / `assert` / `logger.warning`.
3. Descended into nested sub-configs (EPLBConfig under ParallelConfig,
   PrefetchOffloadConfig under OffloadConfig, the params dataclasses).
4. Verified every cited predicate against the actual 0.18.1 source line.

## CPU-replayable vs dormant

`kwargs_positive` (FIRES / invalid) and `kwargs_negative` (PASSES / valid) pairs
are emitted only where the type constructs cleanly on a CPU-only host with no
model download, no CUDA query, no distributed setup, and no platform probe.

Carried as `dormant_reason` (no kwargs) instead:
- `SpeculativeConfig.method` / `rejection_sample_method` - `__post_init__`
  (speculative.py:338) resolves a draft model / requires a target_model_config.
- `CompilationConfig.mode` / `compile_cache_save_format` - `__post_init__`
  (compilation.py:232) runs asserts / derivations not guaranteed CPU-safe alone.
- `ParallelConfig.dcp_comm_backend=='a2a'` cross-field - entangled with the
  distributed `__post_init__` (parallel.py:687).
- `LoRAConfig.lora_dtype` - `torch.dtype | LoRADType` union + arbitrary types
  allowed, so the Literal is not cleanly enforced.
- `ObservabilityConfig.otlp_traces_endpoint` - firing depends on whether the
  optional opentelemetry packages are installed (environment-dependent).
- `DeviceConfig.device` - `SkipValidation`-wrapped, so the Device Literal is not
  enforced at all (a dead-Literal observation).

## Strategy-B-specific gains over an entry-point walk

- Declarative `Literal` membership fields with NO coded raise on any path:
  cache_dtype, prefix_caching_hash_algo, mamba_cache_dtype, mamba_cache_mode,
  kv_offloading_backend, max_lora_rank (integer Literal), scheduler policy /
  runner_type, structured-outputs backend, kv_load_failure_policy, kv_events
  publisher, weight_transfer backend (a file with ZERO raise in its body),
  data_parallel_backend, expert_placement_strategy, dcp_comm_backend,
  EPLBConfig policy, profiler kind, mm cache type / encoder tp mode, offload
  backend, attention flash_attn_version.
- Type-driven enum membership via dict lookup (KeyError, not ValueError):
  `AttentionConfig.backend` and `MultiModalConfig.mm_encoder_attn_backend`
  (`AttentionBackendEnum[value.upper()]`).
- Dead-raise observations the type tree surfaces:
  - `EPLBConfig.policy` is single-member `Literal['default']`, so the
    async-policy `raise` (parallel.py:91) is unreachable via direct construction.
  - `KVTransferConfig.kv_role` and `ECTransferConfig.ec_role` are Literal-typed,
    so their in-post_init `get_args` membership raises are unreachable (the
    Literal rejects an out-of-set role first); only the connector-without-role
    conflict is the live half.
  - `DeviceConfig.device` Literal is bypassed by `SkipValidation`.
- Nested sub-config Field ranges the entry-point walk does not construct directly
  (EPLBConfig.num_redundant_experts, PrefetchOffloadConfig.offload_num_in_group).

## 0.18.1 vs 0.19.1

Structurally the same modular config package, one minor earlier. The single
field-set delta found in the walked surface is
`SpeculativeConfig.rejection_sample_method`: `Literal['strict','probabilistic']`
at 0.18.1 (two members) vs the three-member set (adds `'synthetic'`) at 0.19.1.
`SpeculativeMethod` and the `MTPModelTypes` / `EagleModelTypes` unions are
identical across the two minors. All cited line numbers are re-derived from
0.18.1 source and differ from 0.19.1 (e.g. cache.py is shorter, scheduler /
sampling_params line offsets shifted).

## Tally

69 invariants total: 55 net_new (type-tree catches), 14 folded from the PoC
SamplingParams / params surface. 61 carry CPU-replayable kwargs pairs; 8 are
dormant (model-dir / GPU / distributed / optional-package / SkipValidation
gated). The output YAML parses cleanly under `yaml.safe_load`.
