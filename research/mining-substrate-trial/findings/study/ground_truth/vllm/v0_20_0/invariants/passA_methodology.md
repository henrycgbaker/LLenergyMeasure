# Pass A - entry-point / call-graph walk methodology (vLLM 0.20.0)

Engine source: `/tmp/vllm-0.20.0/vllm/` (the `vllm` package tree; VERSION pinned
to 0.20.0 by the task. The bundled `version.py` reads the version from a
vcs-generated `_version.py` that is not shipped in this snapshot, so the version
marker is the task pin plus the source-structure fingerprint - see "Version
fingerprint" below).

Output: `passA_entrypoint.yaml`. This pass is the entry-point / call-graph half
of a two-pass bake-off; a sibling pass does a class-hierarchy walk. Goal:
maximise recall of construction-time validation invariants reachable from
public, user-facing entry points, at a cost the gate can afford.

## Traversal (what I walked)

Starting roots (public surface a benchmark harness actually constructs):

1. `vllm.LLM(...)` -> `EngineArgs` / `AsyncEngineArgs` (vllm/engine/arg_utils.py)
   -> `create_engine_config` -> the per-aspect dataclasses in `vllm/config/*.py`:
   `ModelConfig`, `CacheConfig`, `ParallelConfig` (+ nested `EPLBConfig`),
   `SchedulerConfig`, `DeviceConfig`, `LoadConfig`, `LoRAConfig`,
   `SpeculativeConfig`, `StructuredOutputsConfig`, `CompilationConfig`,
   `MultiModalConfig`, `MambaConfig`, `KVTransferConfig`, `ECTransferConfig`,
   `ObservabilityConfig`, `OffloadConfig`, `PoolerConfig`, and their
   `__post_init__` / `_verify_*` / `@field_validator` / `@model_validator` /
   `Field(...)` constraint chains, plus the `VllmConfig.__post_init__`
   top-level cross-config aggregation and the platform
   `check_and_update_config` hooks (`vllm/platforms/cpu.py`).
2. `vllm.SamplingParams(...)` -> `__post_init__` -> `_verify_args` /
   `_verify_greedy_sampling`; nested `StructuredOutputsParams.__post_init__` and
   `RepetitionDetectionParams.__post_init__`.
3. Enumerated every `__post_init__`, `_verify*`, `validate*`, `@field_validator`,
   `@model_validator`, `raise`, `assert`, `logger.warning`/`warning_once`,
   `warnings.warn`, `Literal[`, and `Field(ge/gt/le/lt/...)` across
   `config/*.py`, `sampling_params.py`, and the reachable platform hook, then
   READ each in source context to classify predicate + outcome + replayability.

## Method

- grep for the validator/raise/literal/field markers above in each file, then
  read each hit in context. Classified each into error / warning / normalise /
  dormant and recorded `native_type` (a dotted importable path the gate
  constructs), `native_field`, `predicate_kind`, `predicate_value`,
  `message_template`, and a `citation{file,line,qualname}` resolved against the
  v0_20_0 source.
- For declarative pydantic `Field(...)` bounds and `Literal[...]` fields I cited
  the field DEFINITION line; for imperative checks I cited the predicate / raise
  line and named the enclosing validator qualname.
- `kwargs_positive` is chosen to fire the rule in isolation; `kwargs_negative`
  is a minimal valid construction. Where two checks share a field I chose values
  that isolate the target check (e.g. `top_k=1.0` to reach the isinstance check
  past the `< -1` guard).

## Version fingerprint (how this differs from 0.19.1)

The 0.20.0 source carries structure absent from the 0.19.1 cell I skimmed for
field shapes:

- `CacheConfig.gpu_memory_utilization` default raised to **0.92** (was 0.9).
- `ParallelConfig` gained the **numa-binding** surface
  (`numa_bind_nodes`/`numa_bind_cpus` field-validators + the `numa_bind=True`
  requirement) and the **decode-context-parallel** checks
  (`tp % dcp == 0`, `dcp_comm_backend='a2a'` requires `dcp>1`).
- `OffloadConfig` was **restructured** into nested `uva` (UVAOffloadConfig) and
  `prefetch` (PrefetchOffloadConfig) sub-configs, adding the
  `offload_prefetch_step >= 1` check.
- New **`MambaConfig`** (string-coercing backend enum + CUDA-only stochastic
  rounding) and new `ObservabilityConfig.kv_cache_metrics_sample` Field(gt=0,le=1).
- `PoolerConfig` gained `logit_bias->logit_mean` / `logit_scale->logit_sigma`
  deprecation-conflict raises and the second `tok_pooling_type` exclusivity raise.
- `CompilationConfig` gained an explicit `compile_cache_save_format` validator
  and the VLLM_COMPILE-backend / encoder-cudagraph checks.

SamplingParams is structurally unchanged from 0.19.1 (same `_verify_args`
predicate set, same VLLMValidationError typing); line numbers drift by a few.

## Provenance

No PoC ground-truth file exists for v0_20_0
(`research/mining-substrate-trial/findings/ground_truth/vllm/v0_20_0/`
`invariants_ground_truth.yaml` is absent), so **every entry is `net_new`** and
was derived directly from the v0_20_0 source. Nothing was copied from the
v0_19_1 cell without re-reading the corresponding v0_20_0 line.

## Coverage

- Full `SamplingParams` + `StructuredOutputsParams` + `RepetitionDetectionParams`
  predicate set (CPU-replayable).
- All self-contained nested-config validators reachable from construction:
  `CacheConfig`, `LoRAConfig`, `SchedulerConfig` field bounds, `EPLBConfig`,
  the CPU-replayable `ParallelConfig` model/field-validators,
  `KVTransferConfig`, `ECTransferConfig`, `StructuredOutputsConfig`,
  `ObservabilityConfig`, `OffloadConfig`, the CPU-replayable `CompilationConfig`
  subset, `MultiModalConfig`, `MambaConfig`, `PoolerConfig`, `DeviceConfig`.
- The dormant `ModelConfig` / `VllmConfig` / platform cross-config checks
  (representative set), flagged with `dormant_reason`.

## Runtime replayability for the downstream gate

CPU-replayable (pure pydantic / dataclass validation, no CUDA, no model dir):
all `SamplingParams`/`StructuredOutputsParams`/`RepetitionDetectionParams`
entries; `CacheConfig`, `LoRAConfig` (except the dual-stream env case),
`SchedulerConfig` field bounds, `EPLBConfig`, the `ParallelConfig` numa /
data-parallel / dcp / num-redundant-experts / elastic-ep / nsight model- and
field-validators, `KVTransferConfig`, `ECTransferConfig`,
`StructuredOutputsConfig`, `ObservabilityConfig` field bounds + tracing combo,
`OffloadConfig`, the `CompilationConfig` custom-ops / mode / cache-format /
backend checks, `MultiModalConfig`, `MambaConfig` backend enum, `PoolerConfig`,
`DeviceConfig` device Literal.

Dormant (need a GPU / specific platform / model dir / resolved config and cannot
be replayed source-only on a CPU host), flagged with `dormant_reason`:

- All `ModelConfig` checks (its `__post_init__` loads the HF config -> needs a
  model dir): sleep-mode, override-attention-dtype, quantization, dtype,
  max_model_len, head-divisibility, pipeline-parallel support.
- All `VllmConfig.__post_init__` cross-config checks (need a fully-resolved
  ModelConfig): torch_shm/spawn, enforce-eager, cudagraph-mode, sequence-parallel.
- All `SpeculativeConfig` checks: `__post_init__` resolves a draft `ModelConfig`
  (and references `target_model_config`) for every method including ngram, so
  bare-kwargs replay is not possible - intentionally NOT emitted as live
  kwargs entries (see Blind spots).
- Platform `CpuPlatform.check_and_update_config` (takes a built VllmConfig).
- `SchedulerConfig.verify_max_model_len` cross-checks (max_model_len arrives via
  the `__post_init__` InitVar from `create_engine_config`, not as a field).
- Platform-gated raises (`ParallelConfig` EPLB CUDA guard, `MambaConfig`
  stochastic rounding, `DeviceConfig` auto-infer, `LoRAConfig` dual-stream)
  whose outcome depends on the gate host's platform/env.
- `ObservabilityConfig.otlp_traces_endpoint` (depends on whether opentelemetry
  is importable in the gate container).

## Blind spots (what a class-hierarchy walk should catch that I did not)

1. **SpeculativeConfig internals.** Its `__post_init__` requires a resolved
   `target_model_config` (a ModelConfig), so I could not exercise the rich
   draft-TP / vocab-size / num-speculative-tokens / synthetic-acceptance-rate /
   suffix-decoding raises with bare kwargs. They are real construction-time
   predicates but uniformly dormant in this source-only environment; I
   deliberately did not emit speculative entries with live kwargs to avoid
   false CPU-replayable claims. A hierarchy walk with a stub ModelConfig could
   surface them.
2. **Validators on base/sibling classes never reached from the public ctor**
   (e.g. pooling-only `PoolerConfig` paths beyond the exclusivity raises,
   nested DummyOptions Field bounds in MultiModalConfig).
3. **`@field_validator` defined on a parent and overridden in a child** - the
   grep-by-def approach can miss override resolution; the MRO-ordered hierarchy
   walk is the right tool.
4. **CompilationConfig deep tree** - I captured the CPU-replayable
   `__post_init__` raises but did not exhaustively walk the inductor / cudagraph
   capture-size assertions that depend on torch internals or a resolved config.
5. **Enum classes reachable only via `from_dict(string)` paths** on classes the
   public ctor does not route to may be missed.
