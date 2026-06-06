# Pass A methodology - vLLM 0.18.1 (entry-point / call-graph walk)

## Goal

Build a high-recall, runtime-gated GROUND-TRUTH denominator of construction-time
validation invariants for vLLM 0.18.1, reachable from the public user entry points.
An invariant = a predicate evaluated when a vLLM config or params object is constructed
that makes the value VALID or INVALID (a `__post_init__` check, a `_verify_args` /
`_verify_with_*` / model_validator / field_validator, a pydantic `Field(...)` constraint
or `Literal[...]`, an explicit `raise`, an `assert`, or a `logger.warning` /
`warning_once` reachable from construction).

## Source

- Tree: `/tmp/vllm-0.18.1/vllm`
- Version confirmed: `vllm/_version.py` -> `__version__ = '0.18.1'`,
  `__version_tuple__ = (0, 18, 1)`, commit `ga26e8dc7f`.
- 0.18.1 is one minor below the already-mined 0.19.1 and uses the same modular
  `vllm/config/*.py` package. Every entry was re-derived from 0.18.1 source against the
  actual line; nothing was copied from the 0.19.1 cell.

## Walk

1. `vllm.LLM(...)` -> `EngineArgs`/`AsyncEngineArgs.create_engine_config` -> the
   per-aspect configs in `vllm/config/*.py` and their `__post_init__` /
   `_verify_args` / model_validator / field_validator chains:
   ModelConfig, CacheConfig, ParallelConfig (+ nested EPLBConfig), SchedulerConfig,
   LoRAConfig, SpeculativeConfig, StructuredOutputsConfig, CompilationConfig,
   ObservabilityConfig, OffloadConfig, KVTransferConfig, ECTransferConfig, PoolerConfig,
   VllmConfig.
2. `vllm.SamplingParams` -> `__post_init__` -> `_verify_args` /
   `_verify_greedy_sampling`; plus the nested `StructuredOutputsParams` and
   `RepetitionDetectionParams` `__post_init__` validators.
3. Enumerated every `__post_init__`, `_verify*`, `def validate*`, `@field_validator`,
   `@model_validator`, `raise`, `assert`, `logger.warning`/`warning_once`, `Literal[`, and
   `Field(ge/gt/le/lt)` across config + params, and read each in context.

## PoC fold

No PoC GT file exists at
`research/mining-substrate-trial/findings/ground_truth/vllm/v0_18_1/invariants_ground_truth.yaml`,
so every invariant is `provenance: net_new`.

## Replayability classification

The downstream gate replays each invariant in a real vLLM 0.18.1 container (CPU-only,
no GPU): construct with `kwargs_positive` (must FIRE) and `kwargs_negative` (must PASS);
kept as GT only if it behaves as declared. Verified canary:
`SamplingParams(temperature=-1.0)` raises `VLLMValidationError` CPU-only.

- CPU-replayable (56): standalone params/config dataclasses that construct without a
  model dir - SamplingParams, StructuredOutputsParams, RepetitionDetectionParams,
  StructuredOutputsConfig, CacheConfig, LoRAConfig, EPLBConfig, PoolerConfig,
  ParallelConfig (`_validate_parallel_config` / `__post_init__` / `_verify_args` raises
  that do not require a distributed world), ObservabilityConfig, KVTransferConfig,
  ECTransferConfig, CompilationConfig.
- dormant (20): need something a CPU-only source host cannot provide -
  - ModelConfig.* : `__post_init__` downloads/parses a real HF config before the
    after-validators and `_verify_*` helpers run.
  - SchedulerConfig cross-field checks : live in `verify_max_model_len(max_model_len)`,
    reached via `__post_init__` whose `max_model_len` and `is_encoder_decoder` are
    InitVars supplied by `create_engine_config` (or `SchedulerConfig.default_factory`);
    flagged dormant but the kwargs include the InitVars so the gate MAY promote them.
  - SpeculativeConfig.* : `__post_init__` requires a real `target_model_config` (and
    often a draft model dir) before `_verify_args` runs.
  - VllmConfig.* : `__post_init__` needs a fully assembled config including a real
    ModelConfig.
  - ObservabilityConfig otlp endpoint : outcome depends on whether OpenTelemetry is
    installed in the replay container.
  - OffloadConfig num_in_group : predicate reads nested `self.prefetch.*`, not settable
    via flat OffloadConfig kwargs.

Several ParallelConfig entries carry probe notes where construction order (e.g.
`__post_init__` GPU-count checks, or a pydantic Literal firing before a normalisation
branch) could pre-empt the declared predicate; these are flagged for the gate to confirm.

## 0.18.1 vs 0.19.1

The sampling / structured-output / repetition params and the standalone config
dataclasses are equivalent in validation logic to 0.19.1 (line numbers shift only).
The notable 0.18.1 DIFFERENCE: `SamplingParams._verify_args` has NO `max_n` /
`n_gt_max_n_sequences` ceiling (added after 0.18.1). Ordering quirks preserved from the
source: the `top_k < -1` range check runs before the `top_k` int-type check, and the
`temperature` clamp-to-min warning runs in `__post_init__` before `_verify_args`.

## Output

- `passA_entrypoint.yaml` : 76 invariants (76 net_new), 56 CPU-replayable, 20 dormant;
  outcomes 70 invalid / 3 warn / 3 normalise. Loads cleanly with `yaml.safe_load`.

## Discipline

GT contributor independent of the mechanical miner. Every predicate verified against the
actual 0.18.1 source line cited. No fabrication; no cross-version copying without
re-derivation. ASCII only, no em-dashes.
