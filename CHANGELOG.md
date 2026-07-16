# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (0.x pre-release series).
Minor version bumps (`0.x.0`) mark milestone completions. Breaking changes can occur between any 0.x release.

## [Unreleased]

## [v0.11.0] - 2026-07-16

### Added

- `ExperimentResult.engine_build_cache_hit`: whether the tensorrt trt-backend engine build
  was served from the on-disk build cache (`true`) or compiled fresh (`false`); `null` when
  the cache is not in play (pytorch backend, other engines, an `engine_path` override, or the
  cache disabled). Detected from TRT-LLM's own `llm_build_stats.engine_dir`, which is
  populated only on the cache-reuse path (the sibling `cache_hitted` flag is unusable - it is
  `True` on both the reuse and fresh-build paths). Annotates `model_load_time_sec` (a hit
  skips the compile). Additive optional field; result schema_version stays 4.0. ([#804])
- `llem doctor` now reports the TensorRT-LLM engine build cache: location, engine-entry count,
  total size, and the manual clean command. Informational only (never affects the exit code) -
  the cache lifecycle is manual and visible, and llem never auto-evicts. ([#804])
- `ExperimentResult.model_load_time_sec`: wall-clock seconds spent in `engine.load_model()`
  (model load plus any engine build/compile performed there - the tensorrt trt backend's
  TRT engine build, vLLM torch.compile / CUDA-graph capture). Captured by the harness
  around the load call, so all three engines gain it uniformly. Non-energy run metadata:
  the phase completes before the NVML energy window opens. Additive optional field;
  result schema_version stays 4.0. ([#803])
- `LLEM_DOCKER_GPUS`: docker `--gpus` request for llem-launched experiment and baseline
  containers (default `all`, the historical behaviour). On shared multi-GPU hosts, pin llem
  to free devices (e.g. `device=2`); restricting at the docker level keeps CUDA and NVML
  indices consistent inside the container. ([#803])
- TensorRT per-request latency metrics under `latency_profiling`: the plugin now sets
  `SamplingParams(return_perf_metrics=True)` when latency profiling is enabled and extracts
  per-request TTFT / E2E / average TPOT from `RequestOutput.metrics_dict` (the TRT-LLM 1.x
  surface, live-verified on both backend legs at 1.2.1; the vLLM-shaped
  `RequestOutput.metrics` namespace does not exist there). Capture mode is
  `per_request_batch`. Default runs are unchanged: the flag is never set without
  latency profiling, and the `latency_profiling_unsupported` degradation warning now
  fires only when profiling was requested but the engine returned no metrics. ([#803])

- TensorRT-LLM schema discovery now mines BOTH backend args classes and unions them into
  one discovered engine-params surface (63 -> 95 fields): `TrtLlmArgs` (the `trt` backend)
  and `TorchLlmArgs` (the `pytorch` backend). Each field carries a `backends` applicability
  list (`[pytorch, trt]`, `[trt]`, or `[pytorch]`) recording which backend's args class
  carries it. Codegen keeps a single `EngineParams`; the metadata is descriptive (surfaced
  as a Backends column in the schema reference), and cross-backend applicability is enforced
  by loud validation rules, never silent dropping. Absence of the key means "all backends",
  so the single-class vllm/transformers schemas are byte-unchanged. ([#801])
- Exposure-time field narrowing via a new optional `exposure_overrides` block in
  `curated.yaml`: narrows a generated config field (`enum` -> `Literal`, `default`) without
  touching the mined schema. Used to expose tensorrt `backend` as `Literal["pytorch", "trt"]`
  defaulting to `"pytorch"` while the mined type stays `str`. ([#801])
- Backend-applicability rules in the tensorrt corpus (27 -> 29): `fast_build` and
  `quant_config` require `backend='trt'` (both exist only on `TrtLlmArgs`; the pytorch
  backend's `TorchLlmArgs` rejects them under `extra='forbid'`). Live-verified by direct
  construction at 1.2.1; enforced at the config-expansion grain, complementing the plugin's
  construction-grain `ConfigError` guards (defense in depth at both grains). ([#801])
- The deterministic cross-field extractor's target table now walks the `TorchLlmArgs`
  validator tree (`stream_interval`, `batch_wait_*`, `ray_*` preconditions,
  `speculative_config`), joining the pytorch-backend validators to the standing candidate
  surface. ([#801])
- Standing plugin-kwarg lint: `scripts/check_plugin_kwargs.py` cross-checks the literal
  constructor-kwarg names each engine plugin's translation layer hand-types against that
  engine's discovered schema at the current pin, with a rationale-carrying allowlist for
  genuine off-surface kwargs (transformers `from_pretrained` open `**kwargs`). Catches
  upstream kwarg renames the mined schema already knows about but the hand-written glue
  code missed (the `quantization` -> `quant_config` case). Wired as
  `make check-plugin-kwargs` and a `plugin-kwarg-check` job in ci.yml. ([#800])

### Changed

- The upstream-image engine-version handshake probe (a ~60s cold `docker run` per engine
  per study) is now cached on disk keyed by the image content digest, under
  `platformdirs.user_cache_dir("llem")/image-probe/`. A warm image resolves from the cache
  (a `docker image inspect`) instead of re-probing, so the per-study cost the F2 plan flagged
  drops to near zero. The in-process memo remains the first tier; a corrupt, unreadable, or
  unwritable cache degrades to a fresh probe and never crashes. Entries never go stale: a
  rebuilt or re-pulled image gets a new digest and a fresh probe, so no TTL is needed. ([#805])
- TensorRT-LLM backends are now selected by constructor class, not a kwarg: `backend='trt'`
  resolves `tensorrt_llm._tensorrt_engine.LLM` and `pytorch`/unset resolves `tensorrt_llm.LLM`,
  validated live at 1.2.1 (the base `LLM` rejects `backend='trt'` at model load). `backend` is
  no longer forwarded as a kwarg; an unsupported value raises `ConfigError` naming
  `{pytorch, trt}`. ([#797])
- TensorRT-LLM pin advanced 1.0.0 -> 1.2.1 through the full bump pipeline: schema re-mined
  byte-stably, the 20-field curation carried forward with no discovery debt, typed config
  regenerated (`max_num_tokens` default now 8192), and the shipped rules corpus grown 15 -> 29
  (construction-confirmed additions plus human-signed residue). ([#792])

### Fixed

- Memory metrics are no longer a silent 0.0 for out-of-process engines. vLLM V1 runs its
  model in the EngineCore child process and TensorRT-LLM in its executor process, so torch's
  per-process allocator in the driver process saw nothing and `extended_metrics.memory`
  `peak_memory_mb` / `model_memory_mb` came back as exactly 0.0 on every vLLM and TRT-LLM
  experiment (real values only for in-process Transformers). Both capture points now fall
  back to NVML device-used memory when the torch reading is implausible (`== 0.0`): the
  harness model-memory baseline, and the plugin peak-memory read (vLLM's NVML fallback was
  previously gated on the already-broken `peak > 0` torch value so it never fired; TRT-LLM
  had no fallback at all). Transformers keeps its authoritative torch reading and never
  consults NVML. The cascade fields that were nulled by the zero (`tokens_per_gb_vram`,
  `model_memory_utilisation`, `kv_cache_memory_ratio`) now populate. NVML `used` is a
  whole-device reading (it includes this process's CUDA context and any co-tenants), so the
  absolute peak/model figures are an upper bound; the derived `inference_memory_mb` delta
  cancels the shared context term and stays meaningful. Under `LLEM_DOCKER_GPUS` pinning the
  container sees only the pinned device(s), so NVML index resolution matches the experiment's
  own GPUs. The raw memory fields are now nullable and any residual 0.0 is coerced to null at
  the domain boundary - the fields are always either a real measurement or null, never a
  silently-wrong zero. ([#816])
- llem now forwards every `NCCL_*` host environment variable into both the experiment
  container (`infra.docker_runner`) and the baseline container (`study.baseline_container`),
  so NCCL tuning/workaround settings set on the host reach the engine process, which runs
  inside the container. The motivating case is `NCCL_P2P_DISABLE=1` on PCIe multi-GPU hosts
  whose topology lacks functional peer-to-peer (P2P, e.g. an inter-GPU link reported as `SYS`
  in `nvidia-smi topo -m`, often because ACS is enabled): without it, tensor-parallel runs
  hang at the first NCCL collective. Vars are emitted as explicit `-e KEY=VALUE` args
  (matching the existing `LLEM_*` forwarding idiom) in sorted key order for a deterministic
  command. ([#814])
- Study GPU advisory locks are now named by the PHYSICAL device a study occupies, not the
  in-container logical index. Lock names came from `_resolve_gpu_indices` (which enumerates
  from 0), but the physical device is chosen at the docker level by `LLEM_DOCKER_GPUS`
  (`--gpus device=N`), which the lock logic never saw. Under pinning the container always sees
  its GPU as logical 0, so a study pinned to `device=2` locked `gpu-0.lock`, and two studies
  pinned to DIFFERENT physical GPUs both contended on `gpu-0.lock` and spuriously serialised.
  A new `env_config.pinned_gpu_lock_ids()` parses the docker selector into physical lock ids
  (`device=2` -> `["2"]`, `device=2,3` -> `["2","3"]`, a `GPU-<uuid>` selector used verbatim
  as a stable per-device id); `all` / unset / unrecognised shapes fall back to the logical
  indices (logical == physical when every GPU is visible). Lock naming only - measurement-side
  index resolution is unchanged. ([#807])
- TensorRT-LLM build cache silently died across containers unless
  `LLEM_TRT_BUILD_CACHE_PATH` was set by hand: the docker runner bind-mounts the host
  `~/.cache/trt-llm` at `/root/.cache/trt-llm`, but TRT-LLM's own default cache root
  (`/tmp/.cache/tensorrt_llm/llmapi/`) is unmounted and evaporates with each ephemeral
  container. The runner now defaults `LLEM_TRT_BUILD_CACHE_PATH` to the mount target (via the
  mount's `extra_env`, mirroring the `HF_HOME` pattern), so compiled engines persist out of
  the box; a host-set value still wins (forwarded last, docker `-e` is last-wins). Verified
  live: upstream `BuildCacheConfig` keying is sound - identical config hits across containers,
  and tensor-parallel / max-shape / quantisation / dtype changes each key to a distinct
  engine hash. ([#804])
- The TRT-LLM pre-quantised-checkpoint preflight message now names what was live-verified at
  1.2.1: AutoAWQ / AutoGPTQ community-format HF checkpoints load on neither backend (the trt
  backend raises `NotImplementedError` on the `quant_method`; the pytorch backend rejects the
  weight layout), and TRT-LLM's supported pre-quantised path is its own ModelOpt export
  (`hf_quant_config.json`). The rejection stands for both backends; the `engine_path` escape
  hatch is unchanged. ([#804])
- vLLM containerized runs at 0.19.1 crashed at engine start with "Cannot re-initialize CUDA
  in forked subprocess": the harness touches torch.cuda (hardware preflight) before the
  plugin constructs the engine, and vLLM's EngineCore worker forks by default. The plugin now
  sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` (setdefault, so a user override wins) - vLLM's
  own prescription for embedding contexts. Found by the first live containerized vllm run at
  this pin. ([#803])
- Container deps-cache stamp is now keyed per engine: the pyproject-verified stamp was shared
  across all images of one python minor, so whichever engine dispatched first suppressed the
  missing-deps probe for the others (the TRT-LLM NGC image, which ships every llem dep,
  masked the vllm image's missing pyarrow -> parquet write crash). ([#803])
- Transformers image seed and dev cache hygiene: `make docker-seed-transformers` no longer runs
  a harmful second push that wrote plain `transformers:latest` / `transformers:v<version>`
  images and a clobber-prone `mode=max` cache to `:latest`. The `-buildcache` ref is now the
  single cache manifest; the canonical tags are written only by the promotion and release
  tag-copies. `docker-compose.yml`'s `cache_from` now targets that per-version buildcache ref
  (the engine pin is exported as `TRANSFORMERS_VERSION` by the `make docker-build` wrapper and
  also passed as the build arg, so local builds install the pinned version and actually reuse
  the seeded FA3 layers), and the dead `LLEM_PKG_VERSION` / `LLEM_EXPCONF_SCHEMA_FINGERPRINT`
  build-args and Makefile exports are removed (the Dockerfile stopped consuming them in 0.10.0).
  The Dockerfile pins its `ghcr.io/astral-sh/uv` base (was `:latest`, so every uv
  release invalidated every subsequent builder layer) and records that `MAX_JOBS` must never
  be overridden via build-arg in CI, since it participates in the FA3 layer's cache key. ([#799])
- Release image publish is now a registry-side tag-copy of the promoted seed digest, not a
  hosted rebuild. `docker-publish.yml` points `transformers:<version>` at the already-promoted
  `transformers:transformers-<pin>` via `docker buildx imagetools create`, eliminating the
  flash-attention FA3 compile that OOM'd the hosted runner; the workflow aborts loudly when the
  seed/promotion source is missing. ([#798])
- TensorRT-LLM quantisation is passed as the native `quant_config` kwarg, not the
  long-removed `quantization` name (`TrtLlmArgs` is `extra='forbid'` at 1.2.1, so the old
  name crashed any quantised run at construction). ([#797])
- The TensorRT-LLM plugin no longer forwards TRT-build-only knobs (`fast_build`,
  `quant_config`, the engine build cache) to the pytorch backend, whose `TorchLlmArgs` rejects
  them under `extra='forbid'` - this crashed the default pytorch-backend config. Declaring
  `quant_config` or `fast_build=True` on the pytorch backend now raises `ConfigError` rather
  than silently measuring a different configuration. ([#797])
- The four silent `ImportError` fallbacks for TensorRT-LLM sub-configs (QuantConfig,
  BuildCacheConfig, KvCacheConfig, SchedulerConfig) now raise `EngineError` when the user
  declared that sub-config and the native class cannot be imported, instead of silently
  dropping it and measuring a different configuration than declared. ([#797])
- Corrected stale TensorRT-LLM version references: the `RequestOutput.metrics` comment (metrics
  are absent at 1.2.1 without `return_perf_metrics`, not "usually absent in 0.21.0") and the
  container entrypoint's engine-version list (now vLLM 0.19.1 / TRT-LLM 1.2.1 / Python 3.12).
  ([#797])
- Absorb sign-off records now carry the full withheld rule body, so a maintainer's
  `human_confirmed` mark re-ships a rule even after the withholding run dropped it from the
  corpus; a bodyless mark fails loudly instead of silently skipping. ([#792])
- The construction-probe gate rejects pydantic type-coercion noise instead of confirming
  false-positive rules, and bare present-flag claims are unprobeable by construction. ([#792])
- TensorRT discovery and probe containers route through the NVIDIA entrypoint: the 1.2.1 NGC
  image moved `LD_LIBRARY_PATH` setup into `/etc/shinit_v2`, so bypassing the entrypoint broke
  `import tensorrt`. ([#792])

## [v0.10.0] - 2026-07-13

The engine-knowledge-as-data milestone. Hand-curated per-engine config was replaced by
typed Pydantic configs code-generated from validation rules mined directly from engine
source; the engine-coupling restructure, per-engine SSOT version pins, a Docusaurus docs
site, study-authoring commands (`llem study init` / bounds mode / sweep idioms / plan
preview), the absorb conductor, and a rewritten hosted-byte-verification CI all landed on
this line. Engine pins advanced to vLLM 0.19.1, TensorRT-LLM 1.0.0, and Transformers 5.7.0.

### Breaking Changes

- **Hand-curated `engine_configs.py` deleted; per-engine typed configs are now
  code-generated.** The ~1100-line hand-maintained `engine_configs.py` was removed. Each
  engine's typed Pydantic config surface is now code-generated from validation rules mined
  directly from that engine's source, keyed to the pinned version under `engine_versions/`.
  A parity gate guarded the deletion. Author-facing YAML is unchanged; only the internal
  config construction path moved to codegen. ([#733], [#734], [#735], [#736], [#738])

- **Engine validation vocabulary renamed from "invariants" to "rules".** The mined corpus,
  its loader surface, and the CLI/docs terminology now say "rules". A single shipped rules
  corpus with a closed severity enum replaces the prior split, and the invariant-era
  compatibility shims were dropped. YAML that referenced the old `engine_invariants` naming
  must move to the `rules` vocabulary. ([#480], [#572], [#737], [#740])

- **`llem run` reduced to session flags only.** The semantic-override flags were removed;
  experiment parameters now live in the YAML config (author one with `llem study init`), and a
  config path is now required. Flags describe the session; YAML describes the experiment. Removed
  flags and their YAML equivalents:
  - `--model` / `-m` -> `task.model`
  - `--engine` / `-e` -> `engine`
  - `--dataset` / `-d` -> `task.dataset.source`
  - `--n-prompts` / `-n` -> `task.dataset.n_prompts`
  - `--cycles` -> `study_execution.n_cycles`
  - `--order` -> `study_execution.experiment_order`
  - `--no-gaps` -> `study_execution.experiment_gap_seconds: 0` (and `cycle_gap_seconds: 0`)
  - `--timeout` -> `study_execution.wall_clock_timeout_hours`
  - `--no-circuit-breaker` -> `study_execution.max_consecutive_failures: 0`
  - `--fail-fast` -> `study_execution.max_consecutive_failures: 1`
  - `--no-dedup` -> `study_execution.deduplicate_equivalent: false`

  Migrate the quick single-run path:
  ```bash
  # before
  llem run --model gpt2 --engine transformers
  # after
  llem study init -m gpt2 --defaults
  llem run study.yaml
  ```

  Retained session flags: `--output`, `--quiet`, `--verbose`, `--resume`, `--resume-dir`,
  `--dry-run`, `--no-lock`, `--skip-preflight`. Passing a removed flag now gives Typer's standard
  "No such option" (exit 2). ([#749])

- **`engine: pytorch` renamed to `engine: transformers`** throughout YAML, CLI, and Python API.
  The `pytorch` identifier has been renamed to `transformers` - the engine runs HuggingFace
  Transformers `.generate()`. PyTorch is the tensor substrate, not the engine, and renaming aligns
  with `pip install transformers` and the library that owns the inference API.

  Migrate with:
  ```bash
  sed -i 's/engine: pytorch/engine: transformers/g; s/^pytorch:/transformers:/g' your-study.yaml
  ```

  Affected: YAML engine value, YAML section key, `PyTorchConfig` class, `ENGINE_PYTORCH` constant,
  `[pytorch]` extra, `LLEM_RUNNER_PYTORCH`/`LLEM_IMAGE_PYTORCH` env vars, Docker image tags.
  Preserved (PyTorch the library - unchanged): `import torch`, `torch_dtype`, `pytorch/pytorch:*`
  base image, `PYTORCH_VERSION` build args, `torch_compile_backend` field. ([#261])

- **`backend:` field and `--backend` flag renamed to `engine:`** in YAML configs, CLI, and result
  JSON. Aligns terminology with how vLLM, TRT-LLM, and HuggingFace use "engine" natively.

  Migrate with:
  ```bash
  sed -i 's/^\(\s*\)backend:/\1engine:/g' your-study.yaml
  ```

  Affected: YAML field, CLI flag (`-b` becomes `-e`), result JSON fields `"backend"` and
  `"backend_version"`, Python symbols `BackendPlugin`, `BackendError`, `BACKEND_*` constants,
  `get_backend()`, `detect_default_backend()`. ([#260])

- **`tensorrt.tp_size` renamed to `tensorrt.tensor_parallel_size`** to match `TrtLlmArgs` native
  naming. `transformers.tp_size` is unchanged (follows the `accelerate` convention). ([#269])

- **Typed-field curation for engine configs.** Applies the maximalist rubric "type anything with a
  plausible energy/throughput/latency path" to each engine's Pydantic surface. Dropped fields
  remain settable via YAML (`extra="allow"` passthrough unless noted). ([#270])

  Transformers: drops `revision` (reproducibility metadata) and `trust_remote_code` (security
  toggle); adds `allow_tf32`, `autocast_enabled`, `autocast_dtype`, `low_cpu_mem_usage`.

  vLLM: drops `sampling.max_tokens` and `beam_search.max_tokens` (duplicates of
  `ExperimentConfig.max_output_tokens`); adds `num_scheduler_steps`, `max_seq_len_to_capture`,
  `distributed_executor_backend`; replaces flat speculative fields with nested `VLLMSpeculativeConfig`.

  TensorRT-LLM: drops `engine_path`, `TensorRTCalibConfig`, `TensorRTBuildCacheConfig`,
  `sampling.return_perf_metrics`, and `backend: Literal["trt"]`; adds `pipeline_parallel_size`
  and `max_num_tokens`.

- **Engines (vLLM, TensorRT-LLM) now run exclusively inside Docker.** Host extras `[vllm]` and
  `[tensorrt]` removed. Only `[transformers]` remains host-installable. ([#498])

- **`dtype:` and `decoder:` fields migrated into per-engine sub-configs.** Top-level
  `ExperimentConfig.dtype` and `ExperimentConfig.decoder` have moved to each engine's own
  configuration section. ([#290], [#291])

- **`--dtype` and `--batch-size` CLI flags removed.** Both fields are now set via YAML config only.
  ([#292])

- **`precision:` field renamed to `dtype:`** with standard value strings (e.g. `float16`, `bfloat16`
  instead of the prior enum). ([#196])

### Added

- `llem doctor` CLI command reports per-engine image status (OK / MISMATCH / UNVERIFIED /
  UNREACHABLE) and exits non-zero on mismatch for CI gating. ([#256])
- Host/container schema fingerprint verification: Docker images stamped at build time with a
  `llem.expconf.schema.fingerprint` OCI label. Mismatches abort with a rebuild hint. Bypassable
  via `LLEM_SKIP_IMAGE_CHECK=1`. ([#256])
- `SchemaLoader` class (`llenergymeasure.config.SchemaLoader`) reads vendored engine schemas via
  `importlib.resources` with per-instance caching and major-version envelope validation. ([#268])
- Engine parameter discovery introspects installed engine packages inside their Docker images.
  The initial standalone script (`scripts/discover_engine_schemas.py`, #266) was later superseded
  by the `scripts/engine_producers/` codegen toolkit (see Breaking Changes). ([#266])
- Vendored engine parameter schemas at `src/llenergymeasure/engines/{vllm,tensorrt,transformers}/`.
  Regenerate with `make discover-schema ENGINE=<engine>`. ([#266])
- Per-engine sub-package layout (`src/llenergymeasure/engines/<engine>/`) co-locating runtime data,
  schema JSON, and engine invariants YAML. ([#570])
- Per-engine SSOT for library version pins (`engine_versions/`) used by Renovate, Dockerfiles,
  and the invariant-mining pipeline. ([#477])
- Engine invariants mining pipeline: static and dynamic miners for all three engines extract
  validation rules as a reproducible corpus. ([#375], [#434], [#444])
- Vendor-replay CI gate validates corpus against live engine packages; TensorRT gate runs on
  self-hosted GPU runner. ([#414], [#440], [#447])
- `probe` primitive for binary miner reusability check. ([#482])
- `ConfigProbe` protocol and per-engine `probe_config()` implementations. ([#293])
- Configurable per-experiment timeout via `study_execution.experiment_timeout_seconds` (default
  600 s), replacing the previous `max(n_prompts * 2, 600)` heuristic. Both local and Docker paths
  honour the same field. ([#250])
- Disk-persisted baseline power cache with configurable strategy and TTL enforcement. ([#242], [#243])
- Per-study JSONL log capturing runtime warnings and container stderr. ([#395])
- `llem report-gaps` command proposes corpus rules from runtime observations. ([#397])
- Study robustness features: circuit breaker, resume-on-failure, GPU locks, container lifecycle
  management. ([#214])
- Live per-experiment progress display with Rich panels and sub-bullet heartbeats. ([#152], [#165])
- `.env`-based runtime config and configurable `device_map` default. ([#275])
- `trust_remote_code` opt-in via `LLEM_TRUST_REMOTE_CODE` env var. ([#274])
- TRT-LLM build cache configurable via `LLEM_TRT_BUILD_CACHE_{ENABLED,DIR}` env vars. ([#277])
- Tensor parallelism fields (`tp_plan`, `tp_size`) for the Transformers engine. ([#161])
- Cross-field operators in vendored-rules loader. ([#410])
- Docusaurus documentation site at `website/` serving user, methodology, API, and architecture
  docs. ([#566])
- Per-engine discovered-schema Markdown digest rendered to `docs/`. ([#560])
- Architecture documentation suite in `docs/architecture/`. ([#433])
- Per-engine engine-invariants and engine-schemas CI workflows with cross-pipeline coordination
  (consolidated from predecessor mine + vendor + parameter-discovery workflows). ([#484], [#486])
- Engine-pipeline orchestrator (`engine-pipeline.yml`) as single reusable workflow entry point.
  ([#514], [#573])
- Cloudflare Pages PR preview deploy workflow. ([#575])
- SSOT audit trail and GHCR image retention policies. ([#546])
- Engine-knowledge SSOT workspace under `engine_versions/` with per-version mined producer
  snapshots, a shared schema-extraction substrate, and workspace-driven codegen for the typed
  engine config models. ([#733], [#734], [#735], [#736])
- Vendored per-version producer snapshots for all three engines (vLLM 0.7.3 / 0.16.0 / 0.18.1 /
  0.19.1, TensorRT-LLM 0.21.0 / 1.0.0 / 1.2.0 / 1.2.1, Transformers 4.57.3 / 5.3.0 / 5.6.2 / 5.7.0).
  ([#599], [#600], [#601], [#636], [#637], [#638], [#639], [#640], [#641], [#642])
- Schema-vs-source drift tool (coverage and added-direction probes, `EXCLUSIONS.yaml`, sticky-comment
  gate) replacing the prior probe primitive. ([#635], [#643], [#644], [#649], [#650])
- `llem study init` scaffold command for authoring a study YAML. ([#745])
- `llem study plan` preview command showing the expanded experiment grid. ([#750])
- Numeric sweep-axis idioms (`span`, `log`, `pow2`) for concise study axes. ([#746])
- Bounds mode with series policies and per-axis overrides. ([#747])
- Sweep configs record the rejecting rule id when a candidate is skipped. ([#741])
- Absorb conductor orchestrates engine-rule refresh across all engines. ([#756], [#771])
- Upstream validator coverage check confirms mined rules cover each engine's validators. ([#757])
- Cold-read analyst proposer for engine rules. ([#755])
- Citation checker (tier 1 of the rule verification ladder). ([#753])
- Construction and identity probe kernel for rule verification. ([#754])
- Observed-collision miner surfaces dormant rule candidates from runtime observations. ([#752])
- Self-hosted Renovate cron; engine-version scanning revived and its config migrated. ([#732],
  [#775], [#776])
- Generated-doc drift gate in the docs-freshness workflow. ([#760], [#761])
- Fan-in gate making the engine-rules-check requireable without deadlock; bump gates made
  reachable and requireable. ([#769], [#772])
- Runtime-literal discovery stage: string-literal candidates pooled from corpus cross-refs,
  upstream AST/docstring scans, LLM proposals, and previous-schema carry-forward, each verified
  by a two-leg construction probe in the pinned container; confirmed literals are recorded in
  the discovered schema and code-generated as union types. Standing census via
  `make check-corpus-literals`. ([#789])
- Miner final-run recall check with report, a probe-confirmed nested vLLM compilation-config
  cross-field rule, and nested-path rule firing tests. ([#788])

### Changed

- Re-typed `tensorrt.backend` as `Literal["trt", "pytorch", "_autodeploy"] | None` (reverses a
  prior incorrect curation-pass drop; `None` lets TRT-LLM auto-pick the runtime path). ([#276])
- Engine-invariants pipeline consolidated from separate mine + vendor + parameter-discovery
  workflows into a single orchestrated flow with sequential downstream pipelines. ([#484], [#573])
- `study_execution` field names updated (execution fields renamed, `reverse`/`latin_square`
  ordering modes added). ([#190])
- Dataset restructured into nested `DatasetConfig` sub-model. ([#195])
- `OutputConfig` extracted from `ExperimentConfig` as a separate sub-model. ([#203])
- `EnergyConfig` flattened to `energy_sampler` + `gpu_telemetry` fields. ([#201])
- `study_name` field replaces generic `name` field in study configs. ([#182])
- `n_prompts` default reduced to 50; `max_output_tokens` default bumped to 256. ([#175], [#213])
- Renovate customManager retargeted from Dockerfile ARGs to `engine_versions/` SSOT. ([#481])
- First-party `Dockerfile.vllm` and `Dockerfile.tensorrt` replaced with upstream-direct images
  plus volume mounts. ([#509])
- Advanced engine pins to vLLM 0.19.1, TensorRT-LLM 1.0.0, and Transformers 5.7.0, adopting the
  generated nested configs at those versions. ([#738])
- Engine pipeline rewritten to hosted byte-verification (the mined corpus and generated configs
  are verified byte-for-byte against a fresh mine, no human source-diffing). ([#758])
- Documentation aligned to the byte-verification engine-knowledge flow and the rules vocabulary;
  stale engine and contributing pages rewritten. ([#759], [#764], [#774])
- Discovery write path inverted: container discovery writes only under `engine_versions/`, with
  `make promote-schemas` as the sole writer of the packaged copies. ([#785])
- Knowledge-production scripts gated by ruff and mypy in Makefile and CI; import-linter layer
  contracts made honest; bundle artefact filenames centralised as constants. ([#784])
- Dead code deleted and duplicated helpers folded across study and scripts layers. ([#786])
- Docs: heading anchors match the live slugifier, version references pinned to `current.yaml`
  sources, and curation-era drift rewritten. ([#787])

### Fixed

- `ImportError: cuKernelGetName` when importing `tensorrt_llm`: LD_LIBRARY_PATH ordering placed
  the bundled compat CUDA 12.2 library ahead of the host-driver mount. Fixed by prepending
  `/usr/local/cuda/compat/lib` so the host-driver mount takes precedence. ([#264])
- Miner `added_at` timestamp lost on re-mine; f-string `message_template` fields now rendered
  correctly. ([#523])
- `Dockerfile.transformers` stale references to the old `[pytorch]` extra and header comments
  corrected. ([#265])
- Config hash mismatch in Docker study runs resolved. ([#176])
- Config-identity hash now covers the active engine's `harness` block (`batch_size`,
  `torch_compile`, `allow_tf32`, `autocast`). These drive execution but were omitted, so a
  `harness.batch_size` sweep collapsed to a single resolved hash and default dedup ran one
  experiment instead of the full sweep. ([#783])
- `measurement.*` methodology fields now join the config-identity hash, so sweeping warmup,
  baseline, energy sampler, or windowing produces distinct runs rather than deduping to one. ([#783])
- `--no-dedup` no longer crashes with a `KeyError` when a sweep canonicalises two grid points
  to the same declared config: manifest entries are built from actual per-hash occurrence
  counts, keeping the manifest aligned with the runner's per-occurrence cycle counter. ([#783])
- Non-matching engine sections stripped correctly during multi-engine grid expansion. ([#171])
- Docker auto-elevation enforced for multi-engine studies. ([#172])
- Baseline cache path resolved before Docker bind-mount. ([#248])
- Purged false-positive and phantom rules from the shipped corpus. ([#767])
- Rule message rendering and loader guards corrected. ([#768])
- Study dedup fallback, grid edge cases, and dormant-candidate visibility repaired. ([#770])
- Schema discovery retargeted at the `current.yaml` pins. ([#765])
- Pin-driven Transformers publish and GHCR prune exemption. ([#766])
- Rules follow-ups: probe operator canonicalisation, loader operator allowlist, message-template
  residue, and `{invariant_id}` renamed to `{rule_id}`. ([#779])
- Per-engine upstream default images resolve from the pinned engine version instead of a
  never-published GHCR ref; `llem doctor` verifies them. ([#780])
- Energy-measurement failure is loud instead of a silent zero: sampler auto-selection warns,
  absent energy stays `None` rather than coercing to `0.0`, sampler failures set a measurement
  warning, and NVML init failures log debug traces. ([#781])
- Documented top-level `images:` study key no longer leaks into experiment configs and rejects
  the whole study; user-facing config copy de-jargonised. ([#782])
- Self-hosted Renovate aborted before opening update PRs: the stability commit-status POST the
  App token cannot make is now disabled in config. ([#791])

### Removed

- Internal helper `llenergymeasure.study.runner._calculate_timeout` (replaced by direct config
  reads). ([#529])
- First-party `Dockerfile.vllm` and `Dockerfile.tensorrt` engine images. ([#509])
- Dead invariant-mining pipeline and the `refresh-invariants` / `schema-diff` scripts. ([#762],
  [#763])
- Orphaned files, dead references, and retired mined-invariants documentation pages. ([#760],
  [#773])
- Predecessor CI workflows: `auto-mine.yml`, `vendor-tensorrt.yml`, `vendor-vllm.yml`,
  `parameter-discovery.yml`, and predecessors. ([#483], [#485])


## [v0.9.0] - 2026-03-20

Docker infrastructure, vLLM engine, TensorRT-LLM engine, package restructure, test hardening, and CI.

### Added

- NVML GPU memory residual check before experiment dispatch (threshold 1 GB), preventing
  stale-process contamination. ([#24], [#26])
- Docker runner infrastructure: container lifecycle management, volume mounts, GPU index
  resolution. ([#27], [#124])
- Docker pre-flight environment checks. ([#28])
- TensorRT-LLM Docker image rewrite with CUDA 12.6.2 upgrade. ([#114])
- `TensorRTConfig` expanded to full TRT-LLM parameter schema. ([#115])
- `mpirun` injection for TensorRT-LLM tensor parallelism. ([#116])
- `BackendPlugin.validate_config` protocol method. ([#121])
- `TensorRTBackend` implementation registered in `get_backend()`. ([#122])
- `TensorRTConfig.engine_path` for pre-compiled engine loading. ([#143])
- 9-layer import-linter architecture enforcement in CI. ([#135], [#144])

### Changed

- Package restructured with file moves, import rewrites, and layer boundary fixes. ([#133], [#134])
- Prompt loading moved outside the NVML measurement window. ([#145])
- Shared backend helpers extracted; dead warmup code removed. ([#140])
- Test suite restructured; `importorskip` guards added for optional dependencies. ([#137], [#138])

### Fixed

- `accelerate` restored as a `[pytorch]` optional dependency (accidentally dropped). ([#132])
- Runner mode auto-detection (local vs Docker) on startup. ([#146])
- Silent `NVMLError`, payload detection, and empty `gpu_indices` guard. ([#141])

### Removed

- Dead code, stale type annotations, and unused dependencies. ([#130])


## [v0.8.0] - 2026-02-27

Multi-experiment study sweeps.

### Added

- `run_study()` public API for multi-experiment studies. ([#23])
- `StudyConfig` with sweep grammar (grid and cycle ordering). ([#23])
- YAML-driven parameter sweeps across models, engines, and precisions. ([#23])
- `StudyRunner` with sequential experiment dispatch. ([#23])
- Study-level aggregation and result collection. ([#23])
- Manifest-based progress tracking with resume support. ([#23])


## [v0.7.0] - 2026-02-27

First end-to-end single-experiment release.

### Added

- `run_experiment()` public API. ([#22])
- `ExperimentConfig` to `ExperimentResult` pipeline. ([#22])
- Energy measurement via CodeCarbon and Zeus backends. ([#22])
- Extended metrics: TPOT, TEI, memory efficiency. ([#22])
- Streaming latency measurement (TTFT / ITL). ([#22])
- Results persistence in Parquet format. ([#22])


---

## Historical (pre-0.x)

> The entries below predate the current 0.x versioning scheme introduced in early 2026.
> They describe the research prototype and early CLI rewrites that were restructured and
> re-versioned starting from v0.1.0. Version numbers v1.x and v2.0.0 referenced here are
> legacy labels from that era; they do not correspond to any published release under the
> current scheme. The 2026-03-04 history reset remapped these to sequential 0.x tags
> (v0.1.0-v0.6.0) for consistency with the current versioning scheme.

### v0.6.0 (2025-12-29) - formerly v1.16.0

Production-ready containerisation with full GPU support and streamlined developer experience.

#### Added

- Multi-stage Dockerfile with `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base image (builder,
  runtime, and dev stages).
- Docker Compose profiles separating production and development workflows (`lem-app`, `lem-dev`).
- VS Code devcontainer configuration with GPU passthrough and Ruff/Pylance extensions.
- Makefile targets for common Docker operations (`make docker-build`, `make experiment`,
  `make datasets`).

#### Changed

- CI workflow reliability improved with concurrency groups preventing parallel releases.
- Dev container runs as root, eliminating permission complexity with virtual environments.

#### Fixed

- Docker CUDA 12.4 base image aligned with host driver requirements.
- Volume permission errors resolved by running dev containers as root.
- Deprecated `torch_dtype` parameter replaced with `dtype` in model loading.
- Removed obsolete `TRANSFORMERS_CACHE` environment variable (superseded by `HF_HOME`).
- CodeCarbon pandas `FutureWarning` suppressed.
- `nvidia-smi` GPU utilisation parsing handles `[N/A]` values gracefully.

---

### v0.5.0 (2025-12-21) - formerly v1.15.0

Comprehensive test coverage ensuring reliability across all components.

#### Added

- End-to-end CLI tests (8 tests) validating complete benchmark workflows.
- Integration tests (47 tests) covering non-GPU workflows.
- Methodology documentation (`docs/methodology.md`) explaining measurement approach.

#### Changed

- Total test count: 416 passing tests (unit + integration + e2e).
- All tests run without GPU access using mocked/simulated data.

#### Removed

- `requirements.txt` (306 frozen packages) - all dependencies now managed via Poetry lockfile.

---

### v0.4.0 (2025-12-21) - formerly v1.13.0

User-friendly command-line interface replacing legacy entry points.

#### Added

- Typer-based CLI (`lem`) with subcommands: `experiment`, `aggregate`, `config validate`,
  `config show`, `results list`, `results show`, `datasets`.
- `ExperimentOrchestrator` with protocol-based dependency injection.
- `ExperimentContext` dataclass for runtime state management.
- Accelerate launcher with configurable retry logic.
- 25 CLI tests and 27 orchestration unit tests.

#### Removed

- Legacy `MAIN_*.py` entry points (6 files).

---

### v0.3.0 (2025-12-20) - formerly v1.10.0

Major architectural refactor establishing clean module boundaries.

#### Breaking Changes

- Package renamed: `llm-bench` to `lem`. All imports now use `llenergymeasure`.

#### Added

- Energy backend plugin registry with automatic CodeCarbon registration.
- `FlopsEstimator` with three-strategy fallback chain (calflops, architecture, parameter
  estimate), each returning a confidence level.
- Results aggregation with temporal overlap detection and GPU attribution verification.
- Export functionality for CSV and JSON formats.
- 296 unit tests covering all new modules.

#### Changed

- Replaced `print()` statements with Loguru structured logging.

---

### v0.2.0 (2025-05-17) - formerly v1.0.0

Research phase complete - stable multi-model benchmarking validated on production hardware.

#### Added

- Multi-model experiment support with scenario-based configuration.
- Experiment suite CSV export with consistent naming conventions.
- Failed experiment detection with cycle tracking and automatic retry.
- Minimum output token enforcement for comparable generation lengths.
- Large model stability improvements (gradient checkpointing, CUDA cache clearing).
- Data wrangling pipelines for experiment result analysis (Pandas-based).
- Plotting functionality for efficiency metrics visualisation.
- FLOPs caching preventing redundant calculations.

---

### v0.1.0 (2025-03-22) - formerly v0.5.0

Core measurement functionality establishing the foundation for all subsequent development.

#### Added

- Distributed results aggregation across multiple GPUs with per-process JSON files.
- FLOPs calculation with quantisation awareness and `calflops` integration.
- Robust process cleanup with signal handlers and distributed barrier synchronisation.
- Optimum benchmark integration for standardised measurements.

#### Changed

- Distributed execution stability improved: proper NCCL initialisation and teardown.
- Major directory restructuring separating config, core, and result handling.


[Unreleased]: https://github.com/henrycgbaker/llenergymeasure/compare/v0.11.0...HEAD
[v0.11.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.11.0
[v0.10.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.10.0
[v0.9.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.9.0
[v0.8.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.8.0
[v0.7.0]: https://github.com/henrycgbaker/llenergymeasure/releases/tag/v0.7.0

[#22]: https://github.com/henrycgbaker/llenergymeasure/pull/22
[#23]: https://github.com/henrycgbaker/llenergymeasure/pull/23
[#24]: https://github.com/henrycgbaker/llenergymeasure/pull/24
[#26]: https://github.com/henrycgbaker/llenergymeasure/pull/26
[#27]: https://github.com/henrycgbaker/llenergymeasure/pull/27
[#28]: https://github.com/henrycgbaker/llenergymeasure/pull/28
[#113]: https://github.com/henrycgbaker/llenergymeasure/pull/113
[#114]: https://github.com/henrycgbaker/llenergymeasure/pull/114
[#115]: https://github.com/henrycgbaker/llenergymeasure/pull/115
[#116]: https://github.com/henrycgbaker/llenergymeasure/pull/116
[#121]: https://github.com/henrycgbaker/llenergymeasure/pull/121
[#122]: https://github.com/henrycgbaker/llenergymeasure/pull/122
[#124]: https://github.com/henrycgbaker/llenergymeasure/pull/124
[#130]: https://github.com/henrycgbaker/llenergymeasure/pull/130
[#132]: https://github.com/henrycgbaker/llenergymeasure/pull/132
[#133]: https://github.com/henrycgbaker/llenergymeasure/pull/133
[#134]: https://github.com/henrycgbaker/llenergymeasure/pull/134
[#135]: https://github.com/henrycgbaker/llenergymeasure/pull/135
[#137]: https://github.com/henrycgbaker/llenergymeasure/pull/137
[#138]: https://github.com/henrycgbaker/llenergymeasure/pull/138
[#140]: https://github.com/henrycgbaker/llenergymeasure/pull/140
[#141]: https://github.com/henrycgbaker/llenergymeasure/pull/141
[#143]: https://github.com/henrycgbaker/llenergymeasure/pull/143
[#144]: https://github.com/henrycgbaker/llenergymeasure/pull/144
[#145]: https://github.com/henrycgbaker/llenergymeasure/pull/145
[#146]: https://github.com/henrycgbaker/llenergymeasure/pull/146
[#147]: https://github.com/henrycgbaker/llenergymeasure/pull/147
[#152]: https://github.com/henrycgbaker/llenergymeasure/pull/152
[#161]: https://github.com/henrycgbaker/llenergymeasure/pull/161
[#165]: https://github.com/henrycgbaker/llenergymeasure/pull/165
[#171]: https://github.com/henrycgbaker/llenergymeasure/pull/171
[#172]: https://github.com/henrycgbaker/llenergymeasure/pull/172
[#175]: https://github.com/henrycgbaker/llenergymeasure/pull/175
[#176]: https://github.com/henrycgbaker/llenergymeasure/pull/176
[#182]: https://github.com/henrycgbaker/llenergymeasure/pull/182
[#190]: https://github.com/henrycgbaker/llenergymeasure/pull/190
[#195]: https://github.com/henrycgbaker/llenergymeasure/pull/195
[#196]: https://github.com/henrycgbaker/llenergymeasure/pull/196
[#201]: https://github.com/henrycgbaker/llenergymeasure/pull/201
[#203]: https://github.com/henrycgbaker/llenergymeasure/pull/203
[#213]: https://github.com/henrycgbaker/llenergymeasure/pull/213
[#214]: https://github.com/henrycgbaker/llenergymeasure/pull/214
[#242]: https://github.com/henrycgbaker/llenergymeasure/pull/242
[#243]: https://github.com/henrycgbaker/llenergymeasure/pull/243
[#248]: https://github.com/henrycgbaker/llenergymeasure/pull/248
[#250]: https://github.com/henrycgbaker/llenergymeasure/pull/250
[#256]: https://github.com/henrycgbaker/llenergymeasure/pull/256
[#260]: https://github.com/henrycgbaker/llenergymeasure/pull/260
[#261]: https://github.com/henrycgbaker/llenergymeasure/pull/261
[#264]: https://github.com/henrycgbaker/llenergymeasure/pull/264
[#265]: https://github.com/henrycgbaker/llenergymeasure/pull/265
[#266]: https://github.com/henrycgbaker/llenergymeasure/pull/266
[#268]: https://github.com/henrycgbaker/llenergymeasure/pull/268
[#269]: https://github.com/henrycgbaker/llenergymeasure/pull/269
[#270]: https://github.com/henrycgbaker/llenergymeasure/pull/270
[#274]: https://github.com/henrycgbaker/llenergymeasure/pull/274
[#275]: https://github.com/henrycgbaker/llenergymeasure/pull/275
[#276]: https://github.com/henrycgbaker/llenergymeasure/pull/276
[#277]: https://github.com/henrycgbaker/llenergymeasure/pull/277
[#290]: https://github.com/henrycgbaker/llenergymeasure/pull/290
[#291]: https://github.com/henrycgbaker/llenergymeasure/pull/291
[#292]: https://github.com/henrycgbaker/llenergymeasure/pull/292
[#293]: https://github.com/henrycgbaker/llenergymeasure/pull/293
[#375]: https://github.com/henrycgbaker/llenergymeasure/pull/375
[#395]: https://github.com/henrycgbaker/llenergymeasure/pull/395
[#397]: https://github.com/henrycgbaker/llenergymeasure/pull/397
[#410]: https://github.com/henrycgbaker/llenergymeasure/pull/410
[#414]: https://github.com/henrycgbaker/llenergymeasure/pull/414
[#433]: https://github.com/henrycgbaker/llenergymeasure/pull/433
[#434]: https://github.com/henrycgbaker/llenergymeasure/pull/434
[#440]: https://github.com/henrycgbaker/llenergymeasure/pull/440
[#444]: https://github.com/henrycgbaker/llenergymeasure/pull/444
[#447]: https://github.com/henrycgbaker/llenergymeasure/pull/447
[#477]: https://github.com/henrycgbaker/llenergymeasure/pull/477
[#480]: https://github.com/henrycgbaker/llenergymeasure/pull/480
[#481]: https://github.com/henrycgbaker/llenergymeasure/pull/481
[#482]: https://github.com/henrycgbaker/llenergymeasure/pull/482
[#483]: https://github.com/henrycgbaker/llenergymeasure/pull/483
[#484]: https://github.com/henrycgbaker/llenergymeasure/pull/484
[#485]: https://github.com/henrycgbaker/llenergymeasure/pull/485
[#486]: https://github.com/henrycgbaker/llenergymeasure/pull/486
[#498]: https://github.com/henrycgbaker/llenergymeasure/pull/498
[#509]: https://github.com/henrycgbaker/llenergymeasure/pull/509
[#514]: https://github.com/henrycgbaker/llenergymeasure/pull/514
[#523]: https://github.com/henrycgbaker/llenergymeasure/pull/523
[#529]: https://github.com/henrycgbaker/llenergymeasure/pull/529
[#546]: https://github.com/henrycgbaker/llenergymeasure/pull/546
[#560]: https://github.com/henrycgbaker/llenergymeasure/pull/560
[#566]: https://github.com/henrycgbaker/llenergymeasure/pull/566
[#570]: https://github.com/henrycgbaker/llenergymeasure/pull/570
[#572]: https://github.com/henrycgbaker/llenergymeasure/pull/572
[#573]: https://github.com/henrycgbaker/llenergymeasure/pull/573
[#575]: https://github.com/henrycgbaker/llenergymeasure/pull/575
[#599]: https://github.com/henrycgbaker/llenergymeasure/pull/599
[#600]: https://github.com/henrycgbaker/llenergymeasure/pull/600
[#601]: https://github.com/henrycgbaker/llenergymeasure/pull/601
[#635]: https://github.com/henrycgbaker/llenergymeasure/pull/635
[#636]: https://github.com/henrycgbaker/llenergymeasure/pull/636
[#637]: https://github.com/henrycgbaker/llenergymeasure/pull/637
[#638]: https://github.com/henrycgbaker/llenergymeasure/pull/638
[#639]: https://github.com/henrycgbaker/llenergymeasure/pull/639
[#640]: https://github.com/henrycgbaker/llenergymeasure/pull/640
[#641]: https://github.com/henrycgbaker/llenergymeasure/pull/641
[#642]: https://github.com/henrycgbaker/llenergymeasure/pull/642
[#643]: https://github.com/henrycgbaker/llenergymeasure/pull/643
[#644]: https://github.com/henrycgbaker/llenergymeasure/pull/644
[#649]: https://github.com/henrycgbaker/llenergymeasure/pull/649
[#650]: https://github.com/henrycgbaker/llenergymeasure/pull/650
[#732]: https://github.com/henrycgbaker/llenergymeasure/pull/732
[#733]: https://github.com/henrycgbaker/llenergymeasure/pull/733
[#734]: https://github.com/henrycgbaker/llenergymeasure/pull/734
[#735]: https://github.com/henrycgbaker/llenergymeasure/pull/735
[#736]: https://github.com/henrycgbaker/llenergymeasure/pull/736
[#737]: https://github.com/henrycgbaker/llenergymeasure/pull/737
[#738]: https://github.com/henrycgbaker/llenergymeasure/pull/738
[#740]: https://github.com/henrycgbaker/llenergymeasure/pull/740
[#741]: https://github.com/henrycgbaker/llenergymeasure/pull/741
[#745]: https://github.com/henrycgbaker/llenergymeasure/pull/745
[#746]: https://github.com/henrycgbaker/llenergymeasure/pull/746
[#747]: https://github.com/henrycgbaker/llenergymeasure/pull/747
[#749]: https://github.com/henrycgbaker/llenergymeasure/pull/749
[#750]: https://github.com/henrycgbaker/llenergymeasure/pull/750
[#752]: https://github.com/henrycgbaker/llenergymeasure/pull/752
[#753]: https://github.com/henrycgbaker/llenergymeasure/pull/753
[#754]: https://github.com/henrycgbaker/llenergymeasure/pull/754
[#755]: https://github.com/henrycgbaker/llenergymeasure/pull/755
[#756]: https://github.com/henrycgbaker/llenergymeasure/pull/756
[#757]: https://github.com/henrycgbaker/llenergymeasure/pull/757
[#758]: https://github.com/henrycgbaker/llenergymeasure/pull/758
[#759]: https://github.com/henrycgbaker/llenergymeasure/pull/759
[#760]: https://github.com/henrycgbaker/llenergymeasure/pull/760
[#761]: https://github.com/henrycgbaker/llenergymeasure/pull/761
[#762]: https://github.com/henrycgbaker/llenergymeasure/pull/762
[#763]: https://github.com/henrycgbaker/llenergymeasure/pull/763
[#764]: https://github.com/henrycgbaker/llenergymeasure/pull/764
[#765]: https://github.com/henrycgbaker/llenergymeasure/pull/765
[#766]: https://github.com/henrycgbaker/llenergymeasure/pull/766
[#767]: https://github.com/henrycgbaker/llenergymeasure/pull/767
[#768]: https://github.com/henrycgbaker/llenergymeasure/pull/768
[#769]: https://github.com/henrycgbaker/llenergymeasure/pull/769
[#770]: https://github.com/henrycgbaker/llenergymeasure/pull/770
[#771]: https://github.com/henrycgbaker/llenergymeasure/pull/771
[#772]: https://github.com/henrycgbaker/llenergymeasure/pull/772
[#773]: https://github.com/henrycgbaker/llenergymeasure/pull/773
[#774]: https://github.com/henrycgbaker/llenergymeasure/pull/774
[#775]: https://github.com/henrycgbaker/llenergymeasure/pull/775
[#776]: https://github.com/henrycgbaker/llenergymeasure/pull/776
[#779]: https://github.com/henrycgbaker/llenergymeasure/pull/779
[#780]: https://github.com/henrycgbaker/llenergymeasure/pull/780
[#781]: https://github.com/henrycgbaker/llenergymeasure/pull/781
[#782]: https://github.com/henrycgbaker/llenergymeasure/pull/782
[#783]: https://github.com/henrycgbaker/llenergymeasure/pull/783
[#784]: https://github.com/henrycgbaker/llenergymeasure/pull/784
[#785]: https://github.com/henrycgbaker/llenergymeasure/pull/785
[#786]: https://github.com/henrycgbaker/llenergymeasure/pull/786
[#787]: https://github.com/henrycgbaker/llenergymeasure/pull/787
[#788]: https://github.com/henrycgbaker/llenergymeasure/pull/788
[#789]: https://github.com/henrycgbaker/llenergymeasure/pull/789
[#791]: https://github.com/henrycgbaker/llenergymeasure/pull/791
[#792]: https://github.com/henrycgbaker/llenergymeasure/pull/792
[#797]: https://github.com/henrycgbaker/llenergymeasure/pull/797
[#798]: https://github.com/henrycgbaker/llenergymeasure/pull/798
[#799]: https://github.com/henrycgbaker/llenergymeasure/pull/799
[#800]: https://github.com/henrycgbaker/llenergymeasure/pull/800
[#801]: https://github.com/henrycgbaker/llenergymeasure/pull/801
[#803]: https://github.com/henrycgbaker/llenergymeasure/pull/803
[#804]: https://github.com/henrycgbaker/llenergymeasure/pull/804
[#805]: https://github.com/henrycgbaker/llenergymeasure/pull/805
[#807]: https://github.com/henrycgbaker/llenergymeasure/pull/807
[#814]: https://github.com/henrycgbaker/llenergymeasure/pull/814
[#816]: https://github.com/henrycgbaker/llenergymeasure/pull/816
