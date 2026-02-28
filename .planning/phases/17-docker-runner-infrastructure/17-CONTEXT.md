# Phase 17: Docker Runner Infrastructure - Context

**Gathered:** 2026-02-28
**Status:** Ready for planning

<domain>
## Phase Boundary

StudyRunner can dispatch experiments to ephemeral Docker containers, with config passed in and results passed out via shared volume. This phase delivers the DockerRunner dispatch path, config/result transfer, per-backend runner configuration, container entrypoint, and auto-elevation logic. Pre-flight checks (Docker/GPU validation) are Phase 18. vLLM backend activation is Phase 19.

</domain>

<decisions>
## Implementation Decisions

### Runner Config Syntax
- Per-backend runner config in study/experiment YAML: `runners: {backend: docker}` or `runners: {backend: "docker:custom/image:tag"}`
- Bare `docker` value resolves to built-in default image for that backend via internal registry
- `docker:full/image:tag` overrides with explicit image — researcher controls exact environment
- All three backends (PyTorch, vLLM, TensorRT) are architecturally equal — all support both `local` and `docker` runners. No backend is special-cased.
- Per-backend env var override: `LLEM_RUNNER_VLLM=docker`, `LLEM_RUNNER_PYTORCH=docker:custom/img`
- Resolution precedence (highest wins): env var > study/experiment YAML > user config (~/.llem/config.yaml) > built-in registry > default
- Runner works for both single experiments (`llem run`) and studies — consistent API

### Default Runner Selection (Docker-First)
- **Docker-first when available**: if Docker + NVIDIA Container Toolkit are detected, default runner is `docker` for all backends (best measurement isolation)
- **Local fallback**: if Docker not detected, default is `local` with a one-liner nudge: "Docker not detected. Install Docker + NVIDIA Container Toolkit for reproducible isolated measurements."
- User-set config always wins — explicit configuration is never overridden

### Auto-Elevation Behaviour
- When auto-elevating (Docker available, no explicit runner config), ALL backends in the study go to Docker — no hybrid local+Docker within a single study
- Mixed runners in a study (user explicitly set different runners per backend): warn "Mixed runners detected. For consistent measurements, consider running all backends in Docker." Then respect user config.
- Incompatible backend locally (e.g. vllm not installed, runner set to local): log error into results, skip that experiment with inline warning, continue study
- Auto-elevation message: minimal one-liner warning, proceed automatically. No confirmation prompt.

### Runner as Metadata (Not Identity)
- Runner is NOT part of the experiment config hash. Config hash = model + backend + params + dataset
- Runner type, image tag, and image digest are captured in `effective_config` metadata on ExperimentResult
- Container ID recorded in EnvironmentSnapshot
- Sweepable runner (treating runner as an experimental variable in sweep grid) deferred to a future enhancement

### Error Surfacing
- Categorised Docker error hierarchy: ImagePull, GPUAccess, OOM, Timeout, Permission
- Pattern-match known Docker and NVIDIA Container Toolkit error strings to translate cryptic messages into actionable guidance
- Every Docker error includes a "try:" fix suggestion (e.g. "Image not found. Try: docker pull ghcr.io/llem/vllm:1.19.0-cuda12")
- Always capture container stdout/stderr (last N lines) on failure and include in the error message
- Cleanup errors (temp dir deletion failures) are logged as warnings, never mask the real error

### Volume and Temp Management
- Exchange directory created in system temp: `tempfile.mkdtemp(prefix='llem-')`
- Config written as JSON to exchange dir: `{config_hash}_config.json`
- Container mounts exchange dir as `/run/llem` and receives `LLEM_CONFIG_PATH` env var pointing to the config file
- Container writes result to same mount: `{config_hash}_result.json`
- StudyRunner reads result from temp dir and copies to study output directory (final location)
- Cleanup policy: delete temp dir on success; keep on failure for debugging (log path: "Debug artifacts at /tmp/llem-abc123/")
- Config hash naming ensures deterministic correlation between config and result files

### Claude's Discretion
- Exact error category hierarchy and pattern-matching strings
- Number of stdout/stderr lines to capture on failure
- Built-in image registry data structure and CUDA version detection logic
- Docker `run` flags (--shm-size, --ipc, network mode) beyond --rm and --gpus all

</decisions>

<specifics>
## Specific Ideas

- Docker `run` command: `docker run --rm --gpus all -v /tmp/llem-{hash}:/run/llem -e LLEM_CONFIG_PATH=/run/llem/{hash}_config.json {image}`
- Container entrypoint calls `ExperimentOrchestrator` directly (library API, not CLI re-entry) — DOCK-11
- AIEnergyScore's `DOCKER_IMAGE` env var pattern extended to per-backend: `LLEM_RUNNER_{BACKEND}=docker:image`
- Testcontainers' pattern of inspecting container state (OOMKilled, ExitCode) on failure worth adopting
- Progressive disclosure: new users never see runner config; Docker-first default handles it. Power users can override per-backend.

</specifics>

<deferred>
## Deferred Ideas

- Sweepable runner (runner as experimental variable in sweep grid) — future enhancement, enables containerisation-overhead research
- Ryuk-style sidecar reaper for crash-safe container cleanup — evaluate if needed after Phase 17 implementation
- Docker Compose orchestration for multi-container studies — out of scope, single-container-per-experiment design

</deferred>

---

*Phase: 17-docker-runner-infrastructure*
*Context gathered: 2026-02-28*
