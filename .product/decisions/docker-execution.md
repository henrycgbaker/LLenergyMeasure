# Docker Execution Architecture

**Status:** Accepted
**Date decided:** 2026-02-19
**Last updated:** 2026-02-28
**Research:** [../research/13-execution-isolation-patterns.md](../research/13-execution-isolation-patterns.md)

## Decision

Ephemeral `docker run` per experiment. Config via env var, results via shared volume, completion signalled by process exit. Container entrypoint calls `ExperimentOrchestrator` directly (not CLI re-entry). TRT-LLM engines compiled and cached to `~/.llenergymeasure/trt-engines/` with content-addressed key. No user-facing container lifecycle toggle — ephemeral is the only model.

---

## Context

Multi-backend studies (e.g. comparing PyTorch vs vLLM in one study) cannot run backends in the same Python process — vLLM and TensorRT-LLM have CUDA/driver conflicts that make process-level isolation a correctness requirement. Docker provides inter-backend isolation at the container level.

This decision covers the v2.2 Docker orchestration design: how experiments are dispatched to containers, how config and results flow between host and container, and how TRT-LLM's engine compilation overhead is managed. Local single-backend execution (v2.0) uses a parallel but distinct isolation model — see [experiment-isolation.md](experiment-isolation.md).

**Constraints:**
- Container lifecycle must not introduce measurement error (startup latency is not part of the measurement window)
- The host orchestrator (`StudyRunner`) is the single source of timing control, including thermal gaps between experiments
- No peer tool exposes container lifecycle as a user-facing setting
- Implementation complexity must be justified by real benefit

---

## Considered Options

### D1: Container lifecycle model

| Option | Pros | Cons |
|--------|------|------|
| **Ephemeral `docker run` per experiment (chosen — like AIEnergyScore)** | Clean isolation per experiment; no state leaks between experiments; simple orchestration (subprocess.run blocks on exit); no HTTP API server required inside containers | Model must be loaded from disk for each experiment in a parameter sweep (mitigated by TRT disk cache and HF model cache) |
| Persistent containers (keep model in GPU memory between experiments) | Saves disk→GPU load time on repeated cycles of the same config | Requires an HTTP API server inside every backend image; two orchestration code paths; disk engine caching achieves the same for TRT; no peer tool exposes this as a toggle |
| `docker compose up` with long-lived services | Familiar to Docker Compose users | Same problems as persistent containers; adds compose dependency; no peer precedent |

---

### D2: Container entrypoint — CLI re-entry vs library API

| Option | Pros | Cons |
|--------|------|------|
| **Library API directly — thin Python entrypoint calling `ExperimentOrchestrator` (chosen)** | Config is already a fully-validated `ExperimentConfig` object (serialised to JSON); no re-parsing overhead; single code path for experiment execution | Container image must include the full library (already the case) |
| Re-entry via `llem run` CLI | Familiar to users debugging containers manually | Re-parses config through a second code path; adds overhead; validation already done by host |

---

### D3: Host–container communication protocol

| Option | Pros | Cons |
|--------|------|------|
| **Shared volume (results) + environment variable (config) — no IPC protocol (chosen)** | Simplest possible mechanism for ephemeral containers; completion signal is process exit; no API surface to maintain | Config size is limited by env var maximum (mitigated by mounted config file fallback) |
| HTTP API inside container | Supports persistent container model | Requires running server in every backend image; over-engineered for ephemeral model |
| `docker exec` for dispatch | Doesn't require a server | No peer tool uses it for experiment dispatch; tight coupling to container internals; harder timeouts; weak isolation |

---

### D4: TRT-LLM engine compilation strategy

| Option | Pros | Cons |
|--------|------|------|
| **Compile-and-cache to `~/.llenergymeasure/trt-engines/` (chosen)** | Cross-session and cross-study: compile once, reuse indefinitely; cache mounted into container as a volume; unique config → unique cache key | First run of each unique config is slow — inherent TRT constraint, not an LLenergyMeasure choice |
| Recompile every run | Always current; no stale cache | TRT compilation takes minutes per unique config; makes parameter sweeps impractical |
| Persistent container (avoid recompile) | Saves disk→GPU load on repeated cycles | See D1 rejection: lifecycle complexity not justified |

Cache key must include every field that changes the compiled engine architecture:

```python
def trt_compile_key(config: ExperimentConfig) -> str:
    """Canonical cache key for a TRT-LLM compiled engine."""
    trt = config.tensorrt or TensorRTConfig()
    key_fields = {
        "model": config.model,
        "precision": config.precision,
        "tp_size": trt.tp_size,
        "pp_size": trt.pp_size,
        "max_batch_size": trt.max_batch_size,
        "max_seq_len": trt.max_seq_len,
        "quantization": trt.quantization,
        "trt_version": _get_trt_version(),
        # builder_opt_level excluded — affects compilation strategy but not
        # engine architecture; same engine is functionally equivalent at
        # different opt levels
    }
    import hashlib, json
    return hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]
```

> **Updated (2026-02-26):** `trt_version` added to cache key. TRT-LLM engine format changes
> between library versions; without this, cached engines become stale silently after a
> TRT-LLM upgrade, producing incorrect results or crashes. `_get_trt_version()` returns
> `tensorrt_llm.__version__` (or falls back to `tensorrt.__version__`). See DECISION-AUDIT.md §2.17 P2.3.

Cache strategy:
- Cache key: `trt_compile_key(config)` — 16-char SHA-256 (includes TRT-LLM version)
- Cache location: `~/.llenergymeasure/trt-engines/{hash}/`
- Cache mounted into container as a volume
- First run per unique config: compile engine, write to cache
- Subsequent runs (same config): load from cache, no recompile
- TRT-LLM version upgrade: new cache key automatically, no manual invalidation needed

---

### D5: User-facing container lifecycle toggle

| Option | Pros | Cons |
|--------|------|------|
| **No toggle — ephemeral is the only supported model (chosen)** | Simplicity; one code path; `runner: docker` in study YAML selects Docker, lifecycle is always ephemeral | No flexibility for power users who understand persistent containers |
| `container_lifecycle: ephemeral \| persistent` in study YAML | Flexibility | Requires HTTP API server in every backend image; two orchestration code paths; different result-retrieval mechanism; no peer tool exposes this |

---

## Consequences

**Positive:**
- Clean per-experiment isolation — no state leaks between experiments
- Simple orchestration: `subprocess.run` blocks on container exit, no polling or API required
- TRT cache amortises compilation cost across sessions and studies
- No API surface inside containers to maintain

**Negative / Trade-offs:**
- Model is loaded from disk for each experiment in a parameter sweep (mitigated by HF model cache and TRT engine cache)
- TRT parameter sweeps over unique configs are inherently slow — this is a TRT-LLM constraint
- Docker is not available in all environments (HPC clusters with Apptainer/Singularity not addressed in v2.2)

**Neutral / Follow-up decisions triggered:**
- Deferred: `llem compile-engines study.yaml` pre-compilation command — not in v2.2
- Docker images published to Docker Hub under `llenergymeasure/` org — v2.2 planning
- Apptainer/Singularity HPC compatibility — deferred past v2.2

---

### D6: Default runner — Docker-first vs local-first

> **Added (2026-02-28):** Supersedes the original position that `local` is always the default.

| Option | Pros | Cons |
|--------|------|------|
| **Docker-first when available (chosen)** | Best measurement isolation by default; consistent CUDA/driver environment; progressive disclosure — new users never configure runners; aligns with AIEnergyScore's Docker-as-primary model | Requires Docker + NVIDIA Container Toolkit installed; first-run image pull is slow |
| Local always default (original position) | Zero dependencies; works immediately | Measurements affected by host environment variability; multi-backend studies impossible without explicit config; new users must learn about Docker to get good results |

**Decision:** When Docker + NVIDIA Container Toolkit are detected on the host, the default runner for all backends is `docker`. When Docker is not available, the default is `local` with a one-liner nudge recommending Docker installation. User-set runner config always takes precedence.

**Rationale:** For an energy measurement tool, the environment directly affects measurement quality. Docker provides process isolation, CUDA pinning, and cleaner energy baselines. Making it the default when available means new users get the best measurements without configuration. Users without Docker (or who explicitly prefer local) set `runners:` in their config — progressive disclosure.

All three backends (PyTorch, vLLM, TensorRT-LLM) are architecturally equal with respect to runners. Each implements the `InferenceBackend` protocol and can run in either `local` or `docker` mode. PyTorch is not special-cased despite being more commonly installed locally.

**Runner config syntax:**
- `runners: {vllm: docker}` — use built-in default image for the backend
- `runners: {vllm: "docker:ghcr.io/custom/img:tag"}` — explicit image override
- Per-backend env var override: `LLEM_RUNNER_VLLM=docker:image`
- Precedence (highest wins): env var > study YAML > user config > built-in registry > default

**Auto-elevation:** When Docker is available and a multi-backend study has no explicit runner config, all backends are dispatched to Docker. A minimal one-liner warning is shown. All backends go to Docker — no hybrid local+Docker within a single study. If user explicitly configures mixed runners (some local, some Docker), a warning is shown but the config is respected.

**Runner as metadata, not identity:** Runner type and image are NOT part of the experiment config hash. They are captured in `effective_config` metadata on `ExperimentResult` (image tag, image digest, container ID).

---

## Relation to Local Execution

The local (non-Docker) execution model uses the same structural pattern — see [experiment-isolation.md](experiment-isolation.md). Both paths have identical orchestration logic in `StudyRunner`; only the isolation primitive differs:

| | Local | Docker |
|---|---|---|
| Isolation primitive | `multiprocessing.Process` | `docker run` (ephemeral) |
| Config in | Function argument | Env var / mounted file |
| Result out | `multiprocessing.Pipe` | Shared volume |
| Completion signal | `process.join()` | `subprocess.run()` returns |

---

## Related

- [experiment-isolation.md](experiment-isolation.md): Local subprocess isolation model (v2.0)
- [decisions/versioning-roadmap.md](versioning-roadmap.md): Docker multi-backend is v2.2
- [decisions/installation.md](installation.md): Docker as system dependency; `runner: docker` scope
- [designs/study-yaml.md](../designs/study-yaml.md): `runner: docker` field
- [research/13-execution-isolation-patterns.md](../research/13-execution-isolation-patterns.md)
