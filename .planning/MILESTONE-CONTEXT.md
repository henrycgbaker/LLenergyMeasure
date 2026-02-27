# M3 Milestone Context

**Captured:** 2026-02-27
**Status:** Paused — shipping v1.18.1 patch first, then resuming M3 setup
**Resume with:** `/gsd:new-milestone`

## Confirmed Scope — M3: Docker Multi-Backend (vLLM)

**Version:** v1.19.0
**Goal:** Docker container infrastructure + vLLM backend activation. First multi-backend capability.

### Features

1. **Docker runner** — ephemeral `docker run` per experiment, auto-detection of multi-backend → Docker
2. **vLLM backend activation** — P0 fixes: streaming broken (CM-07), shm-size missing (CM-09)
3. **Docker images** — official images per backend, CI publish on release tag
4. **Docker pre-flight checks** — NVIDIA Container Toolkit installed, GPU visibility inside container, CUDA/driver version compat
5. **GPU memory cleanup between experiments** — NVML memory check/reset in Docker dispatch path (local path ships in v1.18.1)
6. **Documentation** — full user docs including Docker setup guide (Phase 13 from M2)

### Carried Items (folded into M3)

1. Create `aienergyscore.jsonl` built-in dataset file — carried from M1
2. Confirm `peak_memory_mb` measurement semantics — carried from M1
3. Manual Ctrl+C SIGINT test on GPU hardware — carried from M2 Phase 11
4. Phase 13 documentation (deferred from M2) — includes Docker setup guide

### Product Requirements (from .product/REQUIREMENTS.md)

**Core M3 requirements:**
- CM-02: vLLM inference backend (Docker)
- CM-07: P0 fix: vLLM streaming broken
- CM-08: P0 fix: Docker execution broken
- CM-09: P0 fix: vLLM `--shm-size` missing
- INF-13: Ephemeral `docker run` per experiment
- INF-14: Config via env var injection; results via shared volume
- INF-17: Docker treated as system dependency, checked at pre-flight

**Deferred from M3 to later milestones:**
- CM-03: TensorRT-LLM backend → M4/v1.20.0
- INF-15: TRT engine cache → M4/v1.20.0
- INF-16: Official images (all backends) → partially M3 (vLLM only), rest M4+

**New requirements identified (not in .product/REQUIREMENTS.md yet):**
- Docker pre-flight: NVIDIA Container Toolkit validation
- Docker pre-flight: GPU visibility test inside container (`nvidia-smi`)
- Docker pre-flight: CUDA/driver version compatibility check
- GPU memory cleanup in Docker dispatch path (local path ships in v1.18.1)

## Versioning Decisions

- **v1.18.1** — Patch: GPU memory cleanup in local StudyRunner (shipping first)
- **v1.19.0** — M3: Docker infra + vLLM backend
- **v1.20.0** — M4: TensorRT-LLM backend (builds on Docker infra)
- **v1.21.0** — M5: SGLang backend
- **v2.0.0** — Reserved for full multi-backend completion (after all milestones)

## Scope Boundaries

### In Scope
- Docker ephemeral container lifecycle
- vLLM backend only (not TRT-LLM, not SGLang)
- Docker pre-flight validation (NVIDIA Container Toolkit, GPU visibility, CUDA compat)
- GPU memory cleanup between Docker experiments
- Full user documentation including Docker setup
- Carried items from M1/M2

### Out of Scope
- TensorRT-LLM backend (M4/v1.20.0)
- SGLang backend (M5/v1.21.0)
- `--resume` flag (deferred)
- Traffic simulation, streaming latency (deferred)
- Persistent containers (ephemeral only)
- `llem compile-engines` pre-compilation command (deferred)

## Research Findings (from .product/research/)

Key peer patterns for Docker GPU benchmarking:
- **AIEnergyScore**: Ephemeral `docker run` per experiment (our model). Also does GPU memory cleanup between runs.
- **MLPerf**: Uses `--runtime=nvidia`, `--ipc=host`. Pre-flight health checks.
- **optimum-benchmark**: `multiprocessing.Process` per benchmark (local). Docker for CI only.

Key risks identified:
- NVIDIA Container Toolkit not installed → silent GPU failure
- GPU visibility fails silently inside container
- CUDA/driver version mismatch → silent corruption
- GPU memory fragmentation across containers (mitigated by cleanup)

## Design References

- `.product/decisions/docker-execution.md` — ephemeral model, env var config, shared volume results
- `.product/designs/docker-execution.md` — full implementation spec with code examples
- `.product/decisions/additional-backends.md` — SGLang accelerated to candidate, TRT isolation
- `.product/decisions/experiment-isolation.md` — local subprocess model (already implemented in M2)

## Previous Milestone Context

- M2 ended at phase 15. M3 phases should start from 16.
- Last plan count: 11 plans in M2.
- StudyRunner already has local subprocess isolation (multiprocessing.spawn + Pipe IPC).
- Multi-backend study without Docker currently → hard error at pre-flight (CM-10, shipped in M2).
