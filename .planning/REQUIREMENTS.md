# Requirements: LLenergyMeasure v1.19.0 — M3: Docker + vLLM

**Defined:** 2026-02-27
**Core Value:** Researchers can run broad parameter sweeps across deployment configurations and produce publishable, methodology-sound measurements showing which implementation choices matter most for LLM energy efficiency.

## v1.19.0 Requirements

Requirements for M3. Each maps to roadmap phases.

### Docker Infrastructure

- [ ] **DOCK-01**: StudyRunner dispatches experiments to ephemeral Docker containers (`docker run --rm`) when runner is `docker`
- [ ] **DOCK-02**: Config passed to container via mounted JSON file + `LLEM_CONFIG_PATH` env var
- [ ] **DOCK-03**: Results returned via shared volume (`{config_hash}_result.json`)
- [ ] **DOCK-04**: Container completion signalled by process exit (`subprocess.run` blocking call)
- [ ] **DOCK-05**: Runner defaults to `local` for all backends. Multi-backend study with incompatible backends auto-elevates to Docker with guidance. Users can override per-backend via config (ref: AIEnergyScore `USE_DOCKER` env var)
- [ ] **DOCK-06**: Runner selection configurable per-backend via user config (`runners:` section) and env var (`LLEM_RUNNER_VLLM=docker:image`). Ref: AIEnergyScore `USE_DOCKER` pattern, extended to per-backend granularity
- [ ] **DOCK-07**: Docker pre-flight validates NVIDIA Container Toolkit is installed
- [ ] **DOCK-08**: Docker pre-flight validates GPU visibility inside container (`nvidia-smi` test)
- [ ] **DOCK-09**: Docker pre-flight validates CUDA/driver version compatibility
- [ ] **DOCK-10**: Official vLLM Docker image published to GHCR (`ghcr.io/llenergymeasure/vllm:{version}-cuda{major}`)
- [ ] **DOCK-11**: Container entrypoint calls `ExperimentOrchestrator` directly (library API, not CLI re-entry)

### vLLM Backend

- [ ] **VLLM-01**: vLLM inference backend activated and producing valid ExperimentResult via Docker
- [ ] **VLLM-02**: P0 fix: vLLM streaming broken (CM-07 from .product/REQUIREMENTS.md)
- [ ] **VLLM-03**: P0 fix: vLLM `--shm-size 8g` passed to container (CM-09 from .product/REQUIREMENTS.md)

### Measurement Quality

- [ ] **MEAS-01**: NVML GPU memory verification check before each experiment dispatch (both local and Docker paths). Ref: optimum-benchmark relies on process exit for cleanup; AIEnergyScore does explicit cleanup in non-Docker path. Subprocess isolation is more thorough — verification catches driver edge cases.
- [ ] **MEAS-02**: Warning logged if residual GPU memory exceeds threshold before experiment start
- [ ] **MEAS-03**: `aienergyscore.jsonl` built-in dataset file created (carried from M1)
- [ ] **MEAS-04**: `peak_memory_mb` measurement semantics confirmed and documented (carried from M1)

### Documentation

- [ ] **DOCS-01**: User documentation covering installation, getting started, configuration
- [ ] **DOCS-02**: Docker setup guide (NVIDIA Container Toolkit, host requirements, image selection)
- [ ] **DOCS-03**: Backend configuration guide (PyTorch local, vLLM Docker)
- [ ] **DOCS-04**: Study YAML reference with sweep grammar examples

### Testing

- [ ] **TEST-01**: Manual Ctrl+C SIGINT test on GPU hardware for Docker path (carried from M2)

## Future Requirements

Deferred to subsequent milestones. Tracked but not in current roadmap.

### TensorRT-LLM (M4/v1.20.0)

- **TRT-01**: TensorRT-LLM inference backend activated via Docker
- **TRT-02**: TRT engine cache at `~/.llenergymeasure/trt-engines/{hash}/` (INF-15)
- **TRT-03**: Official TRT Docker image published to GHCR (INF-16)

### SGLang (M5/v1.21.0)

- **SGL-01**: SGLang inference backend activated via Docker
- **SGL-02**: Official SGLang Docker image published to GHCR
- **SGL-03**: RadixAttention energy profile comparison with PagedAttention (vLLM)

### Advanced Features (deferred)

- **ADV-01**: `--resume` flag implementation (STU-10)
- **ADV-02**: Traffic simulation and streaming latency instrumentation
- **ADV-03**: Webhook notifications on experiment completion

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Persistent Docker containers | Ephemeral-only; no HTTP API inside containers. No peer tool exposes lifecycle toggle. |
| `llem compile-engines` pre-compilation | TRT engine caching is implicit; pre-compilation deferred. |
| In-process GPU cleanup (`torch.cuda.empty_cache`) | Subprocess isolation is more thorough than in-process cleanup. NVML verification check is sufficient. Ref: optimum-benchmark uses same approach. |
| Singularity/Apptainer runner | `NotImplementedError` in v2.0. HPC support deferred. |
| lm-eval integration | Quality metrics not v2.0 scope (v3.0). |
| Web platform | Separate product, separate repo (v4.0). |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DOCK-01 | — | Pending |
| DOCK-02 | — | Pending |
| DOCK-03 | — | Pending |
| DOCK-04 | — | Pending |
| DOCK-05 | — | Pending |
| DOCK-06 | — | Pending |
| DOCK-07 | — | Pending |
| DOCK-08 | — | Pending |
| DOCK-09 | — | Pending |
| DOCK-10 | — | Pending |
| DOCK-11 | — | Pending |
| VLLM-01 | — | Pending |
| VLLM-02 | — | Pending |
| VLLM-03 | — | Pending |
| MEAS-01 | — | Pending |
| MEAS-02 | — | Pending |
| MEAS-03 | — | Pending |
| MEAS-04 | — | Pending |
| DOCS-01 | — | Pending |
| DOCS-02 | — | Pending |
| DOCS-03 | — | Pending |
| DOCS-04 | — | Pending |
| TEST-01 | — | Pending |

**Coverage:**
- v1.19.0 requirements: 23 total
- Mapped to phases: 0
- Unmapped: 23 (roadmap not yet created)

---
*Requirements defined: 2026-02-27*
*Last updated: 2026-02-27 after initial definition*
