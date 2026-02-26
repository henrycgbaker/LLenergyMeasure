# Installation & Packaging

**Status:** Accepted
**Date decided:** 2026-02-17
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

Zero backend deps at base install. Pip extras per backend: `[pytorch]`, `[vllm]`, `[tensorrt]`, `[zeus]`, `[codecarbon]`. Docker is a system dependency (not pip-managed) — required for multi-backend studies, but also an optional runner config for single backends where users can choose to run in docker or locally by default. Progressive disclosure: base → extras → Docker.

---

## Context

LLenergyMeasure depends on inference backends (PyTorch/Transformers, vLLM, TensorRT-LLM) and energy measurement backends (Zeus, CodeCarbon) that are mutually incompatible in some combinations, very large as transitive dependencies, and not needed by all users. The packaging model must:

- Allow users to install only what they need
- Fail helpfully rather than silently when a missing backend is required
- Handle Docker as infrastructure (not a pip dependency) for multi-backend studies
- Not require a setup wizard or init ceremony

**Forces:**
- vLLM and TensorRT-LLM cannot coexist in the same Python process (CUDA/driver conflicts)
- Docker cannot be installed via pip — it is a system dependency
- Energy measurement method should be an explicit user choice (scientific claim integrity)
- CO2 estimation (CodeCarbon) is a separate concern from energy measurement (Zeus/NVML)

---

## Considered Options

### I1: Backend dependency model

| Option | Pros | Cons |
|--------|------|------|
| **pip extras per backend — `[pytorch]`, `[vllm]`, `[tensorrt]` (chosen)** | Standard pattern (lm-eval, Zeus, Optimum-Benchmark all use it); avoids large transitive dep trees; explicit user choice; vLLM + TRT process-incompatibility makes a single `[all]` extra incorrect | Users must know which extra to install |
| Include all backends by default | Zero user decision required | Installs conflicting CUDA libraries; multi-GB download for all users; makes process-incompatibility impossible to enforce |
| Single `[all]` extra | One-liner install | vLLM + TRT-LLM cannot coexist — `[all]` is semantically incorrect and would produce a broken environment |

**Rejected: Single `[all]` extra (2026-02-18).** vLLM and TensorRT-LLM cannot coexist in the same Python process. An `[all]` extra would install both and produce a broken environment.

---

### I2: Docker dependency model

| Option | Pros | Cons |
|--------|------|------|
| **Docker as system dependency — not pip-managed (chosen)** | Docker is infrastructure; cannot be installed via pip; tool provides informational warnings when Docker unavailable; hard fail with exact fix instructions when Docker is *required* (multi-backend studies only) | Users must install Docker separately |
| Wrap Docker in a pip helper | Hides infrastructure concern | Technically impossible (Docker requires root / system-level install); false abstraction |

Tool does NOT wrap `docker compose pull`. Users run Docker commands directly. `llem config` shows whether Docker is available.

---

### I3: New user onboarding flow

| Option | Pros | Cons |
|--------|------|------|
| **Progressive disclosure — install → try → get exact hint → install backend → run (chosen)** | Zero-config entry; guides via helpful errors; no ceremony required | User must attempt a failing run to discover the install hint |
| `llem init` setup wizard | Proactive guidance before first run | No peer tool has this (lm-eval, Zeus, MLflow, Optimum-Benchmark all omit init); adds code for a one-time UX; copy-and-edit examples are simpler |
| Required pre-flight ceremony | Explicit environment check | Forces ceremony on users who already know what they need |

**Rejected: `llem init` (2026-02-18).** No peer LLM benchmarking tool implements an init command. Shipping `experiment-example.yaml` and `study-example.yaml` as copy-and-edit templates is simpler and more portable.

---

### I4: Energy backend model

| Option | Pros | Cons |
|--------|------|------|
| **All optional, none default — `[zeus]`, `[codecarbon]` extras (chosen)** | Scientific claim integrity — energy measurement method should be an explicit user choice; base package includes raw NVML polling (via `nvidia-ml-py`, already a transitive dep) | Users must opt in to accurate energy measurement |
| Default energy backend at base install | Zero decision required | Invisible default violates scientific claim integrity; forces large dep on all users |

CO2 estimation (CodeCarbon) is explicitly decoupled from energy measurement (Zeus). Zeus measures joules/watts. CodeCarbon estimates CO2 from a separate model. They are independent extras.

---

### I5: Docker scope — when required vs optional

| Option | Pros | Cons |
|--------|------|------|
| **Docker required for multi-backend studies only; optional for single-backend (chosen)** | Correctness requirement, not a preference: vLLM + TRT cannot coexist in the same process; single-backend runs work locally without Docker | Docker requirement may surprise users mid-study |
| Docker always required | Reproducibility enforced for all runs | Blocks users without Docker access for single-backend experiments |
| Docker never required | Maximum accessibility | Cannot support multi-backend studies at all |

Single-backend studies may optionally use `runner: docker` in study.yaml or `--runner docker` for reproducible formal-mode execution. Docker is then available but not enforced.

---

### I6: CLI command name

| Option | Pros | Cons |
|--------|------|------|
| **`llem` (chosen for v2.0)** | Matches package name `llenergymeasure`; clean break from v1.x | No backwards compatibility with `lem` |
| Keep `lem` | No user migration required | Conflicts with package name; confusing abbreviation |
| Provide both `lem` and `llem` shim | Zero migration friction | Backwards-compatibility shim adds ongoing maintenance; explicit clean break preferred |

**Rejected: `lem → llem` compatibility shim (2026-02-17).** v2.0 is a clean break. No shim.

---

## Decision

We will use **pip extras per backend** with Docker as a system dependency, progressive disclosure onboarding, and no init command.

- Base install: `pip install llenergymeasure` — library + CLI, no inference backends
- Inference extras: `[pytorch]`, `[vllm]`, `[tensorrt]` — pick what you need; vLLM + TRT are mutually exclusive in one process
- Energy extras: `[zeus]`, `[codecarbon]` — additive, all optional; base package includes raw NVML polling
- Docker: system dependency; informational warning if unavailable; hard fail (with exact fix) if a multi-backend study requires it
- No `llem init`; no setup wizard; progressive disclosure via helpful errors
- CLI rename: `lem` → `llem` at v2.0; clean break, no shim
- Only one NVML session active at a time; when Zeus is active, base NVML poller yields (see [decisions/architecture.md](architecture.md))

**Rationale:** pip extras is the standard pattern across lm-eval, Zeus, and Optimum-Benchmark. The process-incompatibility of vLLM + TRT makes this a correctness requirement, not just a preference. Progressive disclosure matches the no-init pattern of all major peer tools.

---

## Consequences

**Positive:**
- Users install only what they need — base install is small
- Process-incompatibility is enforced at the packaging level, not discovered at runtime
- Scientific claim integrity: energy measurement method is always an explicit choice
- No setup ceremony — first run either succeeds or gives exact install instructions

**Negative / Trade-offs:**
- Users must discover the correct extra via a failing run (mitigated by clear error messages)
- Multi-backend study users must install and manage Docker separately
- `lem` → `llem` rename is a breaking change for existing v1.x users

**Neutral / Follow-up decisions triggered:**
- Docker orchestration design: see [decisions/docker-execution.md](docker-execution.md)
- Pre-flight error messages: see [decisions/architecture.md](architecture.md)
- `runner: docker` study field: see [designs/study-yaml.md](../designs/study-yaml.md)

---

## Install Matrix

```bash
# Base — library + CLI, no inference backends
pip install llenergymeasure

# Inference backends (pick what you need; vLLM + TRT incompatible in same process)
pip install llenergymeasure[pytorch]    # PyTorch + Transformers (most users)
pip install llenergymeasure[vllm]       # vLLM (Linux only, continuous batching)
pip install llenergymeasure[tensorrt]   # TensorRT-LLM (Ampere+ GPU required)

# Energy backends (additive, all optional)
pip install llenergymeasure[zeus]       # Zeus ZeusMonitor: accurate NVML (±5W)
pip install llenergymeasure[codecarbon] # CodeCarbon: CO2 estimation

# Future (v3.0)
pip install llenergymeasure[lm-eval]    # Quality-alongside-efficiency integration
```

## Progressive Disclosure Install Flow

```bash
# Step 1: Install base (tiny — no backends, no GPU deps)
pip install llenergymeasure

# Step 2: Try to run → helpful error
llem run --model meta-llama/Llama-3.1-8B
# Error: No inference backend installed.
#   pip install llenergymeasure[pytorch]    # PyTorch + HF Transformers
#   pip install llenergymeasure[vllm]       # vLLM (high throughput)
#   pip install llenergymeasure[tensorrt]   # TensorRT-LLM (NVIDIA-optimised)

# Step 3: Install a backend
pip install llenergymeasure[pytorch]

# Step 4: Zero-config run — works immediately
llem run --model meta-llama/Llama-3.1-8B
# Uses defaults: aienergyscore dataset, 100 prompts, batch_size=1, bf16
# Output: results/llama-3.1-8b_pytorch_2026-02-18T14-30.json

# Step 5: Config-driven (reproducible)
llem run experiment.yaml

# Step 6: Study sweep (YAML determines scope — same llem run command)
llem run batch-size-effects.yaml

# Step 7: Multi-backend study (Docker required — auto-enforced)
llem run multi-backend-study.yaml
# → pre-flight detects pytorch + vllm → requires Docker
# → Error if Docker unavailable: "Multi-backend studies require Docker.
#    Install Docker, then: docker compose pull pytorch vllm"
```

## Docker Setup (Multi-Backend Studies)

Docker images per backend ship at v2.0 alongside pip packages. Published to Docker Hub under `llenergymeasure/` org. CI rebuilds on backend releases.

```bash
# Pull images before a multi-backend study (user runs this manually)
docker compose pull pytorch vllm         # pull only what you need

# Study auto-dispatches to containers
llem run multi-backend-study.yaml
# → pytorch experiments → pytorch container
# → vllm experiments    → vllm container
# → results merged into study summary
```

Tool does NOT wrap `docker compose pull` — users run Docker commands directly. `llem config` shows whether Docker is available.

## CLI Rename

| | Before v2.0 | v2.0+ |
|-|-------------|-------|
| CLI | `lem` | `llem` |
| PyPI | `llenergymeasure` | `llenergymeasure` (unchanged) |
| Import | `import llenergymeasure` | `import llenergymeasure` (unchanged) |

No compatibility shim. v2.0 is a clean break.

## NVML Single-Session Owner

Only one NVML session active at a time. When Zeus is installed and active, base NVML poller must yield. Energy backend layer enforces a single owner. See [decisions/architecture.md](architecture.md).

---

## Related

- [decisions/architecture.md](architecture.md): NVML single-session owner; pre-flight error design
- [decisions/docker-execution.md](docker-execution.md): Docker orchestration for multi-backend studies
- [designs/cli-commands.md](../designs/cli-commands.md): `llem config` shows Docker availability
- [designs/study-yaml.md](../designs/study-yaml.md): `runner: docker` field
