# Packaging & Installation Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/installation.md](../decisions/installation.md)
**Status**: Confirmed

---

## `pyproject.toml` Structure

```toml
[project]
name = "llenergymeasure"
version = "2.0.0"
description = "LLM inference efficiency measurement — energy, throughput, FLOPs"
requires-python = ">=3.10"

# Base dependencies: library + CLI, no ML backends
dependencies = [
    "pydantic>=2.0",
    "typer>=0.9",
    "rich>=13.0",
    "pyyaml>=6.0",
    "platformdirs>=3.0",       # XDG config path resolution
    "nvidia-ml-py>=13.590.48",  # base NVML polling; v13+ for energy counters
    "pyarrow>=14.0",            # Parquet export for time-series sidecar files
]

[project.optional-dependencies]
# Inference backends (vLLM + TRT are process-incompatible — no [all])
pytorch   = ["torch>=2.0", "transformers>=5.0", "accelerate>=0.28"]
vllm      = ["vllm>=0.15"]                        # Linux only
tensorrt  = ["tensorrt-llm>=1.0"]                 # Ampere+ GPU required

# Energy backends (additive, all optional)
zeus      = ["zeus>=0.13.1"]                       # note: PyPI package is 'zeus' not 'zeus-ml'
codecarbon = ["codecarbon>=3.2.2"]  # Pydantic v2 compat: verify before shipping (CodeCarbon historically used Pydantic v1 internally — STACK.md §3)

# Quality integration (v3.0)
# lm-eval = ["lm-eval>=0.4"]   # Not yet

# Development
dev = [
    "pytest>=8.0",
    "pytest-cov",
    "pytest-mock>=3.12",
    "pytest-xdist>=3.5",
    "ruff",
    "mypy",
    "pre-commit",
    "scipy>=1.12",             # bootstrap CI, statistical tests
]

[project.scripts]
llem = "llenergymeasure.cli.app:app"    # ← clean break from `lem` at v2.0

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
# Build tool: uv (replaces Poetry at v2.0). uv reads pyproject.toml natively.
# Install: `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`

[tool.hatch.build.targets.wheel]
packages = ["src/llenergymeasure"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.mypy]
python_version = "3.10"
strict = true
```

---

## `lem` → `llem` Rename

| | Before v2.0 | v2.0+ |
|---|---|---|
| CLI entry point | `lem` | `llem` |
| PyPI name | `llenergymeasure` | `llenergymeasure` (unchanged) |
| Import | `import llenergymeasure` | `import llenergymeasure` (unchanged) |

**No alias, no shim.** v2.0 is a clean break. Users with `lem` in scripts must update to `llem`.

```toml
# pyproject.toml — single entry point
[project.scripts]
llem = "llenergymeasure.cli.app:app"
# lem was here; removed at v2.0.
```

---

## Install Matrix

```bash
# Base — library + CLI, no inference backends (~10 deps)
pip install llenergymeasure

# Inference backends (mutually exclusive in same process)
pip install llenergymeasure[pytorch]    # Most users: PyTorch + Transformers
pip install llenergymeasure[vllm]       # Linux only, continuous batching
pip install llenergymeasure[tensorrt]   # Ampere+ GPU, NVIDIA-optimised

# Energy backends (additive)
pip install llenergymeasure[zeus]       # ZeusMonitor: ±5% NVML accuracy
pip install llenergymeasure[codecarbon] # CO2 estimation

# Combined (e.g. pytorch + zeus)
pip install llenergymeasure[pytorch,zeus]
```

**Why no `[all]`**: vLLM and TensorRT-LLM cannot coexist in the same Python process
(CUDA/driver conflicts). An `[all]` extra would pull in incompatible packages.

---

## Progressive Disclosure Install Flow

```bash
# Step 1: Install base (tiny — no backends, no GPU deps)
pip install llenergymeasure

# Step 2: Try to run → helpful pre-flight error
llem run --model meta-llama/Llama-3.1-8B
# Error: No inference backend installed.
#   pytorch:   pip install llenergymeasure[pytorch]   (PyTorch + HF Transformers)
#   vllm:      pip install llenergymeasure[vllm]      (vLLM, high throughput)
#   tensorrt:  pip install llenergymeasure[tensorrt]  (TensorRT-LLM, NVIDIA-optimised)

# Step 3: Install a backend
pip install llenergymeasure[pytorch]

# Step 4: Zero-config run succeeds
llem run --model meta-llama/Llama-3.1-8B
# → defaults: aienergyscore dataset, n=100, bf16
# → output: results/llama-3.1-8b_pytorch_2026-02-19T14-30/result.json

# Step 5: Config-driven
llem run experiment.yaml

# Step 6: Study sweep (YAML determines scope — single command)
llem run study.yaml

# Step 7: Multi-backend study (Docker required — auto-enforced)
llem run multi_backend_study.yaml
# → pre-flight detects pytorch + vllm → requires Docker
# → Error if Docker unavailable:
#   "Multi-backend studies require Docker.
#    Install Docker, then: docker compose pull pytorch vllm"
```

---

## Docker Setup (Multi-Backend Studies)

Docker images per backend ship at a later v2.0 milestone. Published to GitHub Container Registry.

```bash
# Pull images before a multi-backend study (user runs this manually)
docker compose pull pytorch vllm

# Study auto-dispatches to containers (llem run detects study YAML)
llem run multi_backend_study.yaml
# → pytorch experiments → pytorch container
# → vllm experiments    → vllm container
# → results merged into study summary
```

Tool does NOT wrap `docker compose pull` — users run Docker commands directly.
`llem config` shows whether Docker is available.

**Docker Compose file** (ships with package, for image pulling convenience):
```yaml
# docker-compose.yml (published to GHCR, not the user's project)
services:
  pytorch:
    image: ghcr.io/llenergymeasure/pytorch:${LLEM_VERSION}-cuda12.4
  vllm:
    image: ghcr.io/llenergymeasure/vllm:${LLEM_VERSION}-cuda12.4
  tensorrt:
    image: ghcr.io/llenergymeasure/tensorrt:${LLEM_VERSION}-cuda12.4
```

<!-- TODO: Where does the docker-compose.yml file live? Options:
     a) Shipped inside the pip package (accessible via `python -m llenergymeasure compose`)
     b) Published separately as a GitHub release artifact
     c) Documented in README for copy-paste
     Recommendation: (b) or (c) — shipping docker-compose.yml inside a pip package is
     unusual. Confirm before Docker milestone implementation. -->

---

## Example Files (Ship with Package)

```
src/llenergymeasure/
  examples/
    experiment.yaml.example    ← template single experiment config
    study.yaml.example         ← template study with sweep
    study_multi_backend.yaml.example  ← template multi-backend study (Docker)
```

No `llem init`. Users copy and edit these. Templates are also documented in README.

---

## NVML Single-Session Owner

Only one NVML session active at a time. When Zeus is installed and active (`[zeus]` installed),
the base NVML poller must yield — they cannot run simultaneously.

```python
# src/llenergymeasure/core/energy/__init__.py
def get_active_energy_backend(config) -> EnergyBackend:
    if _zeus_available():
        return ZeusBackend()   # Zeus owns NVML session
    return NVMLBackend()       # base poller owns session
```

Both `zeus` and `codecarbon` are independent optional extras. Installing both is valid —
they serve different purposes (accuracy vs CO2 estimation).

---

## Peer Reference

| Tool | Base install | Extras pattern | `[all]` |
|---|---|---|---|
| lm-eval | ~12 deps | `[hf,vllm,...]` | ✓ (backends compatible) |
| Zeus | GPU + torch | N/A (single purpose) | N/A |
| optimum-benchmark | Hydra + torch | `[onnxruntime,tensorrt,...]` | ✗ |
| CodeCarbon | Minimal | N/A | N/A |
| LLenergyMeasure | ~10 deps | `[pytorch,vllm,...]` | ✗ (incompatible backends) |

---

## Related

- [../decisions/installation.md](../decisions/installation.md): Decision rationale
- [architecture.md](architecture.md): Module layout (what goes in the package)
- [cli-commands.md](cli-commands.md): `llem` CLI entry point
- [docker-execution.md](docker-execution.md): Docker image publishing (v2.0 Docker milestone)
