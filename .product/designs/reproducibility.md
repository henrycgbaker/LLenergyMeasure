# Reproducibility Design

**Last updated**: 2026-02-25
**Source decisions**: [../decisions/reproducibility.md](../decisions/reproducibility.md)
**Status**: DRAFT — key fields confirmed; TODO items marked below

---

## Design Principle

**Capture what IS controlled. Document what IS NOT. Don't over-promise.**

Reproducibility in energy measurement is inherently probabilistic — GPU thermal state,
OS scheduler, and other processes on the machine all affect power draw. The tool can
guarantee identical workloads; it cannot guarantee identical energy readings.

---

## `EnvironmentSnapshot` Model

Captured at the start of every experiment. Stored in `ExperimentResult`.

```python
# src/llenergymeasure/domain/environment.py

from pydantic import BaseModel
import subprocess, sys
from pathlib import Path


class EnvironmentSnapshot(BaseModel):
    """Records the software environment at measurement time."""

    python_version: str           # "3.11.7"
    cuda_version: str | None      # "12.4" — None if CUDA not available; see detection priority below
    driver_version: str | None    # "550.54.15" — None if NVIDIA driver not found
    llenergymeasure_version: str  # "2.0.0"
    installed_packages: list[str] # pip freeze output (sorted)
    timestamp_utc: str            # ISO-8601 — when snapshot was taken

    # Multi-GPU support (updated 2026-02-19 — was single gpu_name/gpu_vram_gb)
    gpu_names: list[str]          # ["NVIDIA A100-SXM4-80GB", "NVIDIA A100-SXM4-80GB"] — empty if no GPU
    gpu_vram_gb: list[float]      # [80.0, 80.0] — per-device VRAM; same length as gpu_names
    gpu_count: int                # len(gpu_names); 0 if no GPU detected


def capture_environment() -> EnvironmentSnapshot:
    """Captures current environment. Called at experiment start."""
    from importlib.metadata import version as pkg_version

    python_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # CUDA version — multi-source priority (decided 2026-02-19, item 10):
    # 1. torch.version.cuda — most reliable when PyTorch is active backend
    # 2. /usr/local/cuda/version.txt — works when runtime is installed, not just toolkit
    # 3. nvcc --version — only if full CUDA toolkit is installed (rare on servers)
    # 4. None — CUDA unavailable
    cuda_ver = _detect_cuda_version()
    driver_ver = _run_or_none(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits", "--id=0"])

    # Multi-GPU support (decided 2026-02-19, item 11) — lists, not single values
    gpu_names = _detect_gpu_names()
    gpu_vram_gb = _detect_gpu_vram()

    packages = _get_installed_packages()

    return EnvironmentSnapshot(
        python_version=python_ver,
        cuda_version=cuda_ver,
        driver_version=driver_ver.strip() if driver_ver else None,
        llenergymeasure_version=pkg_version("llenergymeasure"),
        installed_packages=packages,
        timestamp_utc=datetime.utcnow().isoformat() + "Z",
        gpu_names=gpu_names,
        gpu_vram_gb=gpu_vram_gb,
        gpu_count=len(gpu_names),
    )


def _detect_cuda_version() -> str | None:
    """Multi-source CUDA version detection. Priority: torch → version.txt → nvcc → None."""
    # Source 1: torch.version.cuda (most reliable when PyTorch backend is active)
    try:
        import torch
        if torch.version.cuda:
            return torch.version.cuda  # e.g. "12.4"
    except ImportError:
        pass

    # Source 2: /usr/local/cuda/version.txt (runtime-only installs have this)
    cuda_txt = Path("/usr/local/cuda/version.txt")
    if cuda_txt.exists():
        content = cuda_txt.read_text().strip()
        # Format: "CUDA Version 12.4.0"
        if "Version" in content:
            parts = content.split()
            return parts[-1].rsplit(".", 1)[0]  # "12.4.0" → "12.4"

    # Source 3: nvcc --version (toolkit install only)
    nvcc_out = _run_or_none(["nvcc", "--version"])
    if nvcc_out:
        for line in nvcc_out.splitlines():
            if "release" in line.lower():
                # "Cuda compilation tools, release 12.4, V12.4.99"
                import re
                m = re.search(r"release (\d+\.\d+)", line)
                if m:
                    return m.group(1)

    return None


def _detect_gpu_names() -> list[str]:
    """Return list of GPU names, one per device. Empty if no NVIDIA GPU detected."""
    result = _run_or_none([
        "nvidia-smi", "--query-gpu=name", "--format=csv,noheader"
    ])
    if not result:
        return []
    return [line.strip() for line in result.splitlines() if line.strip()]


def _detect_gpu_vram() -> list[float]:
    """Return list of VRAM in GB per device. Empty if no NVIDIA GPU detected."""
    result = _run_or_none([
        "nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"
    ])
    if not result:
        return []
    try:
        return [float(v.strip()) / 1024 for v in result.splitlines() if v.strip()]
    except ValueError:
        return []


def _get_installed_packages() -> list[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True, text=True,
    )
    return sorted(result.stdout.strip().splitlines())


def _run_or_none(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() if result.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
```

---

## `random_seed` in ExperimentConfig

```python
class ExperimentConfig(BaseModel):
    random_seed: int = 42   # Controls model sampling; explicit, not hidden
```

Applies to:
- Model sampling (temperature > 0 runs)
- Warmup order randomisation (if shuffled warmup is ever added)
- Synthetic dataset generation (when `dataset.synthetic` is used — inherited as default)

Backends seed their RNG from `random_seed` at the start of inference:
```python
# Inside backend runner
import torch, random, numpy as np
torch.manual_seed(config.random_seed)
random.seed(config.random_seed)
np.random.seed(config.random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.random_seed)
```

<!-- TODO: RNG seeding for vLLM and TensorRT-LLM. These backends may not expose
     a clean RNG seed API. Confirm per-backend seeding approach before implementation.
     For greedy decoding (the default), seeding is irrelevant — document this explicitly. -->

---

## Docker Image Digest (v2.0 — later milestone)

When running via Docker, the container image digest (not just tag) is pinned in the result.
Image tags are mutable — `vllm:latest` can change. The digest is immutable.

```python
class ExperimentResult(BaseModel):
    ...
    docker_image_digest: str | None = None   # None for local runs
    # e.g. "sha256:a1b2c3..." — added in v2.0 Docker milestone, schema_version "2.0"
```

Retrieved via:
```python
result = subprocess.run(
    ["docker", "inspect", "--format={{index .RepoDigests 0}}", image_ref],
    capture_output=True, text=True,
)
digest = result.stdout.strip()
```

---

## `reproducibility_notes` Field

Stored in every `ExperimentResult` — a standard disclaimer about what the config hash does
and does not guarantee:

```python
class ExperimentResult(BaseModel):
    ...
    reproducibility_notes: str = (
        "Energy measurements have variance from thermal and scheduler effects "
        "(NVML accuracy is ±5W; percentage depends on power draw). "
        "Same config_hash guarantees identical workload; it does not guarantee identical "
        "energy readings. See environment_snapshot for the software environment at measurement time."
    )
```

This is a fixed string, not configurable. Its purpose is to appear in every published result
file so downstream consumers understand the limitations without reading the documentation.

---

## What IS Controlled

| Factor | Mechanism |
|---|---|
| Experiment configuration | `config_hash` (SHA-256 of fully-resolved ExperimentConfig) |
| Prompt set and order | Built-in datasets are pinned; JSONL loads in file order |
| Sampling randomness | `random_seed` in ExperimentConfig |
| Software environment | `environment_snapshot` (pip freeze, Python/CUDA version) |
| Hardware identity | GPU model, VRAM captured in `environment_snapshot` |
| Container content (v2.0 — Docker) | `docker_image_digest` (immutable image reference) |

## What IS NOT Controlled

| Factor | Note |
|---|---|
| GPU boost clock speed | Thermal state at measurement time |
| OS scheduler behaviour | Varies with system load |
| Other processes on the same machine | Cannot be isolated without hardware reservation |
| NUMA memory allocation | Non-deterministic under load |
| Network jitter | If model is loaded from remote filesystem |
| GPU firmware microcode | Changes with driver updates |
| GPU power delivery (PSU noise) | Electrical noise in NVML readings |

## Deferred

- Formal determinism mode (lock boost clocks, isolate CPUs) — HPC-specific, v3.x
- Provenance graph (full DAG of inputs → results) — out of scope

---

## Related

- [../decisions/reproducibility.md](../decisions/reproducibility.md): Decision rationale
- [experiment-config.md](experiment-config.md): `config_hash`, `random_seed`
- [result-schema.md](result-schema.md): `environment_snapshot`, `docker_image_digest` fields
- [dataset.md](dataset.md): Built-in dataset pinning
