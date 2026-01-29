# Technology Stack

**Analysis Date:** 2026-01-26

## Languages

**Primary:**
- Python 3.10+ - Core application, CLI, inference engines
- YAML - Configuration files for experiments
- Bash - Docker entrypoint scripts, utility commands

**Secondary:**
- Dockerfile - Multi-stage container builds for backend isolation

## Runtime

**Environment:**
- Python 3.10 (pinned in Docker `nvidia/cuda:12.4.1-runtime-ubuntu22.04`)
- CUDA 12.4.1 - GPU compute platform
- cuDNN bundled with CUDA

**Package Manager:**
- Poetry - Dependency management and publishing
- Lockfile: `poetry.lock` (present, committed)
- Alternative: pip installation via `pyproject.toml`

## Frameworks

**Core CLI:**
- Typer 0.15.0+ - CLI framework with command groups
- Click (via Typer) - Command parsing

**Configuration:**
- Pydantic 2.0+ - Config validation and domain models
- python-dotenv 1.0.0 - Environment variable loading

**Inference Backends (pluggable):**
- Transformers 4.49.0 - HuggingFace model loading
- Torch 2.5.0 - Deep learning framework (core dependency)
- Accelerate 1.4.0 (optional) - Distributed training/inference utilities
- vLLM 0.6.0+ (optional) - High-throughput LLM serving engine
- TensorRT-LLM 0.12.0+ (optional) - NVIDIA optimised inference
- PEFT 0.18.1 (optional) - Parameter-efficient fine-tuning (LoRA adapters)

**Quantization & Optimisation:**
- bitsandbytes 0.45.0 (optional) - 4-bit/8-bit quantization
- calflops 0.2.0 (optional) - FLOPs estimation
- onnxruntime-gpu 1.17.0+ (optional) - ONNX Runtime execution backend

**Data & Datasets:**
- datasets 3.0.0 - HuggingFace dataset loading
- Tokenizers (via transformers) - Fast tokenization

**Monitoring & Metrics:**
- codecarbon 2.8.0 - Energy consumption tracking
- nvidia-ml-py 12.0.0 - NVIDIA GPU monitoring (pynvml wrapper)
- loguru 0.7.0 - Structured logging

**Utilities:**
- tqdm 4.66.0 - Progress bars
- schedule 1.2.2 - Scheduled task execution
- python-jose 3.3.0 (optional API backend) - JWT tokens

## Key Dependencies

**Critical:**
- PyTorch 2.5.0 - Required for all inference, device management
- Transformers 4.49.0 - Model loading from HuggingFace Hub
- Pydantic 2.0 - Config validation, domain model definition
- Typer - CLI command registration and argument parsing

**Energy & GPU:**
- codecarbon 2.8.0 - Energy tracking via system-level APIs
- nvidia-ml-py 12.0.0 - GPU stats (power, memory, utilisation)

**Backend-Specific:**
- **PyTorch backend**: Accelerate 1.4.0 (distributed utilities)
- **vLLM backend**: vLLM 0.6.0+ (has dependency conflicts with TensorRT)
- **TensorRT backend**: tensorrt-llm 0.12.0+ (has dependency conflicts with vLLM)

**Infrastructure (optional API backend):**
- FastAPI 0.115.0 - Web framework
- Uvicorn 0.32.0 - ASGI server
- SQLAlchemy 2.0 - ORM for results database
- asyncpg 0.30.0 - PostgreSQL async driver
- Alembic 1.14.0 - Database migrations

## Configuration

**Environment Variables:**
- `HF_TOKEN` - HuggingFace API token (optional, for gated models)
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `HF_HOME` - HuggingFace cache directory (default: `/app/.cache/huggingface`)
- `LLM_ENERGY_RESULTS_DIR` - Results output directory (default: `results/`)
- `LLM_ENERGY_STATE_DIR` - Experiment state directory (default: `.state/`)
- `LLM_ENERGY_CONFIGS_DIR` - Config directory (Docker only, default: `configs/`)
- `CODECARBON_LOG_LEVEL` - CodeCarbon logging level
- `NVIDIA_DISABLE_REQUIRE` - Suppress NVIDIA driver requirement warnings
- `PIP_NO_CACHE_DIR` - Disable pip cache in Docker

**Build Configuration:**
- `pyproject.toml` - Poetry project metadata, dependencies, extras, version
- `.env.example` - Template environment variables for Docker
- `.env` - Actual environment variables (not committed)
- `Makefile` - Development tasks (format, lint, test, docs generation)
- `docker-compose.yml` - Multi-backend container orchestration

**Linting & Formatting:**
- Ruff 0.8.0+ - Code formatter and linter
  - Config: `pyproject.toml [tool.ruff]`
  - Line length: 100 characters
  - Enabled rules: E (errors), F (pyflakes), I (imports), UP (upgrades), B, SIM, RUF
- MyPy 1.0+ - Static type checker
  - Config: `pyproject.toml [tool.mypy]`
  - Mode: strict (with exceptions for untyped third-party libs)

**Testing Configuration:**
- pytest 8.0+ - Test runner
- pytest-cov 4.0+ - Coverage reporting
- Config: `pyproject.toml [tool.pytest.ini_options]`
- Test paths: `tests/` directory
- Test discovery: `test_*.py` files, `test_*()` functions

**Pre-commit:**
- pre-commit 3.0+ - Git hook framework
- Hooks defined in `.pre-commit-config.yaml`
- Auto-regenerates config docs when SSOT sources change

## Platform Requirements

**Development:**
- Python 3.10+
- NVIDIA CUDA 12.x compatible GPU (optional, for GPU testing)
- Linux recommended (vLLM & TensorRT Linux-only)
- 4GB+ RAM (8GB+ for TensorRT builds)

**Production:**
- **PyTorch backend**: Any NVIDIA GPU (V100+, A100, RTX 30xx/40xx, etc.)
- **vLLM backend**: NVIDIA GPU + Linux-only
- **TensorRT backend**: Ampere+ GPU (A100, A10, RTX 30xx/40xx, H100, L40); NOT V100, T4, RTX 20xx, GTX
  - Requires CUDA 12.x
  - Requires compute capability >= 8.0
- Docker: `nvidia/cuda:12.4.1-runtime-ubuntu22.04` base
- CPU-only mode: Supported but inference will be slow

**GPU Memory Requirements:**
- 8GB minimum (most small models)
- 16GB recommended (7B-13B models)
- 40GB+ (70B models)
- 80GB+ (70B-405B with batching/vLLM)

## Installation

**Local Development:**
```bash
# PyTorch backend (all NVIDIA GPUs)
pip install -e ".[pytorch,dev]"

# vLLM backend (Linux + NVIDIA GPU only)
pip install -e ".[vllm,dev]"

# TensorRT backend (Ampere+ GPU + Linux only)
pip install -e ".[tensorrt,dev]"
```

**Docker (Recommended):**
```bash
# Build PyTorch backend
docker compose build pytorch

# Build vLLM backend
docker compose build vllm

# Build TensorRT backend
docker compose build tensorrt
```

**CLI Entry Points:**
- `lem` - Preferred short alias
- `llenergymeasure` - Full command name
- Both resolve to `llenergymeasure.cli:app` Typer CLI

---

*Stack analysis: 2026-01-26*
