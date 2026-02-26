# Technology Stack

**Analysis Date:** 2026-02-05

## Languages

**Primary:**
- Python 3.10+ - All application code, requires 3.10 minimum for TensorRT-LLM compatibility

**Secondary:**
- YAML - Configuration files (experiment configs, campaign configs, user config)
- Bash - Docker entrypoint scripts, Makefile dev commands

## Runtime

**Environment:**
- Python 3.10+ (3.10/3.12 supported by TensorRT-LLM)
- CUDA 12.4.1 (via NVIDIA base Docker image)
- Virtual environment managed in `/opt/venv` (Docker) or local `.venv`

**Package Manager:**
- Poetry 2.x - Dependency management and packaging
- Lockfile: `poetry.lock` (present in repository)
- Alternative: pip with `pyproject.toml` for editable installs

## Frameworks

**Core:**
- Pydantic v2 - Configuration validation, domain models
- Typer >=0.15.0 - CLI framework with command groups
- PyTorch >=2.5.0 - Default inference backend, tensor operations
- Transformers >=4.49.0 - HuggingFace model loading and generation
- Accelerate >=1.4.0 - Distributed multi-GPU launch for PyTorch backend

**Optional Backends:**
- vLLM >=0.6.0 - High-throughput inference with PagedAttention (Linux only)
- TensorRT-LLM >=0.12.0 - Compiled inference for Ampere+ GPUs (Linux only, CUDA 12.x)
- ONNX Runtime GPU >=1.17.0 - Optional PyTorch optimization via torch.compile

**Testing:**
- pytest >=8.0 - Test runner
- pytest-cov >=4.0 - Coverage reporting

**Build/Dev:**
- Ruff >=0.8.0 - Linter and formatter (100 char line length)
- mypy >=1.0 - Type checking
- pre-commit >=3.0 - Git hooks for quality checks

## Key Dependencies

**Critical:**
- `codecarbon >=2.8.0` - Energy consumption tracking (CPU/GPU/RAM power)
- `nvidia-ml-py >=12.0.0` - GPU monitoring via NVML (replaces deprecated pynvml)
- `loguru >=0.7.0` - Structured logging throughout application
- `datasets >=3.0.0` - HuggingFace dataset loading for prompts
- `peft >=0.18.1` - LoRA adapter loading and merging
- `calflops >=0.2.0` - FLOPs estimation for PyTorch models
- `bitsandbytes >=0.45.0` - 4-bit/8-bit quantization for PyTorch

**Infrastructure:**
- `python-on-whales >=0.70` - Docker container orchestration for campaigns
- `httpx >=0.23.0` - HTTP client for webhook notifications
- `questionary >=2.0` - Interactive CLI prompts (resume, init)
- `schedule >=1.2.2` - Scheduled experiment execution
- `python-dotenv >=1.0.0` - Environment variable loading from `.env`
- `rich` (via Typer) - Rich terminal output, tables, progress bars
- `tqdm >=4.66.0` - Progress bars for inference loops
- `numpy >=1.24` - Statistics, metrics computation, bootstrap CI

**Optional (API backend):**
- `fastapi >=0.115.0` - Web API framework
- `uvicorn >=0.32.0` - ASGI server
- `sqlalchemy >=2.0` - Database ORM
- `asyncpg >=0.30.0` - Async PostgreSQL driver
- `alembic >=1.14.0` - Database migrations
- `pydantic-settings >=2.0` - Settings management
- `python-jose >=3.3.0` - JWT token handling

**Development:**
- `commitizen >=4.0` - Conventional commit enforcement
- `types-pyyaml >=6.0.12` - Type stubs for mypy

## Configuration

**Environment:**
- Environment variables loaded from `.env` file via `python-dotenv`
- User config in `~/.lem-config.yaml` or `.lem-config.yaml` (current dir)
- Key env vars:
  - `HF_TOKEN` - HuggingFace API token for private models
  - `CUDA_VISIBLE_DEVICES` - GPU device selection
  - `LLM_ENERGY_RESULTS_DIR` - Results output directory (default: `results/`)
  - `LLM_ENERGY_STATE_DIR` - Experiment state directory (default: `.state/`)
  - `LLM_ENERGY_CONFIGS_DIR` - Config directory in Docker (default: `configs/`)
  - `PUID`/`PGID` - Docker user/group ID for file ownership
  - `LLM_ENERGY_VERBOSITY` - Logging verbosity (normal/verbose/quiet)
  - `CODECARBON_LOG_LEVEL` - CodeCarbon logging level (default: warning)

**Build:**
- `pyproject.toml` - Poetry project config, tool settings (ruff, mypy, pytest)
- `.pre-commit-config.yaml` - Pre-commit hooks for quality + doc generation
- `Makefile` - Development shortcuts (format, lint, typecheck, test)

## Platform Requirements

**Development:**
- Linux (required for vLLM and TensorRT-LLM)
- NVIDIA GPU with CUDA support
- Docker + Docker Compose (optional but recommended)
- Python 3.10+ with venv
- 16GB+ GPU memory for most models
- Privileged Docker mode for NVML energy metrics

**Production:**
- Deployment: Docker Compose multi-backend setup
- Three backend images: `pytorch`, `vllm`, `tensorrt`
- Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- GPU requirements vary by backend:
  - PyTorch: Any NVIDIA GPU with CUDA
  - vLLM: NVIDIA GPU, Linux only
  - TensorRT-LLM: Ampere+ GPU (compute capability >=8.0), CUDA 12.x, Linux only

**GPU Compute Capabilities:**
- TensorRT supported: A100, A10, RTX 30xx/40xx, H100, L40
- TensorRT NOT supported: V100, T4, RTX 20xx, GTX series

---

*Stack analysis: 2026-02-05*
