# External Integrations

**Analysis Date:** 2026-01-26

## APIs & External Services

**HuggingFace Hub:**
- Model loading from HuggingFace Hub
  - SDK: `transformers` 4.49.0
  - Auth: `HF_TOKEN` environment variable (optional, required for gated models)
  - Implementation: `src/llenergymeasure/core/model_loader.py` loads models via `AutoModelForCausalLM.from_pretrained()`
  - Adapter loading: PEFT (LoRA) via `from_pretrained()` with `peft_model_id`

**HuggingFace Datasets:**
- Dataset loading for experiment prompts
  - SDK: `datasets` 3.0.0
  - Datasets loaded from HuggingFace Hub
  - Built-in curated datasets: `alpaca`, `gsm8k`, `mmlu`, `openorca`, `ultrafeedback` (defined in `src/llenergymeasure/config/models.py`)
  - Custom datasets: Load via `--dataset custom --dataset-source <path>`
  - Implementation: `src/llenergymeasure/core/dataset_loader.py` uses `load_dataset()`
  - Column auto-detection for prompt/text fields

**NVIDIA Container Toolkit:**
- GPU access in Docker
  - Config: `docker-compose.yml` with `deploy.resources.reservations.devices` (nvidia GPU driver)
  - Environment: `NVIDIA_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`
  - Health check: CUDA availability tested on container startup

## Data Storage

**Databases:**
- **File System (Primary)**: JSON files for all results
  - Path: Results directory (configurable via `LLM_ENERGY_RESULTS_DIR`)
  - Structure:
    - `results/raw/{experiment_id}/process_N.json` - Per-process raw results
    - `results/aggregated/{experiment_id}.json` - Aggregated experiment results
  - Format: Pydantic model serialization (`.model_dump_json()`)
  - Implementation: `src/llenergymeasure/results/repository.py` FileSystemRepository

**Experiment State:**
- Path: `.state/` directory (configurable via `LLM_ENERGY_STATE_DIR`)
- Format: JSON files for experiment resumption tracking
- Completion markers: `.completed_{process_index}` files
- Implementation: `src/llenergymeasure/state/experiment_state.py`

**Cache:**
- HuggingFace model cache: `/app/.cache/huggingface` (Docker) or `~/.cache/huggingface` (local)
  - Configurable via `HF_HOME` environment variable
  - Persistent across runs (Docker named volume `hf-cache`)

**File Storage (Optional API backend):**
- PostgreSQL support via SQLAlchemy 2.0
  - Driver: asyncpg 0.30.0
  - ORM: SQLAlchemy with async support
  - Migrations: Alembic 1.14.0
  - Status: Optional (API extras only)

## Authentication & Identity

**Auth Provider:**
- Custom: HuggingFace token-based (no central auth system)
  - Implementation: Environment variable `HF_TOKEN` passed to `transformers` library
  - Used for accessing gated models on HuggingFace Hub

**Optional API Auth (Optional API backend):**
- JWT tokens via python-jose 3.3.0
- Cryptography support included
- Status: Only if API extras installed

## Monitoring & Observability

**Energy Tracking:**
- CodeCarbon 2.8.0
  - Measures CPU, GPU, RAM power consumption
  - Tracks CO2 emissions by grid carbon intensity
  - Implementation: `src/llenergymeasure/core/energy_backends/codecarbon.py` CodeCarbonBackend
  - Backend protocol: `src/llenergymeasure/protocols.py` EnergyBackendProtocol
  - Config: `measure_power_secs=1`, `tracking_mode="process"`
  - Graceful fallback: Returns empty metrics if unavailable (common in containers without NVML access)

**GPU Monitoring:**
- nvidia-ml-py 12.0.0 (pynvml wrapper)
  - Real-time GPU utilisation sampling via NVIDIA Management Library
  - Metrics collected: GPU power, memory, utilisation percentage
  - Implementation: `src/llenergymeasure/core/gpu_utilisation.py` GPUUtilisationSampler
  - Graceful degradation: If unavailable, GPU metrics set to null
  - Background thread: Runs async GPU sampler during inference

**Logging:**
- loguru 0.7.0
  - Structured logging to console and files
  - Configuration: `src/llenergymeasure/logging.py` sets up loguru
  - Log levels controlled per module
  - Output format: Timestamp, level, name, message

**Metrics Collection:**
- FLOPs estimation: calflops 0.2.0 (optional)
  - Computes theoretical floating-point operations
  - Fallbacks: Architecture-based and parameter-count estimation
  - Implementation: `src/llenergymeasure/core/flops.py`

## CI/CD & Deployment

**Hosting:**
- Local execution via CLI or Docker Compose
- Experimental: FastAPI + Uvicorn web API (optional, not primary)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/`)
- Pre-commit hooks via pre-commit framework
  - Auto-generates config documentation
  - Ruff format/lint checks
  - MyPy type checking

**Containerisation:**
- Docker Compose with multi-backend support
  - Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
  - Backend services: pytorch, vllm, tensorrt (mutually exclusive)
  - Volume management: Named volumes for `.cache/huggingface` and `.state`
  - Environment: `docker-compose.yml` defines all backend services
  - PUID/PGID support for host permission mapping

## Environment Configuration

**Required env vars (for functionality):**
- `HF_TOKEN` - HuggingFace API token (optional, for gated models)
- `CUDA_VISIBLE_DEVICES` - GPU device selection (optional)

**Critical env vars (Docker):**
- `PUID`, `PGID` - Host user/group IDs (required for container permission mapping)
- `HF_HOME` - HuggingFace cache path (default: `/app/.cache/huggingface`)
- `LLM_ENERGY_RESULTS_DIR` - Results output directory (default: `results/`)
- `LLM_ENERGY_STATE_DIR` - State directory (default: `.state/`)
- `LLM_ENERGY_CONFIGS_DIR` - Config directory (default: `configs/`)

**Internal env vars (Docker):**
- `CODECARBON_LOG_LEVEL` - CodeCarbon verbosity (set to `warning` by default)
- `NVIDIA_DISABLE_REQUIRE` - Suppress NVIDIA warnings in containers
- `PIP_NO_CACHE_DIR` - Disable pip cache during builds

**Secrets location:**
- `.env` file (not committed, created by `setup.sh`)
- Environment variables passed at runtime
- No hardcoded secrets in code

## Webhooks & Callbacks

**Incoming:**
- None - CLI-driven tool, no webhook endpoints

**Outgoing:**
- None - Results saved to local filesystem only

## Data Format & Export

**Result Export Formats:**
- JSON (primary)
  - Pydantic model serialization
  - Location: `results/raw/` and `results/aggregated/`
- CSV (optional)
  - Exported via `src/llenergymeasure/results/exporters.py` CSVExporter
  - Used for analysis in external tools

## Cross-Backend Integration Points

**Inference Backend Protocol:**
- All backends implement `InferenceBackendProtocol` in `src/llenergymeasure/core/inference_backends/protocols.py`
- Backends: PyTorch, vLLM, TensorRT
- Lazy loading via `get_backend()` factory in `src/llenergymeasure/core/inference_backends/__init__.py`
- CUDA management negotiation: Each backend declares `CudaManagement` (TORCH or BACKEND)

**Energy Backend Protocol:**
- All energy backends implement `EnergyBackendProtocol` in `src/llenergymeasure/protocols.py`
- Currently: CodeCarbon only
- Extensible for other energy tracking methods

## Model Loading Pipeline

**Integrations Used:**
1. HuggingFace Transformers - Load base model (`model_name`)
2. PEFT - Load LoRA adapter if `adapter` specified
3. BitsAndBytes - Apply quantization if `load_in_4bit` or `load_in_8bit`
4. Accelerate - Distribute model across devices if parallelism enabled

**Implementation:** `src/llenergymeasure/core/model_loader.py` HuggingFaceModelLoader

---

*Integration audit: 2026-01-26*
