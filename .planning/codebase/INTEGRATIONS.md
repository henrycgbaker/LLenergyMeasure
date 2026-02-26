# External Integrations

**Analysis Date:** 2026-02-05

## APIs & External Services

**HuggingFace Hub:**
- Model downloading and caching
  - SDK/Client: `transformers` (AutoModel, AutoTokenizer)
  - Auth: `HF_TOKEN` environment variable
  - Cache location: `HF_HOME=/app/.cache/huggingface` (Docker), `~/.cache/huggingface` (local)
  - Used by: All inference backends (pytorch, vllm, tensorrt)
  - Models: Public models or private with HF_TOKEN

**Webhook Notifications:**
- HTTP POST notifications for experiment events
  - Client: `httpx >=0.23.0`
  - Config: `notifications.webhook_url` in `.lem-config.yaml`
  - Events: Experiment completion (`on_complete`), failures (`on_failure`)
  - Timeout: 10 seconds
  - Implementation: `src/llenergymeasure/notifications/webhook.py`
  - Payload: `{"event_type": "complete|failure", "experiment_id": "...", "campaign_id": "...", "timestamp": "..."}`

**HuggingFace Datasets:**
- Dataset loading for prompts
  - SDK/Client: `datasets >=3.0.0`
  - No auth required for public datasets
  - Built-in aliases: `ai-energy-score`, `alpaca`, `sharegpt`, `gsm8k`, `mmlu`
  - Config: `BUILTIN_DATASETS` in `src/llenergymeasure/config/models.py`
  - Custom datasets via HF path or local file

## Data Storage

**Databases:**
- None (file-based storage only)
  - Results: JSON files in `results/raw/` and `results/aggregated/`
  - State: JSON files in `.state/` for experiment resumption
  - Configs: YAML files in `configs/`

**File Storage:**
- Local filesystem only
  - Results directory: `LLM_ENERGY_RESULTS_DIR` env var (default: `results/`)
  - State directory: `LLM_ENERGY_STATE_DIR` env var (default: `.state/`)
  - Model cache: `HF_HOME` env var (default: `.cache/huggingface`)
  - TensorRT engine cache: `.cache/tensorrt-engines/` (Docker volume)
  - File format: JSON (results), YAML (configs), CSV (CodeCarbon emissions)

**Caching:**
- HuggingFace model cache (persistent across runs)
- TensorRT compiled engine cache (persistent across runs)
- Docker named volumes: `lem-hf-cache`, `lem-trt-engine-cache`

## Authentication & Identity

**Auth Provider:**
- None (single-user local tool)
  - Implementation: Direct filesystem access
  - Docker: PUID/PGID env vars for file ownership mapping

**HuggingFace Token:**
- Optional token for private models
  - Storage: `HF_TOKEN` environment variable (from `.env` file)
  - Scope: Model downloading only

## Monitoring & Observability

**Error Tracking:**
- None (local logging only)

**Logs:**
- Structured logging via `loguru`
  - Configuration: `src/llenergymeasure/logging.py`
  - Verbosity levels: `quiet`, `normal`, `verbose`
  - CLI flag: `--verbose`, `--quiet`, `--json`
  - Env var: `LLM_ENERGY_VERBOSITY`
  - Format: Timestamped, coloured console output
  - Subprocess logs: Captured via `subprocess.run(capture_output=True)`

**Energy Metrics:**
- CodeCarbon emissions tracking
  - Client: `codecarbon >=2.8.0`
  - Mode: Process-level tracking (default)
  - Output: CSV file + JSON metrics
  - Backend: `src/llenergymeasure/core/energy_backends/codecarbon.py`

**GPU Metrics:**
- NVIDIA Management Library (NVML)
  - Client: `nvidia-ml-py >=12.0.0` (replaces deprecated pynvml)
  - Metrics: Power, utilization, temperature, memory
  - Sampling: Background thread via `src/llenergymeasure/core/gpu_utilisation.py`
  - Docker: Requires privileged mode for NVML access

## CI/CD & Deployment

**Hosting:**
- Self-hosted (no cloud deployment)
  - Local: Direct Python execution
  - Docker: Multi-backend compose setup

**CI Pipeline:**
- GitHub Actions (minimal)
  - Workflow: `.github/workflows/release.yml`
  - Triggers: Manual release workflow only
  - No automated testing/linting in CI

**Pre-commit Hooks:**
- Local quality enforcement via `pre-commit`
  - Config: `.pre-commit-config.yaml`
  - Hooks: ruff format/lint, mypy, trailing whitespace, YAML validation
  - Doc generation: Auto-regenerates docs when SSOT sources change
  - Branch protection: Prevents direct commits to main

## Environment Configuration

**Required env vars:**
- `PUID` - Docker user ID for file ownership (required for Docker)
- `PGID` - Docker group ID for file ownership (required for Docker)

**Optional env vars:**
- `HF_TOKEN` - HuggingFace token for private models
- `CUDA_VISIBLE_DEVICES` - GPU device selection (defaults to `0` in Docker)
- `LLM_ENERGY_RESULTS_DIR` - Results directory (default: `results/`)
- `LLM_ENERGY_STATE_DIR` - State directory (default: `.state/`)
- `LLM_ENERGY_CONFIGS_DIR` - Config directory in Docker (default: `configs/`)
- `LLM_ENERGY_VERBOSITY` - Logging verbosity (normal/verbose/quiet)
- `LLM_ENERGY_JSON_OUTPUT` - JSON output mode for machine-readable results
- `CODECARBON_LOG_LEVEL` - CodeCarbon logging level (default: warning)
- `HF_HOME` - HuggingFace cache directory
- `NVIDIA_VISIBLE_DEVICES` - NVIDIA container toolkit device visibility
- `NVIDIA_DRIVER_CAPABILITIES` - NVIDIA capabilities (compute, utility)

**Secrets location:**
- `.env` file in project root (git-ignored)
- Generated by `lem doctor` or manually created
- Pattern: `PUID=$(id -u) PGID=$(id -g)`

## Webhooks & Callbacks

**Incoming:**
- None (no API server in default CLI mode)

**Outgoing:**
- Experiment completion/failure webhooks
  - Destination: Configured via `notifications.webhook_url` in `.lem-config.yaml`
  - Method: HTTP POST with JSON payload
  - Client: `httpx` with 10s timeout
  - Events:
    - `complete`: Experiment finished successfully
    - `failure`: Experiment failed with error
  - Payload includes: `event_type`, `experiment_id`, `campaign_id`, `timestamp`, optional `data`
  - Implementation: `src/llenergymeasure/notifications/webhook.py`

## Container Orchestration

**Docker Compose:**
- Multi-backend container management
  - Client: `python-on-whales >=0.70` (Python wrapper for Docker CLI)
  - Config: `docker-compose.yml`
  - Services: `pytorch`, `vllm`, `tensorrt`, `base`
  - Strategies:
    - Ephemeral (default): `docker compose run --rm` per experiment
    - Persistent: `docker compose up -d` + `docker compose exec` for repeated experiments
  - Container lifecycle: `src/llenergymeasure/orchestration/container.py`
  - GPU routing: Auto-infers backend from config, sets `CUDA_VISIBLE_DEVICES` per container

**Campaign Execution:**
- Multi-config, multi-cycle experiment orchestration
  - Orchestrator: `src/llenergymeasure/orchestration/campaign.py`
  - State: `src/llenergymeasure/orchestration/manifest.py` (campaign manifest for resumption)
  - Subprocess management: `subprocess.run()` for Docker Compose commands
  - Thermal gaps: Configurable delays between experiments to cool GPU

## Distributed Execution

**Accelerate:**
- Multi-GPU data parallelism for PyTorch backend
  - Launch: `accelerate launch` via `src/llenergymeasure/orchestration/launcher.py`
  - Config: Automatic device map, no explicit accelerate config file
  - Backend: PyTorch only (vLLM and TensorRT manage their own parallelism)
  - Process coordination: Barrier synchronization, rank-based result aggregation

**Native Backend Parallelism:**
- vLLM: Tensor parallelism via `vllm.tensor_parallel_size`
- TensorRT: Tensor/pipeline parallelism via `tensorrt.tp_size`, `tensorrt.pp_size`
- PyTorch: Data parallelism via Accelerate `num_processes`

## Third-Party Integrations Not Used

- No cloud storage (S3, GCS, Azure Blob)
- No message queues (RabbitMQ, Kafka, Redis)
- No distributed tracing (Jaeger, Datadog, Sentry)
- No secret management services (Vault, AWS Secrets Manager)
- No container registries (Docker Hub, ECR, GCR) - local builds only
- No Kubernetes/orchestrators beyond Docker Compose

---

*Integration audit: 2026-02-05*
