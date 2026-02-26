# Research: Deployment and Packaging Patterns for ML/AI Developer Tools

**Date:** 2026-02-17
**Source:** Research agent transcript (agent-aeecdd7.jsonl, 137KB)
**Status:** Complete -- agent produced an 18,540-character synthesis

---

## Summary

Cross-cutting research into how successful ML/AI tools handle packaging, deployment, configuration, and results management. The dominant pattern is: core library -> CLI wraps library -> web server wraps library, with pip extras for optional backends, Docker for conflicting GPU dependencies, and local-first results with optional remote sync. LLenergyMeasure's current architecture is well-aligned with industry norms.

---

## 1. Library vs CLI vs Web Separation Patterns

**The dominant pattern: core library -> CLI wraps library -> web server wraps library, with the library as the primary interface.** Nobody ships as a monolith that gets split later; the library layer is always the foundation.

### Tool-by-Tool Analysis

**MLflow** -- three-tier, single codebase:
- `mlflow` (PyPI): Full package including tracking client, server, and UI
- `mlflow-skinny` (PyPI): Lightweight client-only (no SQL, server, UI, data science deps)
- Python SDK communicates via REST API; if no tracking server configured, uses local `mlruns/` filesystem backend

**Weights & Biases** -- cleanest separation:
- `wandb` (PyPI): Python client library + CLI
- `wandb/server` (separate GitHub repo): Self-hosted server, deployed via Docker
- Client talks to server via HTTP API, works against managed cloud or self-hosted instance

**ClearML** -- three-component split across separate repos:
- `clearml` (PyPI): Python SDK
- `clearml-server` (separate repo): Backend + web UI, deployed via Docker Compose or K8s
- `clearml-agent` (separate repo): Daemon for remote execution
- Most decoupled architecture among tools surveyed

**Aim** -- bundled approach:
- Single `pip install aim`: Python SDK, CLI, and web UI (React, pre-built) all in one
- `aim up` launches web dashboard locally
- Simplest deployment but least flexible for production

**DVC** -- library-as-CLI:
- Single `dvc` package provides CLI and Python API
- Storage backends are separate packages: `dvc-s3`, `dvc-azure`, `dvc-gdrive`, `dvc-gs`, `dvc-ssh`
- No web UI component

### Summary Table

| Tool | Library | CLI | Web Server | Web UI |
|------|---------|-----|------------|--------|
| MLflow | Same package | Same package | Same package (or skinny) | Same package |
| W&B | `wandb` | `wandb` | `wandb/server` (Docker) | Server ships it |
| ClearML | `clearml` | `clearml` | `clearml-server` (Docker) | Server ships it |
| Aim | `aim` | `aim` | `aim` (bundled) | `aim` (bundled React) |
| DVC | `dvc` | `dvc` | N/A | N/A |

**Key takeaway:** The most scalable pattern (W&B, ClearML) separates the Python SDK from the server early. The library is always `pip install`-able. The server is always Docker-deployable. The CLI ships with the library. The web UI ships with the server.

---

## 2. Dependency Management for GPU Tools

**The standard pattern is pip extras, not separate packages for backends.**

### lm-eval-harness (Most Relevant Precedent)

Since December 2025, base `lm-eval` no longer includes `transformers` or `torch`:

```bash
pip install lm_eval          # Base (no backends)
pip install lm_eval[hf]      # HuggingFace
pip install lm_eval[vllm]    # vLLM
pip install lm_eval[gptq]    # GPTQ quantization
pip install lm_eval[gptq,vllm]  # Multiple backends
```

Each backend lives in a separate module under `models/`.

### PyTorch -- Custom Index URL Pattern

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip3 install torch --index-url https://download.pytorch.org/whl/cpu    # CPU-only
```

Unique to PyTorch due to massive binary sizes; most tools do not replicate this.

### DVC -- Separate Companion Packages

```bash
pip install dvc       # Core
pip install dvc-s3    # S3 support (separate package)
```

Used when backends have genuinely independent release cycles.

### Standard pyproject.toml Pattern

```toml
[project.optional-dependencies]
pytorch = ["torch>=2.0", "transformers>=4.30"]
vllm = ["vllm>=0.4"]
tensorrt = ["tensorrt-llm>=0.10"]
all = ["package[pytorch,vllm,tensorrt]"]
dev = ["pytest", "ruff", "mypy"]
```

**Key takeaway:** pip extras is the industry standard. lm-eval-harness is the closest precedent to LLenergyMeasure's architecture. Separate packages (DVC pattern) are for independent release cycles.

---

## 3. Docker vs Pip for ML Tools

### Current State (2025-2026)

- **pip for development/exploration, Docker for production/reproducibility. Both required.**
- Researchers still predominantly use `pip install` for initial exploration
- Docker adoption growing rapidly for GPU workloads driven by CUDA dependency complexity
- Majority of LLM development has shifted to Docker containers (USDSI, 2026)

### Handling Conflicting Dependencies (vLLM vs TensorRT)

**Docker is the only practical solution.** vLLM and TensorRT-LLM have fundamentally conflicting dependency trees (different PyTorch versions, different CUDA requirements, conflicting pynvml/dask-cuda versions).

NVIDIA TensorRT-LLM GitHub issue #1130 documents conflicts between `dask-cuda`, `pynvml`, `torch-tensorrt`, and `tensorrt`. Recommended approach: separate Docker containers per backend.

### Distribution Patterns

| Tool | pip | Docker | Conda |
|------|-----|--------|-------|
| vLLM | Yes (primary) | Yes (official) | No |
| TensorRT-LLM | Yes (complex) | Yes (recommended) | No |
| MLflow | Yes (primary) | Yes (server) | Yes |
| W&B | Yes (client) | Yes (server only) | Yes |
| ClearML | Yes (client) | Yes (server only) | No |

**Key takeaway:** Ship pip-installable packages for the client/library. Ship Docker images for backends with heavy/conflicting GPU dependencies. Always provide a `pip install` path.

---

## 4. Configuration Systems

**Dominant pattern: Python dataclasses/Pydantic for internal config, CLI args for invocation, YAML for persistence. All three layered.**

### vLLM (Most Relevant)

- Engine arguments defined as Python dataclasses
- CLI args for `vllm serve` command
- YAML config file support: `vllm serve --config config.yaml`
- Priority: **CLI args > config file > defaults** (same as LLenergyMeasure)

### TensorRT-LLM

- Three YAML sections: `model`, `checkpoint`, `build`
- Python API via `LLM` class for programmatic use
- `trtllm-build` CLI for engine compilation
- `trtllm-serve` CLI for serving

### Cross-Tool Comparison

| Tool | Internal Model | User Config | CLI | Priority |
|------|---------------|-------------|-----|----------|
| vLLM | Python dataclass | YAML | argparse | CLI > YAML > defaults |
| TensorRT-LLM | Python dataclass | YAML | argparse | CLI > YAML > defaults |
| lm-eval-harness | Python | YAML (tasks) | argparse | CLI > config |
| MLflow | Pydantic | YAML/env vars | Click | CLI > env > config |
| **LLenergyMeasure** | **Pydantic** | **YAML** | **Typer** | **CLI > YAML > preset > defaults** |

**Key takeaway:** LLenergyMeasure's approach is exactly aligned with industry standard. The SSOT introspection pattern is more sophisticated than most tools, which is a competitive advantage.

---

## 5. Results and Experiment Tracking Patterns

### Local-First Tools

- **MLflow:** Default local `mlruns/`. Remote tracking server is opt-in.
- **DVC:** Git-based versioning with optional remote storage.
- **Aim:** Local storage by default. Remote server optional.

### Cloud-First Tools

- **W&B:** `wandb.init()` syncs to servers by default. Offline mode secondary.
- **Neptune AI:** Cloud-first, with on-premise option.

### Result Upload Patterns

| Platform | Pattern |
|----------|---------|
| HF Open LLM Leaderboard | Users submit models (not results). HF runs evaluations centrally |
| W&B | Automatic upload via SDK during training |
| MLflow | `mlflow.log_metric()` sends to configured URI. Local by default |
| Aim | `aim sync` to push local data to remote server |

### Telemetry Best Practices

- Opt-in for data sharing, opt-out for anonymous usage telemetry
- Opt-in rates below 3% make data "statistically useless" (1984 Ventures)
- Collect minimum necessary; never collect personal info; delete raw after 90 days
- Russ Cox transparent telemetry proposal: upload only aggregate statistics, make data publicly inspectable

**Key takeaway for LLenergyMeasure:** Start local-first (already done). Add optional result upload as separate feature. Make sharing explicitly opt-in. Use `lem results push` (explicit action) rather than automatic upload.

---

## 6. Web Platform Patterns for ML Tools

### Technology Choices

| Tool | Web Framework | Frontend | Backend | Deployment |
|------|--------------|----------|---------|------------|
| W&B | Custom | React | Go + Python | Docker / managed cloud |
| MLflow | FastAPI | React | Python | `mlflow server` / Docker |
| ClearML | Custom | React | Python (APIServer) | Docker Compose / K8s |
| Aim | Custom | React | Python | `aim up` / Docker |
| Gradio | FastAPI | Svelte | Python | Embedded in app |
| Streamlit | Tornado | React | Python | `streamlit run` |

**Key observations:**
- **React dominates** for custom ML dashboards. All major experiment tracking tools use React.
- **Streamlit and Gradio** are for demos and quick prototypes, not production dashboards.
- **FastAPI is the emerging standard** for ML tool backends (2025-2026). Its Pydantic integration is natural for projects already using Pydantic.
- **Self-hosted first, managed later** is the typical evolution.

### Evolution Pattern

1. **Phase 1**: CLI tool + local storage (pip installable)
2. **Phase 2**: Add REST API server (self-hosted, Docker)
3. **Phase 3**: Add web UI (React, bundled with server)
4. **Phase 4**: Offer managed/hosted version

ClearML, W&B, and MLflow all followed this progression.

### Streamlit vs Gradio vs Custom React

| | Streamlit | Gradio | Custom React |
|--|-----------|--------|-------------|
| Best for | Internal dashboards, analytics | Model demos, non-text I/O | Production dashboards |
| Owned by | Snowflake | Hugging Face | N/A |
| Use case | Data exploration | "Share a demo link" | Complex state, real-time, multi-user |
| Used by | -- | HF Leaderboards | W&B, ClearML, MLflow, Aim |

**Key takeaway for LLenergyMeasure:** For v4.0 web platform, standard path is FastAPI backend (natural with Pydantic) + React frontend. Do not use Streamlit/Gradio for the main platform. Consider Gradio only for a quick "try it" demo.

---

## 7. Gap Analysis: LLenergyMeasure vs Industry

| Area | Industry Standard | LLenergyMeasure Current | Gap |
|------|-------------------|------------------------|-----|
| Package structure | Library core + CLI in one, server separate | Monolith package | Need clean library/CLI separation internally |
| Backend deps | pip extras | Already using pip extras | Aligned |
| Docker | Per-backend images | Docker Compose with per-backend services | Aligned |
| Config system | Pydantic + YAML + CLI, layered | Pydantic + YAML + Typer | Aligned (ahead of most) |
| Results | Local-first, optional remote | Local-first | Need optional upload path |
| Web platform | FastAPI + React, self-hosted first | Not yet started | Plan for v4.0 |
| Telemetry | Opt-in for data sharing | Not implemented | Implement as opt-in when upload comes |

---

## Sources

- [MLflow Architecture Overview](https://mlflow.org/docs/latest/self-hosting/architecture/overview/)
- [MLflow Skinny on PyPI](https://pypi.org/project/mlflow-skinny/)
- [MLflow GitHub Issue #6583 -- Separate client/server packages](https://github.com/mlflow/mlflow/issues/6583)
- [W&B Server GitHub Repository](https://github.com/wandb/server)
- [W&B Self-Managed Docs](https://docs.wandb.ai/guides/hosting/hosting-options/self-managed/)
- [ClearML Server GitHub](https://github.com/clearml/clearml-server)
- [ClearML SDK GitHub](https://github.com/clearml/clearml)
- [Aim GitHub Repository](https://github.com/aimhubio/aim)
- [DVC Installation Docs](https://dvc.org/doc/install)
- [lm-eval on PyPI](https://pypi.org/project/lm-eval/)
- [lm-eval-harness Architecture Blog](https://slyracoon23.github.io/blog/posts/2025-03-21_eleutherai-evaluation-methods.html)
- [vLLM Engine Arguments](https://docs.vllm.ai/en/latest/configuration/engine_args/)
- [TensorRT-LLM Build Configuration](https://tensorrt-llm.continuumlabs.ai/llama2-installation/trtllm-build-configuration-file)
- [TensorRT-LLM Dependency Conflict Issue #1130](https://github.com/NVIDIA/TensorRT-LLM/issues/1130)
- [vLLM vs TensorRT-LLM (Northflank)](https://northflank.com/blog/vllm-vs-tensorrt-llm-and-how-to-run-them)
- [Docker for LLM Development 2026 (USDSI)](https://www.usdsi.org/data-science-insights/top-5-docker-containers-transforming-llm-development-in-2026)
- [HF Open LLM Leaderboard Submission Docs](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/submitting)
- [Neptune AI -- Best ML Experiment Tracking Tools](https://neptune.ai/blog/best-ml-experiment-tracking-tools)
- [Open Source Telemetry (1984 Ventures)](https://1984.vc/docs/founders-handbook/eng/open-source-telemetry)
- [Transparent Telemetry (Russ Cox)](https://research.swtch.com/telemetry-opt-in)
- [FastAPI for Microservices (Talent500)](https://talent500.com/blog/fastapi-microservices-python-api-design-patterns-2025/)
- [Streamlit vs Gradio (Squadbase)](https://www.squadbase.dev/en/blog/streamlit-vs-gradio-in-2025-a-framework-comparison-for-ai-apps)
- [PyTorch Installation with uv](https://docs.astral.sh/uv/guides/integration/pytorch/)
