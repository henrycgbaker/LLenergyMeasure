# Research: MLflow Product Architecture

**Date:** 2026-02-17
**Source:** Research agent transcript (agent-a73bbb2.jsonl, 132KB)
**Status:** Complete -- agent produced a 13,744-character synthesis

---

## Summary

MLflow is a library-first ML lifecycle platform with a three-layer architecture (Library -> CLI -> Web UI) sharing a single codebase. Its store abstraction and URI-based routing enable a zero-config local start that scales to team servers and managed enterprise deployments without code changes. MLflow's architecture is the most relevant precedent for LLenergyMeasure's planned evolution from CLI tool to web platform.

---

## 1. The Three-Layer Model

### Layer 1 -- Python Library (the core)

The `mlflow` Python package is the foundation. Everything else is built on top of it.

**Fluent API** (stateful, convenience wrappers):
```python
import mlflow

mlflow.start_run()
mlflow.log_metric("loss", 0.5)
mlflow.log_param("lr", 0.01)
mlflow.end_run()
```
Maintains an active run context via a thread-local stack.

**Client API** (stateless, explicit):
```python
from mlflow import MlflowClient

client = MlflowClient()
run = client.create_run(experiment_id="1")
client.log_metric(run.info.run_id, "loss", 0.5)
```
Requires run IDs for every operation.

**Store abstraction** -- `AbstractStore` with pluggable backends:
- `FileStore` -- local filesystem (`./mlruns`)
- `SqlAlchemyStore` -- PostgreSQL, MySQL, SQLite
- `RestStore` -- HTTP to remote tracking server
- `DatabricksTracingRestStore` -- Databricks integration

The Fluent API wraps `MlflowClient`, which delegates to `TrackingServiceClient`, which resolves a tracking URI to select the appropriate store backend. **The same client code works identically whether writing to a local directory or a remote server -- only the URI changes.**

### Layer 2 -- CLI

The CLI (`mlflow` command) is a **thin wrapper over the library**:

| Command | Purpose |
|---------|---------|
| `mlflow server` | Launch tracking server (FastAPI) |
| `mlflow experiments` | Create, search, manage experiments |
| `mlflow runs` | Create, delete, describe runs |
| `mlflow artifacts` | Upload, list, download artifacts |
| `mlflow models` | Serve, predict, build Docker images |
| `mlflow deployments` | Deploy to SageMaker, Databricks, etc. |
| `mlflow db` | Database schema migrations |

The CLI communicates via `MLFLOW_TRACKING_URI`. Without it set, operations default to local `./mlruns` filesystem storage.

### Layer 3 -- Web UI

The web UI is **bundled inside the tracking server**. `mlflow server` serves both the REST API and the React frontend from the same FastAPI process. The UI consumes the same REST API that the CLI and library use. There is no separate web package.

---

## 2. Store Abstraction and URI-Based Routing

The critical architectural insight: tracking URI resolution is the routing mechanism.

```
User code / CLI / Web UI
        |
    MlflowClient
        |
  TrackingServiceClient
        |
  _resolve_tracking_uri()
        |
  +-----+-----+----------+
  |           |            |
FileStore  SqlAlchemy   RestStore
(local)    Store (DB)   (HTTP -> server)
```

| URI Scheme | Backend | Use Case |
|------------|---------|----------|
| `file://` or empty | `FileStore` | Local `./mlruns` directory |
| `sqlite:///`, `postgresql://`, `mysql://` | `SqlAlchemyStore` | Direct DB access |
| `http://`, `https://` | `RestStore` | Remote tracking server |
| `databricks://` | `DatabricksTracingRestStore` | Managed Databricks |

When using `RestStore`, the client makes HTTP requests to the tracking server. The server itself uses `SqlAlchemyStore` or `FileStore` internally. The server is effectively a **REST proxy** over the store abstraction.

---

## 3. Package Architecture

### Two PyPI Packages from One Monorepo

| Package | Size | Purpose |
|---------|------|---------|
| `mlflow` | ~9.7 MB | Full: tracking client, server, UI, model flavours |
| `mlflow-skinny` | Much smaller | Tracking client only -- no SQL, server, UI, data science deps |
| `mlflow-tracing` | 95% smaller than full | Production tracing SDK only (GenAI) |

**pip extras** for optional dependencies:
- `mlflow[extras]` -- general
- `mlflow[db]` -- database backend
- `mlflow[genai]` -- generative AI
- `mlflow[gateway]` -- AI gateway
- `mlflow[auth]` -- authentication
- `mlflow[databricks]` -- Databricks integration
- `mlflow[langchain]`, `mlflow[mcp]`, `mlflow[mlserver]`, `mlflow[sqlserver]`, `mlflow[aliyun-oss]`, `mlflow[jfrog]`

**Key insight:** `mlflow-skinny` retains the full tracking/logging client but strips the server, UI, and heavy ML dependencies. Users who only need to **log** metrics install skinny; users who need to **view** results or **serve** models install full.

Requires Python >= 3.10.

---

## 4. Tracking Server Architecture

Started via `mlflow server`:

```bash
mlflow server --host 0.0.0.0 --port 5000 \
  --backend-store-uri postgresql://user:pass@db:5432/mlflow \
  --artifacts-destination s3://my-bucket/mlflow-artifacts
```

**Components:**
- **Backend Store** -- metadata (experiments, runs, params, metrics, tags) in SQLAlchemy-compatible DB
- **Artifact Store** -- large binary files (model weights, images) in S3/GCS/Azure Blob/local filesystem
- **REST API** -- all endpoints under `/api/2.0/mlflow/`
- **Web UI** -- served from the same process

**REST API design:**
- Endpoint pattern: `POST /api/2.0/mlflow/experiments/create`, `GET /api/2.0/mlflow/experiments/get`
- Content-Type: `application/json`
- Protocol Buffers define the API contract; JSON for serialisation
- Batch endpoint: `log_batch` accepts up to 1,000 metrics, 100 params, 100 tags per request (1 MB limit)
- Pagination via `max_results` and `page_token`
- Search with SQL-like filter expressions on metrics and params

**Three deployment modes:**
1. **Full server** -- serves both tracking API and artifact proxy
2. **Tracking only** (`--no-serve-artifacts`) -- metadata only, clients access artifacts directly
3. **Artifacts only** (`--artifacts-only`) -- dedicated artifact proxy, separating heavy file I/O

**Security (v3.5.0+):**
- DNS rebinding protection, CORS controls, clickjacking prevention
- Auth via `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` or bearer tokens
- Production: deploy behind NGINX/Apache for TLS

---

## 5. Deployment Model Evolution

MLflow supports a **progressive deployment spectrum**:

| Stage | Mode | Infrastructure | Users |
|-------|------|---------------|-------|
| Local | No server, `./mlruns` | None | Solo developer |
| Local + DB | SQLite backend | SQLite file | Solo, more reliable |
| Team server | `mlflow server` on a VM | PostgreSQL + S3 | Small team |
| Production | Behind reverse proxy, separate artifact server | DB cluster + object storage | Organisation |
| Managed | Databricks Managed MLflow | Fully managed | Enterprise |

**Historical timeline:**
- June 2018 -- MLflow open-sourced by Databricks (4 components: Tracking, Projects, Models, Registry)
- March 2019 -- Databricks Managed MLflow enters public preview
- 2020s -- 10M monthly downloads, 850+ contributors
- 2024 -- Donated to Linux Foundation as top-level Apache project
- 2025-2026 -- MLflow 3.x with GenAI focus (tracing, evaluation, AI Gateway). Currently at v3.10.0rc0

The managed Databricks version adds: Unity Catalog governance, automatic notebook revision capture, enterprise access controls, auto-scaled tracking server, and Workflows integration. **The API is identical** -- `mlflow.log_metric()` works the same on local, self-hosted, and Databricks.

---

## 6. API Design Summary

### Two-Tier Python API

| API | Style | When to use |
|-----|-------|-------------|
| Fluent API (`mlflow.log_metric()`) | Stateful, implicit active run | Interactive scripts, notebooks |
| Client API (`MlflowClient.log_metric(run_id, ...)`) | Stateless, explicit run ID | Multi-threaded, programmatic |

The Fluent API is literally thin wrappers that create an `MlflowClient` instance and target the active run from a thread-local stack.

### Store Abstraction

The `AbstractStore` interface defines the contract. Implementations are selected by URI scheme through a registry pattern (`TrackingStoreRegistryWrapper`):
- Library users never import a specific store -- they set a URI string
- New backends can be added by registering a store class for a new URI scheme
- The server is just another store consumer -- it receives REST calls and delegates to `SqlAlchemyStore`

---

## 7. What Drove Adoption -- Key Architectural Decisions

Based on Linux Foundation analysis and architectural evidence:

1. **Zero-infrastructure start, gradual upgrade path.** `pip install mlflow` + `mlflow.log_metric("loss", 0.5)` writes to `./mlruns`. No server, no database, no configuration. Zero friction to start.

2. **Library-first, not platform-first.** MLflow is a Python library you call from your training script, not a platform you deploy first. The tracking server is optional infrastructure, not a prerequisite.

3. **The store abstraction / URI-based routing.** `MLFLOW_TRACKING_URI` is the entire configuration surface. Changing from local to remote requires changing one string -- no code changes.

4. **Open interface design.** REST API + CLI + Python library all expose the same operations. No proprietary protocols.

5. **Progressive feature adoption.** Four components (Tracking -> Models -> Registry -> Projects) form a natural adoption ladder. Most users start with Tracking alone.

6. **Autologging as an on-ramp.** `mlflow.autolog()` automatically captures metrics from popular frameworks with zero manual instrumentation.

7. **The skinny client package.** `mlflow-skinny` means tracking can be added to production scripts without pulling in Flask, SQLAlchemy, React, and dozens of other dependencies.

---

## 8. Relevance to LLenergyMeasure

| MLflow Pattern | LLenergyMeasure Equivalent |
|---------------|---------------------------|
| Library core with `AbstractStore` | Results repository with pluggable backends |
| Fluent API + Client API | Could have simple `lem.log_result()` + explicit `LemClient` |
| `MLFLOW_TRACKING_URI` routing | `LEM_SERVER_URI` -- same code writes locally or remotely |
| `mlflow server` (FastAPI + UI) | Future web platform (v4.0) served from same process |
| `mlflow-skinny` | A `llenergymeasure-client` package for submission-only |
| REST API `/api/2.0/mlflow/` | `/api/v1/lem/` for the leaderboard server |
| Local `./mlruns` default | Local results directory (already exists) |
| Zero-config start -> server -> managed | CLI-only -> self-hosted server -> central leaderboard |

**Key architectural lesson:** Build the store abstraction and URI-based routing first (v2.0/v3.0), and the web layer becomes a natural extension rather than a rewrite.

---

## Sources

- [MLflow Architecture Overview](https://mlflow.org/docs/latest/self-hosting/architecture/overview/)
- [MLflow CLI Reference](https://mlflow.org/docs/latest/cli.html)
- [MLflow REST API](https://mlflow.org/docs/latest/rest-api.html)
- [MLflow Tracking Server](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [Client and Fluent API (DeepWiki)](https://deepwiki.com/mlflow/mlflow/2.1-client-and-fluent-api)
- [Server and REST API (DeepWiki)](https://deepwiki.com/mlflow/mlflow/2.3-server-and-rest-api)
- [PyPI mlflow](https://pypi.org/project/mlflow/)
- [PyPI mlflow-skinny](https://pypi.org/project/mlflow-skinny/)
- [MLflow Extra Dependencies](https://github.com/mlflow/mlflow/blob/master/EXTRA_DEPENDENCIES.rst)
- [Databricks Managed MLflow](https://www.databricks.com/product/managed-mlflow)
- [Introducing MLflow (2018)](https://www.databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html)
- [10 MLflow Features to 10M Downloads](https://www.linuxfoundation.org/blog/10-mlflow-features-to-10-million-downloads)
- [MLflow Overview](https://mlflow.org/docs/latest/introduction/index.html)
