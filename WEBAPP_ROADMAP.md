# Web Interface Strategy

Strategic planning document for building a web-based UI/webapp to complement the CLI tool.

> **Planning Reference:** See `.claude/plans/ancient-moseying-quokka.md` for the original planning session and decision rationale.

---

## Executive Summary

Build a **community-facing benchmarking platform** with:
- Public leaderboard for LLM efficiency metrics
- Remote experiment execution on self-hosted GPU infrastructure
- React + FastAPI full-stack architecture

**Target Users:** ML researchers, practitioners, and the open-source community

---

## Table of Contents

1. [Current State](#current-state)
2. [Framework Options](#framework-options)
3. [Deployment Platforms](#deployment-platforms)
4. [Existing Inspiration](#existing-inspiration)
5. [Chosen Architecture](#chosen-architecture)
6. [Implementation Roadmap](#implementation-roadmap)
7. [Technical Specifications](#technical-specifications)
8. [Open Questions](#open-questions)
9. [Sources](#sources)

---

## Current State

### What We Have

| Component | Status | Notes |
|-----------|--------|-------|
| CLI tool | Production | `lem experiment` |
| Pydantic models | Production | All data validated, easily serialisable |
| CSV/JSON export | Production | `ResultsExporter` class |
| File-based results | Production | `results/raw/`, `results/aggregated/` |
| Visualisation | None | All output is terminal-based (rich tables) |

### Key Data to Visualise

| Metric | Type | Insight |
|--------|------|---------|
| `tokens_per_joule` | Primary KPI | Efficiency (higher = better) |
| `throughput_tokens_per_sec` | Performance | Speed |
| `total_energy_joules` | Cost | Resource consumption |
| `flops_per_second` | Compute | Hardware utilisation |
| `peak_memory_mb` | Resource | Memory footprint |
| `time_to_first_token_ms` | Latency | User experience |
| `latency_per_token_ms` | Latency | Generation speed |

### User Workflows to Support

1. **Run experiments** - Configure params, launch, monitor progress
2. **Explore results** - Browse, filter, sort experiments
3. **Compare** - Side-by-side across models/backends/configs
4. **Analyse** - Charts, statistical summaries, trends
5. **Export** - Download data, generate reports, API access

---

## Framework Options

### Tier 1: Python-Only (Fastest to Build)

#### Streamlit

**Best for:** Interactive dashboards, data exploration, custom UI

| Pros | Cons |
|------|------|
| Rich widget ecosystem | Reruns entire script on interaction |
| Native Plotly/Altair/Matplotlib | Limited scalability for complex apps |
| Large community, excellent docs | No built-in auth for Streamlit Cloud |
| Good for multi-page apps | Memory-hungry for large datasets |

- **HF Spaces support:** Via Docker SDK
- **Effort:** Low (days to working prototype)

#### Gradio

**Best for:** ML model demos, simple interfaces

| Pros | Cons |
|------|------|
| Native HuggingFace integration | Less UI flexibility |
| Built-in share links | Smaller ecosystem than Streamlit |
| Great for input-output workflows | Less suited for dashboards |
| Runs in Jupyter notebooks | Share links expire in 7 days |

- **HF Spaces support:** Native (first-class)
- **Effort:** Low (days to working prototype)

#### Plotly Dash

**Best for:** Enterprise dashboards, production apps

| Pros | Cons |
|------|------|
| Callback-based (efficient updates) | Steeper learning curve |
| Enterprise features available | Verbose compared to Streamlit |
| Better at scale | Less "magical" than Streamlit |
| Full control over layout | Smaller community |

- **HF Spaces support:** Via Docker
- **Effort:** Medium

### Tier 2: Full-Stack (Maximum Flexibility)

#### FastAPI + React

**Best for:** Production apps, complex interactions, real-time updates

| Pros | Cons |
|------|------|
| Full architectural control | Requires frontend expertise |
| Real-time via WebSocket/SSE | More code to maintain |
| Best performance at scale | Longer development time |
| Can reuse existing Pydantic models | Separate deployment concerns |

- **HF Spaces support:** Via Docker
- **Effort:** High (weeks to production-ready)

### Framework Decision Matrix

| Goal | Best Choice |
|------|-------------|
| Quick HF Spaces demo | Gradio |
| Interactive dashboard prototype | Streamlit |
| Production multi-user app | Dash or FastAPI+React |
| Maximum code reuse (Pydantic) | FastAPI backend + any frontend |

---

## Deployment Platforms

### HuggingFace Spaces

| Feature | Detail |
|---------|--------|
| Free tier | 2 CPU cores, 16GB RAM |
| GPU options | Paid tiers available |
| SDKs | Gradio (native), Streamlit (Docker), Docker (custom) |
| Integration | Native with HF ecosystem, models, datasets |
| Sharing | Public by default, private with paid |

**Why start here:** Zero-friction deployment, ML-native audience, free tier sufficient for demos.

### Railway

| Feature | Detail |
|---------|--------|
| Pricing | $5/mo hobby tier, usage-based |
| Free trial | $5 credit |
| Strengths | No cold starts, git-based deploys |
| Best for | Side projects, prototypes |

### Render

| Feature | Detail |
|---------|--------|
| Free tier | Yes, but sleeps after 15 min inactivity |
| Paid | From $19/mo |
| Strengths | Good scaling path to production |
| Caveats | Free Postgres deleted after 90 days |

### Self-Hosted / Cloud VMs

| Feature | Detail |
|---------|--------|
| Control | Full |
| Cost | Variable (can be cheapest at scale) |
| Best for | GPU-heavy workloads, custom infra |

---

## Existing Inspiration

### LLM-Perf Leaderboard

**URL:** https://huggingface.co/spaces/optimum/llm-perf-leaderboard

Directly relevant - measures **exact same metrics**: energy (via CodeCarbon), throughput, latency, memory.

**Key features:**
- Filter by hardware/backend/optimisation
- Sort by any metric
- Comparison view
- Standardised benchmarking pipeline (Optimum-Benchmark)

### Open LLM Leaderboard

**URL:** https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard

Model quality evaluation (not efficiency), but excellent UX reference.

**Key features:**
- Community submissions
- Model comparator tool
- Detailed per-benchmark breakdowns
- Over 2M unique visitors

### Artificial Analysis LLM Leaderboard

**URL:** https://huggingface.co/spaces/ArtificialAnalysis/LLM-Performance-Leaderboard

Commercial API benchmarking with focus on cost/performance.

**Key features:**
- Cost comparisons across providers
- Quality vs speed tradeoffs
- Time-series data

---

## Chosen Architecture

### Requirements Summary

| Requirement | Choice |
|-------------|--------|
| Audience | Community/public (like Open LLM Leaderboard) |
| Execution | Remote job submission from UI |
| GPU Compute | Self-hosted cluster |
| Frontend | React + Vite |
| Scope | Full platform |

### System Architecture

```
+-------------------------------------------------------------------+
|                    Frontend (React + Vite)                        |
|  +- Public Leaderboard (sortable, filterable)                     |
|  +- Model Explorer (search, compare, detail views)                |
|  +- Analysis Dashboard (charts, statistical viz)                  |
|  +- Experiment Builder (config form with validation)              |
|  +- Submit Benchmark (upload or run remotely)                     |
|  +- User Profiles (submissions, saved comparisons)                |
|  +- Export/API Access                                             |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                    FastAPI Backend                                |
|  +- /api/benchmarks - CRUD, search, filter                        |
|  +- /api/models - model metadata, aggregations                    |
|  +- /api/compare - comparison endpoints                           |
|  +- /api/jobs - submit, status, cancel                            |
|  +- /api/users - auth, profiles, submissions                      |
|  +- Reuses existing Pydantic models from CLI tool                 |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                    Job Execution Layer                            |
|  +- Job Queue (Celery + Redis)                                    |
|  +- GPU Workers (self-hosted cluster)                             |
|  +- Worker Agent (polls queue, runs CLI tool)                     |
|  +- Result Ingestion Pipeline                                     |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                    Data Layer                                     |
|  +- PostgreSQL (experiments, users, metadata)                     |
|  +- S3/MinIO (raw result files, large artifacts)                  |
|  +- Redis (job queue, caching, sessions)                          |
|  +- Optional: HF Dataset mirror for community access              |
+-------------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------------+
|                    Infrastructure                                 |
|  +- Railway/Render/Fly.io (API server)                            |
|  +- Vercel/Cloudflare Pages (React frontend)                      |
|  +- Self-hosted GPU cluster (workers)                             |
|  +- GitHub Actions (CI/CD)                                        |
+-------------------------------------------------------------------+
```

### Self-Hosted GPU Worker Architecture

Workers on your infrastructure connect outbound to the cloud API:

```
+-------------------------------------------------------------------+
|                     Cloud (Public)                                |
|  +------------+  +------------+  +------------+                   |
|  |  Frontend  |  |  FastAPI   |  |   Redis    |                   |
|  |  (Vercel)  |--|  (Railway) |--|  (Railway) |                   |
|  +------------+  +------------+  +-----+------+                   |
+----------------------------------------|---------------------------+
                                         | (job queue)
                                         v
+-------------------------------------------------------------------+
|                  Your Infrastructure                              |
|                                                                   |
|  +--------------------------------------------------------+      |
|  |              Worker Agent (Python)                      |      |
|  |  +- Polls Redis for pending jobs                        |      |
|  |  +- Runs lem CLI                         |      |
|  |  +- Uploads results to API                              |      |
|  |  +- Reports progress via WebSocket                      |      |
|  +--------------------------------------------------------+      |
|                           |                                       |
|           +---------------+---------------+                       |
|           v               v               v                       |
|      +--------+      +--------+      +--------+                   |
|      | GPU 0  |      | GPU 1  |      | GPU N  |                   |
|      | Worker |      | Worker |      | Worker |                   |
|      +--------+      +--------+      +--------+                   |
+-------------------------------------------------------------------+
```

**Worker Agent Responsibilities:**
- Authenticate with API using service token
- Poll Redis queue for jobs (or use Celery worker)
- Download experiment config
- Execute `lem experiment <config>`
- Stream logs to API (real-time progress)
- Upload results on completion
- Handle failures, retries, timeouts

**Security Considerations:**
- Workers connect outbound only (no inbound ports needed)
- API tokens rotated periodically
- Jobs run in isolated containers (Docker)
- Resource limits enforced per job

---

## Implementation Roadmap

### Phase 1: Core Platform

**Scope:**
- FastAPI backend with PostgreSQL
- React frontend with leaderboard + detail views
- Upload results (no remote execution yet)
- Basic auth (GitHub OAuth)
- Deploy to Railway + Vercel

**Deliverables:**
- Public leaderboard page
- Experiment detail view
- Upload JSON results
- User registration

### Phase 2: Analysis Features

**Scope:**
- Interactive comparison charts (Plotly)
- Statistical visualisations (multi-cycle data)
- Advanced filtering and search
- Shareable comparison links
- API documentation

**Deliverables:**
- Efficiency scatter plot
- Model comparison view
- Filter by hardware/backend/precision
- OpenAPI docs + SDK

### Phase 3: Remote Execution

**Scope:**
- Job queue infrastructure
- GPU worker integration
- Config builder UI
- Real-time progress updates (WebSocket)
- Cost estimation before submission

**Deliverables:**
- "Run Benchmark" feature
- Progress monitoring
- Queue management UI

### Phase 4: Community Features

**Scope:**
- User profiles and submission history
- Moderation workflow for submissions
- Hardware standardisation
- HF Dataset export for research
- Embedding/widget for papers

**Deliverables:**
- User dashboard
- Admin moderation panel
- Standard hardware profiles
- Embeddable leaderboard widget

---

## Technical Specifications

### Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frontend | React + Vite | Mature ecosystem, excellent tooling, large community |
| Backend | FastAPI | Pydantic integration, async support, auto-docs |
| Database | PostgreSQL | Production-grade, good for relational data |
| Queue | Celery + Redis | Battle-tested, Python-native |
| GPU Compute | Self-hosted cluster | Existing infrastructure, no cloud costs |
| Object Storage | MinIO or S3 | Result artifacts, large files |
| Hosting (API) | Railway or Render | Easy deployment, good free tiers |
| Hosting (Frontend) | Vercel | Excellent React/Vite support, free tier |

### Key Platform Features

| Feature | Description | Priority |
|---------|-------------|----------|
| Public Leaderboard | Browse all benchmarks, sort by efficiency/speed/energy | P0 |
| Model Search | Filter by size, architecture, backend, hardware | P0 |
| Comparison View | Side-by-side charts for selected models | P0 |
| Submit Results | Upload JSON from local CLI runs | P0 |
| Remote Execution | Configure and run benchmarks on cloud GPUs | P1 |
| User Accounts | Track submissions, save comparisons | P1 |
| API Access | Programmatic access to leaderboard data | P1 |
| Notifications | Email/webhook when jobs complete | P2 |
| Hardware Profiles | Standardised configs (A100, H100, etc.) | P2 |

### Visualisation Requirements

#### Charts (Priority Order)

1. **Efficiency Scatter** (P0)
   - X: Throughput (tokens/sec)
   - Y: Energy efficiency (tokens/joule)
   - Colour: Backend (PyTorch vs vLLM)
   - Size: Model size
   - Hover: Full details

2. **Comparison Bar Charts** (P0)
   - Side-by-side for selected models
   - Metrics: throughput, energy, latency, memory

3. **Multi-Cycle Distribution** (P1)
   - Box plots or violin plots
   - Mean with 95% CI error bars
   - Coefficient of variation indicator

4. **Configuration Sensitivity** (P1)
   - Heatmap: batch_size x precision -> efficiency
   - Line charts: parameter sweeps

5. **Hardware Scaling** (P2)
   - Line chart: efficiency vs GPU count
   - Per-model scaling curves

#### Libraries

| Library | Use Case |
|---------|----------|
| Plotly.js | Interactive charts (primary) |
| Recharts | Simple React charts |
| D3.js | Custom visualisations (if needed) |
| Tanstack Table | Data tables with sorting/filtering |

### Code Reuse Strategy

The existing CLI tool provides:

```
src/llenergymeasure/
+-- domain/models.py      -> API response schemas (direct reuse)
+-- config/models.py      -> Form validation schemas (direct reuse)
+-- results/repository.py -> Storage abstraction (extend for S3)
+-- results/exporter.py   -> CSV/JSON export (reuse for API)
+-- orchestration/        -> Job execution (wrap in worker)
```

**New code needed:**
- `api/` - FastAPI routes, auth, WebSocket handlers
- `worker/` - Queue consumer, job runner, result uploader
- `web/` - React frontend (separate repo or monorepo)

### Data Storage Options

| Option | Pros | Cons |
|--------|------|------|
| Local filesystem | Already works | Not shareable |
| HF Dataset | Free, versioned, API | Read-focused |
| SQLite | Simple, portable | Limited concurrency |
| PostgreSQL | Production-grade | More infrastructure |
| S3/GCS | Scalable, cheap | Requires API layer |

---

## Open Questions

1. **Authentication provider?** GitHub OAuth, Auth0, Clerk, or roll-your-own?
2. **Monorepo vs separate repos?** API + frontend together or split?
3. **Result validation?** How to verify submitted benchmarks are legitimate?
4. **Standardised hardware profiles?** Define A100-80GB, H100, etc. configs?
5. **Pricing model?** Free tier limits? Pay-per-benchmark for heavy users?

---

## Sources

### Framework Comparisons

- [Streamlit vs Gradio 2025](https://www.squadbase.dev/en/blog/streamlit-vs-gradio-in-2025-a-framework-comparison-for-ai-apps)
- [Streamlit vs Dash 2025](https://www.squadbase.dev/en/blog/streamlit-vs-dash-in-2025-comparing-data-app-frameworks)
- [Best Streamlit Alternatives](https://plotly.com/blog/best-streamlit-alternatives-production-data-apps/)
- [Python Framework Survey](https://ploomber.io/blog/survey-python-frameworks/)
- [Gradio vs Streamlit vs Dash vs Flask](https://towardsdatascience.com/gradio-vs-streamlit-vs-dash-vs-flask-d3defb1209a2/)

### Deployment Platforms

- [HuggingFace Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [HF Streamlit Spaces](https://huggingface.co/docs/hub/en/spaces-sdks-streamlit)
- [Heroku Alternatives 2025](https://kuberns.com/blogs/post/the-ultimate-guide-to-heroku-alternatives-in-2025/)
- [Railway vs Heroku](https://docs.railway.com/maturity/compare-to-heroku)
- [Top Heroku Alternatives](https://blog.railway.com/p/top-five-heroku-alternatives)

### Existing Benchmarking UIs

- [LLM-Perf Leaderboard](https://huggingface.co/spaces/optimum/llm-perf-leaderboard)
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [Optimum-Benchmark](https://github.com/huggingface/optimum-benchmark)
- [MLCommons Benchmarks](https://mlcommons.org/benchmarks/)
- [Artificial Analysis Leaderboard](https://huggingface.co/blog/leaderboard-artificial-analysis)

### Full-Stack References

- [FastAPI + React Tutorial 2025](https://craftyourstartup.com/cys-docs/tutorials/full-stack-setup-guide/)
- [FARM Stack Guide](https://www.datacamp.com/tutorial/farm-stack-guide)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [FastAPI Project Template](https://fastapi.tiangolo.com/project-generation/)
