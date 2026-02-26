# Future Version Directions

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

> **Superseded (2026-02-25):** The original version boundaries (v2.1–v2.4) have been
> collapsed. Features previously assigned to v2.1 (measurement depth), v2.2 (Docker),
> and v2.3 (parameter completeness) are now part of v2.0 scope. See
> [versioning-roadmap.md](versioning-roadmap.md) for the updated roadmap. This document
> retains the detailed sub-decisions per feature area — version labels below reflect
> the updated roadmap (v2.0 / v3.0 / v4.0).

---

## Context

This document provides detailed sub-decisions for feature areas beyond the v2.0 core,
and for v2.0 features that need design-level direction. The v2.0 foundation (2 commands
+ 1 flag, local + Docker runner, PyTorch/vLLM/TRT-LLM backends, full measurement depth)
is the complete CLI research tool from which v3.0 and v4.0 extend.

---

## v2.0 Feature Detail — Measurement Depth

> **Updated (2026-02-25):** Previously v2.1. Pulled into v2.0 scope — environment metadata,
> power time-series, and energy backend hierarchy are table-stakes for an energy measurement
> tool. See [versioning-roadmap.md](versioning-roadmap.md).

### Sub-decision 1: Energy Backend Hierarchy

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Zeus → NVML direct → CodeCarbon priority — Chosen** | Zeus is the accuracy leader for NVIDIA; NVML direct avoids requiring Zeus for basic measurement (pynvml already a likely dep); CodeCarbon justified only for CO2 estimates | Adds complexity to backend selection logic |
| Zeus required for energy measurement | Simplest selection logic | Forces Zeus installation on users who just want basic NVML measurement |
| CodeCarbon as primary | CO2 estimates built in | ~10-15% accuracy vs Zeus's ~5%; less precise for raw energy research |

#### Decision

We will use the following runtime priority order:
1. **Zeus** (`[zeus]` extra) — NVML direct, ~5% accuracy, AMD/Apple Silicon future support
2. **NVML direct** (bundled with pynvml, no extra needed) — fallback when Zeus absent
3. **CodeCarbon** (`[codecarbon]` extra) — CO2-only users, ~10-15% accuracy

Rationale: Zeus is the accuracy leader for NVIDIA measurement. NVML direct (via pynvml)
is already a dependency — using it directly as fallback avoids requiring Zeus for basic
measurement. CodeCarbon is only justified when CO2 estimates are needed.

**Open question**: Does pynvml already ship with the base install, or is it a separate extra?
Confirm during v2.1 planning — if pynvml is already present, NVML-direct baseline is free.

**Note on CO2**: CO2 is derived (joules × grid_carbon_intensity), stored with
`estimation_method` field. CO2 calculation is separate from energy measurement. See
`decisions/carbon-intensity.md` for grid intensity lookup strategy.

#### Consequences

Positive: Zero-extra energy measurement available via NVML fallback. Zeus users get best accuracy.
Negative / Trade-offs: Backend selection logic has three-tier priority; must be clearly documented.
Neutral: Triggers `decisions/carbon-intensity.md` for CO2 grid intensity strategy.

---

### Sub-decision 2: Statistical Methodology

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Bootstrap resampling for CIs — Chosen** | No distributional assumptions; research-grade; 95% CI via BCa method | Computationally heavier than mean ± std |
| Mean ± std (ML.ENERGY approach) | Simple; fast | Assumes normality; less robust for skewed distributions |
| Raw scores only (lm-eval approach) | Maximum simplicity | No uncertainty quantification; insufficient for research use |

#### Decision

We will use bootstrap resampling (1000 resamples, BCa method where feasible) for 95%
confidence intervals. Steady-state detection via rolling CV (coefficient of variation) over
last N cycles — continue until CV < 5% over 3 consecutive cycles, or `n_cycles` is
exhausted. Results report: mean, std, p5, p95, CV, n_valid_cycles.

Rationale: Bootstrap CI is the right choice for a research-grade tool — it makes no
distributional assumptions. Peer comparison: ML.ENERGY uses mean ± std; Zeus uses mean;
lm-eval uses raw scores. The `measurement_methodology` field in ExperimentResult documents
what was applied.

**Open question**: Minimum n_cycles for reliable bootstrap CI? Literature suggests ≥5 for
reasonable CI width. Document this as a user warning when n_cycles < 5.

#### Consequences

Positive: Research-grade uncertainty quantification with no distributional assumptions.
Negative / Trade-offs: Computationally heavier than simpler statistics; adds `measurement_methodology` field to result schema.
Neutral: n_cycles minimum recommendation to document in user-facing docs.

---

## v2.0 Feature Detail — Docker Multi-Backend Studies

> **Updated (2026-02-25):** Previously v2.2. Pulled into v2.0 scope — Docker multi-backend
> is required for the tool's core cross-backend comparison value proposition.
> See [versioning-roadmap.md](versioning-roadmap.md).

### Sub-decision: Docker Image Strategy

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Official images on Docker Hub under `llenergymeasure/` org — Chosen** | Standard distribution; auto-tag resolution by host CUDA version; reproducibility via image digests | Requires Docker Hub org setup; build matrix maintenance |
| Users build their own images | No infrastructure | Poor UX; version drift; no reproducibility guarantee |
| Single unified image (all backends) | Simpler distribution | Image size explosion; backend conflicts; not how upstream images are structured |

#### Decision

We will publish official images to Docker Hub under `llenergymeasure/` organisation.
One image per backend per CUDA major version.

Naming convention:
```
llenergymeasure/pytorch:{tool_version}-cuda{cuda_major}
llenergymeasure/vllm:{tool_version}-cuda{cuda_major}
llenergymeasure/tensorrt:{tool_version}-cuda{cuda_major}

# Examples:
llenergymeasure/pytorch:2.2.0-cuda12
llenergymeasure/vllm:2.2.0-cuda12
llenergymeasure/tensorrt:2.2.0-cuda12
```

**Tag resolution** (auto-selected when user doesn't specify):
Host CUDA major version → select matching image tag. Logic in the Docker runner.
Mismatch (e.g. host has CUDA 11, only cuda12 image exists) → error with guidance.

**Base images**: Use official backend images where available:
- vLLM: `vllm/vllm-openai:latest` as base, add llem entrypoint layer
- TensorRT-LLM: `nvcr.io/nvidia/tensorrt:xx-py3` as base
- PyTorch: `pytorch/pytorch:latest` as base

**Build + publish**: GitHub Actions on release tag. Build matrix: `{pytorch, vllm, tensorrt}
× {cuda11, cuda12}`. Push to Docker Hub. SBOM and digest recorded in release notes.

**Maintenance**: Backend images track upstream (pytorch, vllm, trt-llm) releases.
Tool team owns the entrypoint layer; upstream team owns the base image.

**Open questions** (resolve during v2.2 planning):
- Which CUDA minor versions to target within each major (e.g. 12.1 vs 12.4)?
- Multi-arch (amd64 only, or also arm64 for future Apple Silicon support)?
- Private registry option for air-gapped / enterprise users?

#### Consequences

Positive: Reproducible multi-backend studies; users pin by digest; auto CUDA version selection.
Negative / Trade-offs: Build matrix maintenance across 6 image combinations per release; Docker Hub org required.
Neutral: Triggers open questions on CUDA minor targeting and multi-arch that must be resolved in v2.2 planning.

---

## v2.0 Feature Detail — Parameter Completeness

> **Updated (2026-02-25):** Previously v2.3. Pulled into v2.0 scope — parameter coverage
> targets and prefill/decode phase split are core to the research tool.
> See [versioning-roadmap.md](versioning-roadmap.md).

### Sub-decision 1: Coverage Definition and Targets

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Coverage = first-class params / impactful total params — Chosen** | Measurable; tied to actual energy/throughput impact | "Impactful" threshold (≥1%) requires empirical validation |
| Coverage = all possible backend params | Simple to count | Includes irrelevant params (logging, seeds); misleading metric |

#### Decision

Coverage metric per backend:
```
coverage = first_class_params / (first_class_params + known_impactful_missing_params)
```

"Impactful" = changes energy or throughput by ≥1% in benchmarks. Minor kwargs (logging,
seed behaviour) are excluded from the coverage denominator.

**Targets**: PyTorch ≥ 95%, vLLM ≥ 90%, TensorRT-LLM ≥ 95%

**Escape hatch**: `extra: {param: value}` in ExperimentConfig forwards arbitrary kwargs
to the backend. SSOT introspection cannot auto-discover `extra:` params — they are
opaque. This is intentional: first-class params have schema validation, docs, and SSOT
introspection; `extra:` does not.

#### Consequences

Positive: Clear measurable target for parameter coverage; `extra:` escape hatch avoids blocking users with niche params.
Negative / Trade-offs: "≥1% impact" threshold requires benchmark data to classify params.
Neutral: N/A

---

### Sub-decision 2: Prefill/Decode Phase Split

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Separate timing, energy, throughput per phase — Chosen** | Exposes prefill vs decode efficiency tradeoffs; vLLM has TTFT natively; research value is high | Architecturally significant; results schema changes; backends need new instrumentation |
| Aggregate-only metrics | Simpler schema | Hides prefill/decode tradeoffs that are central to KV cache and batch size research |

#### Decision

We will add separate per-phase metrics to ExperimentResult (or a nested sub-object):
```python
prefill: PhaseMetrics      # first-token latency phase
decode: PhaseMetrics       # autoregressive decode phase
# PhaseMetrics: tokens, time_s, energy_j, tokens_per_sec
```

Requires per-phase instrumentation in all three backends. vLLM has TTFT natively;
PyTorch and TRT-LLM require instrumentation.

**Note**: This is architecturally significant — results schema changes, backends need new
instrumentation hooks. Plan carefully; included in v2.0 result schema from the start.

#### Consequences

Positive: Exposes the prefill/decode efficiency tradeoff that is central to this tool's purpose.
Negative / Trade-offs: PyTorch and TRT-LLM need new instrumentation hooks; adds complexity to v2.0 scope.
Neutral: Included in the v2.0 result schema — no migration needed since it ships at launch.

---

## Post-v2.0 — Shareability

> **Updated (2026-02-25):** Previously v2.4. Deferred from v2.0 — trust model for uploads
> is unresolved. Will ship in a post-v2.0 minor release when the trust model is decided.
> See [open-questions.md](open-questions.md).

### Sub-decision: Upload / Sharing API

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Opt-in only; explicit `llem results push` command — Chosen** | Library remains side-effect-free; trust from provenance metadata not volume; no surprise uploads | Users must take explicit action to contribute results |
| Automatic upload at end of experiment | Higher data volume | Unacceptable for library-side-effect-free constraint; privacy concern |
| No upload at all | Maximum privacy | Prevents community result sharing; undermines v4.0 web platform |

#### Decision

We will implement opt-in upload only. `llem results push <file>` is a separate CLI command
requiring explicit invocation. The central DB is a **growing archive** (not a crowdsourced
leaderboard) — trust comes from provenance metadata, not volume.

**Auth model**: API token in `~/.config/llenergymeasure/config.yaml` under `sharing.api_key`.
Never in config files that get committed. Set via `export LLEM_API_KEY=...` or user config.

**Trust model** (see open-questions.md Q2 — unresolved):
Options: (A) submitter attests, (B) provenance metadata required, (C) statistical outlier
detection, (D) manual review. Recommend B as baseline: hardware fingerprint + environment
snapshot required for upload. A without B is unverifiable; C/D add operational complexity
not justified at early DB scale.

**Privacy**: Only ExperimentResult fields are uploaded. No model weights, no user-identifying
data. `environment_snapshot` fields (GPU model, CUDA version, driver) are included —
document clearly in upload confirmation prompt.

**HuggingFace Datasets export**: `llem results export --format hf-datasets` writes a
HF-compatible Parquet file locally. Separate from the central DB upload.

#### Consequences

Positive: Library remains side-effect-free; opt-in preserves user trust; HF export enables independent sharing.
Negative / Trade-offs: Lower data volume than automatic uploads; trust model Q2 still unresolved.
Neutral: Trust model decision (open-questions.md Q2) must be resolved before v2.4 implementation.

---

## v3.0 — Quality + Efficiency (lm-eval)

### Sub-decision: lm-eval Integration

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **Subprocess invocation — Chosen** | Isolates heavy dependencies (transformers, accelerate); matches existing backend isolation model; lm-eval CLI is stable | IPC overhead; result parsing complexity |
| Direct import (`import lm_eval`) | Simpler invocation | Heavy dep conflicts with backend-specific installs; ties our release cycle to lm-eval's |

#### Decision

We will integrate lm-eval as `llenergymeasure[lm-eval]` optional extra, invoking lm-eval
as a subprocess (not imported). Results merged into ExperimentResult.

**Integration pattern**: subprocess (not `import lm_eval`) because:
- lm-eval has heavy dependencies (transformers, accelerate) that may conflict with
  backend-specific installs
- Subprocess isolation matches the existing backend isolation model
- lm-eval CLI is stable and designed for this use pattern

**Task selection**: User specifies `quality: {tasks: [hellaswag, mmlu]}` in ExperimentConfig.
llem runs lm-eval with matching model, records `lm_eval_results: dict` in ExperimentResult.

**Tradeoff metric**: `quality_per_joule` = weighted accuracy score / energy_joules.
Enables Pareto frontier visualisation (accuracy vs efficiency). This is the unique
differentiator: no other tool combines inference efficiency + accuracy in a single run.

**Open question**: How to handle lm-eval's dataset downloads? They can be large (MMLU = 300MB).
Options: (A) pre-download required, (B) llem downloads on first use with progress bar,
(C) skip quality tasks if datasets absent. Recommend B with clear progress indication.

#### Consequences

Positive: `quality_per_joule` is a unique differentiator; subprocess isolation prevents dep conflicts.
Negative / Trade-offs: IPC overhead; result parsing complexity; dataset download UX to resolve.
Neutral: Dataset download strategy (open question) must be resolved during v3.0 planning.

---

## v4.0 — Web Platform

> **Note**: Separate product, separate repo. Full decisions in [web-platform.md](web-platform.md).

### Sub-decision: Tech Stack Direction

#### Considered Options

| Option | Pros | Cons |
|--------|------|------|
| **React + TypeScript + FastAPI — Chosen** | React is dominant for data-heavy frontends; FastAPI naturally exposes Pydantic models already in the result schema; thin API layer | More frontend complexity than a static site |
| Static JSON only (no API) | Zero backend cost; simple | Cannot support dynamic queries, filtering, or aggregation at scale |
| Django + Jinja templates | Single language stack | Not idiomatic for modern data visualisation; slower iteration on frontend |

#### Decision

**Frontend**: React + TypeScript. Vite for build. Recharts or Observable Plot for
data visualisation (both handle structured tabular data well; Observable is more
flexible for custom efficiency plots).

**Backend (v4.1+)**: FastAPI. Pydantic models are already the result schema — FastAPI
naturally exposes them. Thin layer; no business logic in the API server.

**Deployment**:
- Static JSON MVP (v4.0): GitHub Pages or Cloudflare Pages. Zero backend cost.
- Dynamic API (v4.1): FastAPI on a single small VM. Results DB = PostgreSQL or SQLite
  (SQLite is fine until query latency becomes a problem — defer PostgreSQL migration).
- GPU workers: self-hosted (no GPUs at the central server — only at user sites or
  dedicated measurement machines).

**Database**: PostgreSQL for the central results archive. Schema = ExperimentResult
JSON stored as JSONB column + indexed metadata (model, backend, GPU, timestamp).
Avoids ORM complexity for the query patterns we need (filter + aggregate).

**Hosting**: Cloudflare for DNS + CDN. VPS (Hetzner / DigitalOcean) for API server.
Monitoring: UptimeRobot (free tier sufficient for v4.0). No Kubernetes — single VM
+ systemd is the right complexity level until traffic justifies more.

#### Consequences

Positive: Pydantic result schema maps directly to FastAPI; SQLite → PostgreSQL migration deferred until needed.
Negative / Trade-offs: React + TypeScript adds frontend build complexity for a Python-first project.
Neutral: v4.0 is a separate repo — implementation details outside the scope of this document.

---

## Related

- [versioning-roadmap.md](versioning-roadmap.md) — version roadmap (v2.0 / v3.0 / v4.0)
- [release-process.md](release-process.md) — PyPI publishing, Docker image distribution, changelog
- [web-platform.md](web-platform.md) — full v4.0 web platform decisions
- [open-questions.md](open-questions.md) — upload trust model (shareability blocker)
- [installation.md](installation.md) — extras strategy (`[zeus]`, `[codecarbon]`, `[lm-eval]`)
