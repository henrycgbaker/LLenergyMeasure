# Web Platform Patterns

**Sources**: `04-deployment-patterns.md` §3, `07-ml-energy-ecosystem.md` §4

## ML.ENERGY Leaderboard Architecture (Static JSON Pattern)

ML.ENERGY v3 is the clearest validated example of a low-cost ML leaderboard:

```
Benchmark runs (curated, team-run)
    → JSON files committed to GitHub repo
    → React + TypeScript app reads static JSON
    → Hosted on GitHub Pages (zero cost)
```

No server. No database. No ops burden. Deployed immediately. Easy to fork and extend.

**What they validate**: Static JSON leaderboards work at scale. Their data is trusted
precisely because they run the benchmarks themselves — not crowdsourced.

**What this means for us**: v4.0 MVP can follow this exact pattern. `llem results export
--format leaderboard` generates the JSON. CI/CD auto-deploys on new results.

## FastAPI + React Stack

Validated by MLflow (Fluent API server), ML.ENERGY (React frontend), and W&B (FastAPI):

- **Backend**: FastAPI — natural fit with our Pydantic models, async support, OpenAPI docs auto-generated
- **Frontend**: React + TypeScript — dominant in ML tooling UIs
- **Database**: PostgreSQL for dynamic phase (v4.1+)

Deployment options validated by the ecosystem:
- Railway / Render for small-scale API (no infra management)
- Vercel for static React frontend
- User-managed GPU workers via Docker

## Outbound Worker Model (ClearML-Agent Pattern)

```
┌─────────────────────────┐
│  Central coordinator    │  ← web server, job queue
│  (FastAPI + PostgreSQL) │
└────────────┬────────────┘
             │  HTTP (outbound from worker)
             ▼
┌─────────────────────────┐
│  GPU Worker             │  ← user's Docker container
│  (llem + GPU access)    │  ← polls for jobs, runs experiments
└─────────────────────────┘
```

Workers connect **outbound** — firewall-friendly for HPC clusters and university networks.
Worker = `docker run llem-worker --coordinator https://llem.example.com --token <token>`

ClearML-Agent and W&B-Agent both prove this model works in practice.

## Evolution Path

| Phase | Version | Model | Cost | Ops |
|-------|---------|-------|------|-----|
| A — Static | v4.0 | GitHub Pages + static JSON | Free | Near-zero |
| B — Dynamic | v4.1 | FastAPI + PostgreSQL + React | Low | ~2-4h/week |
| C — Live | v4.2 | + WebSocket streaming + campaign GUI | Medium | ~8h/week |

**Key principle**: Validate advocacy value (Phase A) before investing in dynamic
infrastructure (Phase B). Live experiment running for non-technical users is the
hardest problem — defer GPU provisioning question until Phase B/C.
