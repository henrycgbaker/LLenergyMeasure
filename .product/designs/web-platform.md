# Web Platform Design

**Last updated**: 2026-02-17 (migrated to designs/ 2026-02-19)
**Source decisions**: [../decisions/versioning-roadmap.md](../decisions/versioning-roadmap.md)
**Scope**: v4.0+ (separate product, separate repo). Shares library API with CLI.
**Status**: Future — not in scope for v2.0–v3.0

---

## Phase A — Static Leaderboard (v4.0 MVP)

Modelled on ML.ENERGY Leaderboard v3 (React + TypeScript + static JSON).

```
llem results export --format leaderboard
    → JSON files (results, metadata, comparison tables)
    → committed to llenergymeasure-leaderboard GitHub repo
    → React app reads static JSON at build time
    → GitHub Actions deploys to GitHub Pages on merge
```

No server. No database. No ops burden. Validate advocacy value before infrastructure.

**Inputs to leaderboard**: CLI users run `llem results push` (opt-in, v2.4); team curates
selection for public display.

---

## Phase B — Dynamic API (v4.1)

```
┌─────────────────────────────────┐
│  FastAPI server                 │
│  + PostgreSQL                   │  ← Railway / Render
│  + React frontend               │  ← Vercel
└────────────────────┬────────────┘
                     │ REST API
           ┌─────────┴────────────┐
           │                      │
    llem results push        GPU Worker
    (user CLI)               (Docker container)
                             polls for jobs →
                             runs experiments →
                             pushes results
```

Stack confirmed: FastAPI (natural with Pydantic), React + TypeScript, PostgreSQL.

### Worker Model (ClearML-Agent Pattern)

Workers connect **outbound** to the coordinator — no inbound SSH required:

```bash
docker run llem-worker \
  --coordinator https://llem.example.com \
  --token <user-token>
```

Firewall-friendly. Works on HPC clusters, university networks.
Users contribute compute via Docker; coordinator manages job queue.

---

## Phase C — Live Features (v4.2)

- **WebSocket streaming**: Real-time power/temperature during inference, displayed in browser
- **Study designer GUI**: Drag-and-drop parameter sweep builder for non-technical users
- **Demonstration mode**: Side-by-side config comparison (batch=1 vs batch=8) with live power meters

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| API | FastAPI | Pydantic models reused, OpenAPI docs auto-generated |
| Frontend | React + TypeScript | Dominant in ML tooling UIs |
| Database | PostgreSQL (Phase B+) | None needed for Phase A |
| Hosting API | Railway / Render | No infra management |
| Hosting frontend | Vercel / GitHub Pages | Free tier sufficient for Phase A |
| Real-time | WebSockets via FastAPI (Phase C) | |

---

## Separation from CLI Package

The web server is **not** part of `llenergymeasure` package. Separate repository.

Shared: JSON/Parquet results format, library API (`run_experiment`, `run_study`, result types).
Not shared: web server, database, frontend, worker daemon.

---

## Related

- [result-schema.md](result-schema.md): JSON/Parquet result formats consumed by the leaderboard
- [library-api.md](library-api.md): Public library API shared with web worker
