# Web Platform Decisions

**Status:** Accepted
**Date decided:** 2026-02-17
**Last updated:** 2026-02-25
**Research:** N/A

## Decision

Three-phase: static leaderboard (v4.0, GitHub Pages) → dynamic API (v4.1, FastAPI + React + PostgreSQL) → live features (v4.2, WebSocket streaming). Outbound worker model for GPU provisioning. Start with pre-computed results only; GPU model decided at Phase B.

---

## Context

The dual-product vision (see [product-vision.md](product-vision.md)) calls for a web platform
targeting policy makers and the public — audiences who need visual proof that implementation
choices matter but cannot run CLI tools. The platform must serve policy advocacy and community
contribution goals while minimising early operational overhead. GPU provisioning for running
new experiments via the web is an unresolved open question that constrains the architecture.

The ML.ENERGY leaderboard (static JSON + GitHub Pages) was identified as the closest peer
pattern: it demonstrates advocacy value without dynamic infrastructure.

---

## Considered Options

### Phase Structure — How to evolve the platform

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] Three-phase: static → dynamic API → live features** | Validates advocacy value before ops overhead; each phase standalone-useful; mirrors ML.ENERGY evolution | Web platform is distant from v2.0; no early external visibility |
| Launch dynamic API immediately (v4.0) | More feature-complete on launch | GPU provisioning unresolved; database ops overhead while CLI still maturing |
| Single monolithic web launch | One release event | Blocks on all features being production-ready simultaneously |

### Phase A — Static Leaderboard (v4.0 MVP)

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] Static JSON → GitHub Pages / Vercel** | No database, no server, no ops; deploy immediately with existing results; ML.ENERGY v3 proves this model | No community contributions; no live features |
| Server-rendered static site (Next.js SSG) | Better SEO; more flexibility | More build complexity for same outcome |
| PDF reports only | Even simpler | No interactivity; not browsable |

### Phase B — Dynamic API (v4.1)

Stack decision:

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] FastAPI + React + PostgreSQL** | FastAPI natural with Pydantic (already used); PostgreSQL proven for this workload; React standard for dashboard UIs | Three separate stacks to operate |
| Django + HTMX | Single framework | Less natural with Pydantic; HTMX less suited to live WebSocket features |
| Supabase (managed Postgres + auto API) | Reduced ops | Vendor lock-in; less control |

Deployment model:

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] Outbound worker model (ClearML pattern)** | GPU workers connect outbound — firewall-friendly for HPC/university clusters; no inbound firewall rules needed | Polling latency; more complex worker lifecycle |
| SSH-tunnelled workers | Direct connection | Requires firewall exceptions; not viable for HPC |
| Centrally hosted GPU fleet | Simple UX | $2-8/hr per A100; financially unsustainable at research scale |

### GPU Provisioning (Open Question — Phase B/C)

Who provides GPUs for "run new experiments via GUI"?

| Option | Pros | Cons |
|--------|------|------|
| A — Centrally hosted | Simple UX | $2-8/hr per A100 — expensive at research scale |
| B — User-provided cloud credentials | No central cost | Complex UX; credential security risk |
| C — Volunteer compute | No central cost | Reliability and trust issues; gaming risk |
| **[Recommended for now] D — Pre-computed results only** | No GPU cost; validates Phase A value | No live experiment feature |

**Current recommendation**: Start with D (Phase A static), decide GPU model at Phase B.
This remains an open question — see [open-questions.md](open-questions.md).

### Phase C — Live Features (v4.2)

No alternatives considered yet. Proposed features:
- WebSocket streaming: live power/temperature during inference
- Campaign designer GUI: parameter sweep builder for non-technical users
- "Demonstration mode": side-by-side config comparison with live power meters

---

## Decision

We will build the web platform in three phases:

**Phase A (v4.0 MVP):** Static JSON leaderboard, following the ML.ENERGY v3 pattern.
```
llem results export --format leaderboard  → JSON files
→ committed to GitHub repo
→ React + TypeScript reads static JSON
→ GitHub Pages / Vercel (free tier)
```
No database, no server, no ops. Deploy immediately with existing results.

**Phase B (v4.1):** Dynamic API — FastAPI + React + PostgreSQL. Community contributions
via `llem results push`. Outbound worker model (users run Docker containers, workers poll
for jobs). Deployment: Railway/Render for API; Vercel for frontend; user machines for GPU workers.

**Phase C (v4.2):** Live features — WebSocket streaming, campaign designer GUI,
demonstration mode side-by-side comparisons.

Rationale: The static-first approach (Phase A) validates advocacy value before incurring ops
overhead. GPU provisioning for Phase B/C is unresolved and should not block Phase A. The
outbound worker model (ClearML pattern) is the only architecture viable for university/HPC
cluster users who cannot open inbound firewall rules.

The web platform is a separate product, separate repo. It consumes the same JSON results
format as the CLI. The shared surface is the library API (Tier 1-3). The web server is not
part of the CLI package.

---

## Consequences

Positive:
- Phase A can ship immediately using existing CLI results — no new infrastructure
- Outbound worker model allows volunteer compute from HPC-constrained researchers
- Separation from CLI package keeps library dependency surface clean

Negative / Trade-offs:
- No community result contributions until Phase B (v4.1)
- Live features (WebSocket power visualisation) deferred to v4.2
- GPU provisioning model for Phase B/C remains unresolved

Neutral / Follow-up decisions triggered:
- GPU provisioning model must be decided before Phase B — see [open-questions.md](open-questions.md)
- Upload trust model (`llem results push`) needed before shareability ships — see [open-questions.md](open-questions.md)
- `llem results export --format leaderboard` command needed for Phase A

---

## Related

- [product-vision.md](product-vision.md) — dual-product vision that motivates the web platform
- [versioning-roadmap.md](versioning-roadmap.md) — v4.0 in full roadmap context
- [open-questions.md](open-questions.md) — GPU provisioning and upload/trust model (both open)
