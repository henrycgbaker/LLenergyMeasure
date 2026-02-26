# Product Vision

**Status:** Accepted
**Date decided:** 2026-02-05
**Last updated:** 2026-02-25
**Research:** [../research/09-broader-landscape.md](../research/09-broader-landscape.md)

## Decision

Deployment efficiency focus (not model comparison). Dual-product: CLI for researchers + web for policy makers. Library (`import llenergymeasure`) as secondary offering. Opt-in central results DB for community evidence base.

---

## Context

Before any architecture or CLI design work could begin, the fundamental question of what
this tool is for and who it serves needed to be settled. Two positioning alternatives
existed: measure deployment configuration efficiency (implementation choices after model
selection) vs. measure model efficiency (compare models against each other). A third
question was whether the tool should be a standalone CLI, a library, or both.

ML.ENERGY's own data provided the key empirical grounding: a single parameter change
(batch size) produces a **7.5x energy difference** on the same model — yet no tool
systematically quantifies this space.

---

## Considered Options

### Primary Positioning

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen — Option A] Deployment efficiency focus** | Gap nobody else fills; directly actionable for engineers; ML.ENERGY data validates the 7.5x effect | Narrower audience than model comparison |
| Option B — Model comparison leaderboard | Broader audience; familiar format (cf. HF Open LLM Leaderboard) | Already done by ML.ENERGY, HF leaderboard; no differentiation |
| Option C — General ML benchmarking platform | Maximum scope | Massively competitive space; dilutes the unique value proposition |

**Rejected (2026-02-17):** Model comparison (Option B) and general benchmarking (Option C).
We are not competing with ML.ENERGY or the HF Open LLM Leaderboard — we fill a gap they
leave open.

### Product Shape

| Option | Pros | Cons |
|--------|------|------|
| CLI only | Simpler; focused on researcher use case | Policy makers and public cannot use it |
| **[Chosen] Dual-product: CLI + web platform** | CLI for technical measurement; web for advocacy + public access; both share library API | More work; two audiences to serve |
| Web only | Broader reach | Requires centrally hosted GPU infrastructure from day one |

### Library as Offering

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] Library as secondary offering** | `import llenergymeasure` composable with other benchmarking infrastructure; enables CI/CD integration | Two stable APIs to maintain (CLI + library) |
| CLI only, no library API | Simpler | No programmatic integration; blocks lm-eval-style ecosystem adoption |

### Central Results Database

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] Opt-in central DB** | Growing archive for exploration, comparison, and policy case-making; community contribution model | Upload trust model unresolved; ops overhead |
| No central DB | Zero ops | No community evidence base; advocacy use case weakened |

---

## Decision

We are building a dual-product:
- **CLI** (`llem`) for technical ML researchers and engineers: rigorous efficiency
  measurement with statistical guarantees, running on their own hardware.
- **Web platform** for policy makers, decision makers, and the public: democratised access
  and visual proof that implementation choices matter, with no GPU required.

Both products share the same library API and results format. The library (`import
llenergymeasure`) is a secondary offering enabling programmatic integration and CI/CD use.

We chose **deployment efficiency** (Option A) as our primary positioning: we are the only
tool that systematically quantifies how deployment configuration (batch size, quantisation,
backend, parallelism) affects LLM energy efficiency on a given model.

Rationale: ML.ENERGY's own data validates the 7.5x batch-size effect. No existing tool
fills this gap. Model comparison leaderboards (ML.ENERGY, HF) are already well-served.

---

## Consequences

Positive:
- Clear differentiated positioning — not competing with established leaderboards
- Dual-product structure serves both technical researchers and policy audiences
- Library API enables ecosystem integration (lm-eval, CI/CD pipelines)

Negative / Trade-offs:
- Narrower initial audience than a model comparison leaderboard would attract
- Web platform requires sustained ops investment (mitigated by static-first approach in v4.0)
- Central DB trust model is unresolved — see [open-questions.md](open-questions.md)

Neutral / Follow-up decisions triggered:
- Web platform architecture decisions — see [web-platform.md](web-platform.md)
- Library API stability decisions — see [architecture.md](architecture.md)
- Upload and trust model for central DB — see [open-questions.md](open-questions.md)

---

## Target Users

| Product | Users | Environment | Value prop |
|---------|-------|-------------|------------|
| CLI | Technical ML researchers / engineers | Own servers, HPC clusters, cloud instances | Rigorous efficiency measurement with statistical guarantees |
| Web platform | Policy makers, decision makers, public | Browser, no GPU required | Democratised access + visual proof that implementation choices matter |
| Central DB | All of the above + researchers | Web interface | Evidence base for policy advocacy |

---

## Positioning

**We are not**: A model comparison leaderboard (that is ML.ENERGY / HF Open LLM Leaderboard).

**We are**: The only tool that systematically quantifies how deployment configuration
(batch size, quantisation, backend, parallelism) affects LLM energy efficiency on a
given model.

ML.ENERGY's own data validates this: a single parameter change (batch size) produces
**7.5x energy difference** on the same model.

---

## Related

- [versioning-roadmap.md](versioning-roadmap.md) — how the dual-product vision sequences across versions
- [web-platform.md](web-platform.md) — web platform architecture decisions
- [architecture.md](architecture.md) — library-first structure supporting dual-product
- [open-questions.md](open-questions.md) — central DB trust model (unresolved)
- [../research/09-broader-landscape.md](../research/09-broader-landscape.md) — peer landscape analysis including ML.ENERGY
