# Product Vision Discussion (Living Document)

**Started**: 2026-02-05
**Status**: Active discussion — decisions not finalised

---

TODO - i think this is now degraded? are there useful discussions / decisions / ideas we should extract?

## The Two Products

### Product 1: CLI Tool (`lem`)
- **Users**: Technical ML researchers/engineers with GPU access
- **Environment**: Their own servers, HPC clusters, cloud instances
- **Value prop**: Rigorous energy/efficiency measurement with statistical guarantees
- **Deployment**: Containerised (Docker now, SLURM/K8s/Apptainer later) and/or pip install

### Product 2: Web Platform
- **Users**: Non-technical users (policy makers, decision makers, public)
- **Environment**: Browser — no GPU, no CLI knowledge required
- **Value prop**: Democratised access to efficiency measurement + policy advocacy via data
- **Core features**:
  1. Select HF model via GUI
  2. Configure implementation parameters via GUI
  3. Define experiment series / campaigns with parameter bounds
  4. Live visualisation during experiments (streaming responses, timeseries power/energy)
  5. Results visualisation and exploration
- **Goal**: Demonstrate that implementation choices matter for LLM energy efficiency → shape policy agenda

### Product 3: Central Results Database / Leaderboard
- **Feeds from**: Both CLI and web platform users (results saved by default)
- **Purpose**: Growing archive of (model × environment × implementation params → efficiency metrics)
- **Features**: Interactive plotting, exploration, comparison
- **Audience**: Decision makers, public, researchers wanting to see landscape
- **Goal**: Evidence base for policy advocacy

---

## Open Design Questions

### Q1: Web Platform GPU Provisioning
**Problem**: Non-technical users don't have GPUs. Who provides compute?
**Options discussed**: TBD
**Decision**: OPEN

### Q2: CLI Deployment Model
**Context**: GPU workloads — is bare-metal/local install even needed?
**User's thinking**:
- Option A: All containerised, all backends equally (Docker-first)
- Option B: pip install for single backend (local), Docker for multi-backend
- Option C: pip install with backend selection at install time, Docker optional
**Open sub-question**: Later versions → SLURM/K8s/Apptainer for HPC
**Decision**: OPEN

### Q3: Results Database Architecture
**Problem**: Central DB that both CLI and web write to
**Sub-questions**:
- Default opt-in or opt-out for result submission?
- Data quality / trust model?
- Schema versioning?
- Privacy model?
**Decision**: OPEN

### Q4: Campaign System Fate
**Context**: Phase 4 audit says 84% cut for CLI. But web platform needs campaign-like functionality (GUI-defined experiment series).
**Tension**: CLI campaign code is subprocess-based, unsuitable for web. But the *concept* of campaigns is central to web platform.
**Decision**: OPEN

### Q5: Docker Orchestration Scope
**Context**: Current Docker orchestration is 3,150 lines. May be over-engineered for CLI but cross-backend comparison is a core value prop.
**Decision**: OPEN

### Q6: Product Sequencing
**Previous session proposed**: v2.0 CLI → v3.0 library decoupling → v4.0 web
**Under review**: May need revision based on this discussion
**Decision**: OPEN

---

## Decided So Far

| Decision | Detail | Date |
|----------|--------|------|
| Dual-product vision | CLI + Web platform (separate products, shared core) | 2026-02-05 |
| Central results DB | Archive of efficiency measurements, explorable via web | 2026-02-05 |
| Policy advocacy goal | Web platform exists to demonstrate implementation matters for efficiency | 2026-02-05 |
| HPC support | SLURM/K8s/Apptainer deferred to post-v2.0 | 2026-02-05 |

---

## Discussion Log

### Session 2 (2026-02-05, continued)

**User input**: Web platform is for non-technical users to:
- Select models, configure params via GUI
- Run campaigns with parameter bounds
- See live visualisation (streaming + power timeseries)
- Visualise results

Central DB builds over time from both CLI and web experiments. Interactive exploration to make policy case that implementation matters.

CLI deployment: maybe everything containerised since it's GPU work? Or pip install for single backend, Docker for multi-backend?

**Claude's analysis (Session 2)**:

**On the web platform** — Much more compelling than Session 1's "web face on CLI." This is an advocacy and democratisation platform. Genuinely unique value prop: nobody else is building "visual proof that implementation choices change LLM energy consumption."

**GPU provisioning problem raised**: Non-technical users don't have GPUs. Four options identified:
- A: Centrally hosted GPUs (simplest UX, expensive ~$2-8/hr per A100)
- B: Cloud provider integration (user pays, complex UX)
- C: Volunteer compute (unreliable, trust issues)
- D: Pre-computed results only (no GPU cost, but can't test new models)
- Claude recommended: Start as D (results explorer), add experiment-running later

**Central DB analysis**: This is the most architecturally significant feature — it affects CLI too (network client needed, stable schema required, privacy model). Claude recommended opt-in not default upload. Results schema should be stabilised in v2.0 regardless of web timeline.

**CLI deployment pushback**: Claude argued against "all containerised" — researchers prefer bare-metal for benchmarking (container overhead contaminates measurements). Proposed tiered model:
- pip install lem[backend] for single backend (casual/researcher)
- Docker for multi-backend comparison
- SLURM/K8s/Apptainer for HPC (future)

**Campaign system resolution proposed**: Cut aggressively for CLI (current code is subprocess-based, unsuitable for web). When/if web comes, build proper async job infrastructure from scratch. The *concept* of campaigns matters; the *implementation* doesn't transfer.

**Questions posed to user** (awaiting answers):
1. Who pays for GPUs in "run experiments via GUI" scenario?
2. Opt-in vs default upload for results?
3. Keep pip install as primary deployment, or Docker-first?

**User responded** with more detail but session paused before these questions were fully resolved. Discussion is mid-flow.
