# Open Questions

These are the only substantive unresolved decisions as of 2026-02-17. Everything else
is either confirmed or deferred to a later version.

## Q1: Results Upload — Opt-in vs Default?

**Blocker for**: v2.4 shareability design

| Option | Pros | Cons |
|--------|------|------|
| Opt-in (recommended) | Respects privacy, works behind firewalls, principle of least surprise | Fewer results in central DB initially |
| Default (upload unless disabled) | More data faster | Surprises users, blocked by firewalls, breaks air-gapped HPC environments |

**Claude recommends**: Opt-in. The central DB's value is in trust and provenance, not volume.
Uploading by default from a measurement tool running on private infrastructure is a hard no
for most institutional users.

**Decision needed before**: Designing the `llem results push` interface and central DB API.

---

## Q2: Results Trust Model for Central DB

**Blocker for**: v2.4 + v4.1

Central DB is an archive, not a crowdsourced leaderboard. But what prevents bad results?

Options:
- A: No validation — submitter attests accuracy (simplest, lowest friction)
- B: Provenance metadata required (hardware attestation, environment fingerprint)
- C: Statistical outlier detection (flag anomalous results)
- D: Manual review / curation (team reviews before publishing)

**Decision needed before**: Designing the upload API schema.

---

---

## ~~Q3: Runner Defaults~~ — RESOLVED 2026-02-19

**Decision: Option A — `local` always. Docker is opt-in.**

| Option | Result |
|---|---|
| A | `local` default for all. Docker opt-in via user config. | **CHOSEN** |
| B | Docker default if available; `local` fallback with warning. | Rejected — confused defaults; surprising behaviour |
| C | Docker required for studies; `local` only for single experiments. | Rejected — breaks HPC users |
| D | Docker required always; `local` flag opt-out with reproducibility warning. | Rejected — worst cold-start DX |

**Rationale:**
- Docker is a system dependency — cannot be pip-installed, cannot be assumed present
- Many researchers (HPC, air-gapped environments) cannot use Docker
- Zero-config first use (`pip install llenergymeasure[pytorch]` → `llem run`) must work everywhere
- Thermal isolation (the primary measurement concern) is NOT improved by Docker
- optimum-benchmark uses local `multiprocessing.Process` for research-grade benchmarks without Docker
- Docker requirement surfaces naturally: multi-backend studies detect multiple backends and
  hard-error with clear guidance if Docker isn't available (correctness requirement, not preference)

**Runner resolution precedence (updated in architecture.md):**
```
1. LLEM_RUNNER_<BACKEND> env var          ← highest
2. ~/.config/llenergymeasure/config.yaml runners.<backend>
3. CLI flag: llem run --runner docker     ← overrides all backends
4. Default: local                         ← implicit when nothing else is configured
```

**Related sub-questions (resolved):**
- Multi-backend at v2.2 with no user config → **hard error + guidance** (not auto-detect).
  Error message: "Multi-backend study requires user config. Run `llem config` to see setup instructions."
- Default Docker images → auto-select by tool version + CUDA major version. Logic lives in
  the Docker runner. Resolved at v2.2 design time (see decisions/docker-execution.md).
- `llem config` (passive display) is sufficient onboarding. No `llem config init` for v2.0.

---

## Not Open (Resolved Elsewhere)

These were previously listed as open but are now decided:

| Question | Resolution | File |
|----------|-----------|------|
| Phase 5 scope finalisation | Blocked on Phase 4.5 completion + discussion wrap-up; not an open question | [versioning-roadmap.md](versioning-roadmap.md) |
| CLI deployment model (pip vs Docker) | pip primary, Docker for conflicting deps | [installation.md](installation.md) |
| Campaign system fate | Separate module (v2.2), not cut and not in CLI | [architecture.md](architecture.md) |
| Web platform sequencing | Static JSON → dynamic API → live features | [web-platform.md](web-platform.md) |
| Runner default (Q3) | `local` always; Docker opt-in; multi-backend hard-errors without Docker | Q3 section above |
