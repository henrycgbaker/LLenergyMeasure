# Versioning Roadmap

**Status:** Accepted
**Date decided:** 2026-02-18
**Last updated:** 2026-02-25
**Research:** [../research/09-broader-landscape.md](../research/09-broader-landscape.md)

## Decision

v2.0 = the complete CLI research tool (library restructure, measurement depth, Docker multi-backend, parameter completeness). v3.0 = quality + efficiency (lm-eval integration). v4.0 = web platform (FastAPI + React). Each version standalone-useful.

---

## Context

After confirming the library-first architecture and 2-command + 1-flag CLI at v2.0, we needed
an explicit roadmap to sequence the remaining capability areas. The sequencing reflects the
dual-product vision (CLI as primary deliverable, web as separate product) while ensuring each
version delivers standalone value.

The lm-eval and MLflow evolution paths informed the sequencing approach — both tools
delivered a solid CLI first, then layered quality integrations and web interfaces as separate
products only after the core measurement tool was proven.

> **Superseded (2026-02-25):** The original roadmap (2026-02-18) used v2.0–v2.4 micro-versioning
> (v2.0 Clean Foundation, v2.1 Measurement Depth, v2.2 Docker, v2.3 Parameter Completeness,
> v2.4 Shareability). This granularity assumed a linear single-developer path and deferred
> table-stakes features (environment metadata, power time-series, prefill/decode phase split)
> to minor versions where peer tools already ship them. The micro-versions have been collapsed
> into a single v2.0 that delivers the complete CLI research tool. Detailed per-feature scope
> for future versions is in [future-versions.md](future-versions.md).

---

## Considered Options

### Version Sequencing Strategy

| Option | Pros | Cons |
|--------|------|------|
| **[Chosen] v2.0 complete CLI, v3.0 quality, v4.0 web** | Each version standalone-useful; web deferred until CLI proves advocacy value; mirrors lm-eval/MLflow evolution | Web platform delayed; researchers who want GUI must wait |
| Ship web platform at v2.x alongside CLI | Early external visibility; potentially more user traction | GPU provisioning unsolved; web ops overhead while CLI still maturing; distraction from core measurement quality |
| Monorepo CLI + web from v2.0 | Simpler repo management early | Premature; web requirements unknown until CLI usage observed |
| v2.0–v2.4 micro-versioning (original) | Granular milestones; incremental delivery | Defers table-stakes features; over-granular for a single-developer project; peer tools already ship what was in v2.1–v2.3 |

**Rejected (2026-02-18):** Shipping the web platform alongside early CLI versions was rejected because the web MVP (static leaderboard) requires validated results from a working CLI first, and the GPU provisioning model for dynamic web experiments is an unresolved open question. See [web-platform.md](web-platform.md).

**Rejected (2026-02-25):** v2.0–v2.4 micro-versioning. The original roadmap deferred environment metadata, power time-series, and prefill/decode phase-split energy to v2.1–v2.3. Research (FEATURES.md, DECISION-AUDIT.md) shows all three are table-stakes features that every energy-focused peer tool (TokenPowerBench, ML.ENERGY, AIEnergyScore) already ships. Delivering them across four minor versions is over-granular. Collapsed into v2.0.

---

## Decision

We will sequence releases in three major versions: v2.0 builds the complete CLI research tool,
v3.0 adds the unique differentiator (accuracy x efficiency tradeoff via lm-eval), and v4.0
delivers the web platform as a separate product that shares only the library API.

Rationale: Each version must deliver standalone value. Web infrastructure should not be
maintained while CLI measurement quality is still maturing. The sequencing mirrors lm-eval
(CLI first, then integrations) and MLflow (local first, then hosted).

### Roadmap

| Version | Name | Scope |
|---------|------|-------|
| **v2.0** | Complete CLI Research Tool | Library-first restructure, `llem` rename, 2 commands + 1 flag (`llem run`, `llem config`, `llem --version`), P0 bug fixes (4), dead code removal (1,524 lines), local + Docker study execution, subprocess isolation, environment metadata capture, power time-series, prefill/decode phase-split energy, `--dry-run` grid preview, parameter coverage targets, Zeus energy backend, `ExperimentResult` schema (config_hash, measurement_methodology, EnvironmentSnapshot) |
| **v3.0** | Quality + Efficiency | lm-eval integration, accuracy vs efficiency tradeoff metrics, `quality_per_joule` |
| **v4.0** | Web Platform | Static JSON leaderboard (v4.0), dynamic API (v4.1), live features (v4.2). See [web-platform.md](web-platform.md) |

### v2.0 Scope Detail

> **Updated (2026-02-25):** v2.0 is delivered across incremental milestones — each milestone
> ships a usable product. No separate v2.1/v2.2 versions exist. The milestone structure
> will be defined in ROADMAP.md when requirements are finalised.

**P0 bug fixes** (Phase 4 audit):
1. PyTorch model_kwargs bug (L375)
2. vLLM no native streaming
3. Docker broken (single-backend use; multi-backend orchestration also in v2.0)
4. vLLM shm-size missing

**Structural changes:**
- Dead code removal (1,524 lines confirmed removable)
- Library-first restructure: `run_experiment()`, `run_study()`, `ExperimentConfig`, `StudyConfig`, `ExperimentResult`, `StudyResult` in `__init__.py`
- 2 CLI commands + 1 flag: `llem run`, `llem config`, `llem --version`
- `llem run` handles both single experiments and multi-experiment studies — YAML determines scope
- State machine: 6 → 3 states
- `llem` rename (was `lem`); no alias, no shim
- Subprocess isolation: each experiment runs in a fresh `multiprocessing.Process`
- Docker orchestration for multi-backend studies (auto-detected + enforced)
- Docker image publishing per backend
- Study resume from manifest (`llem run --resume study-dir/`)

**Measurement depth (previously v2.1):**
- Environment metadata capture (EnvironmentSnapshot in every ExperimentResult)
- Power time-series capture (Zeus PowerMonitor or direct NVML polling)
- Zeus energy backend with accuracy hierarchy (Zeus → NVML direct → CodeCarbon)
- ExperimentResult schema additions: config_hash, measurement_methodology, steady_state_window

**Parameter completeness (previously v2.3):**
- PyTorch/vLLM/TRT coverage targets (95%/90%/95%)
- Prefill/decode phase-split energy attribution
- `--dry-run` grid preview for studies with VRAM estimation

**Shareability (deferred from original v2.4):**
- `llem results push` — deferred; trust model unresolved. See [open-questions.md](open-questions.md)
- HuggingFace Datasets export — deferred to post-v2.0

---

## Consequences

Positive:
- v2.0 ships a complete, peer-competitive CLI research tool in one release
- Table-stakes features (env metadata, power time-series, phase-split energy) ship at launch
- Web platform deferred until CLI proves value and GPU provisioning model is resolved
- Sequencing is legible to contributors — three clear major milestones

Negative / Trade-offs:
- v2.0 scope is larger than the original micro-versioned plan — longer time to first release
- Web platform is multiple release cycles away; no early external visibility
- lm-eval integration (v3.0) deferred — the accuracy×efficiency differentiator is not
  available to early CLI users

Neutral / Follow-up decisions triggered:
- Docker orchestration design within v2.0 — see [docker-execution.md](docker-execution.md)
- GPU provisioning model for web must be resolved before v4.0 — see [web-platform.md](web-platform.md) and [open-questions.md](open-questions.md)
- `llem results push` trust model needed before shareability — see [open-questions.md](open-questions.md)
- Detailed per-feature sub-decisions for future versions — see [future-versions.md](future-versions.md)

---

## Related

- [architecture.md](architecture.md) — structural changes at v2.0
- [installation.md](installation.md) — rename and packaging changes
- [future-versions.md](future-versions.md) — detailed sub-decisions per future version area
- [web-platform.md](web-platform.md) — v4.0 platform decisions
- [open-questions.md](open-questions.md) — shareability blockers (upload, trust model)
- [../designs/docker-execution.md](../designs/docker-execution.md) — Docker design
- [../research/09-broader-landscape.md](../research/09-broader-landscape.md) — lm-eval and MLflow evolution paths
