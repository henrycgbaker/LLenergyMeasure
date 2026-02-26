# Requirements — v2.0.0 Milestone

TODO: this is now degraded - it needs updating based on new product decisions and desings (see phase 4.5)

27 requirements across 4 releases.

## v1 Requirements

### Measurement Foundations (v1.19.0)

- [x] **MEAS-01**: Capture environment metadata per experiment (CUDA version, driver version, GPU thermal state, power limits, CPU governor, container detection)
- [x] **MEAS-02**: Measure idle GPU baseline power before experiments; report both raw and baseline-adjusted energy readings (raw preserved for backwards compatibility, adjusted fixes 15-30% systematic overestimation)
- [x] **MEAS-03**: Detect and flag thermal throttling during experiments via NVML performance state monitoring
- [x] **MEAS-04**: Collect time-series power/memory/utilisation samples at configurable intervals (extend existing GPUUtilisationSampler)
- [x] **MEAS-05**: Implement warmup convergence detection (continue until CV stabilises, not fixed prompt count)
- [x] **MEAS-06**: Update results schema to v3 with migration path from v2 results
- [x] **MEAS-07**: Add extended efficiency metrics to CSV export (currently JSON-only; memory, GPU utilisation, request latency, batch, KV cache metrics)
- [x] **MEAS-09**: UAT round 1 (fresh clone, install, run one experiment, document pain points)

### Campaign Orchestrator (v1.20.0)

- [ ] **CAMP-01**: Switch from `docker compose run --rm` to long-running containers with `docker compose exec` dispatch
- [ ] **CAMP-02**: Backend-aware programmatic grid generation (cartesian product respecting per-backend param validity)
- [ ] **CAMP-03**: Campaign manifest tracking (exp_id → config → backend → container → status → result_path)
- [ ] **CAMP-04**: Daemon mode for scheduled cycles (run at specific times, not just sequential with delays)
- [ ] **CAMP-05**: Configurable `force_cold_start` option (unload model between experiments for cold-start benchmarking; default: warmup ensures fairness)
- [ ] **CAMP-06**: Container health checks and lifecycle management (start only needed backends, graceful teardown)
- [ ] **CAMP-07**: Retain existing campaign features (randomisation within cycles, interleaved/shuffled/grouped structures, thermal gaps, cycle-as-highest-organisational-principle)
- [ ] **MEAS-08**: Implement bootstrap resampling for 95% confidence intervals (moved from Phase 1 — requires campaign cycles for multi-observation data)
- [ ] **CAMP-08**: UAT round 2 — validate orchestrator dispatches cross-backend campaigns correctly

### Parameter Completeness (v1.21.0)

- [ ] **PARAM-01**: Close known parameter gaps (PyTorch 93.8% → 95%+, vLLM 81.9% → 90%+, TensorRT 93.8% → 95%+)
- [ ] **PARAM-02**: Systematic audit of energy/throughput-impactful kwargs for each backend
- [ ] **PARAM-03**: Validate `extra:` escape hatch works for undocumented/niche parameters
- [ ] **PARAM-04**: Update SSOT introspection to auto-discover any new parameters
- [ ] **PARAM-05**: UAT round 3 — use campaign orchestrator to run parameter audit configs across all backends, verify results

### Polish + UAT (v1.22.0)

- [ ] **UAT-01**: Occam's razor cleanup pass (dead code, naming consistency, simplify over-engineered areas)
- [ ] **UAT-02**: Architectural refactor if pain points emerge during cleanup
- [ ] **UAT-03**: UAT round 4 — full campaign workflow across backends, end-to-end
- [ ] **UAT-04**: Documentation refresh (README, quickstart, deployment guides)
- [ ] **UAT-05**: Docker image rebuild and validation

---

## v2 Requirements (Deferred)

### Precision Metrics (v2.1.0)

- [ ] **PREC-01**: Prefill/decode phase split (separate timing, energy, throughput per phase — PyTorch first, other backends best-effort)
- [ ] **PREC-02**: Profiler integration (optional `torch.profiler` mode for ground-truth FLOPs — PyTorch only)

---

## Out of Scope

- MLflow/W&B integrations — deferred post-v3
- SSM/hybrid architecture support (Mamba, RWKV, Jamba) — future
- 100% exhaustive kwargs coverage — 90%+ energy-impactful with `extra:` escape hatch
- PyPI distribution — install from git sufficient
- Parallelism refactor (unified config block) — discarded, backends handle natively
- NormalisedMetrics (precision-adjusted FLOPs) — deferred to v3 analysis module

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| MEAS-01 | Phase 1 | Complete |
| MEAS-02 | Phase 1 | Complete |
| MEAS-03 | Phase 1 | Complete |
| MEAS-04 | Phase 1 | Complete |
| MEAS-05 | Phase 1 | Complete |
| MEAS-06 | Phase 1 | Complete |
| MEAS-07 | Phase 1 | Complete |
| MEAS-08 | Phase 2 | Pending |
| MEAS-09 | Phase 1 | Complete |
| CAMP-01 | Phase 2 | Pending |
| CAMP-02 | Phase 2 | Pending |
| CAMP-03 | Phase 2 | Pending |
| CAMP-04 | Phase 2 | Pending |
| CAMP-05 | Phase 2 | Pending |
| CAMP-06 | Phase 2 | Pending |
| CAMP-07 | Phase 2 | Pending |
| CAMP-08 | Phase 2 | Pending |
| PARAM-01 | Phase 3 | Pending |
| PARAM-02 | Phase 3 | Pending |
| PARAM-03 | Phase 3 | Pending |
| PARAM-04 | Phase 3 | Pending |
| PARAM-05 | Phase 3 | Pending |
| UAT-01 | Phase 4 | Pending |
| UAT-02 | Phase 4 | Pending |
| UAT-03 | Phase 4 | Pending |
| UAT-04 | Phase 4 | Pending |
| UAT-05 | Phase 4 | Pending |
