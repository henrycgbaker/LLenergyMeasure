# P-08: Traffic Simulation

**Module**: `src/llenergymeasure/core/traffic.py`
**Risk Level**: MEDIUM
**Decision**: Pending — decision required on whether this is in or out of v2.0 scope
**Planning Gap**: Zero mention in any planning document (architecture, cli-ux, energy-backends, observability). The feature exists, is non-trivial, and its v2.0 fate is entirely unaddressed.

---

## What Exists in the Code

**Primary file(s)**: `src/llenergymeasure/core/traffic.py`
**Key classes/functions**:
- `TrafficGenerator` (line 20) — stateful generator of inter-arrival times; supports `poisson` and `constant` modes
- `TrafficGenerator.get_inter_arrival_time()` (line 57) — draws from `numpy.random.default_rng` using exponential distribution for Poisson arrivals or returns fixed `1/target_qps` for constant mode
- `TrafficGenerator.wait_for_next_request()` (line 77) — convenience method that combines delay generation with `time.sleep()`
- `TrafficGenerator.reset()` (line 96) — resets RNG state to original seed, enabling reproducible replays
- `apply_traffic_delay()` (line 109) — module-level entry point; skips delay on first batch (batch_idx==0), then delegates to generator

The module implements MLPerf-style traffic simulation: Poisson arrivals are the industry standard for modelling realistic API request patterns. The `TrafficSimulation` config model (imported from `config.models`) is referenced throughout `summaries.py` and the main config display. The RNG is seeded (`config.seed` or explicit `seed` argument), making runs fully reproducible. The generator tracks `_request_count` for logging and debugging.

## Why It Matters

Without traffic simulation, all measurements assume zero inter-request gap — i.e., every request is sent the instant the previous one completes. This represents a theoretical maximum-throughput scenario that never occurs in production. Poisson-arrival simulation allows researchers to measure energy efficiency and latency at a specified query rate (QPS), which is essential for understanding how LLM inference behaves under realistic API load. The feature is a differentiator from raw throughput benchmarks. If dropped silently, it invalidates any measurement taken with `traffic_simulation.enabled: true` in existing configs.

## Planning Gap Details

The planning documents make no mention of traffic simulation at any level:
- `designs/architecture.md` — no reference
- `decisions/architecture.md` — no reference
- `decisions/cli-ux.md` — no reference (the `llem config` output example also omits it)
- `designs/observability.md` — no reference
- `designs/energy-backends.md` — no reference

The `designs/cli-commands.md` was not read here but is also unlikely to address it given the pattern. The `summaries.py` display code (line 247–277 in the current codebase) actively renders traffic simulation settings in verbose mode, indicating the feature was considered display-worthy — but its v2.0 fate is unresolved.

The config model for `TrafficSimulation` (`target_qps`, `mode`, `seed`, `enabled`) was not listed in the Phase 4.5 "ExperimentConfig removals" session decisions, suggesting it survived the cut — but this was never stated explicitly.

## Recommendation for Phase 5

Make an explicit decision before Phase 5 begins. Two options:

**Option A — Keep at v2.0**: The `TrafficSimulation` config model was not removed in the session 3 cleanup. Keep `TrafficGenerator` and `apply_traffic_delay()` in `core/traffic.py`. Ensure the `traffic_simulation` block survives in `ExperimentConfig`. Wire into the new orchestrator's inference loop (confirm it is still called in `inference_backends/pytorch.py` and vLLM equivalent). Add to the observability design: traffic settings should appear in `llem config`-style pre-experiment summary.

**Option B — Defer to v2.1**: Cut `TrafficSimulation` config block from `ExperimentConfig` in v2.0 (removing ~5 fields). Document as deferred. Risk: existing config files using `traffic_simulation.enabled: true` will break on upgrade.

The existing implementation is clean and requires no rewrite — if kept, it only needs wiring validation. The Poisson arrival logic (lines 68–74) is correct and uses `numpy.random.default_rng` (the modern numpy RNG API).
