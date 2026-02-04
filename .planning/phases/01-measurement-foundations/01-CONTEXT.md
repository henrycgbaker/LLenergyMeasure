# Phase 1 Context: Measurement Foundations

Decisions from discussion — guides research and planning agents.

## 1. Baseline Power Measurement

| Decision | Choice |
|----------|--------|
| When | Auto + cached (measure once, reuse within session) |
| Staleness | Per terminal session — re-measure on new session |
| Failure | Configurable: `baseline.required` (default `false` → warn + continue with raw values) |
| Storage | **Both fields**: `raw_energy` and `adjusted_energy` plus baseline metadata (`baseline_power_watts`, `baseline_method`, `baseline_timestamp`) |

**Rationale**: Auto+cached avoids ~30s overhead per experiment. Both fields preserve backwards compatibility and full auditability.

## 2. Thermal Throttling

| Decision | Choice |
|----------|--------|
| Response | Flag + continue (experiment completes, user decides post-hoc) |
| Detection | NVML throttle reasons API (authoritative, distinguishes thermal vs power vs other) |
| Granularity | Rich metadata: throttle timestamps, duration, severity (clock % reduction), cause |
| Scope | **Per-experiment only** — campaign-level response (retry, skip) is Phase 2 |

**Rationale**: Experiments are the atomic unit. Phase 1 detects and records; Phase 2 orchestrates responses.

## 3. Warmup Convergence

| Decision | Choice |
|----------|--------|
| Max cap | User-configurable (`warmup.max_prompts`, sensible default e.g. 50) |
| CV threshold | Configurable with default (e.g., CV < 5%) |
| Feedback | Progress bar showing: prompts completed, current CV, target CV |
| Non-convergence | Warning flag: `warmup_converged: false` + final CV value in results |

**Rationale**: Cap prevents runaway warmup on noisy hardware. Progress bar gives real-time confidence that convergence is happening.

## 4. Results & Metrics Presentation

| Decision | Choice |
|----------|--------|
| Field strategy | **Richer nested structure**: `energy: {raw, adjusted, baseline}` — breaking change acceptable for v2.0.0 |
| Time-series | Optional export (`--save-timeseries` flag or YAML config), saved as **separate file** from main results JSON |
| CSV columns | Grouped prefixes: `energy_raw_kwh`, `energy_adjusted_kwh`, `gpu_util_pct`, `gpu_mem_gb`, etc. |
| Environment metadata | Full block in results JSON; one-line summary in CLI (e.g., `A100 80GB | CUDA 12.2 | Driver 535.104`) |

**Rationale**: Nested structure is cleaner than flat field proliferation. Separate time-series file keeps results lightweight. Grouped prefixes sort well in spreadsheets.

## Architectural Decision: Campaign-Cycle Model

**Key clarification** — the hierarchy is:

```
Campaign (user interaction level)
  └── Cycle 1 (repetition of entire campaign for statistical robustness)
  │     ├── Experiment A (atomic unit — one config, one run)
  │     ├── Experiment B
  │     └── Experiment C
  └── Cycle 2
        ├── Experiment A
        ├── Experiment B
        └── Experiment C
```

- **Experiments** are the atomic unit of measurement
- **Cycles** are campaign-level repetitions (all experiments run again), NOT per-experiment repetitions
- **Campaigns** are the user-interaction level

**Impact on Phase 1**: Phase 1 focuses on single-experiment measurement accuracy. No cycle logic needed.

**Impact on MEAS-08**: Bootstrap resampling / confidence intervals require multiple observations of the same experiment across cycles → moves to Phase 2 (campaigns).

## Deferred Ideas

| Idea | Deferred to | Reason |
|------|-------------|--------|
| Campaign-level throttle response (retry, skip, continue) | Phase 2 | Requires campaign orchestrator |
| MEAS-08: Bootstrap resampling / confidence intervals | Phase 2 | Requires cycles, which are a campaign concept |
| Per-cycle throttle tracking | Phase 2 | Cycles belong to campaigns |

## Scope Boundaries

Phase 1 delivers:
- Single-experiment measurement accuracy (baseline subtraction, throttle detection, environment metadata)
- Warmup convergence (CV-based, per experiment)
- Extended results schema with nested structure
- Time-series optional export
- Extended CSV export with grouped prefixes
- UAT round 1

Phase 1 does NOT include:
- Campaign orchestration (Phase 2)
- Cycle management (Phase 2)
- Confidence intervals / bootstrap resampling (Phase 2 — needs cycles)
- Campaign-level throttle response (Phase 2)
