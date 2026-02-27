---
phase: 02-campaign-orchestrator
plan: 03
subsystem: results
tags: [bootstrap, confidence-intervals, statistics, campaign-aggregation]
dependency_graph:
  requires: []
  provides: [bootstrap_ci, BootstrapResult, aggregate_campaign_results]
  affects: [02-08]
tech_stack:
  added: []
  patterns: [bootstrap-resampling, percentile-method, graceful-degradation]
key_files:
  created:
    - src/llenergymeasure/results/bootstrap.py
  modified:
    - src/llenergymeasure/results/aggregation.py
decisions: []
metrics:
  duration: 3 min
  completed: 2026-01-29
---

# Phase 02 Plan 03: Bootstrap CI and Campaign Aggregation Summary

Bootstrap resampling (1000 iterations, seed=42, percentile method) for 95% confidence intervals, plus campaign-level cross-cycle aggregation with CIs for energy, throughput, tokens, TTFT, and ITL.

## Tasks Completed

| # | Task | Commit | Key Changes |
|---|------|--------|-------------|
| 1 | Create bootstrap CI module | 94ee6d5 | New `bootstrap.py` with `BootstrapResult` model, `bootstrap_ci()`, `compute_metric_ci()` |
| 2 | Add campaign-level aggregation with CIs | 10d74a7 | `aggregate_campaign_results()` in `aggregation.py` |

## Key Implementation Details

**bootstrap.py**: `BootstrapResult` Pydantic model holds mean, std, CI bounds, sample count, confidence level, and optional warning. `bootstrap_ci()` uses `np.random.default_rng(seed)` for deterministic resampling. Graceful degradation: 1 sample returns no CI, 2 samples returns CI with warning, 3+ returns reliable CI.

**aggregate_campaign_results()**: Takes `dict[str, list[AggregatedResult]]` (config name to cycle results), extracts `total_energy_j`, `avg_tokens_per_second`, `total_tokens` from each cycle, and calls `bootstrap_ci()` per metric. Also extracts `ttft_mean_ms` and `itl_mean_ms` from `latency_stats` when present (streaming mode).

## Verification Results

- All imports succeed
- Bootstrap determinism confirmed (same seed = same CI bounds)
- 5 samples: mean=11.30, CI=[10.40, 12.30]
- 2 samples: warning about unreliable CI
- 1 sample: no CI computed
- 16 existing aggregation tests pass
- ruff check + format pass, mypy passes

## Deviations from Plan

None - plan executed exactly as written.
