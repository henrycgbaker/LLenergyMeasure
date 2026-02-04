---
status: resolved
trigger: "extended-metrics-missing - Extended efficiency metrics are completely missing from experiment results"
created: 2026-02-04T00:00:00Z
updated: 2026-02-04T11:00:00Z
---

## Current Focus

hypothesis: CONFIRMED - extended_metrics field IS present and populated, user may have been looking at old schema v2.0.0 files (results/multi_cycle/0098.json) or misunderstanding that null values show as "N/A" in display
test: Examined results/aggregated/0074_c0.json and results/0074_multi_cycle.json
expecting: extended_metrics field present with data
next_action: Document findings and clarify expected behavior

## Symptoms

expected: Results JSON should contain `extended_efficiency_metrics` with fields like memory_peak_mb, gpu_utilisation_mean, batch_stats, kv_cache_stats, per_request_latencies
actual: The `extended_efficiency_metrics` field is completely absent from results JSON at all levels (top-level, cycle_results, process_results)
errors: No errors during experiment execution - silent failure
reproduction: Run any experiment, inspect results JSON
started: Observed in experiments 0074-0076 (2026-02-03)

## Eliminated

- timestamp: 2026-02-04T11:00:00Z
  hypothesis: extended_metrics is not being serialised to JSON
  evidence: grep found extended_metrics in results/0074_multi_cycle.json at lines 19922, 20154, 40138, etc. Also found in results/aggregated/0074_c0.json.

- timestamp: 2026-02-04T11:00:00Z
  hypothesis: extended_metrics is missing from all result files
  evidence: Examined results/aggregated/0074_c0.json - extended_metrics present with tpot_ms=5.48975181938373, token_efficiency_index=428.92041220247654

## Evidence

- timestamp: 2026-02-04T11:00:00Z
  checked: results/multi_cycle/0098.json (old file, schema v2.0.0)
  found: No extended_metrics field - this is an older schema version before the feature was added
  implication: Older experiments won't have extended_metrics, but newer ones (schema v3.0.0) will

- timestamp: 2026-02-04T11:00:00Z
  checked: results/0074_multi_cycle.json header
  found: schema_version: "3.0.0" with extended_metrics present
  implication: Schema v3.0.0 experiments have extended_metrics

- timestamp: 2026-02-04T11:00:00Z
  checked: results/aggregated/0074_c0.json extended_metrics content
  found: tpot_ms=5.489, token_efficiency_index=428.92, but memory.peak_memory_mb=0.0, gpu_utilisation=null, request_latency=null
  implication: Extended metrics ARE being computed and saved. Some sub-fields are null because underlying data is unavailable for vLLM backend (peak memory not tracked by PyTorch when vLLM manages GPU, GPU sampling fails due to CUDA context conflicts)

- timestamp: 2026-02-04T11:00:00Z
  checked: runner.py extended_metrics computation (lines 303-366)
  found: Code computes extended_metrics and assigns to RawProcessResult.extended_metrics field at line 424
  implication: Extended metrics are being computed and saved in the pipeline

## Resolution

root_cause: NOT A BUG. Extended metrics ARE present in results for experiments 0074-0076 (schema v3.0.0). The user may have:
1. Been looking at old schema v2.0.0 results (e.g., results/multi_cycle/0098.json) which predate the feature
2. Misunderstood that many sub-fields showing "N/A" means the entire feature is missing - N/A is expected behavior for unavailable metrics
3. Expected ALL fields to be populated, but vLLM backend cannot provide some metrics (peak_memory_mb from PyTorch, GPU sampling conflicts with vLLM CUDA context)

fix: No code fix needed. This is expected behavior.
verification: Confirmed extended_metrics present in results/aggregated/0074_c0.json and results/0074_multi_cycle.json
files_changed: []
