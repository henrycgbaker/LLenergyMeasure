---
phase: 02-campaign-orchestrator
plan: 06
subsystem: campaign-orchestration
tags: [campaign, docker-exec, manifest, grid, daemon, cold-start, health-check, bootstrap-ci]
depends_on:
  requires: ["02-01", "02-02", "02-03", "02-04", "02-05"]
  provides: "Integrated campaign orchestrator with Docker exec, manifest tracking, grid support, daemon mode"
  affects: ["02-08"]
tech-stack:
  added: []
  patterns: ["daemon-scheduling", "manifest-state-tracking", "exec-dispatch", "bootstrap-ci-display"]
key-files:
  modified:
    - src/llenergymeasure/orchestration/campaign.py
    - src/llenergymeasure/cli/campaign.py
decisions:
  - "Grid-based experiments use synthetic config_path '<grid:config_name>' to distinguish from file-based"
  - "Daemon mode runs in foreground process; users use nohup/screen for background"
  - "Cold start cache clear uses torch.cuda.empty_cache() + gc.collect() via exec"
  - "ContainerManager lazy import with fallback to legacy docker compose run --rm"
  - "_run_campaign_loop extracts full execution flow for daemon reuse"
metrics:
  duration: "7 min"
  completed: "2026-01-29"
---

# Phase 2 Plan 6: Campaign Orchestrator Integration Summary

Wired all Wave 1+2 modules (ContainerManager, ManifestManager, grid expansion, bootstrap CIs, daemon config) into the existing CampaignRunner and campaign CLI to create the integrated campaign orchestrator.

## What Changed

### orchestration/campaign.py (217 lines added)

Extended CampaignRunner with 6 new methods while preserving all existing functionality:

- `generate_execution_order_from_grid()` -- expands grid config into CampaignExperiment objects with cycle expansion and ordering (interleaved/shuffled/grouped)
- `_apply_ordering()` -- shared helper for cycle expansion + structure ordering
- `create_manifest()` -- creates CampaignManifest with config hashes and links manifest entries to experiments
- `apply_resume_filter()` -- filters execution order to pending/failed experiments from existing manifest
- `should_health_check()` -- returns True per-cycle or at interval_experiments frequency
- `should_cold_start()` -- reads campaign.cold_start.force_cold_start config

Added `manifest_entry: CampaignManifestEntry | None` field to CampaignExperiment dataclass.

### cli/campaign.py (428 lines added)

**New CLI options:**
- `--resume/--no-resume` -- resume from existing manifest
- `--force-cold-start/--no-force-cold-start` -- override cold start config
- `--validate-only` -- validate grid and exit
- `--quiet/-q` -- suppress interactive output (daemon mode)

**Grid support:**
- Grid-based campaigns trigger `expand_campaign_grid()` + `validate_campaign_grid()` before execution
- `_display_validation_summary()` shows generated/valid/filtered/warnings in Rich panel
- `--validate-only` exits after validation summary

**Docker exec dispatch:**
- `_run_single_experiment()` accepts optional `container_mgr` parameter
- When provided, dispatches via `ContainerManager.execute_experiment()` (exec into running container)
- Falls back to legacy `docker compose run --rm` subprocess if python-on-whales not installed
- Grid-based experiments serialise config from Pydantic model instead of reading YAML file

**Manifest tracking:**
- ManifestManager created at campaign start, tracks every experiment lifecycle
- Status updates: pending -> running -> completed/failed (atomic persistence after each)
- Resume: loads existing manifest, links entries to execution order, filters to remaining

**Health checks:**
- After each cycle (or every N experiments), runs `check_and_recover()` on active backends
- Logs warnings for unhealthy services, continues campaign (log-and-continue pattern)

**Cold start:**
- After each experiment when force_cold_start enabled
- Container restart (restart_container=True) or GPU cache clear (default)

**Daemon scheduling:**
- `_wait_until_time()` -- sleeps until target HH:MM, wraps to next day if past
- `_run_campaign_loop()` -- encapsulates full manifest+execution+teardown for reuse
- Daemon wrapper: wait for start time, repeat campaign at interval, enforce total_duration

**Bootstrap CI display:**
- `_display_campaign_ci_summary()` -- loads AggregatedResult from manifest result paths
- Groups by config_name, calls `aggregate_campaign_results()` for bootstrap CIs
- Rich table: config, cycles, energy [CI], throughput [CI], TTFT [CI], ITL [CI]
- Single-cycle campaigns show warning about missing CIs

## Commits

| # | Hash | Description |
|---|------|-------------|
| 1 | 4e0b1e6 | Extend CampaignRunner with grid, manifest, resume, health check, cold start |
| 2 | 3cbd658 | Add CLI options and grid support to campaign command |
| 3 | 32bcc51 | Wire Docker exec dispatch and manifest tracking into CLI |
| 4 | 179b7d1 | Add health checks, cold start, daemon scheduling, CI display |

## Deviations from Plan

None -- plan executed exactly as written.

## Decisions Made

1. **Synthetic grid config paths**: Grid-based experiments use `<grid:config_name>` as config_path to distinguish from file-based experiments. `_run_single_experiment` checks for this prefix to serialise from Pydantic model instead of reading YAML.

2. **Daemon foreground process**: Daemon mode runs in the calling process (no fork/detach). Users use `nohup`, `screen`, or `tmux` for background execution. This avoids process management complexity.

3. **ContainerManager lazy fallback**: If python-on-whales is not installed, campaign falls back to legacy `docker compose run --rm` with a warning. No hard dependency on the campaign extras.

4. **Extracted `_run_campaign_loop`**: Daemon mode needs to repeat the full execution flow, so the manifest+containers+execution+teardown logic is extracted into a reusable function.

## Verification

- `python -c "from llenergymeasure.cli.campaign import campaign_cmd"` -- passes
- All 16 existing campaign unit tests pass
- New CLI options visible in function signature
- CampaignRunner has all 5 new methods
- CampaignExperiment has manifest_entry field

## Next Phase Readiness

Plan 02-08 (integration tests) can proceed. All campaign orchestrator components are wired together. The only untested paths are runtime Docker execution (requires GPU + Docker) and daemon scheduling (requires time-based waiting).
