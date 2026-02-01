---
phase: 02-campaign-orchestrator
plan: 08
subsystem: tests-uat
tags: [unit-tests, uat, campaign, bootstrap, manifest, grid, container]
completed: 2026-01-30
duration: 5 min
requires: [02-06, 02-07]
provides: [phase-2-validation]
affects: [02.1]
tech-stack:
  added: []
  patterns: [mocked-docker, tmp_path-fixtures, pydantic-validation-tests]
key-files:
  created:
    - tests/unit/orchestration/test_container.py
    - tests/unit/orchestration/test_manifest.py
    - tests/unit/orchestration/test_grid.py
    - tests/unit/results/test_bootstrap.py
    - tests/unit/config/test_campaign_config.py
  modified:
    - src/llenergymeasure/cli/campaign.py
decisions:
  - "Docker dispatch UAT deferred to Phase 2.1 — _should_use_docker() always returns False (known bug), .env missing (auto-generation not yet implemented)"
  - "Added file existence check before campaign-vs-experiment YAML detection to fix misleading error"
  - "Added execution mode display (Docker exec / Docker run / Local) to campaign CLI output"
---

# Phase 2 Plan 8: Unit Tests + UAT Summary

68 unit tests across 5 files covering all Phase 2 modules. UAT validated local campaign execution (dry-run, grid validation, 2-cycle interleaved campaign). Docker dispatch UAT deferred to Phase 2.1.

## What Was Done

### Task 1: Unit tests for all Phase 2 modules

Created 5 test files with 68 tests total:

- **test_bootstrap.py** (9 tests): Determinism, boundary cases (1/2/100 samples), confidence levels, constant values, serialisation
- **test_manifest.py** (13 tests): Entry creation, status tracking, get_remaining, round-trip persistence, atomic writes, config hash change detection
- **test_grid.py** (11 tests): Single/multi backend expansion, backend overrides, models axis, empty grid, base config merge, nested keys, Pydantic validation, summary
- **test_container.py** (13 tests): Mocked DockerClient — init, start_services, execute_experiment, health check (healthy/unhealthy/custom threshold), restart, teardown, context manager
- **test_campaign_config.py** (22 tests): Backwards compat, grid mode, validation, health check defaults, cold start, IO paths, daemon scheduling/parsing

All 935 unit tests pass (0 failures, 0 regressions).

### Task 2: UAT checkpoint

**Verified (local execution):**
- `lem campaign uat_campaign.yaml --dry-run` — grid validation (1 valid, 0 filtered), execution plan table
- `lem campaign uat_campaign.yaml --dataset alpaca -n 5` — 2-cycle campaign executed successfully

**UAT findings fixed:**
1. Missing file gives misleading error ("campaign-name required" instead of "file not found") → Added explicit file existence check
2. CLI doesn't show execution mode → Added "Execution: Docker exec / Docker run / Local" line

**Deferred to Phase 2.1:**
- Docker dispatch end-to-end (ContainerManager, docker compose exec) — blocked by `_should_use_docker()` always returning False and missing `.env` auto-generation

## Deviations

- Docker UAT deferred: Phase 2.1 specifically exists to fix the Docker detection and `.env` issues that block this test path. Unit tests verify ContainerManager logic with mocked Docker.

## Test Coverage

| Module | Tests | Key Assertions |
|--------|-------|----------------|
| bootstrap.py | 9 | Determinism, boundary cases, CI width ordering |
| manifest.py | 13 | Round-trip persistence, atomic writes, resume filtering |
| grid.py | 11 | Cartesian product correctness, Pydantic validation |
| container.py | 13 | Docker API calls, health threshold logic |
| campaign_config.py | 22 | Backwards compat, validation, daemon parsing |
