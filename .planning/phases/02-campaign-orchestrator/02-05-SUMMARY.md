---
phase: 02-campaign-orchestrator
plan: 05
subsystem: orchestration
tags: [grid-expansion, cartesian-product, pydantic-validation, ssot, campaign]

requires:
  - phase: 02-01
    provides: CampaignGridConfig with backends, models, shared, backend_overrides
provides:
  - expand_campaign_grid() for cartesian product generation from grid config
  - validate_campaign_grid() for Pydantic dry-run + SSOT hardware warnings
  - GridExpansionResult with valid/filtered/warned configs and summary
affects: [02-06, 02-07, 02-08]

tech-stack:
  added: []
  patterns: [pydantic-dry-run-validation, two-tier-grid-validation]

key-files:
  created:
    - src/llenergymeasure/orchestration/grid.py
  modified: []

key-decisions:
  - "Used Sequence[str | None] type annotation to satisfy mypy list invariance with models axis"

patterns-established:
  - "Two-tier validation: Pydantic instantiation (Tier 1 catches schema errors) + SSOT introspection (Tier 2 adds hardware warnings)"
  - "Backend-specific overrides expand as nested cartesian product per backend"

metrics:
  duration: 3 min
  completed: 2026-01-29
---

# Phase 02 Plan 05: Grid Expansion and Validation Summary

**Campaign grid expansion with Pydantic dry-run validation and SSOT hardware warnings — generates cartesian product from two-level grid config (shared + backend-specific params).**

## Tasks Completed

| # | Task | Commit | Key Changes |
|---|------|--------|-------------|
| 1 | Create grid expansion module | 499ec6c | grid.py: expand_campaign_grid, validate_campaign_grid, GridExpansionResult |

## Key Implementation Details

### Grid Expansion (`expand_campaign_grid`)
- Generates cartesian product: backends x models x shared_params x backend_overrides
- Backend-specific overrides in `backend_overrides[backend]` expand as nested cartesian product
- Supports nested key paths via dot notation (e.g., `decoder.preset`)
- Returns list of raw config dicts ready for Pydantic validation

### Grid Validation (`validate_campaign_grid`)
- **Tier 1 (Pydantic)**: Attempts `ExperimentConfig(**config_dict)` — catches schema errors (e.g., tensorrt + float32)
- **Tier 2 (SSOT)**: Checks `get_param_skip_conditions()` for hardware/environment warnings
- Invalid configs filtered with human-readable reasons; warned configs kept in valid set
- `GridExpansionResult.summary` property provides one-line status

### Verified Behaviours
- Basic expansion: 1 backend x 2 precisions = 2 configs
- Multi-backend: pytorch(2 batch_sizes) + vllm(2 max_num_seqs) = 4 configs
- Pydantic filtering: tensorrt + float32 correctly rejected
- Nested dot notation: `decoder.preset` sets nested dict correctly
- Model sweep: 2 models x 1 precision = 2 configs

## Deviations from Plan

None — plan executed exactly as written.

## Next Phase Readiness

Grid expansion is ready for integration into campaign runner (02-06/07/08).
