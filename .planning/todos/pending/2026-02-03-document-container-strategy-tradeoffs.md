---
created: 2026-02-03T16:55
title: Document container strategy tradeoffs
area: docs
files:
  - docs/deployment.md
  - src/llenergymeasure/cli/campaign.py
---

## Problem

The --container-strategy flag (ephemeral vs persistent) lacks user-facing documentation explaining the tradeoffs:

- **ephemeral** (default): Complete isolation, reproducible results. Each experiment runs in a fresh container (`docker compose run --rm`). No state leakage between experiments.

- **persistent**: Faster execution (no container startup overhead). Containers stay running (`docker compose up` + `exec`). Risk of state leakage (GPU memory, model caching) affecting measurements.

Users need to understand when to use each and the implications for research reproducibility.

## Solution

1. Add tradeoff explanation to docs/deployment.md under Docker section
2. Consider adding brief explanation to `lem campaign --help` output
3. Warning message for persistent mode should mention reproducibility concern
