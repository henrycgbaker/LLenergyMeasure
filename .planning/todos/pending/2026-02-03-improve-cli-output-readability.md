---
created: 2026-02-03T17:15
title: Improve CLI output readability for vLLM/TensorRT
area: cli
files:
  - src/llenergymeasure/cli/experiment.py
  - src/llenergymeasure/cli/campaign.py
---

## Problem

CLI output during experiments (especially vLLM and TensorRT backends) can be noisy and hard to parse. Users need clearer, more useful output with better verbosity controls.

Current state:
- `--verbose` and `--quiet` flags exist at top level
- Backend-specific output (model loading, compilation) can be overwhelming
- Progress indicators may not be clear during long operations
- vLLM/TensorRT have different output patterns than PyTorch

## Solution

1. Audit current output for each backend (PyTorch, vLLM, TensorRT)
2. Design three-tier verbosity:
   - `--quiet`: Errors only, minimal progress
   - Standard (default): Key milestones, progress bars, summary stats
   - `--verbose`: Full logs with timestamps, debug info
3. Consider structured output (JSON mode for scripting)
4. Filter/format backend-specific noise (model loading, compilation progress)
5. Ensure campaign output clearly shows per-experiment status
