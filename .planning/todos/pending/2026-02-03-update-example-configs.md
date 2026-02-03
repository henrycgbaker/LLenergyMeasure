---
created: 2026-02-03T17:00
title: Update example configs for latest functionality
area: docs
files:
  - configs/examples/
---

## Problem

Example configuration files in `configs/examples/` may not showcase all the latest functionality added in recent phases:

- Campaign execution features (Phase 2)
- Container strategy options (ephemeral/persistent)
- Thermal gap settings
- User config integration (.lem-config.yaml)
- Multi-backend campaign examples
- Cycle/warmup configuration

Users looking at examples should see best practices and full feature coverage.

## Solution

Audit and update:
1. `configs/examples/*.yaml` - experiment configs
2. Add campaign config examples if missing
3. Add `.lem-config.yaml` example with comments explaining each option
4. Ensure each backend (pytorch, vllm, tensorrt) has a representative example
