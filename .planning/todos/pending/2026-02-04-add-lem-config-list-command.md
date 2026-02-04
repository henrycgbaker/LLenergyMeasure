---
created: 2026-02-04T14:30
title: Add `lem config list` CLI command
area: cli
files:
  - src/llenergymeasure/cli/config.py
---

## Problem

Currently there's no easy way to list available configuration files in `configs/`. Users need to manually browse the directory to find example configs.

A `lem config list` command would improve discoverability and help users find example configs to base experiments on.

## Solution

Add subcommand to existing `lem config` command group:
- `lem config list` — List YAML files in configs/ directory
- Follow patterns from `lem presets` and `lem datasets` listing commands
