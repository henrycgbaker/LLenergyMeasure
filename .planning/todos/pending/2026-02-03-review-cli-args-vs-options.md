---
created: 2026-02-03T16:45
title: Review CLI args vs options pattern
area: cli
files:
  - src/llenergymeasure/cli/
---

## Problem

The CLI may have positional arguments implemented as options (flags) where they should be required arguments. This affects usability and discoverability — users expect required inputs to be positional args, while optional modifiers should be flags.

Typer convention:
- **Arguments** (`typer.Argument`): Required positional inputs (e.g., `lem experiment CONFIG_PATH`)
- **Options** (`typer.Option`): Optional flags with defaults (e.g., `--cycles 5`, `--output-dir`)

Review the help screen output and CLI command definitions to identify misclassified parameters.

## Solution

TBD - Requires audit of:
1. `lem experiment` command parameters
2. `lem campaign` command parameters
3. `lem config` subcommands
4. Compare against Typer best practices and similar ML tools (mlflow, wandb, etc.)
