---
created: 2026-02-26T13:30:40.079Z
title: Define package contracts layer
area: architecture
files:
  - src/llenergymeasure/core/
  - src/llenergymeasure/api/
  - src/llenergymeasure/cli/
---

## Problem

The package needs a formal contracts layer — a set of canonical types, schemas, path conventions, and config structures that all internal modules depend on. Without this, modules pass raw dicts or ad-hoc types across boundaries, making schema changes invisible and error-prone.

## Solution

- **Pydantic for all contract definitions** — domain types, data schemas, config via Pydantic Settings. Validates structure at module boundaries; Pydantic becomes a core dependency.
- **Dependency direction**: `core/` ← `api/` ← `cli/` (per ADR-0007). Contract definitions in `core/` have zero internal imports — they are the bottom of the dependency graph.
- **All inter-module data exchange uses contract types**, not raw dicts. Schema changes are breaking changes.
- **No business logic in the contract layer** — types and schemas only.
- **Public API surface**: `api/` re-exports contract types directly. Users get typed interfaces without knowing internal structure.
