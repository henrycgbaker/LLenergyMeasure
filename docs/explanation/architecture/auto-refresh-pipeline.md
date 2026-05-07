---
title: Auto-refresh pipeline
description: How engine schemas and invariants are automatically refreshed on library bumps.
---

# Auto-refresh pipeline

> This page is a placeholder. Full content coming in the next documentation update.

The auto-refresh pipeline runs on every Renovate-driven library version bump.
It re-runs the parameter miners, regenerates schema digests, and opens a PR
with the updated corpus. The validation-CI gate then checks that every
invariant in the corpus is still satisfied by the new library version.

For the miner implementation details, see
[Contributing: miner pipeline](/contributing/miner-pipeline).
For the CI wiring, see
[Architecture: CI architecture](/explanation/architecture/ci-architecture).
