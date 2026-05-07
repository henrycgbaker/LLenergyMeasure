---
title: Engine extensibility
description: How to add a new engine to LLenergyMeasure.
---

# Engine extensibility

> This page is a placeholder. Full content coming in the next documentation update.

Adding a new engine requires implementing the `BackendPlugin` protocol,
writing a parameter miner for the engine's config classes, and wiring
the engine into the Docker image and composition layer.

For the practical how-to, see
[Contributing: extending miners](/contributing/extending-miners).
For the harness contract your plugin must satisfy, see
[Harness and plugin model](/explanation/architecture/harness-plugin).
