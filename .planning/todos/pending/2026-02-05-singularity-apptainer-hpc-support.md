---
created: 2026-02-05T18:30
title: Add Singularity/Apptainer support for HPC environments
area: infrastructure
files:
  - src/llenergymeasure/orchestration/docker_dispatch.py
  - src/llenergymeasure/detection/docker_detection.py
  - docker-compose.yml
---

## Problem

HPCs typically use Singularity/Apptainer, not Docker. Docker requires a daemon with root privileges, which HPC admins don't allow. The current container dispatch is Docker-specific (`docker compose run --rm`), so the CLI cannot orchestrate containerised backend execution on HPC clusters.

This is a significant gap for the target user base (ML researchers), many of whom run workloads on university/institutional HPCs.

## Solution

Container-runtime abstraction layer that detects available runtime (Docker vs Singularity/Apptainer) and dispatches accordingly. Existing Docker images can be converted to Singularity SIF files (`singularity pull docker://...`). Dispatch logic needs to support both `docker compose run --rm` and `singularity run` patterns.

Post-v2.0 roadmap item. Docker-only for v2.0; Singularity support as a future phase.
