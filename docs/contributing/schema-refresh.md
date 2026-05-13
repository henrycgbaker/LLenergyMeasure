---
title: Schema refresh (operations guide)
description: Practical guide to triggering and troubleshooting schema-discovery refreshes. For the conceptual treatment, see the Architecture page.
---

# Schema refresh (operations guide)

This page covers the practical operations for refreshing engine
schemas. For the conceptual treatment of how schema discovery works
(and how it parallels the invariant-mining pipeline), see
[engine introspection pipelines](/explanation/architecture/engine-introspection-pipelines).

For the format spec of the `schema.discovered.json` artefact, see
[schema discovered format](/reference/schema-discovered-format).

For the operations side of the parallel mining pipeline, see
[miner pipeline (debugging guide)](/contributing/miner-pipeline).

---

## Manual refresh (workflow_dispatch)

For ad-hoc refreshes outside the Renovate flow:

```bash
# vllm or tensorrt
gh workflow run engine-pipeline.yml \
  --field engine=vllm \
  --field pr_number=123

# transformers: trigger Build engine image. Publish engine image fires
# on its success (workflow_run); schemas-transformers and
# invariants-transformers then fire on the push's success (also
# workflow_run).
gh workflow run engine-pipeline.yml
```

For local refresh against an installed engine:

```bash
./scripts/refresh_discovered_schemas.sh <engine>
```

The script runs `python -m scripts.engine_producers._schemas_runner --engine <engine>`
inside the matching container and writes the result to
`src/llenergymeasure/engines/<engine>/schema.discovered.json`.

---

## Manual version-bump guard

If a developer bumps an engine version ARG in a Dockerfile without
running discovery, the `schema-version-check` job in `ci.yml` catches
the mismatch (path-filtered to `docker/Dockerfile.*`, skips Renovate
PRs). On failure, the developer can either:

- Run locally: `./scripts/refresh_discovered_schemas.sh <engine>`
- Trigger remotely:
  `gh workflow run engine-pipeline.yml --field engine=<engine> --field pr_number=<N>`
  (for transformers, the chain re-fires automatically once the build
  completes).

---

## Prerequisites

- [Mend Renovate](https://github.com/apps/renovate) GitHub App installed
  on the repo (free for open-source).
- Self-hosted GPU runner available for the introspection jobs.
- Docker plus NVIDIA Container Toolkit on the runner.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Renovate not detecting bumps | `fileMatch` pattern doesn't cover the Dockerfile | Check Renovate dashboard, verify `docker/Dockerfile\\..*` matches |
| Renovate not detecting transformers bumps | `customManagers` regex not matching | Verify `ARG TRANSFORMERS_VERSION=X.Y.Z` format in `docker/Dockerfile.transformers` |
| Schema discovery fails to import engine | Container missing `--gpus all` | Verify GPU runner has NVIDIA drivers plus Container Toolkit |
| Version guard fails on non-version change | Should not happen - guard only compares version ARGs | If it does, check `_parse_arg` regex in `scripts/check_discovered_schema_versions.py` |
| NGC registry auth failure | Private image or rate-limited | Add `hostRules` to `renovate.json` |
| Schema unchanged after discovery | Engine version did not change params | Expected: workflow commits nothing and exits cleanly |
| Discovery commits a 2-line diff on every CI run | `LLENERGY_DISCOVERY_FROZEN_AT` not set | The introspector writes a fresh wallclock `discovered_at` on every invocation; set the env var to a stable anchor (typically the author date of the most recent commit touching any input path) |

---

## Adding a new engine

For step-by-step instructions on adding a new engine to llem (which
covers both the introspection-pipeline registration and the broader
plugin / Pydantic / Dockerfile work), see
[engine extensibility](/explanation/architecture/engine-extensibility)
and [extending miners](/contributing/extending-miners).

The introspection-pipeline-specific surface is small:

1. Add a per-engine module under `scripts/engine_producers/`,
   mirroring an existing `*_introspector.py`, and register it in
   `scripts/engine_producers/__init__.py`.
2. Add a case to `scripts/refresh_discovered_schemas.sh`.
3. Run discovery once: `./scripts/refresh_discovered_schemas.sh <engine>`.
4. Add a Renovate `packageRule` in `renovate.json`.
5. If the Dockerfile ARG maps directly to the engine version, add an
   entry to `_ENGINE_SPECS` in
   `scripts/check_discovered_schema_versions.py`.

---

## See also

- Architecture: [engine introspection pipelines](/explanation/architecture/engine-introspection-pipelines) - how schema discovery works (conceptual)
- Reference: [schema discovered format](/reference/schema-discovered-format) - the JSON envelope spec
- Reference: [invariants corpus format](/reference/invariants-corpus-format) - the parallel pipeline's format spec
- Contributing: [miner pipeline (debugging guide)](/contributing/miner-pipeline) - the parallel pipeline's debug guide
- Contributing: [extending miners](/contributing/extending-miners) - adding rules for a new engine
- How-to: [Docker setup](/how-to/docker-setup) - building engine images locally
