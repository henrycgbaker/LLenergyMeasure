---
title: Schema refresh (operations guide)
description: Practical guide to running and troubleshooting schema-discovery refreshes. For the conceptual treatment, see the Architecture pages.
---

# Schema refresh (operations guide)

This page covers the practical operations for refreshing an engine's
discovered schema. For the conceptual treatment - the two engine-knowledge
products and why they are produced locally - see
[Architecture overview](/explanation/architecture/architecture-overview) and
[Pipeline architecture](/explanation/architecture/pipeline-architecture).

For the format spec of the `schema.discovered.json` artefact, see
[schema discovered format](/reference/schema-discovered-format).

For the rules-side counterpart, see
[Local knowledge production](/contributing/knowledge-production).

---

## Refreshing a schema

Schema discovery is a local maintainer task - it introspects the engine
inside its Docker image, so it needs Docker (and, for the GPU engines, an
NVIDIA GPU). CI never runs discovery; it only verifies the committed bytes.

```bash
./scripts/refresh_discovered_schemas.sh <engine>     # vllm | tensorrt | transformers
```

Equivalently, `make discover-schema ENGINE=<engine>`.

The script selects the discovery image from the engine's pinned version
(`vllm/vllm-openai:<ver>` for vllm, `nvcr.io/nvidia/tensorrt-llm/release:<ver>`
for tensorrt, and the first-party `llenergymeasure:transformers-<ver>` image
for transformers - building it if absent), runs
`python -m scripts.engine_producers._schemas_runner` inside the container, and
writes the result to
`src/llenergymeasure/engines/<engine>/schema.discovered.json`. It prints the
`git diff` but does **not** commit: review the diff, `git add`, and open a PR.

After a schema refresh, regenerate the typed config model
(`config.py`) so the two stay in step - the config codegen check in CI fails
otherwise. See [Architecture overview](/explanation/architecture/architecture-overview).

---

## Version-bump guard

If a developer bumps an engine version `ARG` in a Dockerfile without
re-running discovery, the `schema-version-check` job in `ci.yml` catches the
mismatch. The job runs on a hosted CPU runner, is gated on changes under
`docker/`, skips Renovate-labelled PRs, and runs
`scripts/check_discovered_schema_versions.py` per engine. On failure, re-run
discovery locally and commit the refreshed schema:

```bash
./scripts/refresh_discovered_schemas.sh <engine>
```

---

## Prerequisites

- Docker plus the NVIDIA Container Toolkit on the machine running discovery
  (the GPU engines introspect inside a `--gpus all` container).
- [Mend Renovate](https://github.com/apps/renovate) GitHub App installed on
  the repo (free for open source) opens the bump PRs; detection is automatic.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Renovate not detecting bumps | `fileMatch` pattern doesn't cover the Dockerfile | Check Renovate dashboard, verify `docker/Dockerfile\\..*` matches |
| Renovate not detecting transformers bumps | `customManagers` regex not matching | Verify `ARG TRANSFORMERS_VERSION=X.Y.Z` format in `docker/Dockerfile.transformers` |
| Schema discovery fails to import engine | Container missing `--gpus all` | Verify the machine has NVIDIA drivers plus the Container Toolkit |
| Version guard fails on non-version change | Should not happen - guard only compares version ARGs | If it does, check `_parse_arg` regex in `scripts/check_discovered_schema_versions.py` |
| NGC registry auth failure | Private image or rate-limited | Add `hostRules` to `renovate.json` |
| Schema unchanged after discovery | Engine version did not change params | Expected: the script reports no changes and commits nothing |
| Re-discovery shows a 2-line diff on unchanged source | `LLENERGY_DISCOVERY_FROZEN_AT` not set | The introspector writes a fresh wallclock `discovered_at` on every invocation; set the env var to a stable anchor (typically the author date of the most recent commit touching any input path) so re-discovery is byte-stable |

---

## Adding a new engine

For the full picture of adding a new engine to llem (the plugin, the
Dockerfile-or-upstream choice, the pin, and curation), see
[engine extensibility](/explanation/architecture/engine-extensibility).

The schema-discovery surface a new engine adds is small:

1. Add a per-engine module under `scripts/engine_producers/`, mirroring an
   existing `*_introspector.py`, and register it in
   `scripts/engine_producers/__init__.py`.
2. Add a case to `scripts/refresh_discovered_schemas.sh`.
3. Run discovery once: `./scripts/refresh_discovered_schemas.sh <engine>`.
4. Add a Renovate `packageRule` in `renovate.json`.
5. If the Dockerfile `ARG` maps directly to the engine version, add an entry
   to `_ENGINE_SPECS` in `scripts/check_discovered_schema_versions.py`.

---

## See also

- [Architecture overview](/explanation/architecture/architecture-overview) - the two engine-knowledge products
- [Schema discovered format](/reference/schema-discovered-format) - the JSON envelope spec
- [Local knowledge production](/contributing/knowledge-production) - the rules-side operations guide
- [Docker setup](/how-to/docker-setup) - building engine images locally
