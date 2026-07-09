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

## The pin is the SSOT

An engine's version is pinned in one place: its `current.yaml`.

```yaml
# engine_versions/vllm/current.yaml
schema_version: "4.0"
engine: vllm
library:
  pep503_name: vllm
  current_version: 0.19.1
```

Every downstream artefact resolves the version from
`library.current_version` here: the discovery script picks the container
image, the codegen check picks the snapshot directory, and the CI checks read
it to know what to verify against. Bumping an engine means editing this one
field, then re-running the two local production steps below and committing the
results. There is no Dockerfile `ARG` to keep in sync and no separate version
constant anywhere else.

---

## Refreshing a schema

Schema discovery is a local maintainer task - it introspects the engine
inside its Docker image, so it needs Docker (and, for the GPU engines, an
NVIDIA GPU). CI never runs discovery; it only verifies the committed bytes.

```bash
./scripts/refresh_discovered_schemas.sh <engine>     # vllm | tensorrt | transformers
```

Equivalently, `make discover-schema ENGINE=<engine>`.

The script reads the pin from `engine_versions/<engine>/current.yaml`
(`yq '.library.current_version'`), selects the discovery image from it
(`vllm/vllm-openai:<ver>` for vllm, `nvcr.io/nvidia/tensorrt-llm/release:<ver>`
for tensorrt, and the first-party `llenergymeasure:transformers-<ver>` image
for transformers - building it from `docker/Dockerfile.transformers` if
absent), runs `python -m scripts.engine_producers._schemas_runner` inside the
container, and writes the result to
`src/llenergymeasure/engines/<engine>/schema.discovered.json`. It prints the
`git diff` but does **not** commit: the committed JSON is the canonical SSOT,
and authority comes from `git commit`, not from re-running the script. Review
the diff, `git add`, and open a PR.

To make re-discovery byte-stable, the script sets `LLENERGY_DISCOVERY_FROZEN_AT`
so the envelope's `discovered_at` is a fixed anchor rather than a fresh
wallclock on every run (see Troubleshooting).

---

## Regenerate the typed config after a schema change

The typed config model at `src/llenergymeasure/engines/<engine>/config.py` is
generated from the committed snapshot, not hand-written (its header says
`DO NOT EDIT`). After a schema refresh, regenerate it so the two stay in step:

```bash
uv run python scripts/engine_producers/regen_engine_configs.py \
  --engine <engine> --version <ver> --write
```

`--write` overwrites the target file; the default `--check` mode regenerates in
memory and byte-compares against the committed file, exiting non-zero with a
diff on drift. The `config-codegen` matrix job in `engine-rules-check.yml` runs
exactly `--check` per engine, resolving `<ver>` from `current.yaml`, so a
snapshot change that forgot the regen fails at PR time.

---

## What CI verifies

CI never produces these artefacts; it verifies the committed bytes on hosted
CPU runners. Three checks guard the refresh outputs:

| Check | Workflow / job | What it asserts |
|---|---|---|
| `regen_engine_configs.py --check` | `engine-rules-check.yml` / `config-codegen` (matrix over all three engines) | `config.py` is byte-identical to what the committed snapshot regenerates. |
| `check_discovered_schema_versions.py` | `ci.yml` / `schema-version-check` (matrix over all three engines) | The `schema.discovered.json` engine version equals `library.current_version` in `current.yaml`. |
| `check_pydantic_matches_discovered.py` | `ci.yml` / `docs-freshness` (second step) | The Pydantic config surface aligns with the discovered schema (no silently narrowed types or undiscovered fields, modulo an explicit whitelist). |

The two `ci.yml` checks are gated on the `docker` paths filter, which includes
`engine_versions/**`, so a pin bump under `engine_versions/` triggers them even
though nothing under `docker/` changed. See
[CI architecture](/explanation/architecture/ci-architecture) for the full
workflow topology.

---

## Prerequisites

- Docker plus the NVIDIA Container Toolkit on the machine running discovery
  (the GPU engines introspect inside a `--gpus all` container).
- `yq` on `PATH` (the refresh script reads the pin from `current.yaml` with it).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `config-codegen` fails on a snapshot PR | `config.py` was not regenerated after the schema changed | Run `regen_engine_configs.py --engine <e> --version <v> --write` and commit `config.py` |
| `schema-version-check` fails | `schema.discovered.json` version does not match the `current.yaml` pin | Re-run `./scripts/refresh_discovered_schemas.sh <engine>` after the bump and commit the refreshed schema |
| `docs-freshness` Pydantic-alignment step fails | A config field has no discovered counterpart, or a type narrowed/widened | Add the field to the whitelist in `check_pydantic_matches_discovered.py` if intentional, else fix the snapshot or the curation |
| Schema discovery fails to import engine | Container missing `--gpus all` | Verify the machine has NVIDIA drivers plus the Container Toolkit |
| Schema unchanged after discovery | Engine version did not change params | Expected: the script reports no changes and commits nothing |
| Re-discovery shows a 2-line diff on unchanged source | `LLENERGY_DISCOVERY_FROZEN_AT` not set | The introspector writes a fresh wallclock `discovered_at` on every invocation; the refresh script sets this env var to a stable anchor (typically the author date of the most recent commit touching any input path) so re-discovery is byte-stable |

---

## Adding a new engine

For the full picture of adding a new engine to llem (the plugin, the
Dockerfile-or-upstream choice, the pin, and curation), see
[engine extensibility](/explanation/architecture/engine-extensibility).

The schema-discovery surface a new engine adds is small:

1. Add a pin: `engine_versions/<engine>/current.yaml` with the
   `library.current_version` set.
2. Add a per-engine module under `scripts/engine_producers/`, mirroring an
   existing `*_introspector.py`, and register it in
   `scripts/engine_producers/__init__.py`.
3. Add a case to `scripts/refresh_discovered_schemas.sh`.
4. Run discovery once: `./scripts/refresh_discovered_schemas.sh <engine>`, then
   regenerate the config: `regen_engine_configs.py --engine <engine>
   --version <ver> --write`.
5. Add the engine to the `config-codegen` and `schema-version-check` CI
   matrices.

---

## See also

- [Architecture overview](/explanation/architecture/architecture-overview) - the two engine-knowledge products
- [Schema discovered format](/reference/schema-discovered-format) - the JSON envelope spec
- [Local knowledge production](/contributing/knowledge-production) - the rules-side operations guide
- [CI architecture](/explanation/architecture/ci-architecture) - what CI verifies and how
- [Docker setup](/how-to/docker-setup) - building engine images locally
