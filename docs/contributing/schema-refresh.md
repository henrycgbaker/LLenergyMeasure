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

The write path has a single writer per step:

```
container discovery
        |
        v
engine_versions/<engine>/v<safe>/outputs/schema.discovered.json   <- ONLY discovery write target
        |
        v  make promote-schemas (byte-copy; the ONLY writer of the src copy)
src/llenergymeasure/engines/<engine>/schema.discovered.json        <- packaged copy
        |
        v  regen_engine_configs.py (reads outputs/)
src/llenergymeasure/config/generated/<engine>.py
```

The script reads the pin from `engine_versions/<engine>/current.yaml`
(`yq '.library.current_version'`), selects the discovery image from it
(`vllm/vllm-openai:<ver>` for vllm, `nvcr.io/nvidia/tensorrt-llm/release:<ver>`
for tensorrt, and the first-party `llenergymeasure:transformers-<ver>` image
for transformers - building it from `docker/Dockerfile.transformers` if
absent), runs `python -m scripts.engine_producers._schemas_runner` inside the
container, and writes the result to the versioned snapshot
`engine_versions/<engine>/v<safe>/outputs/schema.discovered.json` (the `v<safe>`
directory is derived from the pin via `engine_versions/_outputs.py`, the one
place that name-mangling lives). It then runs `scripts/promote_schemas.py` to
byte-copy that snapshot into the packaged copy
`src/llenergymeasure/engines/<engine>/schema.discovered.json`. Promotion is the
**only** writer of the src copy and does no transformation - if one is ever
needed it belongs in discovery or codegen, not promotion.

The script prints the `git diff` of both files but does **not** commit: the
committed JSON is the canonical SSOT, and authority comes from `git commit`, not
from re-running the script. Review the diff, `git add` both files, and open a PR.
Promotion is also exposed on its own as `make promote-schemas` (all engines, or
`ENGINE=<engine>` for one) for the rare case you have refreshed a snapshot by
hand and only need to re-sync the src copy.

To make re-discovery byte-stable, the script sets `LLENERGY_DISCOVERY_FROZEN_AT`
so the envelope's `discovered_at` is a fixed anchor rather than a fresh
wallclock on every run (see Troubleshooting).

Discovery now also runs the runtime-literal probe as a second stage before the
schema is written: it recovers string literals the static type misses (e.g.
transformers `early_stopping` accepting `"never"`) and records them under a
field's `runtime_literals` key. The runner prints `runtime-literals:` report
lines per engine, including a `NARROWED` line whenever a previously recorded
literal no longer verifies at the new pin - so the diff surface makes any
auto-narrowing visible in review. See
[Local knowledge production](/contributing/knowledge-production#the-runtime-literal-stage)
for the full stage.

---

## Regenerate the typed config after a schema change

The typed config model at `src/llenergymeasure/config/generated/<engine>.py` is
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
CPU runners. The check that guards this page's own output is:

| Check | Workflow / job | What it asserts |
|---|---|---|
| `regen_engine_configs.py --check` | `engine-rules-check.yml` / `config-codegen` (matrix over all three engines) | `config.py` is byte-identical to what the committed snapshot regenerates. |
| `check_discovered_schema_versions.py` | `ci.yml` (matrix over all three engines) | The pin matches the schema version, and the promoted src copy exposes the same parameter surface as the versioned snapshot it was promoted from. This is the drift tripwire for the promotion invariant; with `promote-schemas` in place it should never fire. |

A further byte-identity check in `ci.yml` verifies Pydantic alignment - see
[CI architecture](/explanation/architecture/ci-architecture). These checks are
gated on the `docker` paths filter, which includes `engine_versions/**`, so a
pin bump under `engine_versions/` triggers them even though nothing under
`docker/` changed.

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
