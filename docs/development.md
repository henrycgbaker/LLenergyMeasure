# Development guide

This project enforces an asymmetric runtime contract: **engine code runs only
inside Docker; coordination code runs on host.**

## Layer split

| Layer | Runs on | Why |
|---|---|---|
| Engine code (miners, introspectors, vendor gates, model load) | Docker only | tensorrt-llm loads CUDA bindings on import; a unified host `uv.lock` produced incompatible cross-engine transitive constraints (#437); the multi-gigabyte `tensorrt_llm` wheel OOMed Renovate's lock-update runner. |
| Coordination (CLI, config validation, study runner, energy-measurement scaffolding without engines) | Host | Iteration speed for CLI / config / runner debugging matters; no GPU dependency. |
| Engine-touching tests | Docker only | Tests that import an engine library run inside that engine's image. Host tests gate themselves via `pytest.importorskip(...)` and skip when the engine is absent. |

## Setting up the host environment

```bash
uv sync --dev
```

Installs orchestration dependencies plus dev tools (pytest, ruff, mypy,
import-linter). **No engine libraries are installed on host** —
`import transformers`, `import vllm`, and `import tensorrt_llm` will all fail
on host. That is the contract, not a bug.

If you want host-side energy-measurement scaffolding without engines:

```bash
uv sync --dev --extra zeus --extra codecarbon
```

## Running engine code

Build the engine image once, then `docker run` against it. The image tag is
derived from the SSOT (`engine_versions/{engine}.yaml`):

```bash
VER=$(yq '.library.current_version' engine_versions/transformers.yaml)
docker build -f docker/Dockerfile.transformers \
  --build-arg TRANSFORMERS_VERSION="$VER" \
  -t llenergymeasure:transformers-${VER} .

docker run --rm \
  -v "$(pwd)":/repo -w /repo \
  --entrypoint python3 \
  llenergymeasure:transformers-${VER} \
  -m scripts.engine_miners.build_corpus --engine transformers
```

Replace `transformers` with `vllm` or `tensorrt` (and add `--gpus all` for
those two — they need a CUDA device) for the other engines. The automated
path is three workflow files in `.github/workflows/`: `build-engine-image.yml`
(builds the transformers image once per SSOT version), and
`update-engine-schemas.yml` + `update-engine-invariants.yml` (one workflow per pipeline,
each covering all three engines via per-job `if:` gating on the trigger
source). See "CI pipeline ordering" below for the full sequence.

## Engine image strategy

Per-engine choices about runner type and image source are deliberately
asymmetric:

| Engine | CI runner | GPU required | Image source | Why |
|---|---|---|---|---|
| transformers | `ubuntu-latest` (GH-hosted) | No | First-party `docker/Dockerfile.transformers`, built once by `build-engine-image.yml` per (PR, SSOT version) and consumed downstream via `docker pull` | No upstream provides FA3-included transformers |
| vllm | self-hosted GPU | Yes (CUDA) | `vllm/vllm-openai:<version>` (Docker Hub) | Canonical upstream exists; project source bind-mounted at runtime |
| tensorrt | self-hosted GPU | Yes (CUDA) | `nvcr.io/nvidia/tensorrt-llm/release:<version>` (NGC) | Canonical upstream exists; project source bind-mounted at runtime |

The principled rationale:

1. **vllm and tensorrt use upstream because canonical upstream exists.** Both
   publish per-version images at stable refs that already include the engine
   library plus its CUDA / torch substrate. Our project's value-add (the
   `llenergymeasure` package + miner / introspector scripts) is bind-mounted
   at `/app` with `PYTHONPATH=/app/src:/app -w /app` rather than baked into a
   custom overlay. No first-party Dockerfile means no version drift between
   our image and upstream's release cadence.

2. **transformers needs a first-party image because no upstream provides
   FA3-included transformers.** `pytorch/pytorch:2.5-cuda12.4-cudnn9-runtime`
   has the CUDA + torch substrate but no transformers; `huggingface/transformers-pytorch-gpu`
   has transformers but no FA3 (the hopper-extension build is niche and
   compiled from source). `docker/Dockerfile.transformers` ships transformers
   plus FA2 (PyPI wheel) plus FA3 (compiled from source) plus accelerate /
   bitsandbytes / calflops / sentencepiece / einops pre-installed, plus
   llenergymeasure's runtime non-engine deps (pydantic, typer, pyyaml,
   platformdirs, nvidia-ml-py, numpy, pyarrow, tqdm, rich, python-dotenv,
   filelock). The llenergymeasure package itself is NOT installed into the
   image — it is bind-mounted at runtime via `-v <repo>:/llem-src` +
   `PYTHONPATH=/llem-src`, identically to the vllm + tensorrt cells. This
   keeps image rebuilds dependent only on the engine substrate, not on
   project source edits, so `src/` changes never invalidate the FA3 layer.

3. **Build once, consume many.** Build engine image is the single producer
   of the transformers image; downstream workflows pull rather than rebuild.
   CI builds the same production-equivalent image users get (`INSTALL_FA3`
   defaults to `true` and is not overridden in any workflow). Cold builds
   on a brand-new SSOT version still pay the FA3 compile (~30-60 min); warm
   rebuilds reuse the GHA scope cache + the canonical `:latest` registry
   cache and finish in a few minutes. The previous shape — engine-invariants
   and engine-schemas each running their own buildx step against the same
   per-version GHA scope — was prone to cache-write contention and observed
   to deadlock at PR time on multi-GB layer writes.

## CI pipeline ordering

The transformers engine image flows through four workflows in sequence:

```
build-engine-image.yml   →   publish-engine-image.yml   →   update-engine-schemas.yml :: schemas-transformers
                                                       →   update-engine-invariants.yml :: invariants-transformers
```

Each step chains via `workflow_run: completed` with `conclusion == 'success'` gating.

There is one workflow file per pipeline concern (build, push, schemas, invariants). schemas + invariants cover all three engines; the per-engine asymmetry — transformers needs a pre-built first-party image, vllm and tensorrt pull upstream images directly — is contained inside each workflow via multi-trigger + per-job `if:` gating, not via file proliferation.

When Renovate (or a maintainer) bumps `engine_versions/transformers.yaml` or `docker/Dockerfile.transformers`, the pipeline fires sequentially:

1. **Build engine image** (`build-engine-image.yml`) — builds the transformers image. Exports the layer cache to `ghcr.io/<repo>/transformers-cache:transformers-<VERSION>-buildcache` via `cache-to: type=registry,mode=max`. **Does NOT push a runtime image.** PR builds and main builds both contribute to the cache (per-version content is deterministic given the cache-stable Dockerfile).
2. **Publish engine image** (`publish-engine-image.yml`) — `workflow_run`-triggered on Build engine image's `success`. Tag selection per parent event:
   - `push` to main / `schedule` → canonical `transformers:latest` + `transformers:transformers-<VERSION>` (vetted main code only writes these)
   - `pull_request` → `transformers-cache:transformers-<VERSION>` (PR-time runtime image — exists for the downstream chain to pull, but never claims `:latest`)
   - direct `workflow_dispatch` → canonical tags (ad-hoc recovery path)

   Pulls the layer cache from `:transformers-<VERSION>-buildcache` so the rebuild is essentially free (every layer cache-hits) — only the registry push step actually does network work. Skipped only when the parent run was a `workflow_dispatch` of the BUILD workflow (the build-only ad-hoc test path).
3. **Update engine schemas — transformers** (`schemas-transformers` job in `update-engine-schemas.yml`) — `workflow_run`-triggered on Publish engine image's `success`. Pulls the just-pushed runtime image (canonical or PR-time depending on the trigger that started the chain), runs schema discovery, writes back the discovered JSON + curation digest.
4. **Update engine invariants — transformers** (`invariants-transformers` job in `update-engine-invariants.yml`) — same trigger and ordering. Pulls the same image, runs the miner + vendor + invariants digest, rebases against schemas's writeback before pushing its own commit.

Why split build from push:
- **Push failures don't burn the FA3 compile.** A GHCR permission misconfig, transient registry outage, or rate-limit on the push step previously meant re-running ~50 min of cold compile; now the push retry runs in seconds against the durable cache.
- **PR builds never publish canonical runtime tags.** End users pulling `:latest` always get vetted main code. PR-time runtime images do exist (in the cache repo) — they're necessary so the downstream chain can validate the new Dockerfile — but they're tagged distinctly from canonical.
- **Cleaner observability.** Build duration vs push duration are separately measurable — when the pipeline slows, we see which stage to debug.
- **Ad-hoc recovery path.** If the runtime tag is corrupted or misconfigured, `gh workflow run publish-engine-image.yml --ref main` rebuilds-from-cache + pushes without redoing the FA3 compile.

When Renovate (or a maintainer) bumps `engine_versions/vllm.yaml` or `engine_versions/tensorrt.yaml`, the corresponding `invariants-vllm` / `invariants-tensorrt` / `schemas-vllm` / `schemas-tensorrt` jobs fire on `pull_request: paths` and pull upstream images directly (no first-party build).

A weekly scheduled run of Build engine image (Monday 05:37 UTC) rebuilds the image from scratch with `--no-cache`. If the resulting layer cache diverges from the prior `:transformers-<VERSION>-buildcache`, that surfaces external dependency drift (apt repo, PyPI wheel re-publish, base image silent update) that layer caching alone wouldn't catch. The corresponding push then publishes the new digest to `:latest`.

## Running tests

Host tests (the majority — orchestration, config, energy scaffolding, CLI):

```bash
uv run pytest tests/
```

Engine-touching tests gate themselves via `pytest.importorskip("transformers")`
(or `vllm`, etc.) and are skipped on host. To exercise them, run pytest inside
the matching engine image:

```bash
docker run --rm \
  -v "$(pwd)":/repo -w /repo \
  --entrypoint pytest \
  llenergymeasure:transformers-${VER} \
  tests/unit/scripts/engine_miners/test_transformers_miner.py
```

## Why this contract

The project previously offered three host extras (`[transformers]`, `[vllm]`,
`[tensorrt]`), each pulling its engine library into the host `uv.lock`. Three
problems compounded:

1. `tensorrt-llm 0.21.0` loads CUDA bindings on import, so the host couldn't
   even resolve the `[tensorrt]` extra without GPU drivers (#437).
2. The unified lock fought itself: `tensorrt-llm` transitively forced
   `transformers<4.48` even when only `[transformers]` was installed, breaking
   vLLM's torch in turn (#437, #464).
3. The `tensorrt_llm` wheel is multi-gigabyte; Renovate's lock-update runner
   OOMed every time it tried to refresh the lock.

Engines-in-Docker collapses the trichotomy (Tier 1 host-import, Tier 2 host-
incompatible-Docker, Tier 3 import-requires-GPU) into a single tier: every
engine producer runs inside its own image, period. The host lock has no
engine deps and resolves cleanly; Renovate stops OOMing; CUDA-on-import is
no longer a host problem.

The cost — slower iteration on engine code (Docker build + run vs `python -m`)
— is a non-issue because engine-touching iteration was already Docker-bound
in practice. This contract just stops pretending host imports work for those
paths.
