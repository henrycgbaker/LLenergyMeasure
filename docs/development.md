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
path is three workflow files in `.github/workflows/`: `engine-image-build.yml`
(builds the transformers image once per SSOT version), and
`engine-schemas.yml` + `engine-invariants.yml` (one workflow per pipeline,
each covering all three engines via per-job `if:` gating on the trigger
source). See "CI pipeline ordering" below for the full sequence.

## Engine image strategy

Per-engine choices about runner type and image source are deliberately
asymmetric:

| Engine | CI runner | GPU required | Image source | Why |
|---|---|---|---|---|
| transformers | `ubuntu-latest` (GH-hosted) | No | First-party `docker/Dockerfile.transformers`, built once by `engine-image-build.yml` per (PR, SSOT version) and consumed downstream via `docker pull` | No upstream provides FA3-included transformers |
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
   bitsandbytes / calflops / sentencepiece / einops pre-installed.

3. **Build once, consume many.** Engine Image Build is the single producer
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

There is one workflow file per pipeline (one `engine-schemas.yml`, one
`engine-invariants.yml`) covering all three engines. The per-engine
asymmetry — transformers needs a pre-built first-party image, vllm and
tensorrt pull upstream images directly — is contained inside each workflow
via multi-trigger + per-job `if:` gating, not via file proliferation.

The transformers cell in `engine-invariants.yml` and `engine-schemas.yml`
is gated by `workflow_run: completed` on Engine Image Build (transformers);
the vllm and tensorrt cells in the same files are gated by
`pull_request: paths`. Per-job `if:` clauses select the correct cell for
each trigger source.

When Renovate (or a maintainer) bumps `engine_versions/transformers.yaml`
or `docker/Dockerfile.transformers`, the pipeline fires sequentially:

1. **Engine Image Build** (`engine-image-build.yml`) — builds the transformers
   image once. On PR, pushes to a per-version tag in the
   `transformers-cache` GHCR repo (`ghcr.io/<repo>/transformers-cache:transformers-<VERSION>`).
   On push to `main`, also pushes `:latest` and `:transformers-<VERSION>` to
   the canonical `ghcr.io/<repo>/transformers` repo so external `docker pull`
   users and the next PR cycle's `cache-from :latest` both stay fresh.
2. **Engine Schemas — transformers** (`schemas-transformers` job in
   `engine-schemas.yml`) — `workflow_run`-triggered on Engine Image Build's
   `success`. Pulls the just-built image, runs schema discovery, writes back
   the discovered JSON + curation digest.
3. **Engine Invariants — transformers** (`invariants-transformers` job in
   `engine-invariants.yml`) — `workflow_run`-triggered on Engine Image Build's
   `success`. Pulls the same image, runs the miner + vendor + invariants
   digest, rebases against schemas's writeback before pushing its own commit.

When Renovate (or a maintainer) bumps `engine_versions/vllm.yaml` or
`engine_versions/tensorrt.yaml`, the corresponding `invariants-vllm` /
`invariants-tensorrt` / `schemas-vllm` / `schemas-tensorrt` jobs fire on
`pull_request: paths` and pull upstream images directly (no first-party
build).

A weekly scheduled run of Engine Image Build (Monday 05:37 UTC) rebuilds
the image from scratch with `--no-cache`. If the resulting digest differs
from the cached `:latest`, that surfaces external dependency drift (apt
repo, PyPI wheel re-publish, base image silent update) that layer caching
alone wouldn't catch. Release-time `docker-publish.yml` continues to
publish `:v<pkg-version>` on each release tag.

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
