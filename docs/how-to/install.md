---
title: Advanced install topics
description: Offline install, locked-down environments, custom CUDA, and other advanced scenarios.
---

# Advanced install topics

For a friendly first install, see [Get Started: Install](/get-started/install).
This page covers offline install, locked-down environments, custom CUDA, and
other advanced scenarios.

---

## System Requirements

| Requirement | Minimum | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Hard requirement (TensorRT-LLM compatibility) |
| OS | Linux | Required for vLLM and TensorRT-LLM engines |
| GPU | NVIDIA with CUDA 12.x | Required for all inference engines |
| CUDA (host) | 12.x | For container image compatibility |
| Docker + NVIDIA Container Toolkit | Latest | Required for vLLM and TensorRT-LLM |
| Docker Compose | v2.32+ recommended | Required for build cache (see below). v2.11+ minimum |
| Docker Buildx | v0.17+ recommended | Required for build cache. Bundled with Docker Engine 24+ |

**macOS/Windows:** Transformers engine only. Docker-based engines (vLLM, TensorRT-LLM) require Linux.

---

## Install

The host package is the orchestrator only - it carries no engine
libraries. Install with:

```bash
pip install llenergymeasure
```

### Engine code runs in Docker

Each engine (Transformers, vLLM, TensorRT-LLM) runs inside its own image,
built from the SSOT in `engine_versions/{engine}/current.toml`. There is no host
extra for engines: `import transformers`, `import vllm`, and
`import tensorrt_llm` will fail on host by design. See
[docs/development.md](/contributing/development) for the build/run pattern.

### Available extras

The remaining extras cover host-side energy-measurement scaffolding only:

| Extra | What it installs | When to use |
|-------|-----------------|-------------|
| `zeus` | Zeus energy monitor | GPU energy via Zeus (alternative to NVML) |
| `codecarbon` | CodeCarbon | Carbon-aware energy tracking |

Install with one or both extras together:

```bash
pip install "llenergymeasure[zeus,codecarbon]"
```

---

## Install from Source (Development)

The project uses [uv](https://docs.astral.sh/uv/) as its package manager.

```bash
git clone https://github.com/henrycgbaker/llm-efficiency-measurement-tool.git
cd llm-efficiency-measurement-tool
uv sync --dev
uv run llem --version
```

Engine libraries are not installed on host. See
[docs/development.md](/contributing/development) for how to build and run engine
images locally.

Expected output:

```
llem v0.9.0
```

---

## Docker Setup

For vLLM or TensorRT-LLM engines, Docker with NVIDIA Container Toolkit is required. See the [Docker Setup Guide](/how-to/docker-setup) for a complete walkthrough covering driver installation, toolkit setup, and verification.

---

## BuildKit Builder Setup

Before building Docker images locally, set up a dedicated BuildKit builder with sufficient
cache space. Without this, the default builder may evict cached layers when building
multiple engines, causing expensive recompilation.

```bash
make docker-builder-setup
```

This creates a `llem-builder` with a 200 GiB GC limit. To use it, set
`BUILDX_BUILDER=llem-builder` in your `.env` file or export it in your shell. Run once per
machine. See [Docker Setup - BuildKit](/how-to/docker-setup#buildkit-builder-setup-recommended)
for details.

## Getting Engine Images

Only the Transformers engine is built from a project Dockerfile - vLLM and
TensorRT-LLM use canonical upstream images directly, because no upstream
ships an FA3-included Transformers image but vLLM and TensorRT-LLM both
publish ready-to-use images of their own. The project source is bind-mounted
into the upstream image at run time, so there is no per-release rebuild for
vLLM or TensorRT-LLM.

```bash
# Transformers - build from source (FA3 compile is the slow step)
make docker-build

# vLLM - pull upstream
docker pull vllm/vllm-openai:0.7.3

# TensorRT-LLM - pull upstream (NGC)
docker pull nvcr.io/nvidia/tensorrt-llm/release:0.21.0
```

The pinned versions are the SSOT in `engine_versions/{vllm,tensorrt}.yaml`
under `library.current_version`. Renovate bumps them on each upstream
release.

You can also build the Transformers image with plain `docker build` (no
Compose, no build cache):

```bash
docker build -f docker/Dockerfile.transformers -t llenergymeasure:transformers .
```

Local Transformers builds produce an image tagged `llenergymeasure:transformers`.
When present, `llem` prefers it over the registry image. See
[Image Management](/how-to/docker-setup#image-management) for the full resolution
chain.

> **When to rebuild Transformers.** The Transformers image bundles
> `llenergymeasure` source at build time. If you modify config models,
> engines, or the container entrypoint, rebuild for changes to take effect
> inside the container. The vLLM and TensorRT-LLM containers bind-mount the
> project source at run time, so source edits take effect without a rebuild.
> Local-runner experiments (Transformers without Docker) use the installed
> source directly and do not need a rebuild either.

### Other Docker Make targets

| Target | Description |
|--------|-------------|
| `make docker-pull` | Pull all registry images for your installed version |
| `make docker-images` | Show which image each engine resolves to (local vs registry) |
| `make docker-check` | Validate `docker-compose.yml` configuration |

### Fast rebuilds and first-pull cost

> **Most users never need to build.** `make docker-pull` (or letting `llem run`
> resolve the registry image automatically) gives you a working environment with
> no compilation. Building from source is for contributors and for hosts where
> you've modified `src/llenergymeasure/`.

Only Transformers has a project Dockerfile, so it is the only engine with a
GHCR cache. The image declares `cache_from` pointing at the published GHCR
tags; the `Build engine image` workflow's `transformers` job populates the cache via
`docker/build-push-action`, exporting intermediate layers to
`ghcr.io/henrycgbaker/llenergymeasure/transformers:latest` (rolling) and
`:transformers-<VERSION>` (immutable per SSOT version, written on push to
`main`). This lets fresh machines skip the ~30-min flash-attn FA3 Hopper
compile. vLLM and TensorRT-LLM are pulled from upstream images
(`vllm/vllm-openai`, `nvcr.io/nvidia/tensorrt-llm/release`) and need no
project-side cache.

Measured on `ds01` (AMD EPYC 7742, 128 cores, 504 GB RAM - Docker 27.0.3 / Buildx
v0.32.1 / llenergymeasure 0.9.0):

| Engine | Image size | Cold build | First GHCR pull | Warm local rebuild |
|--------|-----------|------------|-----------------|--------------------|
| Transformers | 7.9 GB | 33m 56s | 2m 33s (10 layers reused) | seconds |
| vLLM | 15.6 GB | 4m 12s | 4m 16s (0 layers reused) | seconds |
| TensorRT-LLM | 50.6 GB | 13m 24s | 13m 32s (0 layers reused) | seconds |

**Reading the table.** Times are measured on a 128-core/504 GB host; on smaller
machines cold builds scale roughly with `MAX_JOBS` (FA3 compile is CPU-bound).

- **Cold build** - fresh builder, `--no-cache`, no GHCR. Simulates an offline
  first-ever build.
- **First GHCR pull** - fresh builder, `cache_from` populated. What a new
  contributor gets after `make docker-builder-setup`.
- **Warm local rebuild** - second and subsequent local builds. The transformers
  image is a kernel substrate (FA3 + engine deps + runtime deps); the
  LLenergyMeasure project source is bind-mounted at runtime, never baked in.
  Source-only edits never invalidate any image layer for any engine.

**Why does the GHCR cache only help Transformers?** vLLM and TensorRT-LLM use
upstream images directly (`vllm/vllm-openai`, `nvcr.io/nvidia/tensorrt-llm/release`)
with no first-party overlay. The dominant cost on a fresh machine is pulling
the upstream base from Docker Hub / NGC, which our GHCR cache cannot accelerate
- there is no first-party Dockerfile to cache. Transformers does have a
first-party Dockerfile (`docker/Dockerfile.transformers`) because no upstream
provides an FA3-included transformers image, and the FA3 compile is the
load-bearing layer that the GHCR cache makes a single-digit-minute pull
instead of a ~30-min cold compile.

Once the upstream base is in local Docker storage (after the first build),
subsequent rebuilds for vLLM/TRT are seconds - the slow part doesn't repeat.

**Build (or pull) as normal:**

```bash
make docker-build                                    # build Transformers from source
docker pull vllm/vllm-openai:0.7.3                   # pull vLLM upstream
docker pull nvcr.io/nvidia/tensorrt-llm/release:0.21.0  # pull TensorRT-LLM upstream
```

**How the cache pipeline is wired:**

The transformers image is published under three GHCR refs, each serving a
distinct consumer. The split exists because three orthogonal questions need
separate answers:

| Ref | Kind | Written by | Consumed by |
|---|---|---|---|
| `transformers-cache:transformers-<VER>-buildcache` | BuildKit cache manifest (`mode=max`, intermediate layer metadata - **not a runnable image**) | `engine-pipeline.yml` on every successful build (PR, main, schedule, dispatch) | Future `docker build` invocations as `cache-from` |
| `transformers-cache:transformers-<VER>` | Runnable PR-time runtime image | `publish-engine-image.yml` when parent build was a `pull_request` | `engine-pipeline.yml` + `engine-pipeline.yml` for PR-time validation against the PR's Dockerfile |
| `transformers:transformers-<VER>` + `transformers:latest` | Runnable canonical runtime image | `publish-engine-image.yml` when parent build was a push to `main`, a schedule, or a `workflow_dispatch` | End users (`docker pull`), `make docker-pull`, main-branch invariants/schemas, downstream Renovate consumers |

The three axes encoded:

1. **`-buildcache` suffix** distinguishes "BuildKit cache metadata" from
   "runnable image". Cache uses `mode=max` so intermediate layers (most
   importantly the ~30-min FA3 compile) are reusable across subsequent
   builds; it cannot be `docker run`'d.
2. **`transformers-cache` repo vs `transformers` repo** distinguishes
   "built from a PR branch" from "built from `main` (vetted)". Only the
   canonical repo serves end users, so a PR build can never accidentally
   claim `:latest`.
3. **Tag (`:latest` vs `:transformers-<VER>`)** within the canonical repo
   is the standard rolling-vs-immutable convention.

Why not collapse them? Two tempting simplifications both lose value:

- *Use `type=inline` cache (cache embedded in runtime image manifest, one
  ref).* Drops `mode=max` intermediate-layer caching; second-build FA3 would
  recompile.
- *Drop the PR-time runtime image, validate against `:latest` only.* PR
  changes to `Dockerfile.transformers` itself would go unvalidated until
  after merge.

Pipeline mechanics:

- `engine-pipeline.yml` runs `build-push-action` with `cache-from` /
  `cache-to` pointing at the buildcache ref. Builds run on every PR, push to
  `main`, schedule, and dispatch. `push: false` - this workflow only exports
  cache, never publishes runnable images.
- `publish-engine-image.yml` is `workflow_run`-triggered on successful
  `engine-pipeline.yml`. It rebuilds (warming off the just-exported
  buildcache, so it's seconds), tags per parent-event (PR → cache repo;
  main / schedule / dispatch → canonical repo), and pushes. The build/push
  split exists so a registry permission failure during push doesn't burn
  the FA3 compile; the cache survives independently.
- `docker-compose.yml` declares `cache_from: [:transformers-<VERSION>,
  :latest]` for the transformers engine - version-pinned first (best layer
  match within a release), rolling-latest as fallback. vllm and tensorrt
  have no first-party `cache_from` chain (they pull upstream directly).
- `make docker-builder-setup` provisions a `docker-container` BuildKit
  driver with a 200 GiB GC limit; the default `docker` driver cannot import
  registry caches at all.
- The Transformers FA3 compile (the only layer where caching is load-bearing)
  runs on a self-hosted runner with sufficient cores + memory; CI rebuilds
  warm off the buildcache ref for every subsequent SSOT bump.
- Pulling any of the three refs is unauthenticated for public packages.

**How to tell if the cache actually warmed:** `make docker-build-{engine}` runs the build
under `BUILDKIT_PROGRESS=plain` and emits a one-line summary when it finishes:

- `✓ transformers build: 4m 18s - GHCR cache imported, 27 layers reused` - cache hit,
  FA3 layer not recompiled.
- `⚠ transformers build: 18m 03s - no GHCR cache imported (cold build)` - silent fallback.
  Cross-check [troubleshooting → Docker rebuild is slow](/how-to/troubleshoot#docker-rebuild-is-slow--recompiling-flash-attn).

The full BuildKit log for the most recent build is at `/tmp/llem-build-{engine}.log`.

**Authentication:** GHCR packages are public. No `docker login` is required to pull them.
If you hit rate limits or are behind a corporate proxy, `docker login ghcr.io` with a
[personal access token](https://github.com/settings/tokens) (scope `read:packages`) may help.

**Push access (contributors).** You do not need push access to develop on this project -
contributors only ever pull cache. Cache publication on releases is fully automated by
`docker-publish.yml` using the repo's auto-issued `GITHUB_TOKEN`, so any merged release
PR ships a fresh cache without human intervention. Manual seeding via
`make docker-seed-transformers` is restricted to the package owner (the packages live
under the `henrycgbaker` user namespace, not an org); this is the standard OSS pattern
for solo-maintained projects and reflects the supply-chain principle that manual pushes
should bypass neither code review nor CI. If you have a legitimate need to push the
cache manually (e.g. infra recovery, base-image emergency reseed), open an issue and
the maintainer can either publish on your behalf or grant per-package collaborator
access in GHCR settings.

**Offline builds:** BuildKit degrades gracefully. When the registry is unreachable the
`cache_from` entries are skipped and the build falls back to local layer cache (cold on a
fresh builder). No errors, just slower.

**First-pull cost:** the first build on any new machine downloads the full cache graph
(sizes above). Subsequent builds are incremental.

### FlashAttention-3

The Transformers Docker image ships with both FlashAttention-2 (FA2) and FlashAttention-3 (FA3)
pre-built. FA3 is compiled from source during the image build, which is the slowest build
step (~20 min). On warm rebuilds the FA3 layer is reused from the GHCR cache (see
[Fast rebuilds and first-pull cost](#fast-rebuilds-and-first-pull-cost) above) and the
build completes in minutes.

FA3 provides Hopper-optimised attention kernels. Use it via
`transformers.attn_implementation: flash_attention_3` in your experiment configs.

**To skip FA3** (e.g. for faster CI builds):

```bash
docker build -f docker/Dockerfile.transformers \
  --build-arg INSTALL_FA3=false \
  -t llenergymeasure:transformers .
```

**Why FA3 takes so long from scratch:** FA3 has no pre-built PyPI wheel. It is compiled from
the `hopper/` subdirectory of the [flash-attention](https://github.com/Dao-AILab/flash-attention)
repository using `nvcc` for CUDA architectures SM 8.0 (A100) and SM 9.0 (H100). CUDA kernel
compilation is inherently slow - each architecture target requires a separate compilation
pass. This is why the build cache is so valuable.

**FA3 hardware requirements:**

| GPU generation | SM | FA2 | FA3 |
|----------------|-----|-----|-----|
| Ampere (A100) | 8.0 | Yes | Yes |
| Hopper (H100) | 9.0 | Yes | Yes (optimised) |
| Ada Lovelace (L40S, RTX 4090) | 8.9 | Yes | Yes |
| Turing or older | < 8.0 | No | No |

**For local (non-Docker) installs**, FA3 must be built manually:

```bash
git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git
pip install flash-attention/hopper --no-build-isolation
```

This produces the `flash_attn_3` and `flash_attn_interface` packages that `transformers`
checks for at runtime.

---

## Verify Installation

Run `llem config` to check your environment:

```bash
llem config
```

Example output:

```
GPU
  NVIDIA A100-SXM4-80GB  80.0 GB
Engines
  transformers: installed
  vllm: not installed  (runs in Docker - see docs/development.md)
  tensorrt: not installed  (runs in Docker - see docs/development.md)
Energy
  Energy: nvml
Config
  Path: /home/user/.config/llenergymeasure/config.yaml
  Status: using defaults (no config file)
Python
  3.12.0
```

What each section means:

- **GPU** - NVIDIA GPU detected via pynvml. If this shows "No GPU detected", experiments will fail.
- **Engines** - Which inference engines are installed. You need at least one to run experiments.
- **Energy** - Active energy sampler. `nvml` (pynvml) is the default and ships with the base install.
- **Config** - Path to the user config file. "Using defaults" is normal for new installs.
- **Python** - Python version in use.

Run `llem config --verbose` for driver version, engine versions, and full config values.

---

## Next Steps

Follow [Getting Started](/tutorials/first-measurement) to run your first experiment.
