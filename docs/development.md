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
path is `engine-invariants.yml` and `engine-schemas.yml` in `.github/workflows/`;
both follow this same pattern across all three engines.

## Engine image strategy

Per-engine choices about runner type and image source are deliberately
asymmetric:

| Engine | CI runner | GPU required | Image source | Why |
|---|---|---|---|---|
| transformers | `ubuntu-latest` (GH-hosted) | No | First-party `docker/Dockerfile.transformers` | No upstream provides FA3-included transformers |
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

3. **Image freshness vs main.** PR-time CI builds the transformers image
   inline using `docker/setup-buildx-action@v4` + `docker/build-push-action@v7`
   with two cache layers: `cache-from` opportunistically warm-hits
   `ghcr.io/<repo>/transformers:latest` (kept fresh by release-time
   `docker-publish.yml`) and `cache-to` writes a per-version GHA scope so
   re-runs of the same SSOT version reuse FA3 + apt + uv layers. Cold builds
   on a brand-new SSOT version still pay the FA3 compile (~30 min with
   `INSTALL_FA3=true`; CI uses `INSTALL_FA3=false` for miner + introspector
   passes since they don't need the kernel). Warm rebuilds are a few minutes.

4. **PR-time inline build, merge-time publish.** When Renovate bumps a
   transformers version, we do NOT publish a per-version image at PR time —
   the PR-time validation contract is "the inline build must succeed."
   Publish-on-merge (path-filtered on `docker/Dockerfile.transformers` and
   `engine_versions/transformers.yaml`) keeps GHCR's `:latest` and
   `:transformers-<library-version>` tags fresh after each merge that
   changes the image inputs. This closes the docs-vs-runtime drift gap for
   external `docker pull` users (the mined corpus + generated docs at HEAD
   describe the same library version that `:latest` ships) and keeps the
   `cache-from :latest` warm-hit rate high for subsequent Renovate cycles.
   Release-time `docker-publish.yml` continues to publish `:v<pkg-version>`
   on each release tag.

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
