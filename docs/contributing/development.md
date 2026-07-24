# Development guide

This project enforces an asymmetric runtime contract: **engine code runs only
inside Docker; coordination code runs on host.**

## Layer split

| Layer | Runs on | Why |
|---|---|---|
| Engine code (schema introspection, rule probes, model load) | Docker only | tensorrt-llm loads CUDA bindings on import; a unified host `uv.lock` produced incompatible cross-engine transitive constraints (#437); the multi-gigabyte `tensorrt_llm` wheel OOMed Renovate's lock-update runner. |
| Coordination (CLI, config validation, study runner, energy-measurement scaffolding without engines) | Host | Iteration speed for CLI / config / runner debugging matters; no GPU dependency. |
| Engine-touching tests | Docker only | Tests that import an engine library run inside that engine's image. Host tests gate themselves via `pytest.importorskip(...)` and skip when the engine is absent. |

## Setting up the host environment

```bash
uv sync --dev
```

Installs orchestration dependencies plus dev tools (pytest, ruff, mypy,
import-linter). **No engine libraries are installed on host** -
`import transformers`, `import vllm`, and `import tensorrt_llm` will all fail
on host. That is the contract, not a bug.

If you want host-side energy-measurement scaffolding without engines:

```bash
uv sync --dev --extra zeus --extra codecarbon
```

## Running engine code

The dispatch path for experiments goes through `docker_runner.py`, which
bind-mounts the project source + a tiny entrypoint script + the host's
runtime-deps cache into the container. The image tag is derived from the
SSOT (`engine_versions/{engine}/current.yaml`); the framework code is bind-mounted
rather than baked.

```bash
VER=$(yq '.library.current_version' engine_versions/transformers/current.yaml)
docker build -f docker/Dockerfile.transformers \
  --build-arg TRANSFORMERS_VERSION="$VER" \
  -t llenergymeasure:transformers-${VER} .

# Direct invocation for ad-hoc schema introspection runs:
docker run --rm \
  -v "$(pwd)":/repo -w /repo \
  --entrypoint python3 \
  llenergymeasure:transformers-${VER} \
  -m scripts.engine_producers._schemas_runner --engine transformers \
  --output /repo/src/llenergymeasure/engines/transformers/schema.discovered.json
```

Most maintainers use the `./scripts/refresh_discovered_schemas.sh <engine>`
wrapper (equivalently `make discover-schema ENGINE=<engine>`) rather than
invoking the runner directly - it selects the right image and prints the diff.

For experiment dispatch (the `llem run` path) docker_runner.py emits a
different shape: the entrypoint script (shipped as package data at
`src/llenergymeasure/infra/_container/container_entrypoint.sh`) is
materialised to a host tempdir, bind-mounted at `/llem-entry.sh`, and set as
`--entrypoint`. The script diffs a runtime requirements list (bind-mounted at
`/llem-requirements.txt`, materialised from the installed distribution
metadata) against the in-container installed dists, pip-installs any missing
ones to a host-mounted cache (`~/.cache/llem/deps/py{N.M}/`, keyed by
container Python minor), sets `PYTHONPATH` to include the cache + `/llem-src`,
then exec's the framework entrypoint module. Shipping the script + deps list
as package data (rather than resolving a repo-root `scripts/` + `pyproject.toml`)
is what makes dispatch work from a plain `pip install`, not only a source
checkout. TRT-LLM dispatches route through `/opt/nvidia/nvidia_entrypoint.sh`
first so `LD_LIBRARY_PATH` is set up for libnvinfer. See "Runtime-deps priming"
below for the full mechanism.

Replace `transformers` with `vllm` or `tensorrt` (and add `--gpus all` for
those two - they need a CUDA device) for the other engines.

Engine knowledge (schemas and rules) is produced locally with these
containers and then committed; CI never runs the engines. It verifies the
committed bytes on hosted CPU runners via `engine-rules-check.yml`. See
[CI architecture](/explanation/architecture/ci-architecture) for the workflow
topology and [Pipeline architecture](/explanation/architecture/pipeline-architecture)
for the transformers image lifecycle.

## Runtime-deps priming

vLLM and TensorRT-LLM use upstream-direct images as the engine substrate,
and those images don't ship every runtime dep `llenergymeasure` needs
(empirical spike 2026-05-12 found `vllm/vllm-openai:v0.7.3` lacks
`platformdirs`, `nvidia-ml-py`, `pyarrow`; the NGC TRT-LLM image lacks
`python-dotenv`). Rather than bake a thin wrapper image per engine, the
in-container entrypoint script primes the missing deps lazily on first
dispatch into a host-mounted persistent cache.

### Mechanism

The entrypoint script (shipped at
`src/llenergymeasure/infra/_container/container_entrypoint.sh`) runs once per
dispatch and:

1. Computes `PY_MINOR` from the container's Python (`sys.version_info`).
2. Sets `PYTHONPATH=/llem-src:/llem-runtime-deps/py{N.M}:...` so the
   probe and subsequent imports see the cache.
3. Fast-paths via a stamp file: `sha256sum` the bind-mounted
   `/llem-requirements.txt`, compare to
   `/llem-runtime-deps/py{N.M}/.llem_requirements_hash_{engine}`.
   Match means "deps probe already done against this requirements list,
   nothing changed, skip the probe." Saves ~200ms per dispatch on warm cache.
   The list is materialised on the host from the installed distribution
   metadata, so a re-install (or edited-then-reinstalled checkout) changes its
   hash and re-triggers the probe.
4. If stamp missing or mismatched: a small Python helper reads the requirement
   specs from `/llem-requirements.txt`, calls
   `importlib.metadata.distribution(name)` per dep, and accumulates the
   missing ones.
5. Pip-installs missing deps via
   `pip install --no-deps --no-cache-dir --only-binary=:all: --target $DEPS_TARGET`.
6. Chowns the cache directory to `LLEM_HOST_UID:LLEM_HOST_GID` (passed
   by docker_runner) so the host can clean it without sudo despite the
   container running as root.
7. Writes the requirements hash to the stamp file.
8. Exec's the framework entrypoint as a single `python3` process, routing
   through `nvidia_entrypoint.sh` when `LLEM_ENGINE=tensorrt` (sets up
   `LD_LIBRARY_PATH` for libnvinfer). Multi-GPU tensorrt is NOT wrapped in
   `mpirun`: TensorRT-LLM's `LLM` API self-manages tensor parallelism by
   spawning its own workers, so one process per container is correct.

### Cache location

The host-side cache lives at `~/.cache/llem/deps/` by default (resolved
via `platformdirs`). Set `LLEM_DEPS_CACHE_DIR` to override - useful when
sharing across machines on cluster storage.

### What this is NOT

- **Not a wrapper image**. The upstream engine image stays untouched.
- **Not an installation step**. There's no `llem doctor` or pre-flight
  ritual; first dispatch primes automatically.
- **Not a permanent host pollution**. The cache is a single bind-mounted
  directory; `rm -rf ~/.cache/llem/deps/` cleans it.
- **Not an alternative to the engine-version gate**. The probed engine
  library version (`vllm.__version__`, `tensorrt_llm.__version__`,
  `transformers.__version__`) is compared at study setup against the SSOT
  pin (`engine_versions/<engine>/current.yaml`) that the wheel-bundled rules
  and discovered schema were generated against, and a mismatch is a hard
  error (see `infra/version_handshake.py`).

## Engine image strategy

Per-engine choices about image source are deliberately asymmetric. For the
full rationale (and the `#518` design record), see
[Pipeline architecture: asymmetric engine architecture](/explanation/architecture/pipeline-architecture#asymmetric-engine-architecture-locked-design-choice).
The developer-relevant summary:

| Engine | Image source | Framework code |
|---|---|---|
| transformers | First-party `docker/Dockerfile.transformers` (flash-attention 3 included; no upstream provides it) | Bind-mounted at runtime |
| vllm | Upstream `vllm/vllm-openai:<version>` (Docker Hub) | Bind-mounted at runtime |
| tensorrt | Upstream `nvcr.io/nvidia/tensorrt-llm/release:<version>` (NGC) | Bind-mounted at runtime |

For all three engines the `llenergymeasure` package is bind-mounted (via
`-v <package-dir>:/llem-src/llenergymeasure` + `PYTHONPATH=/llem-src`), never
baked into the image, so `src/` edits never invalidate an image layer. Only the
package directory is mounted - never its parent - so `/llem-src` exposes just
`llenergymeasure` and never a host site-packages sibling that could shadow a
container-native dependency (`PYTHONPATH` precedes the image's own
site-packages on `sys.path`). The transformers Dockerfile
ships transformers plus FA2/FA3 plus the accelerate / bitsandbytes /
sentencepiece toolchain and llem's non-engine runtime deps; vllm and tensorrt
inherit everything they need from their upstream images.

## Building and publishing the transformers image

Because the flash-attention compile needs more memory than hosted CI runners
have, CI never builds the transformers image. It is produced locally and
promoted by a tag-copy, the same "produce locally, verify in CI" split the
schema and rules follow:

1. **Local build for development.** `make docker-build` builds the image on
   your machine (warm rebuilds pull cache layers from GHCR and finish in a
   few minutes).
2. **Local seed for a bump.** `make docker-seed-transformers` builds the
   runtime image and pushes it to the promotion-source ref
   `transformers-cache:transformers-<VER>` (plus the `mode=max` BuildKit cache
   to `transformers-cache:transformers-<VER>-buildcache`). Run it during a
   transformers bump session.
3. **Merge-time promotion.** When the bump lands on main,
   `publish-engine-image.yml` tag-copies the seeded image to the canonical
   `transformers:transformers-<VER>` and `transformers:latest` tags via
   `docker buildx imagetools create` (no rebuild). A missing seed fails the
   promotion loudly.
4. **Release tag-copy.** `docker-publish.yml` (called by `release.yml`)
   tag-copies the promoted `transformers:transformers-<VER>` image to the
   package-versioned `transformers:<VERSION>` release tag via
   `docker buildx imagetools create` (no rebuild).

See [Pipeline architecture: transformers image lifecycle](/explanation/architecture/pipeline-architecture#transformers-image-lifecycle)
for the full diagram and [CI architecture](/explanation/architecture/ci-architecture)
for the workflow topology.

## Running tests

Host tests (the majority - orchestration, config, energy scaffolding, CLI):

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
  tests/unit/engines/test_engine_protocol.py
```

## Changelog fragments

User-facing changes are recorded as changelog *fragments* rather than by editing
`CHANGELOG.md` directly, so parallel PRs never conflict on the `[Unreleased]`
section. Each PR that changes shipped code (`src/llenergymeasure/`) drops one
small Markdown file per section under `changelog.d/`, named
`<PR-number>.<type>.md`, where `<type>` is one of `added`, `changed`,
`deprecated`, `removed`, `fixed`, or `security` (the Keep a Changelog sections).

The file body is 1-4 sentences of user impact: what changed, what breaks, and
what to do. Keep breaking-change bullets prefixed with the bold
`**Breaking (<area>):**` convention. The PR reference is automatic - the number
in the filename becomes the `([#N])` link when the changelog is assembled. For a
change with genuinely no PR number, prefix the filename with `+` (e.g.
`+my-change.fixed.md`) and it renders without a link.

```bash
# Example: PR #900 adds a config field.
printf 'The study config gains a `foo` field controlling ...\n' > changelog.d/900.added.md
```

Deep format-contract detail (exact field shapes, read-tolerance rules) belongs
in the reference docs (e.g. `docs/reference/results-schema.md`), not the
changelog; the changelog links out to it.

Preview how the fragments will read once assembled (a draft only - it never
consumes the fragments):

```bash
make changelog-draft            # or: uv run towncrier build --draft --version NEXT
```

The real assembly runs once at release time
(`uv run towncrier build --version vX.Y.Z`), which folds every fragment into a
new `CHANGELOG.md` version section and deletes them.

CI enforces this: a PR that touches `src/llenergymeasure/` but adds no fragment
fails the `changelog-check` job. For a change with no user impact (a pure
internal refactor, test-only, or tooling change), apply the `no-changelog` label
to the PR to skip the check.

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

The cost - slower iteration on engine code (Docker build + run vs `python -m`)
- is a non-issue because engine-touching iteration was already Docker-bound
in practice. This contract just stops pretending host imports work for those
paths.
