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

## Engine image strategy

All three engines follow the same pattern: **engine substrate goes in the
image; the project bind-mounts at runtime.** The image contains the engine
library and its CUDA/torch substrate (plus, for transformers, llenergymeasure's
runtime non-engine deps because the upstream `pytorch/pytorch` base does not
bring them transitively). The project source itself (`src/llenergymeasure/`
and `scripts/`) is bind-mounted at runtime via `-v $(pwd):/repo -e
PYTHONPATH=/repo/src:/repo`.

Why first-party for transformers and upstream for vllm/tensorrt:

- **transformers** — no upstream image ships FA3-included transformers, so we
  maintain `docker/Dockerfile.transformers` as a thin first-party image
  layered on `pytorch/pytorch` (FA2 + FA3 + accelerate + ...).
- **vllm** — `vllm/vllm-openai:<version>` is canonical and pulled directly.
- **tensorrt** — `nvcr.io/nvidia/tensorrt-llm/release:<version>` is canonical
  and pulled directly from NGC.

The asymmetry collapses to "is there an upstream image that gives us the
engine substrate we want?" — for vllm and tensorrt, yes; for transformers
with FA3 included, no.

## Running engine code

For transformers, build the image once, then `docker run` against it. The
image tag is derived from the SSOT (`engine_versions/{engine}.yaml`):

```bash
VER=$(yq '.library.current_version' engine_versions/transformers.yaml)
docker build -f docker/Dockerfile.transformers \
  --build-arg TRANSFORMERS_VERSION="$VER" \
  -t llenergymeasure:transformers-${VER} .

docker run --rm \
  -v "$(pwd)":/repo -w /repo \
  -e PYTHONPATH=/repo/src:/repo \
  --entrypoint python3 \
  llenergymeasure:transformers-${VER} \
  -m scripts.engine_miners.build_corpus --engine transformers
```

For vllm and tensorrt, pull the upstream image instead of building. Add
`--gpus all` because those two need a CUDA device:

```bash
VER=$(yq '.library.current_version' engine_versions/vllm.yaml)
docker pull "vllm/vllm-openai:v${VER}"

docker run --rm --gpus all \
  -v "$(pwd)":/repo -w /repo \
  -e PYTHONPATH=/repo/src:/repo \
  --entrypoint python3 \
  "vllm/vllm-openai:v${VER}" \
  -m scripts.engine_miners.build_corpus --engine vllm
```

The automated path is `engine-invariants.yml` and `engine-schemas.yml` in
`.github/workflows/`; both follow this same pattern across all three engines.

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
  -e PYTHONPATH=/repo/src:/repo \
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
