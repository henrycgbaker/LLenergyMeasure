---
title: Engine extensibility
description: What a new inference engine needs to contribute, and what is generated or verified for it.
---

# Engine extensibility

The inference stack evolves quickly: vLLM, TRT-LLM, and SGLang each ship
multiple releases per quarter, and new engines appear regularly. Adding one
to LLenergyMeasure should require as little bespoke code as possible. This
page lists exactly what a contributor must produce, what is generated for
them, and what CI verifies.

For the underlying protocol contract that makes this possible, see
[Harness-plugin model](harness-plugin.md). For how the engine-knowledge
artifacts are produced and kept current, see
[Architecture overview](/explanation/architecture/architecture-overview) and
[Pipeline architecture](/explanation/architecture/pipeline-architecture).

## The contract

A new engine implements the `EnginePlugin` Protocol defined in
[`src/llenergymeasure/engines/protocol.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/engines/protocol.py).
The protocol is `@runtime_checkable`, so any class that provides the required
methods satisfies it without inheritance.

The six required methods are:

| Method | Responsibility |
|---|---|
| `load_model(config, on_substep)` | Load weights into GPU memory; return opaque model object |
| `run_warmup_prompt(config, model, prompt)` | Run one warmup inference; return latency ms (or `0.0` to use kernel-only warmup) |
| `run_inference(config, model, prompts)` | Run batch inference; return `InferenceOutput` |
| `cleanup(model)` | Release GPU memory |
| `check_hardware(config)` | Return compatibility errors (empty list when compatible); must never raise or allocate |
| `capture_observed_params(config, model, output)` | Return dict of effective engine/sampling params for observed-config tracking |

The harness calls these in a fixed order. The plugin never interacts with the
energy sampler, FLOPs estimator, or result model.

## What a contributor writes by hand

The following are engine-specific and cannot be generated:

### 1. Plugin class

Create `src/llenergymeasure/engines/<engine>/plugin.py` with a class
implementing `EnginePlugin`. The three existing engines are concrete examples:

- `TransformersEngine`
  ([`transformers/plugin.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/engines/transformers/plugin.py)) -
  CV-based warmup, batched `model.generate()`, HuggingFace weight loading.
- `VLLMEngine`
  ([`vllm/plugin.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/engines/vllm/plugin.py)) -
  single-pass kernel warmup (`return 0.0`), OpenAI-compatible server, Docker-only.
- `TensorRTEngine`
  ([`tensorrt/plugin.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/engines/tensorrt/plugin.py)) -
  engine compilation outside the NVML window, TRT-LLM executor pattern.

### 2. Dockerfile or upstream image choice

Decide the engine's image path (see [Pipeline architecture](/explanation/architecture/pipeline-architecture#asymmetric-engine-architecture-locked-design-choice)):

- If a canonical upstream image exists and carries the runtime dependencies
  llem needs, use it directly and bind-mount the project source (the vLLM and
  TensorRT-LLM pattern - no Dockerfile).
- If no suitable upstream image exists, add `docker/Dockerfile.<engine>`
  following [`docker/Dockerfile.transformers`](https://github.com/henrycgbaker/llenergymeasure/blob/main/docker/Dockerfile.transformers):
  multi-stage build with a `runtime` target, the engine version pinned via an
  `ARG` sourced from `engine_versions/<engine>/current.yaml`, and
  `llenergymeasure` installed with the relevant extras.

### 3. Engine declaration and pin

Add the new engine to the `Engine` enum in
[`src/llenergymeasure/config/ssot.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/config/ssot.py):

```python
class Engine(str, Enum):
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TENSORRT = "tensorrt"
    SGLANG = "sglang"  # new
```

The `Engine` enum is the single source of truth for engine identifiers
throughout the codebase. Add the version pin at
`engine_versions/<engine>/current.yaml`. CI derives its per-engine check
matrix from these.

### 4. Curation choices

The typed config model is generated from the discovered schema, but which
discovered parameters to expose (and how to narrow their types) is a product
decision. That decision lives in the engine's `curated.yaml` under
`engine_versions/<engine>/<version>/outputs/`. See
[Parameter curation](/explanation/architecture/parameter-curation).

## What is produced locally

Once the engine can be introspected, a maintainer produces its committed
knowledge artifacts on their own machine (these steps need the engine source,
and sometimes a GPU):

- **Discovered schema.** `make discover-schema ENGINE=<engine>` introspects
  the engine at the pinned version and writes
  `src/llenergymeasure/engines/<engine>/schema.discovered.json`.
- **Typed config model.** A code generator turns the discovered schema plus
  the curation choices into
  `src/llenergymeasure/engines/<engine>/config.py`. This is the model users
  configure against under the `engine_config:` key in their study YAML.
- **Validation rules.** `make absorb ENGINE=<engine> SRC=<engine-source>`
  reads the engine source into candidate rules, verifies them against the
  engine, and promotes the confirmed ones into
  `src/llenergymeasure/engines/<engine>/rules.yaml`. See
  [Local knowledge production](/contributing/miner-pipeline).

All three artifacts are committed and ship inside the wheel.

## What CI verifies

CI never produces these artifacts; it checks the committed bytes are
consistent, read-only on hosted CPU runners (see
[CI architecture](/explanation/architecture/ci-architecture)):

- **`config-codegen`** (gating) regenerates `config.py` from the committed
  schema snapshot and curation file and asserts it is byte-identical to the
  committed file. A drifted config fails the check.
- **`rules-coverage`** (advisory) reports validator sites in the engine source
  that no shipped rule covers, without blocking the merge.

Adding `sglang` to those matrices (with its repository slug and package
directory for the coverage checkout) is the only CI wiring a new engine needs.

## Worked forecast: SGLang as the next engine

SGLang is the planned next engine. Given the current contract, the delivery
checklist is:

**Written by hand:**

1. `src/llenergymeasure/engines/sglang/plugin.py` - `SGLangEngine` class
   implementing `EnginePlugin`. SGLang uses a server model similar to vLLM, so
   `VLLMEngine` is the closest reference; `return 0.0` from
   `run_warmup_prompt` to use kernel-only warmup.
2. Image path - SGLang ships an official release container, so use it
   upstream-direct and bind-mount the source (no first-party Dockerfile),
   unless it lacks a runtime dependency llem needs.
3. `Engine.SGLANG = "sglang"` in `ssot.py` and the pin at
   `engine_versions/sglang/current.yaml`.
4. Curation choices in `engine_versions/sglang/<version>/outputs/curated.yaml`.

**Produced locally, then committed:**

5. `schema.discovered.json` via `make discover-schema`.
6. `config.py` via the config generator.
7. `rules.yaml` via `make absorb`.

**Verified by CI:**

8. Add `sglang` to the `config-codegen` matrix (and to `rules-coverage` if its
   config validation fits the validator-site model).

## Why this matters

Keeping measurement methodology current with upstream engine APIs requires
that per-engine bespoke work stay small. The harness-plugin boundary achieves
this: engine authors write inference code, not measurement code; methodology
authors update the harness once, not three times.
