# Troubleshooting

Common issues and solutions for LLenergyMeasure.

---

## Common Issues

### No GPU detected

**Symptom:** `llem doctor` shows no GPU, or measurement fails with a CUDA error.

**Cause:** NVIDIA drivers are not installed, the device is not visible in the current
environment, or the system is CPU-only.

**Fix:**
1. Run `nvidia-smi` to verify the GPU is visible on the host.
2. Run `llem doctor` to see what the tool detects.
3. If `nvidia-smi` works but the engine does not, you may be running outside a container
   that has CUDA - for vLLM and TensorRT-LLM, use `llem run study.yaml` with Docker runners
   (see [docker-setup.md](/how-to/docker-setup)).
4. If `nvidia-smi` fails, install or reinstall the NVIDIA drivers for your OS.

---

### Engine not available on host

**Symptom:** `llem run study.yaml` (with `engine: vllm`) fails immediately with
an import error mentioning the engine package.

**Cause:** Engines have no host install path - they run inside per-engine
Docker images. A host import of `transformers`, `vllm`, or `tensorrt_llm`
will always fail by design.

**Fix:** Build the engine image and dispatch via the Docker runner. The
canonical pattern is in [development.md](/contributing/development); the short form
is:

```bash
VER=$(yq '.library.current_version' engine_versions/transformers/current.yaml)
docker build -f docker/Dockerfile.transformers \
  --build-arg TRANSFORMERS_VERSION="$VER" \
  -t llenergymeasure:transformers-${VER} .
```

Replace `transformers` with `vllm` or `tensorrt` (and add `--gpus all` for
those two) for the other engines. Then run `llem run` with a Docker runner
configured for the engine - see [docker-setup.md](/how-to/docker-setup).

Run `llem doctor` to see the current status of each engine.

---

### Docker pre-flight failed

**Symptom:** `llem run study.yaml` exits early with a pre-flight error about Docker.

**Cause:** One of the Docker pre-flight checks failed. Pre-flight checks verify:
1. Docker CLI is on PATH.
2. NVIDIA Container Toolkit binary is on PATH (`nvidia-container-runtime`, `nvidia-ctk`,
   or `nvidia-container-cli`).
3. Host `nvidia-smi` present (warn only - remote Docker daemon is supported).
4. GPU is visible inside a container (`docker run --gpus all nvidia-smi`).
5. CUDA/driver compatibility (checked from container probe output).

**Fix:** Read the error message - it identifies which check failed.

- `Docker not found`: install Docker Engine ([docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)).
- `NVIDIA Container Toolkit not found`: follow [docker-setup.md](/how-to/docker-setup).
- `GPU not visible inside container`: check that `--gpus all` works with `docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi`.
- `CUDA/driver mismatch`: update the host NVIDIA driver to be compatible with the container's CUDA version.

Skip pre-flight checks for testing or remote daemon setups:

```bash
llem run study.yaml --skip-preflight
```

---

### Out of memory (OOM)

**Symptom:** The run crashes with an OOM error (CUDA out of memory).

**Cause:** The model is too large for the available GPU VRAM at the current configuration.

**Fix (try in order):**
1. Use a smaller model.
2. Reduce `transformers.llem_execution.batch_size` (default is 1 - already minimal for Transformers).
3. Switch to lower dtype: `dtype: float16` or `dtype: bfloat16`.
4. Enable BitsAndBytes quantization: `transformers: { engine_params: { load_in_4bit: true } }`.
5. For vLLM: reduce `vllm.engine_params.gpu_memory_utilization` (e.g. 0.7 instead of 0.9).
6. For vLLM: reduce `vllm.engine_params.max_model_len` to cap KV cache allocation.

---

### Permission denied (Docker)

**Symptom:** `docker run` fails with permission denied.

**Cause:** The current user is not in the `docker` group.

**Fix:**

```bash
sudo usermod -aG docker $USER
newgrp docker          # activate without logout
```

Or prefix Docker commands with `sudo`. Permanent fix requires re-login.

---

### Study failed partially

**Symptom:** A study run produces some results but not all. Some experiments are missing
from the output directory.

**Cause:** Individual experiments may fail while the study continues. LLenergyMeasure
uses skip-and-continue: a failed experiment is recorded as an error in the study manifest,
and execution continues with the remaining experiments.

**Fix:**
1. Check the study manifest in `results/` for per-experiment status and error messages.
2. The manifest records the failure reason for each skipped config.
3. Fix the failing config (see the error message) and re-run the specific experiment
   separately before merging results.

---

### Container crashes with "context canceled" or ValidationError

**Symptom:** Experiments using Docker runners fail immediately. The container log shows
`context canceled` or a Pydantic `ValidationError` mentioning unknown fields (e.g.
`dtype: Extra inputs are not permitted` or `dataset: Input should be a valid string`).

**Cause:** The Transformers Docker image was built from an older version of
the source code. The host sends config JSON using the current schema, but
the container rejects it because its bundled code expects the old schema.
This only applies to Transformers; the vLLM and TensorRT-LLM containers
bind-mount host source at run time, so they always see the current schema.

**Fix:** Rebuild the Transformers image from the current source:

```bash
docker build -f docker/Dockerfile.transformers -t ghcr.io/henrycgbaker/llenergymeasure/transformers:v0.6.0 .
```

Replace `v0.6.0` with your installed version (`llem --version`). See
[Installation - Getting Engine Images](/how-to/install#getting-engine-images)
for full instructions.

---

### Results look wrong / energy is 0

**Symptom:** `total_energy_j` is 0.0, or `energy_adjusted_j` is null, or energy values seem too low.

**Cause:** Energy measurement requires NVML (pynvml) and a supported NVIDIA GPU. If NVML
is unavailable, energy measurement falls back gracefully to zero rather than crashing.

**Fix:**
1. Check `llem doctor` - it shows the active energy sampler under `Energy measurement`.
2. Verify pynvml can access the GPU: run `python -c "import pynvml; pynvml.nvmlInit(); print('OK')"`.
3. Check your config. Setting `energy_sampler: null` explicitly disables
   energy measurement (throughput-only mode).
4. If `baseline.enabled: true` (default), ensure the baseline measurement is completing.
   A failed baseline causes `adjusted_j` to be null.
5. For very short inference runs (< 200ms), NVML polling at 100ms intervals may not collect
   enough samples for accurate integration. Use larger `n` values.

Zeus and CodeCarbon are optional extras. If they are not installed, the tool falls back
to NVML. See [energy-measurement.md](/explanation/methodology/energy-measurement) for sampler details.

---

### Warmup takes too long

**Symptom:** Experiments take much longer than expected. Progress stalls for 1-2 minutes
before measurement begins.

**Cause:** Warmup is enabled by default. It runs `n_prompts=5` warmup inferences, then
waits `thermal_floor_seconds=60.0` seconds for GPU temperature to stabilise before
measuring.

**Fix for quick testing:**

```yaml
offline:
  warmup:
    enabled: false
```

Or on the CLI:

```bash
llem run experiment.yaml  # add to YAML for testing
```

For publication-quality measurements, leave warmup enabled. See
[methodology.md](/explanation/methodology/methodology) for why warmup matters.

---

### Stalls or hangs {#stalls-or-hangs}

**Symptom:** `llem run` appears to hang indefinitely - no progress output,
no error, process does not return.

**Cause:** Most hangs fall into five categories:

1. **Docker image pull in progress** - the first run of an engine pulls a
   multi-GB image. Check `docker pull` progress in a separate terminal:
   `docker ps` and `docker images`.

2. **TensorRT-LLM engine compilation (`trt` backend)** - on the compiled `trt`
   backend, TRT-LLM compiles a CUDA engine from model weights on the first run
   of a config. For a 7B model this takes 5-15 minutes with no visible
   progress. Re-run with `LLEM_LOG_LEVEL=DEBUG` to see compilation logs. The
   default `pytorch` backend has no such compile step.

3. **Pre-flight check stalled** - if the Docker daemon is unreachable or the GPU
   is being held by another process, the pre-flight probe hangs. Run
   `llem doctor` in a second terminal to check daemon and GPU state.

4. **Inference timeout not triggered** - `study_execution.experiment_timeout_seconds`
   defaults to 600 seconds. A model with a very long generation (e.g. high
   `max_new_tokens` with a slow sampler) may legitimately exceed this. Increase
   the timeout in the YAML if needed.

5. **Multi-GPU NCCL collective stuck** - on multi-GPU (tensor-parallel) runs the
   process can hang at the very first NCCL collective with no output. This is
   common on PCIe hosts whose GPU topology lacks functional peer-to-peer (P2P)
   - e.g. boxes where the inter-GPU link is `SYS` in `nvidia-smi topo -m`, often
   because ACS is enabled in the BIOS. The standard workaround is
   `NCCL_P2P_DISABLE=1`.

**Fix:**

- For Docker/TRT-LLM hangs: wait, then check `docker logs <container-id>` for progress.
- For pre-flight hangs: run `docker run --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi`
  to verify GPU access from Docker, then re-run `llem run`.
- For genuine inference hangs: raise `study_execution.experiment_timeout_seconds` or
  reduce `max_new_tokens`.
- For a multi-GPU NCCL hang: set the NCCL tuning/workaround var on the host and
  re-run - llem forwards every non-empty `NCCL_*` host variable into each
  workload container it launches (experiment, baseline, and the engine-server
  container in server mode), so `NCCL_P2P_DISABLE=1 llem run ...` (or exporting
  it first) applies inside the container. On a PCIe box without working P2P this
  is the standard fix. In server mode the same hang shows up as a server that
  never becomes ready: `docker logs <container>` stops at the NCCL init line.

---

## Invalid Parameter Combinations

Rejected config combinations and the engine capability matrix are catalogued in
the generated
[Invalid Parameter Combinations](/reference/engines/invalid-combos) reference,
which is regenerated from the config schema on every engine update.

---

## Docker rebuild is slow / recompiling flash-attn

**Symptom:** `make docker-build` takes 15-20 minutes and the post-build
summary line reports `⚠ no GHCR cache imported (cold build)` (or BuildKit
output shows `flash-attn` source downloads and nvcc compilation for every
build).

**Cause:** BuildKit's `cache_from` registry pull was skipped. In rough order
of likelihood:

(a) **`BUILDX_BUILDER` is unset or pointing at the default `docker` driver.**
    The default driver cannot import registry caches at all - `cache_from`
    entries are silently ignored. Confirm with `docker buildx ls`: the row
    marked with `*` (current builder) must show driver `docker-container`,
    not `docker`. Fix by adding `BUILDX_BUILDER=llem-builder` to your `.env`
    (it ships in `.env.example`) and re-running `make docker-builder-setup`
    if the builder doesn't exist yet.
(b) You are on a fresh buildx builder with no local cache (this is normal
    on the very first build - first-pull cost is paid once).
(c) You are offline or GHCR is unreachable.
(d) Your `TRANSFORMERS_VERSION` (from `engine_versions/transformers/current.yaml`)
    does not match any seeded cache ref (`cache_from` resolves to
    `transformers-cache:transformers-<VERSION>-buildcache` and falls through
    to `:latest` - if neither has usable layers, BuildKit silently
    cold-builds). The buildcache ref only exists after a local
    `make docker-seed-transformers` at that pin.

The full BuildKit log for the most recent attempt is at
`/tmp/llem-build-{engine}.log` - grep it for `importing cache manifest` to
see whether the registry was even reached.

**Fix:**

0. Confirm the builder driver: `docker buildx ls`. The active builder
   (marked `*`) must be `docker-container`. If it's `docker`, run
   `make docker-builder-setup` and ensure `BUILDX_BUILDER=llem-builder` is
   in your `.env` (or exported in the shell).
1. Inspect the builder cache: `docker buildx du --builder llem-builder`. If
   it's near-empty, BuildKit has nothing to reuse locally and will pull from
   the registry.
2. Verify network: `curl -I https://ghcr.io/v2/henrycgbaker/llenergymeasure/transformers/manifests/latest`
   should return 200 or 401 (both fine; 000/timeout means no connectivity).
3. If you recently bumped the single source of truth (SSOT) version but haven't run
   `make docker-seed-transformers` at the new pin yet, the per-version
   buildcache ref does not exist and the build falls back to `:latest`
   (already last in the cache_from chain). Seed locally to create it; no
   env-var override needed.
4. If the cache is corrupt, recreate the builder:
   `make docker-builder-rm && make docker-builder-setup`. Note this discards
   all local layer cache; the first subsequent build will repopulate from
   GHCR.
5. Offline is expected-slow. BuildKit degrades gracefully to a cold build -
   no errors, just minutes.

**Release publish fails: promotion source missing (`docker-publish.yml`):**
CI never compiles flash-attention - the FA3 Hopper build needs far more memory
than hosted runners have. The release publish is a registry-side tag-copy of
the image the maintainer seeds locally and the promotion path publishes, so its
only failure mode is a missing source: `docker-publish.yml` aborts loudly when
`transformers:transformers-<VER>` does not exist in the registry.

Fix - seed locally, then let the promotion tag-copy run:

```bash
docker login ghcr.io           # needs write:packages scope
make docker-seed-transformers  # local build + push of the promotion source (~minutes if locally cached)
```

On the next push to main touching `engine_versions/transformers/current.yaml`
or the Dockerfile, `publish-engine-image.yml` tag-copies the seed to
`transformers:transformers-<VER>` (or trigger it manually via
`workflow_dispatch`). Once that source exists, re-run `docker-publish.yml`; the
tag-copy completes in seconds.

---

## Schema skew between host and Docker image

**Symptom:** `llem run study.yaml` aborts before any experiment with a message
like:

```
Docker image 'llenergymeasure:transformers' was built from llenergymeasure 0.6.0
(schema 9988776655ff) but the host is running 0.6.0 (schema a1b2c3d4e5f6).
The container will reject ExperimentConfig fields added on the host after
the image was built.
```

A container stack trace full of `extra_forbidden` Pydantic errors (often with
URLs mixing `errors.pydantic.dev/2.10/…` and `errors.pydantic.dev/2.12/…`, a
tell for version skew).

> **What the runtime gate now catches.** The original
> `llem.expconf.schema.fingerprint` label is gone (bind-mounted source made
> the schema-fingerprint check structurally redundant: in-container source
> always equals the host source). What `StudyRunner._prepare_images` does
> today is a different check: it probes the in-container engine library
> version and compares it against the engine_version envelope on the
> wheel-bundled rules + schema artefacts. The Pydantic-shape errors
> above are exactly the symptom that probe is meant to catch ahead of
> dispatch, so on current builds the gate should hard-error before the
> container ever runs. If you reached this section anyway, either the image
> shipped without engine-version visibility or `LLEM_SKIP_IMAGE_CHECK=1`
> was set.

**Cause:** all three engines now bind-mount the host project source at
runtime, so a Pydantic-shape error here means the engine library inside the
image is at a version that no longer matches what the host code expects (e.g.
the SSOT bumped transformers but the local image is still on an older tag).

**Fix:** rebuild or repull the affected engine image. One of:

```bash
make docker-build                                       # local build, Transformers
docker pull vllm/vllm-openai:v0.19.1                    # repull vLLM upstream
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.1   # repull TensorRT-LLM upstream
make docker-pull                                        # pull the newest published Transformers tag
```

Verify the image actually changed by inspecting the digest:

```bash
docker image inspect llenergymeasure:transformers --format '{{.Id}}'
```

---

## Getting Help

Run `llem doctor --json` to capture full environment details (Python version, installed
engines, GPU info, energy sampler status, config file path). Include this output when
filing a bug report.

File issues at: [github.com/henrycgbaker/llenergymeasure/issues](https://github.com/henrycgbaker/llenergymeasure/issues)
