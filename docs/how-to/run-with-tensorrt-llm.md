---
title: Run an experiment with TensorRT-LLM (Docker)
description: Use the TensorRT-LLM engine (pytorch and trt backends) for inference measurements.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Run an experiment with TensorRT-LLM (Docker)

TensorRT-LLM exposes two runtimes, chosen per experiment by the `backend`
field. `pytorch` (the default) runs the model through TensorRT-LLM's PyTorch
runtime with no ahead-of-time build step. `trt` compiles the model into an
optimised TensorRT engine and runs inference against that engine; the first
run of a given config pays a one-time compile cost (several minutes), and an
on-disk build cache lets later runs of the same config skip it. llem measures
both backends the same way, so `backend` is just another config axis you can
sweep.

:::caution TensorRT-LLM requires Docker and an Ampere-or-newer GPU
The engine runs inside a Docker container and requires an NVIDIA GPU with
SM >= 7.5 (Turing or newer). FP8 quantisation requires SM >= 8.9 (Ada
Lovelace or newer); on A100 (SM 8.0), use INT8 or W4A16_AWQ instead.
:::

## Prerequisites

- `llenergymeasure` installed (host-side orchestrator)
- Docker + NVIDIA Container Toolkit - see [Docker setup](/how-to/docker-setup)
- TensorRT-LLM upstream image, pulled automatically from NVIDIA NGC on first run at the pinned version (`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`, resolved from `engine_versions/tensorrt/current.yaml`) - there is no first-party tensorrt image; pre-fetch per [Docker setup](/how-to/docker-setup)
- NVIDIA GPU with SM >= 7.5 (Turing or newer; e.g. RTX 2000-series, A100, H100)

## 1. Create a config file

<Tabs groupId="surface">
<TabItem value="yaml" label="YAML">

Minimal:

```yaml
engine: tensorrt
task:
  model: meta-llama/Llama-2-7b-hf
  dataset:
    source: aienergyscore
    n_prompts: 50
runners:
  tensorrt: docker
```

With the compiled `trt` backend, explicit quantisation, and a larger batch:

```yaml
engine: tensorrt
task:
  model: meta-llama/Llama-2-7b-hf
  dataset:
    source: aienergyscore
    n_prompts: 50
runners:
  tensorrt: docker
tensorrt:
  engine_params:
    backend: trt
    max_batch_size: 8
    dtype: bfloat16
    quant_config:
      quant_algo: W4A16_AWQ
```

Engine fields nest under `engine_params:` (and sampling under
`sampling_params:`); a field placed directly on the `tensorrt:` block is
rejected. `quant_config` and `fast_build` exist only on the compiled `trt`
backend, so they require `backend: trt` - see
[Choosing a backend](#choosing-a-backend).

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment(
    model="meta-llama/Llama-2-7b-hf",
    engine="tensorrt",
    n_prompts=50,
)
print(result)
```

</TabItem>
</Tabs>

## Choosing a backend

TensorRT-LLM ships two runtimes behind one engine. Select with
`tensorrt.engine_params.backend`:

| `backend` | Runtime | Build step | Best for |
|-----------|---------|-----------|----------|
| `pytorch` (default) | TensorRT-LLM PyTorch runtime | none | quick runs and sweeps, no compile wait |
| `trt` | compiled TensorRT engine | ahead-of-time compile (cached) | repeated runs of a fixed config, peak throughput |

llem selects the backend by constructor class, not a runtime kwarg: `pytorch`
resolves `tensorrt_llm.LLM` and `trt` resolves
`tensorrt_llm._tensorrt_engine.LLM`. Any other value is rejected at config load
with a `{pytorch, trt}` error.

Some engine parameters exist only on the compiled `trt` backend and are
rejected under `pytorch`:

- `quant_config` - build-time quantisation config
- `fast_build` - reduced-optimisation build for faster compiles

Declaring either without `backend: trt` fails at config load with a clear error
rather than silently measuring a different configuration - the pytorch
runtime's argument class has no such field. The on-disk build cache likewise
applies only to the `trt` backend. The full rule set is in
[invalid parameter combinations](/reference/engines/invalid-combos).

## 2. Run the experiment

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
llem run experiment.yaml
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment("experiment.yaml")
```

</TabItem>
</Tabs>

What happens:

1. Pre-flight checks run: Docker CLI, NVIDIA Container Toolkit, GPU
   visibility, SM-version check.
2. The pinned upstream TensorRT-LLM image
   (`nvcr.io/nvidia/tensorrt-llm/release:1.2.1`) is pulled from NVIDIA NGC on
   first run, with the llenergymeasure source bind-mounted into it.
3. The engine loads the model. On the default `pytorch` backend this is a
   normal model load. On the `trt` backend the container compiles a TensorRT
   engine from the model weights - **first run of a given config only, several
   minutes** - and caches it on disk (see below).
4. Inference runs against the loaded (or compiled) model.
5. Results are printed to stdout and saved to `results/`.

:::tip Engine build cache (`trt` backend)
The compiled `trt` engine is keyed by TensorRT-LLM to the full build config -
model, dtype, tensor-parallel size, max-shape (`max_seq_len` / `max_batch_size`
/ `max_input_len`), and quantisation - plus the TRT-LLM version. Running the
same config again reuses the cached engine and skips compilation; changing any
of those inputs (or bumping the engine version) keys to a distinct engine hash
and triggers a fresh build. Because the host directory `~/.cache/trt-llm` is
bind-mounted into the container at `/root/.cache/trt-llm`, an identical config
also hits across containers.

The cache is on out of the box: `.env.example` ships
`LLEM_TRT_BUILD_CACHE_ENABLED=1` and llem defaults the build-cache location to
the mount (override with `LLEM_TRT_BUILD_CACHE_PATH`; delete the enable line to
fall back to TensorRT-LLM's disabled default). Each result records
`engine_build_cache_hit` (`true` on a hit, `false` on a fresh compile, `null`
when the cache is not in play - the pytorch backend, an `engine_path` override,
or the cache disabled) and `model_load_time_sec`, which covers the build and
load and sits outside the energy window. The cache lifecycle is manual and
visible: llem never auto-evicts. See its location, entry count, total size, and
the manual clean command with `llem doctor`.
:::

## Multi-GPU (tensor parallelism)

Set `tensorrt.engine_params.tensor_parallel_size` above `1` to shard the model
across GPUs:

```yaml
engine: tensorrt
tensorrt:
  engine_params:
    tensor_parallel_size: 2
```

TensorRT-LLM's `LLM` API self-manages tensor parallelism: setting
`tensor_parallel_size` makes it spawn and coordinate its own worker processes
inside a single container process. llem launches one `python3` process and does
not wrap the run in `mpirun`.

Pin which physical GPUs llem uses with `LLEM_DOCKER_GPUS` (the `docker run
--gpus` value; empty means every visible GPU). Quote multi-device values so the
comma is not split by the shell:

```bash
LLEM_DOCKER_GPUS="device=2,3" llem run experiment.yaml
```

Restricting at the docker level keeps CUDA and NVML device indices consistent
inside the container (both enumerate from 0).

**PCIe hosts without functional peer-to-peer (P2P).** On boxes whose GPU
topology reports `SYS` links in `nvidia-smi topo -m` (often because ACS is
enabled in the BIOS), tensor-parallel runs can hang at the first NCCL
collective. llem forwards every `NCCL_*` host environment variable into the
experiment and baseline containers, so the standard workaround applies straight
from the host:

```bash
NCCL_P2P_DISABLE=1 LLEM_DOCKER_GPUS="device=0,1" llem run experiment.yaml
```

## Quantisation is engine-owned

llem never quantises or converts checkpoints itself. Quantisation is owned end
to end by TensorRT-LLM: either declare a build-time `quant_config` on the `trt`
backend and let the engine quantise during compilation, or point the experiment
at a checkpoint already quantised in a format TensorRT-LLM supports. The next
section covers which pre-quantised formats load and which do not.

## HF pre-quantised checkpoints

TensorRT-LLM's `LLM` API cannot load AutoAWQ / AutoGPTQ community-format
Hugging Face checkpoints directly on **either** backend (live-verified at
1.2.1 against `Qwen/Qwen2.5-0.5B-Instruct-AWQ`):

- the **trt** (compiled) backend raises `NotImplementedError: Unsupported
  quantization_config` - it only recognises `fp8`/`mxfp4` `quant_method` values
  from `config.json`;
- the **pytorch** backend raises an `AssertionError` in the weight loader - it
  expects the NVIDIA ModelOpt weight layout, not AutoAWQ's `qweight` packing.

TensorRT-LLM's supported pre-quantised path is its own ModelOpt export (a
checkpoint carrying `hf_quant_config.json`), not the AutoAWQ/AutoGPTQ
`config.json` `quantization_config` format that Qwen's official `*-AWQ` repos
ship.

Pre-flight catches AutoAWQ/AutoGPTQ checkpoints and refuses the run with an
actionable error. To benchmark such a checkpoint, either use a ModelOpt-
quantised equivalent, or convert it once with `trtllm-build` and point the
experiment at the build output:

```bash
trtllm-build \
  --checkpoint_dir <path-to-converted-checkpoint> \
  --output_dir /shared/engines/qwen2.5-7b-awq
```

```yaml
task:
  model: Qwen/Qwen2.5-7B-Instruct-AWQ   # original HF id, for tokenizer + metadata
tensorrt:
  engine_params:
    backend: trt
    engine_path: /shared/engines/qwen2.5-7b-awq
```

With `engine_path` set, the pre-flight gate is skipped because the engine
is already in TensorRT-LLM's native format. `engine_path` requires
`backend: trt` (the compiled-engine directory is only readable by the trt
constructor); the config is rejected otherwise.

## 3. Read the results

The output format matches other engines. The result file includes
`engine: tensorrt`, the resolved `engine_version`, and the load/cache
annotations described above (`model_load_time_sec`, and
`engine_build_cache_hit` on the `trt` backend). See
[How to interpret results](/how-to/interpret-results) for a field-by-field
walkthrough.

## Related

- [Tutorial: Your first measurement](/tutorials/first-measurement) - start here if you've never run `llem`
- [How to: run with vLLM](/how-to/run-with-docker-vllm) - sister recipe for the vLLM engine
- [Reference: engine configuration](/reference/engines/configuration) - every TensorRT-LLM-specific config field
- [Reference: invalid parameter combinations](/reference/engines/invalid-combos) - verified parameter constraints across engines
