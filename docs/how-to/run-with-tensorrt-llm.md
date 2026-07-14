---
title: Run an experiment with TensorRT-LLM (Docker)
description: Use the TensorRT-LLM engine for compiled-engine inference measurements.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Run an experiment with TensorRT-LLM (Docker)

TensorRT-LLM compiles models into optimised TensorRT engines, then runs
inference against those engines. The first run compiles the engine
(several minutes); subsequent runs with the same config load the cached
engine and start inference immediately.

:::caution TensorRT-LLM requires Docker and an Ampere-or-newer GPU
The engine runs inside a Docker container and requires an NVIDIA GPU with
SM >= 7.5 (Turing or newer). FP8 quantisation requires SM >= 8.9 (Ada
Lovelace or newer); on A100 (SM 8.0), use INT8 or W4A16_AWQ instead.
:::

## Prerequisites

- `llenergymeasure` installed (host-side orchestrator)
- Docker + NVIDIA Container Toolkit - see [Docker setup](/how-to/docker-setup)
- TensorRT-LLM upstream image (`nvcr.io/nvidia/tensorrt-llm/release`) - pulled automatically from NVIDIA NGC on first run; pre-fetch per [Docker setup](/how-to/docker-setup)
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

With explicit quantisation and engine caching:

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
  max_batch_size: 8
  dtype: bfloat16
  quant_config:
    quant_algo: W4A16_AWQ
```

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
2. The upstream TensorRT-LLM image (`nvcr.io/nvidia/tensorrt-llm/release`) is
   pulled from NVIDIA NGC on first run, with the llenergymeasure source
   bind-mounted into it.
3. The container compiles the TensorRT engine from the model weights.
   **First run only - this takes several minutes.** Progress is shown in
   the terminal.
4. The compiled engine is cached on disk. The host directory
   `~/.cache/trt-llm` is bind-mounted into the container at
   `/root/.cache/trt-llm`, and llem defaults the build-cache location to that
   mount so compiled engines persist across ephemeral containers out of the
   box (override with `LLEM_TRT_BUILD_CACHE_PATH`).
5. Inference runs against the compiled engine.
6. Results are printed to stdout and saved to `results/`.

:::tip Engine caching
The compiled engine is keyed by TensorRT-LLM to the full build config -
model, dtype, tensor-parallel size, max-shape (max_seq_len / max_batch_size /
max_input_len), and quantisation - plus the TRT-LLM version. Running the same
config again reuses the cached engine and skips compilation; changing any of
those inputs (or bumping the engine version) triggers a fresh build. The cache
is manual and visible: llem never auto-evicts. See its location, entry count,
total size, and the manual clean command with `llem doctor`.
:::

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
  engine_path: /shared/engines/qwen2.5-7b-awq
```

With `engine_path` set, the pre-flight gate is skipped because the engine
is already in TensorRT-LLM's native format.

## 3. Read the results

The output format matches other engines. The result file includes
`engine: tensorrt` and a `build_metadata` section with engine compilation
time, GPU architecture, and TRT-LLM version. See
[How to interpret results](/how-to/interpret-results) for a field-by-field
walkthrough.

## Related

- [Tutorial: Your first measurement](/tutorials/first-measurement) - start here if you've never run `llem`
- [How to: run with vLLM](/how-to/run-with-docker-vllm) - sister recipe for the vLLM engine
- [Reference: engine configuration](/reference/engines/configuration) - every TensorRT-LLM-specific config field
- [Reference: invalid parameter combinations](/reference/engines/invalid-combos) - verified parameter constraints across engines
