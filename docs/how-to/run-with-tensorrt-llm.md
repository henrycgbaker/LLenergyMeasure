---
title: Run an experiment with TensorRT-LLM (Docker)
description: Use the TensorRT-LLM engine for compiled-engine inference measurements.
---

# Run an experiment with TensorRT-LLM (Docker)

TensorRT-LLM compiles models into optimised TensorRT engines, then runs
inference against those engines. The first run compiles the engine
(several minutes); subsequent runs with the same config load the cached
engine and start inference immediately.

## Prerequisites

- `llenergymeasure` installed (host-side orchestrator)
- Docker + NVIDIA Container Toolkit — see [Docker setup](/how-to/docker-setup)
- TensorRT-LLM Docker image built or pullable from GHCR — see [Contributing > Development](/contributing/development)
- NVIDIA GPU with SM ≥ 7.5 (Turing or newer; e.g. RTX 2000-series, A100, H100)

## 1. Create a config file

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
  quant:
    quant_algo: W4A16_AWQ
  build_cache:
    max_cache_storage_gb: 100
```

## 2. Run the experiment

```bash
llem run experiment.yaml
```

What happens:

1. Pre-flight checks run: Docker CLI, NVIDIA Container Toolkit, GPU
   visibility, SM-version check.
2. The TensorRT-LLM Docker image is pulled on first run
   (`ghcr.io/henrycgbaker/llenergymeasure/tensorrt:v0.9.0`).
3. The container compiles the TensorRT engine from the model weights.
   **First run only — this takes several minutes.** Progress is shown in
   the terminal.
4. The compiled engine is cached on disk
   (`~/.cache/tensorrt_llm` inside the container, mounted from the host).
5. Inference runs against the compiled engine.
6. Results are printed to stdout and saved to `results/`.

:::tip Engine caching
The compiled engine is keyed to your config (model, dtype, max_batch_size,
tp_size, etc.). Running the same experiment config again skips compilation
and starts inference immediately. Changing any compile-time parameter
triggers a new build.
:::

## 3. Read the results

The output format matches other engines. The result file includes
`engine: tensorrt` and a `build_metadata` section with engine compilation
time, GPU architecture, and TRT-LLM version. See
[How to interpret results](/how-to/interpret-results) for a field-by-field
walkthrough.

## Related

- [Tutorial: Your first measurement](/tutorials/first-measurement) — start here if you've never run `llem`
- [How to: run with vLLM](/how-to/run-with-docker-vllm) — sister recipe for the vLLM engine
- [Reference: engine configuration](/reference/engines/configuration) — every TensorRT-LLM-specific config field
- [Reference: invariants (TensorRT)](/reference/engines/invariants-tensorrt) — mined parameter constraints
