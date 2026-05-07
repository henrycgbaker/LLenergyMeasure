---
title: Run an experiment with vLLM (Docker)
description: Use the vLLM backend for high-throughput inference measurements.
---

# Run an experiment with vLLM (Docker)

This recipe runs a single measurement against the vLLM backend. Use it
when you want to measure inference under vLLM's continuous-batching
runtime rather than HuggingFace `transformers`.

## Prerequisites

- `llenergymeasure` installed (host-side orchestrator)
- Docker + NVIDIA Container Toolkit — see [Docker setup](/how-to/docker-setup)
- vLLM Docker image built or pullable from GHCR — see [Contributing > Development](/contributing/development)

## 1. Create a config file

Create `experiment.yaml`:

```yaml
engine: vllm
task:
  model: gpt2
  dataset:
    source: aienergyscore
    n_prompts: 50
runners:
  vllm: docker
```

## 2. Run the experiment

```bash
llem run experiment.yaml
```

What happens:

1. Pre-flight checks run: Docker CLI, NVIDIA Container Toolkit, GPU
   visibility inside container, CUDA/driver compatibility.
2. The vLLM Docker image is pulled on first run
   (`ghcr.io/henrycgbaker/llenergymeasure/vllm:v0.9.0`).
3. The container launches, runs the experiment, and streams results back.
4. Results are printed to stdout and saved to `results/`.

## 3. Read the results

The output format matches the Transformers track. The key difference is
`engine: vllm` in the experiment ID and result file. See
[How to interpret results](/how-to/interpret-results) for the field-by-field
walkthrough.

## Related

- [Tutorial: Your first measurement](/tutorials/first-measurement) — start here if you've never run `llem`
- [How to: run with TensorRT-LLM](/how-to/run-with-tensorrt-llm) — sister recipe for the TRT-LLM backend
- [Reference: engine configuration](/reference/engines/configuration) — every vLLM-specific config field
