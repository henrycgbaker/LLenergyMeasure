---
title: Install
description: Install LLenergyMeasure in two minutes.
---

# Install

```bash
pip install llenergymeasure
```

This installs the host orchestrator: the part that drives experiments,
collects results, and writes output files. It is small and has no GPU or
engine dependencies.

Inference engines (transformers, vLLM, TensorRT-LLM) run inside per-engine
Docker containers. The host package coordinates them but does not import
them. The base install therefore works on any Python 3.10+ environment;
GPU and Docker are only needed when you actually run an experiment.

---

## Sampler extras

Energy measurement uses GPU telemetry. The default sampler (`nvml`) ships
with the base install via `nvidia-ml-py`. Two optional samplers provide
alternatives:

### `[zeus]`

```bash
pip install "llenergymeasure[zeus]"
```

Installs the [Zeus](https://ml.energy/zeus/) energy monitor as an
alternative to NVML. Zeus provides higher-resolution GPU power
measurements on supported hardware. Use it when you want to cross-check
NVML readings or when your workflow already uses Zeus.

When to install: if you need Zeus-based energy accounting or want to
compare NVML vs Zeus readings for the same experiment.

### `[codecarbon]`

```bash
pip install "llenergymeasure[codecarbon]"
```

Installs [CodeCarbon](https://codecarbon.io/) for carbon-aware energy
tracking. CodeCarbon reports energy in CO2-equivalent terms using local
grid intensity data, useful when you want to report emissions rather than
raw joules.

When to install: if you are reporting carbon footprint alongside energy,
or if your institution requires CO2-equivalent reporting.

---

## Combined install

Install multiple extras together:

```bash
pip install "llenergymeasure[zeus,codecarbon]"
```

---

## Engine dependencies

Engine libraries (PyTorch and transformers, vLLM, TensorRT-LLM) are
intentionally absent from the host package. They run inside Docker
images that the orchestrator launches. There is no `[transformers]`,
`[vllm]`, or `[tensorrt]` extra to install.

To get engine images:

```bash
# Transformers - build from source (one-time, ~30 min cold; seconds from cache)
make docker-build-transformers

# vLLM - pull upstream image
docker pull vllm/vllm-openai:0.7.3

# TensorRT-LLM - pull upstream NGC image
docker pull nvcr.io/nvidia/tensorrt-llm/release:0.21.0
```

Docker and NVIDIA Container Toolkit must be installed first. See
[Docker setup](/how-to/docker-setup) for a full walkthrough.

---

## Verify your install

```bash
llem config
```

Expected output (GPU and sampler sections confirm hardware is visible):

```
GPU
  NVIDIA A100-SXM4-80GB  80.0 GB
Engines
  transformers: not installed  (runs in Docker)
  vllm: not installed          (runs in Docker)
  tensorrt: not installed      (runs in Docker)
Energy
  Energy: nvml
Python
  3.12.0
```

"Not installed" against engine names is expected - engines run in Docker,
not on the host. The important lines are `GPU` (GPU detected) and
`Energy` (sampler resolved).

---

## What's next

Run your first measurement: [Quick start](/get-started/quick-start).

---

## Advanced install topics

For offline install, locked-down environments, custom CUDA versions,
development installs from source, and BuildKit cache setup, see
[Advanced install topics](/how-to/install).
