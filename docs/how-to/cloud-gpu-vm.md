---
title: Running on a cloud GPU VM
description: Quickstart for AWS, GCP, and Azure GPU instances - prerequisites, pip install, llem doctor, first measurement, and a multi-engine study.
---

# Running on a cloud GPU VM

LLenergyMeasure runs unchanged on a cloud GPU VM. There is nothing cloud-specific
in the tool: it needs an NVIDIA GPU, an NVIDIA driver, Docker, and the NVIDIA
Container Toolkit, and it reaches the same public image and model registries from
a cloud instance as it does from a bare-metal host. This page is the fast path for
getting a fresh AWS, GCP, or Azure GPU instance from zero to a multi-engine study.

If your instance already has a working NVIDIA driver and Docker with GPU
passthrough (most GPU-oriented cloud images do - see [Provider images](#provider-images)),
skip straight to [Install LLenergyMeasure](#install-llenergymeasure).

---

## Supported environments

LLenergyMeasure's
dispatch mechanics are written generically - Docker is resolved from `PATH`, GPU
selection is a pure passthrough to `docker run --gpus`, and no host-specific
tuning is baked in as a default - so the same install works across the following:

**Supported:**

- Bare-metal GPU hosts (workstation, on-prem server, datacentre node).
- Any Docker-capable GPU host (Docker Engine plus NVIDIA Container Toolkit on Linux).
- Cloud GPU VMs (AWS, GCP, Azure) with a GPU attached.

**Out of scope for now:**

- Slurm / apptainer (no HPC-scheduler or Singularity/Apptainer integration).
- Windows native (the Docker-based engines are Linux-only).
- Fractional-GPU power measurement (per-slice power on partitioned GPUs - see
  [MIG and partitioned GPUs](#mig-and-partitioned-gpus)).

"Supported" means the path is exercised and documented; it does not mean every
provider image or instance type is individually certified. The requirement is
always the same four pieces (GPU, driver, Docker, NVIDIA Container Toolkit) plus
network egress to the registries below.

---

## Prerequisites

You need an NVIDIA driver, Docker Engine, and the NVIDIA Container Toolkit on a
Linux instance. These are the same prerequisites as any other host - this page
does not duplicate the install steps, it links the canonical guides:

- **NVIDIA driver** - [NVIDIA driver installation guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/).
  Verify with `nvidia-smi`.
- **Docker Engine** - [official Docker Engine install guide](https://docs.docker.com/engine/install/).
- **NVIDIA Container Toolkit** - [NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
  This is what makes `docker run --gpus all` work.

For a from-scratch walkthrough of all three (with verification commands at each
step), see the [Docker Setup Guide](/how-to/docker-setup). Come back here for the
cloud-specific notes.

### Provider images

Most cloud providers publish GPU-oriented base images that ship the driver,
Docker, and (often) the NVIDIA Container Toolkit preinstalled. Using one of these
removes the driver/toolkit install entirely. Known-good starting points:

| Provider | Image family | Typical GPU instance types | Preinstalled |
|----------|-------------|----------------------------|--------------|
| AWS | Deep Learning AMI (DLAMI, GPU) | `g5`/`g6` (A10G, L4), `p4d`/`p4de` (A100), `p5` (H100) | Driver, Docker, NVIDIA Container Toolkit |
| GCP | Deep Learning VM / GPU-optimised image | `g2` (L4), `a2` (A100), `a3` (H100) | Driver (or install script), Docker |
| Azure | Data Science VM / GPU image + NVIDIA GPU Driver Extension | `NCasT4_v3` (T4), `NVadsA10_v5` (A10), `NC A100 v4`, `ND H100 v5` | Driver (via extension), Docker |

These are pointers, not certifications: confirm the three prerequisites on your
specific image with `nvidia-smi` and the GPU-in-Docker check below, rather than
assuming a given AMI or image is complete. On a bare Linux GPU image with nothing
preinstalled, follow the [Docker Setup Guide](/how-to/docker-setup) end to end.

**Verify GPU passthrough into Docker** (the one check that matters most on a fresh
instance):

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

If this prints the GPU table from inside the container, the instance is ready. If
it fails, see [Docker Setup - Troubleshooting](/how-to/docker-setup#troubleshooting).

### Network egress

A cloud instance behind a restrictive security group or firewall must be able to
reach the registries LLenergyMeasure pulls from. Allow outbound HTTPS to:

- `pypi.org` and `files.pythonhosted.org` - `pip install llenergymeasure`.
- `ghcr.io` - the first-party Transformers image.
- Docker Hub (`registry-1.docker.io`, `auth.docker.io`) - the upstream vLLM image
  (`vllm/vllm-openai`).
- `nvcr.io` - the upstream TensorRT-LLM image (NVIDIA NGC).
- `huggingface.co` and `cdn-lfs.huggingface.co` - model and dataset downloads.

An instance that cannot reach these reports the pull failure at run time. For a
genuinely air-gapped or egress-blocked instance, pre-fetch images and models on a
connected host and see the offline notes in
[Advanced install topics](/how-to/install#getting-engine-images).

---

## Install LLenergyMeasure

The host package is the orchestrator only - it carries no GPU or engine
libraries, so it installs on any Python 3.10+ environment:

```bash
pip install llenergymeasure
```

See [Install](/get-started/install) for the sampler extras (`zeus`, `codecarbon`)
and [Advanced install topics](/how-to/install) for locked-down or offline installs.

---

## Check the environment with `llem doctor`

Before running anything, confirm the instance is ready:

```bash
llem doctor
```

`llem doctor` reports the GPU and driver, per-engine availability (importable
locally or via Docker, and whether each image is cached), the energy samplers, the
Docker CLI/daemon and NVIDIA Container Toolkit, and `HF_TOKEN` presence. Every line
is prefixed `[ok]`/`[warn]`/`[fail]` with a `->` fix hint. On a cloud instance the
lines to check first are the `GPU / driver` section (the instance's GPU is
detected) and `Docker` (daemon reachable, toolkit detected).

For gated models (Llama, Mistral, and similar) set your Hugging Face token so
downloads succeed:

```bash
export HF_TOKEN=...   # create one at https://huggingface.co/settings/tokens
```

See [Install - Verify your install](/get-started/install#verify-your-install) for
the full `llem doctor` output format.

---

## First measurement

Run a single experiment to confirm the full path works end to end. Create a YAML
file:

```yaml
# experiment.yaml
serving_mode: offline
engine: vllm
task:
  model: gpt2
  dataset:
    source: aienergyscore
    n_prompts: 50
runners:
  vllm: container
```

Then run it:

```bash
llem run experiment.yaml
```

On first use `llem` pulls the vLLM image (allow a few minutes for the multi-GB
pull on a fresh instance), launches the container, runs the experiment inside it,
and writes the result. For an annotated walkthrough, see
[Your first measurement](/tutorials/first-measurement).

---

## Multi-engine study

Once a single experiment works, a study measures the same model across multiple
engines in one run. `llem` prepares all the required Docker images once up front
(pulling any that are missing concurrently), then runs each experiment. Follow the
[Multi-engine study tutorial](/tutorials/multi-engine-study) for a complete,
annotated study configuration and how to read the comparison.

The first study on a fresh instance pays the one-time image-pull cost for every
engine it uses (the vLLM and TensorRT-LLM upstream images are large). Subsequent
studies reuse the cached images. To pre-fetch during instance setup rather than on
the first run, see [Docker Setup - Pre-fetch images manually](/how-to/docker-setup#pre-fetch-images-manually).

---

## MIG and partitioned GPUs

A100 and H100 GPUs can be partitioned into Multi-Instance GPU (MIG) slices, and
some cloud SKUs ship pre-partitioned. LLenergyMeasure runs on a MIG slice - GPU
selection is a pure passthrough to `docker run --gpus`, which accepts a MIG UUID.

List the MIG instances on the host:

```bash
nvidia-smi -L
```

Each MIG instance appears with a `MIG-<uuid>` identifier. Pin `llem` to one slice
by setting `LLEM_DOCKER_GPUS` to the matching `--gpus` selector (quote it so the
shell keeps the value intact):

```bash
export LLEM_DOCKER_GPUS="device=MIG-<uuid>"
llem run experiment.yaml
```

`LLEM_DOCKER_GPUS` forwards verbatim to the engine container's `--gpus` flag; the
same variable pins whole devices on a shared multi-GPU host (see
[Docker Setup - GPU not visible inside container](/how-to/docker-setup#gpu-not-visible-inside-container)).

> **Power telemetry on MIG is limited.** NVML reports power for the parent
> physical GPU, not per MIG instance, so energy readings on a MIG slice include
> all instances sharing the card and cannot be attributed to a single slice.
> Per-slice (fractional-GPU) power measurement is out of scope. This is a
> hardware/NVML limitation, not a tool setting; see
> [Energy measurement - Limitations](/explanation/methodology/energy-measurement#limitations)
> for the full methodology caveat. Throughput and latency on a MIG slice are
> measured normally.

---

## See also

- [Docker Setup Guide](/how-to/docker-setup) - full driver, Docker, and toolkit walkthrough.
- [Install](/get-started/install) and [Advanced install topics](/how-to/install) - install, extras, offline.
- [Your first measurement](/tutorials/first-measurement) and [Multi-engine study](/tutorials/multi-engine-study).
- [Running on a single or consumer GPU](/how-to/single-gpu-or-limited-hardware) - low-VRAM knobs.
- [Troubleshooting](/how-to/troubleshoot) - common failure modes.
