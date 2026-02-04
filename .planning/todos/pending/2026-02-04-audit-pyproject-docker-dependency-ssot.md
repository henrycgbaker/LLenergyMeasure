---
created: 2026-02-04T11:33
title: Audit pyproject.toml and Docker dependency architecture
area: tooling
files:
  - pyproject.toml
  - docker/Dockerfile.base
  - docker/Dockerfile.pytorch
  - docker/Dockerfile.vllm
  - docker/Dockerfile.tensorrt
---

## Problem

Current Docker images install dependencies in two ways:
1. Some from pyproject.toml extras (`pip install ".[pytorch]"`)
2. Some hardcoded in Dockerfiles (explicit pip installs)

This creates confusion about SSOT (Single Source of Truth) for dependencies:
- Should pyproject.toml define ALL dependencies, with Dockerfiles just selecting extras?
- Or do Dockerfiles need separate dependency specifications for CUDA version pinning?

Specific issues observed:
- vLLM/TensorRT Dockerfiles use `--no-deps` then manually list deps (torch version conflicts)
- PyTorch Dockerfile uses pyproject.toml extras directly
- flash-attn not included anywhere (mentioned in error messages, should be bundled)
- bitsandbytes, safetensors duplicated between pyproject.toml and Dockerfiles

Industry question: What's the standard pattern for ML projects with GPU dependencies?

## Solution

Research and implement consistent pattern:
1. Research how vLLM, HuggingFace, Lightning do Docker + pyproject.toml
2. Decide on SSOT approach (likely: pyproject.toml for deps, Dockerfiles for install order/CUDA pinning)
3. Audit each Dockerfile for missing/redundant dependencies
4. Add flash-attn to appropriate backends (pytorch, vllm images)
5. Document the dependency architecture pattern
