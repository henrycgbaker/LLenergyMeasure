---
created: 2026-02-18T17:50:07.713Z
title: Backend parameter completeness audit
area: testing
files:
  - src/llenergymeasure/config/backend_configs.py
  - src/llenergymeasure/config/models.py
  - src/llenergymeasure/core/inference_backends/pytorch.py
  - src/llenergymeasure/core/inference_backends/vllm.py
  - src/llenergymeasure/core/inference_backends/tensorrt.py
---

## Problem

The backend config models (PyTorchConfig, VLLMConfig, TensorRTConfig in backend_configs.py)
define the parameter surface, but there are two separate gaps:

1. **Coverage gaps** — some important backend API params are not yet in the config models:
   - All backends: `trust_remote_code` (required for some model architectures)
   - PyTorch: `device_map` (multi-GPU layer sharding, distinct from data-parallel `num_processes`),
     `revision` (model version pinning for reproducibility)
   - vLLM: `tokenizer_mode` (needed for Mistral-style tokenizers), explicit `dtype`
   - TensorRT: `beam_width` (beam search at Executor runtime), `remove_input_padding` (packed
     inference, significant throughput improvement)

2. **Wiring gaps** — params that ARE in the config model but may not be passed through to the
   actual backend API call in the inference backend implementations. Having the field ≠ using it.

3. **No integration tests** — no tests that verify a config field set to a non-default value
   actually changes backend behaviour. Unit tests mock the backend; we need tests that call
   the real API.

4. **No parameter reference docs** — users have no complete reference of what each backend
   param does, its valid range, and its effect on energy/throughput/latency.

## Solution

Target Phase 6 (Parameter Completeness, v2.3). Four-part work:

1. **Coverage audit** — systematically compare config model fields against official backend API
   signatures (transformers.AutoModelForCausalLM.from_pretrained + model.generate,
   vLLM LLM() constructor + SamplingParams, TRT-LLM trtllm-build + Executor).
   Add any missing fields with correct types, defaults, and validation.

2. **Wiring audit** — trace each config field from ExperimentConfig through to the backend
   API call. Flag any fields that are parsed but not forwarded. Fix wiring gaps.

3. **Integration tests** — write one integration test per backend that:
   - Sets a non-default value for a config field (e.g., batch_size=4 for PyTorch)
   - Runs a minimal experiment (1-2 prompts)
   - Asserts the backend actually used that value (e.g., via mock or log inspection)
   Priority fields: batch_size (PyTorch), tensor_parallel_size (vLLM), quantization (all backends)

4. **Parameter reference docs** — one .md doc per backend listing all config fields with:
   - Parameter name + type + default
   - What it controls (1 sentence)
   - Effect on energy/throughput/latency (measured or expected)
   - Backend API source (which function/class it maps to)
   - Valid range and constraints
   Store in docs/backends/ or .planning/designs/backends/

Note: This work is NOT needed for Phase 5 (Clean Foundation) which only requires the config
model structure to be correct. Wiring validation and integration tests can follow in Phase 6.
