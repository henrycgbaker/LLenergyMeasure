# Multi-GPU Support

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** [../research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)

---

## Context

Multi-GPU inference is common for any model larger than ~7B at fp16 on a 40GB GPU. A measurement
tool that ignores multi-GPU configuration cannot produce reproducible results — two runs with
different tensor parallelism (TP) or pipeline parallelism (PP) settings will have different
energy profiles and throughput figures, yet appear identical in the output if the configuration
is not recorded.

Three approaches were available:

1. **Passive** — detect multi-GPU configuration, record it, aggregate measurements. No new sweep parameters.
2. **Active** — expose TP/PP size as first-class sweep parameters in ExperimentConfig.
3. **Defer entirely** — treat multi-GPU as a later-version concern.

The backends handle TP/PP differently (`vllm.tensor_parallel_size`, `tensorrt.tp_size` + `pp_size`,
PyTorch via `device_map="auto"` or `torchrun`). Normalising these across backends into a common
sweep grammar requires the full parameter taxonomy work planned for v2.3.

TokenPowerBench (arxiv 2512.03024) validates our phasing: it benchmarks energy across TP/PP
configurations and confirms that aggregate GPU-level energy (NVML) is sufficient for efficiency
comparison without specialised power meters.

---

## Considered Options

### Primary multi-GPU stance for v2.0

| Option | Pros | Cons |
|--------|------|------|
| **Passive — detect, record, aggregate (chosen)** | Immediately useful; researchers running multi-GPU models get correct energy aggregation; reproducibility preserved via EnvironmentSnapshot; no cross-backend normalisation required yet | TP/PP size not a sweep dimension; users must use backend-specific config sections (`vllm.tensor_parallel_size`) to vary parallelism |
| Active — expose TP/PP as top-level sweep params | Unified grammar for parallelism sweep | Requires normalising 3 different backend APIs (vLLM `tensor_parallel_size`, TRT `tp_size`+`pp_size`, PyTorch `torchrun`); premature for v2.0; high risk of abstraction leaks |
| Defer entirely | Zero complexity | Incorrect energy aggregation for multi-GPU runs; non-reproducible results; unacceptable for a measurement tool |

**Rejected (2026-02-19):** Defer entirely — energy measurements on multi-GPU runs would be
incorrect without aggregation; EnvironmentSnapshot would be incomplete for reproducibility.

**Rejected (2026-02-19):** Active TP/PP sweep params — normalising across 3 backend parallelism
APIs requires the parameter taxonomy work (v2.3); premature for v2.0.

### PyTorch multi-GPU mechanism

| Option | Pros | Cons |
|--------|------|------|
| **`device_map="auto"` (Accelerate pipeline parallelism) — chosen for v2.0** | Supported by HuggingFace Transformers natively; no subprocess orchestration; works as a config field | Pipeline parallelism, not tensor parallelism; less efficient than TP for energy |
| `torchrun --nproc-per-node N` (true TP via Transformers ≥4.47) | True tensor parallelism; better efficiency | Requires subprocess orchestration (`torchrun` launcher); complex; out of scope v2.0 |

**Rejected for v2.0 (2026-02-19):** PyTorch TP via `torchrun` — requires subprocess
orchestration via `torchrun --nproc-per-node N`; complex; deferred to v2.3.

### Energy aggregation

| Option | Pros | Cons |
|--------|------|------|
| **Aggregate across all GPUs, with per-GPU breakdown (chosen)** | Total system power draw is what matters for efficiency comparison; per-GPU breakdown diagnoses PP load imbalance cheaply via Zeus `Measurement.gpu_energy` dict | Slightly more data in result schema |
| Aggregate only (no per-GPU) | Simpler schema | Loses diagnostic value for PP imbalance |
| Per-GPU only (no aggregate) | Maximum granularity | Cannot compare efficiency across different GPU counts without manual normalisation |

### Primary efficiency metric

| Option | Pros | Cons |
|--------|------|------|
| **`energy_per_output_token` (chosen)** | Accounts for GPU count, batch size, and output length implicitly; most comparable across different parallelism configs | Does not capture prefill vs decode energy split |
| `total_energy_joules` | Simple | Not comparable across different batch sizes or GPU counts |
| `energy_per_input_token` | Captures prefill cost | Less meaningful as primary metric for generation workloads |

---

## Decision

We will support multi-GPU passively in v2.0: detect, record in `EnvironmentSnapshot`, and
aggregate energy and FLOPs across all devices. No new sweep parameters for TP/PP size in v2.0.

Rationale:
- Researchers already run multi-GPU models (anything >7B at fp16 on a 40GB GPU)
- Recording the GPU configuration is essential for result reproducibility
- Adding `tp_size` sweep dimensions requires coordinating across 3 backends with different
  parallelism APIs — premature for v2.0
- TokenPowerBench (arxiv 2512.03024) confirms pure TP outperforms PP for energy efficiency;
  we should record TP configuration but not yet expose it as a sweep variable

### Consequences

Positive: Correct energy measurements for multi-GPU runs immediately; reproducibility
preserved; no cross-backend normalisation complexity.
Negative / Trade-offs: TP/PP size is not a sweep dimension until v2.3; users who want to
compare parallelism configurations must use backend-specific fields and run multiple studies.
Neutral: PyTorch `torchrun` TP deferred to v2.3; `device_map="auto"` is the supported
multi-GPU path for PyTorch in v2.0.

---

## Confirmed Sub-Decisions

| Decision | Rationale | Date |
|----------|-----------|------|
| `EnvironmentSnapshot.gpu_count: int` — required | Number of GPUs actually used by the experiment | 2026-02-19 |
| `EnvironmentSnapshot.gpu_names: list[str]` — per-device | Heterogeneous GPU clusters exist; must record all | 2026-02-19 |
| `EnvironmentSnapshot.gpu_vram_gb: list[float]` — per-device | Enables cross-hardware normalisation | 2026-02-19 |
| Energy: aggregate across all GPUs via NVML/Zeus | Total system power draw is what counts for efficiency comparison | 2026-02-19 |
| Report per-GPU energy breakdown too | Useful for diagnosing PP load imbalance; Zeus `Measurement.gpu_energy` dict provides this cheaply | 2026-02-19 |
| FLOPs: sum across all GPUs | Total compute consumed, not per-device utilisation | 2026-02-19 |
| `energy_per_output_token` as primary efficiency metric | Accounts for GPU count, batch size, and output length implicitly; most comparable across different parallelism configs | 2026-02-19 |
| v2.0 TP/PP: use backend-specific config sections (not top-level ExperimentConfig) | `vllm.tensor_parallel_size`, `tensorrt.tp_size` already exist; normalising across backends → v2.3 | 2026-02-19 |
| PyTorch TP (torchrun) out of scope for v2.0 | Requires subprocess orchestration via `torchrun --nproc-per-node N`; complex; `device_map="auto"` is the supported multi-GPU path for PyTorch | 2026-02-19 |
| Explicit TP/PP sweep grammar → v2.3 (parameter taxonomy release) | Normalising parallelism params across backends requires the full param taxonomy work | 2026-02-19 |

---

## Backend Notes

| Backend | Multi-GPU mechanism | v2.0 support |
|---------|---------------------|-------------|
| **PyTorch** | `device_map="auto"` (Accelerate pipeline) or `tp_plan="auto"` (Transformers ≥4.47 true TP via torchrun) | `device_map="auto"` supported; true TP via torchrun out of scope v2.0 |
| **vLLM** | `tensor_parallel_size=N` in engine args | Passive — `vllm.tensor_parallel_size` in VLLMConfig already |
| **TensorRT-LLM** | `tp_size` + `pp_size` in build config | Passive — `tensorrt.tp_size` in TensorRTConfig already |

"Passive" = the field exists and is recorded; it is NOT swept in v2.0.

---

## Energy Aggregation

Zeus measures multi-GPU energy via `measurement.gpu_energy` dict (per-device NVML counters).
NVML fallback: poll each device separately and sum.
Implementation: [../designs/energy-backends.md](../designs/energy-backends.md) § "Multi-GPU Energy Aggregation".

> **Research annotation (2026-02-25):** `.planning/research/PITFALLS.md` (MP-2) flags that
> NVML energy counters measure per-GPU **board power** only. They do **not** capture:
>
> - **NVLink power** (NVLink 4.0 on H100: up to 50W per link at full bandwidth)
> - **NVSwitch power** (on DGX systems: ~100W per switch)
> - **PCIe root complex power**
> - **Host CPU power** for memory copies and synchronisation
>
> For tensor parallelism, all-reduce operations consume significant NVLink bandwidth whose
> energy is not captured by per-GPU NVML counters. Summing per-GPU energy **understates**
> the true total energy by approximately 3–10% depending on NVLink traffic.
>
> **Recommendation:** Document this limitation in `MultiGPUMetrics` and in user-facing docs.
> For publishable results, researchers should note this limitation in their methodology section.
> No correction factor is feasible without specialised power meters.
>
> **Decision not overridden** — the per-GPU sum approach is correct (matches all peer tools).
> This annotation documents a known measurement limitation.
> Sources: [MLPerf Power (arXiv:2410.12032)](https://arxiv.org/html/2410.12032v2),
> [NVIDIA NVLink Blog](https://developer.nvidia.com/blog/nvidia-nvlink-and-nvidia-nvswitch-supercharge-large-language-model-inference/).

---

## GPU Detection

GPU names, VRAM, and count are detected via `nvidia-ml-py` (NVML bindings) at experiment
start and stored in `EnvironmentSnapshot`. Single-GPU case: single-item lists.
Implementation: [../designs/reproducibility.md](../designs/reproducibility.md) § `EnvironmentSnapshot`.

---

## TokenPowerBench Findings (arxiv 2512.03024)

TokenPowerBench benchmarks energy across TP/PP configurations:
- Pure tensor parallelism delivers best energy efficiency (lower J/token than pipeline parallelism)
- Gap between TP and PP widens as workload grows (40 J/token → 60+ J/token under high throughput)
- Multi-GPU measurement: aggregate GPU-level energy without specialised power meters (NVML)

This validates our approach: record TP/PP config, let users compare across configurations
manually. Automated sweep will come when we have the parameter taxonomy (v2.3).

---

## ExperimentResult Fields

`MultiGPUMetrics` model and `ExperimentResult.multi_gpu` field:
see [../designs/result-schema.md](../designs/result-schema.md) § "Multi-GPU Result Fields".

---

## Related

- [../designs/reproducibility.md](../designs/reproducibility.md): EnvironmentSnapshot schema
- [decisions/flops-estimation.md](flops-estimation.md): FLOPs aggregation across GPUs
- [decisions/parameter-taxonomy.md](parameter-taxonomy.md): v2.3 TP/PP sweep grammar
- [../research/14-flops-warmup-lora-multiGPU.md](../research/14-flops-warmup-lora-multiGPU.md)
- NEEDS_ADDRESSING.md item 23
