# Reproducibility

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Context

LLenergyMeasure produces energy and performance measurements that researchers intend to cite
in papers and compare across runs. Measurement reproducibility is therefore a scientific
requirement, not just an engineering convenience. However, energy measurements in particular
are subject to physical non-determinism (thermal state, OS scheduler, GPU boost clocks) that
cannot be fully eliminated by software controls.

The key tension: we must be honest about what we can and cannot guarantee, without either
over-promising (claiming full reproducibility that doesn't exist) or under-providing (omitting
controls we could reasonably implement).

## Considered Options

### Environment Capture

| Option | Pros | Cons |
|--------|------|------|
| **Capture `environment_snapshot` in every result — bold** | Full audit trail of what was installed. Essential for reproducing results cited in papers. Matches MLflow, lm-eval, Optimum-Benchmark patterns. | Adds a `pip freeze` subprocess call at startup. Moderate I/O cost. |
| Capture only llenergymeasure version | Minimal overhead | Insufficient — backend library versions (torch, vllm, etc.) directly affect measurements |
| No environment capture | Zero overhead | No basis for reproduction; fundamentally insufficient for research use |

### Random Seed

| Option | Pros | Cons |
|--------|------|------|
| **`random_seed: int = 42` in `ExperimentConfig`, explicit default — bold** | Controls model sampling (temperature > 0 runs). Default is explicit, not hidden. Reproducible across runs by default. | Does not control GPU non-determinism (boost clocks, CUDA op ordering). |
| No random seed control | Simpler config | Sampling varies between runs; comparisons invalid for temperature > 0 configs. |
| Hidden/hardcoded seed | Reproducible | Conceals an implicit assumption; confusing when users set temperature > 0. |

### Determinism Mode

| Option | Pros | Cons |
|--------|------|------|
| Formal determinism mode (lock boost clocks, isolate CPUs, pin NUMA) | Maximum reproducibility for HPC environments | HPC-specific; requires elevated privileges; out of scope for v2.0 target (local GPU). |
| **Document uncontrolled sources explicitly — bold** | Scientific honesty. Sets correct expectations. Matches what peers (lm-eval, Optimum-Benchmark) actually do. | Does not eliminate variance; users must run multiple cycles themselves. |

### Dataset Pinning

| Option | Pros | Cons |
|--------|------|------|
| **Built-in datasets shipped with the package (pinned) — bold** | Same prompts, same order, every run. Eliminates one major source of variance across machines. | Increases package size. Remote datasets not pinned (out of scope). |
| Download datasets at runtime | Smaller package | Dataset versions can change; prompts can differ; not reproducible without version pinning |

## Decision

We will capture `environment_snapshot` in every `ExperimentResult`, add `random_seed: int = 42`
to `ExperimentConfig`, ship built-in datasets pinned with the package, and document explicitly
what is and is not controlled. Docker image digest will be added to results at v2.2.

Rationale: These controls match the practices of lm-eval, Optimum-Benchmark, and MLflow.
They provide a meaningful audit trail without over-promising. Formal determinism mode is
deferred because it requires HPC-specific privileges not available in the v2.0 local target
environment.

## Consequences

Positive:
- Every result carries a full software audit trail usable in paper citations
- Controlled sources of variance are documented and traceable
- `config_hash` + `environment_snapshot` together uniquely identify the measurement conditions
- Honest `reproducibility_notes` field prevents misuse in claims

Negative / Trade-offs:
- `pip freeze` subprocess at experiment start adds minor I/O overhead
- `random_seed` does not control all sources of non-determinism (thermal state, OS scheduler,
  GPU boost clocks, NUMA allocation, network jitter, GPU firmware microcode)
- Energy measurements retain variance from uncontrolled physical sources (NVML accuracy is
  ±5W absolute, not ±5% — see `.planning/research/PITFALLS.md` CP-1 for details)

Neutral / Follow-up decisions triggered:
- Formal determinism mode (lock boost clocks, isolate CPUs) deferred to v3.x (HPC-specific)
- Provenance graph (full DAG of inputs → results) deferred as out of scope
- Docker image digest pinning deferred to v2.2

## `environment_snapshot` Specification

Captured at the start of every experiment via `pip list --format=freeze` and
`subprocess.run(["nvcc", "--version"])`.

See [`../designs/result-schema.md`](../designs/result-schema.md) for the Pydantic model
definition. Fields: `python_version`, `cuda_version`, `driver_version`,
`llenergymeasure_version`, `installed_packages`, `timestamp_utc`.

## What IS Controlled

- Experiment configuration (`config_hash` of resolved `ExperimentConfig`)
- Prompt set and order (built-in datasets are pinned)
- Random seed (model sampling, warmup order)
- Software environment (captured in `environment_snapshot`)
- Hardware (auto-detected GPU model, VRAM captured in result)

## What IS NOT Controlled

- GPU boost clock speed (thermal state at measurement time)
- OS scheduler behaviour
- Other processes on the same machine
- NUMA memory allocation (non-deterministic under load)
- Network jitter (if model is loaded from remote filesystem)
- GPU firmware microcode (changes with driver updates)

Documented in results with a `reproducibility_notes` field (static string):

```json
{
  "reproducibility_notes": "Energy measurements have variance from thermal and scheduler effects (NVML accuracy is ±5W; percentage depends on power draw). Same config_hash guarantees identical workload; it does not guarantee identical energy readings."
}
```

> **Research annotation (2026-02-25):** `.planning/research/PITFALLS.md` (CP-1) clarifies
> that NVML accuracy is ±5 **watts** (not ±5%). At 300W (A100 under load) this is ~1.7%;
> at 40W (idle) it is ~12.5%. The previous "±5–15%" phrasing was misleading. The
> `reproducibility_notes` field above has been updated accordingly. Additionally, research
> identifies several unaddressed pre-flight checks that would improve reproducibility:
>
> - GPU persistence mode (`nvidia-smi -pm 1`) — UA-1 in PITFALLS.md
> - ECC memory status — UA-2
> - GPU clock frequency at measurement time — UA-3
> - GPU power limit vs default — mP-3
>
> These are candidates for the `EnvironmentSnapshot` model in `designs/reproducibility.md`.
> Source: [NVML API Reference](https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html),
> [Part-time Power Measurements (arXiv:2312.02741)](https://arxiv.org/html/2312.02741v2).

## Related

- [`cli-ux.md`](cli-ux.md) — maximally-explicit UX principle
- [`../designs/result-schema.md`](../designs/result-schema.md) — `ExperimentResult` and `EnvironmentSnapshot` field definitions
- [`../designs/experiment-config.md`](../designs/experiment-config.md) — `random_seed` field definition
- [`warmup-strategy.md`](warmup-strategy.md) — warmup controls that reduce (but do not eliminate) thermal variance
