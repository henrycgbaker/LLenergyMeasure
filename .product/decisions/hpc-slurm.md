# HPC / SLURM Support

**Status:** Proposed
**Date decided:** 2026-02-19
**Last updated:** 2026-02-25
**Research:** N/A

## Context

A significant share of LLM inference research happens on HPC clusters (university and
national compute facilities) running SLURM as the job scheduler. These environments impose
hard constraints that differ from local and Docker execution: no Docker daemon (institutional
policy prohibits it), no persistent processes (jobs are submitted to the scheduler), shared
filesystems, scheduler-controlled GPU allocation, and often no outbound internet access.

Supporting HPC properly is a non-trivial scope addition. The question is whether to attempt
partial support at v2.0/v2.2 or defer until the core is stable and a proper design can be
done.

The v2.0 Layer 1 runner system is already structured to accommodate HPC without breaking
changes — the runner key (e.g. `singularity:`) is a placeholder that can be implemented later
without altering experiment.yaml or study.yaml.

## Considered Options

### When to support HPC

| Option | Pros | Cons |
|--------|------|------|
| Partial HPC support at v2.0 or v2.2 | Reaches HPC researchers sooner | SLURM, Singularity/Apptainer, and shared filesystem complexity not warranted until core is stable. Risk of premature design lock-in. |
| **Defer to v3.x — bold** | Core stable before adding HPC complexity. Design can be informed by actual user feedback from v2.x. Runner abstraction already accommodates it without breaking changes. | HPC researchers cannot use the tool until v3.x. |
| Never support HPC | Zero complexity | Excludes a major research audience. |

### SLURM Job Submission Surface

| Option | Pros | Cons |
|--------|------|------|
| `llem run --submit-slurm` flag | Integrated; no separate tool | Couples SLURM logic to the main CLI; complex to test; unclear UX for multi-backend studies. |
| Separate `llem-slurm` plugin | Clean separation; optional dependency | More packaging surface; plugin discovery complexity. |
| **No design committed yet — bold** | Avoids premature decisions. Design at v3.x when real HPC user feedback is available. | No implementation path defined. |

### Container Runtime

| Option | Pros | Cons |
|--------|------|------|
| **Singularity/Apptainer — bold** | Standard HPC container runtime. Institutional policy compliant. No root required. | Different image format from Docker; separate build pipeline needed. |
| Docker in rootless mode | Familiar tooling | Institutional policy often prohibits Docker entirely on HPC. Not reliable. |

## Decision

We will defer HPC/SLURM support to v3.x. The Layer 1 runner system is already designed to
accommodate HPC without breaking changes to experiment.yaml or study.yaml. No design is
committed at this stage — "planned but not designed" is the correct state.

Rationale: SLURM, Singularity/Apptainer, and shared filesystem complexity are not warranted
until the core local and Docker execution paths are stable and have received user feedback.
The runner abstraction in architecture.md already reserves the `singularity:` runner key,
avoiding any breaking change when HPC is eventually implemented.

## Consequences

Positive:
- No premature design lock-in before HPC-specific user needs are understood
- Runner abstraction already accommodates HPC; deferral is zero-cost structurally
- v3.x planning can be informed by real v2.x user feedback on pain points

Negative / Trade-offs:
- HPC researchers cannot use the tool until v3.x at the earliest
- Intended pattern (below) is a placeholder only — it may change significantly at design time

Neutral / Follow-up decisions triggered:
- v3.x scope must include: SLURM job submission, Singularity/Apptainer image build pipeline,
  shared filesystem result storage, offline model cache (pre-download), array job support
- HPC design must resolve: job submission surface (flag vs plugin), image distribution
  strategy, result aggregation across SLURM array jobs

## Why HPC is Different

These constraints distinguish HPC from local and Docker execution:

- **No Docker daemon**: Institutional policy typically prohibits Docker. Singularity/Apptainer
  is the standard container runtime.
- **No persistent processes**: Jobs are submitted to SLURM scheduler; a long-running process
  cannot be held between experiments.
- **Shared filesystems**: Results must be written to shared storage (Lustre, GPFS), not local
  disk.
- **GPU allocation**: GPUs are allocated per-job, not persistent. `CUDA_VISIBLE_DEVICES` is
  set by the scheduler.
- **Network restrictions**: Outbound internet access often blocked. Model downloads must
  happen before job submission (offline mode required).

## Intended Pattern (Placeholder — Not Designed)

This is a sketch for planning purposes only. The actual design will be done at v3.x.

```yaml
# ~/.llenergymeasure/config.yaml — HPC profile (placeholder)
runners:
  pytorch:   singularity:/shared/images/llenergymeasure-pytorch.sif
  vllm:      singularity:/shared/images/llenergymeasure-vllm.sif
  tensorrt:  singularity:/shared/images/llenergymeasure-tensorrt.sif
```

SLURM job submission would be handled by a future `llem run --submit-slurm` flag or a
separate `llem-slurm` plugin. Design TBD at v3.x.

## Deferred Work (v3.x)

- SLURM job submission (`sbatch` integration)
- Singularity/Apptainer image build pipeline
- Shared filesystem result storage
- Pre-download model cache population (offline mode)
- Array job support for parallel experiment execution

## Related

- [`architecture.md`](architecture.md) — Layer 1 runner system; `singularity:` runner key placeholder
- [`versioning-roadmap.md`](versioning-roadmap.md) — v3.x scope (HPC + lm-eval integration)
- [`docker-execution.md`](docker-execution.md) — Docker isolation strategy (v2.2); prerequisite understanding for HPC container design
