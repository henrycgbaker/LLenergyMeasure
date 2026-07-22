---
title: Multi-engine implementation-parameter study
description: Measure how four implementation choices change energy and efficiency for the same model across three inference engines.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Multi-engine implementation-parameter study

This is the flagship tutorial. We'll measure how four representative
implementation choices - numerical precision, batching, attention
backend, KV-cache reuse - affect the energy and efficiency profile of
the same model across Transformers, vLLM, and TensorRT-LLM. By the end
you'll have a structured result you can chart, and you'll know how to
design your own implementation-parameter sweeps.

These four are illustrative, not exhaustive. Every parameter declared
by an engine is exposed programmatically and can be swept; we picked
four that exercise different layers of the stack. The point of the
exercise is the workflow, not the parameter list.

The framing matters. This is the question `llem` is built to answer:
*given a fixed open-source model, how do downstream implementation
choices shape its inference cost?* Most adjacent tools optimise the
other axis - "given a fixed implementation, how do models compare?"
That's a useful question too, but it's not this one.

> **Compute time:** ~30 minutes on a single A100-class GPU once Docker
> images are pulled. TensorRT-LLM runs on its default `pytorch` backend
> here, so there is no engine-compilation wait; switching a cell to the
> compiled `trt` backend would add a one-time (cached) engine build.

## Prerequisites

- A working install of `llenergymeasure` - see
  [How to install](/how-to/install).
- Docker + NVIDIA Container Toolkit operational and `llem doctor`
  passing - see [Docker setup](/how-to/docker-setup).
- All three engine images either built locally or pullable from GHCR -
  see [Contributing > Development](/contributing/development) for the
  build pattern.
- ~30 GB of free disk for caches (model weights + TRT-LLM compiled
  engines).
- You've completed the
  [first-measurement tutorial](/tutorials/first-measurement) so the
  shape of `llem run` and `result.json` is familiar.

## Step 1 - Sketch the question

Before writing config, sketch what you expect. This is a research
discipline, not a measurement step.

The four parameters in this tutorial are chosen because they exercise
*different layers of the inference stack* and have plausible - but
not pre-assumed - energy effects:

| Parameter | What it changes | Expected effect on energy |
|-----------|-----------------|---------------------------|
| **dtype** (`float16` vs `bfloat16`) | Per-element precision of weights/activations | Probably small for a 0.5B model; bf16 may shift error patterns without changing FLOPs |
| **Batch size / max_num_seqs / max_batch_size** | How many prompts share a kernel call | Larger batches → higher GPU utilisation → lower J/token (until VRAM saturates) |
| **Attention backend** (`sdpa` vs `flash_attention_2`; `flash_attn` vs `flashinfer`) | Kernel implementation of attention | Flash-attn-style backends typically reduce HBM traffic → lower energy |
| **KV-cache reuse** (vLLM prefix caching; TRT-LLM block reuse) | Whether prefix tokens are recomputed across overlapping prompts | Workload-dependent - large effect if prompts share prefix, near-zero otherwise |

We don't pre-assume the answers. The point of measurement is to find
them. But **writing them down before running** is what separates
"I ran a benchmark" from "I tested a hypothesis."

## Step 2 - Read the study config

The shipped config lives at
[`configs/tutorials/tutorial-multi-engine.yaml`](https://github.com/henrycgbaker/llenergymeasure/blob/main/configs/tutorials/tutorial-multi-engine.yaml).
Walk through it section by section.

```yaml
study_name: tutorial-multi-engine

runners:
  transformers: docker
  vllm: docker
  tensorrt: docker
```

All three engines in Docker. `runners` is what pins each engine to its
isolated image, so the host doesn't need to import any engine. Docker is
the recommended way to compare engines side-by-side: engine dependency
closures (e.g. transformers vs vllm pinned versions) are too divergent to
coexist on one host, so pinning each engine to its own image is what lets
you run all three from a single machine.

:::caution Docker is recommended, and elevation is precedence-based
Docker elevation is precedence-based: an engine you pin explicitly (`runners:`,
env var, or user config) keeps that pin; only engines left on auto-detection are
elevated to Docker. An engine pinned to `local` is checked for host
importability at preflight, so `runners:` pinning every engine to `local` runs
without Docker - **but only do this if the host genuinely provides every engine;
Docker per engine remains the recommended isolation.** `llem` raises a
`PreFlightError` before any inference starts if a local-pinned engine is not
importable, or if an auto-resolved engine needs Docker but Docker is
unavailable. See [Docker setup](/how-to/docker-setup) if your environment is not
yet configured.
:::

```yaml
task:
  model: Qwen/Qwen2.5-0.5B
  random_seed: 42
  dataset:
    source: aienergyscore
    n_prompts: 30
    order: interleaved
  max_input_tokens: 256
  max_output_tokens: 256
```

Same model + same prompts + same token budget across every cell of the
sweep. This is what makes cross-cell comparison legitimate - only the
varied parameters differ.

`max_input_tokens` and `max_output_tokens` *control* the FLOPs budget.
If you let prompt and generation lengths float, you can't separate
"this implementation is more efficient" from "this implementation
generated less text."

```yaml
measurement:
  energy_sampler: auto
  baseline:
    enabled: true
    duration_seconds: 30.0
  warmup:
    enabled: true
    n_prompts: 3
    thermal_floor_seconds: 30.0
```

`energy_sampler: auto` probes the host and picks the highest-fidelity
sampler available (NVML → Zeus → CodeCarbon, in that order). The
30-second baseline measures idle GPU power *before* each experiment so
the result file's `Adjusted` energy figure reflects only the inference
work - see [How to interpret results](/how-to/interpret-results).

The sweep section is where the implementation parameters live:

```yaml
sweep:
  # 1. Numerical precision - applies to all three engines.
  transformers.engine_params.dtype: [float16, bfloat16]
  vllm.engine_params.dtype: [float16, bfloat16]
  tensorrt.engine_params.dtype: [float16, bfloat16]

  # 2. Batching strategy - engine-native parameter names.
  harness.transformers.batch_size: [4, 16]
  vllm.engine_params.max_num_seqs: [64, 256]
  tensorrt.engine_params.max_batch_size: [4, 16]

  # 3. Attention backend - measurable energy effect on prefill-heavy work.
  transformers.engine_params.attn_implementation: [sdpa, flash_attention_2]
  vllm.engine_params.attention.backend: [flash_attn, flashinfer]

  # 4. KV-cache reuse - affects steady-state throughput and energy.
  vllm.engine_params.enable_prefix_caching: [true, false]
  tensorrt.kv_cache_reuse:
    - {}
    - tensorrt.engine_params.kv_cache_config.enable_block_reuse: true
      tensorrt.engine_params.kv_cache_config.free_gpu_memory_fraction: 0.9
```

A subtle but important point: each axis is **engine-scoped** by its
key prefix (`transformers.`, `vllm.`, `tensorrt.` - or `harness.<engine>.`
for per-engine harness knobs like batch size). When the sweep
expander processes an experiment cell whose engine is `vllm`, only
`vllm.*` axes are applied to it; the `transformers.*` and
`tensorrt.*` axes are skipped. This is what makes cross-engine sweeps
sensible - you don't end up with `vllm.engine_params.max_num_seqs=64` mixed
into a Transformers experiment.

The `tensorrt.kv_cache_reuse` group illustrates a **dependent group**:
within-group entries are alternatives (unioned, not crossed), while
the group as a whole is crossed against other axes. The empty `{}`
is the baseline; the second entry sets two related fields together.
This is the right way to sweep parameters that travel in pairs.

## Step 3 - Dry-run, then run

Before kicking off the real run, validate the config and see how many
experiments will actually execute:

```bash
llem run configs/tutorials/tutorial-multi-engine.yaml --dry-run
```

Dry-run is a CLI-only path; the Python API does not currently expose an
equivalent flag on `run_study`.

The dry-run resolves the sweep, applies engine-scoped filtering,
deduplicates equivalent cells, and prints a manifest. You should see
something like:

```text
Study: tutorial-multi-engine
  Resolved: 36 experiments (84 expanded → 36 after dedup)
  Per-engine breakdown:
    transformers: 16 (dtype × batch_size × attn × cycles)
    vllm: 16 (dtype × max_num_seqs × attn × prefix_caching)
    tensorrt:  4 (dtype × max_batch_size × kv_cache_reuse - 1 cycle)
  VRAM estimate (per engine): ~4 GB peak (Qwen2.5-0.5B in bf16)
  Estimated wall-clock: 28 min (excluding TRT-LLM first-build ~5 min)
```

> **Sample output above; your numbers will differ** depending on host
> resolution of the Cartesian product, dedup hit rate, and TRT-LLM
> compilation cache state.

If the resolved count and per-engine breakdown match your expectation,
launch the real run:

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
llem run configs/tutorials/tutorial-multi-engine.yaml
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_study

study_result = run_study("configs/tutorials/tutorial-multi-engine.yaml")
print(f"Completed {study_result.summary.completed} experiments")
```

</TabItem>
</Tabs>

You'll see a progress indicator with experiment counters and the
running cell's identifier. Each result lands in
`results/tutorial-multi-engine_<timestamp>/<NNN_cN_*>/result.json`.

## Step 4 - Inspect the manifest and a single result

After the run completes, the study directory looks roughly like this:

```text
results/tutorial-multi-engine_2026-05-07T14-32-08/
├── manifest.json                       # study-level: timing, config, completion
├── 001_c0_qwen-transformers_a1b2c3.../ # one experiment cell
│   ├── result.json                     # measurement metrics only (bundle_version 2.0)
│   ├── config.json                     # engine/model/methodology + resolved config + provenance
│   └── timeseries.parquet              # GPU power/thermal/memory samples
├── 002_c0_qwen-transformers_d4e5f6.../
└── ... (36 cells)
```

`manifest.json` is the study-level record. It contains the resolved
experiment list, study timing, completion status per cell, and the
effective study-level config. This is what you load when you want to
*reason about the study as a whole* rather than one cell.

A single `result.json` is measurement output (bundle_version 2.0). Configuration
moved to the sidecar; only `engine` and `model_name` remain as convenience
copies so the file is self-describing when separated from its directory:

```json
{
  "experiment_id": "qwen-transformers-bf16-bs16-fa2-2026-05-07T14-32-08",
  "bundle_version": "2.0",
  "measurement_mode": "offline",
  "engine": "transformers",
  "model_name": "Qwen/Qwen2.5-0.5B",
  "input_tokens": 6144,
  "output_tokens": 1536,
  "total_tokens": 7680,
  "total_inference_time_sec": 9.8,
  "avg_tokens_per_second": 783.7,
  "total_energy_j": 891.4,
  "mj_per_tok_total": 116.1,
  "mj_per_tok_adjusted": 100.4,
  "total_flops": 1.18e+12,
  "flops_per_output_token": 1.54e+8,
  "energy_breakdown": { "baseline_power_w": 12.3, "...": "..." }
}
```

The authoritative home for engine, model, and methodology identity is the
`config.json` sidecar next to `result.json` in the same cell directory -
alongside the fully resolved config (`declared_config`) and per-field
provenance:

```json
{
  "bundle_version": "2.0",
  "engine": "transformers",
  "engine_version": "4.57.1",
  "model_name": "Qwen/Qwen2.5-0.5B",
  "measurement_methodology": "total",
  "declared_config": { "...": "..." },
  "provenance": { "...": "..." }
}
```

> **Sample numbers above; real values depend on hardware + prompt
> sample.** The structure is stable.

The two energy-per-token figures are the headline:
- `mj_per_tok_total` - millijoules per output token, raw GPU energy
- `mj_per_tok_adjusted` - same, with idle baseline subtracted

For cross-cell comparison the **adjusted** figure is the right pick -
it isolates inference work from the cost of having a GPU plugged in.
The full reasoning is on the
[methodology page](/explanation/methodology/methodology) and the
[energy-measurement explanation](/explanation/methodology/energy-measurement).

## Step 5 - Compare across engines in Python

Loading and grouping results uses the public API. Drop this snippet
into a Python file alongside your study directory:

```python
import json
from collections import defaultdict
from pathlib import Path

study_dir = Path("results/tutorial-multi-engine_2026-05-07T14-32-08")

# Each experiment cell holds result.json (metrics) plus a config.json
# sidecar (engine/model identity + the resolved config). Join them per
# cell: engine and dtype come from config.json, the metric from result.json.
groups: dict[tuple[str, str], list[float]] = defaultdict(list)
for result_path in study_dir.glob("*/result.json"):
    cell = result_path.parent
    result = json.loads(result_path.read_text())
    config = json.loads((cell / "config.json").read_text())

    engine = config["engine"]
    dtype = config["declared_config"][engine]["engine_params"]["dtype"]
    groups[(engine, dtype)].append(result["mj_per_tok_adjusted"])

print(f"{'engine':<14} {'dtype':<10} {'mJ/tok (adj, mean)':>20} {'n':>4}")
for (engine, dtype), values in sorted(groups.items()):
    mean = sum(values) / len(values)
    print(f"{engine:<14} {dtype:<10} {mean:>20.2f} {len(values):>4}")
```

You should see something like:

```text
engine         dtype      mJ/tok (adj, mean)    n
tensorrt       bfloat16                  72.4    2
tensorrt       float16                   68.9    2
transformers   bfloat16                 102.3    8
transformers   float16                  108.7    8
vllm           bfloat16                  84.1    8
vllm           float16                   86.5    8
```

> **Sample numbers above; the *ordering* of magnitudes is what's
> directionally meaningful**, not the precise values. On A100 you
> typically see TRT-LLM lowest, vLLM middle, Transformers highest
> for the per-token energy figure - but the gap between dtypes is
> often within noise for a 0.5B model.

To go further: group by batch size and plot mJ/token vs batch size, or
pivot on `attn_implementation` / `attention.backend` to compare attention
kernels within each engine - all read from each cell's `config.json`
(`declared_config`) joined to its `result.json`. The two-file schema makes
this kind of analysis a few lines of Python.

## Step 6 - What you've learned and where to go next

You've now exercised the full `llem` workflow:

- **Designing** an implementation-parameter sweep with engine-scoped
  axes and dependent groups
- **Resolving** that sweep with `--dry-run` to validate the
  experiment count and VRAM estimates before running
- **Running** a multi-engine study with isolated Docker per engine
- **Inspecting** both the study-level manifest and individual
  result.json files
- **Comparing** results in Python using the universal output metrics
  (mJ/token, tokens/sec) that allow cross-engine comparison

The shape of every research workflow with `llem` looks like this. The
*specifics* - which parameters, which model, which task - change with
your question.

### Sister recipes (How-to)

- [Run with vLLM (Docker)](/how-to/run-with-docker-vllm) - single-engine recipe
- [Run with TensorRT-LLM (Docker)](/how-to/run-with-tensorrt-llm) - single-engine recipe
- [Interpret results](/how-to/interpret-results) - field-by-field walkthrough of `result.json`
- [Troubleshoot](/how-to/troubleshoot) - when a cell fails or a metric looks wrong

### Reference

- [Study config](/reference/study-config) - full sweep / runner / measurement field listing
- [CLI](/reference/cli) - every `llem run` session flag (resume, output, etc.)
- [Engine configuration](/reference/engines/configuration) - per-engine parameter spaces

### Conceptual depth (Explanation)

- [Methodology](/explanation/methodology/methodology) - warmup, baseline, thermal management
- [What we measure](/explanation/methodology/what-we-measure) - energy / throughput / FLOPs
- [Parameter discovery](/explanation/architecture/parameter-discovery) - how a submitted config is validated against the engine's rules
- [Comparison context](/explanation/methodology/comparison-context) - relationship to MLPerf, AI Energy Score
