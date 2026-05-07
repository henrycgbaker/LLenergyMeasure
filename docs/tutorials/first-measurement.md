---
title: Your first measurement
description: Measure GPT-2 energy in five minutes — install, run, read the result.
---

# Your first measurement

In the next five minutes you'll install `llenergymeasure`, run a measurement on GPT-2, and read the result file. By the end you'll know:

- How `llem` is structured (host orchestrator + per-engine Docker images)
- What a single experiment measures (energy, throughput, FLOPs)
- Where results are written and how to read them

This is a **tutorial** — guided and linear. For goal-driven recipes
(e.g. *"how do I run with vLLM?"*), see [How-to](/how-to/install).

## Prerequisites

- `llenergymeasure` installed — see [How to install](/how-to/install)
- Docker + NVIDIA Container Toolkit — every engine, including PyTorch
  Transformers, runs inside a per-engine Docker image
- An NVIDIA GPU available

## Step 1: Verify your environment

```bash
llem config
```

Check that the output shows your GPU detected and an energy sampler
selected. Engines will show as "not installed" on host — that is expected;
they run inside Docker. See the [Docker setup how-to](/how-to/docker-setup)
if any of the pre-flight checks fail.

## Step 2: Run your first experiment

```bash
llem run --model gpt2 -e pytorch
```

This runs GPT-2 (124M parameters). On first run, the model downloads from
HuggingFace (~500 MB). Subsequent runs use the cache.

Default settings: 100 prompts, `aienergyscore` dataset, `bfloat16` dtype.

You'll see a progress indicator on stderr, then results printed to stdout:

```
Result: gpt2-pytorch-bf16-2026-05-07T14-32-08     ← unique experiment ID

Energy                                          ← GPU energy consumed
  Total          847 J                          ← total joules for all 100 prompts
  Baseline       12.3 W                         ← idle GPU power (subtracted from total)
  Adjusted       723 J                          ← energy minus baseline × duration

Performance                                     ← throughput and compute
  Throughput     312 tok/s                      ← output tokens per second (all 100 prompts)
  FLOPs          4.21e+11 (roofline, medium)    ← estimated FLOPs (method, confidence)

Timing                                          ← wall-clock time
  Duration       1m 38s                         ← total experiment wall time
  Warmup         5 prompts excluded             ← thermal stabilisation prompts (not in metrics)
```

## Step 3: Read the results

Each field maps to a measurement decision:

| Field | What it measures |
|-------|-----------------|
| `Total` (J) | Raw GPU energy consumed during the experiment |
| `Baseline` (W) | Idle GPU power measured before the run |
| `Adjusted` (J) | Energy minus `Baseline × Duration` — net inference energy |
| `Throughput` (tok/s) | Output tokens generated per second across all prompts |
| `FLOPs` | Estimated floating-point operations (method and confidence shown) |
| `Duration` | Wall-clock time for the full experiment |
| `Warmup` | Prompts run for thermal stabilisation, excluded from metrics |

Why subtract baseline? Because you're measuring **inference**, not "GPU
plugged in." The full reasoning is on the
[methodology page](/methodology/methodology) and the
[energy-measurement explanation](/methodology/energy-measurement).

## Step 4: Inspect the output files

Results are written to `results/` by default:

```
results/
└── gpt2-pytorch-bf16-2026-05-07T14-32-08/
    └── result.json        # full record (all metrics, config, metadata)
```

The JSON file is the scientific record — all raw metrics, the resolved
config, timestamps, and any measurement warnings. See
[How to interpret results](/how-to/interpret-results) for a walkthrough.

Specify a different output directory with `--output`:

```bash
llem run --model gpt2 -e pytorch --output /data/experiments
```

## What you've learned

- `llem run` runs one experiment end-to-end, writes a `result.json`, prints a
  human-readable summary.
- Even Transformers/PyTorch runs go through a Docker image — keeps the
  measurement environment reproducible.
- Energy is measured *and* baseline-adjusted; both numbers ship in the
  result file so you can pick the framing your study needs.

## Where to go next

- [How to: run with vLLM (Docker)](/how-to/run-with-docker-vllm)
- [How to: run with TensorRT-LLM (Docker)](/how-to/run-with-tensorrt-llm)
- [Reference: study config](/reference/study-config) — sweep syntax for multi-experiment studies
- [Reference: CLI](/reference/cli) — every `llem` flag
