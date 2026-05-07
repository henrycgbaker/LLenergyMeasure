---
title: Quick start
description: Run your first energy measurement in 60 seconds.
---

# Quick start

## Run your first measurement

#### CLI

```bash
llem run --model gpt2 -e transformers
```

#### Python

```python
from llenergymeasure import run_experiment

result = run_experiment(model="gpt2", engine="transformers")
print(result)
```

---

On first run, GPT-2 (~500 MB) downloads from HuggingFace and the
Docker image is resolved. Subsequent runs use the local cache and
complete in under two minutes.

You will see a progress indicator on stderr. When the experiment
finishes, the result prints to stdout:

```
Result: gpt2-transformers-bf16-2026-05-07T14-32-08

Energy
  Total          847 J
  Baseline       12.3 W
  Adjusted       723 J

Performance
  Throughput     312 tok/s
  FLOPs          4.21e+11 (roofline, medium)

Timing
  Duration       1m 38s
  Warmup         5 prompts excluded
```

---

## Read the result

| Field | What it means |
|-------|---------------|
| `Total` (J) | Raw GPU energy for all 100 prompts |
| `Baseline` (W) | Idle GPU power measured before the run |
| `Adjusted` (J) | Total minus `Baseline x Duration` - net inference energy |
| `Throughput` (tok/s) | Output tokens per second across all prompts |
| `FLOPs` | Estimated floating-point operations (method + confidence shown) |
| `Duration` | Wall-clock time for the full experiment |
| `Warmup` | Prompts excluded for thermal stabilisation |

The most useful number for comparing models is `Adjusted` - it isolates
the energy attributable to inference rather than the GPU sitting idle.

For the full schema of every field in the result JSON, see
[Reference: results schema](/reference/results-schema).

---

## Result file

Results are written to `results/` in your working directory:

```
results/
└── gpt2-transformers-bf16-2026-05-07T14-32-08/
    └── result.json
```

The experiment ID encodes the model, engine, dtype, and timestamp.

---

## What's next

- **Deeper tutorial** - [Your first measurement](/tutorials/first-measurement)
  walks through each step with full explanations.
- **Understand the result** - [How to interpret results](/how-to/interpret-results)
  explains what numbers are normal and how to compare across runs.
- **Find the right workflow** - [Choose your path](/get-started/choose-your-path)
  routes you to the right guide for your use case.
