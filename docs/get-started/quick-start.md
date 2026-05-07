---
title: Quick start
description: Run your first energy measurement in 60 seconds.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quick start

## Run your first measurement

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
llem run --model gpt2 --engine transformers
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment(model="gpt2", engine="transformers")
print(result)
```

</TabItem>
</Tabs>

On first run, GPT-2 (around 500 MB) downloads from HuggingFace and the
engine Docker image is resolved. Subsequent runs use the local cache and
typically complete in under two minutes.

A progress indicator prints to stderr. When the experiment finishes, a
short summary prints to stdout and a structured `result.json` is written
under `results/`. Numeric values vary by hardware; the shape is:

```
Result: gpt2-transformers-bf16-<timestamp>

Energy
  Total          <joules>
  Baseline       <watts>
  Adjusted       <joules - baseline * duration>

Performance
  Throughput     <tokens/sec>
  FLOPs          <estimate>

Timing
  Duration       <wall-clock>
  Warmup         <n prompts excluded>
```

---

## Read the result

| Field | What it means |
|-------|---------------|
| `Total` (J) | Raw GPU energy across the prompt set |
| `Baseline` (W) | Idle GPU power measured before the run |
| `Adjusted` (J) | `Total` minus `Baseline x Duration`: the energy attributable to inference |
| `Throughput` (tok/s) | Output tokens per second across all prompts |
| `FLOPs` | Estimated floating-point operations (validity check, not a headline metric) |
| `Duration` | Wall-clock time for the full experiment |
| `Warmup` | Prompts excluded for thermal stabilisation |

`Adjusted` is the most useful number when comparing configurations: it
isolates the inference energy from idle-GPU draw.

For the full field list in `result.json`, see
[Results schema](/reference/results-schema).

---

## Result file

Results are written to `results/` in your working directory:

```
results/
└── gpt2-transformers-bf16-<timestamp>/
    └── result.json
```

The experiment ID encodes the model, engine, dtype, and a timestamp.

---

## What's next

- **Deeper tutorial** - [Your first measurement](/tutorials/first-measurement)
  walks through each step with full explanations.
- **Understand the result** - [How to interpret results](/how-to/interpret-results)
  explains what numbers are normal and how to compare across runs.
- **Find the right workflow** - [Choose your path](/get-started/choose-your-path)
  routes you to the right guide for your use case.
