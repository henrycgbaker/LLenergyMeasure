---
title: Run an experiment with vLLM (Docker)
description: Use the vLLM engine for high-throughput inference measurements.
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Run an experiment with vLLM (Docker)

This recipe runs a single measurement against the vLLM engine. Use it
when you want to measure inference under vLLM's continuous-batching
runtime rather than HuggingFace `transformers`.

:::caution vLLM requires Docker
The vLLM engine runs inside a Docker container. Attempting to run vLLM
without Docker raises a `PreFlightError` at preflight. Ensure Docker and
the NVIDIA Container Toolkit are installed before proceeding.
:::

## Prerequisites

- `llenergymeasure` installed (host-side orchestrator)
- Docker + NVIDIA Container Toolkit - see [Docker setup](/how-to/docker-setup)
- vLLM upstream image (`vllm/vllm-openai`) - pulled automatically from Docker Hub on first run; pre-fetch per [Docker setup](/how-to/docker-setup)

## 1. Create a config file

<Tabs groupId="surface">
<TabItem value="yaml" label="YAML">

Create `experiment.yaml`:

```yaml
engine: vllm
task:
  model: gpt2
  dataset:
    source: aienergyscore
    n_prompts: 50
runners:
  vllm: container
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment(
    model="gpt2",
    engine="vllm",
    n_prompts=50,
)
print(result)
```

</TabItem>
</Tabs>

## 2. Run the experiment

<Tabs groupId="surface">
<TabItem value="cli" label="CLI">

```bash
llem run experiment.yaml
```

</TabItem>
<TabItem value="python" label="Python">

```python
from llenergymeasure import run_experiment

result = run_experiment("experiment.yaml")
```

</TabItem>
</Tabs>

What happens:

1. Pre-flight checks run: Docker CLI, NVIDIA Container Toolkit, GPU
   visibility inside container, CUDA/driver compatibility.
2. The upstream vLLM image (`vllm/vllm-openai`) is pulled from Docker Hub on
   first run, with the llenergymeasure source bind-mounted into it.
3. The container launches, runs the experiment, and streams results back.
4. Results are printed to stdout and saved to `results/`.

## 3. Read the results

The output format matches the Transformers track. The key difference is
`engine: vllm` in the experiment ID and result file. See
[How to interpret results](/how-to/interpret-results) for the field-by-field
walkthrough.

## Related

- [Tutorial: Your first measurement](/tutorials/first-measurement) - start here if you've never run `llem`
- [How to: run with TensorRT-LLM](/how-to/run-with-tensorrt-llm) - sister recipe for the TRT-LLM engine
- [Reference: engine configuration](/reference/engines/configuration) - every vLLM-specific config field
