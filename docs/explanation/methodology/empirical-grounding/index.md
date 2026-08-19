---
slug: /methodology/empirical-grounding
title: Empirical grounding
description: How LLenergyMeasure's defaults and methodology choices are grounded. Every shipped number traces to a measurement study run on real hardware or to a cited source, and carries the envelope within which it was validated.
---

# Empirical grounding

LLenergyMeasure is a measurement instrument, so its defaults are held to a
measurement standard. Every shipped default and methodology choice catalogued
here is grounded in one of two ways: a **measurement study** run on real
hardware, or an explicit **citation** to established practice. Where a number
could be studied rather than assumed, it was studied, and those studies are
**pre-registered where possible**, with their pass and fail criteria fixed
before any data was collected. Every grounded number below traces to a study
page or a named source, and carries the validity envelope within which it holds.

This page is a registry. The study pages linked from it explain how each number
was measured.

---

## Registry

| Grounded default or claim | Shipped value | What grounds it | Validity envelope |
|---|---|---|---|
| Server measurement window duration | 240 s (default) | [Minimum window duration study](/explanation/methodology/empirical-grounding/min-window-duration): energy-per-token variability drops below a coefficient of variation of 0.05 at 240 s for the slowest-to-stabilise request rate. | vLLM, Qwen2.5-0.5B-Instruct, one A100-PCIE-40GB, fixed 256-token outputs. Re-confirm on other engines, models, and hosts. |
| Server ramp exclusion | 30 s, absolute (12.5% of the 240 s window) | Same study: the batch-fill power transient is longest (30 s) at near-saturation; sub-saturation rates reach operating power within 0 to 10 s. | Same envelope. Saturation-dependent, so the fixed value is taken at the near-saturation worst case. |
| Per-level stability tolerance | Coefficient of variation of 0.05 | Same study: the measured curves cross 0.05 at a clean knee for every rate. Mirrors the offline steady-state detector's tolerance. | Same envelope. |
| Per-level agreement rule | 3 consecutive windows within tolerance | Same study: window-to-window energy-per-token deviation stayed at or below 0.049 at 120 s windows and 0.040 at 240 s windows, so the 3-window rule has margin. | Same envelope. |
| Per-window diagnostic sub-windows | 4 sub-windows | Same study: the calibration constant behind the reported coefficient of variation. Pinned, not configurable. | Same envelope. |
| Server warmup convergence gate | Power plateau AND temperature settled AND zero active thermal throttle | [Loaded thermal-equilibrium study](/explanation/methodology/empirical-grounding/loaded-thermal-equilibrium): die temperature settles 90 to 192 s **after** power in every measured cell. | vLLM, Qwen2.5-0.5B-Instruct, one A100-PCIE-40GB. The temperature observable is load-bearing here and is expected to matter more on hotter workloads. |
| Why the temperature observable is required | Power stability alone is not sufficient | Same study: opening the window on power alone biases energy-per-token by up to -12.9%, an error the temperature observable prevents and that exceeds the 0.05 tolerance. | Same envelope. |
| Server warmup fixed-mode duration | 300 s (default) | Same study: the cold-start worst-case equilibration time was 252 s, rounded up to 300 s on the study grid. | Same envelope. A 60 s value stays available as an explicit fast choice, documented as a convenience floor, not a thermal-equilibrium claim. |
| Server warmup convergence timeout | 900 s | Same study: three times the worst measured equilibration time, clamped to the 900 s ceiling. A never-hang failsafe, not an operating point. | Same envelope. |
| Offline warmup prompt count | 5 prompts | Cited practice: DeepSpeed uses 5 to 10 warmup rounds, Zeus uses 10, AI Energy Score uses 10. | Offline (batch) measurement path. |
| Offline thermal floor wait | 60 s | Chosen conservative idle-settling default, not externally mandated: the approximate timescale on which datacenter GPUs settle after a step change in load. | Offline path **only**. An idle wait would bias server energy-per-token (see note below). |
| Open-loop Poisson arrivals | Inter-arrival coefficient of variation of about 1, rates 1 to 100 req/s | [Open-loop arrivals contract](/explanation/methodology/empirical-grounding/open-loop-arrivals): continuously enforced by a conformance test in CI. | Server-mode load generation. Gamma arrivals track a configurable burstiness parameter instead. |
| Energy sampler polling (NVML) | 100 ms interval, trapezoidal integration, instantaneous power reading | Cited: the 100 ms interval matches the A100 power-reading refresh period, so polling faster returns stale values ([NVML reference](https://docs.nvidia.com/deploy/nvml-api/index.html)); trapezoidal integration is standard for non-uniform timesteps and is sensor-noise-limited. See [Energy measurement](/explanation/methodology/energy-measurement). | The +/-5% sensor floor is a hardware limit shared by all NVML-based samplers. On-board sensors also read intermittently, so polled power under-resolves excursions between readings (Burtscher et al., 2014; Bridges et al., 2016). |
| Baseline measurement location | Measured in the same CUDA environment as the work it is subtracted from | Controlled host-versus-container comparison: a host-measured baseline under-counts container idle power by about 8.7 W per A100, roughly 19% of adjusted energy on a 4-GPU, 120 s run. See [Baseline power](/explanation/methodology/methodology#baseline-power). | A100-PCIE-40GB, PyTorch image. A per-image cache key keeps the correction apples-to-apples across engine images. |
| Default output and input length | 256 output tokens, 256 input tokens | A peer-benchmark survey grounds the 256 **output** length: it is the modal vLLM synthetic-throughput default, close to the ShareGPT median, and it balances the prefill-to-decode ratio. The 256 **input** length is the shipped default following a later change; the original survey premised and recommended a 512 input length, so the shipped input length is not grounded by it. See [Dataset choice](/explanation/methodology/dataset-context). | Development and iteration default; users raise it for generation-heavy tasks and publication-grade runs. |
| Transformers in server mode | Rejected at config validation (v0.7) | Upstream verdict at the pinned Transformers version: `transformers serve` is scoped to evaluation, experimentation, and moderate load; it exposes no first-class liveness endpoint, so a real-probe readiness check cannot be satisfied; and it auto-unloads the model after 300 seconds idle. | vLLM and TensorRT-LLM are the server-mode engines at v0.7. Transformers server support is a fast-follow. |

---

## Why the offline thermal floor is offline-only

The offline measurement path warms the GPU with warmup prompts and then waits a
fixed **thermal floor** (60 s by default) for temperature to plateau before it
starts measuring a batch. The wait is correct there: the die is idle between the
warmup burst and the batch, and the wait lets the transient settle.

Server mode is different. A serving engine under a steady request rate is at a
**loaded** thermal equilibrium, not an idle one. Inserting an idle wait before a
server measurement window would let the die cool, so the window would then open
on a die that is re-warming under load, which is exactly the bias the
[loaded thermal-equilibrium study](/explanation/methodology/empirical-grounding/loaded-thermal-equilibrium)
measures. Server mode therefore reaches equilibrium with warmup **traffic** and
gates on the loaded thermal state, never on an idle wait.

---

## Studies

- **[Minimum window duration for stable energy-per-token](/explanation/methodology/empirical-grounding/min-window-duration).**
  How long a server measurement window must be for its energy-per-token figure to
  be repeatable. Grounds the 240 s window default, the 30 s ramp exclusion, and
  the per-level stability rule.

- **[Loaded thermal equilibrium and the warmup gate](/explanation/methodology/empirical-grounding/loaded-thermal-equilibrium).**
  A pre-registered study of how far die temperature lags power under serving load,
  and why the server warmup gate needs a temperature observable. Grounds the
  three-observable gate, the 300 s fixed-mode floor, and the 900 s timeout.

- **[Open-loop Poisson arrivals](/explanation/methodology/empirical-grounding/open-loop-arrivals).**
  Why server load is generated open-loop, the measured arrival-process contract,
  and the conformance test that enforces it in CI.

---

## Release-gate validation

Grounding a default once is not enough for an instrument that ships new versions.
Each milestone release is gated on a GPU regression run that re-confirms the
measurement pipeline end-to-end on real hardware before the version is tagged, so
a regression in any grounded behaviour blocks the release rather than shipping.
See the [release process](/contributing/release-process) for the broader release
flow.

## Research-track boundary

This section documents the numbers and methodology that ship **inside the
measurement instrument**. LLenergyMeasure also runs a separate research track on
engine-configuration knowledge (how the tool learns each engine's valid
parameter space). That work is a different concern with its own lifecycle and is
not part of this instrument's measurement grounding, so it is not catalogued
here.
