---
title: Measurement warnings
description: What the tool's runtime warnings mean, how they affect measurement validity, and when to re-run.
---

# Measurement warnings

LLenergyMeasure emits structured log warnings at runtime when it detects
conditions that may reduce the accuracy or reproducibility of a measurement.
This page documents each warning category, what triggers it, and how to
interpret the downstream metrics in light of it.

---

## Warmup did not converge

**Logged by:** `llenergymeasure.harness.warmup`

**Message:** `Warmup did not converge after N prompts (final CV=X.XXX, target=Y.YYY)`

**What it means:** The warmup phase runs up to `warmup.max_prompts` inferences
and checks whether latency has stabilised (coefficient of variation below
`warmup.cv_threshold`, default 0.05). This warning fires when the CV threshold
was not reached before the maximum iteration count.

Common causes, in order of likelihood:

1. **Sustained thermal throttling** - the GPU is not reaching a stable clock
   state between warmup prompts. Typical on consumer cards without active
   cooling control.
2. **Memory allocation jitter** - the first several inferences trigger CUDA
   memory allocator activity that settles out slowly for large models.
3. **Background GPU load** - another process is contending for GPU time,
   creating latency variance the warmup cannot overcome.
4. **`warmup.max_prompts` set too low for the model** - larger models take more
   warmup inferences to reach steady state.

**Effect on measurement validity:** When warmup does not converge, the GPU
may still be in a transient thermal or memory state at measurement start. The
energy and throughput figures are still technically correct (the harness measures
what actually happened), but the transient state means the measurement does not
reflect steady-state operation. Figures are likely to be higher variance than
usual and may not match a run on a pre-warmed system.

---

## Sampling rate gap detected

**Logged by:** `llenergymeasure.energy.nvml`

**Message:** `Power sampling gap of Xms detected (target: 100ms). Energy calculation may be less accurate.`

**What it means:** NVML power samples are collected every 100 ms by default.
This warning fires when any consecutive gap between samples exceeds 200 ms
(twice the target interval). The energy figure is computed by trapezoidal
integration over the sample sequence; a large gap degrades the integration
accuracy proportionally to how much power variation occurred during the gap.

Common causes:

1. **OS scheduling jitter** - the sampling thread was preempted. More common
   under high system load.
2. **Python GIL contention** - occurs rarely, typically during model loading.
3. **Container overhead** - Docker adds a small scheduling indirection that can
   widen gaps under load.

**Effect on measurement validity:** For a gap of 300-400 ms, the energy error
is typically well within the 5% NVML accuracy floor and not meaningfully
actionable. For gaps exceeding 1,000 ms, the integration error may be larger
than the hardware-level measurement uncertainty, particularly during high-power
transients (e.g. the first few seconds of a vLLM engine init).

**When to re-run vs accept:**

- A single gap of 200-500 ms during a 30+ second run: accept. The error is
  smaller than NVML's intrinsic accuracy limit.
- Multiple gaps, or a gap during a short run (< 5 seconds): re-run on a quieter
  system, or increase `n_prompts` to produce a longer measurement window that
  dilutes the impact.

---

## Thermal throttling detected

**Recorded in result.json:** `throttle.thermal.any: true` or `throttle.thermal.hw: true`

**What it means:** NVML detected active thermal throttling during the measurement
window - the GPU reduced its clock speed to prevent overheating. This is
recorded in `result.json` under `throttle` and logged as a summary at
run end.

`ThrottleInfo` is symmetric across two axes, `thermal` and `power`. Each axis is a
`ThrottleAxis` with the same three flags:

| Field | Meaning |
|-------|---------|
| `throttle.thermal.any` | Any thermal throttle detected (hardware or software) |
| `throttle.thermal.hw` | Hardware thermal limit triggered (GPU too hot) |
| `throttle.thermal.sw` | Software thermal slowdown triggered |
| `throttle.power.any` | Any power throttle detected (hardware or software) |
| `throttle.power.hw` | Hardware power-brake slowdown triggered |
| `throttle.power.sw` | Software power cap triggered |
| `throttle.throttle_duration_sec` | Seconds of throttled operation |
| `throttle.max_temperature_c` | Peak GPU temperature during the run (Celsius) |

**Effect on measurement validity:** Thermal throttling directly reduces GPU
clock speed, which reduces throughput and also reduces instantaneous power draw.
The energy figure reflects the throttled run - it is lower than what you would
measure on a thermally stable system. This means:

- Throughput is understated relative to peak capacity.
- Energy per token may appear artificially good (lower power, same token count)
  or artificially poor (lower throughput while still drawing base power), depending
  on throttle severity.

For cross-system comparisons, a thermally-throttled run is not a fair baseline.

**When to re-run vs accept:**

- For publication: re-run after allowing the GPU to cool (> 10 minutes idle, or
  increase `study_execution.experiment_gap_seconds`). If throttling persists, the
  system cooling is insufficient for sustained inference and you should document this.
- For quick iteration: note the warning. Relative comparisons within the same
  thermal state are still meaningful.

---

## Warmup prompt failed

**Logged by:** `llenergymeasure.harness.warmup`

**Message:** `Warmup prompt N failed: <exception>`

**What it means:** An individual warmup inference raised an exception. The prompt
is skipped and warmup continues. This is a soft failure: the warmup loop is
resilient by design because early prompts sometimes trigger edge-case model
behaviour.

**Effect on measurement validity:** One or two failed warmup prompts out of 5-10
total rarely affect the measurement. If a large fraction of warmup prompts fail,
the warmup is effectively shorter than configured, which may contribute to a
"warmup did not converge" warning (see above). If every warmup prompt fails,
the run proceeds without warmup.

**When to re-run vs accept:** If the failure message indicates a configuration
problem (e.g. OOM, invalid token, model load error), fix the root cause before
re-running. Intermittent failures (network timeout fetching model weights, race
condition in multi-GPU init) are safe to re-run.

---

## Energy tracking unavailable

**Logged by:** `llenergymeasure.energy.codecarbon`

**Messages:**
- `Energy tracking unavailable: <reason>`
- `Continuing without energy metrics (common in containers)`

**What it means:** The CodeCarbon energy sampler failed to initialise. This
warning is emitted when `energy_sampler: codecarbon` is configured but
CodeCarbon cannot access power hardware.

**Effect on measurement validity:** If CodeCarbon was your intended sampler,
energy fields in `result.json` will be null. The run continues and captures
throughput metrics. NVML (the default sampler) is not affected by this warning.

**When to re-run vs accept:** Switch to `energy_sampler: nvml` (default) unless
you specifically need CodeCarbon's CPU+DRAM+GPU combined tracking. NVML is more
reliable inside Docker containers.

---

## Baseline container dispatch failed

**Logged by:** `llenergymeasure.study.baseline_container`

**Messages:**
- `Baseline container dispatch failed: docker binary not found on PATH`
- Various Docker-related warnings about container start/stop failures

**What it means:** The baseline power measurement runs inside a separate
lightweight container to isolate the GPU idle state. If Docker is unavailable
or the container fails to start, the baseline measurement is skipped.

**Effect on measurement validity:** Without a baseline, `adjusted_energy_j` in
`result.json` is null. `total_energy_j` is still measured and valid, but you
cannot subtract idle power. Cross-experiment comparisons that use adjusted energy
are not possible.

**When to re-run vs accept:** Fix the Docker issue (see [How to: troubleshoot](/how-to/troubleshoot))
and re-run if adjusted energy is required. For throughput-focused studies, total
energy without baseline subtraction is acceptable.

---

## GPU memory measurement failed

**Logged by:** `llenergymeasure.study.gpu_memory`

**Message:** `GPU memory measurement failed: <reason>` (or similar)

**What it means:** The memory snapshot before or after inference could not be
taken. This affects `gpu_memory_delta_mb` and related memory fields but not
energy or throughput.

**Effect on measurement validity:** Memory fields in `result.json` will be null
or zero. Energy and throughput figures are unaffected.

---

## Interpreting warnings in combination

Some warning combinations carry specific interpretations:

| Combination | Likely cause | Implication |
|-------------|-------------|-------------|
| Warmup did not converge + thermal throttling | GPU entering run too hot | Both throughput and energy are from a non-steady-state run |
| Sampling gap + thermal throttling | System under load | Energy figure has both integration error and throttle bias |
| Warmup did not converge (no thermal throttling) | Memory jitter or low `max_prompts` | Increase `max_prompts`; re-run |
| All warnings present | Overloaded system | Dedicated GPU environment strongly recommended |

---

## See also

- [Methodology: energy measurement](/explanation/methodology/energy-measurement) - NVML sampling and baseline subtraction
- [Methodology: glossary](/explanation/methodology/glossary) - definitions for thermal stabilisation, baseline, NVML
- [How to: interpret results](/how-to/interpret-results) - reading result.json fields
- [Reference: study config](/reference/study-config#warmup-warmup) - warmup configuration options
