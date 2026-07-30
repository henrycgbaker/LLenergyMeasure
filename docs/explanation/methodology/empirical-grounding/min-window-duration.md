---
slug: /methodology/empirical-grounding/min-window-duration
title: Minimum window duration for stable energy-per-token
description: A measurement study establishing how long a server measurement window must be for its energy-per-token figure to be repeatable, and the three defaults it grounds.
---

# Minimum window duration for stable energy-per-token

In server mode, LLenergyMeasure measures energy-per-token over a fixed-duration
**window** while an inference server handles a steady request rate. How long must
that window be?

There is a tension. If the window is too short, the energy-per-token figure is
dominated by counting noise (few requests complete, so the ratio of energy to
tokens swings from window to window) and is not repeatable. If the window is too
long, it wastes GPU time and slows every study. This study finds the shortest
window whose energy-per-token figure is stable, and derives three shipped
defaults from the result.

> Traceability: this is study E2, run 2026-07-23. The values below are its
> ratified results.

---

## Question

For each request rate, what is the shortest measurement window whose
energy-per-token is stable, where "stable" means its variability is at or below a
coefficient of variation of 0.05 and stays there for all longer windows?

The single default the tool ships must cover the worst (slowest-to-stabilise)
rate, so the window-duration default is the largest per-rate answer.

---

## Method

**Capture, then slice.** Rather than run one measurement per candidate duration,
the study takes one long steady capture per request rate and slices candidate
windows out of it. Each rate ran a 60 s warmup at the target rate (to reach
loaded equilibrium) followed by a single 540 s measured capture. Analysis uses
only the measured capture; the warmup is discarded.

**Coefficient of variation.** For a candidate window of duration `d`, the window
is split into 4 consecutive sub-windows of length `d/4`. Energy-per-token is
computed for each sub-window (energy by trapezoidal integration of the GPU power
samples; output tokens attributed to the sub-window in which each request
completes, counted client-side). The window's coefficient of variation is the
population standard deviation divided by the mean across those 4 sub-window
values. This is the same coefficient-of-variation statistic the offline
steady-state detector uses.

**Sliding placements.** A window of duration `d` is slid across the 540 s capture
one sub-window at a time, and the reported figure is the **mean** coefficient of
variation across all placements (the expected variability at that duration,
robust to where the window happens to fall).

**Per-rate floor.** For each rate, the floor is the shortest duration whose mean
coefficient of variation is at or below 0.05 and stays at or below 0.05 for every
longer candidate. The window-duration default is the largest floor across rates.

**Rate grid.** Four offered rates spanning sub-saturation to near-saturation:
2, 10, 40, and 75 req/s. The 75 req/s point is near-saturation, about 0.78 of the
measured sustained peak throughput of roughly 95 to 98 req/s for this
engine, model, and GPU. True saturation (queue-limited) was deliberately avoided
as an unstable operating point. Achieved throughput matched the offered rate at
all four levels with zero failed requests.

The operating points, for context (the expected efficiency-versus-utilisation
curve, a sanity check on the pipeline):

| Rate (req/s) | Achieved (req/s) | Tokens/s | p50 latency (s) | Mean power (W) | Energy/token (J) |
|---|---|---|---|---|---|
| 2 | 2.0 | 503 | 0.64 | 141.5 | 0.282 |
| 10 | 10.2 | 2610 | 0.65 | 170.6 | 0.065 |
| 40 | 40.1 | 10272 | 0.83 | 190.1 | 0.019 |
| 75 (near-saturation) | 74.9 | 19174 | 1.68 | 229.4 | 0.012 |

Energy-per-token falls sharply with load (fixed idle overhead is amortised over
more tokens) while power rises with load.

---

## Result: energy-per-token variability versus window duration

Mean coefficient of variation across sliding placements. The threshold is 0.05.
**Bold** marks each per-rate floor (the shortest duration at or below 0.05 that
stays at or below 0.05 for all longer durations).

| Rate (req/s) | 15 s | 30 s | 60 s | 120 s | 240 s | 480 s | Per-rate floor |
|---|---|---|---|---|---|---|---|
| 2 | 0.273 | 0.196 | 0.108 | 0.096 | **0.048** | 0.013 | **240 s** |
| 10 | 0.117 | 0.085 | 0.067 | **0.040** | 0.024 | 0.025 | **120 s** |
| 40 | 0.065 | **0.048** | 0.030 | 0.025 | 0.012 | 0.021 | **30 s** |
| 75 (near-saturation) | 0.055 | **0.035** | 0.024 | 0.013 | 0.010 | 0.009 | **30 s** |

Per-rate floors: 240 s at 2 req/s, 120 s at 10 req/s, 30 s at 40 and 75 req/s.
**The maximum across rates is 240 s.**

The knee is clean and monotone: variability falls with both duration and rate.
Higher rates stabilise faster because more tokens complete per sub-window, so
there is less counting noise; the sparsest rate (2 req/s) is the hardest and sets
the fail-safe default.

---

## The three defaults

### 1. Window duration: 240 s

The maximum across the per-rate floors, governed by the 2 req/s rate. It is
already a clean value on the candidate grid, so no rounding was needed.

### 2. Ramp exclusion: 30 s (absolute)

Before a window opens, an initial slice is excluded so the batch-fill transient
(the running batch filling to its equilibrium occupancy) does not bias the
measurement. Measured as a power rise time, this transient is 0 s at 2 and
10 req/s, 10 s at 40 req/s, and 30 s at 75 req/s: it grows with load.

The default is the near-saturation worst case, **30 s**, parameterised as an
absolute value rather than a fraction, because the transient is a physics effect
tied to how fast the batch reaches equilibrium occupancy, independent of window
length. As a fraction of the 240 s window it is 12.5%.

### 3. Per-level stability rule: coefficient of variation 0.05, over 3 consecutive windows

A rate level passes when its energy-per-token agrees to within a coefficient of
variation of 0.05 across 3 consecutive windows, stable through the end of the
level. Both numbers are grounded here:

- The **0.05 tolerance** is where the measured curves cross a clean, well-separated
  knee for every rate, and it matches the offline steady-state detector's
  tolerance.
- The **3 consecutive windows** rule is comfortably satisfied by the data:
  window-to-window energy-per-token deviation across consecutive windows was at or
  below 0.049 at 120 s windows and 0.040 at 240 s windows across all rates, so a
  "3 windows within 5%" gate is both satisfiable and well calibrated.

---

## Borderline note at 2 req/s

The 240 s default is the faithful rounded-up answer, but the sparsest rate is
only just inside the threshold at that duration. At 240 s, the 2 req/s rate has a
mean coefficient of variation of 0.048 (0.0476 precisely), a maximum across
placements of 0.060, and only half of individual placements below 0.05. It
reaches full per-placement robustness at 480 s.

So 240 s is the computed floor and the honest maximum-across-rates answer. A
maintainer who wants extra margin at the sparsest rates can round up (roughly
300 s, or 480 s for full per-placement robustness at 2 req/s). Every other rate
is comfortably stable at its floor.

---

## Limitations

The default is validated for the configuration measured, and should be
re-confirmed outside it:

- **Single engine (vLLM).** The floor may differ on another continuous-batching
  runtime. 240 s should be treated as the vLLM-derived default until re-confirmed.
- **Single model (Qwen2.5-0.5B-Instruct) and single fixed output length**
  (256 tokens per request). A larger model, a variable output-length
  distribution, or a burstier arrival process would change per-sub-window token
  variance and could shift the floor.
- **Single host, one A100-PCIE-40GB.** A different accelerator, multiple GPUs, or
  a noisier shared-host power environment could raise variability at short
  durations and push the floor up.
- **Near-saturation is one point (75 req/s).** Behaviour above roughly 80 req/s
  (the queue-limited regime) is not characterised for the window floor.
- **Ramp is the load transient only.** This study measures the batch-fill
  transient, not thermal equilibrium. The thermal contribution is a separate
  study; see [Loaded thermal equilibrium](/explanation/methodology/empirical-grounding/loaded-thermal-equilibrium).
