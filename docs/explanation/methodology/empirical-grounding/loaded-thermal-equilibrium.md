---
slug: /methodology/empirical-grounding/loaded-thermal-equilibrium
title: Loaded thermal equilibrium and the warmup gate
description: A pre-registered study of how far GPU die temperature lags power under serving load, why the server warmup gate needs a temperature observable, and the floor and timeout it grounds.
---

# Loaded thermal equilibrium and the warmup gate

When a GPU goes from idle to sustained serving load, its **power draw** settles
within seconds. Its **die temperature** does not: it keeps climbing for much
longer. Temperature changes leakage power and clock behaviour, so a measurement
window opened as soon as power looks flat can be taken while the die is still
warming, biasing energy-per-token.

This study asks whether that lag is large enough to matter, and, if so, whether
the server warmup gate needs a temperature observable or whether power stability
alone is enough.

> Traceability: this is study E3, run 2026-07-29. The values below are its
> ratified results.

---

## Pre-registration

This study was **pre-registered**. Before any GPU measurement was taken, a
protocol document fixed the hypotheses, the three signals and their numeric
thresholds, the ablation grid, and the exact decision rules (the temperature
settle criterion, the tolerances, and the arithmetic that turns the measured
times into a floor and a timeout). Only after that protocol was reviewed was any
data collected.

Pre-registration matters because the conclusion here is a pass or fail verdict on
a design choice. If the pass and fail thresholds were chosen after seeing the
data, they could be nudged to favour a preferred answer. Fixing them in advance
removes that freedom. None of the GPU energy-measurement papers surveyed for this
work use formal pre-registration, so the pre-registered protocol is itself part
of the contribution.

---

## Method

**Grid.** Eight cells: rates of 2, 10, 40, and 75 req/s, each in two arms. The
**cold** arm starts each cell from an idle-cooled die (the worst case a study's
first run faces). The **warm** arm runs the rates as one contiguous ascending
sweep with no cooldown between levels, so each level starts from the die state the
previous one left (the realistic sweep case). The load shape is identical to the
[minimum-window-duration study](/explanation/methodology/empirical-grounding/min-window-duration):
open-loop Poisson arrivals, fixed 256-token outputs.

**Windows are measured, not gated.** Every cell runs a fixed 600 s observation
window. The window is not stopped on convergence; instead the power-settle time,
the temperature-settle time, and any throttle time are detected afterward from
the trace. Equilibration time is therefore **measured**, not assumed.

**Three signals, one poll.** All three come from a single NVML poll (the same
kind of sampler used for energy measurement):

- **Power settle** (`t_P`): the power series enters and holds its steady band,
  using the same stable-through-end detector as the rest of the harness.
- **Temperature settle** (`t_T`): the trailing-60 s temperature range is at or
  below 2 C and stays there through a further 30 s confirmation.
- **Throttle clear**: the last time any NVML thermal-slowdown bit was active
  (0 if it never fired).

**The bias.** For each cell, energy-per-token is computed over the
power-settled-but-warming interval (between `t_P` and `t_T`) and over the
equilibrated tail. The bias `b` is the signed relative difference between the two.
A negative bias means the early interval reads low.

---

## Per-cell results

All times in seconds from the start of traffic. The gap is `t_T - t_P`. The
equilibration time for a cell is the latest of the three signals; because the
throttle bit never fired and temperature always settled last, it equals the
temperature-settle time in every cell.

| Cell | Power settle `t_P` (s) | Temp settle `t_T` (s) | Gap (s) | Energy/token bias | Mean power (W) | Steady temp (C) | Peak temp (C) |
|---|---|---|---|---|---|---|---|
| Cold, 2 req/s | 0 | 183 | 183 | -5.5% | 148 | 52.0 | 53 |
| Cold, 10 req/s | 80 | 233 | 153 | +2.2% | 178 | 56.1 | 58 |
| Cold, 40 req/s | 60 | 161 | 101 | -5.3% | 197 | 60.5 | 62 |
| Cold, 75 req/s | 60 | 252 | 192 | -12.9% | 225 | 68.8 | 70 |
| Warm, 2 req/s | 30 | 120 | 90 | +1.9% | 149 | 51.6 | 58 |
| Warm, 10 req/s | 0 | 93 | 93 | +4.1% | 178 | 57.2 | 58 |
| Warm, 40 req/s | 0 | 175 | 175 | -7.6% | 196 | 60.0 | 62 |
| Warm, 75 req/s | 20 | 132 | 112 | -2.5% | 238 | 69.0 | 69 |

---

## Findings

**Temperature settles after power in every cell.** The gap spans 90 to 192 s:
power settles in 0 to 80 s while temperature takes 93 to 252 s. The clearest
single case is the cold 2 req/s cell, where power settled at 0 s (as the
minimum-window-duration study also measured) while the die warmed from 29 to 53 C
and did not thermally settle until 183 s. A detector watching power alone would
have opened that window 183 s early.

**The lag biases energy-per-token, by up to -12.9%.** Measuring over the
power-settled-but-warming interval biased energy-per-token low at three of four
cold-arm rates, peaking at **-12.9%** at cold 75 req/s (a cooler die has lower
leakage, so energy-per-token reads low until the die equilibrates). In the warm
arm the sign can flip for the low rates, because a cell that starts from a
preloaded hot die cools toward a lower steady state, so its early interval reads
high instead. Either direction, the temperature observable is what prevents the
bias.

**Verdict: the temperature observable is load-bearing.** The worst bias
(-12.9%) exceeds the 0.05 tolerance the tool already accepts. Every cell whose
bias exceeds that tolerance also has a power-to-temperature gap over 30 s, and no
cell shows excess bias without such a gap, so the bias is explained by, and
preventable by, the temperature observable. Power stability alone is **not** a sufficient warmup signal. The
server warmup gate therefore requires all three observables (power plateau AND
temperature settled AND zero active thermal throttle).

**The throttle bit never fired.** Peak temperatures were 53 to 70 C, comfortably
below the roughly 83 C software thermal threshold, so the throttle observable is
validated as wired, sampled, and correctly reading zero, rather than exercised
under real throttling. Note the 0.5B model was not as thermally mild as feared:
the die still reached 70 C under the 75 req/s load (about 92% of the board's
roughly 250 W power budget), so the thermal ramp, and the divergence, is a
genuine, well-resolved signal here.

---

## Deriving the floor and the timeout

The study grid rounds each measured time up to the next candidate value:
15, 30, 60, 120, 240, 300, 600, 900 s.

**Fixed-mode floor = 300 s.** The floor is the worst cold-start equilibration time
rounded up to the grid:

```
floor = round_up_to_grid( max cold-arm equilibration time )
      = round_up_to_grid( max(183, 233, 161, 252) )
      = round_up_to_grid( 252 )
      = 300 s
```

A 60 s value reaches thermal equilibrium for none of these cells, so if a fixed
warmup duration must actually reach equilibrium for this class of workload, it is
300 s. A 60 s value stays available as an explicit fast choice, documented as a
convenience floor and **not** a thermal-equilibrium claim.

The warm-arm floor is lower:

```
warm floor = round_up_to_grid( max(120, 93, 175, 132) ) = round_up_to_grid(175) = 240 s
```

240 s (warm) is below 300 s (cold): a pre-warmed die needs less warmup. This is
why an adaptive convergence gate is cheaper across a rate sweep than a fixed
cold-sized warmup, since later levels exit fast once the die is already hot.

**Convergence timeout = 900 s.** The timeout is a never-hang failsafe, set to
three times the worst measured equilibration time and clamped to the 900 s
ceiling:

```
timeout = clamp( round_up_to_grid( 3 x 252 ), 300, 900 )
        = clamp( round_up_to_grid( 756 ), 300, 900 )
        = clamp( 900, 300, 900 )
        = 900 s
```

The timeout is a guard, not an operating point: a healthy warmup converges long
before it, and at the timeout the tool proceeds with a loud "timed out"
disclosure rather than hanging or silently passing.

---

## Deviations, disclosed

The study followed its protocol's own rule to disclose every departure:

- **Measured GPU.** The minimum-window-duration study used host GPU 0; here GPU 0
  had a co-tenant workload, so this study measured an isolated idle GPU of the
  identical A100-PCIE-40GB part. All GPUs were logged, and the co-tenant GPU was a
  stable, constant neighbour (temperature standard deviation 1.22 C), so it
  contributed at most a fixed offset to intake air, not a time-varying confound.
- **GPU pinning.** The container-runtime device flags mis-resolved to the wrong
  physical GPU on this host, so the die was pinned unambiguously by its UUID
  through an environment variable instead.
- **Two-shot capture.** The single-shot driver completed the full cold arm, but
  its background task was reaped during the first warm cell while the server
  container survived. The warm arm was re-run reusing that container (no model
  reload, same GPU) after a preload re-established the loaded-hot seed. The two
  capture shards cover disjoint time spans; no cell straddles the split; no
  captured data was affected.
- **An unsupported server flag** (not part of the other study's flag set) was
  removed at the readiness check, before any measurement.
- **Grid ceiling.** The pre-registered grid topped out at 600 s, which would have
  made the timeout arithmetic collapse rather than round up to the intended 900 s
  ceiling, so 900 s was added to the grid. This does not change the floor (252
  rounds up to 300 either way).

---

## Limitations

- **Single engine, model, host, and GPU** (vLLM, Qwen2.5-0.5B-Instruct, one
  A100-PCIE-40GB), the same envelope as the minimum-window-duration study.
- **The throttle observable is unexercised.** Peaks stayed below the software
  thermal threshold, so it is validated as reading zero, not tested under real
  throttling.
- **Temperature is coarse.** It is integer-Celsius NVML telemetry, lightly
  smoothed; the 2 C over 60 s criterion is the pre-registered anchor.
- **The power-settle time absorbs some power creep.** Over a 600 s window the
  die's heating raises power slightly through leakage, shifting the steady
  reference upward, so the measured power-settle time is a conservative (larger)
  estimate rather than the pure batch-fill transient. Temperature still settles
  later than power in every cell, so the divergence conclusion is robust to this.
- **Near-saturation is one point** (75 req/s).

---

## What a follow-up would add

A **thermal-stress arm**: a larger model, more GPUs, or a power-capped GPU that
drives the die to the throttle threshold. That would exercise the throttle
observable under real throttling and measure how much wider the
power-to-temperature divergence gets on genuinely hot hardware, where hotter dies
and longer thermal ramps are expected to widen the gap further.
