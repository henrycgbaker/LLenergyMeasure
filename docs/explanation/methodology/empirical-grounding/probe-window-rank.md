---
slug: /methodology/empirical-grounding/probe-window-rank
title: Probe windows preserve configuration rankings
description: A pre-registered study of how short a measurement window can be while still ranking serving configurations by energy-per-token the same way the full-length window does. Grounds the 15 s probe window adopted for the optimiser's triage tier.
---

# Probe windows preserve configuration rankings

The optimiser's triage tier has to compare many candidate serving
configurations quickly, so it measures each with a short **probe** rather than
the full measurement protocol. A probe's job is not an accurate absolute
number - probe results are marked by tier and never reported as
measurements - its job is to **rank** configurations well enough that the
survivors it promotes to full-protocol measurement are the right ones. That
makes the probe window length a rank-fidelity question: how short can the
window get before the ranking it produces stops agreeing with the ranking the
full 240 s window produces?

> Traceability: this study was pre-registered, collected 2026-08-20 to
> 2026-08-25, and its verdict was ratified 2026-08-25. The values below are
> its banked results. One registered quantity was amended before analysis,
> disclosed under [Registration amendment](#registration-amendment).

---

## Question

At what abbreviated window length does the rank ordering of serving
configurations by energy-per-token stop agreeing with the full-protocol
ordering?

## Method

**A configuration grid built to be hard to rank.** 11 vLLM serving
configurations spanning the axes the optimiser searches: a vendor-defaults
baseline; the four corners plus a midpoint of the batching plane
(`max_num_batched_tokens` x `max_num_seqs`); two `gpu_memory_utilization`
levels; a prefix-caching flip; eager mode (CUDA graphs off); and a combined
stressor that moves several axes at once. Every cell serves the same model
(Qwen2.5-0.5B-Instruct) at the same offered rate (15 req/s, open-loop Poisson
arrivals, seeded), with the same prompt pool and a 256-token output budget, on
one A100-PCIE-40GB.

**Full protocol first, slices second.** Each cell was measured 3 times under
the full protocol - one server launch per visit, convergence-gated warmup, a
30 s ramp exclusion, then three contiguous 240 s windows - with visit order
interleaved across cells in seeded random rounds, never a sequential block per
cell. Each candidate probe window (15, 30, 60, 120 s) is then computed
**offline as a prefix slice** of each measured 240 s window: the power series
and the per-token receipt timestamps were captured continuously on one clock,
so the first `w` seconds of a window is itself a valid `w`-second window. That
design holds everything except window length constant - same launch, same
warmup, same rate, same requests - so window length is the only factor.

**Estimator, fixed before data.** A window's energy is the trapezoidal
integral of the cleaned power series over the window span; its token count is
the number of client-side output-token receipts inside the span. A visit's
estimate pools energy over pooled tokens across its three windows (never a
mean of ratios); a cell's value is the mean of its three visits; ties in the
ordering break by cell index.

**Decision rule, fixed before data.** For each candidate window: Spearman rank
correlation and top-3 overlap between the candidate's cell ordering and the
240 s reference ordering. The chosen default is the shortest window with
rho >= 0.7 AND top-3 overlap >= 2 that also holds both at every longer
candidate (monotone-stable). If no candidate at or below 120 s clears both,
the full window stays the default and the triage tier's economy collapses to
its fallback.

## Registration amendment

The grid was registered at 12 cells, including a tensor-parallel-2 cell. That
cell could not run: at the pinned vLLM version the tensor-parallel worker
rendezvous hangs inside server containers, and engine versions are frozen for
the release cycle, so the cell's schedule slots were skipped by the driver's
hole mechanism (which preserves every other cell's seeded order). The top-k
denominator was amended from 3-of-12 to 3-of-11 by an explicit ratified
ruling **before** any analysis ran - the analysis code asserts the registered
grid size and refused to emit a verdict until the amendment was recorded. No
threshold, estimator, replicate count, or stability requirement changed.
The consequence is stated under [Envelope](#envelope-and-limits):
tensor-parallel rank behaviour under short windows is unmeasured.

## Results

| window | Spearman rho vs 240 s | top-3 overlap | passes |
| --- | ---: | ---: | :---: |
| **15 s** | 0.818 | 2 of 3 | yes |
| 30 s | 0.845 | 2 of 3 | yes |
| 60 s | 0.927 | 2 of 3 | yes |
| 120 s | 0.927 | 2 of 3 | yes |

Every candidate passes and the agreement strengthens monotonically with
length, so the monotone-stability requirement is met at the shortest
candidate: **the chosen probe window is 15 s**.

The ordering being preserved is a resolvable one, not noise re-shuffled: the
across-cell spread of energy-per-token (coefficient of variation 0.279,
max/min ratio 2.81x) is 87x the median within-cell replicate noise (0.0032).
The structure behind the numbers: the two eager-mode cells are far more
efficient per token than everything else (about 2.7x) and hold ranks 1-2 at
**every** window length; the one top-3 substitution under short windows comes
from a statistically flat mid-field where adjacent cells differ by less than
one replicate spread - exactly the regime the overlap floor of 2 was
registered to tolerate.

## What this grounds

| Grounded default | Value |
| --- | --- |
| Probe window length for the optimiser's triage tier (`optimize.probe.window_seconds`; the tier ships in a later release, and this value ships with it) | 15 s |

Probe results remain tier-marked and are never reported as measurements; this
study licenses the ranking they feed, not their absolute values.

## Envelope and limits

- **One operating point.** vLLM at the pinned engine version,
  Qwen2.5-0.5B-Instruct, one A100-PCIE-40GB, 15 req/s. Re-confirm on other
  engines, models, hosts, and rates before leaning on the ranking elsewhere.
- **Tensor parallelism is unmeasured.** The amended grid carries no
  tensor-parallel cell (see the amendment above), and it is the configuration
  whose power profile most plausibly differs in shape from the rest.
- **Prefix slices, not live short windows.** The abbreviated windows share
  their visit's full-length warmup and launch. This study therefore answers
  "is a shorter window enough?" in isolation; the abbreviated protocol as the
  triage tier actually runs it - short warmup as well as short windows - is a
  separate claim, validated by its own pre-registered companion study before
  the tier ships.
- **Warmup cost is untouched.** The window is the only thing shortened here;
  a probe still pays the launch and warmup floor, which is why the optimiser's
  answer to a large search space is fewer candidates, not ever-cheaper probes.
