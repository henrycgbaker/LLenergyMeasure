---
slug: /methodology/empirical-grounding/open-loop-arrivals
title: Open-loop Poisson arrivals
description: Why server-mode load is generated open-loop, the measured arrival-process contract, and the conformance test that enforces it continuously in CI.
---

# Open-loop Poisson arrivals

Server-mode measurement drives an inference server with generated request
traffic. How that traffic is generated is a measurement choice with real
consequences, so it is grounded and continuously checked.

---

## Why open-loop

There are two ways to generate load. A **closed-loop** generator issues a new
request only after a previous one finishes, so it keeps a fixed number of
requests in flight. An **open-loop** generator issues requests on a schedule
fixed in advance, independent of how fast the server is keeping up.

Closed-loop generation has a measurement flaw. When the server slows down, a
closed-loop generator automatically slows down with it (it is waiting for
in-flight requests to complete before sending more). The requests that would have
piled up and exposed the slowdown are simply never sent, so the latency
percentiles look better than the server actually behaved. This failure mode is
known as **coordinated omission**: the load generator quietly omits exactly the
samples that would reveal the tail.

LLenergyMeasure generates load **open-loop**. The issuance schedule is computed up
front from the arrival process and the seed, and requests fire on that schedule
regardless of how the server progresses. A stalled or capped server never slows
issuance, so queueing time counts against the system under test instead of
disappearing from the measurement.

---

## Latency anchoring

Because issuance does not wait for the server, each request records three
timestamps: its **ideal scheduled time** (the latency anchor), the time it was
actually handed to the transport, and the time it completed. Latency is measured
from the ideal scheduled time, so if the server falls behind, the time a request
spends waiting to be dispatched is counted as latency rather than hidden. This
follows the schedule-anchored convention established by the MLPerf load
generator.

An optional concurrency cap can limit how many requests are in flight at once. It
gates dispatch without ever touching the schedule, and if a cap is materially
binding, that fact is disclosed in the run's report rather than silently
absorbed.

---

## The arrival-process contract

The default arrival process is **Poisson**: inter-arrival gaps are drawn from an
exponential (memoryless) distribution, whose coefficient of variation is exactly
1. A Poisson process is the standard model for independent request arrivals and
is what makes the offered rate meaningful.

The contract the tool guarantees:

- **Poisson arrivals have a coefficient of variation of about 1** across the
  supported rate span of 1 to 100 req/s.
- **The mean inter-arrival gap is 1 divided by the rate**, so the schedule
  realises the offered rate.
- **The schedule is deterministic under its seed**: the same rate, arrival
  process, and seed produce a byte-identical schedule, so runs are reproducible.

For workloads that are burstier or smoother than Poisson, a **gamma** arrival
process is available; its coefficient of variation tracks a configurable
burstiness parameter (equal to 1 reproduces Poisson, greater than 1 is burstier,
less than 1 is smoother).

The two measurement studies on this site (the
[minimum-window-duration](/explanation/methodology/empirical-grounding/min-window-duration)
and [loaded thermal-equilibrium](/explanation/methodology/empirical-grounding/loaded-thermal-equilibrium)
studies) both drive load through this same open-loop Poisson process, so their
results are stated for a well-defined arrival shape.

---

## Continuously enforced

The arrival-process contract is not a one-time claim. A **conformance test**
asserts it on every change in CI: it builds schedules at rates spanning 1 to
100 req/s and checks that the Poisson coefficient of variation stays within a
tight band of 1, that the mean gap matches 1 divided by the rate, and that the
gamma process tracks its burstiness parameter. Because the test runs
continuously, the guarantee holds for the code as it ships, not just for the code
as it was first written.
