---
title: Server-mode serving study
description: Measure energy and latency under traffic-driven serving windows across a request-rate sweep.
---

# Server-mode serving study

Offline mode measures a batch ceiling: hand an engine a fixed prompt set,
let it run flat out, and read off the best-case energy per token. Server
mode asks a different question - *what does inference cost when requests
arrive over time and the engine has to keep up?* It drives a running
server with an open-loop arrival process, opens measurement windows once
the server reaches steady state, and reports energy and latency at each
request rate you sweep.

This tutorial builds a rate sweep on vLLM: the same model and prompts as
an offline study, but measured as a live service at several arrival rates.
By the end you will have one per-window result bundle per measurement
window and know how to read the request log behind it.

The framing matters. Offline answers "how cheap can this get?"; server
mode answers "how does cost move as load rises, and where does latency
break?" The two are complementary, and they share an energy-per-token
definition so you can put them side by side (with one caveat about the
pre-window protocol - see [Step 6](#server-mode-rules)).

> **Compute time:** longer than the offline tutorials. Each rate level
> warms the server to thermal equilibrium, then measures three
> consecutive windows at the default 240 s each, so budget roughly
> 20-30 minutes per rate level on a single A100-class GPU once the vLLM
> image is pulled. Shorten `server.traffic.window_seconds` for a quick
> smoke test (at the cost of measurement quality).

## Prerequisites

- A working install of `llenergymeasure` - see
  [How to install](/how-to/install).
- Docker + NVIDIA Container Toolkit operational and `llem doctor`
  passing - see [Docker setup](/how-to/docker-setup).
- The vLLM engine image built locally or pullable from GHCR - see
  [Run with Docker + vLLM](/how-to/run-with-docker-vllm).
- You've completed the
  [multi-engine study tutorial](/tutorials/multi-engine-study) so the
  shape of a study YAML, `llem run --dry-run`, and a result bundle are
  familiar.

Server mode at v0.7 runs on **vLLM** and **TensorRT-LLM**. Transformers
in `serving_mode: server` is rejected at config validation (see
[Step 6](#server-mode-rules)); this tutorial uses vLLM.

## Step 1 - What server mode measures

An offline experiment is one engine lifetime that produces one measured
window over a fixed prompt batch. A server experiment is one server
lifetime that produces *many* windows - one measurement window per
request rate - driven by traffic that arrives over time.

The unit of measurement is the **window**: a fixed span during which the
server is held at a target arrival rate and steady state, and over which
energy and latency are attributed. A rate sweep produces one *level* per
rate, and each level measures several windows so a stability gate can
confirm the numbers settled.

Two things move from offline to server mode:

| Axis | Offline | Server |
|------|---------|--------|
| Workload driver | fixed prompt batch, run flat out | open-loop arrival process at a target rate |
| Measurement unit | one window per experiment | one window per rate level (several per level) |
| Latency | not the headline | TTFT / ITL percentiles under load, plus SLO attainment |
| Pre-window wait | idle thermal-floor wait | loaded warmup to equilibrium (no idle wait) |

The shared energy-per-token, tokens-per-second, and gross/net energy
definitions are identical across both modes, so cross-mode comparison is
legitimate for those figures. The [methodology page](/explanation/methodology/methodology#server-mode-measurement)
covers the measurement model in depth.

## Step 2 - Write the study config

Here is a complete, runnable server-mode study. Save it as
`server-study.yaml`.

```yaml
study_name: tutorial-server-mode

serving_mode: server

engine: vllm            # single serving engine for this study

runners:
  vllm: container

study_execution:
  n_cycles: 1
  experiment_order: interleave

task:
  model: Qwen/Qwen2.5-0.5B
  random_seed: 42
  dataset:
    source: aienergyscore
    n_prompts: 200
    order: interleaved
  max_input_tokens: 256
  max_output_tokens: 256

measurement:
  energy_sampler: auto
  baseline:
    enabled: true
    duration_seconds: 30.0

server:
  traffic:
    rate: 2                # base placeholder; the sweep below sets it per cell
    arrival: poisson       # memoryless arrivals (CV=1)
    # window_seconds unset -> defaults to 240s
    # ramp_exclusion_seconds unset -> defaults to 30s
  warmup:
    mode: composite        # wait for the thermal-equilibrium gate (default)

sweep:
  server.traffic.rate: [2, 4, 8]

output:
  results_dir: ./results
  save_timeseries: true
```

Walking through the server-specific parts:

`serving_mode: server` is required and selects the online-serving
measurement path. It is a conditioning identity axis, so a server cell
and an offline cell never deduplicate together.

`server.traffic` is the arrival spec. `rate` is requests per second;
`arrival: poisson` gives memoryless arrivals (set `arrival: gamma` with a
`burstiness` value for bursty traffic). Leaving `window_seconds` unset
takes the 240 s default measured-span duration, and leaving
`ramp_exclusion_seconds` unset takes the 30 s default ramp that is
excluded from the span. Both defaults are grounded in the E2
minimum-window-duration study.

`server.warmup.mode: composite` is the default: the harness warms the
server with traffic at the target rate and opens the window only once the
GPU reaches thermal equilibrium (with a 900 s failsafe). `mode: fixed`
warms for a fixed `duration_seconds` (default 300 s) instead. See the
[methodology page](/explanation/methodology/methodology#server-warmup) for
the gate semantics.

`sweep: server.traffic.rate: [2, 4, 8]` is an ordinary study-level list
axis: it expands to three independent measurement cells, one per rate,
each with a distinct config hash. The base `rate: 2` is a placeholder the
sweep overrides.

The full field reference is on the
[study config reference](/reference/study-config#server-mode-server) page
(server section, warmup, traffic, and SLO tables).

## Step 3 - Dry-run and launch

Resolve the sweep before running anything:

```bash
llem run server-study.yaml --dry-run
```

You should see three resolved cells, one per rate. Because this study
uses `experiment_order: interleave` and `n_cycles: 1`, the three cells
fold into a single server launch that is reused across the sweep - the
loader logs an INFO hint if you pick an order that would launch a fresh
server per cell instead (see [Step 6](#server-mode-rules)).

When the counts match your expectation, launch the run:

**CLI**

```bash
llem run server-study.yaml
```

**Python**

```python
from llenergymeasure import run_study

study_result = run_study("server-study.yaml")
print(f"Completed {study_result.summary.completed} cells")
```

## Step 4 - What happens at run time

A server *session* is one server lifetime; under this tutorial's
`interleave` order one session serves all three rate cells as levels over
a single launch. Inside a session the harness runs a fixed sequence:

1. **Launch.** The engine server starts in a container: the harness
   allocates a free loopback port and runs the image detached (no
   `--rm`, so a crash-on-startup container survives for `docker logs`).
2. **Readiness probe.** A liveness poll of the server's `/health`
   endpoint is necessary but never sufficient; readiness is only
   satisfied once a *real* inference request driven through the serving
   path returns HTTP 200.
3. **Warmup.** The harness drives issuer traffic at the level's target
   rate, drawn from the same request-shape distribution the measurement
   will use. In composite mode it opens the window only once all three
   equilibrium observables hold at once - GPU power plateaued,
   temperature settled, and no active thermal throttle - with a 900 s
   failsafe that proceeds (stamping `timed_out`) rather than hanging.
   Warmup re-runs before every rate level.
4. **Ramp exclusion.** When load begins for a level, the first 30 s
   (default) is excluded from the measured span prospectively - it is
   never trimmed retroactively.
5. **Measurement windows.** The harness holds the rate and measures
   three consecutive windows of 240 s each (defaults). A per-level
   stability gate checks that per-window energy per token is stable
   across the windows.
6. **Drain.** After the windows close, in-flight requests are drained to
   completion so latency percentiles include their real end-to-end
   times; energy accounting never extends into the drain.
7. **Teardown.** The server is shut down (`docker stop` then a
   force-remove) so nothing leaks, even on the error path.

A rate sweep repeats steps 4-6 per level within the one launched server
(under `interleave`), with an optional inter-level cooldown
(`server.cooldown_seconds`, default 0).

## Step 5 - Inspect the per-window results

A server cell produces **one result bundle per measurement window**. The
study directory looks like this:

```text
results/tutorial-server-mode_2026-05-07T14-32-08/
├── manifest.json
├── 001_c0_Qwen2.5-0.5B-vllm_<hashA>/          # rate 2, window 0
│   ├── result.json                            # window metrics (serving_mode: server)
│   ├── config.json                            # resolved config + provenance
│   ├── system.json                            # environment + session facts
│   └── requests.parquet                       # per-window request log
├── 001_c0_Qwen2.5-0.5B-vllm_<hashA>_1/        # rate 2, window 1 (collision suffix)
├── 001_c0_Qwen2.5-0.5B-vllm_<hashA>_2/        # rate 2, window 2
├── 001_c0_Qwen2.5-0.5B-vllm_<hashB>/          # rate 4, window 0
├── 001_c0_Qwen2.5-0.5B-vllm_<hashB>_1/        # rate 4, window 1
├── 001_c0_Qwen2.5-0.5B-vllm_<hashC>/          # rate 8, window 0
└── ...                                        # nine bundles total: three rates x three windows
```

All nine bundles share the `001_c0_Qwen2.5-0.5B-vllm_` prefix, because this
study interleaves a single server launch across the sweep: the config hash
differs per rate, and the second and third window of each rate take a `_1` /
`_2` collision suffix.

Each window's `result.json` carries `serving_mode: "server"` and the
shared energy metrics; server-distinct metrics (TTFT/ITL percentiles,
goodput, SLO attainment) live in the derived server-metrics block, and
mode-inapplicable fields (`input_tokens`, `total_tokens`) are `null`
because server mode counts client-side streamed output tokens as the
canonical denominator. The `requests.parquet` sidecar is the raw
per-request log behind those metrics - one row per issued request, with
its own physical facts (issue time, first-token time, terminal status,
client-counted output tokens). See the
[results schema reference](/reference/results-schema#server-mode-results)
for the full column set and how the window results roll up into the study
manifest.

A quick read of one window's request log with DuckDB:

```python
import duckdb

rows = duckdb.sql("""
    SELECT status, count(*) AS n, median(ttft_ms) AS p50_ttft_ms
    FROM 'results/tutorial-server-mode_2026-05-07T14-32-08/*/requests.parquet'
    WHERE in_measurement_window AND NOT is_ramp
    GROUP BY status
""").df()
print(rows)
```

## Step 6 - Rules to know at v0.7 {#server-mode-rules}

Server mode is new in v0.7, and a few staging restrictions apply.

- **One `serving_mode` per study.** A single study must be all offline or
  all server; a mixed study is rejected at config load. Mixed-mode
  studies arrive in a later release. Run one study per mode for now.
- **A rate list is a sweep axis.** `server.traffic.rate: [2, 4, 8]`
  expands to independent measurement cells with distinct config hashes -
  it is not a single multi-rate run.
- **Launch economics depend on experiment order.** Consecutive
  same-cycle cells that differ only by rate fold into one server launch.
  Under `interleave` (used here) each sweep pass reuses a single launch.
  Under the default `sequential` order with `n_cycles > 1`, each cell's
  cycles are adjacent, so the sweep launches a fresh server per cell per
  cycle; the loader emits an INFO hint suggesting `interleave` when it
  detects this.
- **Transformers server mode is rejected.** At v0.7 the server engines
  are vLLM and TensorRT-LLM. `serving_mode: server` with
  `engine: transformers` fails at config validation: the pinned
  `transformers serve` is positioned for evaluation and moderate load,
  not the sustained load a measurement harness drives. A transformers
  server adapter is a tracked fast-follow
  ([issue #896](https://github.com/henrycgbaker/llenergymeasure/issues/896)).
- **Windows are duration-grounded.** `server.traffic.window_requests`
  (count-bound windows) is rejected at v0.7; use `window_seconds`.

## What you've learned and where to go next

You've now run the server-mode workflow end to end:

- **Framed** a serving measurement as a rate sweep over traffic-driven
  windows rather than a batch ceiling.
- **Written** a server-mode study with an open-loop arrival spec and a
  composite warmup gate.
- **Understood** the run-time sequence: launch, real-probe readiness,
  loaded warmup, ramp exclusion, measurement windows, drain, teardown.
- **Inspected** the per-window result bundles and the request log behind
  them.

### Sister recipes (How-to)

- [Run with Docker + vLLM](/how-to/run-with-docker-vllm) - the single-engine Docker recipe this tutorial builds on.
- [Interpret results](/how-to/interpret-results) - field-by-field walkthrough of a result bundle.

### Reference

- [Study config](/reference/study-config#server-mode-server) - full server / traffic / warmup / SLO field listing.
- [Results schema](/reference/results-schema#server-mode-results) - the per-window bundle and `requests.parquet` columns.

### Conceptual depth (Explanation)

- [Methodology - Server-mode measurement](/explanation/methodology/methodology#server-mode-measurement) - windows, warmup gate, and the comparability caveats.
- [Server measurement architecture](/explanation/architecture/server-measurement) - the traffic source, server session, and window manager seams.
