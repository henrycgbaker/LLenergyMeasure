---
title: Server-measurement architecture
description: How the traffic source, server session, and window manager add online-serving measurement without re-keying the offline sweep loop.
---

# Server-measurement architecture

Offline measurement is one engine lifetime that produces one measured window
over a fixed prompt batch. Server-mode measurement is one server lifetime that
produces many windows, driven by traffic that arrives over time. This page
explains the three seams that add server mode without re-keying the offline
sweep loop or duplicating the energy-window mechanics: the **traffic source**,
the **server session**, and the **window manager**.

For the measurement model these seams implement (windows, warmup gate, the
comparability caveats), see
[Methodology: server-mode measurement](/explanation/methodology/methodology#server-mode-measurement).

## The problem

The offline session seam already separates a session's *lifetime*
(acquire -> produce -> release) from *result production*, and an offline session
deliberately produces exactly one result per lifetime. Server mode needs the
same lifetime discipline but a different result cardinality: one launched
server, held alive across a rate sweep, producing one result per measurement
window. It also needs a load driver the offline path never had, and it must
keep the energy-window mechanics the harness already owns rather than
reimplementing them.

Three seams answer this. Each is a narrow interface so the online-serving parts
plug in beside the offline parts instead of rewriting them.

## The TrafficSource seam

`TrafficSource` is a `typing.Protocol` in
[`src/llenergymeasure/harness/traffic.py:247`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/harness/traffic.py#L247).
It has a single method - `run(transport, *, drain_timeout=None) -> IssuerReport` -
and it is the one window-manager-facing surface for driving online load. It is
a deliberate plugin point: a load-generator-backed source can arrive later
behind the same interface without touching the window manager.

The shipped implementation is `OpenLoopPoissonSource`
([`traffic.py:263`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/harness/traffic.py#L263)).
It builds a deterministic arrival schedule from `(rate, arrival, burstiness,
seed)` and issues each request on schedule as a detached task, without awaiting
the transport call. That is the open-loop guarantee, and it is why a stalled
transport cannot inflate the measured issuance duration or stall the schedule:
the issue loop only sleeps to the next scheduled offset, and a bounded drain
(`drain_timeout`) cancels whatever is still pending. The transport itself is a
companion `Transport` protocol
([`traffic.py:84`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/harness/traffic.py#L84)),
injected and never owned by the traffic module.

Per-request bookkeeping lives on `RequestRecord`
([`traffic.py:96`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/harness/traffic.py#L96)):
`issued_at` (the ideal scheduled time, the latency anchor), `dispatched_at`
(when the request actually left for the transport), and `completed_at` (when it
reached a terminal state). The gap between issued and dispatched is what a
concurrency cap or a slow transport shows up as - it is recorded, not hidden.

## The ServerSession

`ServerSession`
([`src/llenergymeasure/study/server_session.py:642`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L642))
is the one-dispatch, N-results session type - the sibling the offline session
seam was built to admit. It duck-types the same `ExperimentSession` protocol
the offline `SubprocessSession` and `DockerSession` implement, but its single
lifetime produces many results:

- **`__enter__`**
  ([`server_session.py:742`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L742))
  launches the engine server and waits for readiness (the launch and the model
  load ride inside one instrumented phase). A failure during acquisition tears
  down whatever was acquired and re-raises, so a failed launch never leaks.
- **`run()`**
  ([`server_session.py:790`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L790))
  warms the server before each rate level and drives the window manager,
  producing one per-window result for every window of every level.
- **`__exit__`**
  ([`server_session.py:835`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L835))
  drains and finalises the bundles, then tears the server down. It is
  idempotent and swallows cleanup faults with a loud warning, so a fault during
  teardown never converts a completed measurement into a failure.

### Session grouping

A rate sweep does not have to relaunch the server per rate. `partition_server_groups`
([`server_session.py:1935`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L1935))
folds a run of consecutive server cells that are identical except
`server.traffic.rate` **and** belong to the same cycle into one session - one
launch driving each rate as its own level. A non-server cell, any non-rate
difference, or a cycle-number change ends the group. This is why experiment
order changes launch economics: under `interleave` a sweep pass folds into one
launch, while under the default `sequential` order with more than one cycle each
cell's cycles are adjacent, so the sweep dispatches singleton sessions (one
launch per cell per cycle).

Each window's identity is minted by `_window_experiment_id`
([`server_session.py:1506`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L1506))
as `server-{config_hash}-c{cycle}-L{level_index}-W{window_index}`. The cycle
component keeps cycle 1 and cycle 2 of one grid point distinct so their bundles
do not collide once a reader keys on `experiment_id`.

## The window manager

The window manager owns the per-window mechanics. A window is described by a
`WindowSpec`
([`src/llenergymeasure/harness/window_manager.py:125`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/harness/window_manager.py#L125)) -
rate, duration, ramp exclusion, and attribution policy. For each level the
manager excludes the ramp once and prospectively, then opens and closes a fixed
number of contiguous windows, bracketing each with a fresh measurement bracket
so the energy-window mechanics are reused unchanged from the offline path.

Validity is decided per level, not per window, by `validate_level_stability`
([`window_manager.py:617`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/harness/window_manager.py#L617)).
Each window's energy per token is its integrated GPU energy over the tokens
attributed to the span; a level passes only when the coefficient of variation
of energy per token across the consecutive windows is stable through the end of
the level (below the same auto threshold the offline steady-state detector
uses). A window with zero attributed tokens makes the diagnostic unformable and
fails the level. A failing level is stamped invalid with a reason, never
silently dropped.

## Interrupt semantics

On SIGINT mid-session the interrupt watcher cancels the task driving the current
level, and the session preserves whatever level state exists before re-raising
so `__exit__` reaps the server. Crediting is conservative
([`server_session.py:1244`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/study/server_session.py#L1244)):
only cleanly-closed levels - those whose validation resolved and whose bundles
are finalised - are credited as completed, so a resume does not re-run them.
In-flight, aborted, and unreached cells stay marked running for the sweep loop's
interrupt downgrade; their work is not credited. If a grouped session is fully
invalid (a warmup abort, or no valid window), it is counted as its full cell
count of failures rather than one.

## The server lifecycle underneath

The session composes the container and process primitives in
[`src/llenergymeasure/serving/lifecycle.py`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/serving/lifecycle.py).
Launch allocates a free loopback port and runs the image detached; readiness
(`await_ready`,
[`lifecycle.py:224`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/serving/lifecycle.py#L224))
runs a liveness health poll and then a real inference request through the
serving path, and is satisfied only when that request returns HTTP 200; shutdown
([`lifecycle.py:437`](https://github.com/henrycgbaker/llenergymeasure/blob/main/src/llenergymeasure/serving/lifecycle.py#L437))
stops and force-removes the container so nothing leaks even if the stop failed.

## Why this matters

The offline and server paths share the harness's energy-window mechanics and
the study runner's session seam. Adding online-serving measurement did not mean
a second measurement stack: it meant one load-driver protocol, one N-results
session beside the one-result offline sessions, and a window manager that reuses
the same measurement bracket. A load-generator-backed traffic source can later
replace `OpenLoopPoissonSource` behind the `TrafficSource` protocol without any
of the session or window logic changing.
