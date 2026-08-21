"""Server-mode warmup execution: the convergence-composite gate + fixed opt-out.

This fills the warmup-hook seam the :class:`WindowManager` reserves
(``harness/window_manager.py``): the manager runs a :class:`ServerWarmup` ONCE
before each rate level's measured windows open, re-warming per level (fails
safe; an already-equilibrated later level exits fast). Two modes:

- **composite** (default): warm the server with issuer-driven traffic at the
  target rate, drawn from the MEASURED traffic's shape distribution
  (``WindowContext.spec`` carries ``rate`` / ``arrival``; the traffic source
  itself comes from the injected factory so it reuses the level's request-shape
  source - never a canned-prompt loop), and open the measured window only once
  ALL THREE thermal-equilibrium observables hold together, each computed from the
  SAME ``PowerThermalSampler`` poll and gating INDEPENDENTLY:

  * **power plateau** - the power series is stable-through-end at CoV <= 0.05,
    reusing ``windowing.py``'s ``_detect_steady_state`` (no math changes);
  * **temperature settled** - every monitored GPU's trailing-90s temperature
    range (a 60s settle window held through a 30s confirmation) is below 2C (the
    ``dT/dt`` anchor), which also imposes the ~90s loaded-observation floor;
  * **throttle clear** - no active thermal-throttle bit in the trailing window
    (the per-sample ``thermal_throttle`` flag = the "while active" reading of
    ``ThrottleInfo.thermal.any``).

  The gate is evaluated off the event loop, and the POWER PLATEAU observable reads a
  1 Hz view of the poll rather than the raw series: steady-state detection costs
  super-quadratically in the sample count, so on a series that grows for as long as
  warmup lasts it has to be given a bounded view. The other two observables are
  extreme-value statistics that a decimated series would go blind to, so they read the
  poll at the sampler's full cadence, as does the energy integration (``_gate_view``,
  ``_evaluate_off_loop``). A hard ``timeout_seconds`` failsafe (default 900s) PROCEEDS
  with a loud ``timed_out`` stamp rather than hanging or silently passing.

- **fixed** (explicit opt-out): the same issuer-driven traffic path, no gate, for
  ``duration_seconds`` (default 300s; 0 skips warmup traffic entirely).

There is NO idle cooldown anywhere in this path (the server's loaded
equilibrium IS the measured thermal posture; an idle settle would bias
energy-per-token favourably). Detector MATH is shared with the offline warmup
by reusing ``windowing.py`` - no controller framework.

Result provenance: each invocation records a :class:`ServerWarmupResult` on the
instance (``results``) so the session layer can stamp the outcome and the
per-mode pre-window protocol description into the bundle - the cross-mode
divergence label the server-mode docs render offline-vs-server side by side.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llenergymeasure.harness.windowing import (
    _AUTO_CV_THRESHOLD,
    _AUTO_MIN_WINDOW_SAMPLES,
    _clean_samples,
    _detect_steady_state,
)
from llenergymeasure.utils.exceptions import LLEMError

if TYPE_CHECKING:
    from llenergymeasure.config.models import ServerWarmupConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.harness.traffic import ShapeSource, TrafficSource, Transport
    from llenergymeasure.harness.window_manager import WarmupContext
    from llenergymeasure.serving.transport import RequestShape
    from llenergymeasure.serving.types import ProbeRequest


@runtime_checkable
class _SamplerLike(Protocol):
    """The slice of :class:`PowerThermalSampler` the composite gate drives.

    A Protocol seam (like the window manager's ``_BracketLike``) so a fresh sampler
    is minted per level via the factory and test fakes inject cleanly.
    """

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_samples(self) -> list[PowerThermalSample]: ...


class WarmupTrafficError(LLEMError):
    """The warmup traffic mechanism failed, so no valid warmup was delivered.

    Raised (rather than proceeding) when the warmup traffic task dies before the
    gate/duration finishes, or when warmup ends with zero successfully completed
    requests. This is a DIFFERENT failure class from the composite gate's
    proceed-on-timeout: a
    convergence timeout is disclosed uncertainty (the warmup happened, equilibrium
    is merely unconfirmed), whereas dead traffic means the warmup did not happen at
    all - and the measured window's traffic on the same transport would fail too.
    The hook raising propagates through the window manager's run_level, so the level
    fails loudly BEFORE a measurement window is opened.
    """


class _CountingTransport:
    """Wraps the injected transport to count SUCCESSFUL warmup completions.

    The cheapest completion observation the warmup seam allows (no change to the
    TrafficSource / Transport protocols): a request that returns without raising is
    a completion; a request whose call raises (the issuer catches it as bookkeeping)
    is not counted. Lets the warmup detect a traffic-alive-but-every-request-failing
    server, which the composite gate's NVML observables cannot see.
    """

    def __init__(self, inner: Transport) -> None:
        self._inner = inner
        self.completed = 0

    async def __call__(self, request: RequestShape) -> Any:
        result = await self._inner(request)
        self.completed += 1
        return result


def _traffic_cause(task: asyncio.Task[Any]) -> BaseException | None:
    """The done traffic task's exception, or None for a clean (early) exit."""
    if task.cancelled():
        return None
    return task.exception()


def _traffic_death_message(elapsed: float, cause: BaseException | None) -> str:
    if cause is not None:
        return (
            f"warmup traffic failed at t={elapsed:.1f}s: the traffic source raised "
            "before warmup completed, so no valid warmup was delivered."
        )
    return (
        f"warmup traffic ended at t={elapsed:.1f}s, before warmup completed: the traffic "
        "source exhausted its schedule early, so the warmup was not sustained."
    )


def _zero_completions_message(elapsed: float) -> str:
    return (
        f"warmup delivered zero successfully completed requests in {elapsed:.1f}s: the "
        "warmup traffic never reached the server (check the transport / serving endpoint)."
    )


__all__ = [
    "ObservableState",
    "ServerWarmup",
    "ServerWarmupResult",
    "WarmupTrafficError",
    "build_probe_request",
    "describe_server_warmup_protocol",
]

# --- Composite-gate calibration. Not user-configurable: the
# thresholds are calibrated constants, like windowing.py's k=4 / 0.05. ----------

#: Trailing window over which the temperature "delta" is measured (the dT/dt anchor).
_TEMP_SETTLE_WINDOW_S = 60.0
#: Confirmation the settled state must hold before the gate accepts it.
_TEMP_CONFIRM_WINDOW_S = 30.0
#: Total trailing temperature window (settle + confirm); also the loaded-observation
#: floor the gate structurally imposes (~90s).
_TEMP_TOTAL_WINDOW_S = _TEMP_SETTLE_WINDOW_S + _TEMP_CONFIRM_WINDOW_S
#: Temperature is "settled" when its trailing-window range stays below this (Celsius).
_TEMP_SETTLED_DELTA_C = 2.0
#: Trailing window over which an ACTIVE thermal throttle vetoes window-open.
_THROTTLE_ACTIVE_WINDOW_S = _TEMP_CONFIRM_WINDOW_S
#: Default gate-evaluation cadence.
_GATE_POLL_INTERVAL_S = 2.0
#: Cadence of the DOWNSAMPLED series view the POWER PLATEAU observable is computed over.
#: The throttle and temperature observables, and the energy integration, all keep the
#: sampler's own (100ms) cadence. See :func:`_gate_view` for why the split.
_GATE_VIEW_INTERVAL_S = 1.0


# ---------------------------------------------------------------------------
# Gate observables (each computed from one PowerThermalSampler poll, in the
# sampler's own perf_counter clock; each gates independently).
# ---------------------------------------------------------------------------


def _gate_view(samples: list[PowerThermalSample]) -> list[PowerThermalSample]:
    """Decimate one poll to at most one sample per GPU per ``_GATE_VIEW_INTERVAL_S``.

    Feeds the POWER PLATEAU observable only. The other two observables are extreme-value
    statistics over a trailing window - ``any()`` over the throttle bit, max-minus-min
    over temperature - and a decimated series is systematically blind to a short-lived
    extreme: a 100ms throttle episode survives sub-sampling to 1 Hz about one time in
    ten. They therefore stay at the sampler's full cadence, which costs nothing worth
    counting because both are linear in the sample count (about 2ms together at the
    9000 samples a 900s warmup collects, against ~1.4s for the plateau).

    WHY the plateau observable needs a downsampled view: it runs
    ``windowing._detect_steady_state``, whose cost grows super-quadratically with the
    sample count (it slides a window over the series and, from every candidate onset,
    re-tests every window through to the end). The gate re-evaluates on a series that
    GROWS for as long as warmup lasts, so at the sampler's full cadence a single
    evaluation eventually costs more than the whole poll interval, and its worst case
    is exactly a series that stops being stable at the very end - which is what a
    stalled issuing loop produces. Decimating the view does not make the cost flat; it
    re-bases the same curve on the warmup's failsafe timeout in SECONDS instead of on
    the capture cadence, which is ~3.9s per evaluation at the 900s default. That
    timeout carries no upper bound, so the cost still runs away if it is set far above
    the default (~30s per evaluation at 1800s); the loop stays responsive throughout
    because the evaluation runs off it, but an abandoned worker thread does keep
    burning a core until it finishes.

    WHAT the decimation changes, deliberately, for the plateau observable. Its
    thresholds are duration-based and so survive unchanged: the plateau window is a
    FRACTION of the series, so it spans the same wall-clock stretch at either cadence,
    and sub-sampling a stationary series leaves its coefficient of variation unchanged
    in expectation. Two SAMPLE-COUNT constants in the same pipeline do not survive
    unchanged, and both are accepted:

    * ``windowing._MEDIAN_KERNEL`` = 3 smooths over 3 samples, so its dropout filter
      spans 3s of the view rather than 0.3s of the raw series. It exists to kill
      single-sample NVML transients, and a 1 Hz view has no sub-second transients left
      to kill.
    * ``windowing._AUTO_MIN_WINDOW_SAMPLES`` = 4 makes this observable need 8 samples,
      so its own history floor becomes 8s rather than 0.8s. It is never the binding
      floor: the temperature observable already requires ~90s of history before the
      gate can pass, which is 90 samples of the view, 22x the detector's 4-sample
      minimum.

    The view is also blind to power ripple faster than its own Nyquist frequency, where
    the raw series would read the ripple as instability. That is the acceptable
    direction here: the plateau question is whether the series has settled at a level
    over a trailing window, not whether consecutive samples differ.

    Decimation is per GPU (the plateau pools per GPU), and the NEWEST sample of each GPU
    is always kept, so the view never lags the series it summarises. The MEASUREMENT is
    untouched by all of this - ``_integrate_sampler_energy`` integrates the sampler's
    own full series, not this view.
    """
    by_gpu: dict[int, list[PowerThermalSample]] = {}
    for s in samples:
        by_gpu.setdefault(s.gpu_index, []).append(s)

    view: list[PowerThermalSample] = []
    for gpu_samples in by_gpu.values():
        ordered = sorted(gpu_samples, key=lambda s: s.timestamp)
        last_index = len(ordered) - 1
        kept_at: float | None = None
        for i, s in enumerate(ordered):
            due = kept_at is None or s.timestamp - kept_at >= _GATE_VIEW_INTERVAL_S
            if due or i == last_index:
                view.append(s)
                kept_at = s.timestamp
    view.sort(key=lambda s: s.timestamp)
    return view


def _power_plateau(samples: list[PowerThermalSample]) -> bool:
    """True iff the power series is stable-through-end (CoV <= 0.05).

    Reuses windowing.py's steady-state detector verbatim (no math changes): the
    cleaned power series has a stable-through-end onset iff its tail has plateaued.
    Pooled across GPUs exactly as windowing does (a conservative multi-GPU proxy).
    """
    cleaned = _clean_samples(samples)
    if len(cleaned) < _AUTO_MIN_WINDOW_SAMPLES * 2:
        return False
    # One pass so the power/time series are co-indexed by construction (the shape
    # _detect_steady_state expects), rather than two predicate-matched comprehensions.
    powers: list[float] = []
    times: list[float] = []
    for s in cleaned:
        if s.power_w is not None:
            powers.append(s.power_w)
            times.append(s.timestamp)
    return _detect_steady_state(powers, times) is not None


def _temperature_settled(samples: list[PowerThermalSample]) -> bool:
    """True iff every monitored GPU's trailing-90s temperature range is below 2C.

    The trailing-60s delta held through a +30s confirmation is operationalised
    as the temperature RANGE over the trailing ``_TEMP_TOTAL_WINDOW_S`` staying below
    ``_TEMP_SETTLED_DELTA_C`` - a conservative single-pass reading that also requires
    at least that much loaded history (the ~90s floor). Per GPU (not pooled): a
    cross-GPU temperature spread is not thermal drift.
    """
    by_gpu: dict[int, list[tuple[float, float]]] = {}
    for s in samples:
        if s.temperature_c is not None:
            by_gpu.setdefault(s.gpu_index, []).append((s.timestamp, s.temperature_c))
    if not by_gpu:
        return False
    for series in by_gpu.values():
        series.sort()
        span = series[-1][0] - series[0][0]
        if span < _TEMP_TOTAL_WINDOW_S:
            return False  # not enough loaded history yet (the structural floor)
        window_lo = series[-1][0] - _TEMP_TOTAL_WINDOW_S
        temps = [temp for ts, temp in series if ts >= window_lo]
        if len(temps) < 2 or (max(temps) - min(temps)) >= _TEMP_SETTLED_DELTA_C:
            return False
    return True


def _throttle_clear(samples: list[PowerThermalSample]) -> bool:
    """True iff no thermal throttle is ACTIVE in the trailing window.

    The "while active" reading (section 16): the per-sample ``thermal_throttle``
    flag (NVML thermal-slowdown bits at that instant) over the trailing window,
    rather than the sampler's all-history throttle aggregate - a transient
    cold-start throttle must not veto the gate forever.
    """
    if not samples:
        return False  # no evidence yet - not proven clear
    now = max(s.timestamp for s in samples)
    window_lo = now - _THROTTLE_ACTIVE_WINDOW_S
    return not any(s.thermal_throttle for s in samples if s.timestamp >= window_lo)


@dataclass(frozen=True)
class ObservableState:
    """The three composite-gate observables from one poll; ``all_hold`` is the gate."""

    power_plateau: bool
    temperature_settled: bool
    throttle_clear: bool

    @property
    def all_hold(self) -> bool:
        return self.power_plateau and self.temperature_settled and self.throttle_clear


def _evaluate_observables(samples: list[PowerThermalSample]) -> ObservableState:
    """The three observables from one raw poll; only the plateau reads a decimated view.

    The split is the whole point (see :func:`_gate_view`): the plateau observable is the
    one whose cost forces a bounded view, and the throttle and temperature observables
    are the two that must not lose a short-lived extreme, so they read the poll as
    captured. Applying the view HERE keeps the decision in one place instead of leaving
    each observable to pick a cadence. This runs off the event loop - keep it free of
    anything but the observables.
    """
    return ObservableState(
        power_plateau=_power_plateau(_gate_view(samples)),
        temperature_settled=_temperature_settled(samples),
        throttle_clear=_throttle_clear(samples),
    )


async def _evaluate_off_loop(samples: list[PowerThermalSample]) -> ObservableState:
    """Evaluate the gate in a worker thread, so the issuing loop keeps running.

    The gate's cost is bounded by the decimated view, but "bounded" is not "small
    enough to block the loop that issues the warmup traffic": a stalled loop stops the
    traffic, the GPU falls to idle, and the series tail becomes the detector's worst
    case - a self-reinforcing stall that also starves the failsafe deadline, because
    the deadline is checked by that same loop. Evaluating off the loop keeps traffic
    flowing and keeps the deadline enforceable whatever one evaluation costs.
    """
    return await asyncio.to_thread(_evaluate_observables, samples)


# ---------------------------------------------------------------------------
# Result + protocol description (divergence labeling)
# ---------------------------------------------------------------------------


def _integrate_sampler_energy(samples: list[PowerThermalSample]) -> float | None:
    """GPU energy (J) over a warmup sampler's power series, or None if unformable.

    Reuses the window manager's energy machinery (``windowing._clean_samples`` +
    ``energy.nvml.integrate_power_samples``, summed across GPUs) so the per-level
    warmup energy is NOT a parallel integration path: the same PowerThermal
    sampler the convergence gate already polls also feeds the energy denominator.
    Returns None when fewer than two usable samples were collected.
    """
    if len(samples) < 2:
        return None
    from llenergymeasure.energy.nvml import integrate_power_samples

    cleaned = _clean_samples(samples)
    if len(cleaned) < 2:
        return None
    return sum(integrate_power_samples(cleaned).values())


@dataclass(frozen=True)
class ServerWarmupResult:
    """One level's server-warmup outcome, recorded for result provenance.

    ``converged`` is True when the composite gate was satisfied (composite) or the
    fixed duration completed (fixed). ``timed_out`` is True only when composite hit
    its failsafe and PROCEEDED anyway - a loud disclosure, never a silent pass.
    ``pre_window_protocol`` is the per-mode description the server-mode docs use to
    label the offline-vs-server divergence. ``energy_j`` is the GPU energy measured
    over the warmup phase (from the SAME sampler the gate uses), or None when unmeasured.
    """

    level_index: int
    mode: str
    converged: bool
    timed_out: bool
    elapsed_s: float
    final_observables: ObservableState | None
    pre_window_protocol: str
    energy_j: float | None = None


def describe_server_warmup_protocol(config: ServerWarmupConfig) -> str:
    """Human-readable description of the server pre-window warmup protocol."""
    if config.mode == "fixed":
        return (
            "server fixed-duration warmup: issuer-driven traffic at the target rate for "
            f"{config.duration_seconds:g}s, no convergence gate"
        )
    return (
        "server convergence-composite warmup: issuer-driven traffic at the target rate "
        f"until GPU power plateaus (CoV <= {_AUTO_CV_THRESHOLD:g}), temperature settles "
        f"(trailing-{_TEMP_SETTLE_WINDOW_S:g}s range < {_TEMP_SETTLED_DELTA_C:g}C held "
        f"+{_TEMP_CONFIRM_WINDOW_S:g}s), and no thermal throttle is active; the plateau "
        f"is evaluated on a {1.0 / _GATE_VIEW_INTERVAL_S:g} Hz view of the sampler series "
        "and the other two observables at its full cadence; failsafe timeout "
        f"{config.timeout_seconds:g}s (proceeds with timed_out on expiry)"
    )


# ---------------------------------------------------------------------------
# Readiness-probe request shape (warmup owns the SHAPE; the serving layer owns the mechanics)
# ---------------------------------------------------------------------------


def build_probe_request(
    shape_source: ShapeSource, *, path: str, method: str = "POST", index: int = 0
) -> ProbeRequest:
    """Draw the readiness probe's request SHAPE from the traffic shape distribution.

    Warm (and probe) the path you measure. The probe body is a representative
    request drawn from the SAME shape source the measured traffic uses; ``path`` /
    ``method`` are the engine's serving endpoint (the serving layer's mechanics
    feed this to ``await_ready``). A non-dict payload becomes a bodyless probe.
    """
    from llenergymeasure.serving.types import ProbeRequest

    shape: RequestShape = shape_source(index)
    payload = shape.payload if isinstance(shape.payload, dict) else None
    return ProbeRequest(path=path, payload=payload, method=method)


# ---------------------------------------------------------------------------
# ServerWarmup - the WarmupHook implementation
# ---------------------------------------------------------------------------

#: Builds the warmup TrafficSource for a level from its context (spec.rate /
#: spec.arrival ride in the WarmupContext) and a horizon in seconds. The session
#: layer supplies it, reusing the level's measured request-shape source so warmup draws from the
#: MEASURED traffic distribution.
WarmupTrafficFactory = Callable[["WarmupContext", float], "TrafficSource"]


class ServerWarmup:
    """The server-mode :data:`~llenergymeasure.harness.window_manager.WarmupHook`.

    Constructed by the session layer with the resolved warmup config, a
    warmup-traffic factory, the server transport, and a sampler factory; passed to
    the :class:`WindowManager` as its ``warmup_hook``. The manager awaits it once
    per level; each call records a :class:`ServerWarmupResult` on :attr:`results`.
    """

    def __init__(
        self,
        config: ServerWarmupConfig,
        *,
        traffic_factory: WarmupTrafficFactory,
        transport: Transport,
        sampler_factory: Callable[[], _SamplerLike],
        poll_interval: float = _GATE_POLL_INTERVAL_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._traffic_factory = traffic_factory
        self._transport = transport
        self._sampler_factory = sampler_factory
        self._poll_interval = poll_interval
        self._sleep = sleep
        self._clock = clock
        self.results: list[ServerWarmupResult] = []

    async def __call__(self, context: WarmupContext) -> None:
        """Warm the server for one level; record the outcome for provenance."""
        if self._config.mode == "fixed":
            result = await self._warm_fixed(context)
        else:
            result = await self._warm_composite(context)
        self.results.append(result)

    async def _warm_composite(self, context: WarmupContext) -> ServerWarmupResult:
        """Composite gate: issuer-driven traffic until all three observables hold.

        The gate is evaluated OFF this loop (see :func:`_evaluate_off_loop`) and the
        loop only ever polls the in-flight evaluation for a verdict, so neither the
        warmup traffic nor the failsafe deadline can be starved by an evaluation that
        turns out slow: every iteration reaches the deadline check.

        Fails fast if the traffic dies before the gate resolves (a dead warmup
        mechanism is not disclosed uncertainty) or if warmup delivered zero
        completed requests.
        """
        sampler = self._sampler_factory()
        counting = _CountingTransport(self._transport)
        # Provision the traffic schedule two poll intervals LONGER than the gate
        # window so a clean full-duration run does not exhaust exactly at the
        # boundary (the finally block stops it explicitly, so the slack is free on
        # the happy path).
        source = self._traffic_factory(
            context, self._config.timeout_seconds + 2 * self._poll_interval
        )
        start = self._clock()
        deadline = start + self._config.timeout_seconds
        timed_out = False
        traffic_died = False
        cause: BaseException | None = None
        observables = ObservableState(False, False, False)
        energy_j: float | None = None

        sampler.start()
        traffic_task: asyncio.Task[Any] = asyncio.create_task(source.run(counting))
        # At most ONE gate evaluation is ever in flight: a poll that finds the previous
        # one still running does not queue another (which would pile threads up behind
        # a slow evaluation), it just re-checks the deadline and sleeps.
        eval_task: asyncio.Task[ObservableState] | None = None
        try:
            while True:
                # Watch the traffic each poll. A completion strictly BEFORE the
                # deadline means the warmup mechanism died early (a raise OR a clean
                # early exit); a completion AT OR AFTER the deadline is the schedule
                # ending at the gate boundary, which is a timeout-proceed (a loud
                # timed_out disclosure), NOT traffic death.
                if traffic_task.done():
                    if self._clock() >= deadline:
                        timed_out = True
                        break
                    traffic_died = True
                    cause = _traffic_cause(traffic_task)
                    break
                if eval_task is None:
                    eval_task = asyncio.create_task(_evaluate_off_loop(sampler.get_samples()))
                if eval_task.done():
                    observables = eval_task.result()
                    eval_task = None
                    if observables.all_hold:
                        break
                if self._clock() >= deadline:
                    timed_out = True
                    break
                await self._sleep(self._poll_interval)
        finally:
            # Dropping the wait on an unfinished evaluation abandons its worker thread
            # to run itself out; that is safe only because the gate view bounds what
            # one evaluation can cost.
            if eval_task is not None:
                eval_task.cancel()
                with contextlib.suppress(BaseException):
                    await eval_task
            await self._stop_traffic(traffic_task)
            sampler.stop()
            energy_j = _integrate_sampler_energy(sampler.get_samples())

        elapsed = self._clock() - start
        if traffic_died:
            raise WarmupTrafficError(_traffic_death_message(elapsed, cause)) from cause
        if counting.completed == 0:
            raise WarmupTrafficError(_zero_completions_message(elapsed))

        return ServerWarmupResult(
            level_index=context.level_index,
            mode="composite",
            converged=not timed_out,
            timed_out=timed_out,
            elapsed_s=elapsed,
            final_observables=observables,
            pre_window_protocol=describe_server_warmup_protocol(self._config),
            energy_j=energy_j,
        )

    async def _warm_fixed(self, context: WarmupContext) -> ServerWarmupResult:
        """Fixed opt-out: issuer-driven traffic for a fixed duration, no gate.

        Same fail-fast contract as composite: dead traffic or zero completions is an
        error, not a silent full-duration sleep.
        """
        duration = self._config.duration_seconds
        protocol = describe_server_warmup_protocol(self._config)
        if duration <= 0.0:
            # Explicit skip: no warmup traffic at all (the extreme opt-out).
            return ServerWarmupResult(
                level_index=context.level_index,
                mode="fixed",
                converged=True,
                timed_out=False,
                elapsed_s=0.0,
                final_observables=None,
                pre_window_protocol=protocol,
            )
        counting = _CountingTransport(self._transport)
        source = self._traffic_factory(context, duration)
        start = self._clock()
        traffic_died = False
        cause: BaseException | None = None
        energy_j: float | None = None
        # A sampler runs purely to measure the warmup phase's GPU energy (the fixed
        # mode has no convergence gate, so the sampler is energy-only here); it is
        # the SAME PowerThermal sampler machinery, not a parallel energy path.
        sampler = self._sampler_factory()
        traffic_task: asyncio.Task[Any] = asyncio.create_task(source.run(counting))

        # Race the fixed duration against the traffic: whichever finishes first wins.
        # The sleeper still awaits self._sleep(duration) exactly once, so the duration
        # wait is preserved; a traffic death simply wins the race and short-circuits it.
        # (Wrapped in a coroutine because the injected sleep is a plain Awaitable.)
        async def _await_duration() -> None:
            await self._sleep(duration)

        sleeper: asyncio.Task[None] = asyncio.create_task(_await_duration())
        sampler.start()
        try:
            done, _pending = await asyncio.wait(
                {traffic_task, sleeper}, return_when=asyncio.FIRST_COMPLETED
            )
            if traffic_task in done:
                traffic_died = True
                cause = _traffic_cause(traffic_task)
        finally:
            sleeper.cancel()
            with contextlib.suppress(BaseException):
                await sleeper
            await self._stop_traffic(traffic_task)
            sampler.stop()
            energy_j = _integrate_sampler_energy(sampler.get_samples())

        elapsed = self._clock() - start
        if traffic_died:
            raise WarmupTrafficError(_traffic_death_message(elapsed, cause)) from cause
        if counting.completed == 0:
            raise WarmupTrafficError(_zero_completions_message(elapsed))

        return ServerWarmupResult(
            level_index=context.level_index,
            mode="fixed",
            converged=True,
            timed_out=False,
            elapsed_s=elapsed,
            final_observables=None,
            pre_window_protocol=protocol,
            energy_j=energy_j,
        )

    @staticmethod
    async def _stop_traffic(traffic_task: asyncio.Task[Any]) -> None:
        """Cancel the warmup traffic and await its unwind (mirrors the manager)."""
        traffic_task.cancel()
        with contextlib.suppress(BaseException):
            await traffic_task
