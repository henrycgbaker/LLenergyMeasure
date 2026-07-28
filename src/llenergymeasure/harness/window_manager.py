"""First-class measurement window + multi-level window manager for server mode.

This module owns server mode's measured-window abstraction: the :class:`WindowSpec`
(the first-class window object), the event-driven delineation that brackets one
window's energy, the dual boundary policies that keep energy and latency
accounting distinct, per-level steady-state validation, and the multi-level
orchestration that drives a rate sweep.

Design anchors (server-mode plan, section 14):

- **D7 - first-class window object + two coexisting boundary policies.** A window
  is ``{rate, duration_or_count, attribution_policy, ramp_exclusion}``. The ENERGY
  policy amortizes steady-state energy over the measured span ``[span_start,
  span_end]`` (denominator = client-counted tokens RECEIVED in that span). The
  LATENCY policy is drain-before-close: every request ISSUED in the measured span
  gets a full latency record, followed to completion PAST ``span_end`` - but energy
  accounting never extends into the drain. The two policies are never conflated
  into one number.
- **D19 - event-driven delineation.** The manager emits explicit
  start-window / stop-window events to a :class:`WindowEnergySink`; the sink (by
  default a :class:`MeasurementBracket`-backed one) opens and closes the energy
  measurement in response. No component infers window membership from timestamps
  alone - the events are the protocol, a drop-in for a future remote sampling
  agent.
- **C2 / C5 - reuse.** The default sink reuses
  :class:`~llenergymeasure.harness.bracket.MeasurementBracket` (a bracket brackets a
  WINDOW, not a Python call). Per-level stability reuses windowing.py's CV /
  stable-through-end machinery unchanged; only the multi-level orchestration,
  prospective ramp exclusion, and 3-consecutive-window validation are new.
- **E2 - ratified numeric defaults.** Measured span 240s, ramp exclusion 30s
  absolute, per-level stability CoV <= 0.05 sustained over 3 consecutive
  sub-windows (mirroring ``windowing.py``'s threshold). The duration / ramp
  defaults are config-exposed under ``server.traffic``; this module carries them as
  the :class:`WindowSpec` dataclass defaults.

Out of scope (later slices): warmup EXECUTION behind the warmup-hook seam (SM8),
server session lifecycle / persistence (SM9 / SM10), and metrics derivation
(J/token, percentiles, goodput) beyond this module's own bookkeeping (SM12).
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from llenergymeasure.config.models import (
    DEFAULT_RAMP_EXCLUSION_SECONDS,
    DEFAULT_WINDOW_SECONDS,
)

# REUSE BINDING (server-mode plan section 14, maintainer-confirmed): the CV /
# steady-state detector is consumed from windowing.py, never reimplemented. These
# live in the same package (Layer 3), so import altitude does not require an
# extraction into a shared home - a direct import is the minimal faithful reuse.
# NO math changes.
from llenergymeasure.harness.windowing import (
    _AUTO_CV_THRESHOLD,
    _coefficient_of_variation,
    _is_stable_through_end,
)

if TYPE_CHECKING:
    from llenergymeasure.config.models import MeasurementConfig, TrafficConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.domain.progress import ProgressCallback
    from llenergymeasure.harness.bracket import MeasuredWindowCore
    from llenergymeasure.harness.traffic import (
        IssuerReport,
        RequestRecord,
        TrafficSource,
        Transport,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The single attribution policy v0.7 ships (maintainer ratification 2026-07-29):
#: steady-state span amortization - the J/token denominator is client-counted
#: tokens RECEIVED within the measured span. Disclosed as a result field, not a
#: configurable knob; requests.parquet (SM11) keeps per-request timestamps so
#: alternative attributions stay re-derivable offline.
ATTRIBUTION_STEADY_STATE_SPAN = "steady_state_span"

#: Consecutive-window agreement count for per-level stability (E2 ratified;
#: perf_analyzer's "3 consecutive windows within tolerance").
STABILITY_CONSECUTIVE_WINDOWS = 3

#: Number of equal sub-windows the measured span is sliced into for the stability
#: CoV. Slicing granularity is a methodology choice (the E2 record is archived
#: local-only); the CoV threshold (0.05) and the 3-consecutive count are the
#: ratified values. 8 sub-windows over a 240s span (~30s each) leaves ample room
#: for the 3-consecutive check.
DEFAULT_STABILITY_SUBWINDOW_COUNT = 8


# ---------------------------------------------------------------------------
# WindowSpec - the first-class window object (D7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSpec:
    """One measured window's definition: ``{rate, duration_or_count, ramp, policy}``.

    ``duration_seconds`` and ``request_count`` are the two forms of
    ``duration_or_count``; exactly one is resolved (``duration_seconds`` defaults to
    the E2 floor when neither is given). ``ramp_exclusion_seconds`` is excluded
    PROSPECTIVELY - the measured span starts after the ramp, it is never trimmed
    afterwards. ``attribution_policy`` is disclosed, not chosen (single v0.7 value).
    """

    rate: float
    duration_seconds: float | None = DEFAULT_WINDOW_SECONDS
    request_count: int | None = None
    ramp_exclusion_seconds: float = DEFAULT_RAMP_EXCLUSION_SECONDS
    attribution_policy: str = ATTRIBUTION_STEADY_STATE_SPAN
    arrival: str = "poisson"

    def __post_init__(self) -> None:
        if self.rate <= 0.0:
            raise ValueError(f"WindowSpec.rate must be > 0 (got {self.rate}).")
        if self.duration_seconds is not None and self.request_count is not None:
            raise ValueError(
                "WindowSpec sets both duration_seconds and request_count; a window is "
                "duration-bounded XOR count-bounded."
            )
        if self.duration_seconds is None and self.request_count is None:
            # Neither given: apply the E2 measured-span default (mirrors the config).
            object.__setattr__(self, "duration_seconds", DEFAULT_WINDOW_SECONDS)
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("WindowSpec.duration_seconds must be > 0.")
        if self.request_count is not None and self.request_count < 1:
            raise ValueError("WindowSpec.request_count must be >= 1.")
        if self.ramp_exclusion_seconds < 0.0:
            raise ValueError("WindowSpec.ramp_exclusion_seconds must be >= 0.")

    @classmethod
    def from_traffic_config(cls, traffic: TrafficConfig) -> WindowSpec:
        """Build a :class:`WindowSpec` from an SM4 :class:`TrafficConfig`.

        ``TrafficConfig`` has already resolved its window default (window_seconds
        defaults to the E2 floor when neither window field is set), so this is a
        straight projection. The attribution policy is the single ratified value.
        """
        return cls(
            rate=traffic.rate,
            duration_seconds=traffic.window_seconds,
            request_count=traffic.window_requests,
            ramp_exclusion_seconds=traffic.ramp_exclusion_seconds,
            arrival=traffic.arrival,
        )


# ---------------------------------------------------------------------------
# Event-driven delineation (D19)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStartEvent:
    """Emitted when the measured span opens (after the prospective ramp)."""

    level_index: int
    spec: WindowSpec
    monotonic_at: float


@dataclass(frozen=True)
class WindowStopEvent:
    """Emitted when the measured span closes (energy accounting ends here)."""

    level_index: int
    spec: WindowSpec
    monotonic_at: float


@runtime_checkable
class WindowEnergySink(Protocol):
    """Consumes window events and controls the energy measurement (D19).

    The single seam between the window manager and the energy sampler: the manager
    EMITS events, the sink translates them into energy start/stop. The default sink
    drives a :class:`MeasurementBracket`; a future remote agent implements the same
    two methods over the wire.
    """

    def open_window(self, event: WindowStartEvent) -> None:
        """Begin energy measurement for the window the event opens."""
        ...

    def close_window(self, event: WindowStopEvent) -> MeasuredWindowCore | None:
        """End energy measurement and return the measured core (or None if unavailable)."""
        ...


@runtime_checkable
class _BracketLike(Protocol):
    """The subset of :class:`MeasurementBracket` the default sink drives."""

    def __enter__(self) -> Any: ...
    def __exit__(self, *exc: Any) -> None: ...
    def finish(self) -> MeasuredWindowCore: ...


BracketFactory = Callable[[], _BracketLike]


class BracketEnergySink:
    """Default energy sink: brackets each window with a fresh MeasurementBracket (C2).

    ``open_window`` enters a new bracket (energy tracker + thermal sampler start);
    ``close_window`` exits it (thermal sampler stop) and calls ``finish()`` (energy
    tracker stop), returning the :class:`MeasuredWindowCore`. A bracket is
    single-use, so one is minted per window. The manual enter/exit (rather than a
    ``with`` block) is required because the window body is the concurrently-running
    traffic, not a synchronous call - the bracket brackets the WINDOW, not a call.
    """

    def __init__(self, *, bracket_factory: BracketFactory) -> None:
        self._bracket_factory = bracket_factory
        self._bracket: _BracketLike | None = None

    @classmethod
    def from_measurement_config(
        cls,
        measurement_config: MeasurementConfig,
        gpu_indices: list[int] | None,
        progress: ProgressCallback | None = None,
    ) -> BracketEnergySink:
        """Build a sink that mints real MeasurementBrackets from ``measurement_config``."""
        from llenergymeasure.harness.bracket import MeasurementBracket

        def factory() -> _BracketLike:
            return MeasurementBracket(
                measurement_config,
                gpu_indices,
                progress,
                measure_detail="server measurement window",
            )

        return cls(bracket_factory=factory)

    def open_window(self, event: WindowStartEvent) -> None:
        if self._bracket is not None:
            raise RuntimeError("open_window called while a window is already open.")
        bracket = self._bracket_factory()
        bracket.__enter__()
        self._bracket = bracket

    def close_window(self, event: WindowStopEvent) -> MeasuredWindowCore | None:
        if self._bracket is None:
            raise RuntimeError("close_window called with no open window.")
        bracket = self._bracket
        self._bracket = None
        bracket.__exit__(None, None, None)
        return bracket.finish()


# ---------------------------------------------------------------------------
# Warmup-hook seam (SM8 fills; no-op default here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupContext:
    """Everything the per-level warmup needs (opaque to SM7)."""

    level_index: int
    spec: WindowSpec


#: The warmup-hook seam: an opaque per-level callable run BEFORE a window opens.
#: SM7 defines the signature and a no-op default; SM8 fills it with the
#: convergence-composite warmup (re-warm per level is SM8's policy, behind this
#: seam). The hook may be sync or async - the manager awaits it when awaitable.
WarmupHook = Callable[[WarmupContext], Awaitable[None] | None]


async def _noop_warmup(context: WarmupContext) -> None:
    """Default warmup hook: does nothing (SM8 replaces it)."""
    return None


# ---------------------------------------------------------------------------
# Per-request token receipt seam (client-side counting is SM11 / O8)
# ---------------------------------------------------------------------------

#: Returns the monotonic receipt timestamps of a request's output tokens (one per
#: token), used to attribute tokens to the energy span by RECEIPT time. Client-side
#: token counting is SM11's job (O8: client counts are the canonical J/token
#: denominator); until it lands the default returns no receipts, so the energy
#: denominator is 0 and SM11 wires the real counter in. Tests inject fakes.
TokenReceiptFn = Callable[["RequestRecord"], Sequence[float]]


def _no_token_receipts(record: RequestRecord) -> Sequence[float]:
    return ()


# ---------------------------------------------------------------------------
# Dual boundary-policy bookkeeping (D7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBoundaries:
    """The window's monotonic-clock boundaries (shared clock with SM5's issuer).

    ``window_start`` is when load began; ``span_start = window_start + ramp`` is
    when the measured (energy) span opened; ``span_end`` is when it closed. Requests
    keep being followed to completion past ``span_end`` for latency (drain), but
    energy never extends past it.
    """

    window_start: float
    span_start: float
    span_end: float


@dataclass(frozen=True)
class LatencyRecord:
    """One request's latency record under the drain-before-close policy.

    Present iff the request was ISSUED within the measured span; ``completed_at``
    (and hence ``latency_s``) may fall PAST ``span_end`` (the drain) or be ``None``
    if the request never completed (e.g. cancelled at a drain timeout).
    """

    index: int
    issued_at: float
    completed_at: float | None
    latency_s: float | None


@dataclass(frozen=True)
class WindowBookkeeping:
    """The window's own bookkeeping: the two boundary policies, kept distinct.

    ``energy_denominator_tokens`` (ENERGY policy) counts client-counted output
    tokens whose RECEIPT time fell within ``[span_start, span_end]`` - across ALL
    requests, regardless of when issued. ``latency_records`` (LATENCY policy) covers
    requests ISSUED within the span, drained to completion. A boundary-straddling
    request (issued in-span, completing after ``span_end``) appears in
    ``latency_records`` with its full latency yet contributes only its in-span
    tokens to ``energy_denominator_tokens`` - the two numbers never collapse.

    SM12 derives J/token, percentiles, and goodput from these; SM7 only classifies.
    """

    boundaries: WindowBoundaries
    attribution_policy: str
    energy_denominator_tokens: int
    latency_records: list[LatencyRecord]
    issued_in_span_count: int
    completed_in_span_count: int
    straddling_count: int


def _tokens_in_span(receipts: Sequence[float], span_start: float, span_end: float) -> int:
    return sum(1 for t in receipts if span_start <= t <= span_end)


def build_window_bookkeeping(
    boundaries: WindowBoundaries,
    report: IssuerReport,
    *,
    token_receipt_fn: TokenReceiptFn = _no_token_receipts,
    attribution_policy: str = ATTRIBUTION_STEADY_STATE_SPAN,
) -> WindowBookkeeping:
    """Classify an issuer report into the two boundary policies (never conflated).

    ENERGY denominator: tokens received in ``[span_start, span_end]`` across every
    request (a request issued in the ramp but still generating during the span
    still contributes its in-span tokens; a request completing after ``span_end``
    contributes only the tokens it delivered before ``span_end``).

    LATENCY records: one per request ISSUED in ``[span_start, span_end]``, carrying
    its full latency even when completion falls in the drain past ``span_end``.
    """
    span_start = boundaries.span_start
    span_end = boundaries.span_end

    energy_tokens = 0
    latency_records: list[LatencyRecord] = []
    issued_in_span = 0
    completed_in_span = 0
    straddling = 0

    for record in report.records:
        energy_tokens += _tokens_in_span(token_receipt_fn(record), span_start, span_end)

        issued_in_span_flag = span_start <= record.issued_at <= span_end
        if not issued_in_span_flag:
            continue

        issued_in_span += 1
        completed_at = record.completed_at
        latency_s = (completed_at - record.issued_at) if completed_at is not None else None
        latency_records.append(
            LatencyRecord(
                index=record.index,
                issued_at=record.issued_at,
                completed_at=completed_at,
                latency_s=latency_s,
            )
        )
        if completed_at is not None and completed_at <= span_end:
            completed_in_span += 1
        else:
            # Issued in-span but completing after span_end (or never): a
            # boundary-straddling request - full latency, energy does not extend.
            straddling += 1

    return WindowBookkeeping(
        boundaries=boundaries,
        attribution_policy=attribution_policy,
        energy_denominator_tokens=energy_tokens,
        latency_records=latency_records,
        issued_in_span_count=issued_in_span,
        completed_in_span_count=completed_in_span,
        straddling_count=straddling,
    )


# ---------------------------------------------------------------------------
# Per-level stability validation (E2 thresholds, reused windowing.py math)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelValidation:
    """Per-level steady-state verdict. A failing level is stamped, never dropped."""

    valid: bool
    reason: str | None
    cov_max: float | None
    subwindow_count: int


def _subwindow_means(power_series: Sequence[float], subwindow_count: int) -> list[float]:
    """Slice ``power_series`` into ``subwindow_count`` contiguous groups, mean each.

    Index-based slicing (matching windowing.py's index-based sliding window). Every
    group is non-empty because the caller clamps ``subwindow_count`` to the sample
    count.
    """
    n = len(power_series)
    means: list[float] = []
    for i in range(subwindow_count):
        lo = i * n // subwindow_count
        hi = (i + 1) * n // subwindow_count
        group = power_series[lo:hi]
        means.append(sum(group) / len(group))
    return means


def validate_level_stability(
    power_series: Sequence[float],
    *,
    subwindow_count: int = DEFAULT_STABILITY_SUBWINDOW_COUNT,
    consecutive: int = STABILITY_CONSECUTIVE_WINDOWS,
) -> LevelValidation:
    """Validate a level's measured-span stability against the E2 threshold.

    ``power_series`` is the measured span's per-sample power (already time-ordered,
    non-positive samples dropped by the caller). It is sliced into equal
    sub-windows; the level is STABLE iff the coefficient of variation over every
    ``consecutive`` sub-windows stays at or below ``windowing.py``'s
    ``_AUTO_CV_THRESHOLD`` (0.05) through the end of the span - the reused
    stable-through-end rule (the ramp already removed the onset transient, so the
    check runs from the first sub-window). Too few samples for a meaningful
    ``consecutive``-window check is a validation FAILURE with a reason, not a pass.
    """
    n = len(power_series)
    k = min(subwindow_count, n)
    if k < consecutive:
        return LevelValidation(
            valid=False,
            reason=(
                f"insufficient power samples for the {consecutive}-consecutive-window "
                f"stability check: {n} sample(s) yield {k} sub-window(s), need "
                f"at least {consecutive}."
            ),
            cov_max=None,
            subwindow_count=k,
        )

    signals = _subwindow_means(power_series, k)
    stable = _is_stable_through_end(signals, 0, consecutive)
    cov_max = max(
        _coefficient_of_variation(signals[s : s + consecutive])
        for s in range(len(signals) - consecutive + 1)
    )
    reason = (
        None
        if stable
        else (
            f"steady-state not met: worst coefficient of variation over "
            f"{consecutive} consecutive sub-windows was {cov_max:.4f}, exceeding the "
            f"{_AUTO_CV_THRESHOLD} threshold."
        )
    )
    return LevelValidation(valid=stable, reason=reason, cov_max=cov_max, subwindow_count=k)


# ---------------------------------------------------------------------------
# Multi-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class LevelPlan:
    """One rate level's inputs for the window manager.

    ``traffic_source`` and ``transport`` are pre-built by the caller (SM9) from the
    level's config. CONTRACT: the source must keep issuing throughout the measured
    span ``[ramp, ramp + duration]`` - the manager controls the energy-window timing
    and does not resize the schedule.
    """

    spec: WindowSpec
    traffic_source: TrafficSource
    transport: Transport
    token_receipt_fn: TokenReceiptFn = _no_token_receipts


@dataclass
class LevelOutcome:
    """Everything one level produced: energy, dual-policy bookkeeping, verdict."""

    level_index: int
    spec: WindowSpec
    energy: MeasuredWindowCore | None
    bookkeeping: WindowBookkeeping
    validation: LevelValidation
    issuer_report: IssuerReport
    start_event: WindowStartEvent
    stop_event: WindowStopEvent


def _core_power_series(core: MeasuredWindowCore | None) -> list[float]:
    """Extract the ordered, positive power readings from a measured core."""
    if core is None:
        return []
    samples: list[PowerThermalSample] = core.timeseries_samples
    return [s.power_w for s in samples if s.power_w is not None and s.power_w > 0.0]


class WindowManager:
    """Drives a rate sweep as a list of measured windows (D7 / D19 / E2).

    Per level: run the warmup hook (SM8), start the open-loop traffic, exclude the
    ramp PROSPECTIVELY, emit start-window (energy opens), hold the measured span,
    emit stop-window (energy closes), drain the traffic to completion for latency,
    then validate and cool down before the next level. The energy window is defined
    by the emitted events - never by post-hoc timestamp diffing.
    """

    def __init__(
        self,
        energy_sink: WindowEnergySink,
        *,
        cooldown_seconds: float = 0.0,
        warmup_hook: WarmupHook | None = None,
        subwindow_count: int = DEFAULT_STABILITY_SUBWINDOW_COUNT,
        drain_timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be >= 0.")
        self._energy_sink = energy_sink
        self._cooldown_seconds = cooldown_seconds
        self._warmup_hook: WarmupHook = warmup_hook if warmup_hook is not None else _noop_warmup
        self._subwindow_count = subwindow_count
        self._drain_timeout = drain_timeout
        self._sleep = sleep
        self._clock = clock

    async def run_levels(self, levels: Sequence[LevelPlan]) -> list[LevelOutcome]:
        """Run every level in order, cooling down between (not after the last)."""
        outcomes: list[LevelOutcome] = []
        last = len(levels) - 1
        for level_index, level in enumerate(levels):
            outcomes.append(await self.run_window(level_index, level))
            if level_index != last and self._cooldown_seconds > 0.0:
                await self._sleep(self._cooldown_seconds)
        return outcomes

    async def run_window(self, level_index: int, level: LevelPlan) -> LevelOutcome:
        """Run one level: warmup -> open -> drive traffic -> close -> drain -> validate."""
        spec = level.spec
        if spec.duration_seconds is None:
            raise ValueError(
                "count-based measured windows (request_count without duration_seconds) "
                "are not implemented at v0.7: the measured-span timing and sub-window "
                "stability are duration-grounded (E2). Set a duration."
            )

        await self._run_warmup_hook(WarmupContext(level_index=level_index, spec=spec))

        window_start = self._clock()
        traffic_task: asyncio.Task[IssuerReport] = asyncio.create_task(
            level.traffic_source.run(level.transport, drain_timeout=self._drain_timeout)
        )
        try:
            # Prospective ramp exclusion: the energy span opens AFTER the ramp.
            await self._sleep(spec.ramp_exclusion_seconds)
            span_start = self._clock()
            start_event = WindowStartEvent(
                level_index=level_index, spec=spec, monotonic_at=span_start
            )
            self._energy_sink.open_window(start_event)

            # Hold the measured span.
            await self._sleep(spec.duration_seconds)
            span_end = self._clock()
            stop_event = WindowStopEvent(level_index=level_index, spec=spec, monotonic_at=span_end)
            core = self._energy_sink.close_window(stop_event)

            # Drain-before-close: energy has stopped; wait for every in-flight
            # request to complete so its latency record is captured.
            report = await traffic_task
        except BaseException:
            traffic_task.cancel()
            with contextlib.suppress(BaseException):
                await traffic_task
            raise

        boundaries = WindowBoundaries(
            window_start=window_start, span_start=span_start, span_end=span_end
        )
        bookkeeping = build_window_bookkeeping(
            boundaries,
            report,
            token_receipt_fn=level.token_receipt_fn,
            attribution_policy=spec.attribution_policy,
        )
        validation = validate_level_stability(
            _core_power_series(core), subwindow_count=self._subwindow_count
        )
        return LevelOutcome(
            level_index=level_index,
            spec=spec,
            energy=core,
            bookkeeping=bookkeeping,
            validation=validation,
            issuer_report=report,
            start_event=start_event,
            stop_event=stop_event,
        )

    async def _run_warmup_hook(self, context: WarmupContext) -> None:
        result = self._warmup_hook(context)
        if inspect.isawaitable(result):
            await result


__all__ = [
    "ATTRIBUTION_STEADY_STATE_SPAN",
    "DEFAULT_STABILITY_SUBWINDOW_COUNT",
    "STABILITY_CONSECUTIVE_WINDOWS",
    "BracketEnergySink",
    "LatencyRecord",
    "LevelOutcome",
    "LevelPlan",
    "LevelValidation",
    "TokenReceiptFn",
    "WarmupContext",
    "WarmupHook",
    "WindowBookkeeping",
    "WindowBoundaries",
    "WindowEnergySink",
    "WindowManager",
    "WindowSpec",
    "WindowStartEvent",
    "WindowStopEvent",
    "build_window_bookkeeping",
    "validate_level_stability",
]
