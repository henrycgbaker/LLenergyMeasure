"""First-class measurement window + multi-level window manager for server mode.

This module owns server mode's measured-window abstraction: the :class:`WindowSpec`
(the first-class window object), the event-driven delineation that brackets one
window's energy, the dual boundary policies that keep energy and latency
accounting distinct, per-level steady-state validation, and the multi-level
orchestration that drives a rate sweep.

Design anchors:

- **First-class window object + two coexisting boundary policies.** A window
  is ``{rate, duration_or_count, attribution_policy, ramp_exclusion}``. The ENERGY
  policy amortizes steady-state energy over the measured span ``[span_start,
  span_end]`` (denominator = client-counted tokens RECEIVED in that span). The
  LATENCY policy is drain-before-close: every request ISSUED in the measured span
  gets a full latency record, followed to completion PAST ``span_end`` - but energy
  accounting never extends into the drain. The two policies are never conflated
  into one number.
- **Event-driven delineation.** The manager emits explicit
  start-window / stop-window events to a :class:`WindowEnergySink`; the sink (by
  default a :class:`MeasurementBracket`-backed one) opens and closes the energy
  measurement in response. No component infers window membership from timestamps
  alone - the events are the protocol, a drop-in for a future remote sampling
  agent.
- **Reuse.** The default sink reuses
  :class:`~llenergymeasure.harness.bracket.MeasurementBracket` (a bracket brackets a
  WINDOW, not a Python call). The stability gate reuses windowing.py's
  coefficient-of-variation, stable-through-end, clean, and clip machinery plus the
  trapezoidal integrator unchanged; only the multi-window / multi-level
  orchestration is new.
- **Ratified numeric defaults + gate formulation.** Measured span 240s, ramp
  exclusion 30s absolute (both config-exposed under ``server.traffic``, carried here
  as :class:`WindowSpec` defaults). The stability gate is calibrated on J/TOKEN, not
  power (power is near-noise-free at these timescales and would always pass):

  * Per window (DIAGNOSTIC, disclosed, feeds the derived-metrics overlay): the coefficient of variation
    over ``k = 4`` contiguous sub-windows' J/token. Each sub-window's J/token is the
    trapezoidal integral of the power series over the sub-window divided by the
    client-counted tokens attributed to it. ``k`` is fixed at 4: the 0.05 threshold
    is calibrated at ``k = 4`` and changing it silently invalidates the threshold.
  * Per level (the GATE): a level runs ``windows_per_level`` (default 3) consecutive
    measured windows at the configured duration, contiguous at the same rate with NO
    re-warmup between them; the level passes iff the window-level J/token values
    agree within 0.05 (CoV over every 3 consecutive windows, stable through the end
    of the level - the perf_analyzer convention). A failing level is
    stamped invalid-with-reason, never dropped.

Out of scope (built elsewhere): warmup EXECUTION behind the warmup-hook seam,
server session lifecycle / persistence, and metrics derivation
(percentiles, goodput) beyond this module's own bookkeeping.
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
from llenergymeasure.energy.nvml import integrate_power_samples

# REUSE BINDING (server-mode plan section 14, maintainer-confirmed): the CV /
# steady-state / clean / clip machinery is consumed from windowing.py, never
# reimplemented. These live in the same package (Layer 3), so import altitude does
# not require an extraction into a shared home - a direct import is the minimal
# faithful reuse. NO math changes.
from llenergymeasure.harness.windowing import (
    _AUTO_CV_THRESHOLD,
    _clean_samples,
    _coefficient_of_variation,
    _filter_to_window,
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
#: configurable knob; requests.parquet keeps per-request timestamps so
#: alternative attributions stay re-derivable offline.
ATTRIBUTION_STEADY_STATE_SPAN = "steady_state_span"

#: Consecutive-window agreement count for the LEVEL gate
#: (perf_analyzer's "3 consecutive windows within tolerance").
STABILITY_CONSECUTIVE_WINDOWS = 3

#: Consecutive measured windows run per rate level by default (adjustable on the
#: manager). Default 3 = the gate's consecutive-window count, so a default level
#: yields exactly one 3-consecutive check.
DEFAULT_WINDOWS_PER_LEVEL = 3

#: Sub-windows per window for the intra-window diagnostic CoV. FIXED at 4: this is
#: the calibration constant - the 0.05 threshold is calibrated at k = 4, so
#: changing k silently invalidates the threshold. Not configurable by design.
_STABILITY_SUBWINDOWS = 4


# ---------------------------------------------------------------------------
# WindowSpec - the first-class window object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSpec:
    """One measured window's definition: ``{rate, duration_or_count, ramp, policy}``.

    ``duration_seconds`` and ``request_count`` are the two forms of
    ``duration_or_count``; exactly one is resolved (``duration_seconds`` defaults to
    the calibrated floor when neither is given). ``request_count`` is represented for a
    future release but rejected at config validation at v0.7 (the manager also
    guards it as an internal belt). ``ramp_exclusion_seconds`` is excluded
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
            # Neither given: apply the calibrated measured-span default (mirrors the config).
            object.__setattr__(self, "duration_seconds", DEFAULT_WINDOW_SECONDS)
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("WindowSpec.duration_seconds must be > 0.")
        if self.request_count is not None and self.request_count < 1:
            raise ValueError("WindowSpec.request_count must be >= 1.")
        if self.ramp_exclusion_seconds < 0.0:
            raise ValueError("WindowSpec.ramp_exclusion_seconds must be >= 0.")

    @classmethod
    def from_traffic_config(cls, traffic: TrafficConfig) -> WindowSpec:
        """Build a :class:`WindowSpec` from a :class:`TrafficConfig`.

        ``TrafficConfig`` has already resolved its window default (window_seconds
        defaults to the calibrated floor) and rejected count-bound windows at v0.7, so this
        is a straight projection. The attribution policy is the single ratified value.
        """
        return cls(
            rate=traffic.rate,
            duration_seconds=traffic.window_seconds,
            request_count=traffic.window_requests,
            ramp_exclusion_seconds=traffic.ramp_exclusion_seconds,
            arrival=traffic.arrival,
        )


# ---------------------------------------------------------------------------
# Event-driven delineation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowStartEvent:
    """Emitted when a measured window opens (after the level's prospective ramp)."""

    level_index: int
    window_index: int
    spec: WindowSpec
    monotonic_at: float


@dataclass(frozen=True)
class WindowStopEvent:
    """Emitted when a measured window closes (energy accounting ends here)."""

    level_index: int
    window_index: int
    spec: WindowSpec
    monotonic_at: float


@dataclass(frozen=True)
class WindowAbortEvent:
    """Emitted when an OPEN window is torn down early (cancellation / exception).

    The manager owns the sink lifecycle, so when an exception or cancellation fires
    while a window is open, it delivers this explicit event (an abort is an
    event, never inferred) so the sink RELEASES its live sampler/tracker. A window
    is either closed or aborted, never both. ``cause`` describes the triggering
    exception for the invalid-with-reason stamp.
    """

    level_index: int
    window_index: int
    spec: WindowSpec
    monotonic_at: float
    cause: str


@runtime_checkable
class WindowEnergySink(Protocol):
    """Consumes window events and controls the energy measurement.

    The single seam between the window manager and the energy sampler: the manager
    EMITS events, the sink translates them into energy start/stop/abort. The default
    sink drives a :class:`MeasurementBracket`; a future remote agent implements the
    same three methods over the wire.
    """

    def open_window(self, event: WindowStartEvent) -> None:
        """Begin energy measurement for the window the event opens."""
        ...

    def close_window(self, event: WindowStopEvent) -> MeasuredWindowCore | None:
        """End energy measurement and return the measured core (or None if unavailable)."""
        ...

    def abort_window(self, event: WindowAbortEvent) -> None:
        """Release an OPEN window's live measurement without producing a core.

        Called when a window is torn down early; must free the sampler/tracker and
        must be safe when no window is open (a no-op). The manager guarantees a
        window is closed XOR aborted, so this is never paired with close_window.
        """
        ...


@runtime_checkable
class _BracketLike(Protocol):
    """The subset of :class:`MeasurementBracket` the default sink drives."""

    def __enter__(self) -> Any: ...
    def __exit__(self, *exc: Any) -> None: ...
    def finish(self) -> MeasuredWindowCore: ...


BracketFactory = Callable[[], _BracketLike]


class BracketEnergySink:
    """Default energy sink: brackets each window with a fresh MeasurementBracket.

    ``open_window`` enters a new bracket (energy tracker + thermal sampler start);
    ``close_window`` exits it (thermal sampler stop) and calls ``finish()`` (energy
    tracker stop), returning the :class:`MeasuredWindowCore`. A bracket is
    single-use, so one is minted per window (one bundle per window). The manual
    enter/exit (rather than a ``with`` block) is required because the window body is
    the concurrently-running traffic, not a synchronous call - the bracket brackets
    the WINDOW, not a call.
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

    def abort_window(self, event: WindowAbortEvent) -> None:
        """Release the live bracket (thermal sampler + energy tracker), discard the core.

        Best-effort per step so BOTH resources are freed even if one teardown raises;
        pops the bracket first so a subsequent close/abort cannot double-release. A
        no-op when no window is open.
        """
        if self._bracket is None:
            return
        bracket = self._bracket
        self._bracket = None
        with contextlib.suppress(BaseException):
            bracket.__exit__(None, None, None)  # stop the thermal sampler
        with contextlib.suppress(BaseException):
            bracket.finish()  # stop the energy tracker; the returned core is discarded


# ---------------------------------------------------------------------------
# Warmup-hook seam (the warmup protocol fills; no-op default here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupContext:
    """Everything the per-level warmup needs (opaque to the window manager)."""

    level_index: int
    spec: WindowSpec


#: The warmup-hook seam: an opaque per-level callable run ONCE before a level's
#: windows open. The window manager defines the signature and a no-op default; the
#: warmup protocol fills it with the convergence-composite warmup. The hook may be
#: sync or async - the manager
#: awaits it when awaitable.
WarmupHook = Callable[[WarmupContext], Awaitable[None] | None]


async def _noop_warmup(context: WarmupContext) -> None:
    """Default warmup hook: does nothing (the warmup protocol replaces it)."""
    return None


# ---------------------------------------------------------------------------
# Per-request token receipt seam (client-side counting)
# ---------------------------------------------------------------------------

#: Returns the monotonic receipt timestamps of a request's output tokens (one per
#: token), the ONE mechanism that feeds BOTH the energy denominator and the
#: stability gate's per-sub-window J/token. Client-side token counting is the
#: request-logging layer's job (client counts are the canonical denominator). Two
#: granularities are legal:
#:
#: - token-granular (one timestamp per token) -> counting per interval is
#:   span-received counting, the ratified energy-denominator rule;
#: - request-granular (all of a request's tokens stamped at its completion time,
#:   i.e. ``[completed_at] * n_tokens``) -> a request's whole token count falls in
#:   the interval containing its completion, which is EXACTLY the calibrated
#:   completion-timestamp attribution rule for sub-window J/token.
#:
#: Until client-side counting lands the default returns no receipts, so denominators are 0 (the gate
#: reports invalid-with-reason and the energy denominator is 0). Tests inject fakes.
TokenReceiptFn = Callable[["RequestRecord"], Sequence[float]]


def _no_token_receipts(record: RequestRecord) -> Sequence[float]:
    return ()


# ---------------------------------------------------------------------------
# Dual boundary-policy bookkeeping
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBoundaries:
    """The window's monotonic-clock boundaries (shared clock with the traffic issuer).

    ``window_start`` is when the level's load began; ``span_start`` is when this
    window's measured (energy) span opened; ``span_end`` is when it closed. Requests
    keep being followed to completion past ``span_end`` for latency (drain), but
    energy never extends past it.
    """

    window_start: float
    span_start: float
    span_end: float


@dataclass(frozen=True)
class WindowBookkeeping:
    """The window's span-clipped energy denominator and attribution policy.

    ``energy_denominator_tokens`` (ENERGY policy) counts client-counted output
    tokens whose RECEIPT time fell within ``[span_start, span_end]`` - across ALL
    requests, regardless of when issued. ``attribution_policy`` names the energy
    attribution rule in force for the window.

    The derived-metrics overlay derives percentiles and goodput from the per-request
    RequestLogRow rows (build_request_rows_by_window), not from this bookkeeping,
    which carries only the span-clipped energy denominator and the attribution
    policy; the window manager only classifies.
    """

    boundaries: WindowBoundaries
    attribution_policy: str
    energy_denominator_tokens: int


def _count_tokens_in_interval(
    records: Sequence[RequestRecord],
    receipt_fn: TokenReceiptFn,
    lo: float,
    hi: float,
    *,
    closed_hi: bool,
) -> int:
    """Count output-token receipts in ``[lo, hi)`` (``[lo, hi]`` when ``closed_hi``).

    Half-open by default so contiguous sub-windows partition without double-counting;
    the final sub-window and the whole-span denominator close the upper edge.
    """
    total = 0
    for record in records:
        for t in receipt_fn(record):
            if lo <= t < hi or (closed_hi and t == hi):
                total += 1
    return total


def build_window_bookkeeping(
    boundaries: WindowBoundaries,
    report: IssuerReport,
    *,
    token_receipt_fn: TokenReceiptFn = _no_token_receipts,
    attribution_policy: str = ATTRIBUTION_STEADY_STATE_SPAN,
) -> WindowBookkeeping:
    """Compute the window's span-clipped energy denominator from an issuer report.

    ENERGY denominator: tokens received in ``[span_start, span_end]`` across every
    request (a request issued in the ramp but still generating during the span
    still contributes its in-span tokens; a request completing after ``span_end``
    contributes only the tokens it delivered before ``span_end``).
    """
    span_start = boundaries.span_start
    span_end = boundaries.span_end

    energy_tokens = _count_tokens_in_interval(
        report.records, token_receipt_fn, span_start, span_end, closed_hi=True
    )
    return WindowBookkeeping(
        boundaries=boundaries,
        attribution_policy=attribution_policy,
        energy_denominator_tokens=energy_tokens,
    )


# ---------------------------------------------------------------------------
# J/token measurement (window + intra-window CoV), reusing windowing.py
# ---------------------------------------------------------------------------


def _integrate_interval(cleaned: list[PowerThermalSample], lo_ts: float, hi_ts: float) -> float:
    """Trapezoidal energy (J) over ``[lo_ts, hi_ts]`` with endpoint interpolation.

    Reuses windowing.py's clip-with-interpolation (``_filter_to_window``) and the
    trapezoidal integrator (``integrate_power_samples``); summed across GPUs.
    """
    if hi_ts <= lo_ts:
        return 0.0
    clipped = _filter_to_window(cleaned, lo_ts, 0.0, hi_ts - lo_ts)
    return sum(integrate_power_samples(clipped).values())


def _intra_window_cov(
    cleaned: list[PowerThermalSample],
    boundaries: WindowBoundaries,
    report: IssuerReport,
    receipt_fn: TokenReceiptFn,
    power_lo: float,
    power_hi: float,
) -> float | None:
    """CoV over ``k = 4`` sub-window J/token values (diagnostic; None if unformable).

    Each clock is partitioned into 4 equal quarters of its OWN measured span - the
    power series in its sampler clock, the token receipts in the issuer's monotonic
    clock - so the quarters are fraction-aligned to the same physical span despite
    the two clocks' different epochs (perf_counter vs monotonic). A sub-window with
    zero attributed tokens makes the ratio unformable, so the diagnostic is None.
    """
    k = _STABILITY_SUBWINDOWS
    power_q = (power_hi - power_lo) / k
    token_q = (boundaries.span_end - boundaries.span_start) / k
    if power_q <= 0.0 or token_q <= 0.0:
        return None

    j_per_token: list[float] = []
    for i in range(k):
        energy = _integrate_interval(cleaned, power_lo + i * power_q, power_lo + (i + 1) * power_q)
        lo = boundaries.span_start + i * token_q
        hi = boundaries.span_start + (i + 1) * token_q
        tokens = _count_tokens_in_interval(
            report.records, receipt_fn, lo, hi, closed_hi=(i == k - 1)
        )
        if tokens <= 0:
            return None
        j_per_token.append(energy / tokens)
    return _coefficient_of_variation(j_per_token)


def _window_measurements(
    core: MeasuredWindowCore | None,
    boundaries: WindowBoundaries,
    report: IssuerReport,
    receipt_fn: TokenReceiptFn,
    window_tokens: int,
) -> tuple[float | None, float | None, float | None]:
    """Return ``(window_energy_j, window_j_per_token, intra_window_cov)``.

    Energy is the trapezoidal integral of the (cleaned) power series over the
    window; J/token is that over the window's client-counted token denominator; the
    intra-window CoV is the k=4 sub-window diagnostic.
    """
    samples: list[PowerThermalSample] = list(core.timeseries_samples) if core is not None else []
    if len(samples) < 2:
        return None, None, None
    cleaned = _clean_samples(samples)
    if len(cleaned) < 2:
        return None, None, None

    power_lo = cleaned[0].timestamp
    power_hi = cleaned[-1].timestamp
    window_energy = _integrate_interval(cleaned, power_lo, power_hi)
    window_j_per_token = (window_energy / window_tokens) if window_tokens > 0 else None
    intra = _intra_window_cov(cleaned, boundaries, report, receipt_fn, power_lo, power_hi)
    return window_energy, window_j_per_token, intra


# ---------------------------------------------------------------------------
# Per-level stability validation (window-to-window J/token gate)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelValidation:
    """Per-level steady-state verdict. A failing level is stamped, never dropped."""

    valid: bool
    reason: str | None
    cov: float | None
    windows_considered: int


def validate_level_stability(
    window_j_per_token: Sequence[float | None],
    *,
    consecutive: int = STABILITY_CONSECUTIVE_WINDOWS,
) -> LevelValidation:
    """Validate a level's window-to-window J/token stability (the stability gate).

    The level passes iff the window-level J/token values agree within
    ``windowing.py``'s ``_AUTO_CV_THRESHOLD`` (0.05) over every ``consecutive``
    windows, sustained through the end of the level (the reused stable-through-end
    rule). Fewer than ``consecutive`` windows, or any window with no valid J/token
    (zero attributed tokens), is a validation FAILURE with a reason - never a pass.
    """
    n = len(window_j_per_token)
    if n < consecutive:
        return LevelValidation(
            valid=False,
            reason=(
                f"the level ran {n} measured window(s); the stability gate needs at "
                f"least {consecutive} consecutive windows."
            ),
            cov=None,
            windows_considered=n,
        )
    if any(v is None for v in window_j_per_token):
        return LevelValidation(
            valid=False,
            reason=(
                "one or more windows produced no J/token (zero attributed output "
                "tokens), so window-to-window stability cannot be assessed."
            ),
            cov=None,
            windows_considered=n,
        )

    values = [float(v) for v in window_j_per_token if v is not None]
    stable = _is_stable_through_end(values, 0, consecutive)
    cov = max(
        _coefficient_of_variation(values[s : s + consecutive]) for s in range(n - consecutive + 1)
    )
    reason = (
        None
        if stable
        else (
            f"window-to-window J/token not steady: worst coefficient of variation "
            f"over {consecutive} consecutive windows was {cov:.4f}, exceeding the "
            f"{_AUTO_CV_THRESHOLD} threshold."
        )
    )
    return LevelValidation(valid=stable, reason=reason, cov=cov, windows_considered=n)


# ---------------------------------------------------------------------------
# Multi-level orchestration
# ---------------------------------------------------------------------------


@dataclass
class LevelPlan:
    """One rate level's inputs for the window manager.

    ``traffic_source`` and ``transport`` are pre-built by the caller (the server session) from the
    level's config. CONTRACT: the source must keep issuing throughout the whole
    level - the ramp plus ``windows_per_level`` measured spans - because the manager
    controls the energy-window timing and does not resize the schedule.
    """

    spec: WindowSpec
    traffic_source: TrafficSource
    transport: Transport
    token_receipt_fn: TokenReceiptFn = _no_token_receipts


@dataclass
class WindowRecord:
    """One measured window's product within a level (one bundle per window)."""

    window_index: int
    boundaries: WindowBoundaries
    energy: MeasuredWindowCore | None
    bookkeeping: WindowBookkeeping
    window_energy_j: float | None
    window_j_per_token: float | None
    intra_window_cov: float | None


@dataclass
class LevelOutcome:
    """Everything one rate level produced: its windows and the window-to-window verdict."""

    level_index: int
    spec: WindowSpec
    windows: list[WindowRecord]
    validation: LevelValidation
    issuer_report: IssuerReport


#: Attribute name under which :func:`WindowManager.run_level` attaches an
#: :class:`AbortedLevel` to the propagating exception when a level fails with any
#: measured cores to preserve (or a window open). The exception itself is re-raised
#: unchanged (CancelledError stays CancelledError); this carries the partial state so
#: the never-silently-dropped posture holds without converting the error.
ABORTED_LEVEL_ATTR = "llem_aborted_level"


@dataclass
class AbortedLevel:
    """Partial state attached to the exception that fails a level.

    ``reason`` names the failure site and cause, one of three disclosed forms:

    - ``"aborted: <cause>"`` - an exception fired while a window was OPEN; its live
      sampler/tracker was released via an abort event.
    - ``"close failed: <cause>"`` - ``close_window`` itself raised; that window's
      bracket state is untrustworthy (no abort event - close was already attempted).
    - ``"drain failed: <cause>"`` - every window closed cleanly but the
      post-measurement drain failed. The energy cores STAND; only straddler latency
      records are lost or truncated - the two-policy separation working as designed
      (energy is intact, latency is best-effort past close).

    ``aborted_window_index`` is the open-or-failed-close window's index, or None for a
    drain failure (all windows closed cleanly; the level failed post-measurement).
    ``completed_cores`` holds the measured cores of the CLEANLY-CLOSED windows (in
    order) so the caller (the server session) can still persist them (drain fields null). Full
    bookkeeping is not reconstructed here because the level's traffic report is
    unavailable once the issuer task is cancelled - that is the session layer's job.
    """

    level_index: int
    aborted_window_index: int | None
    reason: str
    completed_cores: list[MeasuredWindowCore | None]


class WindowManager:
    """Drives a rate sweep as a list of levels, each a run of measured windows.

    Per level: run the warmup hook ONCE, start the open-loop traffic, exclude
    the ramp PROSPECTIVELY once, then run ``windows_per_level`` contiguous measured
    windows (no re-warm between them) - each emitting start-window (energy opens) and
    stop-window (energy closes) events - and finally drain the traffic to completion
    for latency. The energy window is defined by the emitted events, never by
    post-hoc timestamp diffing. Levels are validated on window-to-window J/token
    stability; the caller drives one level per :meth:`run_level` call and owns any
    inter-level cooldown.

    The manager owns the sink lifecycle. If the level fails, it preserves whatever
    was measured: a window OPEN at failure has its live sampler released via an
    explicit abort event (exactly once); a failure while draining after every window
    closed cleanly, or a ``close_window`` that itself raised, keeps the measured
    cores of the cleanly-closed windows. The partial state is attached to the
    propagating exception via an :class:`AbortedLevel`, and the original exception is
    re-raised unchanged (CancelledError stays CancelledError). See
    :class:`AbortedLevel` for the three disclosed failure sites.
    """

    def __init__(
        self,
        energy_sink: WindowEnergySink,
        *,
        windows_per_level: int = DEFAULT_WINDOWS_PER_LEVEL,
        warmup_hook: WarmupHook | None = None,
        drain_timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if windows_per_level < 1:
            raise ValueError("windows_per_level must be >= 1.")
        self._energy_sink = energy_sink
        self._windows_per_level = windows_per_level
        self._warmup_hook: WarmupHook = warmup_hook if warmup_hook is not None else _noop_warmup
        self._drain_timeout = drain_timeout
        self._sleep = sleep
        self._clock = clock

    async def run_level(self, level_index: int, level: LevelPlan) -> LevelOutcome:
        """Run one level: warmup -> ramp -> N contiguous windows -> drain -> validate."""
        spec = level.spec
        if spec.duration_seconds is None:
            raise ValueError(
                "count-based measured windows (request_count without duration_seconds) "
                "are not supported at v0.7: the measured-span timing and the stability "
                "gate are duration-grounded. Set a duration."
            )

        await self._run_warmup_hook(WarmupContext(level_index=level_index, spec=spec))

        window_start = self._clock()
        traffic_task: asyncio.Task[IssuerReport] = asyncio.create_task(
            level.traffic_source.run(level.transport, drain_timeout=self._drain_timeout)
        )
        emitted: list[
            tuple[WindowBoundaries, WindowStartEvent, WindowStopEvent, MeasuredWindowCore | None]
        ] = []
        # Failure-site tracking. ``open_event`` is the currently-OPEN window's start
        # event (the only await inside an open window is the measured-span sleep, so
        # a failure there means a window is genuinely open). ``closing_index`` is set
        # only across the synchronous close_window call, so it identifies a window
        # whose close raised. Both are cleared (sync, no await) so close XOR abort
        # holds exactly once.
        open_event: WindowStartEvent | None = None
        closing_index: int | None = None
        try:
            # Prospective ramp exclusion, ONCE per level: the first window opens after
            # the batch-fill transient; subsequent windows are contiguous (no re-warm).
            await self._sleep(spec.ramp_exclusion_seconds)
            for window_index in range(self._windows_per_level):
                span_start = self._clock()
                start_event = WindowStartEvent(
                    level_index=level_index,
                    window_index=window_index,
                    spec=spec,
                    monotonic_at=span_start,
                )
                self._energy_sink.open_window(start_event)
                open_event = start_event
                await self._sleep(spec.duration_seconds)
                span_end = self._clock()
                stop_event = WindowStopEvent(
                    level_index=level_index,
                    window_index=window_index,
                    spec=spec,
                    monotonic_at=span_end,
                )
                open_event = None  # a close attempt counts as closed: never also abort
                closing_index = window_index
                core = self._energy_sink.close_window(stop_event)
                closing_index = None  # closed cleanly
                boundaries = WindowBoundaries(
                    window_start=window_start, span_start=span_start, span_end=span_end
                )
                emitted.append((boundaries, start_event, stop_event, core))
            # Drain-before-close: energy has stopped; wait for every in-flight request
            # to complete so its latency record is captured.
            report = await traffic_task
        except BaseException as exc:
            traffic_task.cancel()
            with contextlib.suppress(BaseException):
                await traffic_task
            self._attach_level_abort(exc, level_index, spec, open_event, closing_index, emitted)
            raise

        windows = self._build_window_records(emitted, report, level.token_receipt_fn, spec)
        validation = validate_level_stability([w.window_j_per_token for w in windows])
        return LevelOutcome(
            level_index=level_index,
            spec=spec,
            windows=windows,
            validation=validation,
            issuer_report=report,
        )

    @staticmethod
    def _build_window_records(
        emitted: list[
            tuple[WindowBoundaries, WindowStartEvent, WindowStopEvent, MeasuredWindowCore | None]
        ],
        report: IssuerReport,
        token_receipt_fn: TokenReceiptFn,
        spec: WindowSpec,
    ) -> list[WindowRecord]:
        records: list[WindowRecord] = []
        for boundaries, start_event, _stop_event, core in emitted:
            bookkeeping = build_window_bookkeeping(
                boundaries,
                report,
                token_receipt_fn=token_receipt_fn,
                attribution_policy=spec.attribution_policy,
            )
            energy_j, j_per_token, intra = _window_measurements(
                core, boundaries, report, token_receipt_fn, bookkeeping.energy_denominator_tokens
            )
            records.append(
                WindowRecord(
                    window_index=start_event.window_index,
                    boundaries=boundaries,
                    energy=core,
                    bookkeeping=bookkeeping,
                    window_energy_j=energy_j,
                    window_j_per_token=j_per_token,
                    intra_window_cov=intra,
                )
            )
        return records

    async def _run_warmup_hook(self, context: WarmupContext) -> None:
        result = self._warmup_hook(context)
        if inspect.isawaitable(result):
            await result

    def _attach_level_abort(
        self,
        exc: BaseException,
        level_index: int,
        spec: WindowSpec,
        open_event: WindowStartEvent | None,
        closing_index: int | None,
        emitted: list[
            tuple[WindowBoundaries, WindowStartEvent, WindowStopEvent, MeasuredWindowCore | None]
        ],
    ) -> None:
        """Preserve partial state on the propagating exception (never converts it).

        Attaches an :class:`AbortedLevel` unless the failure was a pure ramp-phase
        error with nothing measured and no window open. The original ``exc`` is
        re-raised unchanged by the caller.
        """
        completed_cores = [core for (_, _, _, core) in emitted]
        cause = self._describe_cause(exc)
        aborted_index: int | None
        if open_event is not None:
            # Failure mid-window: the sampler is live, so release it exactly once
            # (best-effort - a raising sink must never mask exc).
            abort_event = WindowAbortEvent(
                level_index=level_index,
                window_index=open_event.window_index,
                spec=spec,
                monotonic_at=self._clock(),
                cause=cause,
            )
            with contextlib.suppress(BaseException):
                self._energy_sink.abort_window(abort_event)
            reason = f"aborted: {cause}"
            aborted_index = open_event.window_index
        elif closing_index is not None:
            # close_window itself raised: the window's bracket state is untrustworthy,
            # and close was already attempted (no abort - close XOR abort stands).
            reason = f"close failed: {cause}"
            aborted_index = closing_index
        elif emitted:
            # Drain failed after every window closed cleanly: the cores stand; only
            # straddler latency records are lost (the two-policy separation).
            reason = f"drain failed: {cause}"
            aborted_index = None
        else:
            return  # ramp-phase failure: nothing measured, nothing open
        setattr(
            exc,
            ABORTED_LEVEL_ATTR,
            AbortedLevel(
                level_index=level_index,
                aborted_window_index=aborted_index,
                reason=reason,
                completed_cores=completed_cores,
            ),
        )

    @staticmethod
    def _describe_cause(exc: BaseException) -> str:
        """Short cause string for the abort stamp (CancelledError reads as 'cancelled')."""
        if isinstance(exc, asyncio.CancelledError):
            return "cancelled"
        return repr(exc)


__all__ = [
    "ABORTED_LEVEL_ATTR",
    "ATTRIBUTION_STEADY_STATE_SPAN",
    "DEFAULT_WINDOWS_PER_LEVEL",
    "STABILITY_CONSECUTIVE_WINDOWS",
    "AbortedLevel",
    "BracketEnergySink",
    "LevelOutcome",
    "LevelPlan",
    "LevelValidation",
    "TokenReceiptFn",
    "WarmupContext",
    "WarmupHook",
    "WindowAbortEvent",
    "WindowBookkeeping",
    "WindowBoundaries",
    "WindowEnergySink",
    "WindowManager",
    "WindowRecord",
    "WindowSpec",
    "WindowStartEvent",
    "WindowStopEvent",
    "build_window_bookkeeping",
    "validate_level_stability",
]
