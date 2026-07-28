"""First-class measurement window + multi-level window manager for server mode.

This module owns server mode's measured-window abstraction: the :class:`WindowSpec`
(the first-class window object), the event-driven delineation that brackets one
window's energy, the dual boundary policies that keep energy and latency
accounting distinct, per-level steady-state validation, and the multi-level
orchestration that drives a rate sweep.

Design anchors (server-mode plan, section 14 as ratified):

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
  WINDOW, not a Python call). The stability gate reuses windowing.py's
  coefficient-of-variation, stable-through-end, clean, and clip machinery plus the
  trapezoidal integrator unchanged; only the multi-window / multi-level
  orchestration is new.
- **E2 - ratified numeric defaults + gate formulation.** Measured span 240s, ramp
  exclusion 30s absolute (both config-exposed under ``server.traffic``, carried here
  as :class:`WindowSpec` defaults). The stability gate is calibrated on J/TOKEN, not
  power (power is near-noise-free at these timescales and would always pass):

  * Per window (DIAGNOSTIC, disclosed, feeds SM12): the coefficient of variation
    over ``k = 4`` contiguous sub-windows' J/token. Each sub-window's J/token is the
    trapezoidal integral of the power series over the sub-window divided by the
    client-counted tokens attributed to it. ``k`` is fixed at 4: the 0.05 threshold
    is calibrated at ``k = 4`` and changing it silently invalidates the threshold.
  * Per level (the GATE): a level runs ``windows_per_level`` (default 3) consecutive
    measured windows at the configured duration, contiguous at the same rate with NO
    re-warmup between them; the level passes iff the window-level J/token values
    agree within 0.05 (CoV over every 3 consecutive windows, stable through the end
    of the level - the perf_analyzer convention E2 confirmed). A failing level is
    stamped invalid-with-reason, never dropped.

Out of scope (later slices): warmup EXECUTION behind the warmup-hook seam (SM8),
server session lifecycle / persistence (SM9 / SM10), and metrics derivation
(percentiles, goodput) beyond this module's own bookkeeping (SM12).
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
#: configurable knob; requests.parquet (SM11) keeps per-request timestamps so
#: alternative attributions stay re-derivable offline.
ATTRIBUTION_STEADY_STATE_SPAN = "steady_state_span"

#: Consecutive-window agreement count for the LEVEL gate (E2 ratified;
#: perf_analyzer's "3 consecutive windows within tolerance").
STABILITY_CONSECUTIVE_WINDOWS = 3

#: Consecutive measured windows run per rate level by default (adjustable on the
#: manager). Default 3 = the gate's consecutive-window count, so a default level
#: yields exactly one 3-consecutive check.
DEFAULT_WINDOWS_PER_LEVEL = 3

#: Sub-windows per window for the intra-window diagnostic CoV. FIXED at 4: this is
#: the E2 calibration constant - the 0.05 threshold is calibrated at k = 4, so
#: changing k silently invalidates the threshold. Not configurable by design.
_STABILITY_SUBWINDOWS = 4


# ---------------------------------------------------------------------------
# WindowSpec - the first-class window object (D7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowSpec:
    """One measured window's definition: ``{rate, duration_or_count, ramp, policy}``.

    ``duration_seconds`` and ``request_count`` are the two forms of
    ``duration_or_count``; exactly one is resolved (``duration_seconds`` defaults to
    the E2 floor when neither is given). ``request_count`` is represented for a
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
        defaults to the E2 floor) and rejected count-bound windows at v0.7, so this
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
# Event-driven delineation (D19)
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
    single-use, so one is minted per window (one bundle per window, SM10). The manual
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


# ---------------------------------------------------------------------------
# Warmup-hook seam (SM8 fills; no-op default here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WarmupContext:
    """Everything the per-level warmup needs (opaque to SM7)."""

    level_index: int
    spec: WindowSpec


#: The warmup-hook seam: an opaque per-level callable run ONCE before a level's
#: windows open. SM7 defines the signature and a no-op default; SM8 fills it with
#: the convergence-composite warmup. The hook may be sync or async - the manager
#: awaits it when awaitable.
WarmupHook = Callable[[WarmupContext], Awaitable[None] | None]


async def _noop_warmup(context: WarmupContext) -> None:
    """Default warmup hook: does nothing (SM8 replaces it)."""
    return None


# ---------------------------------------------------------------------------
# Per-request token receipt seam (client-side counting is SM11 / O8)
# ---------------------------------------------------------------------------

#: Returns the monotonic receipt timestamps of a request's output tokens (one per
#: token), the ONE mechanism that feeds BOTH the energy denominator and the
#: stability gate's per-sub-window J/token. Client-side token counting is SM11's job
#: (O8: client counts are the canonical denominator). Two granularities are legal:
#:
#: - token-granular (one timestamp per token) -> counting per interval is
#:   span-received counting, the ratified energy-denominator rule;
#: - request-granular (all of a request's tokens stamped at its completion time,
#:   i.e. ``[completed_at] * n_tokens``) -> a request's whole token count falls in
#:   the interval containing its completion, which is EXACTLY E2's
#:   completion-timestamp attribution rule for sub-window J/token.
#:
#: Until SM11 lands the default returns no receipts, so denominators are 0 (the gate
#: reports invalid-with-reason and the energy denominator is 0). Tests inject fakes.
TokenReceiptFn = Callable[["RequestRecord"], Sequence[float]]


def _no_token_receipts(record: RequestRecord) -> Sequence[float]:
    return ()


# ---------------------------------------------------------------------------
# Dual boundary-policy bookkeeping (D7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowBoundaries:
    """The window's monotonic-clock boundaries (shared clock with SM5's issuer).

    ``window_start`` is when the level's load began; ``span_start`` is when this
    window's measured (energy) span opened; ``span_end`` is when it closed. Requests
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

    SM12 derives percentiles and goodput from these; SM7 only classifies.
    """

    boundaries: WindowBoundaries
    attribution_policy: str
    energy_denominator_tokens: int
    latency_records: list[LatencyRecord]
    issued_in_span_count: int
    completed_in_span_count: int
    straddling_count: int


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

    energy_tokens = _count_tokens_in_interval(
        report.records, token_receipt_fn, span_start, span_end, closed_hi=True
    )
    latency_records: list[LatencyRecord] = []
    issued_in_span = 0
    completed_in_span = 0
    straddling = 0

    for record in report.records:
        if not (span_start <= record.issued_at <= span_end):
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
# Per-level stability validation (E2 window-to-window J/token gate)
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
    """Validate a level's window-to-window J/token stability (the E2 gate).

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

    ``traffic_source`` and ``transport`` are pre-built by the caller (SM9) from the
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
    """One measured window's product within a level (one bundle per window, SM10)."""

    window_index: int
    boundaries: WindowBoundaries
    energy: MeasuredWindowCore | None
    bookkeeping: WindowBookkeeping
    window_energy_j: float | None
    window_j_per_token: float | None
    intra_window_cov: float | None
    start_event: WindowStartEvent
    stop_event: WindowStopEvent


@dataclass
class LevelOutcome:
    """Everything one rate level produced: its windows and the window-to-window verdict."""

    level_index: int
    spec: WindowSpec
    windows: list[WindowRecord]
    validation: LevelValidation
    issuer_report: IssuerReport


class WindowManager:
    """Drives a rate sweep as a list of levels, each a run of measured windows.

    Per level: run the warmup hook ONCE (SM8), start the open-loop traffic, exclude
    the ramp PROSPECTIVELY once, then run ``windows_per_level`` contiguous measured
    windows (no re-warm between them) - each emitting start-window (energy opens) and
    stop-window (energy closes) events - and finally drain the traffic to completion
    for latency. The energy window is defined by the emitted events, never by
    post-hoc timestamp diffing. Levels are validated on window-to-window J/token
    stability and separated by an optional cooldown.
    """

    def __init__(
        self,
        energy_sink: WindowEnergySink,
        *,
        windows_per_level: int = DEFAULT_WINDOWS_PER_LEVEL,
        cooldown_seconds: float = 0.0,
        warmup_hook: WarmupHook | None = None,
        drain_timeout: float | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if windows_per_level < 1:
            raise ValueError("windows_per_level must be >= 1.")
        if cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be >= 0.")
        self._energy_sink = energy_sink
        self._windows_per_level = windows_per_level
        self._cooldown_seconds = cooldown_seconds
        self._warmup_hook: WarmupHook = warmup_hook if warmup_hook is not None else _noop_warmup
        self._drain_timeout = drain_timeout
        self._sleep = sleep
        self._clock = clock

    async def run_levels(self, levels: Sequence[LevelPlan]) -> list[LevelOutcome]:
        """Run every level in order, cooling down between (not after the last)."""
        outcomes: list[LevelOutcome] = []
        last = len(levels) - 1
        for level_index, level in enumerate(levels):
            outcomes.append(await self.run_level(level_index, level))
            if level_index != last and self._cooldown_seconds > 0.0:
                await self._sleep(self._cooldown_seconds)
        return outcomes

    async def run_level(self, level_index: int, level: LevelPlan) -> LevelOutcome:
        """Run one level: warmup -> ramp -> N contiguous windows -> drain -> validate."""
        spec = level.spec
        if spec.duration_seconds is None:
            raise ValueError(
                "count-based measured windows (request_count without duration_seconds) "
                "are not supported at v0.7: the measured-span timing and the stability "
                "gate are duration-grounded (E2). Set a duration."
            )

        await self._run_warmup_hook(WarmupContext(level_index=level_index, spec=spec))

        window_start = self._clock()
        traffic_task: asyncio.Task[IssuerReport] = asyncio.create_task(
            level.traffic_source.run(level.transport, drain_timeout=self._drain_timeout)
        )
        emitted: list[tuple[WindowBoundaries, WindowStartEvent, WindowStopEvent, Any]] = []
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
                await self._sleep(spec.duration_seconds)
                span_end = self._clock()
                stop_event = WindowStopEvent(
                    level_index=level_index,
                    window_index=window_index,
                    spec=spec,
                    monotonic_at=span_end,
                )
                core = self._energy_sink.close_window(stop_event)
                boundaries = WindowBoundaries(
                    window_start=window_start, span_start=span_start, span_end=span_end
                )
                emitted.append((boundaries, start_event, stop_event, core))
            # Drain-before-close: energy has stopped; wait for every in-flight request
            # to complete so its latency record is captured.
            report = await traffic_task
        except BaseException:
            traffic_task.cancel()
            with contextlib.suppress(BaseException):
                await traffic_task
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
        emitted: list[tuple[WindowBoundaries, WindowStartEvent, WindowStopEvent, Any]],
        report: IssuerReport,
        token_receipt_fn: TokenReceiptFn,
        spec: WindowSpec,
    ) -> list[WindowRecord]:
        records: list[WindowRecord] = []
        for boundaries, start_event, stop_event, core in emitted:
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
                    start_event=start_event,
                    stop_event=stop_event,
                )
            )
        return records

    async def _run_warmup_hook(self, context: WarmupContext) -> None:
        result = self._warmup_hook(context)
        if inspect.isawaitable(result):
            await result


__all__ = [
    "ATTRIBUTION_STEADY_STATE_SPAN",
    "DEFAULT_WINDOWS_PER_LEVEL",
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
    "WindowRecord",
    "WindowSpec",
    "WindowStartEvent",
    "WindowStopEvent",
    "build_window_bookkeeping",
    "validate_level_stability",
]
