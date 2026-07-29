"""Unit tests for the server-mode window object + multi-level window manager (SM7).

Host-only, no GPU, no real server: the traffic source, transport, and energy sink
are always injected fakes, and the clock/sleep are injected so the async
orchestration is deterministic and instant.

Charter (server-mode plan section 4, Wave 3 / SM7, as re-ruled 2026-07-29):
- window boundaries are EVENT-driven, not clock-diff (D19);
- the ramp is excluded PROSPECTIVELY, once per level (the first window starts after
  it; subsequent windows are contiguous, no re-warm);
- the two boundary policies never collapse into one number (D7) - a
  boundary-straddling request appears in latency records yet contributes only its
  in-span tokens to the energy denominator;
- the stability gate is calibrated on J/TOKEN (not power): a per-window k=4
  sub-window J/token CoV (diagnostic) and a per-level window-to-window J/token gate
  over >= 3 consecutive windows, reusing windowing.py's CV / stable-through-end /
  clean / clip machinery and the trapezoidal integrator;
- MeasurementBracket is reused, not re-hardwired (C2/C5).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from llenergymeasure.config.models import (
    DEFAULT_RAMP_EXCLUSION_SECONDS,
    DEFAULT_WINDOW_SECONDS,
    MeasurementConfig,
    TrafficConfig,
)
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.traffic import IssuerReport, RequestRecord, RequestShape
from llenergymeasure.harness.window_manager import (
    ABORTED_LEVEL_ATTR,
    ATTRIBUTION_STEADY_STATE_SPAN,
    DEFAULT_WINDOWS_PER_LEVEL,
    BracketEnergySink,
    LevelPlan,
    WindowAbortEvent,
    WindowBoundaries,
    WindowManager,
    WindowSpec,
    WindowStartEvent,
    WindowStopEvent,
    _window_measurements,
    build_window_bookkeeping,
    validate_level_stability,
)
from llenergymeasure.harness.windowing import _AUTO_CV_THRESHOLD

# ---------------------------------------------------------------------------
# Test doubles + builders
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it and yields to the loop."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)  # let the concurrent traffic task run


class FakeTransport:
    async def __call__(self, request: Any) -> Any:  # pragma: no cover - never called by fakes
        return None


class FakeTrafficSource:
    """Returns a pre-built report; records that the seam's one method was driven."""

    def __init__(self, report: IssuerReport) -> None:
        self._report = report
        self.run_calls = 0

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        self.run_calls += 1
        await asyncio.sleep(0)
        return self._report


class RecordingEnergySink:
    """Records events and returns a fixed (possibly None) core per window."""

    def __init__(self, trace: list[tuple[str, int]] | None = None, core: Any = None) -> None:
        self._trace = trace if trace is not None else []
        self._core = core
        self.events: list[tuple[str, Any]] = []

    def open_window(self, event: WindowStartEvent) -> None:
        self._trace.append(("open", event.window_index))
        self.events.append(("open", event))

    def close_window(self, event: WindowStopEvent) -> Any:
        self._trace.append(("close", event.window_index))
        self.events.append(("close", event))
        return self._core

    def abort_window(self, event: WindowAbortEvent) -> None:
        self._trace.append(("abort", event.window_index))
        self.events.append(("abort", event))


class ProducingEnergySink:
    """Builds a flat-power core spanning each window's actual [span_start, span_end]."""

    def __init__(self, power_w: float = 100.0, samples: int = 41) -> None:
        self._power = power_w
        self._n = samples
        self._open_at: float | None = None

    def open_window(self, event: WindowStartEvent) -> None:
        self._open_at = event.monotonic_at

    def close_window(self, event: WindowStopEvent) -> Any:
        assert self._open_at is not None
        lo, hi = self._open_at, event.monotonic_at
        self._open_at = None
        return SimpleNamespace(timeseries_samples=_flat_samples(lo, hi, self._n, self._power))

    def abort_window(self, event: WindowAbortEvent) -> None:
        self._open_at = None


class AbortTrackingSink:
    """Records open/close/abort calls; can raise from close_window or abort_window."""

    def __init__(
        self,
        core: Any = None,
        *,
        abort_raises: bool = False,
        close_raises_on: int | None = None,
    ) -> None:
        self.core = core
        self._abort_raises = abort_raises
        self._close_raises_on = close_raises_on
        self.calls: list[tuple[str, int]] = []

    def open_window(self, event: WindowStartEvent) -> None:
        self.calls.append(("open", event.window_index))

    def close_window(self, event: WindowStopEvent) -> Any:
        self.calls.append(("close", event.window_index))
        if self._close_raises_on is not None and event.window_index == self._close_raises_on:
            raise RuntimeError("close teardown failed")
        return self.core

    def abort_window(self, event: WindowAbortEvent) -> None:
        self.calls.append(("abort", event.window_index))
        if self._abort_raises:
            raise RuntimeError("abort teardown failed")


class RaisingTrafficSource:
    """A traffic source whose run() raises - to exercise the drain-failure site."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        raise self._exc


class RaisingClock:
    """FakeClock that raises a chosen exception on the Nth sleep call."""

    def __init__(self, raise_on_call: int, exc: BaseException, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []
        self._raise_on = raise_on_call
        self._exc = exc
        self._calls = 0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self._calls += 1
        if self._calls == self._raise_on:
            raise self._exc
        self.sleeps.append(seconds)
        self.now += seconds
        await asyncio.sleep(0)


def _pts(ts: float, power: float = 100.0) -> PowerThermalSample:
    return PowerThermalSample(timestamp=ts, power_w=power, gpu_index=0)


def _flat_samples(
    lo: float, hi: float, n: int = 41, power: float = 100.0
) -> list[PowerThermalSample]:
    return [_pts(lo + (hi - lo) * i / (n - 1), power) for i in range(n)]


def _rec(index: int, issued_at: float, completed_at: float | None) -> RequestRecord:
    return RequestRecord(
        index=index,
        issued_at=issued_at,
        request=RequestShape(index=index),
        completed_at=completed_at,
    )


def _report(records: list[RequestRecord]) -> IssuerReport:
    completed = sum(1 for r in records if r.completed_at is not None)
    return IssuerReport(
        records=records,
        issued_count=len(records),
        completed_count=completed,
        cap_bound_fraction=0.0,
        issuance_duration_s=0.0,
        concurrency_cap=None,
    )


def _one_token_per_record(times: list[float]) -> tuple[IssuerReport, Any]:
    """A report with one request per token receipt time (issued at that time)."""
    records = [_rec(i, t, t + 0.01) for i, t in enumerate(times)]
    receipts = {i: [t] for i, t in enumerate(times)}
    return _report(records), (lambda r: receipts[r.index])


# ---------------------------------------------------------------------------
# WindowSpec - the first-class window object (D7)
# ---------------------------------------------------------------------------


class TestWindowSpec:
    def test_defaults_are_e2_values(self) -> None:
        spec = WindowSpec(rate=10.0)
        assert spec.duration_seconds == DEFAULT_WINDOW_SECONDS == 240.0
        assert spec.ramp_exclusion_seconds == DEFAULT_RAMP_EXCLUSION_SECONDS == 30.0
        assert spec.attribution_policy == ATTRIBUTION_STEADY_STATE_SPAN
        assert spec.request_count is None

    def test_single_attribution_policy_value(self) -> None:
        assert WindowSpec(rate=1.0).attribution_policy == "steady_state_span"

    def test_both_duration_and_count_rejected(self) -> None:
        with pytest.raises(ValueError, match="XOR"):
            WindowSpec(rate=10.0, duration_seconds=240.0, request_count=100)

    def test_count_only_leaves_duration_none(self) -> None:
        spec = WindowSpec(rate=10.0, duration_seconds=None, request_count=100)
        assert spec.duration_seconds is None
        assert spec.request_count == 100

    def test_neither_defaults_to_e2_duration(self) -> None:
        spec = WindowSpec(rate=10.0, duration_seconds=None, request_count=None)
        assert spec.duration_seconds == DEFAULT_WINDOW_SECONDS

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"rate": 0.0},
            {"rate": -1.0},
            {"rate": 10.0, "duration_seconds": 0.0},
            {"rate": 10.0, "duration_seconds": -5.0},
            {"rate": 10.0, "ramp_exclusion_seconds": -1.0},
            {"rate": 10.0, "duration_seconds": None, "request_count": 0},
        ],
    )
    def test_invalid_values_rejected(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            WindowSpec(**kwargs)

    def test_ramp_zero_allowed(self) -> None:
        assert WindowSpec(rate=10.0, ramp_exclusion_seconds=0.0).ramp_exclusion_seconds == 0.0

    def test_from_traffic_config_projects_fields(self) -> None:
        traffic = TrafficConfig(rate=7.0, window_seconds=120.0, ramp_exclusion_seconds=15.0)
        spec = WindowSpec.from_traffic_config(traffic)
        assert spec.rate == 7.0
        assert spec.duration_seconds == 120.0
        assert spec.ramp_exclusion_seconds == 15.0
        assert spec.request_count is None
        assert spec.arrival == "poisson"

    def test_from_traffic_config_carries_default_window(self) -> None:
        spec = WindowSpec.from_traffic_config(TrafficConfig(rate=2.0))
        assert spec.duration_seconds == DEFAULT_WINDOW_SECONDS
        assert spec.ramp_exclusion_seconds == DEFAULT_RAMP_EXCLUSION_SECONDS

    def test_from_traffic_config_carries_count_representation(self) -> None:
        # A bare TrafficConfig still admits window_requests (issuer feature); the
        # WindowSpec represents it (the manager rejects it, a server config rejects it).
        spec = WindowSpec.from_traffic_config(TrafficConfig(rate=5.0, window_requests=500))
        assert spec.duration_seconds is None
        assert spec.request_count == 500


# ---------------------------------------------------------------------------
# Dual boundary-policy bookkeeping (D7): the two policies never collapse
# ---------------------------------------------------------------------------


class TestDualPolicyBookkeeping:
    # measured span = [1030, 1270] (window_start 1000 + ramp 30, duration 240).
    BOUNDARIES = WindowBoundaries(window_start=1000.0, span_start=1030.0, span_end=1270.0)

    def _fixture(self) -> tuple[IssuerReport, Any]:
        records = [
            _rec(0, issued_at=1010.0, completed_at=1050.0),  # issued in RAMP
            _rec(1, issued_at=1100.0, completed_at=1150.0),  # fully in-span
            _rec(2, issued_at=1260.0, completed_at=1300.0),  # STRADDLES span_end
            _rec(3, issued_at=1280.0, completed_at=1290.0),  # issued AFTER span
        ]
        receipts = {
            0: [1035.0],  # ramp-issued but token received in-span
            1: [1110.0, 1120.0, 1130.0],  # all in-span
            2: [1265.0, 1290.0],  # one in-span, one after span_end
            3: [1285.0],  # after span_end
        }
        return _report(records), (lambda r: receipts[r.index])

    def test_straddling_request_in_latency_but_only_in_span_tokens_in_energy(self) -> None:
        report, receipt_fn = self._fixture()
        bk = build_window_bookkeeping(self.BOUNDARIES, report, token_receipt_fn=receipt_fn)

        straddler = next(r for r in bk.latency_records if r.index == 2)
        assert straddler.completed_at == 1300.0
        assert straddler.latency_s == pytest.approx(40.0)  # 1300 - 1260

        # Total energy denominator = ramp(1) + in-span(3) + straddler(1) + after(0).
        assert bk.energy_denominator_tokens == 5
        assert bk.attribution_policy == ATTRIBUTION_STEADY_STATE_SPAN

    def test_ramp_issued_request_feeds_energy_but_not_latency(self) -> None:
        report, receipt_fn = self._fixture()
        bk = build_window_bookkeeping(self.BOUNDARIES, report, token_receipt_fn=receipt_fn)
        assert all(r.index != 0 for r in bk.latency_records)

    def test_latency_membership_is_issued_in_span(self) -> None:
        report, receipt_fn = self._fixture()
        bk = build_window_bookkeeping(self.BOUNDARIES, report, token_receipt_fn=receipt_fn)
        assert sorted(r.index for r in bk.latency_records) == [1, 2]
        assert bk.issued_in_span_count == 2
        assert bk.completed_in_span_count == 1  # only index 1 completed in-span
        assert bk.straddling_count == 1  # index 2

    def test_never_completed_request_is_straddling_with_null_latency(self) -> None:
        report = _report([_rec(0, issued_at=1100.0, completed_at=None)])
        bk = build_window_bookkeeping(self.BOUNDARIES, report)
        assert len(bk.latency_records) == 1
        assert bk.latency_records[0].latency_s is None
        assert bk.straddling_count == 1
        assert bk.completed_in_span_count == 0

    def test_default_token_receipts_yield_zero_energy_denominator(self) -> None:
        report, _ = self._fixture()
        bk = build_window_bookkeeping(self.BOUNDARIES, report)
        assert bk.energy_denominator_tokens == 0


# ---------------------------------------------------------------------------
# J/token measurement: window J/token + intra-window CoV (reused integrator)
# ---------------------------------------------------------------------------


class TestWindowMeasurements:
    def test_flat_power_gives_expected_window_j_per_token_and_zero_intra_cov(self) -> None:
        # Window [0, 4]s, flat 100 W: window energy 400 J. One token per k=4 quarter:
        # each quarter 100 J / 1 token -> J/token 100, intra CoV 0. Window 400 J / 4
        # tokens -> 100 J/token.
        boundaries = WindowBoundaries(window_start=0.0, span_start=0.0, span_end=4.0)
        core = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 4.0, 41, 100.0))
        report, receipt_fn = _one_token_per_record([0.5, 1.5, 2.5, 3.5])

        energy_j, j_per_token, intra = _window_measurements(core, boundaries, report, receipt_fn, 4)
        assert energy_j == pytest.approx(400.0, rel=1e-6)
        assert j_per_token == pytest.approx(100.0, rel=1e-6)
        assert intra == pytest.approx(0.0, abs=1e-9)

    def test_request_granular_receipts_attribute_by_completion(self) -> None:
        # E2's completion-timestamp rule falls out of the ONE seam: a request whose
        # tokens are all stamped at its completion time lands entirely in the quarter
        # containing that completion. One 2-token request completing per quarter ->
        # each quarter 2 tokens, intra CoV 0.
        boundaries = WindowBoundaries(window_start=0.0, span_start=0.0, span_end=4.0)
        core = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 4.0, 41, 100.0))
        completions = [0.5, 1.5, 2.5, 3.5]
        records = [_rec(i, issued_at=t - 0.1, completed_at=t) for i, t in enumerate(completions)]
        receipts = {i: [t, t] for i, t in enumerate(completions)}  # 2 tokens, both at completion
        report = _report(records)

        _, j_per_token, intra = _window_measurements(
            core, boundaries, report, lambda r: receipts[r.index], 8
        )
        assert j_per_token == pytest.approx(400.0 / 8, rel=1e-6)
        assert intra == pytest.approx(0.0, abs=1e-9)

    def test_intra_cov_none_when_a_subwindow_has_no_tokens(self) -> None:
        boundaries = WindowBoundaries(window_start=0.0, span_start=0.0, span_end=4.0)
        core = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 4.0, 41, 100.0))
        # Tokens only in q0 and q3 -> q1/q2 empty -> intra unformable (None), but the
        # window-level J/token is still computed.
        report, receipt_fn = _one_token_per_record([0.5, 3.5])
        energy_j, j_per_token, intra = _window_measurements(core, boundaries, report, receipt_fn, 2)
        assert energy_j == pytest.approx(400.0, rel=1e-6)
        assert j_per_token == pytest.approx(200.0, rel=1e-6)
        assert intra is None

    def test_zero_window_tokens_gives_none_j_per_token(self) -> None:
        boundaries = WindowBoundaries(window_start=0.0, span_start=0.0, span_end=4.0)
        core = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 4.0, 41, 100.0))
        energy_j, j_per_token, _ = _window_measurements(core, boundaries, _report([]), None, 0)
        assert energy_j == pytest.approx(400.0, rel=1e-6)
        assert j_per_token is None

    def test_no_core_or_too_few_samples_gives_all_none(self) -> None:
        boundaries = WindowBoundaries(window_start=0.0, span_start=0.0, span_end=4.0)
        assert _window_measurements(None, boundaries, _report([]), None, 5) == (None, None, None)
        one = SimpleNamespace(timeseries_samples=[_pts(0.0)])
        assert _window_measurements(one, boundaries, _report([]), None, 5) == (None, None, None)


# ---------------------------------------------------------------------------
# Per-level stability validation (E2 window-to-window J/token gate)
# ---------------------------------------------------------------------------


class TestLevelValidation:
    def test_three_equal_window_j_per_token_is_valid(self) -> None:
        v = validate_level_stability([100.0, 100.0, 100.0])
        assert v.valid
        assert v.reason is None
        assert v.cov == pytest.approx(0.0)
        assert v.windows_considered == 3

    def test_deviating_window_is_invalid_with_reason(self) -> None:
        v = validate_level_stability([100.0, 100.0, 140.0])
        assert not v.valid
        assert v.reason is not None
        assert "coefficient of variation" in v.reason

    def test_fewer_than_three_windows_is_invalid(self) -> None:
        v = validate_level_stability([100.0, 100.0])
        assert not v.valid
        assert v.reason is not None
        assert "at least 3" in v.reason
        assert v.windows_considered == 2

    def test_a_window_with_no_j_per_token_is_invalid(self) -> None:
        v = validate_level_stability([100.0, None, 100.0])
        assert not v.valid
        assert v.reason is not None
        assert "no J/token" in v.reason

    def test_threshold_is_the_reused_windowing_constant(self) -> None:
        assert _AUTO_CV_THRESHOLD == 0.05
        below = validate_level_stability([100.0, 105.0, 100.0])
        above = validate_level_stability([100.0, 120.0, 100.0])
        assert below.valid and below.cov is not None and below.cov < 0.05
        assert not above.valid and above.cov is not None and above.cov > 0.05

    def test_stable_through_end_rejects_late_drift(self) -> None:
        # Steady early, drifting at the end: the last 3-consecutive group fails.
        v = validate_level_stability([100.0, 100.0, 100.0, 140.0, 100.0])
        assert not v.valid


# ---------------------------------------------------------------------------
# BracketEnergySink - reuses MeasurementBracket (C2)
# ---------------------------------------------------------------------------


class FakeBracket:
    def __init__(self, core: Any) -> None:
        self.core = core
        self.log: list[str] = []

    def __enter__(self) -> FakeBracket:
        self.log.append("enter")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.log.append("exit")

    def finish(self) -> Any:
        self.log.append("finish")
        return self.core


def _start(level: int = 0, window: int = 0, at: float = 1.0) -> WindowStartEvent:
    return WindowStartEvent(
        level_index=level, window_index=window, spec=WindowSpec(rate=1.0), monotonic_at=at
    )


def _stop(level: int = 0, window: int = 0, at: float = 2.0) -> WindowStopEvent:
    return WindowStopEvent(
        level_index=level, window_index=window, spec=WindowSpec(rate=1.0), monotonic_at=at
    )


class TestBracketEnergySink:
    def test_open_enter_close_exit_finish_order(self) -> None:
        core: Any = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 1.0, 5))
        bracket = FakeBracket(core)
        sink = BracketEnergySink(bracket_factory=lambda: bracket)

        sink.open_window(_start())
        assert bracket.log == ["enter"]
        returned = sink.close_window(_stop())
        assert bracket.log == ["enter", "exit", "finish"]
        assert returned is core

    def test_double_open_rejected(self) -> None:
        sink = BracketEnergySink(bracket_factory=lambda: FakeBracket(None))
        sink.open_window(_start())
        with pytest.raises(RuntimeError, match="already open"):
            sink.open_window(_start())

    def test_close_without_open_rejected(self) -> None:
        sink = BracketEnergySink(bracket_factory=lambda: FakeBracket(None))
        with pytest.raises(RuntimeError, match="no open window"):
            sink.close_window(_stop())

    def test_fresh_bracket_per_window(self) -> None:
        # One bundle per window (SM10): a new bracket is minted for each window.
        built: list[FakeBracket] = []

        def factory() -> FakeBracket:
            b = FakeBracket(None)
            built.append(b)
            return b

        sink = BracketEnergySink(bracket_factory=factory)
        sink.open_window(_start(window=0))
        sink.close_window(_stop(window=0))
        sink.open_window(_start(window=1))
        sink.close_window(_stop(window=1))
        assert len(built) == 2

    def test_default_factory_builds_real_measurement_bracket(self) -> None:
        from llenergymeasure.harness.bracket import MeasurementBracket

        sink = BracketEnergySink.from_measurement_config(MeasurementConfig(), gpu_indices=None)
        assert isinstance(sink._bracket_factory(), MeasurementBracket)


# ---------------------------------------------------------------------------
# WindowManager orchestration (async, event-driven)
# ---------------------------------------------------------------------------


def _level(
    spec: WindowSpec, report: IssuerReport, receipt_fn: Any = None, source: Any = None
) -> LevelPlan:
    src = source if source is not None else FakeTrafficSource(report)
    plan = LevelPlan(spec=spec, traffic_source=src, transport=FakeTransport())
    if receipt_fn is not None:
        plan.token_receipt_fn = receipt_fn
    return plan


class TestWindowManagerOrchestration:
    def test_prospective_ramp_and_contiguous_windows(self) -> None:
        clock = FakeClock(start=1000.0)
        manager = WindowManager(
            RecordingEnergySink(core=None), windows_per_level=3, sleep=clock.sleep, clock=clock
        )
        spec = WindowSpec(rate=10.0, duration_seconds=10.0, ramp_exclusion_seconds=30.0)

        outcome = asyncio.run(manager.run_level(0, _level(spec, _report([]))))

        # Ramp excluded once (prospective): window 0 opens at window_start + ramp.
        b0 = outcome.windows[0].boundaries
        assert b0.window_start == 1000.0
        assert b0.span_start == 1030.0
        assert b0.span_end == 1040.0
        # Windows are contiguous (no re-warm): each opens where the last closed.
        spans = [(w.boundaries.span_start, w.boundaries.span_end) for w in outcome.windows]
        assert spans == [(1030.0, 1040.0), (1040.0, 1050.0), (1050.0, 1060.0)]

    def test_events_are_ordered_and_indexed(self) -> None:
        clock = FakeClock()
        trace: list[tuple[str, int]] = []
        manager = WindowManager(
            RecordingEnergySink(trace=trace, core=None),
            windows_per_level=3,
            sleep=clock.sleep,
            clock=clock,
        )
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        asyncio.run(manager.run_level(0, _level(spec, _report([]))))
        assert trace == [
            ("open", 0),
            ("close", 0),
            ("open", 1),
            ("close", 1),
            ("open", 2),
            ("close", 2),
        ]

    def test_default_windows_per_level(self) -> None:
        clock = FakeClock()
        manager = WindowManager(RecordingEnergySink(core=None), sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        outcome = asyncio.run(manager.run_level(0, _level(spec, _report([]))))
        assert len(outcome.windows) == DEFAULT_WINDOWS_PER_LEVEL == 3

    def test_traffic_source_driven_once_per_level(self) -> None:
        clock = FakeClock()
        manager = WindowManager(RecordingEnergySink(core=None), sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        level = _level(spec, _report([]))
        asyncio.run(manager.run_level(0, level))
        assert level.traffic_source.run_calls == 1

    def test_end_to_end_valid_level(self) -> None:
        # Flat power + equal tokens per window -> equal window J/token -> level valid.
        clock = FakeClock(start=1000.0)
        manager = WindowManager(
            ProducingEnergySink(power_w=100.0), windows_per_level=3, sleep=clock.sleep, clock=clock
        )
        spec = WindowSpec(rate=10.0, duration_seconds=10.0, ramp_exclusion_seconds=30.0)
        # 3 tokens per window across the known fake-clock spans.
        times = [1032, 1035, 1038, 1042, 1045, 1048, 1052, 1055, 1058]
        report, receipt_fn = _one_token_per_record([float(t) for t in times])

        outcome = asyncio.run(manager.run_level(0, _level(spec, report, receipt_fn)))
        assert [w.window_j_per_token for w in outcome.windows] == pytest.approx([1000.0 / 3] * 3)
        assert outcome.validation.valid
        assert outcome.issuer_report is report

    def test_end_to_end_invalid_level_is_stamped_not_dropped(self) -> None:
        # Same energy per window but a token spike in window 2 -> its J/token drops
        # far below the others -> window-to-window gate fails, stamped with a reason.
        clock = FakeClock(start=1000.0)
        manager = WindowManager(
            ProducingEnergySink(power_w=100.0), windows_per_level=3, sleep=clock.sleep, clock=clock
        )
        spec = WindowSpec(rate=10.0, duration_seconds=10.0, ramp_exclusion_seconds=30.0)
        # windows 0/1: 3 tokens; window 2: 12 tokens -> J/token far lower in window 2.
        times = [1032, 1035, 1038, 1042, 1045, 1048] + [1050 + 0.5 * i for i in range(12)]
        report, receipt_fn = _one_token_per_record([float(t) for t in times])

        outcome = asyncio.run(manager.run_level(0, _level(spec, report, receipt_fn)))
        assert not outcome.validation.valid
        assert outcome.validation.reason is not None
        assert len(outcome.windows) == 3  # not dropped

    def test_count_based_window_rejected(self) -> None:
        clock = FakeClock()
        manager = WindowManager(RecordingEnergySink(core=None), sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=None, request_count=100)
        with pytest.raises(ValueError, match="count-based"):
            asyncio.run(manager.run_level(0, _level(spec, _report([]))))

    def test_invalid_windows_per_level_rejected(self) -> None:
        with pytest.raises(ValueError, match="windows_per_level"):
            WindowManager(RecordingEnergySink(core=None), windows_per_level=0)


class TestMultiLevelAndCooldown:
    def _run_two_levels(self, cooldown: float) -> FakeClock:
        clock = FakeClock()
        manager = WindowManager(
            RecordingEnergySink(core=None),
            windows_per_level=3,
            cooldown_seconds=cooldown,
            sleep=clock.sleep,
            clock=clock,
        )
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        levels = [_level(spec, _report([])), _level(spec, _report([]))]
        outcomes = asyncio.run(manager.run_levels(levels))
        assert [o.level_index for o in outcomes] == [0, 1]
        return clock

    def test_cooldown_between_levels_not_after_last(self) -> None:
        clock = self._run_two_levels(cooldown=5.0)
        # per level: ramp(1) + 3 x duration(2); cooldown(5) once, between the levels.
        one_level = [1.0, 2.0, 2.0, 2.0]
        assert clock.sleeps == [*one_level, 5.0, *one_level]

    def test_zero_cooldown_inserts_no_pause(self) -> None:
        clock = self._run_two_levels(cooldown=0.0)
        one_level = [1.0, 2.0, 2.0, 2.0]
        assert clock.sleeps == [*one_level, *one_level]

    def test_negative_cooldown_rejected(self) -> None:
        with pytest.raises(ValueError, match="cooldown_seconds"):
            WindowManager(RecordingEnergySink(core=None), cooldown_seconds=-1.0)


class TestWarmupHookSeam:
    def test_noop_default_hook_runs_clean(self) -> None:
        clock = FakeClock()
        manager = WindowManager(RecordingEnergySink(core=None), sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        outcome = asyncio.run(manager.run_level(0, _level(spec, _report([]))))
        assert outcome.level_index == 0

    def test_sync_hook_runs_once_per_level_before_windows(self) -> None:
        clock = FakeClock()
        trace: list[tuple[str, int]] = []

        def hook(ctx: Any) -> None:
            trace.append(("warmup", ctx.level_index))

        manager = WindowManager(
            RecordingEnergySink(trace=trace, core=None),
            windows_per_level=2,
            warmup_hook=hook,
            sleep=clock.sleep,
            clock=clock,
        )
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        levels = [_level(spec, _report([])), _level(spec, _report([]))]
        asyncio.run(manager.run_levels(levels))
        # Warmup fires ONCE per level, before that level's windows open.
        assert trace == [
            ("warmup", 0),
            ("open", 0),
            ("close", 0),
            ("open", 1),
            ("close", 1),
            ("warmup", 1),
            ("open", 0),
            ("close", 0),
            ("open", 1),
            ("close", 1),
        ]

    def test_async_hook_is_awaited(self) -> None:
        clock = FakeClock()
        seen: list[int] = []

        async def hook(ctx: Any) -> None:
            await asyncio.sleep(0)
            seen.append(ctx.level_index)

        manager = WindowManager(
            RecordingEnergySink(core=None), warmup_hook=hook, sleep=clock.sleep, clock=clock
        )
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        asyncio.run(manager.run_level(3, _level(spec, _report([]))))
        assert seen == [3]

    def test_hook_receives_level_spec(self) -> None:
        clock = FakeClock()
        captured: list[Any] = []

        def hook(ctx: Any) -> None:
            captured.append(ctx.spec)

        manager = WindowManager(
            RecordingEnergySink(core=None), warmup_hook=hook, sleep=clock.sleep, clock=clock
        )
        spec = WindowSpec(rate=42.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        asyncio.run(manager.run_level(0, _level(spec, _report([]))))
        assert captured == [spec]
        assert captured[0].rate == 42.0


async def _capture_abort(coro: Any, exc_type: type[BaseException]) -> BaseException:
    """Await ``coro`` and return the ``exc_type`` it raises (same instance).

    Catching at this level (rather than via asyncio.run's outer Task boundary)
    preserves the raised instance and any attached AbortedLevel - exactly how an
    immediate awaiter (SM9) sees it.
    """
    try:
        await coro
    except exc_type as exc:
        return exc
    raise AssertionError("expected the level to abort, but it completed")


class TestAbortReleasesOnError:
    """Release-on-error at the owner: an abort during an OPEN window frees the sink."""

    def _spec(self) -> WindowSpec:
        return WindowSpec(rate=10.0, duration_seconds=10.0, ramp_exclusion_seconds=30.0)

    def test_cancellation_mid_span_aborts_once_and_propagates(self) -> None:
        # Raise CancelledError on sleep call 2 (window 0's measured span) while open.
        clock = RaisingClock(raise_on_call=2, exc=asyncio.CancelledError())
        sink = AbortTrackingSink()
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)

        exc = asyncio.run(
            _capture_abort(
                manager.run_level(0, _level(self._spec(), _report([]))), asyncio.CancelledError
            )
        )
        # Exactly one abort for the open window, and it was never also closed.
        assert sink.calls == [("open", 0), ("abort", 0)]
        aborted = getattr(exc, ABORTED_LEVEL_ATTR)
        assert aborted.aborted_window_index == 0
        assert aborted.reason == "aborted: cancelled"
        assert aborted.completed_cores == []

    def test_plain_exception_mid_span_aborts_once_and_propagates(self) -> None:
        clock = RaisingClock(raise_on_call=2, exc=ValueError("boom"))
        sink = AbortTrackingSink()
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)

        exc = asyncio.run(
            _capture_abort(manager.run_level(0, _level(self._spec(), _report([]))), ValueError)
        )
        assert sink.calls == [("open", 0), ("abort", 0)]
        aborted = getattr(exc, ABORTED_LEVEL_ATTR)
        assert aborted.reason.startswith("aborted:")
        assert "boom" in aborted.reason

    def test_abort_raising_does_not_mask_or_double_close(self) -> None:
        # The sink's abort_window raises; the ORIGINAL CancelledError must still
        # propagate (not the abort's RuntimeError), and the window is not also closed.
        clock = RaisingClock(raise_on_call=2, exc=asyncio.CancelledError())
        sink = AbortTrackingSink(abort_raises=True)
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)

        exc = asyncio.run(
            _capture_abort(
                manager.run_level(0, _level(self._spec(), _report([]))), asyncio.CancelledError
            )
        )
        assert isinstance(exc, asyncio.CancelledError)
        assert sink.calls == [("open", 0), ("abort", 0)]  # abort attempted once, no close
        assert hasattr(exc, ABORTED_LEVEL_ATTR)  # partial state still attached

    def test_completed_windows_preserved_on_abort(self) -> None:
        # Abort during window 1's span (sleep call 3): window 0 completed normally,
        # so its core is preserved on the AbortedLevel; window 1 is the aborted one.
        sentinel = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 1.0, 5))
        clock = RaisingClock(raise_on_call=3, exc=asyncio.CancelledError())
        sink = AbortTrackingSink(core=sentinel)
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)

        exc = asyncio.run(
            _capture_abort(
                manager.run_level(0, _level(self._spec(), _report([]))), asyncio.CancelledError
            )
        )
        assert sink.calls == [("open", 0), ("close", 0), ("open", 1), ("abort", 1)]
        aborted = getattr(exc, ABORTED_LEVEL_ATTR)
        assert aborted.aborted_window_index == 1
        assert aborted.completed_cores == [sentinel]

    def test_happy_path_emits_no_abort(self) -> None:
        clock = FakeClock()
        sink = AbortTrackingSink(core=None)
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        asyncio.run(manager.run_level(0, _level(spec, _report([]))))
        assert not any(kind == "abort" for kind, _ in sink.calls)
        assert sink.calls == [
            ("open", 0),
            ("close", 0),
            ("open", 1),
            ("close", 1),
            ("open", 2),
            ("close", 2),
        ]

    def test_drain_failure_after_clean_windows_preserves_all_cores(self) -> None:
        # All 3 windows close cleanly, then the post-measurement drain raises: the
        # cores stand (drain-failed site), no abort event, no double-close.
        clock = FakeClock()
        sentinel = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 1.0, 5))
        sink = AbortTrackingSink(core=sentinel)
        source = RaisingTrafficSource(RuntimeError("transport gone"))
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)

        exc = asyncio.run(
            _capture_abort(
                manager.run_level(0, _level(spec, _report([]), source=source)), RuntimeError
            )
        )
        assert not any(kind == "abort" for kind, _ in sink.calls)
        assert [k for k, _ in sink.calls] == ["open", "close", "open", "close", "open", "close"]
        aborted = getattr(exc, ABORTED_LEVEL_ATTR)
        assert aborted.aborted_window_index is None
        assert aborted.reason.startswith("drain failed:")
        assert aborted.completed_cores == [sentinel, sentinel, sentinel]

    def test_cancellation_during_drain_stays_cancelled_with_cores(self) -> None:
        clock = FakeClock()
        sentinel = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 1.0, 5))
        sink = AbortTrackingSink(core=sentinel)
        source = RaisingTrafficSource(asyncio.CancelledError())
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)

        exc = asyncio.run(
            _capture_abort(
                manager.run_level(0, _level(spec, _report([]), source=source)),
                asyncio.CancelledError,
            )
        )
        assert isinstance(exc, asyncio.CancelledError)
        aborted = getattr(exc, ABORTED_LEVEL_ATTR)
        assert aborted.reason == "drain failed: cancelled"
        assert aborted.completed_cores == [sentinel, sentinel, sentinel]
        assert not any(kind == "abort" for kind, _ in sink.calls)

    def test_close_window_raising_preserves_prior_cores_without_abort(self) -> None:
        # Windows 0, 1 close cleanly; window 2's close_window raises: it is the failed
        # window (no abort event), and the two prior cores are preserved.
        clock = FakeClock()
        sentinel = SimpleNamespace(timeseries_samples=_flat_samples(0.0, 1.0, 5))
        sink = AbortTrackingSink(core=sentinel, close_raises_on=2)
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)

        exc = asyncio.run(
            _capture_abort(manager.run_level(0, _level(spec, _report([]))), RuntimeError)
        )
        assert not any(kind == "abort" for kind, _ in sink.calls)  # no abort for the failed close
        aborted = getattr(exc, ABORTED_LEVEL_ATTR)
        assert aborted.aborted_window_index == 2
        assert aborted.reason.startswith("close failed:")
        assert aborted.completed_cores == [sentinel, sentinel]  # windows 0 and 1 only

    def test_ramp_phase_failure_attaches_nothing(self) -> None:
        # Failure before any window opens (ramp sleep, call 1): nothing to preserve.
        clock = RaisingClock(raise_on_call=1, exc=ValueError("ramp boom"))
        sink = AbortTrackingSink()
        manager = WindowManager(sink, windows_per_level=3, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)

        exc = asyncio.run(
            _capture_abort(manager.run_level(0, _level(spec, _report([]))), ValueError)
        )
        assert not hasattr(exc, ABORTED_LEVEL_ATTR)
        assert sink.calls == []
