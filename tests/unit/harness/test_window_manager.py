"""Unit tests for the server-mode window object + multi-level window manager (SM7).

Host-only, no GPU, no real server: the traffic source, transport, and energy sink
are always injected fakes, and the clock/sleep are injected so the async
orchestration is deterministic and instant.

Charter (server-mode plan section 4, Wave 3 / SM7):
- window boundaries are EVENT-driven, not clock-diff (D19);
- the ramp is excluded PROSPECTIVELY (the measured span starts after it);
- the two boundary policies never collapse into one number (D7) - a
  boundary-straddling request appears in latency records yet contributes only its
  in-span tokens to the energy denominator;
- per-level stability uses the E2 thresholds via the reused windowing.py machinery;
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
from llenergymeasure.harness.traffic import IssuerReport, RequestRecord, RequestShape
from llenergymeasure.harness.window_manager import (
    ATTRIBUTION_STEADY_STATE_SPAN,
    BracketEnergySink,
    LevelPlan,
    WindowBoundaries,
    WindowManager,
    WindowSpec,
    WindowStartEvent,
    WindowStopEvent,
    build_window_bookkeeping,
    validate_level_stability,
)
from llenergymeasure.harness.windowing import _AUTO_CV_THRESHOLD

# ---------------------------------------------------------------------------
# Test doubles
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


class TracingEnergySink:
    """Records the ordered (kind, level_index) trace of emitted window events."""

    def __init__(self, trace: list[tuple[str, int]], core: Any = None) -> None:
        self._trace = trace
        self._core = core
        self.events: list[tuple[str, Any]] = []

    def open_window(self, event: WindowStartEvent) -> None:
        self._trace.append(("open", event.level_index))
        self.events.append(("open", event))

    def close_window(self, event: WindowStopEvent) -> Any:
        self._trace.append(("close", event.level_index))
        self.events.append(("close", event))
        return self._core


def _core_with_power(powers: list[float]) -> Any:
    """A MeasuredWindowCore-like duck holding a power series."""
    return SimpleNamespace(timeseries_samples=[SimpleNamespace(power_w=p) for p in powers])


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


def _rec(index: int, issued_at: float, completed_at: float | None) -> RequestRecord:
    return RequestRecord(
        index=index,
        issued_at=issued_at,
        request=RequestShape(index=index),
        completed_at=completed_at,
    )


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
        # v0.7 ships exactly one policy (disclosed, not configurable).
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
        # TrafficConfig already resolved the window default (neither field set).
        traffic = TrafficConfig(rate=2.0)
        spec = WindowSpec.from_traffic_config(traffic)
        assert spec.duration_seconds == DEFAULT_WINDOW_SECONDS
        assert spec.ramp_exclusion_seconds == DEFAULT_RAMP_EXCLUSION_SECONDS

    def test_from_traffic_config_count_window(self) -> None:
        traffic = TrafficConfig(rate=5.0, window_requests=500)
        spec = WindowSpec.from_traffic_config(traffic)
        assert spec.duration_seconds is None
        assert spec.request_count == 500


# ---------------------------------------------------------------------------
# Dual boundary-policy bookkeeping (D7): the two policies never collapse
# ---------------------------------------------------------------------------


class TestDualPolicyBookkeeping:
    # measured span = [1030, 1270] (window_start 1000 + ramp 30, duration 240).
    BOUNDARIES = WindowBoundaries(window_start=1000.0, span_start=1030.0, span_end=1270.0)

    def _fixture(self) -> tuple[IssuerReport, dict[int, list[float]]]:
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
        return _report(records), receipts

    def test_straddling_request_in_latency_but_only_in_span_tokens_in_energy(self) -> None:
        report, receipts = self._fixture()
        bk = build_window_bookkeeping(
            self.BOUNDARIES, report, token_receipt_fn=lambda r: receipts[r.index]
        )

        # LATENCY policy: the straddling request (index 2, issued in-span, completes
        # PAST span_end) has a full latency record via drain-before-close.
        straddler = next(r for r in bk.latency_records if r.index == 2)
        assert straddler.completed_at == 1300.0
        assert straddler.latency_s == pytest.approx(40.0)  # 1300 - 1260

        # ENERGY policy: it contributes only its ONE in-span token (1265), not the
        # 1290 token received after span_end.
        # Total energy denominator = ramp(1) + in-span(3) + straddler(1) + after(0).
        assert bk.energy_denominator_tokens == 5
        assert bk.attribution_policy == ATTRIBUTION_STEADY_STATE_SPAN

    def test_ramp_issued_request_feeds_energy_but_not_latency(self) -> None:
        # Double dissociation: the ramp-issued request (index 0) is NOT a latency
        # record (issued before span_start) yet its in-span token DOES count toward
        # the energy denominator (energy is receipt-based, independent of issue time).
        report, receipts = self._fixture()
        bk = build_window_bookkeeping(
            self.BOUNDARIES, report, token_receipt_fn=lambda r: receipts[r.index]
        )
        assert all(r.index != 0 for r in bk.latency_records)
        # index 0 contributed 1 token (1035 in-span) - included in the total of 5.

    def test_latency_membership_is_issued_in_span(self) -> None:
        report, receipts = self._fixture()
        bk = build_window_bookkeeping(
            self.BOUNDARIES, report, token_receipt_fn=lambda r: receipts[r.index]
        )
        # Only indices 1 and 2 were ISSUED within [1030, 1270].
        assert sorted(r.index for r in bk.latency_records) == [1, 2]
        assert bk.issued_in_span_count == 2
        assert bk.completed_in_span_count == 1  # only index 1 completed in-span
        assert bk.straddling_count == 1  # index 2

    def test_never_completed_request_is_straddling_with_null_latency(self) -> None:
        # A request issued in-span but never completed (drain-timeout cancellation)
        # keeps a latency record with a null latency and counts as straddling.
        report = _report([_rec(0, issued_at=1100.0, completed_at=None)])
        bk = build_window_bookkeeping(self.BOUNDARIES, report)
        assert len(bk.latency_records) == 1
        assert bk.latency_records[0].latency_s is None
        assert bk.straddling_count == 1
        assert bk.completed_in_span_count == 0

    def test_default_token_receipts_yield_zero_energy_denominator(self) -> None:
        # SM11 wires the real client-side counter; until then the default is empty.
        report, _ = self._fixture()
        bk = build_window_bookkeeping(self.BOUNDARIES, report)
        assert bk.energy_denominator_tokens == 0


# ---------------------------------------------------------------------------
# Per-level stability validation (E2 thresholds, reused windowing.py math)
# ---------------------------------------------------------------------------


class TestLevelValidation:
    def test_flat_series_is_valid(self) -> None:
        v = validate_level_stability([100.0] * 40)
        assert v.valid
        assert v.reason is None
        assert v.cov_max == pytest.approx(0.0)

    def test_rising_ramp_is_invalid_with_reason(self) -> None:
        v = validate_level_stability([float(x) for x in range(50, 250, 5)])
        assert not v.valid
        assert v.reason is not None
        assert "coefficient of variation" in v.reason

    def test_too_few_samples_is_invalid_not_pass(self) -> None:
        # Fewer than `consecutive` sub-windows must FAIL (never vacuously pass).
        v = validate_level_stability([100.0, 101.0])
        assert not v.valid
        assert v.reason is not None
        assert "insufficient" in v.reason
        assert v.subwindow_count == 2

    def test_threshold_is_the_reused_windowing_constant(self) -> None:
        # Just under 0.05 passes; a step just over 0.05 fails - proving the E2
        # threshold is the reused windowing.py constant, not a private copy.
        assert _AUTO_CV_THRESHOLD == 0.05
        below = validate_level_stability([100.0, 105.0, 100.0], subwindow_count=3)
        above = validate_level_stability([100.0, 120.0, 100.0], subwindow_count=3)
        assert below.valid and below.cov_max is not None and below.cov_max < 0.05
        assert not above.valid and above.cov_max is not None and above.cov_max > 0.05

    def test_stable_through_end_rejects_late_drift(self) -> None:
        # Stable at the start but drifting at the end must FAIL (the stable-through-end
        # rule reused from windowing.py).
        signals = [100.0] * 6 + [100.0, 140.0, 100.0]
        v = validate_level_stability(signals, subwindow_count=len(signals))
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


class TestBracketEnergySink:
    def test_open_enter_close_exit_finish_order(self) -> None:
        core = _core_with_power([100.0] * 10)
        bracket = FakeBracket(core)
        sink = BracketEnergySink(bracket_factory=lambda: bracket)

        spec = WindowSpec(rate=1.0)
        sink.open_window(WindowStartEvent(level_index=0, spec=spec, monotonic_at=1.0))
        assert bracket.log == ["enter"]
        returned = sink.close_window(WindowStopEvent(level_index=0, spec=spec, monotonic_at=2.0))
        assert bracket.log == ["enter", "exit", "finish"]
        assert returned is core

    def test_double_open_rejected(self) -> None:
        sink = BracketEnergySink(bracket_factory=lambda: FakeBracket(None))
        spec = WindowSpec(rate=1.0)
        sink.open_window(WindowStartEvent(level_index=0, spec=spec, monotonic_at=1.0))
        with pytest.raises(RuntimeError, match="already open"):
            sink.open_window(WindowStartEvent(level_index=0, spec=spec, monotonic_at=1.0))

    def test_close_without_open_rejected(self) -> None:
        sink = BracketEnergySink(bracket_factory=lambda: FakeBracket(None))
        spec = WindowSpec(rate=1.0)
        with pytest.raises(RuntimeError, match="no open window"):
            sink.close_window(WindowStopEvent(level_index=0, spec=spec, monotonic_at=2.0))

    def test_default_factory_builds_real_measurement_bracket(self) -> None:
        # C2 reuse: the production path mints a real MeasurementBracket (not entered
        # here - entering needs a GPU; we only assert the factory reuses the type).
        from llenergymeasure.harness.bracket import MeasurementBracket

        sink = BracketEnergySink.from_measurement_config(MeasurementConfig(), gpu_indices=None)
        bracket = sink._bracket_factory()
        assert isinstance(bracket, MeasurementBracket)


# ---------------------------------------------------------------------------
# WindowManager orchestration (async, event-driven)
# ---------------------------------------------------------------------------


def _level(spec: WindowSpec, report: IssuerReport, receipts: Any = None) -> LevelPlan:
    plan = LevelPlan(
        spec=spec,
        traffic_source=FakeTrafficSource(report),
        transport=FakeTransport(),
    )
    if receipts is not None:
        plan.token_receipt_fn = lambda r: receipts[r.index]
    return plan


class TestWindowManagerOrchestration:
    def test_prospective_ramp_and_event_boundaries(self) -> None:
        clock = FakeClock(start=1000.0)
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=240.0, ramp_exclusion_seconds=30.0)

        outcome = asyncio.run(manager.run_window(0, _level(spec, _report([]))))

        # The measured span STARTS after the ramp (prospective exclusion): span_start
        # = window_start + ramp; span_end = span_start + duration.
        b = outcome.bookkeeping.boundaries
        assert b.window_start == 1000.0
        assert b.span_start == 1030.0
        assert b.span_end == 1270.0
        # Boundaries are the EMITTED events, not a post-hoc timestamp diff (D19).
        assert outcome.start_event.monotonic_at == 1030.0
        assert outcome.stop_event.monotonic_at == 1270.0
        assert outcome.start_event.monotonic_at < outcome.stop_event.monotonic_at

    def test_event_ordering_open_before_close(self) -> None:
        clock = FakeClock()
        trace: list[tuple[str, int]] = []
        sink = TracingEnergySink(trace=trace, core=_core_with_power([100.0] * 40))
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)

        asyncio.run(manager.run_window(0, _level(spec, _report([]))))
        assert trace == [("open", 0), ("close", 0)]

    def test_traffic_source_seam_driven_exactly_once(self) -> None:
        clock = FakeClock()
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        level = _level(spec, _report([]))

        asyncio.run(manager.run_window(0, level))
        assert level.traffic_source.run_calls == 1

    def test_end_to_end_bookkeeping_and_validation(self) -> None:
        clock = FakeClock(start=1000.0)
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=240.0, ramp_exclusion_seconds=30.0)
        report = _report([_rec(1, issued_at=1100.0, completed_at=1150.0)])
        receipts = {1: [1110.0, 1120.0]}

        outcome = asyncio.run(manager.run_window(0, _level(spec, report, receipts)))
        assert outcome.bookkeeping.energy_denominator_tokens == 2
        assert outcome.bookkeeping.issued_in_span_count == 1
        assert outcome.validation.valid  # flat 100W series is steady
        assert outcome.issuer_report is report

    def test_count_based_window_rejected(self) -> None:
        clock = FakeClock()
        sink = TracingEnergySink(trace=[], core=None)
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=None, request_count=100)
        with pytest.raises(ValueError, match="count-based"):
            asyncio.run(manager.run_window(0, _level(spec, _report([]))))

    def test_failing_level_stamped_invalid_not_dropped(self) -> None:
        # A level whose measured span never steadies is stamped invalid-with-reason
        # and still RETURNED (never silently dropped).
        clock = FakeClock()
        rising = [float(x) for x in range(50, 250, 5)]
        sink = TracingEnergySink(trace=[], core=_core_with_power(rising))
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)

        outcome = asyncio.run(manager.run_window(0, _level(spec, _report([]))))
        assert not outcome.validation.valid
        assert outcome.validation.reason is not None


class TestMultiLevelAndCooldown:
    def _run_levels(self, cooldown: float) -> FakeClock:
        clock = FakeClock()
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        manager = WindowManager(sink, cooldown_seconds=cooldown, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        levels = [_level(spec, _report([])), _level(spec, _report([]))]
        outcomes = asyncio.run(manager.run_levels(levels))
        assert [o.level_index for o in outcomes] == [0, 1]
        return clock

    def test_cooldown_applied_between_levels_not_after_last(self) -> None:
        clock = self._run_levels(cooldown=5.0)
        # per level: ramp(1) + duration(2); cooldown(5) only BETWEEN the two levels.
        assert clock.sleeps == [1.0, 2.0, 5.0, 1.0, 2.0]

    def test_zero_cooldown_inserts_no_pause(self) -> None:
        clock = self._run_levels(cooldown=0.0)
        assert clock.sleeps == [1.0, 2.0, 1.0, 2.0]

    def test_negative_cooldown_rejected(self) -> None:
        sink = TracingEnergySink(trace=[], core=None)
        with pytest.raises(ValueError, match="cooldown_seconds"):
            WindowManager(sink, cooldown_seconds=-1.0)


class TestWarmupHookSeam:
    def test_noop_default_hook_runs_clean(self) -> None:
        clock = FakeClock()
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        manager = WindowManager(sink, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        # No warmup_hook passed: the no-op default must not error.
        outcome = asyncio.run(manager.run_window(0, _level(spec, _report([]))))
        assert outcome.level_index == 0

    def test_sync_hook_runs_before_window_opens_per_level(self) -> None:
        clock = FakeClock()
        trace: list[tuple[str, int]] = []
        sink = TracingEnergySink(trace=trace, core=_core_with_power([100.0] * 40))

        def hook(ctx: Any) -> None:
            trace.append(("warmup", ctx.level_index))

        manager = WindowManager(sink, warmup_hook=hook, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        levels = [_level(spec, _report([])), _level(spec, _report([]))]
        asyncio.run(manager.run_levels(levels))

        # Per level, warmup precedes open which precedes close; re-fires each level.
        assert trace == [
            ("warmup", 0),
            ("open", 0),
            ("close", 0),
            ("warmup", 1),
            ("open", 1),
            ("close", 1),
        ]

    def test_async_hook_is_awaited(self) -> None:
        clock = FakeClock()
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        seen: list[int] = []

        async def hook(ctx: Any) -> None:
            await asyncio.sleep(0)
            seen.append(ctx.level_index)

        manager = WindowManager(sink, warmup_hook=hook, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=10.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        asyncio.run(manager.run_window(3, _level(spec, _report([]))))
        assert seen == [3]

    def test_hook_receives_level_spec(self) -> None:
        clock = FakeClock()
        sink = TracingEnergySink(trace=[], core=_core_with_power([100.0] * 40))
        captured: list[Any] = []

        def hook(ctx: Any) -> None:
            captured.append(ctx.spec)

        manager = WindowManager(sink, warmup_hook=hook, sleep=clock.sleep, clock=clock)
        spec = WindowSpec(rate=42.0, duration_seconds=2.0, ramp_exclusion_seconds=1.0)
        asyncio.run(manager.run_window(0, _level(spec, _report([]))))
        assert captured == [spec]
        assert captured[0].rate == 42.0
