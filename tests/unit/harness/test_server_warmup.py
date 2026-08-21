"""Tests for the server-mode warmup protocol: the three-observable composite
gate, the fixed opt-out, the timeout failsafe, per-level re-warm, the readiness
probe request shape, and the divergence-labeling protocol descriptions.

Covers: warmup draws from the (injected) MEASURED traffic
shape; NO pre-window idle cooldown; re-warm fires at every level; timeout stamps
timed_out; each composite observable gates independently; the gate is evaluated on a
1 Hz view of the power series, off the issuing loop, so neither the warmup traffic nor
the failsafe deadline can be starved by a slow evaluation.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import threading
import time
from dataclasses import replace
from typing import Any

import pytest

from llenergymeasure.config.models import ServerWarmupConfig
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness import server_warmup
from llenergymeasure.harness.server_warmup import (
    ObservableState,
    ServerWarmup,
    ServerWarmupResult,
    WarmupTrafficError,
    _evaluate_observables,
    _gate_view,
    _power_plateau,
    _temperature_settled,
    _throttle_clear,
    build_probe_request,
    describe_server_warmup_protocol,
)
from llenergymeasure.harness.traffic import IssuerReport
from llenergymeasure.harness.window_manager import (
    LevelPlan,
    WarmupContext,
    WindowAbortEvent,
    WindowManager,
    WindowSpec,
    WindowStartEvent,
    WindowStopEvent,
)
from llenergymeasure.serving.transport import RequestShape

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _series(
    *,
    n: int,
    dt: float = 1.0,
    power: float | None = 300.0,
    temp: float | None = 55.0,
    throttle: bool = False,
    rising: bool = False,
    gpu: int = 0,
    t0: float = 1000.0,
) -> list[PowerThermalSample]:
    """A monotone-time sample series (sampler perf_counter clock)."""
    out: list[PowerThermalSample] = []
    for i in range(n):
        # A steep ramp (slope 20) so a sliding-window CoV clears the 0.05 plateau
        # threshold - a gentle drift stays "stable" by design.
        p = (100.0 + i * 20.0) if (rising and power is not None) else power
        out.append(
            PowerThermalSample(
                timestamp=t0 + i * dt,
                power_w=p,
                temperature_c=temp,
                gpu_index=gpu,
                thermal_throttle=throttle,
            )
        )
    return out


class FakeSampler:
    """Returns ``samples`` after ``ready_after`` polls (empty before).

    Mirrors production: the sampler starts EMPTY, so the gate cannot pass on the
    first poll - the warmup loop sleeps (letting the traffic task run) and re-polls.
    ``stop_count`` lets tests assert the sampler is released exactly once.
    """

    def __init__(self, samples: list[PowerThermalSample], *, ready_after: int = 0) -> None:
        self._samples = samples
        self._ready_after = ready_after
        self._polls = 0
        self.started = False
        self.stopped = False
        self.stop_count = 0

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.stop_count += 1

    def get_samples(self) -> list[PowerThermalSample]:
        polls = self._polls
        self._polls += 1
        return self._samples if polls >= self._ready_after else []


def _with_throttle(
    samples: list[PowerThermalSample], *, lo: float, hi: float
) -> list[PowerThermalSample]:
    """Set the throttle bit on the samples whose timestamp falls in ``[lo, hi]``."""
    return [replace(s, thermal_throttle=lo <= s.timestamp <= hi) for s in samples]


def _empty_report(issued: int = 0) -> IssuerReport:
    return IssuerReport(
        records=[],
        issued_count=issued,
        completed_count=issued,
        cap_bound_fraction=0.0,
        issuance_duration_s=0.0,
        concurrency_cap=None,
    )


class FakeSource:
    """Issues requests through the transport until cancelled.

    Mimics the real open-loop issuer: a transport call that raises is caught as
    bookkeeping (it does NOT kill the source), so a traffic-alive-but-all-failing
    server is expressible by pairing this with a failing transport.
    """

    def __init__(self) -> None:
        self.started = False
        self.cancelled = False
        self.issued = 0

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        self.started = True
        try:
            while True:
                with contextlib.suppress(Exception):
                    await transport(RequestShape(index=self.issued))
                self.issued += 1
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class RaisingSource:
    """The traffic source itself fails (not the transport): run() raises."""

    def __init__(self, exc: BaseException | None = None) -> None:
        self.started = False
        self._exc = exc or RuntimeError("traffic source boom")

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        self.started = True
        raise self._exc


class RaiseAfterSource:
    """Issues ``n`` successful requests, then raises mid-warmup."""

    def __init__(self, n: int = 3) -> None:
        self._n = n

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        for i in range(self._n):
            await transport(RequestShape(index=i))
            await asyncio.sleep(0.001)
        raise RuntimeError("traffic died mid-warmup")


class EarlyExitSource:
    """Issues ``n`` successful requests, then returns cleanly (schedule exhausted)."""

    def __init__(self, n: int = 3) -> None:
        self._n = n

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        for i in range(self._n):
            await transport(RequestShape(index=i))
            await asyncio.sleep(0.001)
        return _empty_report(self._n)


class ExhaustAfterSource:
    """Issues ``n`` successful requests then returns cleanly, with NO trailing delay.

    Unlike ``EarlyExitSource`` it never awaits a real timer, so under a fake-clock
    driver it runs to completion on the poll loop's first yield - letting a test
    place the clean schedule exhaustion at a controlled point relative to the
    deadline (at the boundary vs. strictly before it).
    """

    def __init__(self, n: int = 1) -> None:
        self._n = n
        self.issued = 0

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        for i in range(self._n):
            await transport(RequestShape(index=i))
            self.issued += 1
        return _empty_report(self.issued)


class FakeTransport:
    async def __call__(self, request: Any) -> Any:
        return None


class AllFailTransport:
    """Raises on every call: the server is reachable but every request fails."""

    async def __call__(self, request: Any) -> Any:
        raise RuntimeError("request failed")


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _ctx(level: int = 0, rate: float = 10.0) -> WarmupContext:
    return WarmupContext(level_index=level, spec=WindowSpec(rate=rate))


# A settled equilibrium series (flat power, cool, unthrottled, >90s of history).
_SETTLED = _series(n=120)


# ---------------------------------------------------------------------------
# Observables (each computed from one poll; each gates independently)
# ---------------------------------------------------------------------------


class TestPowerPlateau:
    def test_flat_series_plateaued(self):
        assert _power_plateau(_SETTLED) is True

    def test_rising_series_not_plateaued(self):
        assert _power_plateau(_series(n=120, rising=True)) is False

    def test_too_few_samples(self):
        assert _power_plateau(_series(n=4)) is False


class TestTemperatureSettled:
    def test_settled_when_flat_over_90s(self):
        assert _temperature_settled(_SETTLED) is True

    def test_not_settled_without_90s_history(self):
        # Only 30s of history - below the structural loaded-observation floor.
        assert _temperature_settled(_series(n=30)) is False

    def test_not_settled_when_drifting(self):
        drifting = [
            PowerThermalSample(timestamp=1000.0 + i, power_w=300.0, temperature_c=50.0 + i * 0.1)
            for i in range(120)
        ]
        assert _temperature_settled(drifting) is False

    def test_per_gpu_one_gpu_drifting_blocks(self):
        cool = _series(n=120, gpu=0, temp=55.0)
        hot = [
            PowerThermalSample(timestamp=1000.0 + i, temperature_c=50.0 + i * 0.1, gpu_index=1)
            for i in range(120)
        ]
        assert _temperature_settled(cool + hot) is False

    def test_cross_gpu_spread_is_not_drift(self):
        # Two GPUs each individually flat (55C and 65C) - settled despite the spread.
        a = _series(n=120, gpu=0, temp=55.0)
        b = _series(n=120, gpu=1, temp=65.0)
        assert _temperature_settled(a + b) is True


class TestThrottleClear:
    def test_clear_when_no_recent_throttle(self):
        assert _throttle_clear(_SETTLED) is True

    def test_active_throttle_vetoes(self):
        assert _throttle_clear(_series(n=120, throttle=True)) is False

    def test_only_old_throttle_is_clear(self):
        # A cold-start throttle long ago, clear in the trailing window.
        old = _series(n=60, throttle=True, t0=1000.0)
        recent = _series(n=60, throttle=False, t0=1060.0)
        assert _throttle_clear(old + recent) is True

    def test_empty_not_proven_clear(self):
        assert _throttle_clear([]) is False


class TestGateView:
    """The POWER PLATEAU observable reads a 1 Hz view; nothing else does.

    Steady-state detection costs super-quadratically in the sample count, so the
    plateau must be given a bounded view of a series that grows for as long as warmup
    lasts. The temperature and throttle observables are extreme-value statistics over
    a trailing window, so decimating them would lose short-lived extremes that must
    veto the window; they read the poll as captured, and so does the energy
    integration.
    """

    def test_decimates_full_cadence_series_to_one_hz(self):
        # 60s at the sampler's 100ms cadence: one sample per second, plus the newest.
        raw = _series(n=600, dt=0.1)
        view = _gate_view(raw)
        assert len(view) == 61

    def test_keeps_the_newest_sample(self):
        # The trailing temperature / throttle windows are anchored on the newest
        # sample, so the view must never lag the series.
        raw = _series(n=600, dt=0.1)
        assert _gate_view(raw)[-1] is raw[-1]

    def test_series_already_at_one_hz_is_unchanged(self):
        raw = _series(n=120, dt=1.0)
        assert _gate_view(raw) == raw

    def test_decimates_per_gpu(self):
        # Per GPU, because the observables are per GPU (or pooled per GPU): a
        # two-GPU poll must not decimate away one device's series.
        raw = _series(n=600, dt=0.1, gpu=0) + _series(n=600, dt=0.1, gpu=1)
        view = _gate_view(raw)
        assert len([s for s in view if s.gpu_index == 0]) == 61
        assert len([s for s in view if s.gpu_index == 1]) == 61

    def test_sample_count_bounded_at_the_failsafe_horizon(self):
        # The whole point: at the 900s failsafe the gate sees ~900 samples rather than
        # the 9000 the full cadence would hand it, so the per-poll cost stays flat.
        assert len(_gate_view(_series(n=9000, dt=0.1))) <= 902

    def test_only_the_plateau_reads_the_view(self, monkeypatch):
        # Structural guard in both directions: the plateau must never be handed the raw
        # series (that is the defect), and the other two must never be handed the view
        # (that is what blinds them).
        seen: dict[str, int] = {}

        def recorder(name):
            def record(samples):
                seen[name] = len(samples)
                return True

            return record

        monkeypatch.setattr(server_warmup, "_power_plateau", recorder("plateau"))
        monkeypatch.setattr(server_warmup, "_temperature_settled", recorder("temperature"))
        monkeypatch.setattr(server_warmup, "_throttle_clear", recorder("throttle"))
        _evaluate_observables(_series(n=9000, dt=0.1))
        assert seen["plateau"] <= 902
        assert seen["temperature"] == 9000
        assert seen["throttle"] == 9000

    def test_plateau_verdict_survives_decimation(self):
        # The plateau's own decision is unchanged by the view: a settled 120s series at
        # full cadence passes either way, a ramping one fails either way.
        settled = _series(n=1200, dt=0.1)
        ramping = _series(n=1200, dt=0.1, rising=True)
        assert _power_plateau(settled) is _evaluate_observables(settled).power_plateau is True
        assert _power_plateau(ramping) is _evaluate_observables(ramping).power_plateau is False

    def test_throttle_transient_between_view_samples_still_vetoes(self):
        # A 900ms throttle episode landing between two 1 Hz view samples: the view
        # cannot see it at all, so the observable has to read the raw series.
        samples = _series(n=1200, dt=0.1)
        samples = _with_throttle(samples, lo=1114.1, hi=1114.9)
        assert not any(s.thermal_throttle for s in _gate_view(samples))  # invisible
        assert _evaluate_observables(samples).throttle_clear is False

    def test_duty_cycle_throttle_invisible_to_the_view_still_vetoes(self):
        # A throttle cycling on 400ms in every second, phased so that no 1 Hz view
        # sample ever lands on an ON slice: sustained throttling, invisible to the view.
        samples = [
            replace(s, thermal_throttle=2 <= i % 10 <= 5)
            for i, s in enumerate(_series(n=1200, dt=0.1))
        ]
        assert not any(s.thermal_throttle for s in _gate_view(samples))  # invisible
        assert _evaluate_observables(samples).throttle_clear is False

    def test_temperature_spike_between_view_samples_still_vetoes(self):
        # The temperature observable is a range over a trailing window - also an
        # extreme-value statistic, and also lost by decimation. A 3C spike 500ms wide,
        # placed between view samples, must still block the gate.
        samples = [
            replace(s, temperature_c=58.0 if 1100.1 <= s.timestamp <= 1100.4 else 55.0)
            for s in _series(n=1200, dt=0.1)
        ]
        view_temps = [s.temperature_c for s in _gate_view(samples) if s.temperature_c is not None]
        assert len(view_temps) == len(_gate_view(samples))
        assert max(view_temps) - min(view_temps) == 0.0  # invisible
        assert _evaluate_observables(samples).temperature_settled is False

    def test_fast_power_ripple_reads_as_plateaued(self):
        # The accepted consequence, pinned so it stays a decision rather than a
        # surprise: a 2 Hz ripple is above the view's Nyquist frequency, so the view
        # aliases it away and reads a plateau where the raw series reads instability.
        # The plateau asks whether power has settled at a level, not whether
        # consecutive samples differ, and the two observables that must catch
        # short-lived events are the ones kept at full cadence.
        samples = [
            replace(s, power_w=300.0 * (1.0 + 0.1 * math.sin(2.0 * math.pi * 2.0 * i * 0.1)))
            for i, s in enumerate(_series(n=1200, dt=0.1))
        ]
        assert _power_plateau(samples) is False
        assert _evaluate_observables(samples).power_plateau is True


class TestObservableState:
    def test_all_hold_requires_all_three(self):
        assert ObservableState(True, True, True).all_hold is True
        assert ObservableState(False, True, True).all_hold is False
        assert ObservableState(True, False, True).all_hold is False
        assert ObservableState(True, True, False).all_hold is False


# ---------------------------------------------------------------------------
# Probe request shape (drawn from the traffic shape distribution)
# ---------------------------------------------------------------------------


class TestBuildProbeRequest:
    def test_draws_dict_payload_from_shape_source(self):
        pr = build_probe_request(
            lambda i: RequestShape(index=i, payload={"prompt": "hi"}), path="/v1/completions"
        )
        assert pr.path == "/v1/completions"
        assert pr.payload == {"prompt": "hi"}
        assert pr.method == "POST"

    def test_non_dict_payload_is_bodyless(self):
        pr = build_probe_request(lambda i: RequestShape(index=i), path="/health", method="GET")
        assert pr.payload is None
        assert pr.method == "GET"


# ---------------------------------------------------------------------------
# Protocol description (divergence labeling)
# ---------------------------------------------------------------------------


class TestProtocolDescription:
    def test_composite_mentions_three_observables(self):
        desc = describe_server_warmup_protocol(ServerWarmupConfig())
        assert "power" in desc and "temperature" in desc and "throttle" in desc
        assert "900" in desc

    def test_fixed_mentions_duration_no_gate(self):
        desc = describe_server_warmup_protocol(
            ServerWarmupConfig(mode="fixed", duration_seconds=120)
        )
        assert "fixed" in desc and "120" in desc and "no convergence gate" in desc


# ---------------------------------------------------------------------------
# ServerWarmup - composite / fixed / timeout / re-warm
# ---------------------------------------------------------------------------


def _warmup(
    config: ServerWarmupConfig, sampler: FakeSampler, source: Any, transport: Any = None, **kw
):
    return ServerWarmup(
        config,
        traffic_factory=lambda ctx, horizon: source,
        transport=transport if transport is not None else FakeTransport(),
        sampler_factory=lambda: sampler,
        poll_interval=kw.pop("poll_interval", 0.0),
        **kw,
    )


class TestServerWarmupComposite:
    def test_converges_when_gate_holds(self):
        # ready_after=1: gate fails the first poll, so the loop sleeps and the traffic
        # task runs (delivers >= 1 completion) before converging on the second poll.
        sampler = FakeSampler(_SETTLED, ready_after=1)
        source = FakeSource()
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        asyncio.run(sw(_ctx()))
        result = sw.results[0]
        assert isinstance(result, ServerWarmupResult)
        assert result.converged is True
        assert result.timed_out is False
        assert result.final_observables.all_hold is True
        assert result.mode == "composite"
        assert source.issued >= 1  # warmup traffic actually ran

    def test_composite_captures_warmup_energy(self):
        # The gate sampler's power series ALSO feeds the warmup energy (same
        # sampler, not a parallel path). A settled series integrates to > 0 J.
        sampler = FakeSampler(_SETTLED, ready_after=1)
        source = FakeSource()
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        asyncio.run(sw(_ctx()))
        assert sw.results[0].energy_j is not None
        assert sw.results[0].energy_j > 0.0

    def test_traffic_and_sampler_lifecycle(self):
        # ready_after=1: the first poll is empty (gate fails), so the loop sleeps and
        # the traffic task actually runs before convergence on the second poll.
        sampler = FakeSampler(_SETTLED, ready_after=1)
        source = FakeSource()
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        asyncio.run(sw(_ctx()))
        assert sampler.started and sampler.stopped
        assert source.started and source.cancelled

    def test_timeout_proceeds_with_timed_out_stamp(self):
        # A never-settling series (only 30s of temp history) + a fake advancing clock.
        sampler = FakeSampler(_series(n=30))
        source = FakeSource()
        clk = FakeClock()

        async def advancing_sleep(d: float) -> None:
            clk.t += max(d, 1.0)
            await asyncio.sleep(0)  # yield so the traffic task runs

        sw = _warmup(
            ServerWarmupConfig(mode="composite", timeout_seconds=5.0),
            sampler,
            source,
            poll_interval=1.0,
            sleep=advancing_sleep,
            clock=clk,
        )
        asyncio.run(sw(_ctx()))
        result = sw.results[0]
        assert result.timed_out is True
        assert result.converged is False
        # Proceeded (returned) and released everything - never hung.
        assert sampler.stopped and source.cancelled


class TestServerWarmupTimeoutBoundary:
    """Regression: a clean warmup-traffic schedule that ends AT/AFTER the timeout is a
    timeout-proceed (loud timed_out), NOT traffic death; a schedule that ends strictly
    BEFORE the timeout is still genuine early exhaustion and fails loudly. Guards the
    v0.7.0 release-gate defect where a full-duration run failed the experiment cell.
    """

    def test_clean_exit_at_deadline_proceeds_timed_out(self):
        # The traffic schedule exhausts cleanly exactly at the gate boundary while the
        # observables never converge: proceed with the loud timed_out stamp, NO raise.
        sampler = FakeSampler([])  # never converges
        source = ExhaustAfterSource(n=1)  # completes on the first poll's yield
        clk = FakeClock()

        async def advancing_sleep(d: float) -> None:
            clk.t += max(d, 1.0)  # one poll pushes the clock to the deadline
            await asyncio.sleep(0)  # yield so the traffic task runs to completion

        sw = _warmup(
            ServerWarmupConfig(mode="composite", timeout_seconds=1.0),
            sampler,
            source,
            poll_interval=1.0,
            sleep=advancing_sleep,
            clock=clk,
        )
        asyncio.run(sw(_ctx()))  # must NOT raise
        result = sw.results[0]
        assert result.timed_out is True
        assert result.converged is False
        assert source.issued == 1  # the warmup traffic actually ran to the boundary
        assert sampler.stopped  # released, never hung

    def test_clean_exit_before_deadline_still_fails(self):
        # The same clean schedule exhaustion, but WELL before the deadline: genuine
        # early death, so WarmupTrafficError still fires (the fix must not mask it).
        sampler = FakeSampler([])  # never converges
        source = ExhaustAfterSource(n=3)
        clk = FakeClock()

        async def advancing_sleep(d: float) -> None:
            clk.t += max(d, 1.0)
            await asyncio.sleep(0)

        sw = _warmup(
            ServerWarmupConfig(mode="composite", timeout_seconds=100.0),
            sampler,
            source,
            poll_interval=1.0,
            sleep=advancing_sleep,
            clock=clk,
        )
        with pytest.raises(WarmupTrafficError) as exc:
            asyncio.run(sw(_ctx()))
        assert exc.value.__cause__ is None  # a clean early exit chains no cause
        assert "ended" in str(exc.value)
        assert sw.results == []  # no result recorded for a failed level


#: A gate evaluation far slower than the whole warmup under test. Blocking (not
#: awaitable) on purpose: it is the shape of the real cost, a CPU-bound detector call.
_SLOW_EVAL_S = 0.6


class TestSlowGateEvaluation:
    """Regression: the gate is evaluated OFF the issuing loop, so an evaluation that
    turns out slow can neither stall the warmup traffic nor starve the failsafe.

    Guards the shipped defect where the gate was evaluated synchronously on the loop
    that issues warmup traffic: one slow evaluation stopped the traffic, the idle GPU
    made the next evaluation slower still, and the failsafe could never fire because
    its deadline was checked by the loop the evaluation was blocking.
    """

    def test_slow_evaluation_cannot_push_past_the_deadline(self, monkeypatch):
        def slow_evaluate(samples):
            time.sleep(_SLOW_EVAL_S)
            return ObservableState(False, False, False)

        monkeypatch.setattr(server_warmup, "_evaluate_observables", slow_evaluate)
        source = FakeSource()
        sw = _warmup(
            ServerWarmupConfig(mode="composite", timeout_seconds=0.1),
            FakeSampler(_SETTLED),
            source,
            poll_interval=0.005,
        )

        async def drive() -> float:
            started = time.monotonic()
            await sw(_ctx())
            return time.monotonic() - started

        elapsed = asyncio.run(drive())
        assert sw.results[0].timed_out is True
        # The failsafe fired on its own deadline instead of waiting out the evaluation.
        assert elapsed < _SLOW_EVAL_S / 2

    def test_traffic_keeps_flowing_while_the_gate_evaluates(self, monkeypatch):
        source = FakeSource()
        issued_before: list[int] = []
        issued_after: list[int] = []

        def slow_evaluate(samples):
            issued_before.append(source.issued)
            time.sleep(_SLOW_EVAL_S)
            issued_after.append(source.issued)
            return ObservableState(False, False, False)

        monkeypatch.setattr(server_warmup, "_evaluate_observables", slow_evaluate)
        sw = _warmup(
            ServerWarmupConfig(mode="composite", timeout_seconds=0.2),
            FakeSampler([]),
            source,
            poll_interval=0.005,
        )
        asyncio.run(sw(_ctx()))
        assert issued_before  # the evaluation really ran
        # Requests were issued DURING the evaluation: the issuing loop kept its cadence.
        assert issued_after[0] > issued_before[0]

    def test_at_most_one_evaluation_runs_at_a_time(self, monkeypatch):
        # Off-loop evaluation must not turn a slow detector into a thread pile-up: a
        # poll that finds the previous evaluation still running starts no second one.
        # Without that guard this warmup queues one evaluation per poll.
        lock = threading.Lock()
        live = 0
        peak = 0

        def slow_evaluate(samples):
            nonlocal live, peak
            with lock:
                live += 1
                peak = max(peak, live)
            time.sleep(_SLOW_EVAL_S)
            with lock:
                live -= 1
            return ObservableState(False, False, False)

        monkeypatch.setattr(server_warmup, "_evaluate_observables", slow_evaluate)
        sw = _warmup(
            ServerWarmupConfig(mode="composite", timeout_seconds=0.2),
            FakeSampler(_SETTLED),
            FakeSource(),
            poll_interval=0.001,  # ~200 polls inside one evaluation
        )
        asyncio.run(sw(_ctx()))
        assert peak == 1


def _sawtooth(n: int = 1200, dt: float = 0.1) -> list[PowerThermalSample]:
    """A 295-305 W sawtooth of 1 s period on the sampler's 100 ms cadence.

    Deliberately shaped so full cadence and the 1 Hz view integrate to DIFFERENT
    energies: the view lands on the same phase of every tooth (295 W) while the true
    mean is 300 W. A constant-power fixture cannot tell the two apart, so it cannot
    catch the gate view leaking into the measurement. The 1.7% coefficient of variation
    stays well inside the plateau threshold, so the gate still converges on it.
    """
    return [
        replace(s, power_w=295.0 + 10.0 * (i % 10) / 9.0) for i, s in enumerate(_series(n=n, dt=dt))
    ]


class TestCaptureKeepsFullCadence:
    def test_warmup_energy_integrates_the_undecimated_series(self):
        # The gate's 1 Hz view must not reach the measurement: the warmup energy is the
        # integral of the sampler's own full-cadence series.
        from llenergymeasure.energy.nvml import integrate_power_samples
        from llenergymeasure.harness.windowing import _clean_samples

        raw = _sawtooth()
        expected = sum(integrate_power_samples(_clean_samples(raw)).values())
        # Guard the fixture itself: if the two cadences ever integrate alike again, this
        # test is vacuous and must be told so rather than passing.
        decimated = sum(integrate_power_samples(_clean_samples(_gate_view(raw))).values())
        assert abs(decimated - expected) / expected > 0.01

        sampler = FakeSampler(raw, ready_after=1)
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, FakeSource())
        asyncio.run(sw(_ctx()))
        assert sw.results[0].converged is True  # the fixture still passes the gate
        assert sw.results[0].energy_j == pytest.approx(expected)


class TestServerWarmupFixed:
    def test_runs_duration_no_gate(self):
        sampler = FakeSampler([])  # fixed mode does not sample
        source = FakeSource()
        slept: list[float] = []

        async def record_sleep(d: float) -> None:
            slept.append(d)
            await asyncio.sleep(0)  # yield so the traffic task runs

        sw = _warmup(
            ServerWarmupConfig(mode="fixed", duration_seconds=42.0),
            sampler,
            source,
            sleep=record_sleep,
        )
        asyncio.run(sw(_ctx()))
        result = sw.results[0]
        assert result.mode == "fixed"
        assert result.converged is True
        assert result.timed_out is False
        assert result.final_observables is None
        # NO pre-window idle cooldown: the only wait is the warmup duration itself.
        assert slept == [42.0]
        assert source.started and source.cancelled
        assert source.issued >= 1  # warmup traffic actually ran

    def test_zero_duration_skips_warmup_traffic(self):
        source = FakeSource()
        sw = _warmup(
            ServerWarmupConfig(mode="fixed", duration_seconds=0.0), FakeSampler([]), source
        )
        asyncio.run(sw(_ctx()))
        assert sw.results[0].converged is True
        assert sw.results[0].elapsed_s == 0.0
        assert source.started is False  # no traffic issued at all


class TestServerWarmupFailFast:
    """MUST-FIX 1: a dead warmup mechanism FAILS the level, never proceeds silently."""

    def test_composite_source_raises_immediately(self):
        sampler = FakeSampler([])  # never converges, so the loop observes the death
        source = RaisingSource()
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        with pytest.raises(WarmupTrafficError) as exc:
            asyncio.run(sw(_ctx()))
        assert isinstance(exc.value.__cause__, RuntimeError)  # chained, not swallowed
        assert sampler.stop_count == 1  # sampler released exactly once
        assert sw.results == []  # no result recorded for a failed level

    def test_fixed_source_raises_immediately(self):
        source = RaisingSource()
        sw = _warmup(
            ServerWarmupConfig(mode="fixed", duration_seconds=60.0), FakeSampler([]), source
        )
        with pytest.raises(WarmupTrafficError) as exc:
            asyncio.run(sw(_ctx()))
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert sw.results == []

    def test_composite_source_raises_mid_warmup(self):
        sampler = FakeSampler([])  # never converges
        source = RaiseAfterSource(n=3)
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        with pytest.raises(WarmupTrafficError):
            asyncio.run(sw(_ctx()))

    def test_composite_source_exits_cleanly_early(self):
        sampler = FakeSampler([])  # never converges before the schedule exhausts
        source = EarlyExitSource(n=3)
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        with pytest.raises(WarmupTrafficError) as exc:
            asyncio.run(sw(_ctx()))
        # A clean early exit chains no cause but still fails loudly.
        assert exc.value.__cause__ is None
        assert "ended" in str(exc.value)

    def test_composite_all_requests_fail(self):
        # Traffic stays alive but every request fails -> zero completions -> fail fast.
        sampler = FakeSampler(_SETTLED, ready_after=1)
        source = FakeSource()
        sw = _warmup(
            ServerWarmupConfig(mode="composite"), sampler, source, transport=AllFailTransport()
        )
        with pytest.raises(WarmupTrafficError) as exc:
            asyncio.run(sw(_ctx()))
        assert "zero" in str(exc.value)

    def test_fixed_all_requests_fail(self):
        # Traffic stays alive (all requests fail), so the duration sleeper wins the
        # race; use a fast sleep so the 42s duration does not run in real time.
        async def _fast_sleep(_d: float) -> None:
            await asyncio.sleep(0)

        source = FakeSource()
        sw = _warmup(
            ServerWarmupConfig(mode="fixed", duration_seconds=42.0),
            FakeSampler([]),
            source,
            transport=AllFailTransport(),
            sleep=_fast_sleep,
        )
        with pytest.raises(WarmupTrafficError) as exc:
            asyncio.run(sw(_ctx()))
        assert "zero" in str(exc.value)

    def test_manager_fails_level_before_opening_window(self):
        # Driven through the window manager: the hook raising fails the level BEFORE
        # any measurement window opens, and the sampler is released exactly once.
        sampler = FakeSampler([])
        source = RaisingSource()
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        sink = _RecordingSink()
        plan = LevelPlan(
            spec=WindowSpec(rate=10.0, duration_seconds=1.0),
            traffic_source=FakeSource(),  # the measured-window traffic, never reached
            transport=FakeTransport(),
        )
        manager = WindowManager(sink, warmup_hook=sw, windows_per_level=1)
        with pytest.raises(WarmupTrafficError):
            asyncio.run(manager.run_level(0, plan))
        assert sink.opened == 0  # no measurement window was opened
        assert sampler.stop_count == 1


class _RecordingSink:
    """Minimal WindowEnergySink that records whether a window was ever opened."""

    def __init__(self) -> None:
        self.opened = 0

    def open_window(self, event: WindowStartEvent) -> None:
        self.opened += 1

    def close_window(self, event: WindowStopEvent) -> Any:
        return None

    def abort_window(self, event: WindowAbortEvent) -> None:
        pass


class TestPerLevelReWarm:
    def test_rewarm_fires_and_records_per_level(self):
        # A fresh sampler + source per level (the factory is called each invocation).
        sw = ServerWarmup(
            ServerWarmupConfig(mode="composite"),
            traffic_factory=lambda ctx, horizon: FakeSource(),
            transport=FakeTransport(),
            sampler_factory=lambda: FakeSampler(_SETTLED, ready_after=1),
            poll_interval=0.0,
        )
        asyncio.run(sw(_ctx(level=0)))
        asyncio.run(sw(_ctx(level=1)))
        assert [r.level_index for r in sw.results] == [0, 1]

    def test_usable_as_window_manager_warmup_hook(self):
        # The instance is directly the WarmupHook the WindowManager awaits.
        sw = _warmup(
            ServerWarmupConfig(mode="fixed", duration_seconds=0.0), FakeSampler([]), FakeSource()
        )
        coro = sw(_ctx())
        assert asyncio.iscoroutine(coro)
        asyncio.run(coro)
        assert len(sw.results) == 1


def _poll(*, rising_power: bool = False, drift_temp: bool = False, throttle: bool = False):
    """A 120s poll that fails exactly the requested observable(s)."""
    out: list[PowerThermalSample] = []
    for i in range(120):
        out.append(
            PowerThermalSample(
                timestamp=1000.0 + i,
                power_w=(100.0 + i * 20.0) if rising_power else 300.0,
                temperature_c=(50.0 + i * 0.1) if drift_temp else 55.0,
                thermal_throttle=throttle,
            )
        )
    return out


class TestIndependentGatingThroughObservables:
    """Each observable computed from the real functions gates the AND independently."""

    def test_all_hold(self):
        assert _gate(_poll()) is True

    def test_power_alone_fails(self):
        assert _gate(_poll(rising_power=True)) is False

    def test_temperature_alone_fails(self):
        assert _gate(_poll(drift_temp=True)) is False

    def test_throttle_alone_fails(self):
        assert _gate(_poll(throttle=True)) is False


def _gate(samples) -> bool:
    return ObservableState(
        power_plateau=_power_plateau(samples),
        temperature_settled=_temperature_settled(samples),
        throttle_clear=_throttle_clear(samples),
    ).all_hold
