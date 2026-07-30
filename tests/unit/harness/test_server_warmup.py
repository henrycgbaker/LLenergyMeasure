"""Tests for the server-mode warmup protocol (SM8): the three-observable composite
gate, the fixed opt-out, the timeout failsafe, per-level re-warm, the readiness
probe request shape, and the divergence-labeling protocol descriptions.

Covers the SM8 verify charter: warmup draws from the (injected) MEASURED traffic
shape; NO pre-window idle cooldown; re-warm fires at every level; timeout stamps
timed_out; each composite observable gates independently.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llenergymeasure.config.models import ServerWarmupConfig
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.server_warmup import (
    ObservableState,
    ServerWarmup,
    ServerWarmupResult,
    _power_plateau,
    _temperature_settled,
    _throttle_clear,
    build_probe_request,
    describe_server_warmup_protocol,
)
from llenergymeasure.harness.traffic import IssuerReport, RequestShape
from llenergymeasure.harness.window_manager import WarmupContext, WindowSpec

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
    """

    def __init__(self, samples: list[PowerThermalSample], *, ready_after: int = 0) -> None:
        self._samples = samples
        self._ready_after = ready_after
        self._polls = 0
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def get_samples(self) -> list[PowerThermalSample]:
        polls = self._polls
        self._polls += 1
        return self._samples if polls >= self._ready_after else []


class FakeSource:
    """Runs until cancelled; records that it was started and torn down."""

    def __init__(self) -> None:
        self.started = False
        self.cancelled = False

    async def run(self, transport: Any, *, drain_timeout: float | None = None) -> IssuerReport:
        self.started = True
        try:
            while True:
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class FakeTransport:
    async def __call__(self, request: Any) -> Any:  # pragma: no cover - never called by fakes
        return None


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


def _warmup(config: ServerWarmupConfig, sampler: FakeSampler, source: FakeSource, **kw):
    return ServerWarmup(
        config,
        traffic_factory=lambda ctx, horizon: source,
        transport=FakeTransport(),
        sampler_factory=lambda: sampler,
        poll_interval=kw.pop("poll_interval", 0.0),
        **kw,
    )


class TestServerWarmupComposite:
    def test_converges_when_gate_holds(self):
        sampler = FakeSampler(_SETTLED)
        source = FakeSource()
        sw = _warmup(ServerWarmupConfig(mode="composite"), sampler, source)
        asyncio.run(sw(_ctx()))
        result = sw.results[0]
        assert isinstance(result, ServerWarmupResult)
        assert result.converged is True
        assert result.timed_out is False
        assert result.final_observables.all_hold is True
        assert result.mode == "composite"

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

    def test_zero_duration_skips_warmup_traffic(self):
        source = FakeSource()
        sw = _warmup(
            ServerWarmupConfig(mode="fixed", duration_seconds=0.0), FakeSampler([]), source
        )
        asyncio.run(sw(_ctx()))
        assert sw.results[0].converged is True
        assert sw.results[0].elapsed_s == 0.0
        assert source.started is False  # no traffic issued at all


class TestPerLevelReWarm:
    def test_rewarm_fires_and_records_per_level(self):
        # A fresh sampler + source per level (the factory is called each invocation).
        sw = ServerWarmup(
            ServerWarmupConfig(mode="composite"),
            traffic_factory=lambda ctx, horizon: FakeSource(),
            transport=FakeTransport(),
            sampler_factory=lambda: FakeSampler(_SETTLED),
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
