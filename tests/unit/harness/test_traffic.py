"""Conformance tests for the open-loop TrafficSource seam and Poisson issuer.

Host-only, no GPU, no real server: the transport is always an injected fake.
These tests pin the issuer to its ratified semantic contract (section 12 of the
server-mode plan, as amended):

- the arrival schedule is a genuine open-loop Poisson / gamma process (CV~1 for
  Poisson across the rate span; gamma CV tracks its burstiness),
- the schedule is deterministic under a seed,
- the issue loop is non-blocking (a stalled transport never slows issuance),
- a binding concurrency cap delays DISPATCH while preserving the schedule and
  the ``issued_at`` latency anchor, and is disclosed via ``cap_bound_fraction``,
- the window-manager-facing surface is exactly one method.

The async run-loop tests use real time with small, compressed schedules and
structural (not exact-timing) assertions, so they stay fast and robust under a
parallel/loaded CI runner.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from llenergymeasure.config.models import TrafficConfig
from llenergymeasure.harness.traffic import (
    PARTIAL_COMPLETION_ATTR,
    CompletionResult,
    HttpxTransport,
    IssuerReport,
    OpenLoopPoissonSource,
    RequestShape,
    TrafficSource,
    build_schedule,
    require_httpx,
)

# ---------------------------------------------------------------------------
# (a) Poisson schedule: CV ~ 1 across rates spanning 1-100 req/s
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rate", [1.0, 3.0, 10.0, 32.0, 100.0])
def test_poisson_schedule_cv_is_one(rate: float) -> None:
    """Exponential inter-arrivals have CV = 1 (memoryless) at every rate."""
    cfg = TrafficConfig(rate=rate, window_requests=10)
    schedule = build_schedule(cfg, seed=0, count=5000)

    assert schedule.arrival == "poisson"
    assert schedule.count == 5000
    assert abs(schedule.coefficient_of_variation() - 1.0) < 0.08


@pytest.mark.parametrize("rate", [1.0, 10.0, 100.0])
def test_poisson_schedule_mean_matches_rate(rate: float) -> None:
    """Mean inter-arrival is 1 / rate, so the schedule realises the target rate."""
    cfg = TrafficConfig(rate=rate, window_requests=10)
    schedule = build_schedule(cfg, seed=0, count=5000)

    mean_gap = float(schedule.interarrivals().mean())
    assert mean_gap == pytest.approx(1.0 / rate, rel=0.05)


# ---------------------------------------------------------------------------
# (b) Gamma schedule: CV tracks the burstiness parameter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("burstiness", [0.5, 1.0, 2.0])
def test_gamma_schedule_cv_tracks_burstiness(burstiness: float) -> None:
    """Gamma inter-arrivals realise CV = burstiness (>1 burstier, <1 smoother)."""
    cfg = TrafficConfig(rate=10.0, arrival="gamma", burstiness=burstiness, window_requests=10)
    schedule = build_schedule(cfg, seed=0, count=8000)

    cv = schedule.coefficient_of_variation()
    assert abs(cv - burstiness) / burstiness < 0.12


# ---------------------------------------------------------------------------
# (c) Determinism under a seed
# ---------------------------------------------------------------------------


def test_schedule_deterministic_under_seed() -> None:
    """Same (rate, arrival, seed) yields a byte-identical schedule."""
    cfg = TrafficConfig(rate=25.0, window_requests=500)
    first = build_schedule(cfg, seed=1234)
    second = build_schedule(cfg, seed=1234)
    assert first.offsets == second.offsets


def test_schedule_differs_across_seeds() -> None:
    """Different seeds yield different schedules (the RNG is actually seeded)."""
    cfg = TrafficConfig(rate=25.0, window_requests=500)
    assert build_schedule(cfg, seed=1).offsets != build_schedule(cfg, seed=2).offsets


def test_source_uses_config_seed_when_unset() -> None:
    """A source with no explicit seed falls back to the config's traffic.seed."""
    cfg = TrafficConfig(rate=25.0, window_requests=200, seed=99)
    from_source = OpenLoopPoissonSource(cfg).schedule.offsets
    from_config_seed = build_schedule(cfg, seed=99).offsets
    assert from_source == from_config_seed


def test_window_seconds_schedule_bounded_by_duration() -> None:
    """A duration window draws until the schedule covers window_seconds."""
    cfg = TrafficConfig(rate=50.0, window_seconds=2.0)
    schedule = build_schedule(cfg, seed=7)
    assert schedule.count > 0
    assert max(schedule.offsets) <= 2.0
    # ~rate * duration arrivals, within a generous band.
    assert 50 < schedule.count < 150


# ---------------------------------------------------------------------------
# (d) Non-blocking under a stalled transport
# ---------------------------------------------------------------------------


async def _hanging_transport(request: RequestShape) -> object:
    """Never returns: models a fully stalled system-under-test."""
    await asyncio.Event().wait()
    return None  # pragma: no cover - unreachable


def test_issue_loop_non_blocking_under_stall() -> None:
    """A stalled transport must not slow issuance: all requests still issue on schedule."""
    cfg = TrafficConfig(rate=500.0, window_requests=40)
    source = OpenLoopPoissonSource(cfg, seed=3)
    schedule_span = max(source.schedule.offsets)

    async def drive() -> IssuerReport:
        # wait_for is a safety net: a closed-loop (blocking) issuer would stall
        # on the first request forever and trip this, failing loudly.
        return await asyncio.wait_for(
            source.run(_hanging_transport, drain_timeout=0.2), timeout=5.0
        )

    report = asyncio.run(drive())

    # Every request was issued despite zero completions.
    assert report.issued_count == 40
    assert report.completed_count == 0
    # Issuance tracked the (short) schedule span, not the (infinite) stall.
    assert report.issuance_duration_s < schedule_span + 0.5
    # No cap in force, so nothing is cap-bound.
    assert report.cap_bound_fraction == 0.0


# ---------------------------------------------------------------------------
# (e) Concurrency-cap semantics
# ---------------------------------------------------------------------------


class _SlowTransport:
    """Fixed-latency transport: each call takes ``delay_s`` before returning."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.calls = 0

    async def __call__(self, request: RequestShape) -> str:
        self.calls += 1
        await asyncio.sleep(self.delay_s)
        return "ok"


def test_binding_cap_preserves_schedule_and_reports_cap_bound() -> None:
    """A binding cap delays dispatch, preserves the schedule + issued_at anchor, and is disclosed."""
    cfg = TrafficConfig(rate=500.0, window_requests=30, concurrency_cap=1)
    source = OpenLoopPoissonSource(cfg, seed=5)
    schedule_span = max(source.schedule.offsets)
    transport = _SlowTransport(delay_s=0.015)

    report = asyncio.run(asyncio.wait_for(source.run(transport), timeout=10.0))

    # All requests issued and (eventually) completed.
    assert report.issued_count == 30
    assert report.completed_count == 30

    # Schedule preserved: issuance kept pace with the arrival process and was NOT
    # dragged out to the serialized dispatch time (~30 * 15ms).
    assert report.issuance_duration_s < schedule_span + 0.3
    assert report.issuance_duration_s < 0.5 * transport.delay_s * 30

    # issued_at is the SCHEDULED time (the latency anchor), not the dispatch time:
    # inter-issue spacing matches the schedule offsets exactly.
    issued_deltas = [
        report.records[i].issued_at - report.records[0].issued_at
        for i in range(len(report.records))
    ]
    schedule_deltas = [off - source.schedule.offsets[0] for off in source.schedule.offsets]
    assert issued_deltas == pytest.approx(schedule_deltas, abs=1e-9)

    # Dispatch was delayed behind the cap: later requests queue for a slot.
    max_delay = max(r.dispatch_delay_s or 0.0 for r in report.records)
    assert max_delay > transport.delay_s  # at least one full slot-wait

    # The binding cap is disclosed for provenance.
    assert report.concurrency_cap == 1
    assert report.cap_bound_fraction > 0.8


def test_uncapped_run_completes_and_is_not_cap_bound() -> None:
    """Without a cap, every request dispatches promptly and cap_bound_fraction is 0."""
    cfg = TrafficConfig(rate=500.0, window_requests=30)
    source = OpenLoopPoissonSource(cfg, seed=8)
    transport = _SlowTransport(delay_s=0.005)

    report = asyncio.run(asyncio.wait_for(source.run(transport), timeout=10.0))

    assert report.issued_count == 30
    assert report.completed_count == 30
    assert report.concurrency_cap is None
    assert report.cap_bound_fraction == 0.0


def test_transport_error_is_bookkept_not_raised() -> None:
    """A transport exception is recorded per-request; it never stops issuance."""

    async def flaky(request: RequestShape) -> str:
        if request.index % 2 == 0:
            raise RuntimeError("boom")
        return "ok"

    cfg = TrafficConfig(rate=500.0, window_requests=10)
    source = OpenLoopPoissonSource(cfg, seed=2)
    report = asyncio.run(asyncio.wait_for(source.run(flaky), timeout=10.0))

    assert report.issued_count == 10
    assert report.completed_count == 10  # completed_at stamped even on error
    errored = [r for r in report.records if r.error is not None]
    assert len(errored) == 5
    assert all(isinstance(r.error, RuntimeError) for r in errored)


def test_completion_callbacks_do_not_influence_schedule() -> None:
    """Completions are bookkeeping only: schedule is identical with fast vs slow transport."""
    cfg = TrafficConfig(rate=300.0, window_requests=25)

    source_fast = OpenLoopPoissonSource(cfg, seed=11)
    source_slow = OpenLoopPoissonSource(cfg, seed=11)
    report_fast = asyncio.run(source_fast.run(_SlowTransport(delay_s=0.0)))
    report_slow = asyncio.run(source_slow.run(_SlowTransport(delay_s=0.02)))

    issued_fast = [r.issued_at - report_fast.records[0].issued_at for r in report_fast.records]
    issued_slow = [r.issued_at - report_slow.records[0].issued_at for r in report_slow.records]
    assert issued_fast == pytest.approx(issued_slow, abs=1e-9)


# ---------------------------------------------------------------------------
# (f) Exactly one window-manager-facing event surface
# ---------------------------------------------------------------------------


def test_traffic_source_has_exactly_one_surface() -> None:
    """The TrafficSource protocol exposes exactly one driving method: run()."""
    surface = [
        name
        for name, value in vars(TrafficSource).items()
        if not name.startswith("_") and callable(value)
    ]
    assert surface == ["run"]


def test_open_loop_source_satisfies_protocol() -> None:
    """The built-in issuer structurally satisfies TrafficSource."""
    cfg = TrafficConfig(rate=10.0, window_requests=10)
    assert isinstance(OpenLoopPoissonSource(cfg, seed=1), TrafficSource)


# ---------------------------------------------------------------------------
# [server] extra: lazy httpx import with an actionable error
# ---------------------------------------------------------------------------


def test_require_httpx_names_the_server_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """When httpx is absent, the error names the extra to install."""
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError) as excinfo:
        require_httpx()
    message = str(excinfo.value)
    assert "httpx" in message
    assert "llenergymeasure[server]" in message


def test_httpx_transport_construction_requires_server_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HttpxTransport is the lazy-import use site: constructing it without httpx errors."""
    monkeypatch.setitem(sys.modules, "httpx", None)
    with pytest.raises(ImportError) as excinfo:
        HttpxTransport(base_url="http://localhost:8000")
    assert "llenergymeasure[server]" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Client-side streaming token counting - a fake httpx stream
# ---------------------------------------------------------------------------


class _FakeStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


class _FakeStreamCtx:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _FakeStreamResponse:
        return _FakeStreamResponse(self._lines)

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeAsyncClient:
    """Minimal fake of httpx.AsyncClient.stream over canned SSE lines."""

    def __init__(self, lines: list[str], payloads: list[dict], **_kw: object) -> None:
        self._lines = lines
        self._payloads = payloads

    def stream(self, method: str, path: str, *, json: dict) -> _FakeStreamCtx:
        self._payloads.append(json)
        return _FakeStreamCtx(self._lines)

    async def aclose(self) -> None:
        return None


def test_httpx_transport_streams_and_counts_client_side(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The transport counts streamed deltas client-side; server usage is auxiliary."""
    lines = [
        'data: {"choices":[{"text":"Hel","finish_reason":null}]}',
        'data: {"choices":[{"text":"lo","finish_reason":null}]}',
        # An empty finish delta is a step with no text - not a counted token.
        'data: {"choices":[{"text":"","finish_reason":"length"}]}',
        # A usage-only terminal chunk (choices empty) carries the auxiliary count.
        'data: {"choices":[],"usage":{"prompt_tokens":4,"completion_tokens":2}}',
        "data: [DONE]",
    ]
    payloads: list[dict] = []
    fake_httpx = SimpleNamespace(AsyncClient=lambda **kw: _FakeAsyncClient(lines, payloads, **kw))
    monkeypatch.setattr("llenergymeasure.harness.traffic.require_httpx", lambda: fake_httpx)

    transport = HttpxTransport(base_url="http://x", path="/v1/completions")
    result = asyncio.run(
        transport(RequestShape(index=0, payload={"model": "gpt2", "prompt": "hi"}))
    )

    assert isinstance(result, CompletionResult)
    # Two non-empty deltas -> two client-counted tokens (the empty finish delta is not one).
    assert len(result.output_token_times) == 2
    assert result.text == "Hello"
    assert result.first_token_at == result.output_token_times[0]
    assert result.finish_reason == "length"
    # The engine's self-reported usage rides as auxiliary provenance only.
    assert result.server_prompt_tokens == 4
    assert result.server_completion_tokens == 2
    # Streaming was requested with the usage option, and the original payload is preserved.
    assert payloads[0]["stream"] is True
    assert payloads[0]["stream_options"] == {"include_usage": True}
    assert payloads[0]["prompt"] == "hi"


class _FailingStreamResponse:
    def __init__(self, lines: list[str], fail_after: int) -> None:
        self._lines = lines
        self._fail_after = fail_after

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for i, line in enumerate(self._lines):
            if i == self._fail_after:
                raise ConnectionError("stream reset mid-response")
            yield line


class _FailingStreamCtx:
    def __init__(self, lines: list[str], fail_after: int) -> None:
        self._lines = lines
        self._fail_after = fail_after

    async def __aenter__(self) -> _FailingStreamResponse:
        return _FailingStreamResponse(self._lines, self._fail_after)

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FailingAsyncClient:
    def __init__(self, lines: list[str], fail_after: int, **_kw: object) -> None:
        self._lines = lines
        self._fail_after = fail_after

    def stream(self, method: str, path: str, *, json: dict) -> _FailingStreamCtx:
        return _FailingStreamCtx(self._lines, self._fail_after)

    async def aclose(self) -> None:
        return None


def test_httpx_transport_preserves_partial_on_mid_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that dies mid-response attaches the tokens delivered so far."""
    lines = [
        'data: {"choices":[{"text":"Hel","finish_reason":null}]}',
        'data: {"choices":[{"text":"lo","finish_reason":null}]}',
        # The reset strikes here, before any finish / usage / [DONE] chunk.
        'data: {"choices":[{"text":"!","finish_reason":null}]}',
    ]
    fake_httpx = SimpleNamespace(AsyncClient=lambda **kw: _FailingAsyncClient(lines, 2, **kw))
    monkeypatch.setattr("llenergymeasure.harness.traffic.require_httpx", lambda: fake_httpx)

    transport = HttpxTransport(base_url="http://x", path="/v1/completions")
    with pytest.raises(ConnectionError) as excinfo:
        asyncio.run(transport(RequestShape(index=0, payload={"prompt": "hi"})))

    partial = getattr(excinfo.value, PARTIAL_COMPLETION_ATTR)
    assert isinstance(partial, CompletionResult)
    # The two deltas delivered before the reset are preserved (not discarded),
    # so their in-span tokens can still count toward the energy denominator.
    assert len(partial.output_token_times) == 2
    assert partial.text == "Hello"
    assert partial.finish_reason is None  # the stream never finished
