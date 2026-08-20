"""Tests for the server-mode measurement session.

Host-only, no GPU, no real server: the engine (ServerCapable), the window
manager, the energy sink, the traffic source, and the warmup hook are all
injected fakes, so the full launch -> warm up -> windows -> shutdown lifecycle is
exercised deterministically.

Charter:
- one server lifetime produces N window results without re-keying the offline
  loop;
- cleanup-exactly-once on the normal, exception, and interrupt paths, with the
  server reaped on every exit (the session-hardening invariant);
- the three banked contracts: the issuance horizon covers ramp + N windows; the
  client-counted tokens flow through the TokenReceiptFn seam (client-side
  streamed-delta counts); an AbortedLevel is caught at the run_level await
  site and a WarmupTrafficError aborts the session;
- per-level ServerWarmupResult is stamped into each window result (the cross-mode
  divergence label);
- the resolved warmup is read in-process (the serialization-boundary contract).
"""

from __future__ import annotations

import asyncio
import os
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from llenergymeasure.config.models import ExperimentConfig, ServerWarmupConfig
from llenergymeasure.config.user_config import UserConfig, UserServerConfig
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.server_warmup import ServerWarmupResult, WarmupTrafficError
from llenergymeasure.harness.traffic import IssuerReport, RequestRecord
from llenergymeasure.harness.window_manager import (
    ABORTED_LEVEL_ATTR,
    ATTRIBUTION_STEADY_STATE_SPAN,
    AbortedLevel,
    LevelOutcome,
    LevelValidation,
    WindowBookkeeping,
    WindowBoundaries,
    WindowRecord,
    WindowSpec,
)
from llenergymeasure.serving.transport import CompletionResult, RequestShape
from llenergymeasure.study import server_session as ss
from llenergymeasure.study.server_session import (
    ServerSession,
    ServerSessionError,
    ServerSessionResult,
    _CompletionsShapeSource,
    _core_energy_j,
    _drive_levels,
    _level_traffic_source,
    _sum_server_completion_tokens,
    build_request_rows_by_window,
    client_token_receipts,
)

# ---------------------------------------------------------------------------
# Builders / fakes
# ---------------------------------------------------------------------------


def _server_config(**server_overrides: Any) -> ExperimentConfig:
    """A minimal vLLM server-mode config (transformers+server is rejected)."""
    server: dict[str, Any] = {"traffic": {"rate": 10, "window_seconds": 60}}
    server.update(server_overrides)
    return ExperimentConfig(
        task={"model": "gpt2"},
        engine="vllm",
        serving_mode="server",
        server=server,
        measurement={"baseline": {"enabled": False}},
    )


class FakeEngine:
    """A recording ServerCapable engine (no real process/container)."""

    def __init__(
        self, *, launch_error: Exception | None = None, ready_error: Exception | None = None
    ) -> None:
        self.launch_error = launch_error
        self.ready_error = ready_error
        self.launched = 0
        self.readied = 0
        self.shutdowns = 0
        self.handle = SimpleNamespace(base_url="http://127.0.0.1:9", engine="vllm")

    def launch(self, config: Any, placement: Any) -> Any:
        self.launched += 1
        if self.launch_error is not None:
            raise self.launch_error
        return self.handle

    def await_ready(self, handle: Any, probe: Any, *, timeout: float) -> None:
        self.readied += 1
        if self.ready_error is not None:
            raise self.ready_error

    def shutdown(self, handle: Any) -> None:
        self.shutdowns += 1


class FakeManifest:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []

    def mark_running(self, config_hash: str, cycle: int) -> None:
        self.calls.append(("running", (config_hash, cycle)))

    def mark_completed(self, config_hash: str, cycle: int, result_file: str, **kw: Any) -> None:
        self.calls.append(("completed", (config_hash, cycle, result_file, kw)))

    def mark_failed(
        self, config_hash: str, cycle: int, error_type: str, message: str, **kw: Any
    ) -> None:
        self.calls.append(("failed", (config_hash, cycle, error_type, message)))

    @property
    def statuses(self) -> list[str]:
        return [c[0] for c in self.calls]


def _fake_runner(
    *, interrupt_event: Any = None, timeout: float = 30.0, gpu_indices: Any = None
) -> Any:
    study = SimpleNamespace(
        study_execution=SimpleNamespace(experiment_timeout_seconds=timeout, gpu_indices=gpu_indices)
    )
    return SimpleNamespace(
        manifest=FakeManifest(),
        _progress=None,
        _interrupt_event=interrupt_event,
        _runner_specs=None,
        study=study,
    )


class _FakeShapeSource:
    def __call__(self, index: int) -> RequestShape:
        return RequestShape(index=index, payload={"model": "m", "prompt": "hi", "max_tokens": 1})


class FakeManager:
    """A window manager stand-in: run_level returns / raises per a script."""

    def __init__(self, script: list[Any]) -> None:
        # Each entry is a LevelOutcome (returned) or an Exception (raised).
        self._script = script
        self.calls: list[int] = []

    async def run_level(self, level_index: int, level: Any) -> LevelOutcome:
        self.calls.append(level_index)
        await asyncio.sleep(0)
        item = self._script[level_index]
        if isinstance(item, BaseException):
            raise item
        return item


class FakeWarmup:
    """A warmup hook stand-in exposing the ServerWarmup.results surface."""

    def __init__(self, results: list[ServerWarmupResult]) -> None:
        self.results = results

    async def __call__(self, context: Any) -> None:  # pragma: no cover - fake manager skips it
        return None


def _report(records: list[RequestRecord]) -> IssuerReport:
    return IssuerReport(
        records=records,
        issued_count=len(records),
        completed_count=sum(1 for r in records if r.completed_at is not None),
        cap_bound_fraction=0.0,
        issuance_duration_s=0.0,
        concurrency_cap=None,
    )


def _window_record(index: int, *, energy: float | None, j_per_token: float | None) -> WindowRecord:
    boundaries = WindowBoundaries(window_start=0.0, span_start=1.0, span_end=2.0)
    bookkeeping = WindowBookkeeping(
        boundaries=boundaries,
        attribution_policy=ATTRIBUTION_STEADY_STATE_SPAN,
        energy_denominator_tokens=10,
    )
    return WindowRecord(
        window_index=index,
        boundaries=boundaries,
        energy=None,
        bookkeeping=bookkeeping,
        window_energy_j=energy,
        window_j_per_token=j_per_token,
        intra_window_cov=0.01,
    )


def _level_outcome(
    level_index: int = 0, *, valid: bool = True, n_windows: int = 3, energy: float = 10.0
) -> LevelOutcome:
    windows = [_window_record(i, energy=energy, j_per_token=0.5) for i in range(n_windows)]
    validation = LevelValidation(
        valid=valid,
        reason=None if valid else "window-to-window J/token not steady",
        cov=0.01,
        windows_considered=n_windows,
    )
    return LevelOutcome(
        level_index=level_index,
        spec=WindowSpec(rate=10.0),
        windows=windows,
        validation=validation,
        issuer_report=_report([]),
    )


def _flat_core(power_w: float = 100.0, duration: float = 10.0, n: int = 21) -> Any:
    """A measured-core stand-in with a flat power series (energy = power * duration)."""
    samples = [
        PowerThermalSample(timestamp=1000.0 + duration * i / (n - 1), power_w=power_w, gpu_index=0)
        for i in range(n)
    ]
    return SimpleNamespace(timeseries_samples=samples)


def _abort_exc(level_index: int, *, window: int, cores: list[Any]) -> RuntimeError:
    """A run_level-style exception carrying its AbortedLevel partial state."""
    exc = RuntimeError("mid-window transport failure")
    setattr(
        exc,
        ABORTED_LEVEL_ATTR,
        AbortedLevel(
            level_index=level_index,
            aborted_window_index=window,
            reason=f"aborted: window {window}",
            completed_cores=cores,
        ),
    )
    return exc


def _warmup_result(level_index: int = 0) -> ServerWarmupResult:
    return ServerWarmupResult(
        level_index=level_index,
        mode="composite",
        converged=True,
        timed_out=False,
        elapsed_s=1.0,
        final_observables=None,
        pre_window_protocol="server convergence-composite warmup (test)",
    )


def _override(session: ServerSession, name: str, fn: Any) -> None:
    """Attach an untyped test double to a session seam method (mypy/B010-transparent)."""
    session.__dict__[name] = fn


def _make_session(
    engine: FakeEngine,
    *,
    runner: Any = None,
    config: ExperimentConfig | None = None,
) -> ServerSession:
    runner = runner if runner is not None else _fake_runner()
    config = config if config is not None else _server_config()
    session = ServerSession(runner, config, None, config_hash="h", cycle=1, index=1, engine=engine)
    # Stub the heavy real seams so __enter__/run() are host-only. (The seam-override
    # assignments are intentionally untyped test doubles.)
    _override(session, "_make_shape_source", lambda: _FakeShapeSource())
    _override(session, "_make_transport", lambda base_url: SimpleNamespace())
    _override(session, "_make_energy_sink", lambda: object())
    return session


def _wire_run(
    session: ServerSession, manager: FakeManager, warmup: FakeWarmup, plans: list[Any] | None = None
) -> None:
    _override(session, "_make_warmup", lambda shape, transport: warmup)
    _override(session, "_make_manager", lambda sink, wu: manager)
    _override(session, "_make_level_plans", lambda shape, transport: plans or [object()])


# ---------------------------------------------------------------------------
# Contract 2 - client-side token receipts are the canonical denominator
# ---------------------------------------------------------------------------


def _completion(
    token_times: list[float],
    *,
    prompt: int | None = None,
    completion: int | None = None,
    finish_reason: str | None = None,
) -> CompletionResult:
    return CompletionResult(
        text="x" * len(token_times),
        output_token_times=list(token_times),
        first_token_at=token_times[0] if token_times else None,
        server_prompt_tokens=prompt,
        server_completion_tokens=completion,
        finish_reason=finish_reason,
    )


class TestTokenReceipts:
    def test_client_streamed_deltas_are_the_receipts(self) -> None:
        rec = RequestRecord(
            index=0,
            issued_at=1.0,
            request=RequestShape(index=0),
            completed_at=5.0,
            result=_completion([2.0, 3.0, 4.0], completion=99),
        )
        # Token-granular: one receipt at each streamed delta's arrival time. The
        # server-reported usage (99) never reaches the denominator seam.
        assert client_token_receipts(rec) == (2.0, 3.0, 4.0)

    def test_non_completion_result_yields_no_receipts(self) -> None:
        # A legacy dict response (no client-counted deltas) is not a denominator.
        rec = RequestRecord(
            index=0,
            issued_at=1.0,
            request=RequestShape(index=0),
            completed_at=5.0,
            result={"usage": {"completion_tokens": 3}},
        )
        assert client_token_receipts(rec) == ()

    def test_request_without_completion_yields_no_receipts(self) -> None:
        # No CompletionResult on the record (nothing streamed) -> no receipts.
        never = RequestRecord(
            index=0, issued_at=1.0, request=RequestShape(index=0), completed_at=None
        )
        assert client_token_receipts(never) == ()

    def test_errored_or_timed_out_request_preserves_partial_receipts(self) -> None:
        # A mid-stream failure's delivered tokens still count in the denominator,
        # regardless of the error flag or a missing completion timestamp.
        errored = RequestRecord(
            index=1,
            issued_at=1.0,
            request=RequestShape(index=1),
            completed_at=5.0,
            result=_completion([2.0, 3.0]),
            error=RuntimeError("boom"),
        )
        assert client_token_receipts(errored) == (2.0, 3.0)
        timed_out = RequestRecord(
            index=2,
            issued_at=1.0,
            request=RequestShape(index=2),
            completed_at=None,
            result=_completion([2.5]),
        )
        assert client_token_receipts(timed_out) == (2.5,)

    def test_empty_stream_yields_no_receipts(self) -> None:
        rec = RequestRecord(
            index=0,
            issued_at=1.0,
            request=RequestShape(index=0),
            completed_at=5.0,
            result=_completion([]),
        )
        assert client_token_receipts(rec) == ()


# ---------------------------------------------------------------------------
# Per-window request-log rows (attribution flags, client-vs-server count)
# ---------------------------------------------------------------------------


def _win(window_index: int, span_start: float, span_end: float) -> WindowRecord:
    boundaries = WindowBoundaries(window_start=0.0, span_start=span_start, span_end=span_end)
    bookkeeping = WindowBookkeeping(
        boundaries=boundaries,
        attribution_policy=ATTRIBUTION_STEADY_STATE_SPAN,
        energy_denominator_tokens=0,
    )
    return WindowRecord(
        window_index=window_index,
        boundaries=boundaries,
        energy=None,
        bookkeeping=bookkeeping,
        window_energy_j=None,
        window_j_per_token=None,
        intra_window_cov=None,
    )


def _req(
    index: int,
    issued_at: float,
    *,
    result: Any = None,
    completed_at: float | None = None,
    error: BaseException | None = None,
    dispatched_at: float | None = None,
) -> RequestRecord:
    return RequestRecord(
        index=index,
        issued_at=issued_at,
        request=RequestShape(index=index),
        dispatched_at=dispatched_at,
        completed_at=completed_at,
        result=result,
        error=error,
    )


class TestRequestRows:
    def test_attribution_ramp_window_and_drain_tail(self) -> None:
        """Ramp, in-window, and drain-straddling requests get the right boundary flags."""
        windows = [_win(0, 10.0, 20.0), _win(1, 20.0, 30.0)]
        ramp = _req(0, 5.0, completed_at=12.0, result=_completion([11.0, 12.0]))
        straddler = _req(1, 15.0, completed_at=25.0, result=_completion([15.5, 24.0, 25.0]))
        in_win1 = _req(2, 22.0, completed_at=24.0, result=_completion([23.0]))

        rows_by_window = build_request_rows_by_window(
            [ramp, straddler, in_win1], windows, level_index=0
        )

        assert [len(rl) for rl in rows_by_window] == [2, 1]  # ramp + straddler own window 0
        ramp_row, straddler_row = rows_by_window[0]
        assert ramp_row.is_ramp is True
        assert ramp_row.in_measurement_window is False
        assert straddler_row.is_ramp is False
        assert straddler_row.in_measurement_window is True
        assert straddler_row.completed_in_drain is True  # completed after span_end 20
        win1_row = rows_by_window[1][0]
        assert win1_row.window_index == 1
        assert win1_row.in_measurement_window is True
        assert win1_row.completed_in_drain is False

    def test_ttft_and_latency_populated_from_stream(self) -> None:
        """TTFT / e2e / per-token times are derived from the streamed CompletionResult."""
        windows = [_win(0, 10.0, 20.0)]
        rec = _req(
            0,
            12.0,
            dispatched_at=12.01,
            completed_at=13.0,
            result=_completion([12.5, 12.7, 12.9]),
        )
        row = build_request_rows_by_window([rec], windows, level_index=0)[0][0]
        assert row.first_token_at == 12.5
        assert row.ttft_ms == pytest.approx(500.0)  # (12.5 - 12.0) * 1000
        assert row.e2e_latency_ms == pytest.approx(1000.0)  # (13.0 - 12.0) * 1000
        assert row.output_token_times == [12.5, 12.7, 12.9]
        assert row.client_output_tokens == 3

    def test_client_count_is_denominator_server_count_is_auxiliary(self) -> None:
        """The client-counted deltas drive client_output_tokens; server usage rides aside."""
        windows = [_win(0, 10.0, 20.0)]
        # Client streamed 3 deltas; the engine self-reports 99 (a divergence).
        rec = _req(
            0,
            12.0,
            completed_at=13.0,
            result=_completion(
                [12.5, 12.7, 12.9], prompt=41, completion=99, finish_reason="length"
            ),
        )
        row = build_request_rows_by_window([rec], windows, level_index=0)[0][0]
        assert row.client_output_tokens == 3  # canonical = client count, never 99
        assert row.server_completion_tokens == 99  # preserved as auxiliary
        assert row.server_prompt_tokens == 41
        assert row.finish_reason == "length"  # threaded from the stream (SLO input)
        # The window aggregate of the auxiliary is a sum, None when never reported.
        assert _sum_server_completion_tokens([row]) == 99

    def test_status_ok_error_timeout(self) -> None:
        """Rows carry physical facts per status; the consumer filters, the row never does."""
        windows = [_win(0, 10.0, 20.0)]
        ok = _req(0, 11.0, completed_at=12.0, result=_completion([11.5], finish_reason="stop"))
        # Error / timeout requests carry their REAL partial receipts.
        errored = _req(
            1, 12.0, completed_at=12.5, result=_completion([12.2, 12.3]), error=RuntimeError("x")
        )
        timed_out = _req(2, 13.0, completed_at=None, result=_completion([13.2]))
        rows = build_request_rows_by_window([ok, errored, timed_out], windows, level_index=0)[0]

        assert [r.status for r in rows] == ["ok", "error", "timeout"]
        # The token series is real for ALL statuses (the delivered tokens count).
        assert [r.client_output_tokens for r in rows] == [1, 2, 1]
        assert rows[1].output_token_times == [12.2, 12.3]
        # Raw-record: first_token_at / ttft are the REAL physical facts when a token
        # arrived (even on a failed / timed-out row) - first_token_at IS the series[0].
        assert rows[1].first_token_at == 12.2
        assert rows[1].ttft_ms == pytest.approx(200.0)  # (12.2 - 12.0) * 1000
        assert rows[2].first_token_at == 13.2
        assert rows[2].ttft_ms == pytest.approx(200.0)  # (13.2 - 13.0) * 1000
        # finish_reason is real only when a finish chunk arrived; a mid-stream death
        # never finished, so it stays null (truthful) on the error / timeout rows.
        assert [r.finish_reason for r in rows] == ["stop", None, None]
        # e2e is time-to-terminal: the error row carries its to-failure latency, the
        # timeout row (never completed) leaves completed_at / e2e null.
        assert rows[1].e2e_latency_ms == pytest.approx(500.0)  # (12.5 - 12.0) * 1000
        assert rows[1].completed_at == 12.5
        assert rows[2].e2e_latency_ms is None
        assert rows[2].completed_at is None

    def test_mid_stream_failure_tokens_count_in_denominator(self) -> None:
        """Regression: a clean straddler and a mid-stream-failed request with
        identical in-span receipts contribute equally to the energy denominator."""
        from llenergymeasure.harness.window_manager import build_window_bookkeeping

        boundaries = WindowBoundaries(window_start=0.0, span_start=10.0, span_end=20.0)
        in_span = [11.0, 12.0, 13.0, 14.0, 15.0]
        clean = _req(0, 11.0, completed_at=25.0, result=_completion(in_span))  # ok straddler
        failed = _req(
            1, 11.0, completed_at=16.0, result=_completion(in_span), error=RuntimeError("reset")
        )
        report = IssuerReport(
            records=[clean, failed],
            issued_count=2,
            completed_count=1,
            cap_bound_fraction=0.0,
            issuance_duration_s=0.0,
            concurrency_cap=None,
        )
        bk = build_window_bookkeeping(boundaries, report, token_receipt_fn=client_token_receipts)
        assert bk.energy_denominator_tokens == 10  # 5 + 5: the failed request counts equally

    def test_sum_server_completion_none_when_unreported(self) -> None:
        """The auxiliary window total is None (not 0) when no engine reported usage."""
        windows = [_win(0, 10.0, 20.0)]
        rec = _req(0, 11.0, completed_at=12.0, result=_completion([11.5]))  # no server usage
        row = build_request_rows_by_window([rec], windows, level_index=0)[0][0]
        assert row.server_completion_tokens is None
        assert _sum_server_completion_tokens([row]) is None

    def test_no_records_yields_empty_row_lists_per_window(self) -> None:
        """A level with no requests still returns one (empty) row list per window."""
        windows = [_win(0, 10.0, 20.0), _win(1, 20.0, 30.0)]
        assert build_request_rows_by_window([], windows, level_index=0) == [[], []]


# ---------------------------------------------------------------------------
# Request encoding + issuance horizon (contract 1)
# ---------------------------------------------------------------------------


class TestShapeAndHorizon:
    def test_completions_shape_encodes_openai_body(self) -> None:
        src = _CompletionsShapeSource(["a", "b"], model="gpt2", max_tokens=7)
        shape = src(0)
        assert shape.payload == {
            "model": "gpt2",
            "prompt": "a",
            "max_tokens": 7,
            "temperature": 0.0,
        }
        # Prompts cycle by index.
        assert src(3).payload["prompt"] == "b"

    def test_level_traffic_source_covers_full_level_horizon(self) -> None:
        # The issuance schedule must span ramp + windows_per_level * duration as one
        # continuous run (contract 1): a horizon-sized window_seconds is projected.
        config = _server_config(traffic={"rate": 20, "window_seconds": 60, "seed": 7})
        traffic = config.server.traffic
        horizon = 30.0 + 3 * 60.0  # ramp + 3 windows
        source = _level_traffic_source(
            traffic,
            rate=traffic.rate,
            arrival=traffic.arrival,
            horizon_seconds=horizon,
            shape_source=_FakeShapeSource(),
        )
        # The schedule's last offset covers (near) the whole horizon, not one window.
        assert source.schedule.offsets[-1] > 60.0
        assert source.schedule.offsets[-1] <= horizon


# ---------------------------------------------------------------------------
# Contract 3 - the level driver: catch AbortedLevel at the run_level await site
# ---------------------------------------------------------------------------


def _drive(manager: Any, plans: list[Any], **kw: Any) -> tuple[str, list[LevelOutcome], list[Any]]:
    outcomes: list[LevelOutcome] = []
    failures: list[Any] = []
    status = asyncio.run(_drive_levels(manager, plans, outcomes, failures, **kw))
    return status, outcomes, failures


class TestDriver:
    def test_happy_path_collects_every_level(self) -> None:
        manager = FakeManager([_level_outcome(0), _level_outcome(1)])
        status, outcomes, failures = _drive(manager, [object(), object()])
        assert status == "ok"
        assert [o.level_index for o in outcomes] == [0, 1]
        assert failures == []

    def test_warmup_traffic_error_aborts_session(self) -> None:
        # A dead transport dooms later levels: the session aborts, later levels
        # are NOT run, and the failure is recorded (not silently skipped).
        manager = FakeManager([WarmupTrafficError("dead"), _level_outcome(1)])
        status, outcomes, failures = _drive(manager, [object(), object()])
        assert status == "warmup_aborted"
        assert outcomes == []
        assert manager.calls == [0]  # level 1 never attempted
        assert len(failures) == 1 and "warmup failed" in failures[0].reason

    def test_aborted_level_recorded_then_continues(self) -> None:
        # A non-warmup level failure carries its partial state on the exception;
        # the driver catches it at the run_level await site, records it, and
        # continues to the next level.
        exc = RuntimeError("mid-window transport failure")
        setattr(
            exc,
            ABORTED_LEVEL_ATTR,
            AbortedLevel(
                level_index=0,
                aborted_window_index=1,
                reason="aborted: boom",
                completed_cores=[None],
            ),
        )
        manager = FakeManager([exc, _level_outcome(1)])
        status, outcomes, failures = _drive(manager, [object(), object()])
        assert status == "ok"
        assert manager.calls == [0, 1]  # continued past the failed level
        assert [o.level_index for o in outcomes] == [1]
        assert len(failures) == 1
        assert failures[0].reason == "aborted: boom"
        assert failures[0].aborted_window_index == 1

    def test_ramp_phase_failure_recorded_and_continues(self) -> None:
        # A level failure with no AbortedLevel to preserve (e.g. a ramp-phase
        # error) is still recorded invalid-with-reason and the session continues -
        # one bad level never crashes the study (contract 3 / offline parity).
        manager = FakeManager([RuntimeError("ramp-phase blowup"), _level_outcome(1)])
        status, outcomes, failures = _drive(manager, [object(), object()])
        assert status == "ok"
        assert manager.calls == [0, 1]
        assert [o.level_index for o in outcomes] == [1]
        assert len(failures) == 1
        assert "ramp-phase blowup" in failures[0].reason
        assert failures[0].completed_cores == []

    def test_interrupt_preserves_partial_and_reraises(self) -> None:
        # SIGINT bridge: the watcher cancels the driving task; the manager attaches
        # its AbortedLevel to the CancelledError; the driver catches it INLINE
        # (attribute intact), records the partial, and re-raises so __exit__ reaps.
        class CancellingManager:
            def __init__(self) -> None:
                self.calls: list[int] = []

            async def run_level(self, level_index: int, level: Any) -> LevelOutcome:
                self.calls.append(level_index)
                try:
                    await asyncio.sleep(10)
                except BaseException as exc:
                    setattr(
                        exc,
                        ABORTED_LEVEL_ATTR,
                        AbortedLevel(
                            level_index=level_index,
                            aborted_window_index=0,
                            reason="aborted: cancelled",
                            completed_cores=[None],
                        ),
                    )
                    raise
                raise AssertionError("sleep should have been cancelled")  # pragma: no cover

        event = threading.Event()
        event.set()  # interrupt already pending
        manager = CancellingManager()
        outcomes: list[LevelOutcome] = []
        failures: list[Any] = []
        plans: list[Any] = [object()]
        with pytest.raises((asyncio.CancelledError, KeyboardInterrupt)):
            asyncio.run(
                _drive_levels(
                    manager,
                    plans,
                    outcomes,
                    failures,
                    interrupt_event=event,
                    poll_interval=0.001,
                )
            )
        assert len(failures) == 1
        assert failures[0].reason == "aborted: cancelled"
        assert failures[0].completed_cores == [None]

    def test_interrupt_without_aborted_level_records_warmup_partial(self) -> None:
        # A mid-warmup / mid-ramp interrupt raises a bare CancelledError (no
        # AbortedLevel): the driver synthesizes a traceable "interrupted (warmup)"
        # partial rather than leaving the level indistinguishable from
        # nothing-attempted, then propagates so __exit__ reaps (MF4).
        manager = FakeManager([asyncio.CancelledError()])
        outcomes: list[LevelOutcome] = []
        failures: list[Any] = []
        plans: list[Any] = [object()]
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(_drive_levels(manager, plans, outcomes, failures))
        assert len(failures) == 1
        assert failures[0].reason == "interrupted (warmup)"

    def test_system_exit_propagates_not_recorded(self) -> None:
        # SystemExit / GeneratorExit are BaseException-not-Exception: they must
        # propagate, never become "level failed, continue" (MF2).
        plans: list[Any] = [object()]
        for exc_cls in (SystemExit, GeneratorExit):
            manager = FakeManager([exc_cls()])
            outcomes: list[LevelOutcome] = []
            failures: list[Any] = []
            with pytest.raises(exc_cls):
                asyncio.run(_drive_levels(manager, plans, outcomes, failures))
            assert failures == []  # not recorded as a level failure


# ---------------------------------------------------------------------------
# Lifecycle - launch / ready / shutdown, cleanup exactly once
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_enter_launches_readies_and_marks_running(self) -> None:
        engine = FakeEngine()
        runner = _fake_runner()
        session = _make_session(engine, runner=runner)
        with session:
            assert engine.launched == 1
            assert engine.readied == 1
            assert session._handle is not None
            assert session._handle.base_url == engine.handle.base_url
            assert runner.manifest.statuses == ["running"]
        # Exit reaps the server exactly once.
        assert engine.shutdowns == 1

    def test_enter_failure_on_launch_reaps_and_reraises(self) -> None:
        engine = FakeEngine(launch_error=RuntimeError("no image"))
        session = _make_session(engine)
        with pytest.raises(RuntimeError, match="no image"), session:
            pass  # pragma: no cover - never entered
        # Launch failed before a handle existed: nothing to reap, but the failure
        # path ran cleanup once and never marked running.
        assert engine.shutdowns == 0
        assert session._runner.manifest.statuses == []

    def test_enter_failure_on_readiness_reaps_partial_server(self) -> None:
        engine = FakeEngine(ready_error=RuntimeError("never ready"))
        session = _make_session(engine)
        with pytest.raises(RuntimeError, match="never ready"), session:
            pass  # pragma: no cover - never entered
        # The server was launched then failed readiness: it MUST be reaped.
        assert engine.launched == 1
        assert engine.shutdowns == 1

    def test_exit_is_idempotent(self) -> None:
        engine = FakeEngine()
        session = _make_session(engine)
        session.__enter__()
        session.__exit__(None, None, None)
        session.__exit__(None, None, None)  # second exit is a no-op
        assert engine.shutdowns == 1

    def test_cleanup_runs_on_exception_in_body(self) -> None:
        engine = FakeEngine()
        session = _make_session(engine)
        with pytest.raises(RuntimeError, match="body boom"), session:
            raise RuntimeError("body boom")
        assert engine.shutdowns == 1

    def test_non_server_capable_engine_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import llenergymeasure.engines as engines_mod

        # get_engine returns a plain object that is NOT ServerCapable.
        monkeypatch.setattr(engines_mod, "get_engine", lambda name: SimpleNamespace())
        session = ServerSession(
            _fake_runner(), _server_config(), None, config_hash="h", cycle=1, index=1
        )
        with pytest.raises(ServerSessionError, match="does not support server mode"):
            session._resolve_engine()


# ---------------------------------------------------------------------------
# run() - N results, warmup provenance, manifest, contracts
# ---------------------------------------------------------------------------


class TestRun:
    def test_run_produces_n_results_with_warmup_provenance(self) -> None:
        engine = FakeEngine()
        runner = _fake_runner()
        session = _make_session(engine, runner=runner)
        warmup = FakeWarmup([_warmup_result(0)])
        manager = FakeManager([_level_outcome(0, n_windows=3, energy=10.0)])
        _wire_run(session, manager, warmup)

        with session:
            result = session.run()

        assert isinstance(result, ServerSessionResult)
        assert result.valid is True
        assert result.token_counting == ss.TOKEN_COUNTING_CLIENT_STREAMED
        # Warmup provenance stamped into EVERY window result (the cross-mode divergence label).
        level = result.levels[0]
        assert all(w.warmup is not None for w in level.windows)
        assert all(
            w.pre_window_protocol == "server convergence-composite warmup (test)"
            for w in level.windows
        )
        # Bookkeeping energy summed for the manifest summary.
        assert result.total_window_energy_j == pytest.approx(30.0)
        assert "completed" in runner.manifest.statuses

    def test_run_warmup_traffic_error_returns_failure_dict(self) -> None:
        engine = FakeEngine()
        runner = _fake_runner()
        session = _make_session(engine, runner=runner)
        warmup = FakeWarmup([])
        manager = FakeManager([WarmupTrafficError("warmup traffic died")])
        _wire_run(session, manager, warmup)

        with session:
            result = session.run()

        assert isinstance(result, dict)
        assert result["type"] == "WarmupTrafficError"
        assert "warmup failed" in result["message"]
        assert "failed" in runner.manifest.statuses
        assert engine.shutdowns == 1  # server still reaped

    def test_run_invalid_level_returns_failure_dict(self) -> None:
        engine = FakeEngine()
        runner = _fake_runner()
        session = _make_session(engine, runner=runner)
        warmup = FakeWarmup([_warmup_result(0)])
        manager = FakeManager([_level_outcome(0, valid=False)])
        _wire_run(session, manager, warmup)

        with session:
            result = session.run()

        assert isinstance(result, dict)
        assert result["type"] == "ServerSessionInvalid"
        assert "failed" in runner.manifest.statuses

    def test_run_interrupt_returns_partial_and_leaves_manifest_running(self) -> None:
        event = threading.Event()

        class SlowManager:
            def __init__(self) -> None:
                self.calls: list[int] = []

            async def run_level(self, level_index: int, level: Any) -> LevelOutcome:
                self.calls.append(level_index)
                event.set()  # trip the interrupt as soon as the level starts
                try:
                    await asyncio.sleep(10)
                except BaseException as exc:
                    setattr(
                        exc,
                        ABORTED_LEVEL_ATTR,
                        AbortedLevel(
                            level_index=level_index,
                            aborted_window_index=0,
                            reason="aborted: cancelled",
                            completed_cores=[None],
                        ),
                    )
                    raise
                raise AssertionError  # pragma: no cover

        engine = FakeEngine()
        runner = _fake_runner(interrupt_event=event)
        session = _make_session(engine, runner=runner)
        warmup = FakeWarmup([_warmup_result(0)])
        _wire_run(session, SlowManager(), warmup)

        with session:
            result = session.run()

        assert isinstance(result, ServerSessionResult)
        # Interrupted mid-session: the level failure is recorded, manifest NOT
        # resolved here (the sweep loop's mark_interrupted downgrades running).
        assert "completed" not in runner.manifest.statuses
        assert "failed" not in runner.manifest.statuses
        assert runner.manifest.statuses == ["running"]
        assert result.levels[0].invalid_reason == "aborted: cancelled"
        assert engine.shutdowns == 1  # server reaped on the interrupt path


# ---------------------------------------------------------------------------
# Contract 5 - the resolved warmup is read IN-PROCESS (no serialization boundary)
# ---------------------------------------------------------------------------


class TestResolvedWarmupSeam:
    def test_run_reads_overlay_resolved_warmup_in_process(self) -> None:
        # A user-config overlay attaches a resolved warmup as a PrivateAttr; the
        # session reads it directly (in-process), never through a serialized config.
        from llenergymeasure.config.precedence import apply_server_warmup_overlay

        config = _server_config()
        user = UserConfig(server=UserServerConfig(warmup=ServerWarmupConfig(mode="fixed")))
        apply_server_warmup_overlay(config, user)
        assert config.resolved_server_warmup().mode == "fixed"

        engine = FakeEngine()
        session = _make_session(engine, config=config)
        captured: dict[str, Any] = {}

        # Real _make_warmup reads resolved_server_warmup(); capture the config it uses.
        real_make_warmup = ServerSession._make_warmup

        def spy_make_warmup(self: ServerSession, shape: Any, transport: Any) -> Any:
            warmup = real_make_warmup(self, shape, transport)
            captured["mode"] = warmup._config.mode
            return warmup

        _override(session, "_make_warmup", spy_make_warmup.__get__(session, ServerSession))
        _override(session, "_make_manager", lambda sink, wu: FakeManager([_level_outcome(0)]))
        _override(session, "_make_level_plans", lambda shape, transport: [object()])

        with session:
            session.run()

        # The session ran the OVERLAY-RESOLVED protocol (fixed), read in-process.
        assert captured["mode"] == "fixed"


# ---------------------------------------------------------------------------
# Contract 3 - a failed level's cleanly-closed cores are preserved and
# their measured GPU energy still counts
# ---------------------------------------------------------------------------


class TestPreservedCores:
    def test_core_energy_j_flat_series(self) -> None:
        # Flat 100 W over 10 s -> 1000 J (trapezoidal of a flat series is exact).
        assert _core_energy_j(_flat_core(power_w=100.0, duration=10.0)) == pytest.approx(1000.0)

    def test_core_energy_j_none_and_too_short(self) -> None:
        assert _core_energy_j(None) is None
        assert _core_energy_j(SimpleNamespace(timeseries_samples=[])) is None
        one = SimpleNamespace(timeseries_samples=[PowerThermalSample(1000.0, 100.0, gpu_index=0)])
        assert _core_energy_j(one) is None

    def test_mid_level_abort_preserves_cores_and_energy(self) -> None:
        # The verifier's scenario: level 0 aborts on window 3 with 2 clean cores,
        # level 1 succeeds. Both cores land on the failed level's result and their
        # energy is counted in the session total (never silently dropped).
        core0 = _flat_core(power_w=100.0, duration=10.0)  # 1000 J
        core1 = _flat_core(power_w=50.0, duration=10.0)  # 500 J
        engine = FakeEngine()
        runner = _fake_runner()
        session = _make_session(engine, runner=runner)
        warmup = FakeWarmup([_warmup_result(0), _warmup_result(1)])
        manager = FakeManager(
            [_abort_exc(0, window=3, cores=[core0, core1]), _level_outcome(1, energy=10.0)]
        )
        _wire_run(session, manager, warmup, plans=[object(), object()])

        with session:
            result = session.run()

        assert isinstance(result, ServerSessionResult)
        assert result.valid is True  # level 1 passed
        failed_level = result.levels[0]
        assert failed_level.invalid_reason is not None
        assert failed_level.aborted_window_index == 3
        assert failed_level.completed_cores == [core0, core1]
        # 1000 + 500 (preserved cores) + 3 windows x 10 J (level 1) = 1530 J.
        assert result.total_window_energy_j == pytest.approx(1000.0 + 500.0 + 30.0)


# ---------------------------------------------------------------------------
# Real _make_level_plans - issuance horizon spans the whole level (contract 1)
# ---------------------------------------------------------------------------


class TestMakeLevelPlans:
    def test_make_level_plans_covers_full_horizon_and_wires_receipts(self) -> None:
        # Exercise the REAL (unstubbed) _make_level_plans: it builds one LevelPlan
        # whose traffic source issues across ramp + windows_per_level x duration and
        # whose token receipts flow through the interim server-reported seam.
        config = _server_config(traffic={"rate": 20, "window_seconds": 40, "seed": 3})
        engine = FakeEngine()
        session = ServerSession(
            _fake_runner(), config, None, config_hash="h", cycle=1, index=1, engine=engine
        )
        plans = session._make_level_plans(_FakeShapeSource(), SimpleNamespace())
        assert len(plans) == 1
        plan = plans[0]
        horizon = 30.0 + 3 * 40.0  # default ramp + windows_per_level x duration
        # The schedule spans the whole level, not just one 40 s window.
        assert plan.traffic_source.schedule.offsets[-1] > 40.0
        assert plan.traffic_source.schedule.offsets[-1] <= horizon
        assert plan.token_receipt_fn is client_token_receipts
        assert plan.spec.rate == 20.0
        assert plan.spec.duration_seconds == 40.0


# ---------------------------------------------------------------------------
# Real _make_placement - a server container is owned by its study
# ---------------------------------------------------------------------------


class TestMakePlacement:
    """The container leg carries the study's ownership labels.

    Without them a launched server container is invisible to the study-scoped
    cleanup and to the orphan reaper, so a server that outlives its launching
    process (the one case ``shutdown`` cannot cover) leaks holding the GPU.
    """

    @staticmethod
    def _session_with_identity(study_id: str | None, mode: str) -> ServerSession:
        from llenergymeasure.config.runner_spec import RunnerSpec

        runner = _fake_runner()
        runner.study.study_design_hash = study_id
        spec = RunnerSpec(mode=mode, image="img:v1" if mode == "container" else None, source="yaml")
        return ServerSession(
            runner, _server_config(), spec, config_hash="h", cycle=1, index=1, engine=FakeEngine()
        )

    def test_container_placement_carries_ownership_labels(self) -> None:
        session = self._session_with_identity("abcdef1234567890", "container")

        placement = session._make_placement()

        assert placement.mode == "container"
        assert placement.labels is not None
        assert placement.labels["llem.study_id"] == "abcdef1234567890"
        assert placement.labels["llem.parent_pid"] == str(os.getpid())

    def test_process_placement_has_no_labels(self) -> None:
        session = self._session_with_identity("abcdef1234567890", "process")

        placement = session._make_placement()

        assert placement.mode == "process"
        assert placement.labels is None

    def test_container_placement_without_study_identity_is_refused(self) -> None:
        from llenergymeasure.utils.exceptions import StudyError

        session = self._session_with_identity(None, "container")

        with pytest.raises(StudyError, match="study_design_hash"):
            session._make_placement()
