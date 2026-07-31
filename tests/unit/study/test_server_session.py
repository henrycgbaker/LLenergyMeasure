"""Tests for the server-mode measurement session (SM9).

Host-only, no GPU, no real server: the engine (ServerCapable), the window
manager, the energy sink, the traffic source, and the warmup hook are all
injected fakes, so the full launch -> warm up -> windows -> shutdown lifecycle is
exercised deterministically.

Charter (server-mode plan section 4 / section 18):
- one server lifetime produces N window results without re-keying the offline
  loop (C3);
- cleanup-exactly-once on the normal, exception, and interrupt paths, with the
  server reaped on every exit (the F6 session-hardening invariant);
- the three banked contracts: the issuance horizon covers ramp + N windows; the
  client-counted tokens flow through the TokenReceiptFn seam (interim,
  server-reported); an AbortedLevel is caught at the run_level await site and a
  WarmupTrafficError aborts the session;
- per-level ServerWarmupResult is stamped into each window result (D6 divergence
  label);
- the resolved warmup is read in-process (the serialization-boundary contract).
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest

from llenergymeasure.config.models import ExperimentConfig, ServerWarmupConfig
from llenergymeasure.config.user_config import UserConfig, UserServerConfig
from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.server_warmup import ServerWarmupResult, WarmupTrafficError
from llenergymeasure.harness.traffic import IssuerReport, RequestRecord, RequestShape
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
    WindowStartEvent,
    WindowStopEvent,
)
from llenergymeasure.study import server_session as ss
from llenergymeasure.study.server_session import (
    ServerSession,
    ServerSessionError,
    ServerSessionResult,
    _CompletionsShapeSource,
    _core_energy_j,
    _drive_levels,
    _level_traffic_source,
    server_reported_token_receipts,
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
    spec = WindowSpec(rate=10.0)
    boundaries = WindowBoundaries(window_start=0.0, span_start=1.0, span_end=2.0)
    bookkeeping = WindowBookkeeping(
        boundaries=boundaries,
        attribution_policy=ATTRIBUTION_STEADY_STATE_SPAN,
        energy_denominator_tokens=10,
        latency_records=[],
        issued_in_span_count=0,
        completed_in_span_count=0,
        straddling_count=0,
    )
    return WindowRecord(
        window_index=index,
        boundaries=boundaries,
        energy=None,
        bookkeeping=bookkeeping,
        window_energy_j=energy,
        window_j_per_token=j_per_token,
        intra_window_cov=0.01,
        start_event=WindowStartEvent(0, index, spec, 1.0),
        stop_event=WindowStopEvent(0, index, spec, 2.0),
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
# Contract 2 - token receipts flow from what the transport exposes (interim)
# ---------------------------------------------------------------------------


class TestTokenReceipts:
    def test_server_reported_tokens_stamped_at_completion(self) -> None:
        rec = RequestRecord(
            index=0,
            issued_at=1.0,
            request=RequestShape(index=0),
            completed_at=5.0,
            result={"usage": {"completion_tokens": 3}},
        )
        # Request-granular attribution: n tokens at completed_at (E2 rule).
        assert server_reported_token_receipts(rec) == (5.0, 5.0, 5.0)

    def test_no_usage_yields_no_receipts(self) -> None:
        rec = RequestRecord(
            index=0,
            issued_at=1.0,
            request=RequestShape(index=0),
            completed_at=5.0,
            result={"choices": [{"text": "pong"}]},
        )
        assert server_reported_token_receipts(rec) == ()

    def test_incomplete_or_errored_request_yields_no_receipts(self) -> None:
        never = RequestRecord(
            index=0, issued_at=1.0, request=RequestShape(index=0), completed_at=None
        )
        assert server_reported_token_receipts(never) == ()
        errored = RequestRecord(
            index=1,
            issued_at=1.0,
            request=RequestShape(index=1),
            completed_at=5.0,
            result={"usage": {"completion_tokens": 2}},
            error=RuntimeError("boom"),
        )
        assert server_reported_token_receipts(errored) == ()

    def test_zero_or_bool_token_count_rejected(self) -> None:
        for n in (0, -1, True, "3", None):
            rec = RequestRecord(
                index=0,
                issued_at=1.0,
                request=RequestShape(index=0),
                completed_at=5.0,
                result={"usage": {"completion_tokens": n}},
            )
            assert server_reported_token_receipts(rec) == ()


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
        # N results = one per window (C3).
        assert result.window_count == 3
        assert result.token_counting == ss.TOKEN_COUNTING_SERVER_REPORTED
        # Warmup provenance stamped into EVERY window result (point 4 / D6 label).
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
# Contract 3 / O7.4 - a failed level's cleanly-closed cores are preserved and
# their measured GPU energy still counts (the verifier's HIGH must-fix)
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
        assert plan.token_receipt_fn is server_reported_token_receipts
        assert plan.spec.rate == 20.0
        assert plan.spec.duration_seconds == 40.0
