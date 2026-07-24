"""Tests for the study.session seam - context-managed experiment sessions.

Focus: the lifecycle guarantees the sweep loop now leans on:
- ``ExperimentSession`` protocol conformance for both offline implementations;
- ``__exit__`` releases resources exactly once, on the normal path and on the
  interrupt/exception path alike (the SIGINT/circuit-break re-keying);
- the progress-consumer timeout loop tolerates a silent producer and still
  exits on the parent's sentinel.

The dispatch mechanics themselves (pipe drain ordering, kill escalation, docker
translation) are exercised end-to-end through ``StudyRunner.run`` in
tests/unit/study/test_study_runner.py; here we isolate the session lifecycle.
"""

from __future__ import annotations

import queue
import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.models import ExecutionConfig, ExperimentConfig, StudyConfig
from llenergymeasure.study.runner import StudyRunner
from llenergymeasure.study.session import (
    DockerSession,
    ExperimentSession,
    SubprocessSession,
)
from tests.conftest import TEST_CONFIG_HASH
from tests.unit.study.conftest import _make_mock_context, _make_mock_process

# =============================================================================
# Fixtures / helpers
# =============================================================================


def _make_config() -> ExperimentConfig:
    """A minimal ExperimentConfig with baseline disabled (keeps sessions hermetic)."""
    return ExperimentConfig(
        task={"model": "test/model"},
        engine="transformers",
        measurement={"baseline": {"enabled": False}},
        serving_mode="offline",
    )


def _make_study(config: ExperimentConfig) -> StudyConfig:
    return StudyConfig(
        experiments=[config],
        study_name="session-test",
        study_execution=ExecutionConfig(n_cycles=1, experiment_order="sequential"),
        study_design_hash=TEST_CONFIG_HASH,
    )


def _make_runner(tmp_path: Path, *, runner_specs: dict | None = None) -> StudyRunner:
    config = _make_config()
    study = _make_study(config)
    runner = StudyRunner(study, MagicMock(), tmp_path, runner_specs=runner_specs)
    # Env snapshot collection is a real background future; stub it out.
    runner._get_env_snapshot = MagicMock(return_value=MagicMock())  # type: ignore[method-assign]
    return runner


# =============================================================================
# Protocol conformance
# =============================================================================


def test_sessions_satisfy_experiment_session_protocol(tmp_path: Path) -> None:
    """Both offline sessions structurally satisfy the runtime-checkable protocol."""
    runner = _make_runner(tmp_path)
    config = runner.study.experiments[0]
    ctx = _make_mock_context(_make_mock_process())

    sub = SubprocessSession(runner, config, ctx, config_hash="h", cycle=1, index=1)
    assert isinstance(sub, ExperimentSession)

    from llenergymeasure.config.runner_spec import RunnerSpec

    spec = RunnerSpec(mode="container", image="img:v1", source="yaml")
    doc = DockerSession(runner, config, spec, config_hash="h", cycle=1, index=1)
    assert isinstance(doc, ExperimentSession)


# =============================================================================
# SubprocessSession: cleanup exactly once
# =============================================================================


def test_subprocess_session_cleanup_runs_once_normal_path(tmp_path: Path) -> None:
    """Normal path: __exit__ closes the pipe, clears the active handle, and removes
    the staging dir - exactly once, and a redundant __exit__ is a no-op."""
    runner = _make_runner(tmp_path)
    config = runner.study.experiments[0]
    proc = _make_mock_process(is_alive_after_join=False, exitcode=0)
    ctx = _make_mock_context(proc, pipe_data={"status": "ok"})
    parent_conn = ctx.Pipe.return_value[0]

    with (
        patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
        patch.object(runner, "_handle_result"),
    ):
        session = SubprocessSession(runner, config, ctx, config_hash="h", cycle=1, index=1)
        with session:
            result = session.run()

    assert result == {"status": "ok"}
    ts_dir = session._ts_tmpdir
    assert ts_dir is not None and not ts_dir.exists(), "staging dir was not removed on exit"
    assert runner._active_process is None
    parent_conn.close.assert_called_once()

    # A second __exit__ (idempotent guard) must not re-run teardown.
    session.__exit__(None, None, None)
    parent_conn.close.assert_called_once()


def test_subprocess_session_cleanup_runs_once_on_exception(tmp_path: Path) -> None:
    """Interrupt/exception path: an error inside the ``with`` body still triggers a
    single teardown - the live worker is reaped, the pipe closed, the staging dir
    removed - which is the SIGINT/circuit-break cleanup guarantee the loop leans on."""
    runner = _make_runner(tmp_path)
    config = runner.study.experiments[0]
    # is_alive=True: the body raised before run()'s own join/kill sequence, so
    # __exit__ must reap the still-running worker.
    proc = _make_mock_process(is_alive_after_join=True, pid=5150)
    ctx = _make_mock_context(proc, pipe_has_data=False)
    parent_conn = ctx.Pipe.return_value[0]

    with (
        patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
        patch("llenergymeasure.study.session._kill_process_group") as mock_kill,
    ):
        session = SubprocessSession(runner, config, ctx, config_hash="h", cycle=1, index=1)
        ts_dir_before = None
        with pytest.raises(RuntimeError, match="boom"), session:
            ts_dir_before = session._ts_tmpdir
            raise RuntimeError("boom")

    # Teardown ran exactly once despite the exception.
    assert ts_dir_before is not None and not ts_dir_before.exists()
    assert runner._active_process is None
    parent_conn.close.assert_called_once()
    mock_kill.assert_called_once_with(proc.pid, signal.SIGKILL)
    proc.join.assert_called()

    # Idempotent: a second __exit__ does not kill or close again.
    session.__exit__(None, None, None)
    parent_conn.close.assert_called_once()
    mock_kill.assert_called_once()


def test_subprocess_session_active_process_set_during_enter(tmp_path: Path) -> None:
    """__enter__ registers the worker as the runner's active process (SIGINT target)."""
    runner = _make_runner(tmp_path)
    config = runner.study.experiments[0]
    proc = _make_mock_process()
    ctx = _make_mock_context(proc, pipe_data={"status": "ok"})

    with patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"):
        session = SubprocessSession(runner, config, ctx, config_hash="h", cycle=1, index=1)
        session.__enter__()
        try:
            assert runner._active_process is proc
        finally:
            session.__exit__(None, None, None)
    assert runner._active_process is None


def test_subprocess_session_enter_failure_releases_resources(tmp_path: Path) -> None:
    """__enter__ raising at the pre-dispatch GPU-residual check must not strand
    resources. The check fires after the staging dir, pipe, and consumer thread are
    acquired but before the worker starts; since a failing __enter__ means the ``with``
    statement never calls __exit__, __enter__'s own failure path releases what it
    acquired and re-raises."""
    runner = _make_runner(tmp_path)
    config = runner.study.experiments[0]
    proc = _make_mock_process(is_alive_after_join=False)
    ctx = _make_mock_context(proc, pipe_has_data=False)
    parent_conn, child_conn = ctx.Pipe.return_value

    with patch(
        "llenergymeasure.study.gpu_memory.check_gpu_memory_residual",
        side_effect=RuntimeError("residual GPU memory"),
    ):
        session = SubprocessSession(runner, config, ctx, config_hash="h", cycle=1, index=1)
        # The exception propagates out of __enter__ (the `with` body never runs).
        with pytest.raises(RuntimeError, match="residual GPU memory"), session:
            pass

    # The staging dir was created then removed; the consumer thread was started
    # then stopped; both pipe ends were closed (the residual check raised before
    # the normal-path child close ran); the active handle was cleared.
    ts_dir = session._ts_tmpdir
    assert ts_dir is not None and not ts_dir.exists(), "staging dir leaked on __enter__ failure"
    assert session._consumer is not None and not session._consumer.is_alive(), (
        "progress consumer thread stranded on __enter__ failure"
    )
    assert session._consumer_stopped is True
    parent_conn.close.assert_called_once()
    child_conn.close.assert_called_once()
    assert runner._active_process is None


def test_subprocess_session_cleanup_finishes_when_stop_consumer_raises(tmp_path: Path) -> None:
    """_cleanup completes its remaining steps when _stop_consumer raises: the pipe is
    still closed and the staging dir still removed. The consumer-stop call is guarded
    like its process-reap and pipe-close siblings, so a raise there cannot skip them."""
    runner = _make_runner(tmp_path)
    config = runner.study.experiments[0]
    proc = _make_mock_process(is_alive_after_join=False)
    ctx = _make_mock_context(proc, pipe_has_data=False)
    parent_conn = ctx.Pipe.return_value[0]

    with patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"):
        session = SubprocessSession(runner, config, ctx, config_hash="h", cycle=1, index=1)
        session.__enter__()

    ts_dir = session._ts_tmpdir
    assert ts_dir is not None and ts_dir.exists()
    consumer = session._consumer

    # _stop_consumer raises during teardown; the remaining steps must still run.
    with patch.object(session, "_stop_consumer", side_effect=RuntimeError("consumer boom")):
        session.__exit__(None, None, None)

    assert not ts_dir.exists(), "staging dir must be removed even when _stop_consumer raises"
    parent_conn.close.assert_called_once()
    assert runner._active_process is None

    # _stop_consumer was stubbed out, so its sentinel never reached the real
    # consumer thread; reap it here so the test leaves no live daemon behind.
    if consumer is not None:
        session._progress_queue.put(None)
        consumer.join(timeout=2.0)


# =============================================================================
# DockerSession: cleanup exactly once
# =============================================================================


def test_docker_session_cleanup_removes_staging_dir_once(tmp_path: Path) -> None:
    """DockerSession.__exit__ removes the container staging dir exactly once."""
    from llenergymeasure.config.runner_spec import RunnerSpec

    spec = RunnerSpec(mode="container", image="img:v1", source="yaml")
    runner = _make_runner(tmp_path, runner_specs={"transformers": spec})
    runner._images_prepared = True
    config = runner.study.experiments[0]

    docker_ts_dir = tmp_path / "docker-ts"
    docker_ts_dir.mkdir()
    fake_result = MagicMock()
    fake_result.total_energy_j = 1.0

    class _FakeDockerRunner:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def run(self, *args: Any, **kwargs: Any) -> tuple[Any, Path]:
            return fake_result, docker_ts_dir

    with (
        patch("llenergymeasure.infra.docker_runner.DockerRunner", _FakeDockerRunner),
        patch("llenergymeasure.study.gpu_memory.check_gpu_memory_residual"),
        patch.object(runner, "_handle_result"),
    ):
        session = DockerSession(runner, config, spec, config_hash="h", cycle=1, index=1)
        with session:
            session.run()

    assert not docker_ts_dir.exists(), "docker staging dir was not removed on exit"
    # Idempotent second exit does not raise even though the dir is already gone.
    session.__exit__(None, None, None)


# =============================================================================
# Progress-consumer timeout loop (decision 4)
# =============================================================================


class _ScriptedQueue:
    """A queue whose get() replays a script; the ``_EMPTY`` marker raises queue.Empty."""

    EMPTY = object()

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)

    def get(self, timeout: float | None = None) -> Any:
        item = self._script.pop(0)
        if item is self.EMPTY:
            raise queue.Empty
        return item


def test_progress_consumer_tolerates_empty_then_exits_on_sentinel() -> None:
    """The consumer survives ``queue.Empty`` timeouts (silent producer) and exits
    on the parent's None sentinel, forwarding the real events it did receive."""
    from llenergymeasure.study._progress import _consume_progress_events

    progress = MagicMock()
    q = _ScriptedQueue(
        [
            {"event": "step_done", "step": "baseline", "elapsed_sec": 1.5},
            _ScriptedQueue.EMPTY,  # producer silent this tick - must not hang/crash
            _ScriptedQueue.EMPTY,
            None,  # parent sentinel ends the loop
        ]
    )

    # Returns (does not hang) despite the intervening Empty ticks.
    _consume_progress_events(q, study_progress=progress)

    progress.on_step_done.assert_called_once_with("baseline", 1.5)


def test_progress_consumer_thread_stays_responsive_on_silent_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A daemon consumer against a live-but-silent queue keeps polling (does not die
    on Empty) and exits promptly once the sentinel arrives."""
    import threading
    import time

    from llenergymeasure.study._progress import _consume_progress_events

    monkeypatch.setattr("llenergymeasure.study._progress.TIMEOUT_PROGRESS_QUEUE_POLL", 0.01)

    q: queue.Queue = queue.Queue()
    progress = MagicMock()
    q.put({"event": "step_done", "step": "s", "elapsed_sec": 0.1})

    t = threading.Thread(target=_consume_progress_events, args=(q, progress), daemon=True)
    t.start()
    time.sleep(0.1)  # several poll ticks elapse with an empty queue
    assert t.is_alive(), "consumer must keep polling on an empty queue, not exit"

    q.put(None)  # sentinel
    t.join(timeout=2.0)
    assert not t.is_alive(), "consumer must exit on the sentinel"
    progress.on_step_done.assert_called_once_with("s", 0.1)
