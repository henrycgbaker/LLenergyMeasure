"""Unit tests for the engine-agnostic server lifecycle mechanics.

The FULL lifecycle is exercised against the PROCESS leg using the asyncio stub
server (launch / liveness / real probe / ready / shutdown / kill-escalation /
idempotence / failed-launch cleanup / log access). The container leg is tested
without invoking docker: command construction is pure, and stop / remove / logs
are asserted by mocking ``subprocess.run``. The DooD topology error is tested
with monkeypatched container-self detection.
"""

from __future__ import annotations

import signal
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.infra import server_lifecycle as sl
from llenergymeasure.infra.server_lifecycle import (
    ProbeRequest,
    ServerHandle,
    ServerLaunchError,
    ServerReadinessError,
    ServerTopologyError,
)

COMPLETIONS_PROBE = ProbeRequest(
    path="/v1/completions",
    payload={"model": "stub", "prompt": "ping", "max_tokens": 1},
)


# ---------------------------------------------------------------------------
# Full lifecycle against the process leg
# ---------------------------------------------------------------------------


def test_full_lifecycle_process_leg(launch_stub):
    """launch -> liveness -> real probe -> ready -> log access -> shutdown."""
    handle = launch_stub()

    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)

    # Log access is the SM9 failure-artefact hand-off: the stub announced itself.
    assert "stub server listening" in handle.read_logs()
    assert handle.identity.startswith("process pid=")

    sl.shutdown(handle)
    assert handle.process is not None
    assert handle.process.poll() is not None  # reaped, no orphan


def test_readiness_requires_real_probe_not_just_health(launch_stub):
    """/health passing is NEVER sufficient (R8): an always-503 probe times out."""
    # health is 200 immediately, but /v1/completions stays 503 for far longer
    # than the readiness timeout, so readiness must fail on the probe phase.
    handle = launch_stub(completions_ready_after=9999.0)

    with pytest.raises(ServerReadinessError) as excinfo:
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=2.0, poll_interval=0.2)

    assert "readiness probe" in str(excinfo.value)


def test_probe_becomes_ready_after_model_loads(launch_stub):
    """Readiness succeeds once the real probe starts returning 200."""
    handle = launch_stub(completions_ready_after=1.0)

    # Would fail if we accepted /health alone (200 from the start); succeeds
    # only because we wait for the real probe to clear its 503 window.
    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)


# ---------------------------------------------------------------------------
# Shutdown: graceful, kill escalation, idempotence, leak-free
# ---------------------------------------------------------------------------


def test_shutdown_graceful(launch_stub):
    """A server that honours SIGTERM is stopped without escalation."""
    handle = launch_stub()
    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)

    sl.shutdown(handle, grace=5.0)

    assert handle.process is not None
    assert handle.process.returncode == -signal.SIGTERM


def test_shutdown_kill_escalation(launch_stub):
    """A server that ignores SIGTERM is escalated to SIGKILL after the grace."""
    handle = launch_stub(ignore_sigterm=True)
    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)

    grace = 1.0
    t0 = time.monotonic()
    sl.shutdown(handle, grace=grace)
    elapsed = time.monotonic() - t0

    assert handle.process is not None
    # SIGTERM was ignored, so the process was hard-killed: escalation happened.
    assert handle.process.returncode == -signal.SIGKILL
    # And only after waiting out the grace period (not an immediate kill).
    assert elapsed >= grace * 0.5


def test_shutdown_idempotent(launch_stub):
    """A second shutdown is a no-op and never raises."""
    handle = launch_stub()
    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)

    sl.shutdown(handle)
    first_returncode = handle.process.returncode  # type: ignore[union-attr]

    sl.shutdown(handle)  # must not raise, must not change state
    assert handle.process.returncode == first_returncode  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Failed launch cleans its own partial state
# ---------------------------------------------------------------------------


def test_failed_process_launch_cleans_log_file(tmp_path):
    """A launch that cannot start reaps its own log file and raises."""
    log_path = tmp_path / "wont-start.log"

    with pytest.raises(ServerLaunchError):
        sl.launch_process_server(
            ["/nonexistent/llem-not-a-real-binary"],
            base_url="http://127.0.0.1:1",
            engine="stub",
            log_path=log_path,
        )

    assert not log_path.exists()


def test_process_exit_during_startup_is_detected(tmp_path):
    """A server that exits immediately fails readiness fast (not a timeout)."""
    port = sl.allocate_free_port()
    handle = sl.launch_process_server(
        [sys.executable, "-c", "import sys; sys.exit(3)"],
        base_url=sl.base_url_for(port),
        engine="stub",
        log_path=tmp_path / "quick-exit.log",
    )
    try:
        with pytest.raises(ServerLaunchError) as excinfo:
            sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)
        assert "exited during startup" in str(excinfo.value)
    finally:
        sl.shutdown(handle)


# ---------------------------------------------------------------------------
# DooD topology error (monkeypatched container-self detection)
# ---------------------------------------------------------------------------


def _patch_dood(monkeypatch, *, in_container: bool, socket_available: bool) -> None:
    from llenergymeasure.infra import runner_resolution

    monkeypatch.setattr(runner_resolution, "is_running_in_container", lambda: in_container)
    monkeypatch.setattr(
        runner_resolution, "is_container_socket_available", lambda: socket_available
    )


def test_dood_topology_raises_actionable_error(monkeypatch):
    """A sibling container unreachable under DooD raises a topology error, not a timeout."""
    _patch_dood(monkeypatch, in_container=True, socket_available=True)
    # Container-leg handle pointing at an unbound port -> every connect fails.
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-test",
    )

    with pytest.raises(ServerTopologyError) as excinfo:
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=2.0, poll_interval=0.2)

    assert "--network host" in str(excinfo.value)


def test_dood_check_skipped_for_process_leg(monkeypatch):
    """Process-in-container reaches localhost fine: no false topology error."""
    _patch_dood(monkeypatch, in_container=True, socket_available=True)
    # Process-leg handle (no container_name) with no live server -> plain timeout.
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
    )

    with pytest.raises(ServerReadinessError):
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=1.0, poll_interval=0.2)


def test_dood_check_skipped_when_not_in_container(monkeypatch):
    """On a bare host a container-leg connection failure is a plain timeout."""
    _patch_dood(monkeypatch, in_container=False, socket_available=True)
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-test",
    )

    with pytest.raises(ServerReadinessError):
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=1.0, poll_interval=0.2)


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------


def test_allocate_free_port_is_bindable():
    """The allocated port is free enough to bind again immediately."""
    import socket

    port = sl.allocate_free_port()
    assert 1 <= port <= 65535
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))  # must not raise


# ---------------------------------------------------------------------------
# Container leg without docker (command construction + mocked CLI)
# ---------------------------------------------------------------------------


def test_container_argv_has_ruled_flags():
    """docker run argv carries image, --gpus, --network host, and the port."""
    argv = sl.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["Qwen/Qwen2.5-0.5B", "--port", "8123"],
        shm_size="8g",
    )

    assert argv[:2] == ["docker", "run"]
    assert "-d" in argv and "--rm" in argv
    # --network host is unconditional and adjacent.
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--name") + 1] == "llem-vllm-server-abc"
    # Image precedes the serve args; the port lives in the serve args (host net,
    # so no -p publish is emitted).
    img_idx = argv.index("vllm/vllm-openai:v0.19.1")
    assert argv[img_idx + 1 :] == ["Qwen/Qwen2.5-0.5B", "--port", "8123"]
    assert "-p" not in argv


def test_launch_container_server_success_and_cleanup_on_failure():
    """A non-zero docker run force-removes the container; success returns a handle."""
    # Success: docker run -d returns 0 with a container id on stdout.
    ok = MagicMock(returncode=0, stdout="container-id\n", stderr="")
    with patch("subprocess.run", return_value=ok):
        handle = sl.launch_container_server(
            ["docker", "run", "-d", "img"],
            base_url="http://127.0.0.1:8000",
            engine="vllm",
            container_name="llem-vllm-server-ok",
        )
    assert handle.container_name == "llem-vllm-server-ok"
    assert handle.process is None

    # Failure: non-zero exit -> ServerLaunchError AND a force-remove is issued.
    calls: list[list[str]] = []

    def _fake_run(argv, *a, **k):
        calls.append(argv)
        if argv[:3] == ["docker", "run", "-d"]:
            return MagicMock(returncode=1, stdout="", stderr="boom")
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run), pytest.raises(ServerLaunchError):
        sl.launch_container_server(
            ["docker", "run", "-d", "img"],
            base_url="http://127.0.0.1:8000",
            engine="vllm",
            container_name="llem-vllm-server-bad",
        )
    assert any(c[:3] == ["docker", "rm", "-f"] for c in calls), "failed launch must force-remove"


def test_shutdown_container_stops_then_force_removes():
    """Container shutdown issues docker stop (escalating) then docker rm -f."""
    handle = ServerHandle(
        base_url="http://127.0.0.1:8000",
        engine="vllm",
        container_name="llem-vllm-server-xyz",
    )
    calls: list[list[str]] = []

    def _fake_run(argv, *a, **k):
        calls.append(argv)
        return MagicMock(returncode=0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=_fake_run):
        sl.shutdown(handle, grace=3.0)

    assert ["docker", "stop", "-t", "3", "llem-vllm-server-xyz"] in calls
    assert any(c[:3] == ["docker", "rm", "-f"] for c in calls)

    # Idempotent: a second shutdown does nothing further.
    calls.clear()
    with patch("subprocess.run", side_effect=_fake_run):
        sl.shutdown(handle)
    assert calls == []


def test_read_logs_container_leg_shells_docker_logs():
    """read_logs on a container handle shells `docker logs`."""
    handle = ServerHandle(
        base_url="http://127.0.0.1:8000",
        engine="vllm",
        container_name="llem-vllm-server-logs",
    )
    proc = MagicMock(stdout="line one\n", stderr="")
    with patch("subprocess.run", return_value=proc) as run:
        logs = handle.read_logs(tail_lines=10)
    assert "line one" in logs
    argv = run.call_args.args[0]
    assert argv[:2] == ["docker", "logs"]
    assert "llem-vllm-server-logs" in argv
