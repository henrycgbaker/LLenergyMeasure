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

    # Log access is the failure-artefact hand-off: the stub announced itself.
    assert "stub server listening" in handle.read_logs()
    assert handle.identity.startswith("process pid=")

    sl.shutdown(handle)
    assert handle.process is not None
    assert handle.process.poll() is not None  # reaped, no orphan


def test_readiness_requires_real_probe_not_just_health(launch_stub):
    """/health passing is NEVER sufficient: an always-503 probe times out."""
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
    """Permanent DooD-unreachability raises a topology error at the DEADLINE, not eagerly.

    A sibling container that never becomes reachable (unbound port -> every probe
    a transport-level connection failure) raises the actionable topology error -
    but only once the whole readiness budget is spent, never on the first poll
    (the eager-abort defect). The captured `docker logs` tail is attached so a
    container that is in fact healthy and listening on the real host can be told
    apart from a crash.
    """
    _patch_dood(monkeypatch, in_container=True, socket_available=True)
    # The container is reported running, so the connection failure is the topology,
    # not a crash (and no real `docker inspect` runs).
    monkeypatch.setattr(sl, "_container_running", lambda name: True)
    monkeypatch.setattr(sl.ServerHandle, "read_logs", lambda self, **k: "listening on host")
    # Container-leg handle pointing at an unbound port -> every connect fails.
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-test",
    )

    timeout = 1.0
    t0 = time.monotonic()
    with pytest.raises(ServerTopologyError) as excinfo:
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=timeout, poll_interval=0.2)
    elapsed = time.monotonic() - t0

    # Diagnosed at the deadline, not on the first poll: the loop polled for about
    # the whole budget before concluding topology (a first-poll abort would land
    # in a small fraction of the timeout).
    assert elapsed >= timeout * 0.5
    assert "--network host" in str(excinfo.value)
    # The docker logs tail rides along on the topology error.
    assert "listening on host" in str(excinfo.value)


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
    monkeypatch.setattr(sl, "_container_running", lambda name: True)
    monkeypatch.setattr(sl.ServerHandle, "read_logs", lambda self, **k: "")
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-test",
    )

    with pytest.raises(ServerReadinessError):
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=1.0, poll_interval=0.2)


def test_delayed_listen_is_not_aborted_under_dood(monkeypatch, launch_stub):
    """A sibling that starts listening AFTER a startup window is NOT aborted eagerly.

    This is the regression the eager first-poll DooD check hid: with DooD detected,
    the normal startup-window ECONNREFUSED (the server is still loading and not yet
    listening) must be polled through, not read as permanent unreachability. A real
    stub refuses connections for ~1s (several poll intervals) then binds and serves;
    driving it through the CONTAINER leg (where the topology check lives) under
    patched DooD detection, readiness must SUCCEED well within the timeout.
    """
    _patch_dood(monkeypatch, in_container=True, socket_available=True)
    monkeypatch.setattr(sl, "_container_running", lambda name: True)
    # Real stub, refuses connections for the first second, then listens + serves.
    proc_handle = launch_stub(listen_after=1.0)
    # Exercise the container leg against that same real server: a container-leg
    # handle carrying the process's base_url. During the window every probe is a
    # connection failure (the topology-error trigger under the old eager check).
    handle = ServerHandle(
        base_url=proc_handle.base_url,
        engine="vllm",
        container_name="llem-vllm-server-delayed",
    )

    # Must NOT raise: the loop polls through the connection-refused window and
    # succeeds once the stub binds and the real probe returns 200.
    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=20.0, poll_interval=0.2)


def test_http_error_then_success_under_dood(monkeypatch):
    """HTTP errors during startup under DooD still resolve to readiness (never topology).

    A server that answers (503 while the model loads) is reachable, so even under
    DooD detection it must simply wait for the 200 - an HTTP-level answer is never
    a topology signal.
    """
    _patch_dood(monkeypatch, in_container=True, socket_available=True)
    monkeypatch.setattr(sl, "_container_running", lambda name: True)
    calls = {"n": 0}

    def _probe(url, *, method, payload, timeout):
        # 503 while the model loads, then 200 steadily (both readiness phases).
        calls["n"] += 1
        return 503 if calls["n"] <= 2 else 200

    monkeypatch.setattr(sl, "_http_probe", _probe)
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-503",
    )

    # 503, 503, then 200: the liveness phase clears, then the real probe clears.
    sl.await_ready(handle, COMPLETIONS_PROBE, timeout=5.0, poll_interval=0.05)


def test_deadline_with_mixed_failures_is_readiness_not_topology(monkeypatch):
    """A deadline after MIXED failures is a readiness timeout, not a topology error.

    When some probes failed at the transport level but at least one got an HTTP
    answer, the server WAS reachable at least once, so unreachability cannot be
    the explanation - the deadline must raise ServerReadinessError even though the
    DooD topology is detected.
    """
    _patch_dood(monkeypatch, in_container=True, socket_available=True)
    monkeypatch.setattr(sl, "_container_running", lambda name: True)
    monkeypatch.setattr(sl.ServerHandle, "read_logs", lambda self, **k: "still loading")
    calls = {"n": 0}

    def _mixed(url, *, method, payload, timeout):
        calls["n"] += 1
        # Alternate transport failures with a 503 answer; never 200 -> deadline.
        if calls["n"] % 2 == 0:
            raise sl._ConnectionFailed("connection refused")
        return 503

    monkeypatch.setattr(sl, "_http_probe", _mixed)
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-mixed",
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
    assert "-d" in argv
    # No --rm: a crashed container must survive so `docker logs` can recover the
    # startup diagnostic (the failure-artefact hand-off); leak-freeness is explicit in shutdown.
    assert "--rm" not in argv
    # --network host is unconditional and adjacent.
    assert argv[argv.index("--network") + 1] == "host"
    assert argv[argv.index("--gpus") + 1] == "all"
    assert argv[argv.index("--name") + 1] == "llem-vllm-server-abc"
    # Image precedes the serve args; the port lives in the serve args (host net,
    # so no -p publish is emitted).
    img_idx = argv.index("vllm/vllm-openai:v0.19.1")
    assert argv[img_idx + 1 :] == ["Qwen/Qwen2.5-0.5B", "--port", "8123"]
    assert "-p" not in argv


def test_container_argv_carries_ownership_labels():
    """Ownership labels are emitted before the image, so the study owns the server.

    The study-scoped cleanup filters on ``llem.study_id`` and the orphan reaper on
    ``llem.parent_pid``; an unlabelled server container is invisible to both.
    """
    argv = sl.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["m", "--port", "8123"],
        shm_size="8g",
        labels={"llem.study_id": "abcdef12", "llem.parent_pid": "4242"},
    )

    assert "llem.study_id=abcdef12" in argv
    assert "llem.parent_pid=4242" in argv
    for value in ("llem.study_id=abcdef12", "llem.parent_pid=4242"):
        idx = argv.index(value)
        assert argv[idx - 1] == "--label"
        # docker run options must precede the image name.
        assert idx < argv.index("vllm/vllm-openai:v0.19.1")


def test_container_argv_without_labels_emits_none():
    """No labels supplied (e.g. a direct non-study launch) emits no --label flags."""
    argv = sl.build_server_container_argv(
        image="img:v1",
        container_name=None,
        gpu_indices=None,
        serve_args=["m"],
    )

    assert "--label" not in argv


def test_container_argv_mounts_hf_cache(monkeypatch):
    """The server container binds the HF cache + sets HF_HOME (else weights re-download).

    Same LLEM_DOCKER_HF_CACHE-driven mount the offline docker dispatch uses; the
    mount/env precede the image (docker run options come before the image name).
    """
    monkeypatch.setenv("LLEM_DOCKER_HF_CACHE", "/data/hf")
    argv = sl.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=None,
        serve_args=["m", "--port", "8123"],
        shm_size="8g",
    )
    target = "/root/.cache/huggingface"
    # -v <host>:<target> present, and it precedes the image.
    assert f"/data/hf:{target}" in argv
    mount_idx = argv.index(f"/data/hf:{target}")
    assert argv[mount_idx - 1] == "-v"
    assert mount_idx < argv.index("vllm/vllm-openai:v0.19.1")
    # HF_HOME points at the in-container target.
    assert f"HF_HOME={target}" in argv


def test_container_argv_is_pinned_exactly(monkeypatch):
    """The whole argv is pinned, flag order included, for one fully-specified launch.

    The other argv tests assert individual flags and their adjacency; this one
    fixes the complete list so a reordering or an accidental extra flag cannot
    slip through. Every env-driven input is pinned so the expectation is stable
    on any host.
    """
    monkeypatch.delenv("LLEM_DOCKER_GPUS", raising=False)
    monkeypatch.setenv("LLEM_DOCKER_HF_CACHE", "/data/hf")

    argv = sl.build_server_container_argv(
        image="vllm/vllm-openai:v0.19.1",
        container_name="llem-vllm-server-abc",
        gpu_indices=[2, 3],
        serve_args=["Qwen/Qwen2.5-0.5B", "--port", "8123"],
        shm_size="8g",
        labels={"llem.study_id": "abcdef12", "llem.parent_pid": "4242"},
    )

    assert argv == [
        "docker",
        "run",
        "-d",
        "--network",
        "host",
        "--gpus",
        '"device=2,3"',
        "--name",
        "llem-vllm-server-abc",
        "--label",
        "llem.study_id=abcdef12",
        "--label",
        "llem.parent_pid=4242",
        "--shm-size",
        "8g",
        "-v",
        "/data/hf:/root/.cache/huggingface",
        "-e",
        "HF_HOME=/root/.cache/huggingface",
        "vllm/vllm-openai:v0.19.1",
        "Qwen/Qwen2.5-0.5B",
        "--port",
        "8123",
    ]


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


def test_container_crash_during_startup_captures_logs(monkeypatch):
    """A container that exits during startup fails fast with its logs (not lost to --rm)."""
    _patch_dood(monkeypatch, in_container=False, socket_available=False)
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-crash",
    )
    calls: list[list[str]] = []

    def _fake_run(argv, *a, **k):
        calls.append(argv)
        if argv[:2] == ["docker", "inspect"]:
            return MagicMock(returncode=0, stdout="false\n", stderr="")  # exists but stopped
        if argv[:2] == ["docker", "logs"]:
            return MagicMock(returncode=0, stdout="CUDA error: out of memory\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(ServerLaunchError) as excinfo,
    ):
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=5.0, poll_interval=0.2)

    # The diagnostic --rm would have destroyed is preserved in the error, and the
    # crashed container is force-removed only AFTER its logs were captured.
    assert "out of memory" in str(excinfo.value)
    assert any(c[:3] == ["docker", "rm", "-f"] for c in calls)


def test_container_readiness_timeout_captures_logs(monkeypatch):
    """The container-leg readiness timeout carries `docker logs` (--rm no longer eats them)."""
    _patch_dood(monkeypatch, in_container=False, socket_available=False)
    handle = ServerHandle(
        base_url=sl.base_url_for(sl.allocate_free_port()),
        engine="vllm",
        container_name="llem-vllm-server-slow",
    )

    def _fake_run(argv, *a, **k):
        if argv[:2] == ["docker", "inspect"]:
            return MagicMock(returncode=0, stdout="true\n", stderr="")  # running, never ready
        if argv[:2] == ["docker", "logs"]:
            return MagicMock(returncode=0, stdout="still loading weights\n", stderr="")
        return MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("subprocess.run", side_effect=_fake_run),
        pytest.raises(ServerReadinessError) as excinfo,
    ):
        sl.await_ready(handle, COMPLETIONS_PROBE, timeout=1.0, poll_interval=0.2)

    assert "still loading weights" in str(excinfo.value)
