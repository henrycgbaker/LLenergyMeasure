"""Engine-agnostic server lifecycle mechanics for online-serving measurement.

This is the infra-altitude plumbing behind the ``ServerCapable`` engine-plugin
extension (see :mod:`llenergymeasure.engines.protocol`). It owns the concerns
that are identical for every OpenAI-compatible inference server - port
allocation, launching the server (a detached sibling container OR a host
subprocess in its own process group), the readiness wait, and leak-free
shutdown with kill escalation. Per-engine knowledge (the ``vllm serve`` command,
the health path, the probe request shape) lives in the engine adapters, which
compose these primitives.

Design constraints this module enforces:

- **Sibling, not a tightening:** nothing here touches ``run_inference`` or
  the offline dispatch path. The long-lived server lifecycle (launch -> ready ->
  stop) is a parallel sibling of the run-to-completion batch dispatch that
  ``DockerRunner`` owns.
- **Readiness is a real probe:** :func:`await_ready` polls liveness THEN
  drives a real inference request through the serving path; a passing ``/health``
  never satisfies readiness on its own.
- **Leak-free:** :func:`shutdown` is idempotent and escalates SIGTERM -> SIGKILL
  (process leg, via the ``killpg`` process-group precedent) or ``docker stop``
  -> ``docker rm -f`` (container leg). A failed launch removes its own partial
  state (the container is force-removed, the process log file is cleaned up).
- **DooD reachability:** when llenergymeasure itself runs inside a container and
  dispatches a sibling server with ``--network host``, ``localhost`` does not
  route to the sibling unless llenergymeasure's own container is also
  ``--network host``. Rather than a-priori network-mode detection (ruled out), a
  transient connection failure is NEVER treated as terminal - the readiness loop
  keeps polling - and only once the whole deadline is spent with every probe
  having failed at the transport level does the unreachability surface as a
  topology-specific actionable error (see :func:`_is_dood_topology`) instead of a
  generic readiness timeout.

Probe transport: a narrow stdlib ``urllib`` client (no ``httpx`` / third-party
dependency, so no server extra is pulled in just to probe readiness). The
traffic-issuer transport seam (the ``TrafficSource``) is a separate,
richer async client; unifying this readiness probe behind that seam is deferred
to the server session, which owns both the issuer and the readiness wait.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llenergymeasure.config.ssot import (
    RUNNER_CONTAINER,
    TIMEOUT_DOCKER_CLI,
    TIMEOUT_SIGTERM_GRACE,
    RunnerMode,
)

# The server container's `docker run` argv is built beside the other container
# shapes, in the one argv-construction home. Re-exported here so a server
# adapter composes launch, readiness and shutdown from this module alone.
from llenergymeasure.infra.docker.command import build_server_container_argv
from llenergymeasure.utils.exceptions import LLEMError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HEALTH_PATH",
    "ProbeRequest",
    "ServerHandle",
    "ServerLaunchError",
    "ServerLifecycleError",
    "ServerPlacement",
    "ServerReadinessError",
    "ServerTopologyError",
    "allocate_free_port",
    "await_ready",
    "build_server_container_argv",
    "default_server_log_path",
    "launch_container_server",
    "launch_process_server",
    "server_container_name",
    "shutdown",
]

#: The liveness endpoint OpenAI-compatible servers (vLLM, trtllm-serve,
#: transformers serve) expose. Overridable per engine at :func:`await_ready`.
DEFAULT_HEALTH_PATH = "/health"

#: Loopback host every launched server binds to. Under container-leg
#: ``--network host`` the sibling binds the real host loopback, so the same URL
#: reaches it from a co-located client (the standard genai-perf / vllm
#: benchmark_serving topology).
_LOCALHOST = "127.0.0.1"

#: Recent-log tail length (characters) attached to launch/readiness errors.
_LOG_TAIL_CHARS = 2000


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ServerLifecycleError(LLEMError):
    """Base class for server lifecycle failures (launch / readiness / topology)."""


class ServerLaunchError(ServerLifecycleError):
    """The server process/container could not be launched, or exited during startup."""


class ServerReadinessError(ServerLifecycleError):
    """The server did not become ready (liveness + real probe) within the timeout."""


class ServerTopologyError(ServerLifecycleError):
    """The server is unreachable because of a docker-outside-of-docker network topology.

    Raised (instead of a generic readiness timeout) when llenergymeasure runs
    inside a container with a mounted Docker socket and dispatched a sibling
    server container: the sibling's ``--network host`` binds the real host
    network, which llenergymeasure's own container cannot reach via localhost
    unless it too runs ``--network host``. The message is actionable.
    """


# ---------------------------------------------------------------------------
# Value types (the ServerCapable protocol surface references these)
# ---------------------------------------------------------------------------


@dataclass
class ProbeRequest:
    """The readiness probe's request shape.

    This lifecycle layer owns the probe MECHANICS (drive this request through the
    serving path); the server warmup protocol owns the request SHAPE (drawn from
    the measured traffic distribution - warm the path you measure), supplied here
    as a parameter. ``payload`` is the JSON
    body (``None`` for a bodyless request); ``path`` is the serving endpoint
    (e.g. ``/v1/completions``).
    """

    path: str
    payload: dict[str, Any] | None = None
    method: str = "POST"


@dataclass
class ServerPlacement:
    """Where a server runs: process (host subprocess) or container (sibling image).

    ``image``, ``gpu_indices``, and ``labels`` are consumed only by the container
    leg; the adapter resolves a ``None`` image via the image registry.
    Constructed by the caller (the server session, from the resolved
    ``RunnerSpec`` + ``study_execution.gpu_indices`` + the study's container
    ownership labels); tests construct it directly.

    ``labels`` carries the study's ownership labels
    (``study.container_lifecycle.generate_container_labels``) so a launched
    server container is visible to the same leak protection the study's
    experiment containers get: the study-scoped cleanup and the orphan reaper
    both select on ``llem.study_id``. A container-mode placement built by the
    study path always carries them.
    """

    mode: RunnerMode
    image: str | None = None
    gpu_indices: list[int] | None = None
    labels: dict[str, str] | None = None


@dataclass
class ServerHandle:
    """A launched server's identity + access, returned by ``ServerCapable.launch``.

    Exposes the ``base_url`` the issuer talks to, the process/container identity
    (exactly one of ``process`` / ``container_name`` is set), and log access -
    :meth:`read_logs` is the server-session failure-artefact hand-off (it reads the
    process log file, or shells ``docker logs`` for the container leg).
    """

    base_url: str
    engine: str
    process: subprocess.Popen[bytes] | None = None
    container_name: str | None = None
    log_path: Path | None = None
    _log_file: Any = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)

    @property
    def identity(self) -> str:
        """Human-readable process/container identity for logs and diagnostics."""
        if self.container_name is not None:
            return f"container {self.container_name}"
        if self.process is not None:
            return f"process pid={self.process.pid}"
        return "<unlaunched>"

    def read_logs(self, *, tail_lines: int | None = None) -> str:
        """Return the server's captured logs (best-effort, never raises).

        Process leg reads the redirected stdout/stderr log file; container leg
        shells ``docker logs``. Returns ``""`` when logs are unavailable.
        """
        if self.container_name is not None:
            cmd = ["docker", "logs"]
            if tail_lines is not None:
                cmd += ["--tail", str(tail_lines)]
            cmd.append(self.container_name)
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=TIMEOUT_DOCKER_CLI
                )
            except (OSError, subprocess.SubprocessError):
                return ""
            return result.stdout + result.stderr
        if self.log_path is not None and self.log_path.exists():
            try:
                text = self.log_path.read_text(errors="replace")
            except OSError:
                return ""
            if tail_lines is not None:
                return "\n".join(text.splitlines()[-tail_lines:])
            return text
        return ""

    def _close_log(self) -> None:
        """Close the process-leg log file handle (idempotent, never raises)."""
        if self._log_file is not None:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None


# ---------------------------------------------------------------------------
# Port allocation + container naming + log paths
# ---------------------------------------------------------------------------


def allocate_free_port() -> int:
    """Return a currently-free TCP port on the loopback interface.

    Binds to port 0 (the OS assigns a free port), reads it back, and releases
    the socket. There is an inherent TOCTOU window between release and the
    server binding it; it is the standard approach and the window is small.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((_LOCALHOST, 0))
        return int(sock.getsockname()[1])


def base_url_for(port: int) -> str:
    """Return the loopback base URL a launched server is reached at."""
    return f"http://{_LOCALHOST}:{port}"


def server_container_name(engine: str) -> str:
    """Return a unique container name for an engine server (for --name / cleanup)."""
    return f"llem-{engine}-server-{uuid.uuid4().hex[:12]}"


def default_server_log_path(engine: str, port: int) -> Path:
    """Return a temp-dir log path for a process-leg server's captured output."""
    return Path(tempfile.gettempdir()) / f"llem-{engine}-server-{port}.log"


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def launch_process_server(
    argv: list[str],
    *,
    base_url: str,
    engine: str,
    log_path: Path,
) -> ServerHandle:
    """Launch a server as a host subprocess in its own process group.

    ``start_new_session=True`` makes the child a session/group leader (the
    parent-side analogue of the worker's ``os.setpgrp()`` precedent), so
    :func:`shutdown` can signal the whole group - engine worker subprocesses and
    all - via ``killpg``. stdout+stderr are redirected to ``log_path`` for the
    failure-artefact hand-off. A launch that cannot even start (bad argv,
    missing executable) cleans up its own log file and raises
    :class:`ServerLaunchError`.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "wb")  # noqa: SIM115 - handle lives on the ServerHandle
    try:
        proc: subprocess.Popen[bytes] = subprocess.Popen(
            argv,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        # Failed launch cleans its own partial state: close + remove the log file.
        with contextlib.suppress(Exception):
            log_file.close()
        with contextlib.suppress(OSError):
            log_path.unlink()
        raise ServerLaunchError(
            f"failed to launch {engine} server process {argv[0]!r}: {exc}"
        ) from exc
    logger.info("Launched %s server (pid=%s), logs -> %s", engine, proc.pid, log_path)
    return ServerHandle(
        base_url=base_url,
        engine=engine,
        process=proc,
        log_path=log_path,
        _log_file=log_file,
    )


def launch_container_server(
    argv: list[str],
    *,
    base_url: str,
    engine: str,
    container_name: str,
) -> ServerHandle:
    """Launch a server as a detached sibling container.

    ``docker run -d`` blocks only until the container is created (pulling the
    image first if absent), then returns; readiness is polled separately. A
    non-zero exit or an exec failure force-removes the container (failed launch
    cleans its own partial state) and raises :class:`ServerLaunchError`.
    """
    try:
        result = subprocess.run(argv, capture_output=True, text=True)
    except (OSError, subprocess.SubprocessError) as exc:
        _force_remove_container(container_name)
        raise ServerLaunchError(f"failed to run docker for {engine} server: {exc}") from exc
    if result.returncode != 0:
        _force_remove_container(container_name)
        raise ServerLaunchError(
            f"docker run failed for {engine} server (exit {result.returncode}): "
            f"{result.stderr.strip()}"
        )
    logger.info("Launched %s server container %s", engine, container_name)
    return ServerHandle(base_url=base_url, engine=engine, container_name=container_name)


# ---------------------------------------------------------------------------
# Readiness (liveness poll THEN a real probe through the serving path)
# ---------------------------------------------------------------------------


def await_ready(
    handle: ServerHandle,
    probe_request: ProbeRequest,
    *,
    timeout: float,
    poll_interval: float = 0.5,
    request_timeout: float = 5.0,
    health_path: str = DEFAULT_HEALTH_PATH,
) -> None:
    """Block until the server is ready, or raise.

    Two phases: (1) a liveness poll of ``health_path`` - necessary but
    NEVER sufficient; (2) a REAL inference request (``probe_request``) driven
    through the serving path - readiness is only satisfied when that completes
    with HTTP 200. Raises :class:`ServerReadinessError` on timeout (with recent
    logs), :class:`ServerLaunchError` if the process/container exited during
    startup, or :class:`ServerTopologyError` in the docker-outside-of-docker
    topology.

    ``timeout`` is a SINGLE shared deadline spanning BOTH phases, not a per-phase
    budget: the same wall-clock deadline is computed once and passed to each
    phase, so a slow liveness phase eats into the probe phase's remaining time
    (each phase is still guaranteed at least one attempt).
    """
    deadline = time.monotonic() + timeout
    base = handle.base_url.rstrip("/")
    _wait_for(
        handle,
        deadline,
        poll_interval,
        url=base + health_path,
        method="GET",
        payload=None,
        request_timeout=request_timeout,
        phase=f"liveness probe (GET {health_path})",
    )
    _wait_for(
        handle,
        deadline,
        poll_interval,
        url=base + probe_request.path,
        method=probe_request.method,
        payload=probe_request.payload,
        request_timeout=request_timeout,
        phase=f"readiness probe ({probe_request.method} {probe_request.path})",
    )
    logger.info("%s server ready at %s", handle.engine, handle.base_url)


def _wait_for(
    handle: ServerHandle,
    deadline: float,
    poll_interval: float,
    *,
    url: str,
    method: str,
    payload: dict[str, Any] | None,
    request_timeout: float,
    phase: str,
) -> None:
    """Poll ``url`` until it returns HTTP 200 or the deadline passes.

    A transient connection failure (the server is still loading and not yet
    listening) is NEVER terminal on its own: during the startup window it is
    indistinguishable from permanent DooD-unreachability, so the loop keeps
    polling. Topology is diagnosed only when the whole deadline is spent (see the
    deadline branch), never on the first failure. ``saw_http_response`` tracks
    whether any probe ever got an HTTP-level answer (as opposed to a transport
    failure) - one answer proves the server was reachable, which downgrades a
    deadline to a plain readiness timeout rather than a topology error.
    """
    last: Any = "no attempt"
    saw_http_response = False
    while True:
        # Fast-fail if the server died during startup (process leg: exited pid;
        # container leg: State.Running=false). Both capture the logs into the
        # ServerLaunchError before any cleanup, so the diagnostic is preserved.
        _ensure_process_alive(handle)
        _ensure_container_alive(handle)
        try:
            status = _http_probe(url, method=method, payload=payload, timeout=request_timeout)
            if status == 200:
                return
            saw_http_response = True
            last = f"HTTP {status}"
        except _ConnectionFailed as exc:
            # Transient during the startup window; never terminal on its own -
            # keep polling. Topology is diagnosed only at the deadline (below).
            last = f"connection failed ({exc})"
        if time.monotonic() >= deadline:
            # Diagnosis is a DEADLINE decision, never a first-failure one: a
            # normal startup-window connection failure (model still loading,
            # nothing listening yet) is indistinguishable from permanent
            # DooD-unreachability, so an eager raise aborts valid launches.
            # read_logs() runs BEFORE the caller's cleanup (shutdown removes the
            # container), so the error carries the tail; with --rm dropped the
            # container still exists here, so `docker logs` succeeds.
            logs = _tail(handle.read_logs())
            if not saw_http_response and _is_dood_topology(handle):
                # Every probe failed at the transport level AND we sit behind the
                # DooD topology: the sibling may in fact be healthy and listening
                # on the real host (the logs disambiguate) while llenergymeasure's
                # own container simply cannot route to it. Explain the topology.
                raise ServerTopologyError(f"{_dood_message(handle)}\nRecent logs:\n{logs}")
            # A single HTTP response anywhere proves the server was reachable, so
            # this is an ordinary readiness timeout, not a topology problem.
            raise ServerReadinessError(
                f"{handle.engine} server {phase} did not succeed within the readiness "
                f"timeout (last result: {last}). Recent logs:\n{logs}"
            )
        time.sleep(poll_interval)


def _ensure_process_alive(handle: ServerHandle) -> None:
    """Fail fast (process leg) if the server exited during startup."""
    proc = handle.process
    if proc is not None and proc.poll() is not None:
        raise ServerLaunchError(
            f"{handle.engine} server process exited during startup "
            f"(exit code {proc.returncode}). Recent logs:\n{_tail(handle.read_logs())}"
        )


def _ensure_container_alive(handle: ServerHandle) -> None:
    """Fail fast (container leg) if the server container crashed during startup.

    Mirrors :func:`_ensure_process_alive`. When the container EXISTS but is no
    longer running (``State.Running == false``), it crashed: capture its logs
    NOW (while the container - and therefore ``docker logs`` - still exists, the
    reason ``--rm`` was dropped), force-remove it, and raise a
    :class:`ServerLaunchError` carrying the log tail. An unknown state (docker
    unavailable, or the container not yet visible) is NOT treated as a crash -
    the poll simply continues.
    """
    if handle.container_name is None:
        return
    if _container_running(handle.container_name) is False:
        logs = _tail(handle.read_logs())
        _force_remove_container(handle.container_name)
        raise ServerLaunchError(
            f"{handle.engine} server container {handle.container_name} exited during "
            f"startup. Recent logs:\n{logs}"
        )


def _container_running(container_name: str) -> bool | None:
    """Return the container's running state via ``docker inspect``.

    ``True`` running, ``False`` exists-but-stopped (a definitive crash signal),
    ``None`` unknown - the container is absent (a startup race, or already
    removed) or docker is unavailable, neither of which should be read as a
    crash. Never raises.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    state = result.stdout.strip().lower()
    if state == "true":
        return True
    if state == "false":
        return False
    return None


def _is_dood_topology(handle: ServerHandle) -> bool:
    """Whether an unreachable container-leg server sits behind the DooD topology.

    True only for the container leg when llenergymeasure itself runs inside a
    container (``is_running_in_container``) with a mounted Docker socket
    (``is_container_socket_available`` - the DooD signal): a sibling server on
    ``--network host`` binds the real host network, which llenergymeasure's own
    container cannot reach via localhost unless it too runs ``--network host``.
    Imported at use-site so detection is monkeypatchable in tests. This is a pure
    predicate - the readiness loop decides at the DEADLINE (not the first failure)
    whether the topology is the actual explanation for the unreachability.
    """
    if handle.container_name is None:
        return False
    from llenergymeasure.infra.runner_resolution import (
        is_container_socket_available,
        is_running_in_container,
    )

    return is_running_in_container() and is_container_socket_available()


def _dood_message(handle: ServerHandle) -> str:
    """The actionable DooD topology explanation (the ``docker logs`` tail is
    appended separately by the caller, since a healthy sibling listening on the
    real host is disambiguated from a crash only by its logs)."""
    return (
        f"Cannot reach the {handle.engine} server at {handle.base_url} from inside "
        "llenergymeasure's own container. The server runs as a sibling container with "
        "--network host (bound to the real host network); llenergymeasure's container is "
        "in a different network namespace, so localhost does not route to it. Fix: run "
        "llenergymeasure's own container with --network host, or pin this engine to "
        f"process mode (runners.{handle.engine}=process)."
    )


# ---------------------------------------------------------------------------
# Shutdown (idempotent, leak-free, with kill escalation)
# ---------------------------------------------------------------------------


def shutdown(handle: ServerHandle, *, grace: float = TIMEOUT_SIGTERM_GRACE) -> None:
    """Stop the server, escalating to a hard kill; idempotent and best-effort.

    Process leg: SIGTERM the whole process group, wait ``grace`` seconds, then
    SIGKILL if still alive. Container leg: ``docker stop -t grace`` (docker's own
    SIGTERM -> SIGKILL escalation) then ``docker rm -f`` so nothing leaks even if
    stop failed. A second call is a no-op. Never raises - it runs on cleanup /
    ``__exit__`` paths.
    """
    if handle._closed:
        return
    handle._closed = True
    if handle.process is not None:
        _shutdown_process(handle.process, grace)
    if handle.container_name is not None:
        _shutdown_container(handle.container_name, grace)
    handle._close_log()


def _shutdown_process(proc: subprocess.Popen[bytes], grace: float) -> None:
    """SIGTERM the process group, then SIGKILL after ``grace`` if still alive."""
    if proc.poll() is not None:
        return  # already exited
    pid = proc.pid
    _kill_process_group(pid, signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    # Escalation: the graceful stop did not land within the grace period.
    logger.warning("server pid=%s ignored SIGTERM; escalating to SIGKILL", pid)
    _kill_process_group(pid, signal.SIGKILL)
    with contextlib.suppress(Exception):
        proc.wait(timeout=grace)


def _kill_process_group(pid: int, sig: int) -> None:
    """Signal the whole process group rooted at ``pid`` (killpg precedent).

    ``start_new_session=True`` at launch makes the child a group leader, so its
    PID equals its PGID. Errors are suppressed - the group may already be gone.
    Reimplemented at the infra altitude (the study-layer worker helper cannot be
    imported downward) but faithful to that precedent.
    """
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pid, sig)


def _shutdown_container(container_name: str, grace: float) -> None:
    """``docker stop`` (graceful, escalating) then force-remove - leak-free."""
    _run_docker_quiet(["docker", "stop", "-t", str(int(grace)), container_name])
    _force_remove_container(container_name)


def _force_remove_container(container_name: str) -> None:
    """Force-remove a container, ignoring 'no such container' (idempotent)."""
    _run_docker_quiet(["docker", "rm", "-f", container_name])


def _run_docker_quiet(argv: list[str]) -> None:
    """Run a docker CLI command best-effort (never raises)."""
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        subprocess.run(argv, capture_output=True, timeout=TIMEOUT_DOCKER_CLI)


# ---------------------------------------------------------------------------
# HTTP probe transport (narrow stdlib client; see module docstring / server-session seam)
# ---------------------------------------------------------------------------


class _ConnectionFailed(Exception):
    """Transport-level failure (server not listening / unreachable), not an HTTP status."""


def _http_probe(
    url: str,
    *,
    method: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> int:
    """Send one request; return the HTTP status code.

    A response of ANY status (including 4xx/5xx) returns its code - the caller
    decides whether it means "ready". A transport-level failure (connection
    refused, DNS, timeout) raises :class:`_ConnectionFailed`, which the poll loop
    treats as "not up yet" (and the DooD check inspects).
    """
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response.read()
            return int(response.status)
    except urllib.error.HTTPError as exc:
        # The server responded, just not 2xx (e.g. 503 while the model loads).
        return int(exc.code)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise _ConnectionFailed(str(exc)) from exc


def _tail(text: str) -> str:
    """Return the last ``_LOG_TAIL_CHARS`` characters of ``text`` (for error messages)."""
    text = text.strip()
    if len(text) <= _LOG_TAIL_CHARS:
        return text or "<no logs captured>"
    return "..." + text[-_LOG_TAIL_CHARS:]


# Re-export the container-mode constant the launch router pattern-matches
# placement.mode against, so a server adapter needs only this module.
CONTAINER_MODE: RunnerMode = RUNNER_CONTAINER
