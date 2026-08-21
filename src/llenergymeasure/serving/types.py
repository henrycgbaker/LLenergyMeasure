"""Value types and error types for the online-serving lifecycle.

The vocabulary half of the serving layer. Everything here is data plus its
failure modes: where a server should run (:class:`ServerPlacement`), what a
launched server is (:class:`ServerHandle`), what shape the readiness probe takes
(:class:`ProbeRequest`), and the four errors the lifecycle raises. A consumer
that only needs to NAME these things - an annotation, a constructor call, an
``except`` clause - imports this module and never pulls in the launch/readiness/
shutdown mechanics that live in :mod:`llenergymeasure.serving.lifecycle`.

The ``ServerCapable`` engine-plugin extension (see
:mod:`llenergymeasure.engines.protocol`) is typed entirely in terms of these
three value types, so an engine adapter's signatures depend on the vocabulary
without depending on the plumbing.
"""

from __future__ import annotations

import contextlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llenergymeasure.config.ssot import TIMEOUT_DOCKER_CLI, RunnerMode
from llenergymeasure.utils.exceptions import LLEMError

__all__ = [
    "ProbeRequest",
    "ServerHandle",
    "ServerLaunchError",
    "ServerLifecycleError",
    "ServerPlacement",
    "ServerReadinessError",
    "ServerTopologyError",
]


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

    The serving layer owns the probe MECHANICS (drive this request through the
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
    (``infra.docker.ownership.generate_container_labels``) so a launched
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
