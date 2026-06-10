"""Container lifecycle management for Docker-backed studies.

Provides deterministic container naming, label generation, atexit cleanup
handlers, a SIGTERM bridge, and a startup reaper for orphaned containers.

Four-layer strategy to prevent Docker container leaks on study abort:

1. Named containers: deterministic ``llem-{hash8}-{index:04d}`` names.
2. Labels: ``llem.study_id``, ``llem.parent_pid``, ``llem.started_at`` for targeted cleanup.
3. atexit handler: stops containers with matching study_id label on exit.
4. SIGTERM bridge: converts SIGTERM to sys.exit(0) so atexit handlers fire.
5. Startup reaper: stops orphaned containers whose parent PID is dead.
"""

from __future__ import annotations

import atexit
import logging
import os
import shutil
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import TIMEOUT_DOCKER_CLI, TIMEOUT_DOCKER_STOP

if TYPE_CHECKING:
    from llenergymeasure.utils.exceptions import DockerError

__all__ = [
    "cleanup_study_containers",
    "copy_artefact",
    "generate_container_labels",
    "generate_container_name",
    "install_sigterm_bridge",
    "persist_failure_artefacts",
    "reap_orphaned_containers",
    "register_container_cleanup",
]

logger = logging.getLogger(__name__)


def generate_container_name(study_id: str, experiment_index: int) -> str:
    """Return a deterministic container name for a given study and experiment.

    Format: ``llem-{study_id_short}-{index:04d}``

    ``study_id_short`` is the first 8 characters of the study_design_hash.
    Falls back to ``unknown`` when study_id is empty.

    Args:
        study_id:         Study design hash (typically a full hex string).
        experiment_index: 1-based experiment index within the study.

    Returns:
        Container name string, e.g. ``"llem-abcdef12-0001"``.
    """
    short = study_id[:8] if study_id else "unknown"
    return f"llem-{short}-{experiment_index:04d}"


def generate_container_labels(study_id: str) -> dict[str, str]:
    """Return Docker labels that enable targeted cleanup and reaper identification.

    Labels:
        ``llem.study_id``:   Study design hash - used to filter containers by study.
        ``llem.parent_pid``: PID of the host process that launched the container.
        ``llem.started_at``: UTC ISO-8601 timestamp when the labels were generated.

    Args:
        study_id: Study design hash.

    Returns:
        Dict of label key -> value pairs.
    """
    return {
        "llem.study_id": study_id,
        "llem.parent_pid": str(os.getpid()),
        "llem.started_at": datetime.now(timezone.utc).isoformat(),
    }


def cleanup_study_containers(study_id: str) -> None:
    """Stop any running containers with this study's label.

    Intended as an atexit handler. Uses ``docker ps --filter`` to list
    containers with the matching ``llem.study_id`` label, then sends a
    graceful ``docker stop -t 5`` to each.

    This function must never raise - atexit handlers that raise produce
    confusing output and may mask the original exception.

    Args:
        study_id: Study design hash used as the label filter value.
    """
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"label=llem.study_id={study_id}"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
        for cid in result.stdout.strip().splitlines():
            if cid.strip():
                subprocess.run(
                    ["docker", "stop", "-t", "5", cid.strip()],
                    capture_output=True,
                    timeout=TIMEOUT_DOCKER_STOP,
                )
    except Exception:
        pass  # Best-effort; atexit handlers must never raise


def register_container_cleanup(study_id: str) -> None:
    """Register an atexit handler that stops containers for this study.

    Calling this multiple times with the same study_id is safe - Python's
    atexit module allows multiple registrations and runs them LIFO.

    Args:
        study_id: Study design hash passed to cleanup_study_containers.
    """
    atexit.register(cleanup_study_containers, study_id)


def install_sigterm_bridge() -> Any:
    """Install a SIGTERM handler that calls sys.exit(0) to trigger atexit.

    Python's default SIGTERM disposition terminates the process without
    running atexit handlers. This bridge converts SIGTERM into a clean
    exit so atexit-registered cleanup functions (including
    cleanup_study_containers) execute.

    Returns:
        The previous SIGTERM handler (for restoration in finally blocks).
        Returns None if signal handling is not available (e.g. non-main thread).
    """

    def _sigterm_handler(signum: int, frame: Any) -> None:
        sys.exit(0)

    try:
        original = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _sigterm_handler)
        return original
    except (OSError, ValueError):
        # ValueError raised when called from non-main thread
        return None


def reap_orphaned_containers() -> int:
    """Stop containers whose parent PID is no longer alive.

    Called at study start. Queries all running containers with the
    ``llem.study_id`` label, then checks whether each container's recorded
    ``llem.parent_pid`` is still alive using ``os.kill(pid, 0)``.

    Containers whose parent is dead are stopped with ``docker stop -t 5``.
    Containers whose parent is alive (or owned by another user) are skipped.

    This function never raises - errors are swallowed so they cannot block
    study start.

    Returns:
        Count of containers reaped (stopped).
    """
    reaped = 0
    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "--filter",
                "label=llem.study_id",
                "--format",
                '{{.ID}} {{.Label "llem.parent_pid"}}',
            ],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            cid, pid_str = parts
            try:
                os.kill(int(pid_str), 0)  # Signal 0 = aliveness probe
            except (ProcessLookupError, ValueError):
                logger.warning("Reaping orphaned container %s (PID %s dead)", cid, pid_str)
                subprocess.run(
                    ["docker", "stop", "-t", "5", cid],
                    capture_output=True,
                    timeout=TIMEOUT_DOCKER_STOP,
                )
                reaped += 1
            except PermissionError:
                pass  # Process exists but owned by another user - not orphaned
    except Exception:
        pass  # Never block study start
    return reaped


def copy_artefact(src: Path, dest: Path) -> str | None:
    """Copy a single file, returning the dest filename on success or None."""
    if not src.exists():
        return None
    try:
        shutil.copy2(src, dest)
        logger.debug("Artefact persisted to %s", dest)
        return dest.name
    except Exception as copy_exc:
        logger.warning("Failed to persist %s to %s: %s", src.name, dest, copy_exc)
        return None


def persist_failure_artefacts(
    exc: DockerError,
    study_dir: Path,
    config_hash: str,
    cycle: int,
    result: dict[str, Any],
) -> None:
    """Copy failure artefacts from the Docker exchange dir into ``failed-runs/``.

    Copies ``container.log`` and any ``*_error.json`` from the exchange
    directory. Adds a ``log_file`` key to *result* so the manifest records
    where the log can be found.
    """
    exchange_dir_str = getattr(exc, "exchange_dir", None)
    if not exchange_dir_str:
        return

    exchange_dir = Path(exchange_dir_str)
    failed_runs_dir = study_dir / "failed-runs"
    prefix = f"{config_hash}_cycle{cycle}"

    try:
        failed_runs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as mkdir_exc:
        logger.warning("Failed to create failed-runs/: %s", mkdir_exc)
        return

    # Copy container.log (Docker stderr capture)
    log_file = copy_artefact(
        exchange_dir / "container.log",
        failed_runs_dir / f"{prefix}_container.log",
    )
    if log_file:
        result["log_file"] = f"failed-runs/{log_file}"

    # Copy error JSON (structured traceback from container entrypoint).
    # The error JSON uses the Docker config hash (output_dir=/run/llem),
    # which differs from the study-level config_hash, so glob for it.
    for src in exchange_dir.glob("*_error.json"):
        copy_artefact(src, failed_runs_dir / f"{prefix}_error.json")
        break  # only one expected
