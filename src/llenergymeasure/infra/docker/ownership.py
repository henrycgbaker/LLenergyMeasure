"""Who owns a container, and the machinery that reclaims it.

Every container llenergymeasure launches is owned by exactly one study, and this
module is where that ownership is stamped on and acted upon: deterministic
container naming, label generation, the atexit cleanup handler, the SIGTERM
bridge that makes atexit fire at all, and the startup reaper for containers whose
launching process is gone.

Layered strategy to prevent container leaks on abort:

1. Named containers: deterministic ``llem-{hash8}-{index:04d}`` names.
2. Labels: ``llem.study_id``, ``llem.parent_pid``, ``llem.started_at`` for targeted cleanup.
3. atexit handler: stops containers with matching study_id label on exit, keeps
   their log tails, and then removes them.
4. SIGTERM bridge: converts SIGTERM to sys.exit(0) so atexit handlers fire.
5. Startup reaper: stops orphaned containers whose parent PID is dead.

Every layer keys on the study design hash, so that identity is mandatory:
:func:`require_study_id` refuses a study without one, and the StudyError it
raises spells out why. That is the one place study vocabulary surfaces here: the
identity being enforced belongs to the caller's study, so the refusal is phrased
in the caller's terms even though the mechanics live at this layer.

These are mechanics only. WHETHER to install the atexit net and the SIGTERM
bridge for a given run is a decision the study layer makes; nothing here reaches
back up to make it.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from llenergymeasure.config.ssot import TIMEOUT_DOCKER_CLI, TIMEOUT_DOCKER_STOP
from llenergymeasure.utils.exceptions import StudyError

__all__ = [
    "cleanup_study_containers",
    "generate_container_labels",
    "generate_container_name",
    "install_sigterm_bridge",
    "reap_orphaned_containers",
    "register_container_cleanup",
    "require_study_id",
]

logger = logging.getLogger(__name__)

# How much of an abandoned container's output to keep. Long enough to hold an
# engine's startup log plus whatever killed it, short enough that a study
# aborting with several containers up does not write out tens of megabytes.
_ABANDONED_LOG_TAIL_LINES: Final = 500

# Filename stem for a persisted log tail, so the files are obviously grouped and
# obviously not per-experiment failure artefacts.
_ABANDONED_LOG_PREFIX: Final = "abandoned-container-"


def require_study_id(study_design_hash: str | None) -> str:
    """Return the study identity every container of this study is owned by.

    Required, not best-effort: the StudyError raised below is where the reason
    lives, since that message is what an operator actually sees.

    Args:
        study_design_hash: ``StudyConfig.study_design_hash``, possibly None.

    Returns:
        The non-empty study identity.

    Raises:
        StudyError: If the hash is missing, empty, or whitespace-only.
    """
    study_id = (study_design_hash or "").strip()
    if not study_id:
        raise StudyError(
            "Study has no study_design_hash, so its Docker containers cannot be "
            "managed safely and the study is refused. The hash is the container "
            "ownership key: it names containers, it is the llem.study_id label, "
            "and it is the filter the cleanup handler and the startup reaper "
            "select on. Substituting a shared placeholder would make every "
            "unidentified study share one ownership key, so cleaning up after "
            "one of them would stop the containers of every concurrently "
            "running trial that shares it. Studies loaded from YAML always "
            "carry the hash; a programmatically constructed StudyConfig must "
            "have it computed before containers are launched (see "
            "https://github.com/henrycgbaker/llenergymeasure/issues/886)."
        )
    return study_id


def generate_container_name(study_id: str, experiment_index: int) -> str:
    """Return a deterministic container name for a given study and experiment.

    Format: ``llem-{study_id_short}-{index:04d}``

    ``study_id_short`` is the first 8 characters of the study_design_hash.

    Args:
        study_id:         Study design hash (typically a full hex string).
        experiment_index: 1-based experiment index within the study.

    Returns:
        Container name string, e.g. ``"llem-abcdef12-0001"``.

    Raises:
        StudyError: If study_id is empty - see :func:`require_study_id`.
    """
    short = require_study_id(study_id)[:8]
    return f"llem-{short}-{experiment_index:04d}"


def generate_container_labels(study_id: str | None) -> dict[str, str]:
    """Return Docker labels that enable targeted cleanup and reaper identification.

    Labels:
        ``llem.study_id``:   Study design hash - used to filter containers by study.
        ``llem.parent_pid``: PID of the host process that launched the container.
        ``llem.started_at``: UTC ISO-8601 timestamp when the labels were generated.

    Every container a study launches (experiment, engine server, baseline) wears
    these labels, so the four-layer leak protection sees all of them.

    Args:
        study_id: Study design hash, possibly None - it is put through
            :func:`require_study_id` here, so callers hand over the raw
            ``StudyConfig.study_design_hash`` rather than pre-validating it.

    Returns:
        Dict of label key -> value pairs.

    Raises:
        StudyError: If study_id is empty - see :func:`require_study_id`.
    """
    study_id = require_study_id(study_id)
    return {
        "llem.study_id": study_id,
        "llem.parent_pid": str(os.getpid()),
        "llem.started_at": datetime.now(timezone.utc).isoformat(),
    }


def _docker_quiet(argv: list[str], *, timeout: float) -> bool:
    """Run a docker command best-effort; return whether it exited zero.

    Never raises: this whole path runs at interpreter exit, where an exception
    would garble the output and could mask the error that ended the study.
    """
    try:
        result = subprocess.run(argv, capture_output=True, timeout=timeout)
    except Exception:
        return False
    return result.returncode == 0


def _read_container_log_tail(container_id: str) -> str | None:
    """Return the container's log tail, or ``None`` if it could not be read.

    ``None`` covers both a docker call that failed and a container that is no
    longer there; the caller distinguishes those, because they mean opposite
    things about whether anything is left to lose.
    """
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(_ABANDONED_LOG_TAIL_LINES), container_id],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    # docker logs replays the container's stdout on stdout and its stderr on
    # stderr; engines log to both, so the record needs both.
    return result.stdout + result.stderr


def _persist_container_log_tail(container_id: str, log_dir: Path, log_tail: str) -> bool:
    """Write *log_tail* into *log_dir*; return whether it landed on disk."""
    dest = log_dir / f"{_ABANDONED_LOG_PREFIX}{container_id}.log"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(log_tail or "(container produced no output)", encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "Container %s left in place: its log tail could not be written to %s (%s)",
            container_id,
            dest,
            exc,
        )
        return False
    logger.info("Container %s stopped at exit; its log tail is at %s", container_id, dest)
    return True


def _stop_and_reclaim(container_id: str, log_dir: Path) -> None:
    """Stop one container, keep its log tail, then remove it.

    Removal is conditional on the log tail being safely on disk. Reclaiming disk
    space is worth having; destroying the only surviving record of what a
    container did is not worth having, so a failure to persist leaves the
    container exactly where it is and says so.
    """
    _docker_quiet(["docker", "stop", "-t", "5", container_id], timeout=TIMEOUT_DOCKER_STOP)

    log_tail = _read_container_log_tail(container_id)
    if log_tail is None:
        if _docker_quiet(
            ["docker", "container", "inspect", container_id], timeout=TIMEOUT_DOCKER_CLI
        ):
            logger.warning(
                "Container %s left in place: its logs could not be read, and removing "
                "it would destroy the only record of what it did",
                container_id,
            )
        # Otherwise the container is simply gone: the run-to-completion shapes are
        # launched with ``--rm``, so docker reaps them the moment they stop and
        # there is nothing left to persist or reclaim.
        return

    if not _persist_container_log_tail(container_id, log_dir, log_tail):
        return  # already warned; the container stays put

    _docker_quiet(["docker", "rm", container_id], timeout=TIMEOUT_DOCKER_STOP)


def cleanup_study_containers(study_id: str, log_dir: Path) -> None:
    """Stop this study's containers, keep their logs, then remove them.

    Intended as an atexit handler. ``docker ps --filter`` lists the containers
    wearing this study's ``llem.study_id`` label and each is stopped gracefully
    with ``docker stop -t 5``.

    Stopping alone is not enough. The run-to-completion shapes are launched with
    ``--rm``, so docker reaps them as soon as they stop and nothing is left
    behind. The engine-server shape deliberately is NOT ``--rm`` - a
    crash-on-startup has to survive for its logs to be recoverable - so a stop
    leaves it sitting on the host as an exited container that no code will ever
    look at again. It is removed here, but only once its log tail is safely
    written into *log_dir*. If that write cannot happen the container is LEFT IN
    PLACE and a warning names it: a stray container is untidy and fixable by
    hand, whereas discarding the last record of why a study died is not.

    Never raises. An atexit handler that raises garbles the shutdown output and
    can mask the original exception. An empty study_id is refused silently for
    the same reason - the loud refusal belongs at registration time
    (:func:`register_container_cleanup`), and an unscoped filter here could reach
    containers this study does not own.

    Args:
        study_id: Study design hash used as the label filter value.
        log_dir:  Directory the log tails are written into. Chosen by the caller,
            so the study's on-disk layout stays owned by the layer that owns the
            study's output directory.
    """
    if not (study_id or "").strip():
        logger.warning("Container cleanup skipped: no study identity to scope it to")
        return
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"label=llem.study_id={study_id}"],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
        for cid in result.stdout.strip().splitlines():
            if cid.strip():
                _stop_and_reclaim(cid.strip(), log_dir)
    except Exception:
        pass  # Best-effort; atexit handlers must never raise


def register_container_cleanup(study_id: str, log_dir: Path) -> None:
    """Register an atexit handler that reclaims this study's containers.

    Calling this multiple times with the same study_id is safe - Python's
    atexit module allows multiple registrations and runs them LIFO.

    Args:
        study_id: Study design hash passed to cleanup_study_containers.
        log_dir:  Directory abandoned containers' log tails are written into.

    Raises:
        StudyError: If study_id is empty - see :func:`require_study_id`.
    """
    atexit.register(cleanup_study_containers, require_study_id(study_id), log_dir)


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

    Reaped containers are stopped but NOT removed, unlike
    :func:`cleanup_study_containers`. This reaper is host-wide: the orphans it
    finds belong to somebody else's abandoned study, so their evidence is not
    ours to destroy, and we have no output directory of theirs to preserve it in.
    Freeing the GPU is the whole job here.

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
