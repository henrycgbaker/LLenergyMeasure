"""Failure artefacts: what a failed experiment leaves behind in ``failed-runs/``.

A study that loses an experiment must stay debuggable afterwards, so every
failure route lands its evidence under ``{study_dir}/failed-runs/`` with a shared
``{config_hash}_cycle{cycle}`` stem, and records a ``log_file`` pointer in the
result dict so the manifest can find it again:

- container dispatch: the in-container ``container.log`` and the structured
  ``*_error.json`` the entrypoint wrote, copied out of the exchange dir;
- subprocess and in-process dispatch: the captured traceback string, which has no
  exchange dir to be copied from.

Every helper here is best-effort by design. A persistence failure must never mask
the original error, so write problems are logged and swallowed rather than
raised.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.utils.exceptions import DockerError

__all__ = [
    "copy_artefact",
    "persist_failure_artefacts",
    "persist_failure_traceback",
]

logger = logging.getLogger(__name__)


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


def _ensure_failed_runs_dir(
    study_dir: Path, config_hash: str, cycle: int
) -> tuple[Path, str] | None:
    """Create ``{study_dir}/failed-runs/`` and return it with the artefact prefix.

    Returns ``(failed_runs_dir, prefix)`` where ``prefix`` is the shared
    ``{config_hash}_cycle{cycle}`` stem used to name every persisted failure
    artefact. Returns ``None`` (logging a warning) if the directory cannot be
    created - failure persistence is best-effort and must never mask the
    original error.
    """
    failed_runs_dir = study_dir / "failed-runs"
    try:
        failed_runs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as mkdir_exc:
        logger.warning("Failed to create failed-runs/: %s", mkdir_exc)
        return None
    return failed_runs_dir, f"{config_hash}_cycle{cycle}"


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
    ensured = _ensure_failed_runs_dir(study_dir, config_hash, cycle)
    if ensured is None:
        return
    failed_runs_dir, prefix = ensured

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


def persist_failure_traceback(
    study_dir: Path,
    config_hash: str,
    cycle: int,
    traceback_str: str,
    result: dict[str, Any],
) -> None:
    """Persist a captured traceback into ``failed-runs/`` for a local failure.

    The Docker path keeps the real in-container traceback debuggable via
    :func:`persist_failure_artefacts` (it copies the ``*_error.json`` the
    container entrypoint wrote). Local/subprocess dispatch has no exchange dir,
    but the subprocess worker and the single-experiment in-process path still
    capture a full traceback string on failure - this helper gives that string
    the same on-disk home (``failed-runs/{prefix}_traceback.txt``) and the same
    ``log_file`` manifest pointer, so a local failure is as debuggable as a
    Docker one regardless of dispatch mode.

    Best-effort: a persistence failure must never mask the original error, so
    write problems are logged and swallowed. Dispatch-neutral - it takes a plain
    traceback string, not a ``DockerError``.
    """
    if not traceback_str:
        return

    ensured = _ensure_failed_runs_dir(study_dir, config_hash, cycle)
    if ensured is None:
        return
    failed_runs_dir, prefix = ensured

    dest = failed_runs_dir / f"{prefix}_traceback.txt"
    try:
        dest.write_text(traceback_str, encoding="utf-8")
    except OSError as write_exc:
        logger.warning("Failed to persist traceback to %s: %s", dest, write_exc)
        return

    result["log_file"] = f"failed-runs/{dest.name}"
