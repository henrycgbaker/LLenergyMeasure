"""Per-GPU advisory locks for study concurrency safety.

Uses filelock.FileLock (kernel-backed via fcntl.flock on Linux) which
auto-releases on process death including SIGKILL. Lock files live at
~/.cache/llem/gpu-{id}.lock, where ``id`` names the PHYSICAL device the study
occupies (a device index like ``2`` under ``--gpus device=2``, or a
``GPU-<uuid>`` string) - never the in-container logical index, which always
starts at 0 under docker pinning. Callers derive the ids from the docker
pinning; see ``utils.env_config.pinned_gpu_lock_ids``.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

from filelock import FileLock, Timeout

from llenergymeasure.utils.exceptions import StudyError

__all__ = ["acquire_gpu_locks", "release_gpu_locks"]


def acquire_gpu_locks(
    lock_ids: list[str],
    lock_dir: Path | None = None,
) -> list[FileLock]:
    """Acquire advisory file locks for the given physical GPU ids, in sorted order.

    Each id names a PHYSICAL GPU the study will occupy - a device index like
    ``"2"`` (under ``--gpus device=2``) or a ``GPU-<uuid>`` string. Callers
    derive these from the docker pinning (see
    ``utils.env_config.pinned_gpu_lock_ids``) so two studies on different
    physical GPUs never share a lock. The ids are NOT the in-container logical
    indices, which always start at 0 under pinning and address the energy
    samplers, not host-side locks.

    Sorted acquisition (Dijkstra's resource ordering) prevents deadlocks when
    multiple studies attempt to acquire overlapping GPU sets concurrently. The
    sort is lexicographic on the id strings; any globally consistent order
    suffices for deadlock freedom.

    The locks are non-blocking (timeout=0). If any GPU is already locked by
    another process, all previously acquired locks are released (atomic all-or-none
    rollback) and a StudyError is raised.

    Args:
        lock_ids: Physical GPU lock identifiers to lock (e.g. ``["2", "3"]``).
        lock_dir: Directory for lock files. Defaults to ~/.cache/llem.

    Returns:
        List of acquired FileLock objects in sorted id order.

    Raises:
        StudyError: If any GPU is locked by another process.
    """
    if lock_dir is None:
        lock_dir = Path.home() / ".cache" / "llem"

    lock_dir.mkdir(parents=True, exist_ok=True)

    # Sort to prevent deadlocks (Dijkstra's resource ordering)
    sorted_ids = sorted(lock_ids)

    acquired: list[FileLock] = []
    failed_ids: list[str] = []

    for lock_id in sorted_ids:
        lock_path = lock_dir / f"gpu-{lock_id}.lock"
        lock = FileLock(str(lock_path), timeout=0)
        try:
            lock.acquire()
            acquired.append(lock)
        except Timeout:
            failed_ids.append(lock_id)
            # Atomic rollback: release all already-acquired locks
            for held_lock in acquired:
                with contextlib.suppress(Exception):
                    held_lock.release()
            raise StudyError(
                f"GPU(s) {failed_ids} locked by another process. Use --no-lock to override."
            ) from None

    return acquired


def release_gpu_locks(locks: list[FileLock]) -> None:
    """Release all acquired GPU locks, suppressing any errors.

    Args:
        locks: List of FileLock objects previously returned by acquire_gpu_locks.
    """
    for lock in locks:
        with contextlib.suppress(Exception):
            lock.release()
