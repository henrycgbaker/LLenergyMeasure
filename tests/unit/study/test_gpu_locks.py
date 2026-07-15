"""Tests for GPU advisory lock acquisition and release."""

from __future__ import annotations

from pathlib import Path

import filelock
import pytest
from filelock import FileLock

from llenergymeasure.study.gpu_locks import acquire_gpu_locks, release_gpu_locks
from llenergymeasure.utils.exceptions import StudyError

# =============================================================================
# Helpers
# =============================================================================


def _lock_path(lock_dir: Path, lock_id: str) -> Path:
    return lock_dir / f"gpu-{lock_id}.lock"


# =============================================================================
# Tests
# =============================================================================


def test_single_gpu_lock_acquired_and_released(tmp_path: Path) -> None:
    """Single GPU lock acquired successfully and released without error."""
    locks = acquire_gpu_locks(["0"], lock_dir=tmp_path)

    assert len(locks) == 1
    assert _lock_path(tmp_path, "0").exists()

    # Verify the lock is actually held (re-acquiring non-blocking should raise Timeout)
    held_lock = FileLock(str(_lock_path(tmp_path, "0")), timeout=0)
    with pytest.raises(filelock.Timeout):
        held_lock.acquire()

    release_gpu_locks(locks)


def test_multi_gpu_locks_acquired_in_sorted_order(tmp_path: Path) -> None:
    """Multi-GPU locks are acquired in sorted order (lock files for all ids exist)."""
    # Pass ids in reverse order to verify sorting
    locks = acquire_gpu_locks(["2", "0", "1"], lock_dir=tmp_path)

    assert len(locks) == 3
    for i in ("0", "1", "2"):
        assert _lock_path(tmp_path, i).exists(), f"Lock file for GPU {i} missing"

    release_gpu_locks(locks)


def test_physical_lock_ids_name_the_physical_device(tmp_path: Path) -> None:
    """A study pinned to physical device 2 locks gpu-2.lock, not gpu-0.lock."""
    locks = acquire_gpu_locks(["2"], lock_dir=tmp_path)

    assert _lock_path(tmp_path, "2").exists()
    assert not _lock_path(tmp_path, "0").exists()

    release_gpu_locks(locks)


def test_different_physical_devices_do_not_collide(tmp_path: Path) -> None:
    """Two studies pinned to distinct physical GPUs acquire disjoint locks."""
    locks_a = acquire_gpu_locks(["2"], lock_dir=tmp_path)
    # A second acquisition on a DIFFERENT physical device must succeed even while
    # the first is still held - this is the concurrency bug the fix closes.
    locks_b = acquire_gpu_locks(["3"], lock_dir=tmp_path)

    assert len(locks_a) == 1
    assert len(locks_b) == 1

    release_gpu_locks(locks_a)
    release_gpu_locks(locks_b)


def test_same_physical_device_collides(tmp_path: Path) -> None:
    """Two studies pinned to the SAME physical GPU contend on one lock."""
    external_lock = FileLock(str(_lock_path(tmp_path, "2")))
    external_lock.acquire()

    try:
        with pytest.raises(StudyError, match=r"GPU\(s\).*locked by another process"):
            acquire_gpu_locks(["2"], lock_dir=tmp_path)
    finally:
        external_lock.release()


def test_uuid_lock_id(tmp_path: Path) -> None:
    """A UUID lock id names its own lock file verbatim."""
    locks = acquire_gpu_locks(["GPU-abc-123"], lock_dir=tmp_path)

    assert _lock_path(tmp_path, "GPU-abc-123").exists()

    release_gpu_locks(locks)


def test_contention_raises_study_error(tmp_path: Path) -> None:
    """Acquiring the same GPU twice raises StudyError with a clear message."""
    # Hold GPU 0 with an external lock
    external_lock = FileLock(str(_lock_path(tmp_path, "0")))
    external_lock.acquire()

    try:
        with pytest.raises(StudyError, match=r"GPU\(s\).*locked by another process"):
            acquire_gpu_locks(["0"], lock_dir=tmp_path)
    finally:
        external_lock.release()


def test_atomic_rollback_on_contention(tmp_path: Path) -> None:
    """If GPU 1 is locked, attempting ["0", "1"] releases GPU 0 (atomic rollback)."""
    # Hold GPU 1 externally
    external_lock = FileLock(str(_lock_path(tmp_path, "1")))
    external_lock.acquire()

    try:
        with pytest.raises(StudyError):
            acquire_gpu_locks(["0", "1"], lock_dir=tmp_path)

        # GPU 0 must be released after rollback - we can acquire it again
        locks = acquire_gpu_locks(["0"], lock_dir=tmp_path)
        assert len(locks) == 1
        release_gpu_locks(locks)
    finally:
        external_lock.release()


def test_release_gpu_locks_suppresses_errors(tmp_path: Path) -> None:
    """release_gpu_locks does not raise even if a lock raises on release."""

    # Create a mock lock that raises on release
    class BrokenLock:
        def release(self) -> None:
            raise RuntimeError("simulated release error")

    # Should not propagate the exception
    release_gpu_locks([BrokenLock()])  # type: ignore[list-item]


def test_release_gpu_locks_empty_list(tmp_path: Path) -> None:
    """release_gpu_locks handles empty list without error."""
    release_gpu_locks([])


def test_lock_dir_created_if_missing(tmp_path: Path) -> None:
    """acquire_gpu_locks creates lock_dir if it does not exist."""
    new_dir = tmp_path / "nested" / "llem"
    assert not new_dir.exists()

    locks = acquire_gpu_locks(["0"], lock_dir=new_dir)
    assert new_dir.exists()
    release_gpu_locks(locks)
