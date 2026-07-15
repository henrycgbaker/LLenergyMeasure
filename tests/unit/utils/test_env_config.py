"""Tests for utils/env_config.py physical-GPU lock-id parsing.

pinned_gpu_lock_ids() parses the LLEM_DOCKER_GPUS docker --gpus selector into
per-physical-device lock identifiers so study GPU locks name the physical
device, not the in-container logical index (which always starts at 0 under
pinning). See study/gpu_locks.py.
"""

from __future__ import annotations

import pytest

from llenergymeasure.utils.env_config import ENV_DOCKER_GPUS, pinned_gpu_lock_ids


def test_unset_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unset LLEM_DOCKER_GPUS -> None (every GPU visible; use logical indices)."""
    monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
    assert pinned_gpu_lock_ids() is None


def test_all_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explicit 'all' -> None (logical == physical fallback)."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "all")
    assert pinned_gpu_lock_ids() is None


def test_empty_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty value falls back to 'all' -> None."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "")
    assert pinned_gpu_lock_ids() is None


def test_single_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=2 -> ["2"] (the physical device index)."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2")
    assert pinned_gpu_lock_ids() == ["2"]


def test_multi_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=2,3 -> ["2", "3"]."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2,3")
    assert pinned_gpu_lock_ids() == ["2", "3"]


def test_whitespace_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surrounding whitespace on tokens is stripped."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2, 3")
    assert pinned_gpu_lock_ids() == ["2", "3"]


def test_empty_tokens_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty tokens from stray commas are dropped."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2,,3")
    assert pinned_gpu_lock_ids() == ["2", "3"]


def test_empty_body_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """device= with no ids -> None (nothing to pin)."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=")
    assert pinned_gpu_lock_ids() is None


def test_uuid_single(monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU-UUID selector is used verbatim as a stable per-device lock id."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=GPU-abc-123")
    assert pinned_gpu_lock_ids() == ["GPU-abc-123"]


def test_uuid_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple UUIDs each become their own lock id."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=GPU-abc,GPU-def")
    assert pinned_gpu_lock_ids() == ["GPU-abc", "GPU-def"]


def test_count_form_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """count=N does not name specific devices -> None (fall back to logical)."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "count=2")
    assert pinned_gpu_lock_ids() is None


def test_bare_count_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare integer count (docker's --gpus 2 form) -> None."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "2")
    assert pinned_gpu_lock_ids() is None


def test_unrecognised_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unrecognised selector shape -> None."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "garbage")
    assert pinned_gpu_lock_ids() is None


def test_traversal_sanitized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path separators in a pathological value cannot escape the lock dir."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=../../etc")
    ids = pinned_gpu_lock_ids()
    assert ids is not None
    assert "/" not in ids[0]
    assert ids == [".._.._etc"]


def test_different_devices_yield_different_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distinct physical devices produce distinct lock ids (no false collision)."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2")
    ids_two = pinned_gpu_lock_ids()
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=3")
    ids_three = pinned_gpu_lock_ids()
    assert ids_two != ids_three


def test_same_device_yields_same_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same physical device always maps to the same lock id (real collision)."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2")
    first = pinned_gpu_lock_ids()
    second = pinned_gpu_lock_ids()
    assert first == second == ["2"]
