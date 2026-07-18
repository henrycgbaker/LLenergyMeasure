"""Tests for utils/env_config.py physical-GPU lock-id parsing.

pinned_gpu_lock_ids() parses the LLEM_DOCKER_GPUS docker --gpus selector into
per-physical-device lock identifiers so study GPU locks name the physical
device, not the in-container logical index (which always starts at 0 under
pinning). See study/gpu_locks.py.
"""

from __future__ import annotations

import logging

import pytest

from llenergymeasure.utils.env_config import (
    ENV_DOCKER_GPUS,
    ENV_DOCKER_SHM_SIZE,
    docker_gpus,
    docker_gpus_arg,
    docker_gpus_cache_token,
    docker_shm_size,
    pinned_gpu_lock_ids,
    warn_on_gpu_selector_conflict,
)


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


# ---------------------------------------------------------------------------
# docker_gpus_arg: multi-device selectors must be quoted for `docker run --gpus`
# ---------------------------------------------------------------------------


def test_gpus_arg_multi_device_is_quoted(monkeypatch: pytest.MonkeyPatch) -> None:
    """device=1,3 -> '"device=1,3"' so docker does not split the comma into a count."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=1,3")
    assert docker_gpus_arg() == '"device=1,3"'


def test_gpus_arg_single_device_unquoted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single device=N has no comma and needs no quoting."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=2")
    assert docker_gpus_arg() == "device=2"


def test_gpus_arg_multi_uuid_is_quoted(monkeypatch: pytest.MonkeyPatch) -> None:
    """A comma-separated UUID device list is quoted too."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "device=GPU-abc,GPU-def")
    assert docker_gpus_arg() == '"device=GPU-abc,GPU-def"'


def test_gpus_arg_all_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """'all' (and the unset default) is returned verbatim."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "all")
    assert docker_gpus_arg() == "all"
    monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
    assert docker_gpus_arg() == "all"


def test_gpus_arg_count_verbatim(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare count is not a device list and is returned verbatim."""
    monkeypatch.setenv(ENV_DOCKER_GPUS, "2")
    assert docker_gpus_arg() == "2"


# ---------------------------------------------------------------------------
# Config gpu_indices -> --gpus selector, with env>config precedence
# ---------------------------------------------------------------------------


class TestConfigGpuIndices:
    """study_execution.gpu_indices scopes containers via --gpus device=<indices>,
    but the LLEM_DOCKER_GPUS env var overrides it (env>config).
    """

    def test_no_env_no_config_is_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Neither set -> 'all' (historical default preserved)."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert docker_gpus(None) == "all"
        assert docker_gpus_arg(None) == "all"

    def test_config_single_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A single config index -> device=N (no comma, no quoting)."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert docker_gpus([2]) == "device=2"
        assert docker_gpus_arg([2]) == "device=2"

    def test_config_multi_index(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Multiple config indices -> device=a,b, quoted for the docker arg."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert docker_gpus([2, 3]) == "device=2,3"
        assert docker_gpus_arg([2, 3]) == '"device=2,3"'

    def test_env_overrides_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LLEM_DOCKER_GPUS wins over config indices (env>config)."""
        monkeypatch.setenv(ENV_DOCKER_GPUS, "device=5")
        assert docker_gpus([2, 3]) == "device=5"
        assert docker_gpus_arg([2, 3]) == "device=5"

    def test_lock_ids_from_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config indices become physical lock ids (no env pinning present)."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert pinned_gpu_lock_ids([2, 3]) == ["2", "3"]

    def test_lock_ids_env_wins_over_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both set, lock ids come from the env selector (env>config)."""
        monkeypatch.setenv(ENV_DOCKER_GPUS, "device=5")
        assert pinned_gpu_lock_ids([2, 3]) == ["5"]


class TestDockerGpusCacheToken:
    """docker_gpus_cache_token qualifies the baseline cache key by physical GPU."""

    def test_unpinned_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'all' selector -> None (baseline cache key stays unqualified)."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert docker_gpus_cache_token(None) is None

    def test_config_pin_sanitised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Config indices -> filename-safe token (no '=' or ',')."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert docker_gpus_cache_token([2, 3]) == "device_2_3"

    def test_env_pin_sanitised(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Env selector is sanitised the same way and wins over config."""
        monkeypatch.setenv(ENV_DOCKER_GPUS, "device=7")
        assert docker_gpus_cache_token([2, 3]) == "device_7"

    def test_distinct_pins_distinct_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        assert docker_gpus_cache_token([2, 3]) != docker_gpus_cache_token([4, 5])


class TestGpuSelectorConflictWarning:
    def test_warns_when_both_set(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A warning fires (naming both) when env and config both scope GPUs."""
        monkeypatch.setenv(ENV_DOCKER_GPUS, "device=5")
        with caplog.at_level(logging.WARNING):
            warn_on_gpu_selector_conflict([2, 3])
        assert any("Env wins" in rec.message for rec in caplog.records)
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_no_warning_env_only(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Env set but no config indices -> no conflict, no warning."""
        monkeypatch.setenv(ENV_DOCKER_GPUS, "device=5")
        with caplog.at_level(logging.WARNING):
            warn_on_gpu_selector_conflict(None)
        assert caplog.records == []

    def test_no_warning_config_only(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Config indices but no env -> no conflict, no warning."""
        monkeypatch.delenv(ENV_DOCKER_GPUS, raising=False)
        with caplog.at_level(logging.WARNING):
            warn_on_gpu_selector_conflict([2, 3])
        assert caplog.records == []


class TestDockerShmSize:
    def test_default_is_8g(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Unset LLEM_DOCKER_SHM_SIZE -> the historical 8g default."""
        monkeypatch.delenv(ENV_DOCKER_SHM_SIZE, raising=False)
        assert docker_shm_size() == "8g"

    def test_empty_is_8g(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty value falls back to 8g."""
        monkeypatch.setenv(ENV_DOCKER_SHM_SIZE, "")
        assert docker_shm_size() == "8g"

    def test_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A set value is forwarded verbatim."""
        monkeypatch.setenv(ENV_DOCKER_SHM_SIZE, "16g")
        assert docker_shm_size() == "16g"
