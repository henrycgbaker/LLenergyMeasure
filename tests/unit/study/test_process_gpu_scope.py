"""Tests for physical GPU scoping on the process runner path.

The container runner scopes devices with ``docker run --gpus``; the process
runner has only ``CUDA_VISIBLE_DEVICES``. These tests pin the two consequences:
a spawned worker restricts itself (and nothing else) to the allowed devices, and
a scoped single experiment never takes the in-process fast path (where the
scope could not be enforced without mutating the caller's environment).
"""

from __future__ import annotations

import os

import pytest

from llenergymeasure.study.orchestration import _takes_single_fast_path
from llenergymeasure.study.worker import _cuda_visible_devices_value, _scope_to_gpu_indices


@pytest.fixture(autouse=True)
def _hermetic_cuda_visible_devices():
    """Snapshot and restore CUDA_VISIBLE_DEVICES around every test in this file.

    The code under test (``_scope_to_gpu_indices``) writes ``os.environ``
    directly, and pytest's MonkeyPatch records nothing for a key that was absent
    when ``delenv(raising=False)`` ran - so without this, a value like ``"2,3"``
    leaks into the process and poisons later tests in the same run (translation
    via ``to_cuda_logical_indices`` then silently empties unrelated index lists).
    """
    saved = os.environ.get("CUDA_VISIBLE_DEVICES")
    yield
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved


# ---------------------------------------------------------------------------
# Spawned worker: CUDA_VISIBLE_DEVICES from the allowed physical indices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("indices", "expected"),
    [([2], "2"), ([2, 3], "2,3"), ([0, 1, 2, 3], "0,1,2,3")],
)
def test__cuda_visible_devices_value(indices, expected) -> None:
    assert _cuda_visible_devices_value(indices) == expected


def test_scope_sets_cuda_visible_devices(monkeypatch) -> None:
    """An allowlist becomes the process's CUDA_VISIBLE_DEVICES."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _scope_to_gpu_indices([2, 3])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_scope_is_a_noop_without_an_allowlist(monkeypatch) -> None:
    """No allowlist leaves the variable exactly as it was - today's behaviour."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _scope_to_gpu_indices(None)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_scope_overrides_an_inherited_value(monkeypatch) -> None:
    """The allowlist is absolute: it names physical devices, so it replaces any
    inherited visibility rather than indexing into it."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    _scope_to_gpu_indices([1])
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_worker_scopes_before_importing_torch() -> None:
    """The scoping call must precede every engine/harness import in the worker body.

    torch reads CUDA_VISIBLE_DEVICES when it initialises CUDA, so a later call
    would silently do nothing. Guarded structurally: the source order is the
    contract, and it is easy to break by moving one line.
    """
    import inspect

    from llenergymeasure.study import worker

    body = inspect.getsource(worker._run_experiment_worker)
    assert body.index("_scope_to_gpu_indices(gpu_indices)") < body.index("capture_runtime_obs")


# ---------------------------------------------------------------------------
# Dispatch: a scoped single experiment never takes the in-process fast path
# ---------------------------------------------------------------------------


def _single_offline_study(gpu_indices: list[int] | None):
    from llenergymeasure.config.models import ExperimentConfig, StudyConfig

    return StudyConfig(
        experiments=[ExperimentConfig(task={"model": "m1"}, engine="vllm", serving_mode="offline")],
        study_execution={"n_cycles": 1, "gpu_indices": gpu_indices},
    )


def test_unscoped_offline_single_keeps_the_fast_path() -> None:
    """Without a GPU scope the offline single experiment runs in-process."""
    assert _takes_single_fast_path(_single_offline_study(None)) is True


def test_scoped_single_routes_through_the_runner() -> None:
    """A resolved scope forces StudyRunner, whose worker subprocess enforces it.

    The fast path runs in the calling process, where enforcing the scope would
    mean mutating the caller's CUDA_VISIBLE_DEVICES - so a scoped single must
    not take it. Enforcement, not a warning.
    """
    assert _takes_single_fast_path(_single_offline_study([2, 3])) is False
