"""Tests for physical GPU scoping on the process runner path.

The container runner scopes devices with ``docker run --gpus``; the process
runner has only ``CUDA_VISIBLE_DEVICES``. These tests pin the two consequences:
a spawned worker restricts itself (and nothing else) to the allowed devices, and
the in-process single-experiment leg says out loud that it cannot.
"""

from __future__ import annotations

import logging

import pytest

from llenergymeasure.study.single import _warn_unenforceable_gpu_scope
from llenergymeasure.study.worker import _cuda_visible_devices_value, _scope_to_gpu_indices

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
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_scope_is_a_noop_without_an_allowlist(monkeypatch) -> None:
    """No allowlist leaves the variable exactly as it was - today's behaviour."""
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _scope_to_gpu_indices(None)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_scope_overrides_an_inherited_value(monkeypatch) -> None:
    """The allowlist is absolute: it names physical devices, so it replaces any
    inherited visibility rather than indexing into it."""
    import os

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
# In-process leg: honest about what it cannot enforce
# ---------------------------------------------------------------------------


def test_in_process_leg_warns_when_a_scope_cannot_be_enforced(caplog) -> None:
    """The warning names the scope, the reason, and both ways out."""
    with caplog.at_level(logging.WARNING):
        _warn_unenforceable_gpu_scope([2, 3])

    messages = [rec.getMessage() for rec in caplog.records]
    assert len(messages) == 1
    assert "GPU scope [2, 3] cannot be enforced" in messages[0]
    assert "CUDA_VISIBLE_DEVICES=2,3" in messages[0]
    assert "container runner" in messages[0]


def test_in_process_leg_silent_without_a_scope(caplog) -> None:
    """No allowlist, nothing to warn about."""
    with caplog.at_level(logging.WARNING):
        _warn_unenforceable_gpu_scope(None)
    assert caplog.records == []


def test_in_process_leg_never_mutates_the_callers_environment(monkeypatch) -> None:
    """The warning is the whole of the in-process response: no env is touched."""
    import os

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    _warn_unenforceable_gpu_scope([2, 3])
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
