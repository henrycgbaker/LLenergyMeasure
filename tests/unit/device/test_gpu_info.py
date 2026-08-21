"""Unit tests for device/gpu_info.py NVML lifecycle helpers.

Fully mocked - no GPU or NVIDIA driver required.
"""

from __future__ import annotations

import logging
import sys
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.device.gpu_info import nvml_context


def test_nvml_context_yields_on_failure_and_logs_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """nvml_context() stays fail-soft (still yields) but leaves a debug trace on failure.

    Regression guard: an NVML init/permission failure must not be swallowed silently -
    it should leave a diagnosable trace while preserving the swallow-and-continue contract.
    """
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.side_effect = RuntimeError("driver not loaded")

    yielded = False
    with (
        caplog.at_level(logging.DEBUG, logger="llenergymeasure.device.gpu_info"),
        patch.dict(sys.modules, {"pynvml": fake_pynvml}),
        nvml_context(),
    ):
        yielded = True  # fail-soft: caller body still runs

    assert yielded is True
    assert any("NVML unavailable" in rec.message for rec in caplog.records)


def test_nvml_context_no_failure_trace_on_success() -> None:
    """nvml_context() initialises and shuts down cleanly when NVML is available."""
    fake_pynvml = MagicMock()

    with patch.dict(sys.modules, {"pynvml": fake_pynvml}), nvml_context():
        pass

    fake_pynvml.nvmlInit.assert_called_once()
    fake_pynvml.nvmlShutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Physical <-> CUDA-visible index spaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2,3", [2, 3]),
        ("3,1", [3, 1]),
        (" 0 , 1 ", [0, 1]),
        ("1", [1]),
        ("", None),
        ("   ", None),
        ("GPU-abcdef", None),
        ("MIG-abcdef", None),
    ],
)
def test_cuda_visible_physical_order(monkeypatch, raw, expected) -> None:
    """The visible set parses in visibility order; UUID and empty forms yield None."""
    from llenergymeasure.device.gpu_info import cuda_visible_physical_order

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", raw)
    assert cuda_visible_physical_order() == expected


def test_cuda_visible_physical_order_unset(monkeypatch) -> None:
    """Unset means "no restriction", reported as None rather than an empty list."""
    from llenergymeasure.device.gpu_info import cuda_visible_physical_order

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert cuda_visible_physical_order() is None


def test_to_cuda_logical_indices_translates_under_a_restriction(monkeypatch) -> None:
    """Physical indices become positions in the visible set - what torch and Zeus want."""
    from llenergymeasure.device.gpu_info import to_cuda_logical_indices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert to_cuda_logical_indices([2, 3]) == [0, 1]
    assert to_cuda_logical_indices([3]) == [1]


def test_to_cuda_logical_indices_honours_visibility_order(monkeypatch) -> None:
    """The mapping follows the declared order, not numeric order."""
    from llenergymeasure.device.gpu_info import to_cuda_logical_indices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,1")
    assert to_cuda_logical_indices([3, 1]) == [0, 1]


def test_to_cuda_logical_indices_drops_invisible_devices(monkeypatch) -> None:
    """A physical device that is not visible has no logical counterpart."""
    from llenergymeasure.device.gpu_info import to_cuda_logical_indices

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert to_cuda_logical_indices([1, 2]) == [0]


def test_to_cuda_logical_indices_is_identity_when_unrestricted(monkeypatch) -> None:
    """Unset or UUID-valued CUDA_VISIBLE_DEVICES leaves the indices untouched."""
    from llenergymeasure.device.gpu_info import to_cuda_logical_indices

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert to_cuda_logical_indices([0, 2]) == [0, 2]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abcdef")
    assert to_cuda_logical_indices([0, 2]) == [0, 2]


def test_host_gpu_count_returns_none_when_nvml_absent(monkeypatch) -> None:
    """Fail-soft: no NVML means "unknown", not zero devices."""
    from llenergymeasure.device.gpu_info import host_gpu_count

    monkeypatch.setitem(sys.modules, "pynvml", None)
    assert host_gpu_count() is None


def test_host_gpu_count_reads_the_nvml_census(monkeypatch) -> None:
    """A working NVML yields the device count."""
    from llenergymeasure.device.gpu_info import host_gpu_count

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetCount.return_value = 4
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    assert host_gpu_count() == 4
