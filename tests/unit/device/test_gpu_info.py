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
