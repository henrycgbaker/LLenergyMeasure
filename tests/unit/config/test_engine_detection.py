"""Unit tests for config.engine_detection.py.

Covers:
- is_engine_available() - tries importing the engine package

All import-level side effects are mocked via unittest.mock.patch; no real
engine packages are imported or required.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager

import pytest

from llenergymeasure.config.engine_detection import (
    is_engine_available,
)


@contextmanager
def _hide_module(name: str):
    """Temporarily poison *name* in sys.modules so imports see it as missing.

    Snapshots and restores **all** submodules (``name.*``) to prevent
    order-dependent failures when pytest-randomly reorders tests.
    """
    saved = {k: sys.modules[k] for k in list(sys.modules) if k == name or k.startswith(f"{name}.")}
    sys.modules[name] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        # Remove any entries added during the poisoned import attempt
        for k in list(sys.modules):
            if k == name or k.startswith(f"{name}."):
                sys.modules.pop(k, None)
        sys.modules.update(saved)


# ---------------------------------------------------------------------------
# is_engine_available
# ---------------------------------------------------------------------------


class TestIsEngineAvailable:
    def test_returns_true_when_torch_importable(self):
        pytest.importorskip("torch")
        result = is_engine_available("transformers")
        assert result is True

    def test_returns_false_when_torch_not_importable(self):
        with _hide_module("torch"):
            result = is_engine_available("transformers")
        assert result is False

    def test_returns_false_when_vllm_not_importable(self):
        with _hide_module("vllm"):
            result = is_engine_available("vllm")
        assert result is False

    def test_returns_false_when_tensorrt_not_importable(self):
        with _hide_module("tensorrt_llm"):
            result = is_engine_available("tensorrt")
        assert result is False

    def test_returns_false_for_unknown_engine(self):
        result = is_engine_available("unknown_backend_xyz")
        assert result is False

    def test_returns_false_when_oserror_on_import(self):
        """OSError (e.g. missing .so) should be caught and return False."""
        import llenergymeasure.config.engine_detection as _bd_mod

        real_import = __import__

        def _raise_oserror(name, *args, **kwargs):
            if name == "tensorrt_llm":
                raise OSError("libcudart.so not found")
            return real_import(name, *args, **kwargs)

        original_builtins_import = _bd_mod.__builtins__.get("__import__")  # type: ignore[union-attr]
        with _hide_module("tensorrt_llm"):
            _bd_mod.__builtins__["__import__"] = _raise_oserror  # type: ignore[index]
            try:
                result = is_engine_available("tensorrt")
            finally:
                if original_builtins_import is not None:
                    _bd_mod.__builtins__["__import__"] = original_builtins_import  # type: ignore[index]
                else:
                    del _bd_mod.__builtins__["__import__"]  # type: ignore[attr-defined]

        assert result is False

    def test_unexpected_error_during_import_propagates(self):
        """A non-import error during import propagates (not masked as 'unavailable').

        Only ImportError / OSError mean "engine not installed"; an unexpected
        error (e.g. a CUDA init failure inside the package) is a real bug and
        must surface rather than being silently swallowed.
        """
        import llenergymeasure.config.engine_detection as _bd_mod

        real_import = __import__

        def _raise_runtime(name, *args, **kwargs):
            if name == "vllm":
                raise RuntimeError("CUDA init failed")
            return real_import(name, *args, **kwargs)

        original_builtins_import = _bd_mod.__builtins__.get("__import__")  # type: ignore[union-attr]
        with _hide_module("vllm"):
            _bd_mod.__builtins__["__import__"] = _raise_runtime  # type: ignore[index]
            try:
                with pytest.raises(RuntimeError, match="CUDA init failed"):
                    is_engine_available("vllm")
            finally:
                if original_builtins_import is not None:
                    _bd_mod.__builtins__["__import__"] = original_builtins_import  # type: ignore[index]
                else:
                    del _bd_mod.__builtins__["__import__"]  # type: ignore[attr-defined]
