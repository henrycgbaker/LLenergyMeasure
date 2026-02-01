"""Tests for backend_detection module."""

from unittest.mock import patch

from llenergymeasure.config.backend_detection import (
    KNOWN_BACKENDS,
    get_available_backends,
    get_backend_install_hint,
    is_backend_available,
)


class TestBackendConstants:
    """Tests for backend constants."""

    def test_known_backends_list(self):
        """KNOWN_BACKENDS contains expected backends."""
        assert "pytorch" in KNOWN_BACKENDS
        assert "vllm" in KNOWN_BACKENDS
        assert "tensorrt" in KNOWN_BACKENDS
        assert len(KNOWN_BACKENDS) == 3


class TestBackendAvailability:
    """Tests for backend availability detection."""

    def test_is_backend_available_pytorch(self):
        """PyTorch backend available if torch importable."""
        # In dev environment, torch is likely installed
        # Test both possible outcomes
        result = is_backend_available("pytorch")
        assert isinstance(result, bool)

    def test_is_backend_available_unknown(self):
        """Unknown backend returns False."""
        result = is_backend_available("nonexistent_backend")
        assert result is False

    @patch("llenergymeasure.config.backend_detection.is_backend_available")
    def test_get_available_backends(self, mock_is_available):
        """get_available_backends returns list of available backends."""

        # Mock pytorch available, others not
        def side_effect(backend):
            return backend == "pytorch"

        mock_is_available.side_effect = side_effect

        result = get_available_backends()

        assert isinstance(result, list)
        assert "pytorch" in result
        assert len(result) == 1

    @patch("llenergymeasure.config.backend_detection.is_backend_available")
    def test_get_available_backends_multiple(self, mock_is_available):
        """get_available_backends handles multiple backends."""
        # Mock all backends available
        mock_is_available.return_value = True

        result = get_available_backends()

        assert len(result) == 3
        assert "pytorch" in result
        assert "vllm" in result
        assert "tensorrt" in result


class TestBackendInstallHints:
    """Tests for backend installation hints."""

    def test_get_backend_install_hint_pytorch(self):
        """PyTorch install hint correct."""
        hint = get_backend_install_hint("pytorch")
        assert hint == "pip install llenergymeasure"

    def test_get_backend_install_hint_vllm(self):
        """vLLM hint steers users to Docker."""
        hint = get_backend_install_hint("vllm")
        assert "Docker" in hint

    def test_get_backend_install_hint_tensorrt(self):
        """TensorRT hint steers users to Docker."""
        hint = get_backend_install_hint("tensorrt")
        assert "Docker" in hint

    def test_get_backend_install_hint_unknown(self):
        """Unknown backend returns reasonable fallback."""
        hint = get_backend_install_hint("unknown_backend")
        assert "pip install llenergymeasure[unknown_backend]" in hint


class TestBackendImportErrors:
    """Tests for import error handling."""

    def test_is_backend_available_import_error(self):
        """is_backend_available handles ImportError gracefully."""
        # Mock torch module to raise ImportError when accessed
        with patch.dict("sys.modules", {"torch": None}):
            result = is_backend_available("pytorch")

        # Should return False, not raise
        assert result is False

    def test_is_backend_available_vllm_unavailable(self):
        """is_backend_available returns False for vLLM when not installed."""
        # Mock vllm module to raise ImportError
        with patch.dict("sys.modules", {"vllm": None}):
            result = is_backend_available("vllm")

        assert result is False
