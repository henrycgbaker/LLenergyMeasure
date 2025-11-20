"""
Unit tests for custom exceptions.

Tests exception hierarchy and error types.
"""

import pytest

from llm_efficiency.utils.exceptions import (
    LLMEfficiencyError,
    ModelLoadingError,
    InferenceError,
    ConfigurationError,
    DataError,
    MetricsError,
    StorageError,
    NetworkError,
    QuantizationError,
)


class TestExceptionHierarchy:
    """Tests for exception hierarchy."""

    def test_base_exception(self):
        """Test base LLMEfficiencyError."""
        error = LLMEfficiencyError("base error")
        assert str(error) == "base error"
        assert isinstance(error, Exception)

    def test_model_loading_error(self):
        """Test ModelLoadingError inherits from base."""
        error = ModelLoadingError("model loading failed")
        assert isinstance(error, LLMEfficiencyError)
        assert isinstance(error, Exception)

    def test_inference_error(self):
        """Test InferenceError inherits from base."""
        error = InferenceError("inference failed")
        assert isinstance(error, LLMEfficiencyError)

    def test_configuration_error(self):
        """Test ConfigurationError inherits from base."""
        error = ConfigurationError("invalid config")
        assert isinstance(error, LLMEfficiencyError)

    def test_data_error(self):
        """Test DataError inherits from base."""
        error = DataError("data loading failed")
        assert isinstance(error, LLMEfficiencyError)

    def test_metrics_error(self):
        """Test MetricsError inherits from base."""
        error = MetricsError("metrics calculation failed")
        assert isinstance(error, LLMEfficiencyError)

    def test_storage_error(self):
        """Test StorageError inherits from base."""
        error = StorageError("storage failed")
        assert isinstance(error, LLMEfficiencyError)

    def test_network_error(self):
        """Test NetworkError inherits from base."""
        error = NetworkError("network request failed")
        assert isinstance(error, LLMEfficiencyError)

    def test_quantization_error(self):
        """Test QuantizationError inherits from base."""
        error = QuantizationError("quantization not supported")
        assert isinstance(error, LLMEfficiencyError)


class TestExceptionCatching:
    """Tests for exception catching."""

    def test_catch_specific_exception(self):
        """Test catching specific exception type."""
        with pytest.raises(ModelLoadingError):
            raise ModelLoadingError("test")

    def test_catch_base_exception(self):
        """Test catching via base exception."""
        with pytest.raises(LLMEfficiencyError):
            raise InferenceError("test")

    def test_raise_with_cause(self):
        """Test raising with cause chain."""
        original = ValueError("original error")
        
        with pytest.raises(ModelLoadingError) as exc_info:
            try:
                raise original
            except ValueError as e:
                raise ModelLoadingError("wrapped error") from e
        
        assert exc_info.value.__cause__ == original
