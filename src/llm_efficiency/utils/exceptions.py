"""
Custom exceptions for LLM Efficiency framework.

Provides specific exception types for better error handling and debugging.
"""


class LLMEfficiencyError(Exception):
    """Base exception for all LLM Efficiency errors."""

    pass


class ModelLoadingError(LLMEfficiencyError):
    """Raised when model loading fails."""

    pass


class InferenceError(LLMEfficiencyError):
    """Raised when inference fails."""

    pass


class ConfigurationError(LLMEfficiencyError):
    """Raised when configuration is invalid."""

    pass


class DataError(LLMEfficiencyError):
    """Raised when data loading or processing fails."""

    pass


class MetricsError(LLMEfficiencyError):
    """Raised when metrics calculation fails."""

    pass


class StorageError(LLMEfficiencyError):
    """Raised when results storage/loading fails."""

    pass


class NetworkError(LLMEfficiencyError):
    """Raised when network operations fail."""

    pass


class QuantizationError(LLMEfficiencyError):
    """Raised when quantization is not supported or fails."""

    pass
