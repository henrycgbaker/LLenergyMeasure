"""Tests for exception hierarchy."""

import pytest

from llm_bench.exceptions import (
    AggregationError,
    ConfigurationError,
    DistributedError,
    EnergyTrackingError,
    InferenceError,
    LLMBenchError,
    ModelLoadError,
    RetryableError,
)


class TestExceptionHierarchy:
    """Test that all exceptions inherit from LLMBenchError."""

    def test_all_exceptions_inherit_from_base(self):
        exceptions = [
            ConfigurationError,
            ModelLoadError,
            InferenceError,
            EnergyTrackingError,
            AggregationError,
            DistributedError,
            RetryableError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, LLMBenchError)

    def test_base_exception_is_exception(self):
        assert issubclass(LLMBenchError, Exception)


class TestRetryableError:
    """Tests for RetryableError specifics."""

    def test_default_max_retries(self):
        err = RetryableError("test error")
        assert err.max_retries == 3

    def test_custom_max_retries(self):
        err = RetryableError("test error", max_retries=5)
        assert err.max_retries == 5

    def test_message_preserved(self):
        err = RetryableError("GPU OOM")
        assert str(err) == "GPU OOM"

    def test_can_be_caught_as_base(self):
        with pytest.raises(LLMBenchError):
            raise RetryableError("test")


class TestExceptionMessages:
    """Test exception instantiation with messages."""

    @pytest.mark.parametrize(
        "exc_class,message",
        [
            (ConfigurationError, "Invalid config file"),
            (ModelLoadError, "Model not found"),
            (InferenceError, "Generation failed"),
            (EnergyTrackingError, "CodeCarbon unavailable"),
            (AggregationError, "Missing process results"),
            (DistributedError, "GPU communication failed"),
        ],
    )
    def test_exception_with_message(self, exc_class, message):
        err = exc_class(message)
        assert str(err) == message
