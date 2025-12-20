"""Tests for resilience utilities."""

from unittest.mock import MagicMock, patch

import pytest

from llm_bench.exceptions import RetryableError
from llm_bench.resilience import cleanup_gpu_memory, retry_on_error, safe_cleanup


class TestRetryOnError:
    """Tests for retry_on_error decorator."""

    def test_succeeds_first_try(self):
        call_count = 0

        @retry_on_error(max_retries=3)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_retries_on_retryable_error(self):
        call_count = 0

        @retry_on_error(max_retries=2, delay_seconds=0.01)
        def fails_twice():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableError("transient error")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        @retry_on_error(max_retries=2, delay_seconds=0.01)
        def always_fails():
            raise RetryableError("persistent error")

        with pytest.raises(RetryableError, match="persistent error"):
            always_fails()

    def test_does_not_retry_other_exceptions(self):
        call_count = 0

        @retry_on_error(max_retries=3, delay_seconds=0.01)
        def raises_value_error():
            nonlocal call_count
            call_count += 1
            raise ValueError("not retryable")

        with pytest.raises(ValueError):
            raises_value_error()

        assert call_count == 1

    def test_backoff_factor(self):
        delays = []

        def mock_sleep(seconds):
            delays.append(seconds)

        @retry_on_error(max_retries=3, delay_seconds=0.1, backoff_factor=2.0)
        def always_fails():
            raise RetryableError("error")

        with patch("time.sleep", mock_sleep), pytest.raises(RetryableError):
            always_fails()

        # Should have 3 delays: 0.1, 0.2, 0.4
        assert len(delays) == 3
        assert delays[0] == pytest.approx(0.1)
        assert delays[1] == pytest.approx(0.2)
        assert delays[2] == pytest.approx(0.4)


class TestCleanupGpuMemory:
    """Tests for cleanup_gpu_memory function."""

    def test_no_error_without_torch(self):
        # Should not raise even if torch not available
        with patch.dict("sys.modules", {"torch": None}):
            cleanup_gpu_memory()

    def test_calls_cuda_methods_when_available(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            # Re-import to get the mock
            from llm_bench import resilience

            resilience.cleanup_gpu_memory()

        mock_torch.cuda.empty_cache.assert_called_once()
        mock_torch.cuda.synchronize.assert_called_once()


class TestSafeCleanup:
    """Tests for safe_cleanup wrapper."""

    def test_successful_cleanup(self):
        cleanup_called = False

        def my_cleanup():
            nonlocal cleanup_called
            cleanup_called = True

        wrapped = safe_cleanup(my_cleanup)
        wrapped()
        assert cleanup_called is True

    def test_exception_suppressed(self):
        def failing_cleanup():
            raise RuntimeError("cleanup failed")

        wrapped = safe_cleanup(failing_cleanup)
        # Should not raise
        wrapped()

    def test_preserves_function_name(self):
        def named_cleanup():
            pass

        wrapped = safe_cleanup(named_cleanup)
        assert wrapped.__name__ == "named_cleanup"
