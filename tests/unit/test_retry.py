"""
Unit tests for retry utilities.

Tests exponential backoff, retry decorators, and context managers.
"""

import pytest
import time
from unittest.mock import Mock, patch

from llm_efficiency.utils.retry import (
    retry_with_exponential_backoff,
    retry_on_exception,
    RetryContext,
)


class TestRetryWithExponentialBackoff:
    """Tests for retry_with_exponential_backoff decorator."""

    def test_success_on_first_try(self):
        """Test function succeeds on first try."""
        mock_func = Mock(return_value="success")
        
        decorated = retry_with_exponential_backoff(max_retries=3)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 1

    def test_success_after_retries(self):
        """Test function succeeds after several retries."""
        mock_func = Mock(side_effect=[ValueError("fail"), ValueError("fail"), "success"])
        
        decorated = retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError,)
        )(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 3

    def test_fails_after_max_retries(self):
        """Test function fails after max retries."""
        mock_func = Mock(side_effect=ValueError("always fails"))
        
        decorated = retry_with_exponential_backoff(
            max_retries=2,
            initial_delay=0.01,
            exceptions=(ValueError,)
        )(mock_func)
        
        with pytest.raises(ValueError, match="always fails"):
            decorated()
        
        assert mock_func.call_count == 3  # Initial + 2 retries

    def test_exponential_backoff_delays(self):
        """Test that delays increase exponentially."""
        call_times = []
        
        def failing_func():
            call_times.append(time.time())
            raise ValueError("fail")
        
        decorated = retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.1,
            backoff_factor=2.0,
            exceptions=(ValueError,)
        )(failing_func)
        
        with pytest.raises(ValueError):
            decorated()
        
        # Check delays are increasing
        assert len(call_times) == 4  # Initial + 3 retries
        # Delays should be approximately 0.1, 0.2, 0.4 seconds
        # Allow some tolerance for test execution time
        if len(call_times) >= 4:
            delay1 = call_times[1] - call_times[0]
            delay2 = call_times[2] - call_times[1]
            assert delay2 > delay1  # Second delay should be larger

    def test_only_retries_specified_exceptions(self):
        """Test only specified exceptions are retried."""
        mock_func = Mock(side_effect=KeyError("not retryable"))
        
        decorated = retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError,)  # Only retry ValueError
        )(mock_func)
        
        with pytest.raises(KeyError):
            decorated()
        
        # Should fail immediately without retries
        assert mock_func.call_count == 1

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retry_calls = []
        
        def on_retry_cb(attempt, exception):
            retry_calls.append((attempt, str(exception)))
        
        mock_func = Mock(side_effect=[ValueError("fail1"), ValueError("fail2"), "success"])
        
        decorated = retry_with_exponential_backoff(
            max_retries=3,
            initial_delay=0.01,
            exceptions=(ValueError,),
            on_retry=on_retry_cb
        )(mock_func)
        
        decorated()
        
        assert len(retry_calls) == 2  # Called after each retry
        assert retry_calls[0] == (1, "fail1")
        assert retry_calls[1] == (2, "fail2")

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        call_times = []
        
        def failing_func():
            call_times.append(time.time())
            raise ValueError("fail")
        
        decorated = retry_with_exponential_backoff(
            max_retries=5,
            initial_delay=0.1,
            backoff_factor=10.0,  # Large factor
            max_delay=0.15,  # Cap at 0.15s
            exceptions=(ValueError,)
        )(failing_func)
        
        with pytest.raises(ValueError):
            decorated()
        
        # Later delays should not exceed max_delay
        if len(call_times) >= 3:
            delay3 = call_times[3] - call_times[2]
            assert delay3 <= 0.2  # Allow small margin


class TestRetryOnException:
    """Tests for retry_on_exception function."""

    def test_success_on_first_try(self):
        """Test successful function call."""
        func = Mock(return_value="result")
        
        result = retry_on_exception(func, max_retries=3, delay=0.01)
        
        assert result == "result"
        assert func.call_count == 1

    def test_success_after_retries(self):
        """Test success after some failures."""
        func = Mock(side_effect=[ValueError("fail"), "success"])
        
        result = retry_on_exception(
            func,
            max_retries=3,
            delay=0.01,
            exceptions=(ValueError,)
        )
        
        assert result == "success"
        assert func.call_count == 2

    def test_fails_after_max_retries(self):
        """Test failure after all retries exhausted."""
        func = Mock(side_effect=ValueError("always fails"))
        
        with pytest.raises(ValueError):
            retry_on_exception(func, max_retries=2, delay=0.01)


class TestRetryContext:
    """Tests for RetryContext context manager."""

    def test_success_on_first_try(self):
        """Test successful operation on first try."""
        attempts = 0
        
        with RetryContext(max_retries=3, delay=0.01) as retry:
            while retry.should_retry():
                attempts += 1
                retry.success()
                break
        
        assert attempts == 1

    def test_success_after_retries(self):
        """Test success after multiple retries."""
        attempts = 0
        
        with RetryContext(max_retries=5, delay=0.01, exceptions=(ValueError,)) as retry:
            while retry.should_retry():
                attempts += 1
                try:
                    if attempts < 3:
                        raise ValueError("fail")
                    retry.success()
                    break
                except ValueError as e:
                    retry.failure(e)
        
        assert attempts == 3

    def test_fails_after_max_retries(self):
        """Test failure after max retries."""
        attempts = 0
        
        with pytest.raises(ValueError):
            with RetryContext(max_retries=2, delay=0.01, exceptions=(ValueError,)) as retry:
                while retry.should_retry():
                    attempts += 1
                    retry.failure(ValueError("fail"))
        
        assert attempts == 2

    def test_should_retry_returns_false_after_success(self):
        """Test should_retry returns False after success."""
        with RetryContext(max_retries=5) as retry:
            assert retry.should_retry() is True
            retry.success()
            assert retry.should_retry() is False

    def test_exponential_backoff_in_context(self):
        """Test backoff factor is applied."""
        call_times = []
        
        with pytest.raises(ValueError):
            with RetryContext(
                max_retries=3,
                delay=0.05,
                backoff_factor=2.0,
                exceptions=(ValueError,)
            ) as retry:
                while retry.should_retry():
                    call_times.append(time.time())
                    retry.failure(ValueError("fail"))
        
        # Should have 3 attempts
        assert len(call_times) == 3
