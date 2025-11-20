"""
Retry utilities with exponential backoff.

Provides decorators and functions for retrying operations that may fail.
"""

import logging
import time
from functools import wraps
from typing import Callable, Optional, Tuple, Type

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    max_delay: float = 60.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[int, Exception], None]] = None,
) -> Callable:
    """
    Decorator to retry a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
        max_delay: Maximum delay between retries
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Optional callback called on each retry (retry_count, exception)

    Example:
        @retry_with_exponential_backoff(max_retries=5)
        def download_model(url):
            # May fail with network errors
            return requests.get(url)
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt == max_retries:
                        logger.error(
                            f"{func.__name__} failed after {max_retries} retries: {e}"
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )

                    if on_retry:
                        on_retry(attempt + 1, e)

                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)

            # Should never reach here, but just in case
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


def retry_on_exception(
    func: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> any:
    """
    Retry a function call with simple fixed delay.

    Args:
        func: Function to retry
        max_retries: Maximum number of attempts
        delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            return func()
        except exceptions as e:
            last_exception = e
            logger.warning(
                f"Attempt {attempt + 1}/{max_retries} failed: {e}, "
                f"retrying in {delay}s..."
            )
            if attempt < max_retries - 1:
                time.sleep(delay)

    if last_exception:
        raise last_exception


class RetryContext:
    """
    Context manager for retrying operations.

    Example:
        with RetryContext(max_retries=3, delay=2.0) as retry:
            while retry.should_retry():
                try:
                    result = risky_operation()
                    retry.success()
                    break
                except Exception as e:
                    retry.failure(e)
    """

    def __init__(
        self,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
    ):
        self.max_retries = max_retries
        self.initial_delay = delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions
        self.attempt = 0
        self.current_delay = delay
        self.last_exception = None
        self._success = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and not self._success:
            if issubclass(exc_type, self.exceptions):
                logger.error(f"All {self.max_retries} retry attempts failed")
        return False

    def should_retry(self) -> bool:
        """Check if should continue retrying."""
        if self._success:
            return False

        if self.attempt == 0:
            self.attempt += 1
            return True

        if self.attempt >= self.max_retries:
            if self.last_exception:
                raise self.last_exception
            return False

        logger.warning(
            f"Retry attempt {self.attempt + 1}/{self.max_retries} "
            f"after {self.current_delay:.1f}s delay"
        )
        time.sleep(self.current_delay)
        self.current_delay *= self.backoff_factor
        self.attempt += 1
        return True

    def success(self):
        """Mark operation as successful."""
        self._success = True

    def failure(self, exception: Exception):
        """Record a failure."""
        self.last_exception = exception
        logger.debug(f"Operation failed: {exception}")
