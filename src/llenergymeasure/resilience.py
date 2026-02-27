"""Resilience utilities for llenergymeasure."""

import functools
import logging
import time
from collections.abc import Callable
from typing import TypeVar

from llenergymeasure.exceptions import LLEMError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_on_error(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (LLEMError,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying functions on transient errors.

    Args:
        max_retries: Maximum number of retry attempts.
        delay_seconds: Initial delay between retries.
        backoff_factor: Multiplier for delay after each retry.
        exceptions: Tuple of exception types to catch and retry.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exception: Exception | None = None
            delay = delay_seconds

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                            attempt + 1,
                            max_retries + 1,
                            e,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error("All %d attempts failed: %s", max_retries + 1, e)

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper

    return decorator
