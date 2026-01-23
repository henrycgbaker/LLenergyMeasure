"""Resilience utilities for LLM Bench framework."""

import functools
import gc
import time
from collections.abc import Callable
from typing import TypeVar

from loguru import logger

from llenergymeasure.exceptions import RetryableError

T = TypeVar("T")


def retry_on_error(
    max_retries: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (RetryableError,),
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
                            f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed: {e}")

            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state in retry logic")

        return wrapper

    return decorator


def cleanup_gpu_memory() -> None:
    """Attempt to free GPU memory after an error or between experiments."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.debug("GPU memory cache cleared")
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")


def safe_cleanup(cleanup_func: Callable[[], None]) -> Callable[[], None]:
    """Wrap a cleanup function to catch and log exceptions.

    Args:
        cleanup_func: Function to wrap.

    Returns:
        Wrapped function that won't raise exceptions.
    """

    @functools.wraps(cleanup_func)
    def wrapper() -> None:
        try:
            cleanup_func()
        except Exception as e:
            logger.warning(f"Cleanup function {cleanup_func.__name__} failed: {e}")

    return wrapper
