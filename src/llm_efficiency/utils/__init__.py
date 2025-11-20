"""Utility modules for LLM Efficiency framework."""

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
from llm_efficiency.utils.retry import (
    retry_with_exponential_backoff,
    retry_on_exception,
    RetryContext,
)
from llm_efficiency.utils.profiling import (
    PerformanceProfiler,
    ProfileResult,
    profile_function,
    timer,
    get_memory_usage,
    get_cpu_usage,
)
from llm_efficiency.utils.cache import (
    LRUCacheWithTTL,
    DiskCache,
    cached_with_ttl,
)

__all__ = [
    # Exceptions
    "LLMEfficiencyError",
    "ModelLoadingError",
    "InferenceError",
    "ConfigurationError",
    "DataError",
    "MetricsError",
    "StorageError",
    "NetworkError",
    "QuantizationError",
    # Retry utilities
    "retry_with_exponential_backoff",
    "retry_on_exception",
    "RetryContext",
    # Profiling
    "PerformanceProfiler",
    "ProfileResult",
    "profile_function",
    "timer",
    "get_memory_usage",
    "get_cpu_usage",
    # Caching
    "LRUCacheWithTTL",
    "DiskCache",
    "cached_with_ttl",
]
