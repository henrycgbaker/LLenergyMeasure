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
from llm_efficiency.utils.logging import (
    JSONFormatter,
    StructuredLogger,
    LoggingConfig,
    configure_logging,
    get_structured_logger,
    set_module_level,
    log_execution_time,
    log_progress,
    log_metrics,
    setup_logging,
    get_logger,
)
from llm_efficiency.utils.validation import (
    validate_positive_int,
    validate_positive_float,
    validate_in_range,
    validate_choice,
    validate_path_exists,
    validate_model_name,
    validate_precision,
    validate_batch_config,
    validate_quantization_config,
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
    # Logging
    "JSONFormatter",
    "StructuredLogger",
    "LoggingConfig",
    "configure_logging",
    "get_structured_logger",
    "set_module_level",
    "log_execution_time",
    "log_progress",
    "log_metrics",
    "setup_logging",
    "get_logger",
    # Validation
    "validate_positive_int",
    "validate_positive_float",
    "validate_in_range",
    "validate_choice",
    "validate_path_exists",
    "validate_model_name",
    "validate_precision",
    "validate_batch_config",
    "validate_quantization_config",
]
