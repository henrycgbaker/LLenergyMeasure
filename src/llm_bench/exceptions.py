"""Exception hierarchy for LLM Bench framework."""


class LLMBenchError(Exception):
    """Base exception for llm-bench."""


class ConfigurationError(LLMBenchError):
    """Invalid or missing configuration."""


class ModelLoadError(LLMBenchError):
    """Failed to load model or tokenizer."""


class InferenceError(LLMBenchError):
    """Error during model inference."""


class EnergyTrackingError(LLMBenchError):
    """Error in energy measurement backend."""


class AggregationError(LLMBenchError):
    """Error aggregating experiment results."""


class DistributedError(LLMBenchError):
    """Error in distributed/multi-GPU setup."""


class RetryableError(LLMBenchError):
    """Transient errors that can be retried (e.g., OOM, GPU issues).

    Use this for errors where a retry might succeed, such as:
    - CUDA out of memory (after cleanup)
    - Transient GPU communication errors
    - Temporary resource unavailability
    """

    def __init__(self, message: str, max_retries: int = 3):
        super().__init__(message)
        self.max_retries = max_retries
