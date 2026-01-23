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


# Backend-specific errors


class BackendError(LLMBenchError):
    """Base class for inference backend errors."""


class BackendNotAvailableError(BackendError):
    """Backend is not installed or not usable.

    Raised when a backend's dependencies are missing or the backend
    reports it cannot be used on the current system.
    """

    def __init__(self, backend: str, install_hint: str | None = None):
        msg = f"Backend '{backend}' is not available"
        if install_hint:
            msg += f". Install with: {install_hint}"
        super().__init__(msg)
        self.backend = backend
        self.install_hint = install_hint


class BackendInitializationError(BackendError):
    """Failed to initialize backend (model loading, memory allocation, etc.)."""


class BackendInferenceError(BackendError):
    """Error during inference execution."""


class BackendTimeoutError(BackendError):
    """Inference exceeded timeout limit."""

    def __init__(self, backend: str, timeout_sec: float):
        super().__init__(
            f"Inference timed out after {timeout_sec}s on backend '{backend}'. "
            "Consider reducing batch size or prompt count."
        )
        self.backend = backend
        self.timeout_sec = timeout_sec


class BackendConfigError(BackendError):
    """Configuration is incompatible with the selected backend."""

    def __init__(self, backend: str, param: str, message: str):
        super().__init__(f"Backend '{backend}': parameter '{param}' - {message}")
        self.backend = backend
        self.param = param


# State machine errors


class InvalidStateTransitionError(LLMBenchError):
    """Invalid state machine transition attempted.

    Raised when code attempts to transition an experiment or process
    to a state that is not valid from the current state.
    """

    def __init__(self, from_status: str, to_status: str, entity: str = "experiment"):
        super().__init__(f"Invalid {entity} state transition: {from_status} -> {to_status}")
        self.from_status = from_status
        self.to_status = to_status
        self.entity = entity
