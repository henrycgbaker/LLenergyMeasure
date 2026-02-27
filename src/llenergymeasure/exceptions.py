"""Exception hierarchy for llenergymeasure."""


class LLEMError(Exception):
    """Base exception for llenergymeasure."""


class ConfigError(LLEMError):
    """Invalid or missing configuration."""


class BackendError(LLEMError):
    """Error from an inference backend (load, run, timeout)."""


class PreFlightError(LLEMError):
    """Pre-flight check failed before GPU allocation."""


class ExperimentError(LLEMError):
    """Error during experiment execution."""


class StudyError(LLEMError):
    """Error during study orchestration."""


class InvalidStateTransitionError(ExperimentError):
    """Invalid state machine transition."""

    def __init__(self, from_state: str, to_state: str):
        super().__init__(f"Invalid transition: {from_state} -> {to_state}")
        self.from_state = from_state
        self.to_state = to_state


# ---------------------------------------------------------------------------
# v1.x compatibility aliases — removed in a later phase when consumers migrate
# ---------------------------------------------------------------------------
ConfigurationError = ConfigError
AggregationError = ExperimentError
BackendInferenceError = BackendError
BackendInitializationError = BackendError
BackendNotAvailableError = BackendError
BackendConfigError = ConfigError
BackendTimeoutError = BackendError
