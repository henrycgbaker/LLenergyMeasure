"""Structured logging setup for LLM Bench using loguru.

Supports three verbosity modes:
- quiet: WARNING+ only, no progress bars
- normal: INFO+ with simplified format, progress bars enabled
- verbose: DEBUG+ with full format (timestamps, module names)

Backend log filtering:
- In quiet/normal mode: Suppress noisy backend library logs (vLLM, TensorRT, HF)
- In verbose mode: Show all backend logs for debugging
"""

from __future__ import annotations

import os
import sys
from typing import Any, Literal

from loguru import logger

__all__ = ["VERBOSE_FORMAT", "configure_backend_log_filtering", "logger", "setup_logging"]

# Remove default handler at module load
logger.remove()

# Full format with timestamps and module info (verbose mode)
VERBOSE_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

# Simplified format without timestamps and module info (normal mode)
SIMPLE_FORMAT = "<level>{level: <8}</level> | <level>{message}</level>"

# JSON format for structured logging
JSON_FORMAT = "{message}"

# Type alias for verbosity levels
VerbosityType = Literal["quiet", "normal", "verbose"]

# Noisy loggers to filter in default/quiet modes
# These libraries produce excessive initialization logs that obscure CLI output
BACKEND_NOISY_LOGGERS = [
    # vLLM - ModelRunner initialization spam
    "vllm",
    "vllm.worker",
    "vllm.model_executor",
    "vllm.engine",
    # TensorRT - Engine building logs
    "tensorrt",
    "tensorrt_llm",
    # HuggingFace transformers/tokenizers
    "transformers",
    "transformers.tokenization_utils",
    "transformers.modeling_utils",
    "tokenizers",
    # Ray distributed (if used)
    "ray",
    "ray.worker",
    # Accelerate
    "accelerate",
]


def _get_verbosity_from_env() -> VerbosityType:
    """Get verbosity from LLM_ENERGY_VERBOSITY environment variable.

    Returns:
        Verbosity level string.
    """
    env_value = os.environ.get("LLM_ENERGY_VERBOSITY", "normal").lower()
    if env_value in ("quiet", "normal", "verbose"):
        return env_value  # type: ignore[return-value]
    return "normal"


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
    verbosity: VerbosityType | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, output logs as JSON for machine parsing.
        log_file: Optional file path to write logs to.
        verbosity: Override verbosity level ("quiet", "normal", "verbose").
                   If None, reads from LLM_ENERGY_VERBOSITY env var.
    """
    logger.remove()

    # Determine effective verbosity
    effective_verbosity = verbosity or _get_verbosity_from_env()

    # Map verbosity to log level and format
    if effective_verbosity == "quiet":
        effective_level = "WARNING"
        log_format = SIMPLE_FORMAT
    elif effective_verbosity == "verbose":
        effective_level = level if level != "INFO" else "DEBUG"
        log_format = VERBOSE_FORMAT
    else:  # normal
        effective_level = level
        log_format = SIMPLE_FORMAT

    if json_output:
        logger.add(
            sys.stderr,
            format=JSON_FORMAT,
            serialize=True,
            level=effective_level,
        )
    else:
        logger.add(
            sys.stderr,
            format=log_format,
            level=effective_level,
            colorize=True,
        )

    if log_file:
        # Always use verbose format for log files
        logger.add(
            log_file,
            format=VERBOSE_FORMAT,
            level="DEBUG",  # Capture everything to file
            rotation="10 MB",
            retention="7 days",
        )

    # Configure backend-specific filtering (suppress noisy libs in non-verbose mode)
    configure_backend_log_filtering(effective_verbosity)


def setup_logging_for_verbosity(verbosity: VerbosityType) -> None:
    """Convenience function to setup logging with verbosity level.

    Used by CLI to configure logging based on --quiet/--verbose flags.

    Args:
        verbosity: Verbosity level.
    """
    setup_logging(verbosity=verbosity)


def get_logger(name: str = "llenergymeasure") -> Any:
    """Get a logger instance bound with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Logger instance bound with the name context.
    """
    return logger.bind(name=name)


def configure_backend_log_filtering(verbosity: VerbosityType) -> None:
    """Configure logging filters for backend libraries.

    In quiet/normal mode: Suppress backend initialization logs (set to WARNING+)
    In verbose mode: Show all backend logs (set to DEBUG)

    This sets both:
    1. Python stdlib logging levels for libraries that use logging
    2. Environment variables for backends that print directly to stdout
       (vLLM uses VLLM_LOGGING_LEVEL, TensorRT uses TLLM_LOG_LEVEL)

    Args:
        verbosity: Current verbosity level.
    """
    import logging as stdlib_logging

    # Show everything in verbose mode, suppress INFO/DEBUG in normal/quiet
    level = stdlib_logging.DEBUG if verbosity == "verbose" else stdlib_logging.WARNING

    for logger_name in BACKEND_NOISY_LOGGERS:
        stdlib_logging.getLogger(logger_name).setLevel(level)

    # Set backend-specific environment variables for libraries that bypass Python logging
    # These must be set BEFORE the backend is imported
    if verbosity == "verbose":
        # Show all logs in verbose mode
        os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
        os.environ["TLLM_LOG_LEVEL"] = "DEBUG"
        # Let vLLM configure its own logging
        os.environ.pop("VLLM_CONFIGURE_LOGGING", None)
    else:
        # Suppress backend noise in normal/quiet mode
        os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
        os.environ["TLLM_LOG_LEVEL"] = "WARNING"
        # Disable vLLM's automatic logging configuration
        os.environ["VLLM_CONFIGURE_LOGGING"] = "0"


def is_backend_filtering_active() -> bool:
    """Check if backend log filtering is currently active.

    Returns:
        True if backend logs are being suppressed (non-verbose mode).
    """
    verbosity = _get_verbosity_from_env()
    return verbosity != "verbose"


def add_experiment_log_file(log_dir: str | os.PathLike[str], experiment_id: str) -> str:
    """Add a log file handler for experiment logs.

    Captures ALL logs (DEBUG level) to file, regardless of console verbosity.
    This ensures full debugging information is available even when console
    output is filtered.

    Args:
        log_dir: Directory for log files (e.g., results/<exp_id>/logs/).
        experiment_id: Experiment ID for log filename.

    Returns:
        Path to the log file as a string.
    """
    from pathlib import Path

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"{experiment_id}.log"

    # Add file handler at DEBUG level (capture everything)
    logger.add(
        str(log_file),
        format=VERBOSE_FORMAT,
        level="DEBUG",
        rotation="50 MB",
        retention="7 days",
        enqueue=True,  # Thread-safe
    )

    return str(log_file)
