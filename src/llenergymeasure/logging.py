"""Structured logging setup for LLM Bench using loguru.

Supports three verbosity modes:
- quiet: WARNING+ only, no progress bars
- normal: INFO+ with simplified format, progress bars enabled
- verbose: DEBUG+ with full format (timestamps, module names)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Literal

from loguru import logger

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
