"""Structured logging setup for LLM Bench using loguru."""

import sys
from typing import Any

from loguru import logger

# Remove default handler
logger.remove()

# Default format for console output
DEFAULT_FORMAT = (
    "<green>{time:HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan> - "
    "<level>{message}</level>"
)

# JSON format for structured logging
JSON_FORMAT = "{message}"


def setup_logging(
    level: str = "INFO",
    json_output: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, output logs as JSON for machine parsing.
        log_file: Optional file path to write logs to.
    """
    logger.remove()

    if json_output:
        logger.add(
            sys.stderr,
            format=JSON_FORMAT,
            serialize=True,
            level=level,
        )
    else:
        logger.add(
            sys.stderr,
            format=DEFAULT_FORMAT,
            level=level,
            colorize=True,
        )

    if log_file:
        logger.add(
            log_file,
            format=DEFAULT_FORMAT,
            level=level,
            rotation="10 MB",
            retention="7 days",
        )


def get_logger(name: str = "llm_energy_measure") -> Any:
    """Get a logger instance bound with the given name.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Logger instance bound with the name context.
    """
    return logger.bind(name=name)
