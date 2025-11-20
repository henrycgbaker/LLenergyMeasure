"""
Enhanced logging utilities with structured logging and JSON output support.

Features:
- Structured logging with context
- JSON format for machine-readable logs
- Per-module log level configuration
- Progress tracking integration
- Performance logging
- Backwards compatible with existing setup_logging()
"""

import json
import logging
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from rich.logging import RichHandler


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.

    Outputs log records as JSON objects for easy parsing and analysis.
    """

    def __init__(
        self,
        include_extra: bool = True,
        timestamp_format: str = "iso",
    ):
        """
        Initialize JSON formatter.

        Args:
            include_extra: Include extra fields from log record
            timestamp_format: Timestamp format ('iso', 'unix', 'readable')
        """
        super().__init__()
        self.include_extra = include_extra
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": self._format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add location info
        if record.pathname:
            log_data["location"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields
        if self.include_extra:
            extra = {
                k: v
                for k, v in record.__dict__.items()
                if k not in [
                    "name",
                    "msg",
                    "args",
                    "created",
                    "filename",
                    "funcName",
                    "levelname",
                    "levelno",
                    "lineno",
                    "module",
                    "msecs",
                    "message",
                    "pathname",
                    "process",
                    "processName",
                    "relativeCreated",
                    "thread",
                    "threadName",
                    "exc_info",
                    "exc_text",
                    "stack_info",
                ]
                and not k.startswith("_")
            }
            if extra:
                log_data["extra"] = extra

        return json.dumps(log_data, default=str)

    def _format_timestamp(self, timestamp: float) -> Union[str, float]:
        """Format timestamp according to configuration."""
        if self.timestamp_format == "unix":
            return timestamp
        elif self.timestamp_format == "readable":
            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        else:  # iso
            return datetime.fromtimestamp(timestamp).isoformat()


class StructuredLogger:
    """
    Enhanced logger with structured logging support.

    Provides context-aware logging with automatic field injection.
    """

    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            context: Default context fields to include in all logs
        """
        self.logger = logging.getLogger(name)
        self.context = context or {}

    def _log_with_context(
        self,
        level: int,
        msg: str,
        extra: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Log message with context."""
        merged_extra = {**self.context, **(extra or {})}
        self.logger.log(level, msg, extra=merged_extra, **kwargs)

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, msg, extra, **kwargs)

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Log info message with context."""
        self._log_with_context(logging.INFO, msg, extra, **kwargs)

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, msg, extra, **kwargs)

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Log error message with context."""
        self._log_with_context(logging.ERROR, msg, extra, **kwargs)

    def critical(
        self, msg: str, extra: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, msg, extra, **kwargs)

    def add_context(self, **kwargs: Any) -> None:
        """Add fields to logger context."""
        self.context.update(kwargs)

    def remove_context(self, *keys: str) -> None:
        """Remove fields from logger context."""
        for key in keys:
            self.context.pop(key, None)

    @contextmanager
    def context_scope(self, **kwargs: Any):
        """
        Temporarily add context fields.

        Example:
            with logger.context_scope(experiment_id="exp123"):
                logger.info("Running experiment")
        """
        # Add context
        old_values = {}
        for key, value in kwargs.items():
            if key in self.context:
                old_values[key] = self.context[key]
            self.context[key] = value

        try:
            yield self
        finally:
            # Restore old context
            for key in kwargs:
                if key in old_values:
                    self.context[key] = old_values[key]
                else:
                    self.context.pop(key, None)


class LoggingConfig:
    """
    Centralized logging configuration manager.

    Supports per-module log levels and multiple output formats.
    """

    def __init__(self):
        """Initialize logging configuration."""
        self._configured = False
        self.handlers: Dict[str, logging.Handler] = {}

    def configure(
        self,
        level: Union[str, int] = "INFO",
        format: str = "rich",  # 'rich', 'json', 'standard'
        output_file: Optional[Path] = None,
        module_levels: Optional[Dict[str, Union[str, int]]] = None,
        include_timestamp: bool = True,
    ) -> None:
        """
        Configure logging for the application.

        Args:
            level: Default log level
            format: Output format ('rich', 'json', 'standard')
            output_file: Optional file to write logs to
            module_levels: Per-module log levels
            include_timestamp: Include timestamps in logs
        """
        if self._configured:
            # Reset existing configuration
            logging.root.handlers.clear()
            self.handlers.clear()

        # Convert level to int if string
        if isinstance(level, str):
            level = getattr(logging, level.upper())

        # Configure root logger
        logging.root.setLevel(level)

        # Create console handler
        if format == "rich":
            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=include_timestamp,
                show_path=True,
            )
            console_handler.setLevel(level)
            logging.root.addHandler(console_handler)
            self.handlers["console"] = console_handler

        elif format == "json":
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(JSONFormatter())
            console_handler.setLevel(level)
            logging.root.addHandler(console_handler)
            self.handlers["console"] = console_handler

        else:  # standard
            console_handler = logging.StreamHandler(sys.stdout)
            if include_timestamp:
                fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            else:
                fmt = "%(name)s - %(levelname)s - %(message)s"
            console_handler.setFormatter(logging.Formatter(fmt))
            console_handler.setLevel(level)
            logging.root.addHandler(console_handler)
            self.handlers["console"] = console_handler

        # Add file handler if specified
        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(output_file)
            if format == "json":
                file_handler.setFormatter(JSONFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )
            file_handler.setLevel(level)
            logging.root.addHandler(file_handler)
            self.handlers["file"] = file_handler

        # Configure per-module levels
        if module_levels:
            for module_name, module_level in module_levels.items():
                if isinstance(module_level, str):
                    module_level = getattr(logging, module_level.upper())
                logging.getLogger(module_name).setLevel(module_level)

        # Suppress noisy third-party loggers
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("torch").setLevel(logging.WARNING)

        self._configured = True

    def set_module_level(self, module_name: str, level: Union[str, int]) -> None:
        """Set log level for specific module."""
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logging.getLogger(module_name).setLevel(level)

    def get_structured_logger(
        self, name: str, context: Optional[Dict[str, Any]] = None
    ) -> StructuredLogger:
        """Get structured logger for module."""
        return StructuredLogger(name, context)


# Global configuration instance
_config = LoggingConfig()


def configure_logging(
    level: Union[str, int] = "INFO",
    format: str = "rich",
    output_file: Optional[Path] = None,
    module_levels: Optional[Dict[str, Union[str, int]]] = None,
    include_timestamp: bool = True,
) -> None:
    """
    Configure application logging.

    Args:
        level: Default log level
        format: Output format ('rich', 'json', 'standard')
        output_file: Optional file to write logs to
        module_levels: Per-module log levels
        include_timestamp: Include timestamps in logs

    Example:
        configure_logging(
            level="INFO",
            format="json",
            output_file=Path("./logs/app.log"),
            module_levels={
                "llm_efficiency.core": "DEBUG",
                "llm_efficiency.metrics": "WARNING",
            }
        )
    """
    _config.configure(level, format, output_file, module_levels, include_timestamp)


def get_structured_logger(
    name: str, context: Optional[Dict[str, Any]] = None
) -> StructuredLogger:
    """
    Get structured logger for module.

    Args:
        name: Logger name (usually __name__)
        context: Default context fields

    Returns:
        Structured logger instance

    Example:
        logger = get_structured_logger(__name__, context={"component": "inference"})
        logger.info("Processing batch", extra={"batch_id": 42})
    """
    return _config.get_structured_logger(name, context)


def set_module_level(module_name: str, level: Union[str, int]) -> None:
    """
    Set log level for specific module.

    Args:
        module_name: Module name
        level: Log level

    Example:
        set_module_level("llm_efficiency.core.inference", "DEBUG")
    """
    _config.set_module_level(module_name, level)


@contextmanager
def log_execution_time(
    logger: Union[logging.Logger, StructuredLogger],
    operation: str,
    level: int = logging.INFO,
):
    """
    Context manager to log execution time of an operation.

    Args:
        logger: Logger instance
        operation: Operation name
        level: Log level

    Example:
        with log_execution_time(logger, "model_loading"):
            model = load_model()
    """
    start_time = time.perf_counter()
    extra = {"operation": operation, "phase": "start"}

    if isinstance(logger, StructuredLogger):
        logger._log_with_context(level, f"Starting {operation}", extra)
    else:
        logger.log(level, f"Starting {operation}", extra=extra)

    try:
        yield
    finally:
        elapsed = time.perf_counter() - start_time
        extra = {
            "operation": operation,
            "phase": "complete",
            "duration_seconds": elapsed,
        }

        if isinstance(logger, StructuredLogger):
            logger._log_with_context(
                level, f"Completed {operation} in {elapsed:.3f}s", extra
            )
        else:
            logger.log(level, f"Completed {operation} in {elapsed:.3f}s", extra=extra)


@contextmanager
def log_progress(
    logger: Union[logging.Logger, StructuredLogger],
    operation: str,
    total: int,
    level: int = logging.INFO,
    log_interval: int = 10,
):
    """
    Context manager for logging progress.

    Args:
        logger: Logger instance
        operation: Operation name
        total: Total items to process
        level: Log level
        log_interval: Log every N items

    Yields:
        Function to call with current progress

    Example:
        with log_progress(logger, "processing_batches", total=100) as update:
            for i in range(100):
                # Do work
                update(i + 1)
    """
    start_time = time.perf_counter()
    last_logged = 0

    def update_progress(current: int) -> None:
        """Update progress."""
        nonlocal last_logged

        # Only log at intervals or on completion
        if current - last_logged < log_interval and current < total:
            return

        elapsed = time.perf_counter() - start_time
        percent = (current / total) * 100 if total > 0 else 0
        rate = current / elapsed if elapsed > 0 else 0
        eta = (total - current) / rate if rate > 0 else 0

        extra = {
            "operation": operation,
            "current": current,
            "total": total,
            "percent": percent,
            "rate": rate,
            "elapsed": elapsed,
            "eta": eta,
        }

        msg = f"{operation}: {current}/{total} ({percent:.1f}%) - {rate:.1f} items/s - ETA: {eta:.1f}s"

        if isinstance(logger, StructuredLogger):
            logger._log_with_context(level, msg, extra)
        else:
            logger.log(level, msg, extra=extra)

        last_logged = current

    try:
        yield update_progress
    finally:
        elapsed = time.perf_counter() - start_time
        extra = {
            "operation": operation,
            "total": total,
            "elapsed": elapsed,
            "rate": total / elapsed if elapsed > 0 else 0,
        }

        msg = f"Completed {operation}: {total} items in {elapsed:.3f}s"

        if isinstance(logger, StructuredLogger):
            logger._log_with_context(level, msg, extra)
        else:
            logger.log(level, msg, extra=extra)


def log_metrics(
    logger: Union[logging.Logger, StructuredLogger],
    metrics: Dict[str, Any],
    prefix: str = "",
    level: int = logging.INFO,
) -> None:
    """
    Log metrics dictionary.

    Args:
        logger: Logger instance
        metrics: Metrics dictionary
        prefix: Optional prefix for metric names
        level: Log level

    Example:
        log_metrics(logger, {
            "throughput": 123.45,
            "latency": 0.012,
            "energy": 0.000456,
        }, prefix="inference")
    """
    for key, value in metrics.items():
        metric_name = f"{prefix}.{key}" if prefix else key
        extra = {"metric_name": metric_name, "metric_value": value}

        msg = f"{metric_name}={value}"

        if isinstance(logger, StructuredLogger):
            logger._log_with_context(level, msg, extra)
        else:
            logger.log(level, msg, extra=extra)


# Backwards compatibility
def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    rich_output: bool = True,
) -> None:
    """
    Configure logging for the application (backwards compatible).

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        rich_output: Use Rich for beautiful console output
    """
    format = "rich" if rich_output else "standard"
    configure_logging(level=level, format=format, output_file=log_file)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance (backwards compatible).

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
