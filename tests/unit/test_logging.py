"""Tests for logging utilities."""

import json
import logging
import tempfile
from pathlib import Path

import pytest

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
)


class TestJSONFormatter:
    """Tests for JSONFormatter."""

    def test_basic_formatting(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert "timestamp" in data
        assert data["location"]["line"] == 42

    def test_extra_fields(self):
        """Test extra fields are included."""
        formatter = JSONFormatter(include_extra=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test",
            args=(),
            exc_info=None,
        )
        record.experiment_id = "exp123"
        record.batch_num = 42

        result = formatter.format(record)
        data = json.loads(result)

        assert "extra" in data
        assert data["extra"]["experiment_id"] == "exp123"
        assert data["extra"]["batch_num"] == 42

    def test_exception_logging(self):
        """Test exception logging."""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Error occurred",
                args=(),
                exc_info=True,
            )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert data["exception"]["message"] == "Test error"
        assert "traceback" in data["exception"]


class TestStructuredLogger:
    """Tests for StructuredLogger."""

    def test_context_management(self):
        """Test logger context management."""
        logger = StructuredLogger("test", context={"component": "inference"})

        assert logger.context == {"component": "inference"}

        logger.add_context(experiment_id="exp123")
        assert logger.context["experiment_id"] == "exp123"

        logger.remove_context("experiment_id")
        assert "experiment_id" not in logger.context

    def test_context_scope(self):
        """Test temporary context scope."""
        logger = StructuredLogger("test")

        with logger.context_scope(batch=1, epoch=2):
            assert logger.context["batch"] == 1
            assert logger.context["epoch"] == 2

        # Context should be cleared
        assert "batch" not in logger.context
        assert "epoch" not in logger.context

    def test_nested_context_scopes(self):
        """Test nested context scopes."""
        logger = StructuredLogger("test", context={"component": "test"})

        with logger.context_scope(level=1):
            assert logger.context["level"] == 1

            with logger.context_scope(level=2):
                assert logger.context["level"] == 2

            # Should restore previous value
            assert logger.context["level"] == 1

        # Should restore original state
        assert "level" not in logger.context
        assert logger.context["component"] == "test"


class TestLoggingConfig:
    """Tests for LoggingConfig."""

    def test_basic_configuration(self):
        """Test basic logging configuration."""
        config = LoggingConfig()
        config.configure(level="INFO", format="standard")

        assert config._configured
        assert "console" in config.handlers

    def test_json_format(self):
        """Test JSON format configuration."""
        config = LoggingConfig()
        config.configure(level="DEBUG", format="json")

        assert config._configured
        assert isinstance(config.handlers["console"].formatter, JSONFormatter)

    def test_file_output(self):
        """Test file output configuration."""
        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_file = Path(f.name)

        try:
            config = LoggingConfig()
            config.configure(level="INFO", format="standard", output_file=log_file)

            assert "file" in config.handlers
            assert log_file.exists()
        finally:
            if log_file.exists():
                log_file.unlink()

    def test_per_module_levels(self):
        """Test per-module log level configuration."""
        config = LoggingConfig()
        config.configure(
            level="INFO",
            format="standard",
            module_levels={
                "llm_efficiency.core": "DEBUG",
                "llm_efficiency.metrics": "WARNING",
            },
        )

        core_logger = logging.getLogger("llm_efficiency.core")
        metrics_logger = logging.getLogger("llm_efficiency.metrics")

        assert core_logger.level == logging.DEBUG
        assert metrics_logger.level == logging.WARNING

    def test_set_module_level(self):
        """Test setting module level after configuration."""
        config = LoggingConfig()
        config.configure(level="INFO")

        config.set_module_level("test_module", "DEBUG")
        test_logger = logging.getLogger("test_module")

        assert test_logger.level == logging.DEBUG


class TestConfigureLogging:
    """Tests for configure_logging function."""

    def test_configure_with_defaults(self):
        """Test configuration with default settings."""
        configure_logging()

        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO

    def test_configure_with_json(self):
        """Test JSON format configuration."""
        configure_logging(level="DEBUG", format="json")

        # Check that configuration was applied
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_configure_with_module_levels(self):
        """Test per-module configuration."""
        configure_logging(
            level="INFO",
            module_levels={"test.module1": "DEBUG", "test.module2": "ERROR"},
        )

        module1 = logging.getLogger("test.module1")
        module2 = logging.getLogger("test.module2")

        assert module1.level == logging.DEBUG
        assert module2.level == logging.ERROR


class TestLogExecutionTime:
    """Tests for log_execution_time context manager."""

    def test_logs_execution_time(self, caplog):
        """Test that execution time is logged."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.INFO):
            with log_execution_time(logger, "test_operation"):
                pass

        assert "Starting test_operation" in caplog.text
        assert "Completed test_operation" in caplog.text

    def test_works_with_structured_logger(self, caplog):
        """Test with StructuredLogger."""
        logger = get_structured_logger("test")

        with caplog.at_level(logging.INFO):
            with log_execution_time(logger, "structured_op"):
                pass

        assert "Starting structured_op" in caplog.text
        assert "Completed structured_op" in caplog.text


class TestLogProgress:
    """Tests for log_progress context manager."""

    def test_logs_progress(self, caplog):
        """Test progress logging."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.INFO):
            with log_progress(logger, "processing", total=10, log_interval=5) as update:
                for i in range(10):
                    update(i + 1)

        # Should log at intervals
        assert "processing:" in caplog.text
        assert "Completed processing" in caplog.text

    def test_logs_final_summary(self, caplog):
        """Test final summary is logged."""
        logger = logging.getLogger("test")

        with caplog.at_level(logging.INFO):
            with log_progress(logger, "task", total=5, log_interval=1) as update:
                for i in range(5):
                    update(i + 1)

        assert "Completed task: 5 items" in caplog.text


class TestLogMetrics:
    """Tests for log_metrics function."""

    def test_logs_metrics(self, caplog):
        """Test metrics logging."""
        logger = logging.getLogger("test")

        metrics = {
            "throughput": 123.45,
            "latency": 0.012,
            "energy": 0.000456,
        }

        with caplog.at_level(logging.INFO):
            log_metrics(logger, metrics, prefix="inference")

        assert "inference.throughput=123.45" in caplog.text
        assert "inference.latency=0.012" in caplog.text
        assert "inference.energy=0.000456" in caplog.text

    def test_logs_without_prefix(self, caplog):
        """Test metrics logging without prefix."""
        logger = logging.getLogger("test")

        metrics = {"metric1": 1, "metric2": 2}

        with caplog.at_level(logging.INFO):
            log_metrics(logger, metrics)

        assert "metric1=1" in caplog.text
        assert "metric2=2" in caplog.text
