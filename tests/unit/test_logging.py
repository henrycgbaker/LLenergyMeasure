"""Tests for logging configuration."""

from loguru import logger

from llm_energy_measure.logging import get_logger, setup_logging


class TestSetupLogging:
    """Tests for setup_logging function."""

    def teardown_method(self):
        """Reset logger after each test."""
        logger.remove()

    def test_setup_logging_default(self):
        setup_logging()
        # Should not raise
        logger.info("test message")

    def test_setup_logging_debug_level(self):
        setup_logging(level="DEBUG")
        logger.debug("debug message")

    def test_setup_logging_json_output(self):
        setup_logging(json_output=True)
        logger.info("json test")


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_bound_logger(self):
        log = get_logger("test_module")
        # Should be able to call logging methods
        assert hasattr(log, "info")
        assert hasattr(log, "debug")
        assert hasattr(log, "error")

    def test_get_logger_default_name(self):
        log = get_logger()
        assert hasattr(log, "info")
