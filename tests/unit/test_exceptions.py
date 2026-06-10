"""Unit tests for the exception hierarchy in llenergymeasure.exceptions.

Confirms the flat hierarchy: LLEMError + 5 direct subclasses.
"""

from __future__ import annotations

import pytest

from llenergymeasure.utils.exceptions import (
    ConfigError,
    EngineError,
    ExperimentError,
    LLEMError,
    PreFlightError,
    StudyError,
)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


def test_llem_error_is_base():
    """LLEMError is a subclass of Exception."""
    assert issubclass(LLEMError, Exception)


# ---------------------------------------------------------------------------
# Direct subclasses of LLEMError
# ---------------------------------------------------------------------------


def test_config_error_inherits_llem_error():
    """ConfigError is a subclass of LLEMError."""
    assert issubclass(ConfigError, LLEMError)


def test_engine_error_inherits_llem_error():
    """EngineError is a subclass of LLEMError."""
    assert issubclass(EngineError, LLEMError)


def test_preflight_error_inherits_llem_error():
    """PreFlightError is a subclass of LLEMError."""
    assert issubclass(PreFlightError, LLEMError)


def test_experiment_error_inherits_llem_error():
    """ExperimentError is a subclass of LLEMError."""
    assert issubclass(ExperimentError, LLEMError)


def test_study_error_inherits_llem_error():
    """StudyError is a subclass of LLEMError."""
    assert issubclass(StudyError, LLEMError)


# ---------------------------------------------------------------------------
# Catchability via base class
# ---------------------------------------------------------------------------


def test_all_errors_catchable_via_llem_error():
    """Catching LLEMError catches all 5 direct subclass instances."""
    subclasses = [
        ConfigError("config"),
        EngineError("engine"),
        PreFlightError("preflight"),
        ExperimentError("experiment"),
        StudyError("study"),
    ]
    for exc in subclasses:
        try:
            raise exc
        except LLEMError:
            pass  # expected
        else:
            pytest.fail(f"{type(exc).__name__} was not caught by LLEMError")


# ---------------------------------------------------------------------------
# Message preservation
# ---------------------------------------------------------------------------


def test_error_messages_preserved():
    """Constructing with a message preserves str(e)."""
    for cls in [ConfigError, EngineError, PreFlightError, ExperimentError, StudyError]:
        msg = f"test message for {cls.__name__}"
        exc = cls(msg)
        assert str(exc) == msg
