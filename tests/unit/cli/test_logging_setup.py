"""Tests for _setup_logging handler attachment.

Regression: the package installs a NullHandler at import time. The setup used
to treat any existing handler (including that placeholder) as "already
configured" and never attach the real stream handler, so WARNING-level messages
never reached the user.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from llenergymeasure.cli import _setup_logging


@pytest.fixture
def restore_llem_logger() -> Iterator[None]:
    """Snapshot and restore the ``llenergymeasure`` logger's handlers/level."""
    log = logging.getLogger("llenergymeasure")
    saved_handlers = list(log.handlers)
    saved_level = log.level
    try:
        yield
    finally:
        log.handlers = saved_handlers
        log.setLevel(saved_level)


def test_setup_logging_attaches_stream_handler_over_nullhandler(
    restore_llem_logger: None,
) -> None:
    """A pre-existing NullHandler must not suppress the real stream handler."""
    log = logging.getLogger("llenergymeasure")
    # Reproduce the import-time state: only a NullHandler present.
    log.handlers = [logging.NullHandler()]

    _setup_logging(0)

    stream_handlers = [
        h
        for h in log.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
    ]
    assert stream_handlers, "a real StreamHandler must be attached despite the NullHandler"
    assert log.level == logging.WARNING


def test_setup_logging_default_surfaces_warnings(
    restore_llem_logger: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """With the NullHandler present, a child-logger WARNING still reaches stderr."""
    log = logging.getLogger("llenergymeasure")
    log.handlers = [logging.NullHandler()]

    _setup_logging(0)

    logging.getLogger("llenergymeasure.study.runner").warning("SENTINEL-WARNING")

    captured = capsys.readouterr()
    assert "SENTINEL-WARNING" in captured.err


def test_setup_logging_no_duplicate_stream_handlers(restore_llem_logger: None) -> None:
    """Repeated calls must not stack duplicate stream handlers."""
    log = logging.getLogger("llenergymeasure")
    log.handlers = [logging.NullHandler()]

    _setup_logging(0)
    _setup_logging(0)

    stream_handlers = [
        h
        for h in log.handlers
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
    ]
    assert len(stream_handlers) == 1, "repeated setup must not add duplicate handlers"
