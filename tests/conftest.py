"""Pytest configuration and fixtures for LLM Bench tests."""

import os
import re

# Disable Rich colors in tests to ensure consistent output for assertions
# This must be set before any imports that use Rich
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"  # Additional terminal hint


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    return ansi_escape.sub("", text)
