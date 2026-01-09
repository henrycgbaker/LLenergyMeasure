"""Pytest configuration and fixtures for LLM Bench tests."""

import os

# Disable Rich colors in tests to ensure consistent output for assertions
# This must be set before any imports that use Rich
os.environ["NO_COLOR"] = "1"
