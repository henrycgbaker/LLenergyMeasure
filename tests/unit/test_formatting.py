"""Tests for shared formatting helpers."""

from __future__ import annotations

from llenergymeasure.utils.formatting import format_bytes


def test_format_bytes_scales_binary_units():
    assert format_bytes(0) == "0 B"
    assert format_bytes(512) == "512 B"
    assert format_bytes(1536) == "1.5 KB"
    assert format_bytes(1048576) == "1.0 MB"
    assert format_bytes(3 * 1024**3) == "3.0 GB"
    assert format_bytes(2 * 1024**4) == "2.0 TB"
