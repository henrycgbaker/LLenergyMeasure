"""Tests for scripts/engine_producers/_current.py pin-resolution helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.engine_producers import _current  # noqa: E402


@pytest.mark.parametrize(
    ("engine", "expected"),
    [("transformers", "4.57.3"), ("vllm", "0.7.3"), ("tensorrt", "0.21.0")],
)
def test_current_version_reads_the_pin(engine: str, expected: str) -> None:
    assert _current.current_version(engine) == expected


@pytest.mark.parametrize(
    ("engine", "tail"),
    [
        ("transformers", "engine_versions/transformers/v4_57_3/outputs"),
        ("vllm", "engine_versions/vllm/v0_7_3/outputs"),
        ("tensorrt", "engine_versions/tensorrt/v0_21_0/outputs"),
    ],
)
def test_current_outputs_dir_resolves_from_pin(engine: str, tail: str) -> None:
    outputs = _current.current_outputs_dir(engine)
    assert outputs.is_dir()
    assert str(outputs).endswith(tail)


def test_current_version_raises_on_malformed_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_current, "load_current", lambda engine: {"library": {}})
    with pytest.raises(ValueError, match=r"no string library\.current_version"):
        _current.current_version("transformers")
