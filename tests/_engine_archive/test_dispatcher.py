"""Unit tests for the version-aware machinery dispatcher.

Exercises ``load_machinery`` for every (engine, version, producer) cell
populated by PR-0; verifies the LANDMARKS contract; verifies the
no-fallback rule (unknown version raises ``ModuleNotFoundError``).
"""

from __future__ import annotations

import pytest

from llenergymeasure._engine_archive._dispatcher import load_machinery
from scripts.engine_miners._ssot import safe_version

# ---------------------------------------------------------------------------
# safe_version mangling
# ---------------------------------------------------------------------------


def test_safe_version_dotted_to_underscore() -> None:
    assert safe_version("0.7.3") == "v0_7_3"
    assert safe_version("4.57.3") == "v4_57_3"
    assert safe_version("1.2.0") == "v1_2_0"


def test_safe_version_handles_hyphenated_prerelease() -> None:
    # Hyphens (e.g. PEP-440 dev / pre tags written as 1.2.0-rc1) are also
    # mapped to underscores; the safe form remains a legal identifier.
    assert safe_version("1.2.0-rc1") == "v1_2_0_rc1"


def test_safe_version_rejects_non_alphanumeric() -> None:
    with pytest.raises(ValueError):
        safe_version("0.7.3+cuda12")  # build metadata not supported


# ---------------------------------------------------------------------------
# Dispatcher resolution
# ---------------------------------------------------------------------------


_SUPPORTED_CELLS = [
    ("vllm", "0.7.3", "static"),
    ("vllm", "0.7.3", "discovery"),
    ("tensorrt", "0.21.0", "static"),
    ("tensorrt", "0.21.0", "discovery"),
    ("transformers", "4.57.3", "static"),
    ("transformers", "4.57.3", "discovery"),
]


@pytest.mark.parametrize("engine,version,producer", _SUPPORTED_CELLS)
def test_load_machinery_returns_module_with_landmarks(
    engine: str, version: str, producer: str
) -> None:
    """Every (engine, version, producer) cell in PR-0 exposes LANDMARKS."""
    machinery = load_machinery(engine=engine, version=version, producer=producer)
    landmarks = machinery.LANDMARKS
    assert isinstance(landmarks, tuple), (
        f"{engine}/{version}/{producer}: LANDMARKS must be a tuple, got {type(landmarks).__name__}"
    )
    assert all(isinstance(s, str) for s in landmarks), (
        f"{engine}/{version}/{producer}: every LANDMARKS entry must be a str"
    )
    assert len(landmarks) > 0, f"{engine}/{version}/{producer}: LANDMARKS must be non-empty"
    assert all("." in s for s in landmarks), (
        f"{engine}/{version}/{producer}: LANDMARKS entries must be dotted attribute paths"
    )


# ---------------------------------------------------------------------------
# No-fallback contract
# ---------------------------------------------------------------------------


def test_unknown_version_raises_loud() -> None:
    """Plan's no-fallback rule: an unknown version surfaces ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        load_machinery(engine="vllm", version="999.999.999", producer="static")


def test_unknown_engine_raises_loud() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_machinery(engine="not-a-real-engine", version="0.7.3", producer="static")


def test_unknown_producer_raises_loud() -> None:
    """Dispatcher accepts only the three SSOT producer kinds."""
    with pytest.raises(ModuleNotFoundError):
        # ``introspector`` is the user-facing name; the SSOT key is ``discovery``.
        load_machinery(engine="vllm", version="0.7.3", producer="introspector")  # type: ignore[arg-type]
