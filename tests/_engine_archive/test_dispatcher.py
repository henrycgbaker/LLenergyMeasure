"""Unit tests for the version-aware machinery dispatcher.

Engine-agnostic tests only: the ``safe_version`` mangling helper, the
no-fallback contract (unknown engines / versions / producers raise
``ModuleNotFoundError``), and the actionable error-message contract that
fires when a Renovate-driven version bump outpaces the per-version vendoring.

Per-engine ``load_producer`` resolution tests live in each engine's
vendor PR alongside the engine's archive subpackage.
"""

from __future__ import annotations

import pytest

from engine_versions._dispatcher import load_producer
from scripts.engine_producers._current import safe_version

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
# No-fallback contract
# ---------------------------------------------------------------------------


def test_unknown_version_raises_loud() -> None:
    """Plan's no-fallback rule: an unknown version surfaces ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        load_producer(engine="vllm", version="999.999.999", producer="static_invariant_miner")


def test_unknown_engine_raises_loud() -> None:
    with pytest.raises(ModuleNotFoundError):
        load_producer(
            engine="not-a-real-engine", version="0.7.3", producer="static_invariant_miner"
        )


def test_unknown_producer_raises_loud() -> None:
    """Dispatcher accepts only the three SSOT producer kinds."""
    with pytest.raises(ModuleNotFoundError):
        # ``introspector`` is the user-facing name; the SSOT key is ``discovery``.
        load_producer(engine="vllm", version="0.7.3", producer="introspector")  # type: ignore[arg-type]


def test_dispatcher_error_message_names_file_to_create() -> None:
    """Missing-producer diagnostic must name the exact file path.

    This is the primary signal a maintainer sees when a Renovate-driven SSOT
    bump outpaces the per-version vendoring. The error must point them at
    the file to create so the next chunk PR is unblocked without spelunking.
    """
    with pytest.raises(ModuleNotFoundError) as exc_info:
        load_producer(engine="vllm", version="999.0.0", producer="static_invariant_miner")
    msg = str(exc_info.value)
    assert "vllm" in msg
    assert "999.0.0" in msg
    assert "engine_versions/vllm/v999_0_0/producers/static_invariant_miner.py" in msg
    assert "LANDMARKS" in msg
