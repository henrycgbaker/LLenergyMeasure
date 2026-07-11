"""Unit tests for the version-aware machinery dispatcher.

Covers:

- ``safe_version`` mangling helper.
- The auto-fallback contract: when no exact-match archive exists at the
  target version, the dispatcher resolves to the highest vendored archive
  ``<= target`` and logs the fallback to stderr.
- The no-fallback diagnostic: when no archive exists at or below the target
  with a ``producers/<producer>.py`` file, the dispatcher raises
  ``ModuleNotFoundError`` with the path the maintainer needs to create.

Per-engine ``load_producer`` resolution tests live in each engine's
vendor PR alongside the engine's archive subpackage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine_versions._dispatcher import _find_fallback_safe_version, load_producer
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
# Fallback unit tests (_find_fallback_safe_version, tmp_path fixture)
# ---------------------------------------------------------------------------


def _make_archive(root: Path, safe: str, producer: str) -> None:
    """Create ``<root>/<safe>/producers/<producer>.py`` + __init__.py chain.

    Mirrors the shape of a real engine archive enough for the dispatcher's
    filesystem check.
    """
    archive_dir = root / safe / "producers"
    archive_dir.mkdir(parents=True)
    (root / safe / "__init__.py").write_text("")
    (archive_dir / "__init__.py").write_text("")
    (archive_dir / f"{producer}.py").write_text("LANDMARKS: tuple[str, ...] = ()\n")


def test_find_fallback_picks_highest_below_target(tmp_path: Path) -> None:
    """v0_7_3 + v0_19_1 vendored; target 0.16.0 falls back to v0_7_3."""
    for safe in ("v0_7_3", "v0_19_1"):
        _make_archive(tmp_path, safe, "static_invariant_miner")
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="0.16.0",
        producer="static_invariant_miner",
    )
    assert result == "v0_7_3"


def test_find_fallback_picks_exact_when_present(tmp_path: Path) -> None:
    """When the target version is itself vendored, it is the highest <= target."""
    for safe in ("v0_7_3", "v0_19_1"):
        _make_archive(tmp_path, safe, "static_invariant_miner")
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="0.19.1",
        producer="static_invariant_miner",
    )
    assert result == "v0_19_1"


def test_find_fallback_returns_none_when_all_above(tmp_path: Path) -> None:
    """All vendored versions above target -> no candidate."""
    _make_archive(tmp_path, "v0_19_1", "static_invariant_miner")
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="0.5.0",
        producer="static_invariant_miner",
    )
    assert result is None


def test_find_fallback_skips_dirs_without_producer_file(tmp_path: Path) -> None:
    """A version dir with only machinery/ (no producers/<producer>.py) is skipped."""
    _make_archive(tmp_path, "v0_7_3", "static_invariant_miner")
    # v0_18_1 has machinery/ but no producers/<producer>.py.
    (tmp_path / "v0_18_1" / "machinery").mkdir(parents=True)
    (tmp_path / "v0_18_1" / "__init__.py").write_text("")
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="0.18.5",
        producer="static_invariant_miner",
    )
    assert result == "v0_7_3"


def test_find_fallback_skips_non_version_dirs(tmp_path: Path) -> None:
    """Dirs like ``__pycache__`` and ``outputs`` are silently skipped."""
    _make_archive(tmp_path, "v0_7_3", "static_invariant_miner")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "outputs").mkdir()
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="0.7.4",
        producer="static_invariant_miner",
    )
    assert result == "v0_7_3"


def test_find_fallback_handles_prerelease_ordering(tmp_path: Path) -> None:
    """Pre-release ordering: v0_7_3rc1 < v0_7_3 < v0_7_4."""
    for safe in ("v0_7_3rc1", "v0_7_3", "v0_7_4"):
        _make_archive(tmp_path, safe, "static_invariant_miner")
    # Target 0.7.3 picks v0_7_3 (rc1 ranks lower).
    assert (
        _find_fallback_safe_version(
            engine_root=tmp_path,
            target_version="0.7.3",
            producer="static_invariant_miner",
        )
        == "v0_7_3"
    )
    # Target 0.7.3rc1 picks v0_7_3rc1 (0.7.3 > rc1, excluded).
    assert (
        _find_fallback_safe_version(
            engine_root=tmp_path,
            target_version="0.7.3rc1",
            producer="static_invariant_miner",
        )
        == "v0_7_3rc1"
    )


def test_find_fallback_returns_none_for_missing_engine_root(tmp_path: Path) -> None:
    """Non-existent engine_root returns None; caller raises."""
    result = _find_fallback_safe_version(
        engine_root=tmp_path / "missing",
        target_version="0.7.3",
        producer="static_invariant_miner",
    )
    assert result is None


def test_find_fallback_returns_none_for_invalid_target_version(tmp_path: Path) -> None:
    """Garbage target_version returns None (no fallback selected)."""
    _make_archive(tmp_path, "v0_7_3", "static_invariant_miner")
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="not-a-version",
        producer="static_invariant_miner",
    )
    assert result is None


def test_find_fallback_filters_per_producer_kind(tmp_path: Path) -> None:
    """A dir with only ``schema_introspector.py`` isn't a fallback for ``static_invariant_miner``."""
    _make_archive(tmp_path, "v0_7_3", "schema_introspector")
    result = _find_fallback_safe_version(
        engine_root=tmp_path,
        target_version="0.7.4",
        producer="static_invariant_miner",
    )
    assert result is None


# ---------------------------------------------------------------------------
# load_producer: integration against real engine_versions/ tree
# ---------------------------------------------------------------------------


def test_load_producer_exact_match() -> None:
    """When the exact version is vendored, the dispatcher returns that one."""
    module = load_producer(engine="vllm", version="0.7.3", producer="static_invariant_miner")
    assert module.__name__ == "engine_versions.vllm.v0_7_3.producers.static_invariant_miner"


def test_load_producer_falls_back_when_above_all_vendored(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A target above all vendored versions falls back to the highest <= target."""
    module = load_producer(engine="vllm", version="999.0.0", producer="static_invariant_miner")
    assert module.__name__.startswith("engine_versions.vllm.")
    assert module.__name__.endswith(".producers.static_invariant_miner")
    # The fallback version comes from the live tree; just assert it's not v999.
    assert "v999_0_0" not in module.__name__
    # Stderr log surfaces the fallback choice.
    captured = capsys.readouterr()
    assert "no exact match" in captured.err
    assert "vllm==999.0.0" in captured.err
    assert "falling back to" in captured.err


def test_load_producer_falls_back_to_v0_7_3_for_intermediate_target(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """vllm has v0_7_3 vendored; target 0.7.4 (unvendored) falls back to v0_7_3."""
    module = load_producer(engine="vllm", version="0.7.4", producer="static_invariant_miner")
    assert module.__name__ == "engine_versions.vllm.v0_7_3.producers.static_invariant_miner"
    captured = capsys.readouterr()
    assert "falling back to engine_versions.vllm.v0_7_3.producers.static_invariant_miner" in (
        captured.err
    )


# ---------------------------------------------------------------------------
# No-fallback contract: raise loud when nothing exists at or below target
# ---------------------------------------------------------------------------


def test_below_lowest_vendored_raises_loud() -> None:
    """All vendored vllm versions are >= 0.7.3; target 0.0.0 has no candidate."""
    with pytest.raises(ModuleNotFoundError):
        load_producer(engine="vllm", version="0.0.0", producer="static_invariant_miner")


def test_unknown_engine_raises_loud() -> None:
    """Non-existent engine root has no vendored archives; raises with create-this-file path."""
    with pytest.raises(ModuleNotFoundError):
        load_producer(
            engine="not-a-real-engine",
            version="0.7.3",
            producer="static_invariant_miner",
        )


def test_unknown_producer_raises_loud() -> None:
    """A typo in the producer kind matches no archives; raises."""
    with pytest.raises(ModuleNotFoundError):
        # ``introspector`` is the user-facing probe name; the archive uses
        # the long form ``schema_introspector``.
        load_producer(engine="vllm", version="0.7.3", producer="introspector")  # type: ignore[arg-type]


def test_dispatcher_error_message_names_file_to_create() -> None:
    """No-fallback diagnostic names the path the maintainer needs to create.

    This is the primary signal a maintainer sees when neither an exact-match
    archive nor any earlier vendored archive can serve the bumped version.
    The error message must point at the file to create so the next chunk PR
    is unblocked without spelunking.
    """
    with pytest.raises(ModuleNotFoundError) as exc_info:
        load_producer(engine="vllm", version="0.0.0", producer="static_invariant_miner")
    msg = str(exc_info.value)
    assert "vllm" in msg
    assert "0.0.0" in msg
    assert "engine_versions/vllm/v0_0_0/producers/static_invariant_miner.py" in msg
    assert "LANDMARKS" in msg
