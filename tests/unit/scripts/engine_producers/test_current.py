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


# ---------------------------------------------------------------------------
# previous_pin_outputs_dir
# ---------------------------------------------------------------------------


def _fake_engine_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    engine: str,
    current: str,
    versions_with_outputs: list[str],
    versions_without_outputs: list[str] | None = None,
) -> Path:
    """Lay out a synthetic ``engine_versions/{engine}/`` tree and point the
    module's repo-root + current-version resolvers at it."""
    engine_dir = tmp_path / "engine_versions" / engine
    for v in versions_with_outputs:
        outputs = engine_dir / _current.safe_version(v) / "outputs"
        outputs.mkdir(parents=True)
        (outputs / "rules.proposed.yaml").write_text("invariants: []\n")
    for v in versions_without_outputs or []:
        (engine_dir / _current.safe_version(v)).mkdir(parents=True)
    monkeypatch.setattr(_current, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(_current, "current_version", lambda e: current)
    return engine_dir


def test_previous_pin_none_when_pin_is_only_outputs_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Today's state: the current pin is the only version dir with outputs/,
    # and the newer vendored version dirs have none. Resolver returns None
    # (the decay alarm + surface trend are a structural no-op).
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="vllm",
        current="0.7.3",
        versions_with_outputs=["0.7.3"],
        versions_without_outputs=["0.16.0", "0.19.1"],
    )
    assert _current.previous_pin_outputs_dir("vllm") is None


def test_previous_pin_picks_most_recent_prior_with_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="vllm",
        current="0.21.0",
        versions_with_outputs=["0.18.1", "0.19.1", "0.21.0"],
    )
    prev = _current.previous_pin_outputs_dir("vllm")
    assert prev is not None
    assert prev.parent.name == "v0_19_1"


def test_previous_pin_ignores_higher_versions_and_empty_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine_dir = _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="tensorrt",
        current="1.2.0",
        versions_with_outputs=["0.21.0", "1.2.0"],
        versions_without_outputs=["1.2.1"],
    )
    # An empty outputs/ on an otherwise-qualifying prior is not a candidate.
    (engine_dir / _current.safe_version("1.1.0") / "outputs").mkdir(parents=True)
    prev = _current.previous_pin_outputs_dir("tensorrt")
    assert prev is not None
    assert prev.parent.name == "v0_21_0"


def test_previous_pin_none_for_unknown_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_current, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(_current, "current_version", lambda e: "1.0.0")
    assert _current.previous_pin_outputs_dir("nonexistent") is None


# ---------------------------------------------------------------------------
# is_major_bump
# ---------------------------------------------------------------------------


def test_is_major_bump_false_when_no_prior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="vllm",
        current="0.7.3",
        versions_with_outputs=["0.7.3"],
    )
    assert _current.is_major_bump("vllm") is False


def test_is_major_bump_false_on_minor_crossing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="vllm",
        current="0.21.0",
        versions_with_outputs=["0.19.1", "0.21.0"],
    )
    assert _current.is_major_bump("vllm") is False


def test_is_major_bump_true_on_major_crossing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The live tensorrt 0.x -> 1.x case the design flags as the strongest
    # un-run cell.
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="tensorrt",
        current="1.2.1",
        versions_with_outputs=["0.21.0", "1.2.1"],
    )
    assert _current.is_major_bump("tensorrt") is True
