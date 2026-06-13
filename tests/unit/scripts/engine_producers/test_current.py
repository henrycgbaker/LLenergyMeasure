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
    [("transformers", "5.7.0"), ("vllm", "0.7.3"), ("tensorrt", "0.21.0")],
)
def test_current_version_reads_the_pin(engine: str, expected: str) -> None:
    assert _current.current_version(engine) == expected


@pytest.mark.parametrize(
    ("engine", "tail"),
    [
        ("transformers", "engine_versions/transformers/v5_7_0/outputs"),
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
# carry_forward_inputs
# ---------------------------------------------------------------------------


def _carry_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    engine: str,
    current: str,
    prior: str | None,
    prior_files: dict[str, str] | None = None,
    current_files: dict[str, str] | None = None,
) -> Path:
    """Lay out a synthetic engine tree for carry-forward tests.

    ``prior_files`` populates the prior pin's outputs/; ``current_files``
    pre-seeds the current pin's outputs/ (to test the no-clobber path).
    Wires the module's repo-root + current-version resolvers at the tree.
    Returns the current pin's outputs/ directory.
    """
    engine_dir = tmp_path / "engine_versions" / engine
    if prior is not None:
        prior_out = engine_dir / _current.safe_version(prior) / "outputs"
        prior_out.mkdir(parents=True)
        # A prior pin must carry rules to qualify as previous_pin_outputs_dir.
        (prior_out / "rules.proposed.yaml").write_text("invariants: []\n")
        for name, body in (prior_files or {}).items():
            (prior_out / name).write_text(body, encoding="utf-8")
    current_out = engine_dir / _current.safe_version(current) / "outputs"
    current_out.mkdir(parents=True)
    for name, body in (current_files or {}).items():
        (current_out / name).write_text(body, encoding="utf-8")
    monkeypatch.setattr(_current, "_find_repo_root", lambda start: tmp_path)
    monkeypatch.setattr(_current, "current_version", lambda e: current)
    return current_out


def test_carry_forward_seeds_curated_and_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Fresh new pin, prior carries both maintainer inputs -> both forwarded.
    current_out = _carry_tree(
        monkeypatch,
        tmp_path,
        engine="transformers",
        current="5.8.1",
        prior="5.7.0",
        prior_files={
            "curated.yaml": "engine: transformers\n",
            "overlay.yaml": "narrowings: {temperature: {le: 2.0}}\n",
        },
    )
    copied = _current.carry_forward_inputs("transformers")
    assert set(copied) == {"curated.yaml", "overlay.yaml"}
    assert (current_out / "curated.yaml").read_text() == "engine: transformers\n"
    assert "temperature" in (current_out / "overlay.yaml").read_text()


def test_carry_forward_curated_only_when_prior_has_no_overlay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Prior carries curated but no overlay -> curated forwarded, no overlay,
    # no error (overlay is optional).
    current_out = _carry_tree(
        monkeypatch,
        tmp_path,
        engine="transformers",
        current="5.0.0",
        prior="4.57.3",
        prior_files={"curated.yaml": "engine: transformers\n"},
    )
    copied = _current.carry_forward_inputs("transformers")
    assert copied == ["curated.yaml"]
    assert (current_out / "curated.yaml").exists()
    assert not (current_out / "overlay.yaml").exists()


def test_carry_forward_is_noop_when_current_already_has_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The current pin already carries the maintainer files (e.g. a re-run, or a
    # maintainer edit). Carry must NOT clobber them.
    current_out = _carry_tree(
        monkeypatch,
        tmp_path,
        engine="transformers",
        current="5.8.1",
        prior="5.7.0",
        prior_files={
            "curated.yaml": "engine: transformers\nfrom: prior\n",
            "overlay.yaml": "from: prior\n",
        },
        current_files={
            "curated.yaml": "engine: transformers\nfrom: maintainer-edit\n",
            "overlay.yaml": "from: maintainer-edit\n",
        },
    )
    copied = _current.carry_forward_inputs("transformers")
    assert copied == []
    assert "maintainer-edit" in (current_out / "curated.yaml").read_text()
    assert "maintainer-edit" in (current_out / "overlay.yaml").read_text()


def test_carry_forward_raises_when_no_curated_anywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Brand-new engine: neither the current pin nor any prior carries a
    # curated.yaml. That needs a bootstrap, out of scope for the per-bump loop.
    _carry_tree(
        monkeypatch,
        tmp_path,
        engine="newengine",
        current="1.0.0",
        prior=None,
    )
    with pytest.raises(FileNotFoundError, match=r"required input curated\.yaml missing"):
        _current.carry_forward_inputs("newengine")


def test_carry_forward_raises_when_prior_lacks_curated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A prior pin exists but somehow has no curated.yaml (corrupt window): the
    # required input cannot be forwarded -> hard error, not a silent gap.
    _carry_tree(
        monkeypatch,
        tmp_path,
        engine="transformers",
        current="5.8.1",
        prior="5.7.0",
        prior_files={"overlay.yaml": "narrowings: {}\n"},
    )
    with pytest.raises(FileNotFoundError, match=r"required input curated\.yaml missing"):
        _current.carry_forward_inputs("transformers")


def test_carry_forward_main_emits_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _carry_tree(
        monkeypatch,
        tmp_path,
        engine="transformers",
        current="5.8.1",
        prior="5.7.0",
        prior_files={"curated.yaml": "engine: transformers\n"},
    )
    assert _current._main(["--engine", "transformers", "--carry-forward"]) == 0
    out = capsys.readouterr().out
    assert "seeded curated.yaml" in out
    # A second pass is a no-op.
    assert _current._main(["--engine", "transformers", "--carry-forward"]) == 0
    assert "nothing carried" in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# _main (GITHUB_OUTPUT emission)
# ---------------------------------------------------------------------------


def test_main_emits_repo_relative_prev_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The decay-alarm re-gate consumes prev_outputs INSIDE the engine
    # container, where the checkout mounts at /repo - an absolute runner
    # path dangles there. The emitted form must be repo-relative (it also
    # resolves host-side, where consumers run from the checkout root).
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="vllm",
        current="0.21.0",
        versions_with_outputs=["0.19.1", "0.21.0"],
    )
    assert _current._main(["--engine", "vllm"]) == 0
    out = capsys.readouterr().out
    assert "prev_outputs=engine_versions/vllm/v0_19_1/outputs\n" in out
    assert "major_bump=false\n" in out


def test_main_emits_empty_prev_outputs_when_no_prior(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _fake_engine_tree(
        monkeypatch,
        tmp_path,
        engine="vllm",
        current="0.7.3",
        versions_with_outputs=["0.7.3"],
    )
    assert _current._main(["--engine", "vllm"]) == 0
    out = capsys.readouterr().out
    assert "prev_outputs=\n" in out
