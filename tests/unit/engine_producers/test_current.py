"""Unit tests for ``scripts/engine_producers/_current.py``.

Covers the lookup helpers used across the corpus pipeline:

- :func:`current_path` / :func:`load_current` / :func:`safe_version` -
  small helpers exercised end-to-end via the live ``engine_versions/`` tree.
- :func:`current_outputs_dir` - the per-version archive path that
  ``regen_engine_corpus.py`` (and future Move 3 callers) compose into the
  SSOT-to-shadow sync. Exercised with monkeypatching so a synthetic
  ``current.toml`` can stand in for the real one.

``safe_version`` already has fast unit coverage in
``tests/_engine_archive/test_dispatcher.py``; the tests here cover the
remaining helpers without duplicating that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import _current  # noqa: E402

# ---------------------------------------------------------------------------
# current_path / load_current (sanity against the live tree)
# ---------------------------------------------------------------------------


class TestCurrentPath:
    def test_returns_absolute_path_under_engine_versions(self) -> None:
        path = _current.current_path("vllm")
        assert path.is_absolute()
        assert path.name == "current.toml"
        assert path.parent.name == "vllm"
        assert path.parent.parent.name == "engine_versions"

    def test_live_current_toml_exists_for_each_engine(self) -> None:
        for engine in ("vllm", "tensorrt", "transformers"):
            assert _current.current_path(engine).is_file()


class TestLoadCurrent:
    def test_loads_real_engine_current_toml_to_dict(self) -> None:
        data = _current.load_current("vllm")
        assert isinstance(data, dict)
        assert data["engine"] == "vllm"
        assert isinstance(data["library"], dict)
        assert data["library"]["pep503_name"] == "vllm"

    def test_raises_on_missing_toml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_current, "current_path", lambda engine: tmp_path / "nope.toml")
        with pytest.raises(FileNotFoundError):
            _current.load_current("vllm")


# ---------------------------------------------------------------------------
# current_outputs_dir
# ---------------------------------------------------------------------------


def _write_synthetic_toml(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")


class TestCurrentOutputsDir:
    def test_resolves_live_engine_to_engine_versions_outputs(self) -> None:
        # Live tree: vllm pins 0.7.3 -> v0_7_3/outputs/.
        path = _current.current_outputs_dir("vllm")
        assert path.is_absolute()
        assert path.name == "outputs"
        assert path.parent.name.startswith("v")  # safe_version form
        assert path.parent.parent.name == "vllm"
        assert path.parent.parent.parent.name == "engine_versions"
        # Sanity: the resolved version matches what ``load_current`` returns.
        data = _current.load_current("vllm")
        expected_safe = _current.safe_version(data["library"]["current_version"])
        assert path.parent.name == expected_safe

    def test_uses_safe_version_mangling(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth = tmp_path / "current.toml"
        _write_synthetic_toml(
            synth,
            "schema_version = 1\n"
            'engine = "transformers"\n'
            "[library]\n"
            'pep503_name = "transformers"\n'
            'current_version = "4.57.3"\n',
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        path = _current.current_outputs_dir("transformers")
        # ``current_outputs_dir`` composes the safe-version + "outputs"
        # under ``current_path(engine).parent``; with current_path
        # monkeypatched to a tmp file, only the trailing segments are
        # under test (the engine subdir is implicit in current_path).
        assert path.parts[-2:] == ("v4_57_3", "outputs")

    def test_raises_when_library_block_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth = tmp_path / "current.toml"
        _write_synthetic_toml(
            synth,
            'schema_version = 1\nengine = "vllm"\n',
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        with pytest.raises(ValueError, match=r"'library' must be a mapping"):
            _current.current_outputs_dir("vllm")

    def test_raises_when_current_version_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth = tmp_path / "current.toml"
        _write_synthetic_toml(
            synth,
            'schema_version = 1\nengine = "vllm"\n[library]\npep503_name = "vllm"\n',
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        with pytest.raises(ValueError, match=r"'library\.current_version'"):
            _current.current_outputs_dir("vllm")

    def test_raises_when_current_version_non_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # TOML happily parses bare numbers as floats; the guard must reject
        # them so ``safe_version`` never receives a non-string and
        # accidentally produces ``v473`` for ``4.73``.
        synth = tmp_path / "current.toml"
        _write_synthetic_toml(
            synth,
            "schema_version = 1\n"
            'engine = "vllm"\n'
            "[library]\n"
            'pep503_name = "vllm"\n'
            "current_version = 4.73\n",
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        with pytest.raises(ValueError, match=r"'library\.current_version'"):
            _current.current_outputs_dir("vllm")
