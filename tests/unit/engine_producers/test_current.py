"""Unit tests for ``scripts/engine_producers/_current.py``.

Covers the lookup helpers used across the corpus pipeline:

- :func:`current_path` / :func:`load_current` / :func:`safe_version` -
  small helpers exercised end-to-end via the live ``engine_versions/`` tree.
- :func:`current_outputs_dir` - the per-version archive path that
  ``regen_engine_corpus.py`` (and future Move 3 callers) compose into the
  SSOT-to-shadow sync. Exercised with monkeypatching so a synthetic
  ``current.yaml`` can stand in for the real one.

``safe_version`` already has fast unit coverage in
``tests/_engine_archive/test_dispatcher.py``; the tests here cover the
remaining helpers without duplicating that.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Make scripts/ importable for direct module access in tests.
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
        assert path.name == "current.yaml"
        assert path.parent.name == "vllm"
        assert path.parent.parent.name == "engine_versions"

    def test_live_current_yaml_exists_for_each_engine(self) -> None:
        for engine in ("vllm", "tensorrt", "transformers"):
            assert _current.current_path(engine).is_file()


class TestLoadCurrent:
    def test_loads_real_engine_current_yaml_to_dict(self) -> None:
        data = _current.load_current("vllm")
        assert isinstance(data, dict)
        assert data["engine"] == "vllm"
        assert isinstance(data["library"], dict)
        assert data["library"]["pep503_name"] == "vllm"

    def test_raises_on_missing_yaml(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Point ``current_path`` at a non-existent file to verify the error.
        monkeypatch.setattr(_current, "current_path", lambda engine: tmp_path / "nope.yaml")
        with pytest.raises(FileNotFoundError):
            _current.load_current("vllm")

    def test_raises_on_non_mapping_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bad = tmp_path / "current.yaml"
        bad.write_text("- this\n- is\n- a\n- list\n")
        monkeypatch.setattr(_current, "current_path", lambda engine: bad)
        with pytest.raises(ValueError, match="did not parse to a mapping"):
            _current.load_current("vllm")


# ---------------------------------------------------------------------------
# current_outputs_dir
# ---------------------------------------------------------------------------


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
        # Synthetic current.yaml at version 4.57.3 -> v4_57_3/outputs/.
        synth = tmp_path / "current.yaml"
        synth.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "engine": "transformers",
                    "library": {
                        "pep503_name": "transformers",
                        "current_version": "4.57.3",
                    },
                }
            )
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        path = _current.current_outputs_dir("transformers")
        # ``_find_repo_root`` walks up from _current.py, not from the
        # synthetic dir, so the returned path is rooted at the live repo;
        # we assert the version-safe suffix shape only.
        assert path.parts[-3:] == ("transformers", "v4_57_3", "outputs")

    def test_raises_when_library_block_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth = tmp_path / "current.yaml"
        synth.write_text(yaml.safe_dump({"schema_version": 1, "engine": "vllm"}))
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        with pytest.raises(ValueError, match="missing required 'library' mapping"):
            _current.current_outputs_dir("vllm")

    def test_raises_when_current_version_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        synth = tmp_path / "current.yaml"
        synth.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "engine": "vllm",
                    "library": {"pep503_name": "vllm"},
                }
            )
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        with pytest.raises(ValueError, match=r"missing 'library\.current_version'"):
            _current.current_outputs_dir("vllm")

    def test_raises_when_current_version_non_string(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # YAML happily parses bare numbers; the guard must reject them so
        # ``safe_version`` never receives a non-string and accidentally
        # produces ``v473`` for ``4.73``.
        synth = tmp_path / "current.yaml"
        synth.write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "engine": "vllm",
                    "library": {"pep503_name": "vllm", "current_version": 4.73},
                }
            )
        )
        monkeypatch.setattr(_current, "current_path", lambda engine: synth)
        with pytest.raises(ValueError, match=r"missing 'library\.current_version'"):
            _current.current_outputs_dir("vllm")
