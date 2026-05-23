"""Unit tests for ``scripts/engine_producers/regen_engine_corpus.py``.

The script copies per-version corpus artefacts from
``engine_versions/<engine>/v<safe>/outputs/`` (SSOT) into
``src/llenergymeasure/engines/<engine>/`` (data shadow the loader reads).
Two modes:

- ``--check``: exit 1 with a unified diff per drifted file. No writes.
- ``--write`` (default): ``shutil.copy2`` each source onto its destination.

Tests use ``tmp_path`` and monkeypatch the path-derivation helpers so the
real ``engine_versions/`` and ``src/llenergymeasure/engines/`` trees stay
untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make scripts/ importable for direct module access in tests.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.engine_producers import regen_engine_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: spin up a synthetic SSOT / shadow tree per test
# ---------------------------------------------------------------------------


_PROPOSED = "schema_version: 1.0.0\nengine: {engine}\ninvariants: []\n"
_VALIDATED = "schema_version: 1.0.0\nengine: {engine}\ncases: []\n"
_DISCOVERED = '{{"schema_version": "1.0.0", "engine": "{engine}"}}\n'


def _write_outputs(outputs_dir: Path, engine: str) -> None:
    """Populate ``outputs_dir`` with the three SSOT artefacts."""
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (outputs_dir / "invariants.proposed.yaml").write_text(_PROPOSED.format(engine=engine))
    (outputs_dir / "invariants.validated.yaml").write_text(_VALIDATED.format(engine=engine))
    (outputs_dir / "schema.discovered.json").write_text(_DISCOVERED.format(engine=engine))


def _install_engine(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    engines: tuple[str, ...] = ("vllm",),
    populate_shadow: bool = False,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Wire monkeypatched per-engine paths and return ``(outputs, shadow)`` maps.

    Each engine gets its own SSOT outputs directory (always populated) and a
    shadow directory (populated only when ``populate_shadow`` is True).
    """
    outputs_map: dict[str, Path] = {}
    shadow_map: dict[str, Path] = {}
    for engine in engines:
        outputs_map[engine] = tmp_path / "engine_versions" / engine / "v0_0_0" / "outputs"
        shadow_map[engine] = tmp_path / "src" / "llenergymeasure" / "engines" / engine
        _write_outputs(outputs_map[engine], engine)
        if populate_shadow:
            _write_outputs(shadow_map[engine], engine)

    monkeypatch.setattr(
        regen_engine_corpus, "current_outputs_dir", lambda engine: outputs_map[engine]
    )
    monkeypatch.setattr(regen_engine_corpus, "_shadow_dir", lambda engine: shadow_map[engine])
    monkeypatch.setattr(regen_engine_corpus, "ENGINES", tuple(engines))
    return outputs_map, shadow_map


# ---------------------------------------------------------------------------
# --check mode
# ---------------------------------------------------------------------------


class TestCheckMode:
    def test_exits_zero_when_shadow_matches_ssot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=True)
        rc = regen_engine_corpus.main(["--check"])
        assert rc == 0

    def test_exits_one_with_diff_on_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _outputs, shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=True
        )
        # Hand-edit the shadow to introduce drift in one file.
        (shadow["vllm"] / "invariants.proposed.yaml").write_text(
            "schema_version: 1.0.0\nengine: vllm\ninvariants: [DRIFTED]\n"
        )
        rc = regen_engine_corpus.main(["--check"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "vllm/invariants.proposed.yaml drift" in captured.err
        assert "DRIFTED" in captured.err
        # Resolution hint is present.
        assert "regen_engine_corpus.py --write" in captured.err

    def test_exits_one_when_shadow_missing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Shadow dir not populated -> every file shows as drift (dst missing).
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False)
        rc = regen_engine_corpus.main(["--check"])
        assert rc == 1
        captured = capsys.readouterr()
        # All three files surface as drift entries.
        assert captured.err.count("vllm/") >= 3

    def test_writes_nothing_in_check_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # populate_shadow=False -> shadow dir doesn't exist before --check.
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False)
        shadow_dir = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
        assert not shadow_dir.exists()
        regen_engine_corpus.main(["--check"])
        # --check must not have created the shadow dir.
        assert not shadow_dir.exists()


# ---------------------------------------------------------------------------
# --write mode
# ---------------------------------------------------------------------------


class TestWriteMode:
    def test_default_mode_is_write(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # No flag -> writes. Shadow starts empty; after main() it has all three.
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False)
        rc = regen_engine_corpus.main([])
        assert rc == 0
        shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
        assert (shadow / "invariants.proposed.yaml").is_file()
        assert (shadow / "invariants.validated.yaml").is_file()
        assert (shadow / "schema.discovered.json").is_file()

    def test_explicit_write_flag_works(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False)
        rc = regen_engine_corpus.main(["--write"])
        assert rc == 0
        shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
        assert (shadow / "invariants.proposed.yaml").read_text() == _PROPOSED.format(engine="vllm")

    def test_write_overwrites_existing_shadow(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _outputs, shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=True
        )
        # Hand-edit the shadow; --write should restore the SSOT byte content.
        (shadow["vllm"] / "invariants.proposed.yaml").write_text("DRIFTED")
        rc = regen_engine_corpus.main(["--write"])
        assert rc == 0
        assert (shadow["vllm"] / "invariants.proposed.yaml").read_text() == _PROPOSED.format(
            engine="vllm"
        )

    def test_write_creates_shadow_dir_if_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Shadow dir doesn't exist (e.g. a freshly-vendored engine).
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False)
        shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
        assert not shadow.exists()
        regen_engine_corpus.main(["--write"])
        assert shadow.is_dir()

    def test_write_preserves_mtime_via_copy2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs, _shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False
        )
        # Set a known mtime on the SSOT file.
        import os

        target = outputs["vllm"] / "invariants.proposed.yaml"
        target_mtime = 1_700_000_000  # arbitrary fixed epoch
        os.utime(target, (target_mtime, target_mtime))
        regen_engine_corpus.main(["--write"])
        shadow_file = (
            tmp_path / "src" / "llenergymeasure" / "engines" / "vllm" / "invariants.proposed.yaml"
        )
        # ``copy2`` preserves mtime to the second; allow 1s slop for FS rounding.
        assert abs(shadow_file.stat().st_mtime - target_mtime) < 2


# ---------------------------------------------------------------------------
# Per-engine iteration
# ---------------------------------------------------------------------------


class TestPerEngineIteration:
    def test_iterates_all_configured_engines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Three engines, all populated -> --write fills all three shadow dirs.
        _install_engine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            engines=("vllm", "tensorrt", "transformers"),
            populate_shadow=False,
        )
        rc = regen_engine_corpus.main(["--write"])
        assert rc == 0
        for engine in ("vllm", "tensorrt", "transformers"):
            shadow = tmp_path / "src" / "llenergymeasure" / "engines" / engine
            assert (shadow / "invariants.proposed.yaml").is_file()
            assert (shadow / "invariants.validated.yaml").is_file()
            assert (shadow / "schema.discovered.json").is_file()

    def test_check_accumulates_drift_across_engines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _outputs, shadow = _install_engine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            engines=("vllm", "tensorrt"),
            populate_shadow=True,
        )
        # Drift in two different engines surfaces both in the report.
        (shadow["vllm"] / "invariants.proposed.yaml").write_text("DRIFTED-vllm")
        (shadow["tensorrt"] / "schema.discovered.json").write_text('{"DRIFTED-tensorrt": true}')
        rc = regen_engine_corpus.main(["--check"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "vllm/invariants.proposed.yaml drift" in captured.err
        assert "tensorrt/schema.discovered.json drift" in captured.err

    def test_first_engine_clean_still_reports_second_engine_drift(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Regression guard: --check must not short-circuit on the first clean
        # engine; it must surface drift in any engine.
        _outputs, shadow = _install_engine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            engines=("vllm", "tensorrt", "transformers"),
            populate_shadow=True,
        )
        (shadow["transformers"] / "invariants.validated.yaml").write_text("DRIFTED")
        rc = regen_engine_corpus.main(["--check"])
        assert rc == 1
        assert "transformers/invariants.validated.yaml drift" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Error handling: missing source files
# ---------------------------------------------------------------------------


class TestMissingSource:
    def test_missing_source_file_raises_with_clear_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs, _shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False
        )
        # Delete one SSOT artefact to simulate a freshly-renovated current.yaml
        # whose cells workflow hasn't populated outputs/ yet.
        (outputs["vllm"] / "schema.discovered.json").unlink()
        with pytest.raises(FileNotFoundError) as exc_info:
            regen_engine_corpus.main(["--check"])
        message = str(exc_info.value)
        assert "vllm" in message
        assert "schema.discovered.json" in message
        assert "current.yaml" in message  # Resolution hint mentions cause.

    def test_missing_source_in_write_mode_also_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        outputs, _shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False
        )
        (outputs["vllm"] / "invariants.proposed.yaml").unlink()
        with pytest.raises(FileNotFoundError):
            regen_engine_corpus.main(["--write"])


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------


class TestCliFlags:
    def test_check_and_write_mutually_exclusive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # argparse exits the process with code 2 on bad usage.
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            regen_engine_corpus.main(["--check", "--write"])
        assert exc_info.value.code == 2
