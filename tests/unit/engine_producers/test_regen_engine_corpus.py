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
    def test_default_mode_is_check_not_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # F#9: bare invocation must NOT mutate; it must exit 1 when drift
        # exists (here: empty shadow vs populated SSOT). Unix
        # formatter/linter convention - explicit --write to mutate.
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False)
        rc = regen_engine_corpus.main([])
        assert rc == 1
        shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
        # Crucially: no files written - bare invocation is dry-run.
        assert not (shadow / "invariants.proposed.yaml").exists()

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
    def test_missing_source_file_skips_with_clear_message(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # F#5/F#8: a missing source file is tolerated, not fatal. The
        # Renovate-pre-vendor window (current.toml bumped, cells haven't
        # produced this version's outputs/ yet) is legitimate.
        outputs, _shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=True
        )
        (outputs["vllm"] / "schema.discovered.json").unlink()
        rc = regen_engine_corpus.main(["--check"])
        # The two remaining files match; the missing one is informational.
        assert rc == 0
        err = capsys.readouterr().err
        assert "skipped" in err
        assert "vllm/schema.discovered.json" in err

    def test_missing_outputs_dir_skips_in_write_mode(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # When the entire v<safe>/outputs/ directory is missing (freshly
        # bumped current.toml whose vendor PR hasn't landed), --write
        # skips with a note instead of crashing.
        outputs, _shadow = _install_engine(
            tmp_path=tmp_path, monkeypatch=monkeypatch, populate_shadow=False
        )
        import shutil as _sh

        _sh.rmtree(outputs["vllm"])
        rc = regen_engine_corpus.main(["--write"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "outputs directory not present" in err


# ---------------------------------------------------------------------------
# --engine filter (F#10)
# ---------------------------------------------------------------------------


class TestEngineFilter:
    def test_engine_filter_restricts_run_to_named_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All three engines populated; --engine transformers must only
        # write the transformers shadow.
        _install_engine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            engines=("vllm", "tensorrt", "transformers"),
            populate_shadow=False,
        )
        rc = regen_engine_corpus.main(["--engine", "transformers", "--write"])
        assert rc == 0
        transformers_shadow = (
            tmp_path / "src" / "llenergymeasure" / "engines" / "transformers"
        )
        vllm_shadow = tmp_path / "src" / "llenergymeasure" / "engines" / "vllm"
        assert (transformers_shadow / "invariants.proposed.yaml").is_file()
        assert not (vllm_shadow / "invariants.proposed.yaml").exists()

    def test_engine_filter_repeatable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two of three engines; tensorrt left out.
        _install_engine(
            tmp_path=tmp_path,
            monkeypatch=monkeypatch,
            engines=("vllm", "tensorrt", "transformers"),
            populate_shadow=False,
        )
        rc = regen_engine_corpus.main(
            ["--engine", "vllm", "--engine", "transformers", "--write"]
        )
        assert rc == 0
        for name in ("vllm", "transformers"):
            assert (
                tmp_path / "src" / "llenergymeasure" / "engines" / name / "invariants.proposed.yaml"
            ).is_file()
        assert not (
            tmp_path / "src" / "llenergymeasure" / "engines" / "tensorrt" / "invariants.proposed.yaml"
        ).exists()

    def test_engine_filter_rejects_unknown_engine(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # argparse rejects unknown engines with exit code 2.
        _install_engine(tmp_path=tmp_path, monkeypatch=monkeypatch)
        with pytest.raises(SystemExit) as exc_info:
            regen_engine_corpus.main(["--engine", "bogus"])
        assert exc_info.value.code == 2


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
