"""Unit tests for cli/diagnose_cmd.py - Typer smoke tests.

No live model and no docker: the heavy paths (``OllamaDiagnoseModel`` and
``ContainerGateRunner``) are patched to inert stubs, and the api diagnose
functions are patched to return a canned :class:`DiagnoseResult`. We assert
argument wiring, mode dispatch, the residual-empty short-circuit, and the
gate-confirmed-only emission to ``--out``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

import llenergymeasure.cli.diagnose_cmd as diag_cmd
from llenergymeasure.api.diagnose import DiagnoseResult
from llenergymeasure.cli import app

runner = CliRunner()
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.fixture(autouse=True)
def _inert_live_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the GPU/docker constructors so the command never touches them."""
    import llenergymeasure.api.diagnose as api_diag

    monkeypatch.setattr(api_diag, "OllamaDiagnoseModel", lambda **kw: object())
    monkeypatch.setattr(api_diag, "ContainerGateRunner", lambda **kw: object())


def test_help_lists_modes() -> None:
    result = runner.invoke(app, ["diagnose-bump", "--help"])
    assert result.exit_code == 0
    out = _strip(result.output)
    assert "--mode" in out
    assert "--engine" in out
    assert "--regate-report" in out
    assert "--schema" in out


def test_carried_mode_writes_confirmed_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {
        "engine": "transformers",
        "reconciliation": {"residual": [{"id": "rule_x", "reason": "decay"}]},
    }
    report_path = tmp_path / "regate.json"
    report_path.write_text(json.dumps(report))
    corpus_path = tmp_path / "carried.yaml"
    corpus_path.write_text("schema_version: '1.0.0'\nengine: transformers\ninvariants: []\n")
    out_path = tmp_path / "proposed.yaml"

    captured: dict = {}

    def fake_carried(**kwargs):
        captured.update(kwargs)
        return DiagnoseResult(
            engine="transformers",
            engine_version="5.7.0",
            mode="carried_failure_triage",
            confirmed=[{"id": "rule_x", "added_by": "llm_diagnose"}],
            proposed_yaml="invariants:\n- id: rule_x\n  added_by: llm_diagnose\n",
        )

    monkeypatch.setattr(diag_cmd, "diagnose_carried_failures", fake_carried)

    result = runner.invoke(
        app,
        [
            "diagnose-bump",
            "--engine",
            "transformers",
            "--image",
            "img:5.7.0",
            "--new-version",
            "5.7.0",
            "--old-version",
            "4.57.3",
            "--source-root",
            str(tmp_path),
            "--mode",
            "carried",
            "--regate-report",
            str(report_path),
            "--carried-corpus",
            str(corpus_path),
            "--cited-file",
            "configuration_utils.py",
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # Residual ids were pulled from the report's reconciliation block.
    assert captured["residual_ids"] == ["rule_x"]
    assert captured["cited_files"] == ["configuration_utils.py"]
    # Only the confirmed YAML is written.
    assert out_path.read_text().count("added_by: llm_diagnose") == 1


def test_carried_mode_empty_residual_short_circuits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"engine": "transformers", "reconciliation": {"residual": []}}
    report_path = tmp_path / "regate.json"
    report_path.write_text(json.dumps(report))
    corpus_path = tmp_path / "carried.yaml"
    corpus_path.write_text("invariants: []\n")

    called = {"n": 0}

    def fake_carried(**kwargs):
        called["n"] += 1
        return DiagnoseResult(engine="transformers", engine_version="5.7.0", mode="x")

    monkeypatch.setattr(diag_cmd, "diagnose_carried_failures", fake_carried)

    result = runner.invoke(
        app,
        [
            "diagnose-bump",
            "--engine",
            "transformers",
            "--image",
            "img:5.7.0",
            "--new-version",
            "5.7.0",
            "--old-version",
            "4.57.3",
            "--source-root",
            str(tmp_path),
            "--mode",
            "carried",
            "--regate-report",
            str(report_path),
            "--carried-corpus",
            str(corpus_path),
            "--cited-file",
            "x.py",
        ],
    )
    assert result.exit_code == 0
    # Rung 0 healed everything -> diagnose never runs (1a dormant on this bump).
    assert called["n"] == 0
    assert "Nothing to diagnose" in _strip(result.output)


def test_carried_mode_missing_required_flag_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose-bump",
            "--engine",
            "transformers",
            "--image",
            "img",
            "--new-version",
            "5.7.0",
            "--source-root",
            str(tmp_path),
            "--mode",
            "carried",
        ],
    )
    assert result.exit_code != 0


def test_gap_mode_dispatches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps({"engine": "transformers", "engine_params": {}}))

    captured: dict = {}

    def fake_gap(**kwargs):
        captured.update(kwargs)
        return DiagnoseResult(engine="transformers", engine_version="5.7.0", mode="gap_diagnose")

    monkeypatch.setattr(diag_cmd, "diagnose_gaps", fake_gap)

    result = runner.invoke(
        app,
        [
            "diagnose-bump",
            "--engine",
            "transformers",
            "--image",
            "img:5.7.0",
            "--new-version",
            "5.7.0",
            "--source-root",
            str(tmp_path),
            "--mode",
            "gap",
            "--schema",
            str(schema_path),
            "--surface-file",
            "config.py",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["config_surface_files"] == ["config.py"]


def test_bad_mode_errors(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "diagnose-bump",
            "--engine",
            "transformers",
            "--image",
            "img",
            "--new-version",
            "5.7.0",
            "--source-root",
            str(tmp_path),
            "--mode",
            "sideways",
        ],
    )
    assert result.exit_code != 0
