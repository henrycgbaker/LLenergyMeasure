"""Tests for the ``llem doctor`` CLI command (sectioned health check)."""

from __future__ import annotations

import json
from unittest.mock import patch

from typer.testing import CliRunner

from llenergymeasure.api.health import CheckLine, HealthReport, HealthSection
from llenergymeasure.cli import app

runner = CliRunner()

_BUILD = "llenergymeasure.api.health.build_health_report"


def patch_build(report: HealthReport):  # type: ignore[no-untyped-def]
    return patch(_BUILD, return_value=report)


def _report(*sections: HealthSection) -> HealthReport:
    return HealthReport(sections=list(sections))


def _ok_report() -> HealthReport:
    return _report(HealthSection("GPU / driver", [CheckLine("ok", "GPU 0: A100 (80.0 GB)")]))


def _warn_report() -> HealthReport:
    return _report(
        HealthSection(
            "Energy measurement",
            [CheckLine("warn", "Zeus: not installed", "pip install 'llenergymeasure[zeus]'")],
        )
    )


def _fail_report() -> HealthReport:
    return _report(
        HealthSection(
            "Image schema handshake",
            [CheckLine("fail", "vllm: MISMATCH (img)", "rebuild: make docker-build-vllm")],
        )
    )


def test_default_all_ok_exits_zero() -> None:
    with patch_build(_ok_report()):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Environment health check" in result.output
    assert "[ok]" in result.output
    assert "Summary:" in result.output


def test_default_warnings_still_exit_zero() -> None:
    with patch_build(_warn_report()):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "[warn]" in result.output
    # Fix hint rendered.
    assert "-> pip install" in result.output


def test_default_failure_exits_one() -> None:
    with patch_build(_fail_report()):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "[fail]" in result.output
    assert "MISMATCH" in result.output


def test_check_flag_grades_warnings_as_one() -> None:
    with patch_build(_warn_report()):
        result = runner.invoke(app, ["doctor", "--check"])
    assert result.exit_code == 1


def test_check_flag_grades_errors_as_two() -> None:
    with patch_build(_fail_report()):
        result = runner.invoke(app, ["doctor", "--check"])
    assert result.exit_code == 2


def test_check_flag_all_ok_is_zero() -> None:
    with patch_build(_ok_report()):
        result = runner.invoke(app, ["doctor", "--check"])
    assert result.exit_code == 0


def test_json_output_is_machine_readable() -> None:
    with patch_build(_warn_report()):
        result = runner.invoke(app, ["doctor", "--json"])
    payload = json.loads(result.output)
    assert payload["status"] == "warn"
    assert payload["summary"]["warn"] == 1
    assert payload["sections"][0]["title"] == "Energy measurement"


def test_json_with_check_grades_exit_code() -> None:
    with patch_build(_fail_report()):
        result = runner.invoke(app, ["doctor", "--json", "--check"])
    assert result.exit_code == 2
    # Still valid JSON on stdout.
    json.loads(result.output)


def test_section_titles_rendered() -> None:
    report = _report(
        HealthSection("GPU / driver", [CheckLine("ok", "x")]),
        HealthSection("Docker", [CheckLine("ok", "y")]),
    )
    with patch_build(report):
        result = runner.invoke(app, ["doctor"])
    assert "GPU / driver" in result.output
    assert "Docker" in result.output
