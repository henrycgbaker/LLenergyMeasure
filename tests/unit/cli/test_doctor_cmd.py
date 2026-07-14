"""Tests for ``llem doctor`` CLI command."""

from __future__ import annotations

from unittest.mock import patch

from typer.testing import CliRunner

from llenergymeasure.api.doctor import (
    DoctorReport,
    EngineDoctorResult,
    SchemaStatus,
)
from llenergymeasure.cli import app

runner = CliRunner()


def _report(
    results: list[EngineDoctorResult],
    *,
    host_fp: str = "a" * 64,
    host_pkg: str = "0.9.0",
    skip_active: bool = False,
) -> DoctorReport:
    return DoctorReport(
        host_pkg_version=host_pkg,
        host_fingerprint=host_fp,
        skip_check_active=skip_active,
        results=results,
    )


def test_all_ok_exits_zero() -> None:
    report = _report(
        [
            EngineDoctorResult(
                engine="transformers",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "transformers" in result.output
    assert "OK" in result.output


def test_mismatch_exits_nonzero() -> None:
    report = _report(
        [
            EngineDoctorResult(
                engine="transformers",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="b" * 64,
                status=SchemaStatus.MISMATCH,
                detail="rebuild: make docker-build-pytorch",
            ),
            EngineDoctorResult(
                engine="vllm",
                image="llenergymeasure:vllm",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            ),
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "MISMATCH" in result.output
    assert "rebuild" in result.output


def test_unreachable_is_not_mismatch() -> None:
    report = _report(
        [
            EngineDoctorResult(
                engine="transformers",
                image="llenergymeasure:pytorch",
                pkg_version=None,
                image_fingerprint=None,
                status=SchemaStatus.UNREACHABLE,
                detail="no labels",
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "UNREACHABLE" in result.output


def test_skip_check_warning_rendered() -> None:
    report = _report(
        [
            EngineDoctorResult(
                engine="transformers",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ],
        skip_active=True,
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "LLEM_SKIP_IMAGE_CHECK" in result.output


def test_engine_column_aligns_for_transformers() -> None:
    """The Engine column is wide enough for 'transformers' (12 chars) - no overflow."""
    report = _report(
        [
            EngineDoctorResult(
                engine="transformers",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    lines = result.output.splitlines()
    header_line = next(line for line in lines if line.startswith("Engine"))
    data_line = next(line for line in lines if line.startswith("transformers"))
    # The Image column must start at the same offset in both rows - i.e. the
    # long engine name did not push its row's columns out of alignment.
    assert header_line.index("Image") == data_line.index("llenergymeasure:pytorch")


def test_local_present_column_rendered() -> None:
    """The Local column reports whether the resolved image is cached locally."""
    report = _report(
        [
            EngineDoctorResult(
                engine="vllm",
                image="vllm/vllm-openai:v0.19.1",
                pkg_version=None,
                image_fingerprint=None,
                status=SchemaStatus.UNREACHABLE,
                local_present=True,
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    lines = result.output.splitlines()
    header_line = next(line for line in lines if line.startswith("Engine"))
    assert "Local" in header_line
    data_line = next(line for line in lines if line.startswith("vllm"))
    assert "yes" in data_line


def test_host_footer_rendered() -> None:
    report = _report(
        [
            EngineDoctorResult(
                engine="transformers",
                image="llenergymeasure:pytorch",
                pkg_version="0.9.0",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ]
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])
    assert "Host llenergymeasure version: 0.9.0" in result.output
    assert "Host ExperimentConfig SHA-256:" in result.output


def test_renders_trt_cache_section() -> None:
    """The doctor output includes the TRT-LLM build-cache location, size, and clean hint."""
    from llenergymeasure.api.doctor import TrtCacheHealth

    cache = TrtCacheHealth(
        path="/home/u/.cache/trt-llm",
        exists=True,
        entry_count=3,
        total_bytes=3 * 1024 * 1024 * 1024,
        clean_hint="clean manually (llem never auto-evicts): rm -rf /home/u/.cache/trt-llm/engine-*",
    )
    report = _report(
        [
            EngineDoctorResult(
                engine="tensorrt",
                image="nvcr.io/nvidia/tensorrt-llm/release:1.2.1",
                pkg_version="1.2.1",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
            )
        ]
    )
    report = DoctorReport(
        host_pkg_version=report.host_pkg_version,
        host_fingerprint=report.host_fingerprint,
        skip_check_active=report.skip_check_active,
        results=report.results,
        trt_cache=cache,
    )
    with patch("llenergymeasure.api.doctor.run_doctor_checks", return_value=report):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "engine build cache" in result.output
    assert "/home/u/.cache/trt-llm" in result.output
    assert "3 engine(s)" in result.output
    assert "3.0 GB" in result.output
    assert "auto-evicts" in result.output
