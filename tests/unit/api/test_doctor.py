"""Tests for ``run_doctor_checks`` image resolution + local-presence reporting."""

from __future__ import annotations

from unittest.mock import patch

from llenergymeasure.api.doctor import SchemaStatus, run_doctor_checks
from llenergymeasure.infra.version_handshake import ImageStamp
from llenergymeasure.utils.exceptions import ConfigError

_EMPTY_STAMP = ImageStamp(pkg_version=None, expconf_fingerprint=None)


def test_reports_resolved_image_and_local_presence():
    with (
        patch("llenergymeasure.api.doctor.compute_expconf_fingerprint", return_value="f" * 64),
        patch(
            "llenergymeasure.api.doctor.get_default_image",
            return_value="vllm/vllm-openai:v0.19.1",
        ),
        patch("llenergymeasure.api.doctor.image_present_locally", return_value=True),
        patch("llenergymeasure.api.doctor.inspect_image_stamp", return_value=_EMPTY_STAMP),
    ):
        report = run_doctor_checks(engines=("vllm",))

    (row,) = report.results
    assert row.engine == "vllm"
    assert row.image == "vllm/vllm-openai:v0.19.1"
    assert row.local_present is True


def test_unresolvable_default_becomes_unreachable_row_with_fix():
    def _raise(engine):
        raise ConfigError(f'Set runners.{engine} to "docker:<image>:<tag>"')

    with (
        patch("llenergymeasure.api.doctor.compute_expconf_fingerprint", return_value="f" * 64),
        patch("llenergymeasure.api.doctor.get_default_image", side_effect=_raise),
    ):
        report = run_doctor_checks(engines=("tensorrt",))

    (row,) = report.results
    assert row.status is SchemaStatus.UNREACHABLE
    assert row.local_present is None
    assert row.image == "(unresolved)"
    assert 'runners.tensorrt to "docker:' in row.detail
    assert not report.any_mismatch


# ---------------------------------------------------------------------------
# run_trt_cache_check: TRT-LLM build-cache visibility (manual + visible policy)
# ---------------------------------------------------------------------------


def test_trt_cache_check_absent_dir(tmp_path):
    """A non-existent cache dir reports exists=False, zero entries/bytes."""
    from llenergymeasure.api.doctor import run_trt_cache_check

    missing = tmp_path / "nope"
    with patch("llenergymeasure.api.doctor.trt_build_cache_host_dir", return_value=missing):
        health = run_trt_cache_check()

    assert health.exists is False
    assert health.entry_count == 0
    assert health.total_bytes == 0
    assert "rm -rf" in health.clean_hint


def test_trt_cache_check_counts_entries_and_bytes(tmp_path):
    """Counts engine-* entries and sums file bytes; ignores non-engine files."""
    from llenergymeasure.api.doctor import run_trt_cache_check

    (tmp_path / "engine-aaa").mkdir()
    (tmp_path / "engine-aaa" / "content").write_bytes(b"x" * 100)
    (tmp_path / "engine-bbb").mkdir()
    (tmp_path / "engine-bbb" / "content").write_bytes(b"y" * 50)
    (tmp_path / "not-an-engine.txt").write_bytes(b"z" * 7)  # counted in bytes, not entries

    with patch("llenergymeasure.api.doctor.trt_build_cache_host_dir", return_value=tmp_path):
        health = run_trt_cache_check()

    assert health.exists is True
    assert health.entry_count == 2
    assert health.total_bytes == 157
    assert health.path == str(tmp_path)


def test_run_doctor_checks_includes_trt_cache():
    """The full report carries a trt_cache section (never affects exit code)."""
    with (
        patch("llenergymeasure.api.doctor.compute_expconf_fingerprint", return_value="f" * 64),
        patch(
            "llenergymeasure.api.doctor.get_default_image",
            return_value="vllm/vllm-openai:v0.19.1",
        ),
        patch("llenergymeasure.api.doctor.image_present_locally", return_value=True),
        patch("llenergymeasure.api.doctor.inspect_image_stamp", return_value=_EMPTY_STAMP),
    ):
        report = run_doctor_checks(engines=("vllm",))

    assert report.trt_cache is not None
    assert isinstance(report.trt_cache.entry_count, int)
