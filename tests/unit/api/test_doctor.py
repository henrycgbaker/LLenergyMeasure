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
