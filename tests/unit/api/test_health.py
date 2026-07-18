"""Unit tests for the environment health report backing ``llem doctor``.

Every environment probe is mocked - these tests never touch GPU hardware, a
Docker daemon, or the real filesystem config.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

import llenergymeasure.api.health as health
from llenergymeasure.api.doctor import DoctorReport, EngineDoctorResult, TrtCacheHealth
from llenergymeasure.api.health import (
    CheckLine,
    HealthReport,
    HealthSection,
    build_health_report,
)
from llenergymeasure.config.user_config import UserConfig
from llenergymeasure.infra.runner_resolution import RunnerSpec
from llenergymeasure.infra.version_handshake import SchemaStatus


def _find_spec_for(*available: str):
    """Return a find_spec side effect that reports *available* modules present."""

    def _side_effect(name: str) -> MagicMock | None:
        return MagicMock() if name in available else None

    return _side_effect


# ---------------------------------------------------------------------------
# HealthReport severity / exit-code logic
# ---------------------------------------------------------------------------


def _report(*statuses: str) -> HealthReport:
    lines = [CheckLine(s, f"line {s}") for s in statuses]  # type: ignore[arg-type]
    return HealthReport(sections=[HealthSection("s", lines)])


def test_worst_and_counts() -> None:
    report = _report("ok", "warn", "ok", "fail", "warn")
    assert report.worst == "fail"
    assert report.counts == {"ok": 2, "warn": 2, "fail": 1}


@pytest.mark.parametrize(
    ("statuses", "expected_code"),
    [
        (("ok", "ok"), 0),
        (("ok", "warn"), 1),
        (("warn", "fail"), 2),
    ],
)
def test_check_exit_code(statuses: tuple[str, ...], expected_code: int) -> None:
    assert _report(*statuses).check_exit_code == expected_code


def test_report_to_dict_is_json_serialisable() -> None:
    report = _report("ok", "warn")
    payload = report.to_dict()
    assert payload["status"] == "warn"
    assert payload["summary"] == {"ok": 1, "warn": 1, "fail": 0}
    # Must round-trip through json without error.
    json.loads(json.dumps(payload))


# ---------------------------------------------------------------------------
# GPU / driver section
# ---------------------------------------------------------------------------


def test_gpu_section_with_gpus() -> None:
    gpus = [{"name": "NVIDIA A100", "vram_gb": 80.0}]
    with patch.object(health, "gpu_inventory", return_value=(gpus, "535.129.03")):
        section = health._gpu_section()
    text = " ".join(line.message for line in section.lines)
    assert "NVIDIA A100" in text
    assert "535.129.03" in text
    assert all(line.status == "ok" for line in section.lines)


def test_gpu_section_no_gpu_warns() -> None:
    with patch.object(health, "gpu_inventory", return_value=([], None)):
        section = health._gpu_section()
    statuses = {line.status for line in section.lines}
    assert "warn" in statuses
    warn_line = next(line for line in section.lines if line.status == "warn")
    assert warn_line.fix is not None


# ---------------------------------------------------------------------------
# Engines section
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("importable", "docker_avail", "cached", "expected_status", "msg_substr", "fix_substr"),
    [
        # Importable in the host env -> ok, regardless of runner mode.
        pytest.param(
            True, False, False, "ok", "importable locally (0.19.1)", None, id="importable-locally"
        ),
        # Not importable, Docker up, image already pulled -> ok.
        pytest.param(False, True, True, "ok", "cached locally", None, id="docker-cached"),
        # Not importable, Docker up, image absent -> warn with a pull hint.
        pytest.param(
            False, True, False, "warn", "not cached locally", "docker pull", id="docker-not-cached"
        ),
        # Not importable and no Docker -> warn; engines are not pip extras, so the
        # fix must point at Docker.
        pytest.param(
            False, False, False, "warn", "Docker unavailable", "Docker", id="no-local-no-docker"
        ),
    ],
)
def test_engine_line_status(
    importable: bool,
    docker_avail: bool,
    cached: bool,
    expected_status: str,
    msg_substr: str,
    fix_substr: str | None,
) -> None:
    spec = RunnerSpec(
        mode="docker" if docker_avail else "local", image=None, source="auto_detected"
    )
    find_spec = _find_spec_for("vllm") if importable else _find_spec_for()
    with (
        patch("importlib.util.find_spec", side_effect=find_spec),
        patch.object(health, "_probe_engine_version", return_value="0.19.1"),
        patch.object(health, "is_docker_available", return_value=docker_avail),
        patch.object(health, "get_default_image", return_value="vllm/vllm-openai:v0.19.1"),
        patch.object(health, "image_present_locally", return_value=cached),
    ):
        line = health._engine_line("vllm", spec)
    assert line.status == expected_status
    assert msg_substr in line.message
    if fix_substr is not None:
        assert line.fix is not None
        assert fix_substr in line.fix


# ---------------------------------------------------------------------------
# Energy section
# ---------------------------------------------------------------------------


def test_energy_section_nvml_only() -> None:
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for("pynvml")),
        patch("llenergymeasure.api.health.probe_energy_sampler", return_value="NVMLSampler"),
    ):
        section = health._energy_section()
    by_status = {line.status for line in section.lines}
    text = "\n".join(line.message for line in section.lines)
    assert "NVML" in text
    assert "NVMLSampler" in text
    # Zeus + CodeCarbon absent -> warnings present.
    assert "warn" in by_status


def test_energy_section_none_available() -> None:
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for()),
        patch("llenergymeasure.api.health.probe_energy_sampler", return_value=None),
    ):
        section = health._energy_section()
    messages = "\n".join(line.message for line in section.lines)
    assert "No energy sampler available" in messages


# ---------------------------------------------------------------------------
# Docker section
# ---------------------------------------------------------------------------


def test_docker_section_all_present() -> None:
    with (
        patch.object(health.shutil, "which", return_value="/usr/bin/x"),
        patch.object(health, "docker_daemon_reachable", return_value=True),
    ):
        section = health._docker_section()
    assert all(line.status == "ok" for line in section.lines)
    text = "\n".join(line.message for line in section.lines)
    assert "Docker daemon: reachable" in text


def test_docker_section_cli_absent_warns() -> None:
    with (
        patch.object(health.shutil, "which", return_value=None),
        patch.object(health, "docker_daemon_reachable", return_value=None),
    ):
        section = health._docker_section()
    statuses = [line.status for line in section.lines]
    assert "warn" in statuses
    # No daemon line is emitted when the CLI is absent (already reported).
    assert not any("daemon" in line.message for line in section.lines)


# ---------------------------------------------------------------------------
# Credentials section (HF_TOKEN never printed)
# ---------------------------------------------------------------------------


def test_credentials_token_present(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_supersecretvalue")
    section = health._credentials_section()
    line = section.lines[0]
    assert line.status == "ok"
    # The token value must never appear in the report.
    assert "hf_supersecretvalue" not in line.message
    assert line.fix is None or "hf_supersecretvalue" not in line.fix


def test_credentials_token_absent_warns(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    section = health._credentials_section()
    line = section.lines[0]
    assert line.status == "warn"
    assert line.fix is not None


# ---------------------------------------------------------------------------
# Configuration section
# ---------------------------------------------------------------------------


def test_configuration_defaults() -> None:
    specs = {"vllm": RunnerSpec(mode="docker", image=None, source="auto_detected")}
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    with patch.object(health, "get_user_config_path", return_value=fake_path):
        section = health._configuration_section(UserConfig(), None, specs)
    text = "\n".join(line.message for line in section.lines)
    assert "using built-in defaults" in text
    assert "runner.vllm: docker (source=auto_detected)" in text
    assert all(line.status == "ok" for line in section.lines)


def test_configuration_invalid_config_fails() -> None:
    specs: dict[str, RunnerSpec] = {}
    fake_path = MagicMock()
    fake_path.exists.return_value = True
    with patch.object(health, "get_user_config_path", return_value=fake_path):
        section = health._configuration_section(None, "bad value at runners.vllm", specs)
    fail_line = section.lines[0]
    assert fail_line.status == "fail"
    assert "bad value" in (fail_line.fix or "")


# ---------------------------------------------------------------------------
# Image schema handshake section
# ---------------------------------------------------------------------------


def _doctor_report(status: SchemaStatus, detail: str = "") -> DoctorReport:
    return DoctorReport(
        host_pkg_version="0.13.0",
        host_fingerprint="a" * 64,
        skip_check_active=False,
        results=[
            EngineDoctorResult(
                engine="vllm",
                image="vllm/vllm-openai:v0.19.1",
                pkg_version="0.19.1",
                image_fingerprint="a" * 64,
                status=status,
                local_present=True,
                detail=detail,
            )
        ],
    )


def test_handshake_mismatch_is_fail() -> None:
    report = _doctor_report(SchemaStatus.MISMATCH, detail="rebuild: make docker-build-vllm")
    section = health._image_handshake_section(report)
    fail = next(line for line in section.lines if line.status == "fail")
    assert "MISMATCH" in fail.message
    assert "rebuild" in (fail.fix or "")


def test_handshake_local_tag_shadow_renders_warn_line() -> None:
    report = DoctorReport(
        host_pkg_version="0.13.0",
        host_fingerprint="a" * 64,
        skip_check_active=False,
        results=[
            EngineDoctorResult(
                engine="vllm",
                image="llenergymeasure:vllm",
                pkg_version="0.19.1",
                image_fingerprint="a" * 64,
                status=SchemaStatus.OK,
                local_present=True,
                shadows_default="vllm/vllm-openai:v0.19.1",
            )
        ],
    )
    section = health._image_handshake_section(report)
    shadow = next(line for line in section.lines if "shadows pinned default" in line.message)
    assert shadow.status == "warn"
    assert "llenergymeasure:vllm" in shadow.message
    assert "vllm/vllm-openai:v0.19.1" in shadow.message
    assert "docker rmi llenergymeasure:vllm" in (shadow.fix or "")


def test_handshake_unreachable_is_warn() -> None:
    report = _doctor_report(SchemaStatus.UNREACHABLE)
    section = health._image_handshake_section(report)
    assert any(line.status == "warn" for line in section.lines)
    assert not any(line.status == "fail" for line in section.lines)


def test_handshake_none_degrades_to_warn() -> None:
    section = health._image_handshake_section(None)
    assert section.lines[0].status == "warn"
    assert section.lines[0].fix is not None


def test_handshake_trt_cache_rendered() -> None:
    report = _doctor_report(SchemaStatus.OK)
    cache = TrtCacheHealth(
        path="/cache/trt",
        exists=True,
        entry_count=2,
        total_bytes=2 * 1024**3,
        clean_hint="rm -rf ...",
    )
    report = DoctorReport(
        host_pkg_version=report.host_pkg_version,
        host_fingerprint=report.host_fingerprint,
        skip_check_active=report.skip_check_active,
        results=report.results,
        trt_cache=cache,
    )
    section = health._image_handshake_section(report)
    text = "\n".join(line.message for line in section.lines)
    assert "/cache/trt" in text


# ---------------------------------------------------------------------------
# build_health_report end-to-end (all probes mocked)
# ---------------------------------------------------------------------------


def test_build_health_report_smoke() -> None:
    spec = RunnerSpec(mode="docker", image=None, source="auto_detected")
    with (
        patch.object(health, "_load_user_config_safe", return_value=(UserConfig(), None)),
        patch.object(health, "resolve_runner", return_value=spec),
        patch.object(health, "run_doctor_checks", return_value=_doctor_report(SchemaStatus.OK)),
        patch.object(health, "gpu_inventory", return_value=([], None)),
        patch.object(health, "is_docker_available", return_value=True),
        patch.object(health, "get_default_image", return_value="img:tag"),
        patch.object(health, "image_present_locally", return_value=True),
        patch.object(health, "docker_daemon_reachable", return_value=True),
        patch("importlib.util.find_spec", side_effect=_find_spec_for("pynvml")),
        patch("llenergymeasure.api.health.probe_energy_sampler", return_value="NVMLSampler"),
    ):
        report = build_health_report()

    titles = [section.title for section in report.sections]
    assert titles == [
        "GPU / driver",
        "Engines",
        "Energy measurement",
        "Docker",
        "Credentials",
        "Configuration",
        "Image schema handshake",
    ]
    # Fully serialisable for --json.
    json.loads(json.dumps(report.to_dict()))


def test_build_health_report_survives_image_check_crash() -> None:
    spec = RunnerSpec(mode="local", image=None, source="default")
    with (
        patch.object(health, "_load_user_config_safe", return_value=(UserConfig(), None)),
        patch.object(health, "resolve_runner", return_value=spec),
        patch.object(health, "run_doctor_checks", side_effect=RuntimeError("docker exploded")),
        patch.object(health, "gpu_inventory", return_value=([], None)),
        patch.object(health, "is_docker_available", return_value=False),
        patch.object(health, "docker_daemon_reachable", return_value=None),
        patch("importlib.util.find_spec", side_effect=_find_spec_for("pynvml")),
        patch("llenergymeasure.api.health.probe_energy_sampler", return_value="NVMLSampler"),
    ):
        report = build_health_report()

    handshake = report.sections[-1]
    assert handshake.title == "Image schema handshake"
    assert handshake.lines[0].status == "warn"
    assert report.image_report is None
