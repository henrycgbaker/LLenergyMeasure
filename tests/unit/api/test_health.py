"""Unit tests for the environment health report backing ``llem doctor``.

Every environment probe is mocked - these tests never touch GPU hardware, a
Docker daemon, or the real filesystem config.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

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


def test_check_exit_code_all_ok() -> None:
    assert _report("ok", "ok").check_exit_code == 0


def test_check_exit_code_warnings() -> None:
    assert _report("ok", "warn").check_exit_code == 1


def test_check_exit_code_errors() -> None:
    assert _report("warn", "fail").check_exit_code == 2


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
    with (
        patch.object(health, "_probe_gpu", return_value=gpus),
        patch.object(health, "_probe_driver_version", return_value="535.129.03"),
    ):
        section = health._gpu_section()
    text = " ".join(line.message for line in section.lines)
    assert "NVIDIA A100" in text
    assert "535.129.03" in text
    assert all(line.status == "ok" for line in section.lines)


def test_gpu_section_no_gpu_warns() -> None:
    with patch.object(health, "_probe_gpu", return_value=None):
        section = health._gpu_section()
    statuses = {line.status for line in section.lines}
    assert "warn" in statuses
    warn_line = next(line for line in section.lines if line.status == "warn")
    assert warn_line.fix is not None


# ---------------------------------------------------------------------------
# Engines section
# ---------------------------------------------------------------------------


def test_engine_line_importable_locally() -> None:
    spec = RunnerSpec(mode="local", image=None, source="user_config")
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for("vllm")),
        patch.object(health, "_probe_engine_version", return_value="0.19.1"),
    ):
        line = health._engine_line("vllm", spec)
    assert line.status == "ok"
    assert "importable locally (0.19.1)" in line.message


def test_engine_line_docker_cached() -> None:
    spec = RunnerSpec(mode="docker", image=None, source="auto_detected")
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for()),
        patch.object(health, "is_docker_available", return_value=True),
        patch.object(health, "get_default_image", return_value="vllm/vllm-openai:v0.19.1"),
        patch.object(health, "image_present_locally", return_value=True),
    ):
        line = health._engine_line("vllm", spec)
    assert line.status == "ok"
    assert "cached locally" in line.message


def test_engine_line_docker_not_cached_warns_with_pull_hint() -> None:
    spec = RunnerSpec(mode="docker", image=None, source="auto_detected")
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for()),
        patch.object(health, "is_docker_available", return_value=True),
        patch.object(health, "get_default_image", return_value="vllm/vllm-openai:v0.19.1"),
        patch.object(health, "image_present_locally", return_value=False),
    ):
        line = health._engine_line("vllm", spec)
    assert line.status == "warn"
    assert line.fix is not None
    assert "docker pull" in line.fix


def test_engine_line_no_local_no_docker_warns() -> None:
    spec = RunnerSpec(mode="local", image=None, source="default")
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for()),
        patch.object(health, "is_docker_available", return_value=False),
    ):
        line = health._engine_line("tensorrt", spec)
    assert line.status == "warn"
    assert line.fix is not None
    # Engines are not pip-installable extras - the fix must point at Docker.
    assert "Docker" in line.fix


# ---------------------------------------------------------------------------
# Energy section
# ---------------------------------------------------------------------------


def test_energy_section_nvml_only() -> None:
    with (
        patch("importlib.util.find_spec", side_effect=_find_spec_for("pynvml")),
        patch("llenergymeasure.api.probe_energy_sampler", return_value="NVMLSampler"),
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
        patch("llenergymeasure.api.probe_energy_sampler", return_value=None),
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
        patch.object(health, "_docker_daemon_running", return_value=True),
    ):
        section = health._docker_section()
    assert all(line.status == "ok" for line in section.lines)
    text = "\n".join(line.message for line in section.lines)
    assert "Docker daemon: reachable" in text


def test_docker_section_cli_absent_warns() -> None:
    with (
        patch.object(health.shutil, "which", return_value=None),
        patch.object(health, "_docker_daemon_running", return_value=None),
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
        patch.object(health, "_probe_gpu", return_value=None),
        patch.object(health, "is_docker_available", return_value=True),
        patch.object(health, "get_default_image", return_value="img:tag"),
        patch.object(health, "image_present_locally", return_value=True),
        patch.object(health, "_docker_daemon_running", return_value=True),
        patch("importlib.util.find_spec", side_effect=_find_spec_for("pynvml")),
        patch("llenergymeasure.api.probe_energy_sampler", return_value="NVMLSampler"),
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
        patch.object(health, "_probe_gpu", return_value=None),
        patch.object(health, "is_docker_available", return_value=False),
        patch.object(health, "_docker_daemon_running", return_value=None),
        patch("importlib.util.find_spec", side_effect=_find_spec_for("pynvml")),
        patch("llenergymeasure.api.probe_energy_sampler", return_value="NVMLSampler"),
    ):
        report = build_health_report()

    handshake = report.sections[-1]
    assert handshake.title == "Image schema handshake"
    assert handshake.lines[0].status == "warn"
    assert report.image_report is None
