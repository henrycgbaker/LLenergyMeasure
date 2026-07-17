"""Environment health check backing the ``llem doctor`` command.

Builds a sectioned, machine-serialisable report describing whether the host is
ready to run measurements: GPU/driver, per-engine availability, energy samplers,
Docker + NVIDIA Container Toolkit, credentials, resolved configuration, and the
image schema handshake. Every check degrades gracefully - a missing GPU, absent
Docker daemon, or unreachable image becomes a warning with an actionable fix
hint rather than an exception.

Detection logic is reused from the canonical homes (``runner_resolution``,
``image_registry``, ``docker_preflight``, ``user_config``, ``api.doctor``); this
module only classifies and presents the results.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Literal

from llenergymeasure.api import probe_energy_sampler
from llenergymeasure.api.doctor import DoctorReport, run_doctor_checks
from llenergymeasure.config.ssot import (
    ENGINE_PACKAGES,
    ENV_HF_TOKEN,
    RUNNER_DOCKER,
    TIMEOUT_DOCKER_CLI,
    Engine,
    engine_str,
)
from llenergymeasure.config.user_config import (
    UserConfig,
    get_user_config_path,
    load_user_config,
)

# Reuse the canonical Docker-availability helpers rather than re-implementing the
# PATH probes (their home is docker_preflight; runner_resolution reuses them too).
from llenergymeasure.infra.docker_preflight import (
    DOCKER_INSTALL_URL,
    NVIDIA_TOOLKIT_BINS,
    NVIDIA_TOOLKIT_INSTALL_URL,
)
from llenergymeasure.infra.image_registry import get_default_image, image_present_locally
from llenergymeasure.infra.runner_resolution import RunnerSpec, is_docker_available, resolve_runner
from llenergymeasure.infra.version_handshake import SchemaStatus
from llenergymeasure.utils.exceptions import ConfigError

__all__ = [
    "CheckLine",
    "HealthReport",
    "HealthSection",
    "build_health_report",
]

Status = Literal["ok", "warn", "fail"]

# Severity ranking - the report's overall status is the worst line it contains.
_SEVERITY: dict[Status, int] = {"ok": 0, "warn": 1, "fail": 2}

# Map an image schema-handshake status to a health severity. A MISMATCH is the
# one genuine failure (the CI-gating signal preserved from the original doctor);
# everything else that is merely absent or unverifiable is a warning.
_SCHEMA_TO_STATUS: dict[SchemaStatus, Status] = {
    SchemaStatus.OK: "ok",
    SchemaStatus.MISMATCH: "fail",
    SchemaStatus.UNVERIFIED: "warn",
    SchemaStatus.UNREACHABLE: "warn",
    SchemaStatus.BYPASSED: "warn",
}

_HF_TOKEN_URL = "https://huggingface.co/settings/tokens"
_DOCKER_SETUP_DOC = "docs/how-to/docker-setup.md"


@dataclass(frozen=True)
class CheckLine:
    """One health-check line: a status, a human message, and an optional fix."""

    status: Status
    message: str
    fix: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "message": self.message, "fix": self.fix}


@dataclass(frozen=True)
class HealthSection:
    """A named group of check lines."""

    title: str
    lines: list[CheckLine] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "lines": [line.to_dict() for line in self.lines]}


@dataclass(frozen=True)
class HealthReport:
    """Full environment health report: ordered sections plus image-handshake data."""

    sections: list[HealthSection]
    image_report: DoctorReport | None = None

    @property
    def all_lines(self) -> list[CheckLine]:
        return [line for section in self.sections for line in section.lines]

    @property
    def counts(self) -> dict[Status, int]:
        counts: dict[Status, int] = {"ok": 0, "warn": 0, "fail": 0}
        for line in self.all_lines:
            counts[line.status] += 1
        return counts

    @property
    def worst(self) -> Status:
        worst: Status = "ok"
        for line in self.all_lines:
            if _SEVERITY[line.status] > _SEVERITY[worst]:
                worst = line.status
        return worst

    @property
    def check_exit_code(self) -> int:
        """Exit code for ``--check`` mode: 0 = all ok, 1 = warnings, 2 = errors."""
        return _SEVERITY[self.worst]

    def to_dict(self) -> dict[str, Any]:
        counts = self.counts
        return {
            "status": self.worst,
            "summary": counts,
            "sections": [section.to_dict() for section in self.sections],
            "image_handshake": _image_report_to_dict(self.image_report),
        }


# ---------------------------------------------------------------------------
# Low-level probes (moved from the former cli/config_cmd.py)
# ---------------------------------------------------------------------------


def _probe_gpu() -> list[dict[str, Any]] | None:
    """Return a list of GPU info dicts, or None if none are visible."""
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        gpus: list[dict[str, Any]] = []
        with nvml_context():
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({"name": name, "vram_gb": mem.total / 1e9})
        return gpus if gpus else None
    except Exception:
        return None


def _probe_driver_version() -> str | None:
    """Return the NVIDIA driver version string, or None if unavailable."""
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        with nvml_context():
            raw = pynvml.nvmlSystemGetDriverVersion()
        return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        return None


def _probe_engine_version(engine: str) -> str | None:
    """Return the installed version of *engine*'s host package, or None."""
    try:
        package = ENGINE_PACKAGES[Engine(engine)]
        module = importlib.import_module(package)
    except Exception:
        return None
    version = getattr(module, "__version__", None)
    return str(version) if version else None


def _docker_daemon_running() -> bool | None:
    """Return whether the Docker daemon is reachable.

    None when the Docker CLI is not on PATH (nothing to probe); True/False
    otherwise. Never raises - a missing binary or timeout is reported as False.
    """
    if shutil.which("docker") is None:
        return None
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=TIMEOUT_DOCKER_CLI,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _gpu_section() -> HealthSection:
    lines: list[CheckLine] = []
    gpus = _probe_gpu()
    if gpus:
        for i, gpu in enumerate(gpus):
            lines.append(CheckLine("ok", f"GPU {i}: {gpu['name']} ({gpu['vram_gb']:.1f} GB)"))
        driver = _probe_driver_version()
        if driver:
            lines.append(CheckLine("ok", f"NVIDIA driver: {driver}"))
    else:
        lines.append(
            CheckLine(
                "warn",
                "No GPU detected",
                "a CUDA GPU is required to run measurements; on a remote-Docker-daemon "
                "setup the GPU lives on the daemon host",
            )
        )
    lines.append(CheckLine("ok", f"Python: {sys.version.split()[0]}"))
    return HealthSection("GPU / driver", lines)


def _resolve_image(engine: str) -> tuple[str | None, bool, str | None]:
    """Resolve *engine*'s default image and local-cache presence.

    Returns ``(image, cached, error)`` - *error* is a message when the default
    image cannot be resolved (a broken wheel), in which case *image* is None.
    """
    try:
        image = get_default_image(engine)
    except ConfigError as exc:
        return None, False, str(exc)
    return image, image_present_locally(image), None


def _engine_line(engine: str, spec: RunnerSpec) -> CheckLine:
    importable = importlib.util.find_spec(ENGINE_PACKAGES[Engine(engine)]) is not None
    if importable:
        version = _probe_engine_version(engine)
        installed = f"importable locally ({version})" if version else "importable locally"
        # Availability lives here; runner provenance (source=) belongs to the
        # Configuration section, so it is not repeated on this line.
        return CheckLine("ok", f"{engine}: {installed}; runner={spec.mode}")

    docker_avail = is_docker_available()
    if docker_avail and spec.mode == RUNNER_DOCKER:
        image, cached, error = _resolve_image(engine)
        if error is not None:
            return CheckLine("warn", f"{engine}: default image unresolved", error)
        if cached:
            return CheckLine("ok", f"{engine}: runs via Docker ({image}, cached locally)")
        return CheckLine(
            "warn",
            f"{engine}: runs via Docker ({image}, not cached locally)",
            f"docker pull {image}  (or it is pulled automatically on first run)",
        )

    return CheckLine(
        "warn",
        f"{engine}: not importable locally and Docker unavailable",
        f"install Docker + NVIDIA Container Toolkit - {engine} runs in its container "
        f"(engines are not pip-installable; see {_DOCKER_SETUP_DOC})",
    )


def _engines_section(specs: dict[str, RunnerSpec]) -> HealthSection:
    lines = [_engine_line(engine, specs[engine]) for engine in specs]
    return HealthSection("Engines", lines)


def _energy_section() -> HealthSection:
    lines: list[CheckLine] = []

    has_nvml = importlib.util.find_spec("pynvml") is not None
    has_zeus = importlib.util.find_spec("zeus") is not None
    has_codecarbon = importlib.util.find_spec("codecarbon") is not None

    if has_nvml:
        lines.append(CheckLine("ok", "NVML (nvidia-ml-py): available"))
    else:
        lines.append(
            CheckLine("warn", "NVML (nvidia-ml-py): not installed", "pip install nvidia-ml-py")
        )
    if has_zeus:
        lines.append(CheckLine("ok", "Zeus: available (higher-accuracy energy counter)"))
    else:
        lines.append(
            CheckLine(
                "warn",
                "Zeus: not installed (higher-accuracy energy counter)",
                "pip install 'llenergymeasure[zeus]'",
            )
        )
    if has_codecarbon:
        lines.append(CheckLine("ok", "CodeCarbon: available (fallback sampler)"))
    else:
        lines.append(
            CheckLine(
                "warn",
                "CodeCarbon: not installed (fallback sampler)",
                "pip install 'llenergymeasure[codecarbon]'",
            )
        )

    selected = probe_energy_sampler()
    if selected:
        lines.append(CheckLine("ok", f"Auto-selected sampler: {selected}"))
    else:
        lines.append(
            CheckLine(
                "warn",
                "No energy sampler available on this host",
                "install nvidia-ml-py (NVML) or run inside a GPU container",
            )
        )
    return HealthSection("Energy measurement", lines)


def _docker_section() -> HealthSection:
    lines: list[CheckLine] = []

    docker_cli = shutil.which("docker") is not None
    if docker_cli:
        lines.append(CheckLine("ok", "Docker CLI: found on PATH"))
    else:
        lines.append(
            CheckLine(
                "warn",
                "Docker CLI: not found on PATH",
                f"install Docker Engine - {DOCKER_INSTALL_URL}",
            )
        )

    daemon = _docker_daemon_running()
    if daemon is True:
        lines.append(CheckLine("ok", "Docker daemon: reachable"))
    elif daemon is False:
        lines.append(
            CheckLine(
                "warn",
                "Docker daemon: not reachable",
                "start the Docker daemon (e.g. `systemctl start docker`) and check permissions",
            )
        )
    # daemon is None -> CLI absent, already reported above.

    toolkit = any(shutil.which(tool) is not None for tool in NVIDIA_TOOLKIT_BINS)
    if toolkit:
        lines.append(CheckLine("ok", "NVIDIA Container Toolkit: detected"))
    else:
        lines.append(
            CheckLine(
                "warn",
                "NVIDIA Container Toolkit: not detected",
                f"install the NVIDIA Container Toolkit - {NVIDIA_TOOLKIT_INSTALL_URL}",
            )
        )
    return HealthSection("Docker", lines)


def _credentials_section() -> HealthSection:
    # Detect-and-advise only: NEVER print the token value.
    token = os.environ.get(ENV_HF_TOKEN)
    if token:
        line = CheckLine("ok", "HF_TOKEN: set (value hidden)")
    else:
        line = CheckLine(
            "warn",
            "HF_TOKEN: not set - gated models (e.g. Llama, Mistral) will fail to download",
            f"export HF_TOKEN=... (create one at {_HF_TOKEN_URL}) or add it to .env",
        )
    return HealthSection("Credentials", [line])


def _configuration_section(
    user_cfg: UserConfig | None,
    cfg_error: str | None,
    specs: dict[str, RunnerSpec],
) -> HealthSection:
    lines: list[CheckLine] = []
    config_path = get_user_config_path()

    if cfg_error is not None:
        lines.append(
            CheckLine(
                "fail",
                f"User config: invalid ({config_path})",
                cfg_error,
            )
        )
    elif config_path.exists():
        lines.append(CheckLine("ok", f"User config: {config_path} (loaded)"))
    else:
        lines.append(CheckLine("ok", "User config: none - using built-in defaults"))

    for engine, spec in specs.items():
        image = f", image={spec.image}" if spec.image else ""
        lines.append(CheckLine("ok", f"runner.{engine}: {spec.mode} (source={spec.source}{image})"))

    cfg = user_cfg if user_cfg is not None else UserConfig()
    lines.append(CheckLine("ok", f"energy_sampler: {cfg.measurement.energy_sampler}"))
    lines.append(
        CheckLine(
            "ok",
            "thermal gaps: experiment="
            f"{cfg.execution.experiment_gap_seconds:g}s, cycle={cfg.execution.cycle_gap_seconds:g}s",
        )
    )
    return HealthSection("Configuration", lines)


def _image_handshake_section(report: DoctorReport | None) -> HealthSection:
    if report is None:
        return HealthSection(
            "Image schema handshake",
            [
                CheckLine(
                    "warn",
                    "image schema handshake unavailable",
                    "install/start Docker to verify engine images against the host schema",
                )
            ],
        )

    lines: list[CheckLine] = []
    for row in report.results:
        status = _SCHEMA_TO_STATUS.get(row.status, "warn")
        message = f"{row.engine}: {row.status.value} ({row.image})"
        lines.append(CheckLine(status, message, row.detail or None))

    if report.skip_check_active:
        lines.append(
            CheckLine(
                "warn",
                "LLEM_SKIP_IMAGE_CHECK=1 is active - the runtime schema handshake is bypassed",
                "unset LLEM_SKIP_IMAGE_CHECK to re-enable the runtime image check",
            )
        )

    cache = report.trt_cache
    if cache is not None and cache.exists and cache.entry_count:
        from llenergymeasure.utils.formatting import format_bytes

        lines.append(
            CheckLine(
                "ok",
                f"TensorRT-LLM build cache: {cache.path} "
                f"({cache.entry_count} engine(s), {format_bytes(cache.total_bytes)})",
            )
        )

    lines.append(
        CheckLine(
            "ok", f"host llenergymeasure {report.host_pkg_version} (schema fingerprint present)"
        )
    )
    return HealthSection("Image schema handshake", lines)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _load_user_config_safe() -> tuple[UserConfig | None, str | None]:
    try:
        return load_user_config(), None
    except ConfigError as exc:
        return None, str(exc)


def _resolve_runner_specs(user_cfg: UserConfig | None) -> dict[str, RunnerSpec]:
    runners = user_cfg.runners if user_cfg is not None else None
    return {
        engine_str(engine): resolve_runner(engine_str(engine), user_config=runners)
        for engine in ENGINE_PACKAGES
    }


def _run_image_checks_safe() -> DoctorReport | None:
    try:
        return run_doctor_checks()
    except Exception:
        return None


def build_health_report() -> HealthReport:
    """Probe the host environment and return a full sectioned health report."""
    user_cfg, cfg_error = _load_user_config_safe()
    specs = _resolve_runner_specs(user_cfg)
    image_report = _run_image_checks_safe()

    sections = [
        _gpu_section(),
        _engines_section(specs),
        _energy_section(),
        _docker_section(),
        _credentials_section(),
        _configuration_section(user_cfg, cfg_error, specs),
        _image_handshake_section(image_report),
    ]
    return HealthReport(sections=sections, image_report=image_report)


def _image_report_to_dict(report: DoctorReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    cache = report.trt_cache
    return {
        "host_pkg_version": report.host_pkg_version,
        "host_fingerprint": report.host_fingerprint,
        "skip_check_active": report.skip_check_active,
        "results": [
            {
                "engine": row.engine,
                "image": row.image,
                "pkg_version": row.pkg_version,
                "image_fingerprint": row.image_fingerprint,
                "status": row.status.value,
                "local_present": row.local_present,
                "detail": row.detail,
            }
            for row in report.results
        ],
        "trt_cache": None
        if cache is None
        else {
            "path": cache.path,
            "exists": cache.exists,
            "entry_count": cache.entry_count,
            "total_bytes": cache.total_bytes,
            "clean_hint": cache.clean_hint,
        },
    }
