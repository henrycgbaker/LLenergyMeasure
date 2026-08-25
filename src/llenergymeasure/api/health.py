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
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from llenergymeasure.api import probe_energy_sampler
from llenergymeasure.api.doctor import DoctorReport, run_doctor_checks
from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.config.ssot import (
    ENGINE_PACKAGES,
    ENV_HF_TOKEN,
    RUNNER_CONTAINER,
    Engine,
    engine_str,
)
from llenergymeasure.config.user_config import (
    UserConfig,
    get_user_config_path,
    load_user_config,
)
from llenergymeasure.device.gpu_info import gpu_inventory
from llenergymeasure.harness.preflight import check_engine_installed

# Reuse the canonical host probes rather than re-implementing them (their home is
# docker_preflight; runner_resolution reuses the toolkit list too).
from llenergymeasure.infra.docker_preflight import (
    DOCKER_INSTALL_URL,
    NVIDIA_TOOLKIT_BINS,
    NVIDIA_TOOLKIT_INSTALL_URL,
    docker_daemon_reachable,
)
from llenergymeasure.infra.image_registry import get_default_image, image_present_locally
from llenergymeasure.infra.runner_resolution import is_docker_available, resolve_runner
from llenergymeasure.infra.version_handshake import SchemaStatus
from llenergymeasure.utils.exceptions import ConfigError

if TYPE_CHECKING:
    from llenergymeasure.config.precedence import ResolvedStudySettings

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
# one genuine failure (the CI-gating signal);
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
        counts = self.counts
        if counts["fail"]:
            return "fail"
        if counts["warn"]:
            return "warn"
        return "ok"

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
# Low-level probes
# ---------------------------------------------------------------------------


def _probe_engine_version(engine: str) -> str | None:
    """Return the installed version of *engine*'s host package, or None.

    Thin bridge over ``engines._observed.library_version`` (the SSOT for reading
    an installed library's ``__version__``), mapping its ``"unknown"`` sentinel
    back to None for this module's presence semantics.
    """
    from llenergymeasure.engines._observed import library_version

    version = library_version(ENGINE_PACKAGES[Engine(engine)])
    return None if version == "unknown" else version


def _availability_line(
    present: bool, ok_msg: str, warn_msg: str, fix: str | None = None
) -> CheckLine:
    """An ``ok`` line when *present*, else a ``warn`` line carrying *fix*."""
    if present:
        return CheckLine("ok", ok_msg)
    return CheckLine("warn", warn_msg, fix)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _gpu_section() -> HealthSection:
    lines: list[CheckLine] = []
    gpus, driver = gpu_inventory()
    if gpus:
        for i, gpu in enumerate(gpus):
            lines.append(CheckLine("ok", f"GPU {i}: {gpu['name']} ({gpu['vram_gb']:.1f} GB)"))
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
    importable = check_engine_installed(engine)
    if importable:
        version = _probe_engine_version(engine)
        installed = f"importable locally ({version})" if version else "importable locally"
        # Availability lives here; runner provenance (source=) belongs to the
        # Configuration section, so it is not repeated on this line.
        return CheckLine("ok", f"{engine}: {installed}; runner={spec.mode}")

    docker_avail = is_docker_available()
    if docker_avail and spec.mode == RUNNER_CONTAINER:
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
    selected = probe_energy_sampler()
    lines = [
        _availability_line(
            importlib.util.find_spec("pynvml") is not None,
            "NVML (nvidia-ml-py): available",
            "NVML (nvidia-ml-py): not installed",
            "pip install nvidia-ml-py",
        ),
        _availability_line(
            importlib.util.find_spec("zeus") is not None,
            "Zeus: available (higher-accuracy energy counter)",
            "Zeus: not installed (higher-accuracy energy counter)",
            "pip install 'llenergymeasure[zeus]'",
        ),
        _availability_line(
            importlib.util.find_spec("codecarbon") is not None,
            "CodeCarbon: available (fallback sampler)",
            "CodeCarbon: not installed (fallback sampler)",
            "pip install 'llenergymeasure[codecarbon]'",
        ),
        _availability_line(
            selected is not None,
            f"Auto-selected sampler: {selected}",
            "No energy sampler available on this host",
            "install nvidia-ml-py (NVML) or run inside a GPU container",
        ),
    ]
    return HealthSection("Energy measurement", lines)


def _docker_section() -> HealthSection:
    lines = [
        _availability_line(
            shutil.which("docker") is not None,
            "Docker CLI: found on PATH",
            "Docker CLI: not found on PATH",
            f"install Docker Engine - {DOCKER_INSTALL_URL}",
        )
    ]

    daemon = docker_daemon_reachable()
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

    lines.append(
        _availability_line(
            any(shutil.which(tool) is not None for tool in NVIDIA_TOOLKIT_BINS),
            "NVIDIA Container Toolkit: detected",
            "NVIDIA Container Toolkit: not detected",
            f"install the NVIDIA Container Toolkit - {NVIDIA_TOOLKIT_INSTALL_URL}",
        )
    )
    return HealthSection("Docker", lines)


def _credentials_section() -> HealthSection:
    # Detect-and-advise only: NEVER print the token value.
    line = _availability_line(
        bool(os.environ.get(ENV_HF_TOKEN)),
        "HF_TOKEN: set (value hidden)",
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
    if cfg.execution.gpu_indices is not None:
        lines.append(
            CheckLine(
                "ok",
                f"GPU allowlist: {cfg.execution.gpu_indices} (execution.gpu_indices) - "
                "llem places work and measurement only on these host devices",
            )
        )
    else:
        lines.append(CheckLine("ok", "GPU allowlist: none - all host GPUs permitted"))
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
        if row.shadows_default:
            lines.append(
                CheckLine(
                    "warn",
                    f"{row.engine}: local tag {row.image} shadows pinned default "
                    f"{row.shadows_default}",
                    f"docker rmi {row.image} to restore the pinned default, "
                    f"or pin an explicit image via runners.{row.engine}",
                )
            )

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


def _settings_without_a_study(user_cfg: UserConfig | None) -> ResolvedStudySettings:
    """Resolve the study-wide settings with no study file in play.

    What the diagnostics report on: the same precedence chain a run resolves
    through, minus the study layer, so an env var or user-config pin shows exactly
    what a run would pick up.
    """
    from llenergymeasure.config.precedence import resolve_study_settings

    return resolve_study_settings(
        study_output={},
        study_execution={},
        study_runners=None,
        study_images=None,
        user_config=user_cfg,
    )


def _resolve_runner_specs(user_cfg: UserConfig | None) -> dict[str, RunnerSpec]:
    """The runner each engine would resolve to, via the same chain a run uses."""
    from llenergymeasure.config.runner_spec import pins_from_resolved

    settings = _settings_without_a_study(user_cfg)
    pins = pins_from_resolved(settings.runners, settings.provenance, section="runners")
    return {
        engine_str(engine): resolve_runner(engine_str(engine), pins.get(engine_str(engine)))
        for engine in ENGINE_PACKAGES
    }


def show_image_resolution() -> None:
    """Print which Docker image each engine will resolve to.

    Shows the pin source (env var / user config) or local vs registry default for
    each engine. Used by ``make docker-images`` for quick diagnostics. Lives at
    the api resolution edge because it READS THE USER CONFIG: it derives the pins
    from the same precedence chain a run uses (no study file in play here), so an
    ``LLEM_IMAGE_<ENGINE>`` override shows exactly what a run would use.
    """
    from llenergymeasure.config.runner_spec import pins_from_resolved
    from llenergymeasure.config.ssot import ALL_ENGINES
    from llenergymeasure.infra.image_registry import resolve_image

    settings = _settings_without_a_study(load_user_config())
    pins = pins_from_resolved(settings.images, settings.provenance, section="images")
    print("=== Image resolution ===")
    for engine in sorted(ALL_ENGINES):
        image, source = resolve_image(engine, pin=pins.get(engine))
        print(f"  {engine:10s} -> {image}  ({source})")


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
                "shadows_default": row.shadows_default,
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
