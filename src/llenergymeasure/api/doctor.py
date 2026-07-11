"""Image-health doctor checks for the ``llem doctor`` command."""

from __future__ import annotations

from dataclasses import dataclass

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import Engine
from llenergymeasure.infra.image_registry import get_default_image, image_present_locally
from llenergymeasure.infra.version_handshake import (
    BundledEngineVersionMismatchError,
    SchemaStatus,
    classify_stamp,
    compute_expconf_fingerprint,
    inspect_image_stamp,
    rebuild_hint,
    skip_check_enabled,
)
from llenergymeasure.utils.exceptions import ConfigError

__all__ = [
    "DoctorReport",
    "EngineDoctorResult",
    "SchemaStatus",
    "run_doctor_checks",
]

SUPPORTED_ENGINES: tuple[Engine, ...] = tuple(Engine)

_DETAIL_FOR_STATUS: dict[SchemaStatus, str] = {
    SchemaStatus.OK: "",
    SchemaStatus.UNVERIFIED: "image predates schema-fingerprint label - rebuild to verify",
    SchemaStatus.UNREACHABLE: "no labels (image missing or built pre-handshake)",
}


@dataclass(frozen=True)
class EngineDoctorResult:
    """One row of the doctor table."""

    engine: str
    image: str
    pkg_version: str | None
    image_fingerprint: str | None
    status: SchemaStatus
    local_present: bool | None = None
    detail: str = ""


@dataclass(frozen=True)
class DoctorReport:
    """Full doctor output: per-engine rows plus host-side context."""

    host_pkg_version: str
    host_fingerprint: str
    skip_check_active: bool
    results: list[EngineDoctorResult]

    @property
    def any_mismatch(self) -> bool:
        return any(r.status is SchemaStatus.MISMATCH for r in self.results)


def _detail_for(engine: str, status: SchemaStatus) -> str:
    if status is SchemaStatus.MISMATCH:
        return f"rebuild: {rebuild_hint(engine)}"
    return _DETAIL_FOR_STATUS.get(status, "")


def run_doctor_checks(
    engines: tuple[str, ...] = SUPPORTED_ENGINES,
) -> DoctorReport:
    """Run image-health checks across *engines* and return a structured report.

    Image resolution follows ``get_default_image`` - local build first, then
    the per-engine default remote image (first-party GHCR for transformers,
    upstream ``vllm/vllm-openai`` / ``nvcr.io/nvidia/tensorrt-llm/release`` at
    the pinned engine version for vLLM/TRT). Each row also reports whether the
    resolved image is present in the local Docker cache. Unreachable images
    (docker not installed, no such tag, inspect timeout) become ``UNREACHABLE``
    rows rather than blowing up; an engine whose default cannot be resolved
    (broken wheel) becomes an ``UNREACHABLE`` row carrying the actionable fix.
    """
    host_fp = compute_expconf_fingerprint()
    results: list[EngineDoctorResult] = []

    for engine in engines:
        try:
            image = get_default_image(engine)
        except (ConfigError, BundledEngineVersionMismatchError) as exc:
            results.append(
                EngineDoctorResult(
                    engine=engine,
                    image="(unresolved)",
                    pkg_version=None,
                    image_fingerprint=None,
                    status=SchemaStatus.UNREACHABLE,
                    local_present=None,
                    detail=str(exc),
                )
            )
            continue

        local_present = image_present_locally(image)
        stamp = inspect_image_stamp(image)
        status = classify_stamp(stamp, host_fp)
        results.append(
            EngineDoctorResult(
                engine=engine,
                image=image,
                pkg_version=stamp.pkg_version,
                image_fingerprint=stamp.expconf_fingerprint,
                status=status,
                local_present=local_present,
                detail=_detail_for(engine, status),
            )
        )

    return DoctorReport(
        host_pkg_version=__version__,
        host_fingerprint=host_fp,
        skip_check_active=skip_check_enabled(),
        results=results,
    )
