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
from llenergymeasure.utils.env_config import trt_build_cache_host_dir
from llenergymeasure.utils.exceptions import ConfigError

__all__ = [
    "DoctorReport",
    "EngineDoctorResult",
    "SchemaStatus",
    "TrtCacheHealth",
    "run_doctor_checks",
    "run_trt_cache_check",
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
class TrtCacheHealth:
    """TensorRT-LLM engine build-cache location and footprint.

    Informational only: the cache lifecycle is manual + visible, so this never
    affects the doctor exit code. llem NEVER auto-evicts entries; ``clean_hint``
    is the documented manual clean path.
    """

    path: str
    exists: bool
    entry_count: int
    total_bytes: int
    clean_hint: str


@dataclass(frozen=True)
class DoctorReport:
    """Full doctor output: per-engine rows plus host-side context."""

    host_pkg_version: str
    host_fingerprint: str
    skip_check_active: bool
    results: list[EngineDoctorResult]
    trt_cache: TrtCacheHealth | None = None

    @property
    def any_mismatch(self) -> bool:
        return any(r.status is SchemaStatus.MISMATCH for r in self.results)


def run_trt_cache_check() -> TrtCacheHealth:
    """Report the host TRT-LLM engine build-cache location, entries, and size.

    Host-side only (no container / GPU): stats the directory the docker runner
    bind-mounts as the build cache. Never raises - unreadable entries are
    skipped so a permission hiccup cannot break ``llem doctor``.
    """
    cache_dir = trt_build_cache_host_dir()
    # Entries are written by the container's root process, so a non-root host
    # user needs sudo to remove them - reflect that in the documented path.
    clean_hint = f"clean manually (llem never auto-evicts): sudo rm -rf {cache_dir}/engine-*"
    if not cache_dir.is_dir():
        return TrtCacheHealth(str(cache_dir), False, 0, 0, clean_hint)

    entry_count = 0
    total_bytes = 0
    try:
        entry_count = sum(1 for e in cache_dir.iterdir() if e.name.startswith("engine-"))
        for path in cache_dir.rglob("*"):
            try:
                if path.is_file() and not path.is_symlink():
                    total_bytes += path.stat().st_size
            except OSError:
                continue
    except OSError:
        pass

    return TrtCacheHealth(str(cache_dir), True, entry_count, total_bytes, clean_hint)


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
        trt_cache=run_trt_cache_check(),
    )
