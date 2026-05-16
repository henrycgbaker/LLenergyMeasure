"""Host/container schema-skew detection via OCI image labels + SSOT engine
version probe.

Two-tier handshake:

1. **Label-based fingerprint check** (when present). Images that we build
   ourselves are stamped with ``org.opencontainers.image.version`` (the
   llenergymeasure package version, for display) and
   ``llem.expconf.schema.fingerprint`` (a SHA-256 over
   ``ExperimentConfig.model_json_schema()``). The host computes its own
   fingerprint at runtime and compares it to the label. This protects
   against subtle ExperimentConfig schema drift when the framework is
   baked into the image.

2. **SSOT engine-version probe** (fallback). Under the current dispatch
   architecture (bind-mount canonical), framework code lives on the host
   and is mounted into the container at run time, so the label check is
   tautological for upstream-direct images that have no label. The
   meaningful drift surface is whether the engine library installed in
   the image (e.g. ``vllm.__version__``) matches the version that our
   vendored invariants and discovered schemas in
   ``engine_versions/{engine}/v<current>/outputs/`` were generated against
   (recorded in ``engine_versions/{engine}.yaml::library.current_version``).
   The probe is run only when the label-based check is inconclusive
   (no labels, or a literal ``"unknown"`` fingerprint left by a legacy
   build) so that label-based-OK paths stay cheap.

Bypass with ``LLEM_SKIP_IMAGE_CHECK=1`` when the skew is known harmless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass
from functools import cache, lru_cache

from llenergymeasure.config.ssot import ENGINE_PACKAGES, TIMEOUT_DOCKER_INSPECT, Engine
from llenergymeasure.utils.compat import StrEnum

__all__ = [
    "ENV_SKIP_IMAGE_CHECK",
    "LABEL_IMAGE_VERSION",
    "LABEL_SCHEMA_FINGERPRINT",
    "BundledEngineVersionMismatchError",
    "ImageStamp",
    "SchemaStatus",
    "VersionMismatchError",
    "classify_engine_version",
    "classify_stamp",
    "compute_expconf_fingerprint",
    "inspect_image_stamp",
    "parse_image_stamp",
    "probe_image_engine_version",
    "read_bundled_engine_version",
    "read_ssot_engine_version",
    "rebuild_hint",
    "skip_check_enabled",
]

logger = logging.getLogger(__name__)

ENV_SKIP_IMAGE_CHECK = "LLEM_SKIP_IMAGE_CHECK"
LABEL_SCHEMA_FINGERPRINT = "llem.expconf.schema.fingerprint"
LABEL_IMAGE_VERSION = "org.opencontainers.image.version"

# Soft cap on how long the engine-version probe is allowed to take per
# image. The probe is one ``docker run`` (with the engine library import
# inside) so this is enough headroom for container start + Python import,
# even on TRT-LLM's NGC image with its banner / CUDA compat layer setup.
_TIMEOUT_ENGINE_VERSION_PROBE_SECONDS = 60.0

_ENGINE_VERSION_MARKER = "---LLEM_VER:"


class VersionMismatchError(RuntimeError):
    """Raised when a Docker image's schema fingerprint differs from the host's."""


@dataclass(frozen=True)
class ImageStamp:
    """OCI labels relevant to the schema handshake, pulled from a Docker image."""

    pkg_version: str | None
    expconf_fingerprint: str | None


_EMPTY_STAMP = ImageStamp(pkg_version=None, expconf_fingerprint=None)


class SchemaStatus(StrEnum):
    """Outcome of comparing a Docker image's stamp to the host fingerprint."""

    OK = "OK"
    MISMATCH = "MISMATCH"
    UNVERIFIED = "UNVERIFIED"
    UNREACHABLE = "UNREACHABLE"
    BYPASSED = "BYPASSED"


@cache
def compute_expconf_fingerprint() -> str:
    """Return the SHA-256 hex digest of the ExperimentConfig JSON schema.

    The schema is frozen per-process, so the result is memoised. Callers
    typically display the first 12 hex characters for readability.
    """
    from llenergymeasure.config.introspection import get_experiment_config_schema

    payload = json.dumps(
        get_experiment_config_schema(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def classify_stamp(stamp: ImageStamp, host_fingerprint: str) -> SchemaStatus:
    """Classify an image stamp against the host fingerprint.

    Pure: does not read environment or log. Callers that honour the
    ``LLEM_SKIP_IMAGE_CHECK`` bypass should short-circuit to
    :attr:`SchemaStatus.BYPASSED` themselves before calling this.

    A literal ``"unknown"`` fingerprint is treated as
    :attr:`SchemaStatus.UNVERIFIED`, not :attr:`SchemaStatus.MISMATCH` -
    it indicates the image was built before the build-time hash
    computation was wired up (typical for legacy local dev rebuilds) and
    is not a positive disagreement signal. Callers that want a stronger
    answer in this case should fall back to
    :func:`probe_image_engine_version`.
    """
    if stamp.expconf_fingerprint is None and stamp.pkg_version is None:
        return SchemaStatus.UNREACHABLE
    if stamp.expconf_fingerprint is None or stamp.expconf_fingerprint == "unknown":
        return SchemaStatus.UNVERIFIED
    if stamp.expconf_fingerprint == host_fingerprint:
        return SchemaStatus.OK
    return SchemaStatus.MISMATCH


@lru_cache(maxsize=32)
def probe_image_engine_version(
    image: str,
    engine: str,
    *,
    timeout: float = _TIMEOUT_ENGINE_VERSION_PROBE_SECONDS,
) -> str | None:
    """Probe the engine library's ``__version__`` inside *image*.

    Runs ``docker run --rm`` against *image* with an inline ``import X;
    print(...)`` for the engine module mapped from *engine*. The result
    is cached per (image, engine) because an image's content is fixed by
    its digest - the version inside doesn't change without a new tag.

    Engine-conditional entrypoint: TRT-LLM needs
    ``/opt/nvidia/nvidia_entrypoint.sh`` to set up ``LD_LIBRARY_PATH``
    before ``import tensorrt_llm`` succeeds.

    Returns the parsed version string (e.g. ``"0.7.3"``), or ``None`` if
    the probe fails: docker not installed, image missing, engine module
    unimportable, timeout, or output unparseable. Callers should treat
    ``None`` the same as a missing label - unverified, not mismatch.
    """
    try:
        module_name = ENGINE_PACKAGES[Engine(engine)]
    except (KeyError, ValueError):
        return None

    probe_code = (
        f"import {module_name}; print({_ENGINE_VERSION_MARKER!r} + str({module_name}.__version__))"
    )

    if engine == Engine.TENSORRT:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "/opt/nvidia/nvidia_entrypoint.sh",
            image,
            "python3",
            "-c",
            probe_code,
        ]
    else:
        cmd = [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            "-c",
            probe_code,
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=timeout, text=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("engine-version probe failed for %s/%s: %s", engine, image, exc)
        return None

    if result.returncode != 0:
        logger.debug(
            "engine-version probe exited %s for %s/%s; stderr=%s",
            result.returncode,
            engine,
            image,
            (result.stderr or "")[:200],
        )
        return None

    for line in result.stdout.splitlines():
        if line.startswith(_ENGINE_VERSION_MARKER):
            return line[len(_ENGINE_VERSION_MARKER) :].strip() or None
    return None


class BundledEngineVersionMismatchError(RuntimeError):
    """Raised when the bundled invariants and schema artefacts disagree on engine_version.

    Both artefacts are sourced from the same per-version
    ``engine_versions/<engine>/v<safe>/outputs/`` directory at wheel build
    time, so disagreement here indicates a build-time bundling bug
    (mismatched force-include paths, half-applied refresh, or one artefact
    re-mined without the other). Surfaces loud rather than picking one of
    the two arbitrarily as the runtime expectation.
    """


def read_bundled_engine_version(engine: str) -> str | None:
    """Return the engine_version field from the loaded invariants + schema artefacts.

    The wheel ships per-engine machine artefacts whose envelope carries the
    engine_version they were mined for. This function reads through
    :class:`EngineInvariantsLoader` and :class:`SchemaLoader` (so the
    primary read goes via the same mechanism the runtime experiment path
    uses), cross-checks that both bundled artefacts agree on
    engine_version, and returns the version string.

    Returns ``None`` if either artefact is absent (e.g. wheel built before
    the engine was vendored). Raises
    :class:`BundledEngineVersionMismatchError` if the two bundled artefacts
    disagree - that disagreement only happens via a build-time bundling
    bug, never via a runtime configuration choice, so silent handling
    would mask the real issue.

    Preferred over :func:`read_ssot_engine_version` for the runtime
    handshake because the bundled artefact is in the wheel (so it works
    for installed users without an in-repo SSOT file), and because the
    artefact's engine_version is the one llem actually applies at
    experiment time - the SSOT version describes the build's INTENT;
    the bundled envelope describes what shipped.
    """
    from llenergymeasure.config.engine_invariants.loader import EngineInvariantsLoader
    from llenergymeasure.config.schema_loader import SchemaLoader

    try:
        invariants = EngineInvariantsLoader().load_invariants(engine)
    except FileNotFoundError as exc:
        logger.debug("Bundled invariants read failed for %s: %s", engine, exc)
        return None
    try:
        schema = SchemaLoader().load_schema(engine)
    except (FileNotFoundError, ValueError) as exc:
        # ValueError covers SchemaLoader rejecting an unknown engine name.
        logger.debug("Bundled schema read failed for %s: %s", engine, exc)
        return None

    inv_version = (invariants.engine_version or "").strip()
    sch_version = (schema.engine_version or "").strip()
    if not inv_version and not sch_version:
        return None
    if inv_version != sch_version:
        raise BundledEngineVersionMismatchError(
            f"Bundled artefacts for {engine!r} disagree on engine_version: "
            f"invariants envelope says {inv_version!r}, schema envelope says "
            f"{sch_version!r}. This indicates a build-time bundling bug - "
            f"both artefacts must be sourced from the same per-version "
            f"engine_versions/<engine>/v<safe>/outputs/ directory. Check "
            f"pyproject.toml's [tool.hatch.build.targets.wheel.force-include] "
            f"entries for the engine and rebuild the wheel."
        )
    return inv_version


def read_ssot_engine_version(engine: str) -> str | None:
    """Read ``engine_versions/{engine}/current.yaml::library.current_version``.

    Legacy/dev-mode helper. Use :func:`read_bundled_engine_version` for the
    runtime handshake - it reads from the bundled wheel artefact (works
    for installed users, no dependency on an in-repo
    ``engine_versions/`` tree) and cross-checks invariants/schema agreement.

    This SSOT-file reader returns ``None`` for wheel-installed users
    (engine_versions/ is not in the wheel), so it can't drive the
    runtime handshake on its own. Retained for in-repo dev tooling and
    for the parity guard at ``scripts/ci/check_engine_bundle_pin.sh``.

    Returns ``None`` if the file is absent, unreadable, or the field is
    missing. The repo root is resolved via
    :func:`llenergymeasure.infra.docker_runner._resolve_repo_root`, which
    is the single source of truth for repo-shape paths.
    """
    import yaml

    from llenergymeasure.infra.docker_runner import _resolve_repo_root

    ssot_path = _resolve_repo_root() / "engine_versions" / engine / "current.yaml"
    try:
        with open(ssot_path) as f:
            data = yaml.safe_load(f) or {}
    except (FileNotFoundError, OSError) as exc:
        logger.debug("SSOT read failed for %s at %s: %s", engine, ssot_path, exc)
        return None

    version = data.get("library", {}).get("current_version")
    if not isinstance(version, str) or not version:
        return None
    return version


def classify_engine_version(
    probed: str | None,
    expected: str | None,
) -> SchemaStatus:
    """Compare a probed engine version string to the SSOT expectation.

    Returns :attr:`SchemaStatus.UNREACHABLE` if either side is missing,
    :attr:`SchemaStatus.OK` on match (after stripping a leading ``v``
    prefix, since some images stamp ``v0.7.3`` against an SSOT
    ``0.7.3``), :attr:`SchemaStatus.MISMATCH` otherwise.
    """
    if probed is None or expected is None:
        return SchemaStatus.UNREACHABLE
    if probed.lstrip("v") == expected.lstrip("v"):
        return SchemaStatus.OK
    return SchemaStatus.MISMATCH


def inspect_image_stamp(image: str, *, timeout: float = TIMEOUT_DOCKER_INSPECT) -> ImageStamp:
    """Parse handshake labels from ``docker image inspect`` on *image*.

    Returns an empty stamp on any failure (docker not installed, inspect
    timeout, JSON parse error, missing labels). The caller decides whether the
    absence of labels is a warning or an error.
    """
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("docker image inspect failed for %s: %s", image, exc)
        return _EMPTY_STAMP

    if result.returncode != 0:
        logger.debug(
            "docker image inspect returned %s for %s: %s",
            result.returncode,
            image,
            result.stderr.decode("utf-8", errors="replace") if result.stderr else "",
        )
        return _EMPTY_STAMP

    return parse_image_stamp(result.stdout)


def parse_image_stamp(inspect_stdout: bytes) -> ImageStamp:
    """Extract an ``ImageStamp`` from raw ``docker image inspect`` JSON."""
    try:
        data = json.loads(inspect_stdout)
    except (json.JSONDecodeError, ValueError):
        return _EMPTY_STAMP
    if not data:
        return _EMPTY_STAMP
    labels = data[0].get("Config", {}).get("Labels") or {}
    return ImageStamp(
        pkg_version=labels.get(LABEL_IMAGE_VERSION),
        expconf_fingerprint=labels.get(LABEL_SCHEMA_FINGERPRINT),
    )


def rebuild_hint(engine: str) -> str:
    """Return the user-facing rebuild command for *engine*."""
    return f"make docker-build-{engine}"


def skip_check_enabled() -> bool:
    """True iff ``LLEM_SKIP_IMAGE_CHECK`` is set to a truthy value."""
    from llenergymeasure.utils.env_config import parse_bool_env

    return parse_bool_env(ENV_SKIP_IMAGE_CHECK)
