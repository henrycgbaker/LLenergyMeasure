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
   ``src/llenergymeasure/engines/{engine}/`` were generated against
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
from pathlib import Path

from llenergymeasure.config.ssot import TIMEOUT_DOCKER_INSPECT
from llenergymeasure.utils.compat import StrEnum

__all__ = [
    "ENGINE_TO_IMPORT_MODULE",
    "ENV_SKIP_IMAGE_CHECK",
    "LABEL_IMAGE_VERSION",
    "LABEL_SCHEMA_FINGERPRINT",
    "ImageStamp",
    "SchemaStatus",
    "VersionMismatchError",
    "classify_engine_version",
    "classify_stamp",
    "compute_expconf_fingerprint",
    "inspect_image_stamp",
    "parse_image_stamp",
    "probe_image_engine_version",
    "read_ssot_engine_version",
    "rebuild_hint",
    "skip_check_enabled",
]

logger = logging.getLogger(__name__)

ENV_SKIP_IMAGE_CHECK = "LLEM_SKIP_IMAGE_CHECK"
LABEL_SCHEMA_FINGERPRINT = "llem.expconf.schema.fingerprint"
LABEL_IMAGE_VERSION = "org.opencontainers.image.version"

# Engine name -> Python module to import for ``__version__`` probing.
# TRT-LLM's pip distribution is ``tensorrt-llm`` but the import is
# ``tensorrt_llm``; vLLM's distribution and import name agree.
ENGINE_TO_IMPORT_MODULE = {
    "transformers": "transformers",
    "vllm": "vllm",
    "tensorrt": "tensorrt_llm",
}

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
    module_name = ENGINE_TO_IMPORT_MODULE.get(engine)
    if module_name is None:
        return None

    probe_code = (
        f"import {module_name}; print({_ENGINE_VERSION_MARKER!r} + str({module_name}.__version__))"
    )

    if engine == "tensorrt":
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        logger.debug("engine-version probe failed for %s/%s: %s", engine, image, exc)
        return None

    if result.returncode != 0:
        stderr_repr = result.stderr or ""
        if isinstance(stderr_repr, bytes):
            stderr_repr = stderr_repr.decode("utf-8", errors="replace")
        logger.debug(
            "engine-version probe exited %s for %s/%s; stderr=%s",
            result.returncode,
            engine,
            image,
            stderr_repr[:200],
        )
        return None

    # ``text=True`` should yield str, but tests routinely mock subprocess.run
    # and return MagicMock-shaped results with bytes stdout. Decode defensively
    # so both real and mocked subprocess shapes work.
    stdout = result.stdout
    if isinstance(stdout, bytes):
        stdout = stdout.decode("utf-8", errors="replace")

    for line in stdout.splitlines():
        if line.startswith(_ENGINE_VERSION_MARKER):
            return line[len(_ENGINE_VERSION_MARKER) :].strip() or None
    return None


def read_ssot_engine_version(engine: str) -> str | None:
    """Read ``engine_versions/{engine}.yaml::library.current_version``.

    Returns ``None`` if the file is absent, unreadable, or the field is
    missing. The SSOT lives in the repo root, resolved relative to this
    module's path so the function works regardless of cwd.
    """
    try:
        import yaml
    except ImportError:
        logger.debug("PyYAML not available; cannot read SSOT engine version")
        return None

    # version_handshake.py at src/llenergymeasure/infra/version_handshake.py;
    # repo root is three .parent hops up from __file__.
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    ssot_path = repo_root / "engine_versions" / f"{engine}.yaml"
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
