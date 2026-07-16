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
   vendored rules and discovered schemas in
   ``src/llenergymeasure/engines/{engine}/`` were generated against
   (recorded in ``engine_versions/{engine}/current.yaml::library.current_version``).
   The probe is run only when the label-based check is inconclusive
   (no labels, or a literal ``"unknown"`` fingerprint left by a legacy
   build) so that label-based-OK paths stay cheap.

Bypass with ``LLEM_SKIP_IMAGE_CHECK=1`` when the skew is known harmless.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import platformdirs

from llenergymeasure.config.ssot import ENGINE_PACKAGES, TIMEOUT_DOCKER_INSPECT, Engine
from llenergymeasure.infra.image_registry import inspect_image
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
    "rebuild_hint",
    "resolve_image_engine_version",
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


def probe_image_engine_version(
    image: str,
    engine: str,
    *,
    timeout: float = _TIMEOUT_ENGINE_VERSION_PROBE_SECONDS,
) -> str | None:
    """Probe the engine library's ``__version__`` inside *image*.

    Runs ``docker run --rm`` against *image* with an inline ``import X;
    print(...)`` for the engine module mapped from *engine*. This is the
    cold probe: one container start (~60s on the TRT-LLM NGC image). It is
    intentionally uncached - :func:`resolve_image_engine_version` owns the
    caching tiers (an in-process memo and a persistent digest-keyed cache)
    so a repeat probe of the same image content never pays the container
    cost twice. Callers that want the cached path use that function; this
    one is the primitive it falls back to on a cache miss.

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


# ---------------------------------------------------------------------------
# Digest-keyed persistent probe cache
# ---------------------------------------------------------------------------
#
# The cold container probe above is a ~60s ``docker run`` per image, run once
# per engine per study at study start. Its answer is a pure function of the
# image *content* (the installed engine library version never changes without
# a new image), so it is safe to memoise keyed by the image's content digest.
#
# Three tiers, cheapest first:
#   1. in-process memo (this process, keyed by the image reference);
#   2. persistent on-disk cache keyed by the image DIGEST (survives across
#      processes / studies - the win the F2 plan asked for);
#   3. the cold container probe (:func:`probe_image_engine_version`).
#
# Invalidation is automatic and total: the key is the content digest, so a
# rebuilt or re-pulled image gets a fresh key and a fresh probe. Entries for a
# given digest never go stale - the same digest is byte-identical content, so
# its engine version cannot change. No TTL is needed or wanted.

_PROBE_CACHE_SUBDIR = "image-probe"
_PROBE_CACHE_FILENAME = "engine-versions.json"

# Sentinel distinguishing "cached as a real miss" from "not in cache". The
# cold probe can legitimately return None (probe failed); we do not persist
# that (a transient failure should be retried next run), so a None here always
# means "absent from the persistent cache".
_CACHE_ABSENT = object()

# Tier 1: in-process memo keyed by the image reference (not the digest, so a
# repeat call skips even the ``docker image inspect`` digest resolution).
_probe_memo: dict[tuple[str, str], str | None] = {}


def _probe_cache_path() -> Path:
    """Return the on-disk path of the persistent probe cache file.

    Lives under the same XDG cache root the docker runner uses for its deps
    cache (``platformdirs.user_cache_dir("llem")``), in an ``image-probe``
    subdirectory, so all llem host caches sit together.
    """
    return Path(platformdirs.user_cache_dir("llem")) / _PROBE_CACHE_SUBDIR / _PROBE_CACHE_FILENAME


def _resolve_image_digest(image: str, *, timeout: float = TIMEOUT_DOCKER_INSPECT) -> str | None:
    """Return the local content digest (``Id``) of *image*, or None.

    Reads ``docker image inspect``'s ``Id`` field (the image config digest,
    e.g. ``sha256:abc...``) via the local daemon. This is always present once
    an image is pulled and is stable per image content, which is exactly the
    cache-key property we want.

    Returns None when the digest cannot be resolved - docker missing, the
    inspect timing out, a non-zero exit (image not pulled yet), or malformed
    JSON. A None digest means the caller skips the persistent cache and falls
    back to the cold probe (which pulls the image if absent); it never crashes.
    RepoDigests (the registry digest) is deliberately not used: it is absent
    for locally-built images, whereas ``Id`` is always available locally.
    """
    result = inspect_image(image, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except (json.JSONDecodeError, ValueError):
        return None
    if not data:
        return None
    image_id = data[0].get("Id")
    return image_id if isinstance(image_id, str) and image_id else None


def _read_probe_cache() -> dict[str, dict[str, str]]:
    """Load the persistent probe cache, degrading to empty on any error.

    A missing, unreadable, or corrupt cache file yields an empty mapping so
    the caller re-probes rather than crashing - the file is a pure performance
    optimisation and is always safe to discard.
    """
    path = _probe_cache_path()
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return {}
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Probe cache at %s is corrupt; ignoring", path)
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _read_probe_cache_entry(digest: str, engine: str) -> object:
    """Return the cached version for (*digest*, *engine*), or ``_CACHE_ABSENT``."""
    by_engine = _read_probe_cache().get(digest)
    if not isinstance(by_engine, dict):
        return _CACHE_ABSENT
    version = by_engine.get(engine)
    return version if isinstance(version, str) else _CACHE_ABSENT


def _write_probe_cache_entry(digest: str, engine: str, version: str) -> None:
    """Persist (*digest*, *engine*) -> *version*, swallowing any write error.

    Read-merge-write with an atomic ``os.replace`` so a reader never sees a
    half-written file. No lock is taken: a concurrent writer racing on a
    different digest could clobber this entry, but the only consequence is a
    redundant re-probe next run (correctness is preserved), so the simplicity
    is worth it. Any failure (read-only cache dir, disk full) is logged at
    debug and swallowed - the probe result is already in hand.
    """
    path = _probe_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cache = _read_probe_cache()
        cache.setdefault(digest, {})[engine] = version
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(cache, handle, indent=2, sort_keys=True)
            os.replace(tmp_name, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise
    except OSError as exc:
        logger.debug("Could not persist probe cache to %s: %s", path, exc)


def resolve_image_engine_version(image: str, engine: str) -> str | None:
    """Return the engine library version inside *image*, using the cache tiers.

    This is the cached entry point that callers should use instead of
    :func:`probe_image_engine_version` directly. Tiers, cheapest first:

    1. **in-process memo** - a repeat call in the same process returns
       immediately (no docker calls at all);
    2. **persistent digest cache** - keyed by the image content digest; a hit
       skips the cold container probe entirely, even across processes /
       studies;
    3. **cold container probe** - :func:`probe_image_engine_version`, the
       ~60s ``docker run``, run only on a full miss. Its result is written to
       both caches (the persistent one only when a digest is resolvable and
       the probe actually succeeded).

    Never raises for cache reasons: an unresolvable digest, a corrupt cache
    file, or an unwritable cache dir all degrade to a fresh probe.
    """
    memo_key = (image, engine)
    if memo_key in _probe_memo:
        return _probe_memo[memo_key]

    digest = _resolve_image_digest(image)
    if digest is not None:
        cached = _read_probe_cache_entry(digest, engine)
        if cached is not _CACHE_ABSENT:
            assert isinstance(cached, str)
            _probe_memo[memo_key] = cached
            return cached

    version = probe_image_engine_version(image, engine)
    if digest is not None and version is not None:
        _write_probe_cache_entry(digest, engine, version)
    _probe_memo[memo_key] = version
    return version


class BundledEngineVersionMismatchError(RuntimeError):
    """Raised when the bundled rules and schema artefacts disagree on engine_version.

    Both artefacts are sourced from the same per-version
    ``engine_versions/<engine>/v<safe>/outputs/`` directory at wheel build
    time, so disagreement here indicates a build-time bundling bug
    (mismatched force-include paths, half-applied refresh, or one artefact
    re-mined without the other). Surfaces loud rather than picking one of
    the two arbitrarily as the runtime expectation.
    """


def read_bundled_engine_version(engine: str) -> str | None:
    """Return the engine_version field from the loaded rules + schema artefacts.

    The wheel ships per-engine machine artefacts whose envelope carries the
    engine_version they were mined for. This function reads through
    :class:`EngineRulesLoader` and :class:`SchemaLoader` (so the
    primary read goes via the same mechanism the runtime experiment path
    uses), cross-checks that both bundled artefacts agree on
    engine_version, and returns the version string.

    Returns ``None`` if either artefact is absent (e.g. wheel built before
    the engine was vendored). Raises
    :class:`BundledEngineVersionMismatchError` if the two bundled artefacts
    disagree - that disagreement only happens via a build-time bundling
    bug, never via a runtime configuration choice, so silent handling
    would mask the real issue.

    The bundled artefact is the runtime source of truth: it is in the wheel
    (so it works for installed users without an in-repo SSOT file), and its
    engine_version is the one llem actually applies at experiment time.
    """
    from llenergymeasure.config.engine_rules.loader import EngineRulesLoader
    from llenergymeasure.config.schema_loader import SchemaLoader

    try:
        rules = EngineRulesLoader().load_rules(engine)
    except FileNotFoundError as exc:
        logger.debug("Bundled rules read failed for %s: %s", engine, exc)
        return None
    try:
        schema = SchemaLoader().load_schema(engine)
    except (FileNotFoundError, ValueError) as exc:
        # ValueError covers SchemaLoader rejecting an unknown engine name.
        logger.debug("Bundled schema read failed for %s: %s", engine, exc)
        return None

    inv_version = rules.engine_version.strip()
    sch_version = schema.engine_version.strip()
    if inv_version != sch_version:
        raise BundledEngineVersionMismatchError(
            f"Bundled artefacts for {engine!r} disagree on engine_version: "
            f"rules envelope says {inv_version!r}, schema envelope says "
            f"{sch_version!r}. This indicates a build-time bundling bug - "
            f"both artefacts must be sourced from the same per-version "
            f"engine_versions/<engine>/v<safe>/outputs/ directory. Check "
            f"pyproject.toml's [tool.hatch.build.targets.wheel.force-include] "
            f"entries for the engine and rebuild the wheel."
        )
    return inv_version


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
    result = inspect_image(image, timeout=timeout)
    if result is None:
        logger.debug("docker image inspect failed for %s", image)
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
