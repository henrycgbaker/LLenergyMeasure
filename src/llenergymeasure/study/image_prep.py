"""Study-level Docker image preparation and schema-fingerprint verification.

``_ImageMixin`` holds the stateful image-prep methods mixed into
``StudyRunner``: it checks (or pulls) each Docker engine's image once at study
start, parses display metadata, and verifies the image's schema fingerprint
against the host before any experiments run. Two pure free functions
(``_sanitize_image_for_filename`` and ``_parse_image_metadata``) sit alongside
the mixin.

These methods read ``self._runner_specs`` and ``self._progress`` and set
``self._images_prepared``, all initialised in ``StudyRunner.__init__``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from llenergymeasure._version import __version__ as _HOST_PKG_VERSION
from llenergymeasure.config.ssot import (
    DOCKER_PULL_TIMEOUT,
    RUNNER_DOCKER,
    TIMEOUT_DOCKER_INSPECT,
)
from llenergymeasure.infra.image_registry import inspect_image
from llenergymeasure.infra.version_handshake import (
    ENV_SKIP_IMAGE_CHECK,
    LABEL_SCHEMA_FINGERPRINT,
    BundledEngineVersionMismatchError,
    ImageStamp,
    SchemaStatus,
    VersionMismatchError,
    classify_engine_version,
    classify_stamp,
    compute_expconf_fingerprint,
    parse_image_stamp,
    probe_image_engine_version,
    read_bundled_engine_version,
    rebuild_hint,
    skip_check_enabled,
)

if TYPE_CHECKING:
    from llenergymeasure.domain.progress import StudyProgressCallback
    from llenergymeasure.infra.runner_resolution import RunnerSpec

logger = logging.getLogger(__name__)


def _sanitize_image_for_filename(image: str) -> str:
    """Return a filename-safe slug for a Docker image tag.

    Replaces path separators, tag markers, digest markers, and whitespace with
    underscores. Clipped to 128 chars and stripped of trailing underscores so
    the resulting basename stays well under typical filesystem limits.
    """
    sanitized = image
    for ch in ("/", ":", "@", " ", "\t", "\n"):
        sanitized = sanitized.replace(ch, "_")
    sanitized = sanitized[:128].rstrip("_")
    return sanitized or "unknown"


def _parse_image_metadata(inspect_stdout: bytes) -> dict[str, str] | None:
    """Extract human-readable display metadata from ``docker image inspect`` JSON.

    Returns the id/size/built/layers fields the progress display renders
    inline. Schema-handshake labels are handled separately via
    ``parse_image_stamp`` so the raw 64-char fingerprint never leaks into
    the display metadata.
    """
    try:
        data = json.loads(inspect_stdout)
        if not data:
            return None
        info = data[0]
        meta: dict[str, str] = {}

        image_id = info.get("Id", "")
        if image_id.startswith("sha256:"):
            image_id = image_id[7:19]
        if image_id:
            meta["id"] = image_id

        size_bytes = info.get("Size", 0)
        if size_bytes:
            if size_bytes >= 1_073_741_824:
                meta["size"] = f"{size_bytes / 1_073_741_824:.1f} GB"
            else:
                meta["size"] = f"{size_bytes / 1_048_576:.0f} MB"

        created = info.get("Created", "")
        if created:
            try:
                created_dt = datetime.fromisoformat(created[:26].rstrip("Z"))
                created_dt = created_dt.replace(tzinfo=timezone.utc)
                age = datetime.now(timezone.utc) - created_dt
                if age.days > 0:
                    meta["built"] = f"{age.days}d ago"
                elif age.seconds >= 3600:
                    meta["built"] = f"{age.seconds // 3600}h ago"
                else:
                    meta["built"] = f"{age.seconds // 60}m ago"
            except (ValueError, TypeError):
                pass

        layers = info.get("RootFS", {}).get("Layers", [])
        if layers:
            meta["layers"] = str(len(layers))

        return meta if meta else None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None


class _ImageMixin:
    """Stateful Docker image-prep + fingerprint-verification methods for StudyRunner.

    Relies on attributes set up by ``StudyRunner.__init__``: ``_runner_specs``,
    ``_progress``, and ``_images_prepared``.
    """

    # Attributes provided by StudyRunner.__init__ (declared for the type checker).
    _runner_specs: dict[str, RunnerSpec] | None
    _progress: StudyProgressCallback | None
    _images_prepared: bool

    def _prepare_images(self) -> None:
        """Check/pull Docker images for all Docker engines before experiments.

        Runs once at the start of the study. Each engine's image is verified
        (or pulled) sequentially. On failure, raises so the study aborts early.
        Sets ``_images_prepared`` so per-experiment image_check is skipped.
        """

        if not self._runner_specs:
            return

        docker_engines = [
            (engine_name, spec)
            for engine_name, spec in self._runner_specs.items()
            if spec.mode == RUNNER_DOCKER and spec.image
        ]
        if not docker_engines:
            return

        if self._progress:
            self._progress.begin_image_prep([e for e, _ in docker_engines])

        for engine_name, spec in docker_engines:
            image = spec.image
            assert image is not None  # narrowing for type checker
            t0 = time.monotonic()

            # Check if image exists locally
            check = inspect_image(image, timeout=TIMEOUT_DOCKER_INSPECT)

            if check is not None and check.returncode == 0:
                elapsed = time.monotonic() - t0
                mismatch_error = self._finalise_image(
                    engine_name, image, check.stdout, cached=True, elapsed=elapsed
                )
                if mismatch_error is not None:
                    if self._progress:
                        self._progress.end_image_prep()
                    raise mismatch_error
                continue

            # Image not found locally - try to pull
            logger.info("Image %s not found locally, pulling...", image)
            try:
                pull = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                    timeout=DOCKER_PULL_TIMEOUT,
                )
            except subprocess.TimeoutExpired as exc:
                if self._progress:
                    self._progress.image_failed(engine_name, image, "pull timed out (30min)")
                    self._progress.end_image_prep()
                from llenergymeasure.infra.docker_errors import DockerImagePullError

                raise DockerImagePullError(
                    message=f"Image pull timed out: {image}",
                    fix_suggestion=f"docker compose build {engine_name}",
                ) from exc

            if pull.returncode != 0:
                tip = f"docker compose build {engine_name}"
                if self._progress:
                    self._progress.image_failed(engine_name, image, f"not found \u2014 run: {tip}")
                    self._progress.end_image_prep()
                from llenergymeasure.infra.docker_errors import DockerImagePullError

                raise DockerImagePullError(
                    message=f"Image not found: {image}",
                    fix_suggestion=tip,
                )

            elapsed = time.monotonic() - t0
            try:
                inspect = subprocess.run(
                    ["docker", "image", "inspect", image],
                    capture_output=True,
                    timeout=TIMEOUT_DOCKER_INSPECT,
                )
                inspect_stdout = inspect.stdout if inspect.returncode == 0 else b""
            except Exception:
                inspect_stdout = b""

            mismatch_error = self._finalise_image(
                engine_name, image, inspect_stdout, cached=False, elapsed=elapsed
            )
            if mismatch_error is not None:
                if self._progress:
                    self._progress.end_image_prep()
                raise mismatch_error

        if self._progress:
            self._progress.end_image_prep()

        self._images_prepared = True

    def _finalise_image(
        self,
        engine_name: str,
        image: str,
        inspect_stdout: bytes,
        *,
        cached: bool,
        elapsed: float,
    ) -> Exception | None:
        """Report an image-ready progress event and return any mismatch error.

        Parses display metadata and the schema stamp from *inspect_stdout*,
        classifies the schema status, renders it through
        ``progress.image_ready`` so the engine row appears before any abort,
        and returns the ``VersionMismatchError`` for the caller to raise.
        """
        metadata = _parse_image_metadata(inspect_stdout) or {}
        stamp = parse_image_stamp(inspect_stdout)
        schema_status, mismatch_error = self._verify_image_fingerprint(engine_name, image, stamp)
        metadata["schema"] = schema_status

        if self._progress:
            self._progress.image_ready(
                engine_name, image, cached=cached, elapsed=elapsed, metadata=metadata
            )
        return mismatch_error

    def _verify_image_fingerprint(
        self,
        engine_name: str,
        image: str,
        stamp: ImageStamp,
    ) -> tuple[str, Exception | None]:
        """Compare the host schema fingerprint to *stamp* and classify the result.

        Returns ``(schema_status, error)`` where ``schema_status`` is the
        short display string and ``error`` is a ``VersionMismatchError`` to
        raise (or ``None``). Callers raise after rendering so the offending
        engine appears in the terminal before the study aborts.

        Two-tier classification:

        1. **Label-based fingerprint check** (preferred when present).
           Definitive ``OK`` / ``MISMATCH`` answers are trusted directly.
        2. **Bundled engine-version probe** (fallback). When the label is
           absent or stamped ``"unknown"``, probe the engine library's
           ``__version__`` inside the image and compare against the
           ``engine_version`` envelope field on the bundled invariants +
           schema artefacts (read via :func:`read_bundled_engine_version`,
           which cross-checks both bundled artefacts agree). Closes the
           verification gap for upstream-direct images (vllm, tensorrt)
           which never had a llem schema-fingerprint label, and for legacy
           local images where the fingerprint was stamped as ``"unknown"``.
        """
        if skip_check_enabled():
            return "bypassed", None

        host_fp = compute_expconf_fingerprint()
        label_status = classify_stamp(stamp, host_fp)

        if label_status is SchemaStatus.OK:
            return "ok", None

        if label_status is SchemaStatus.MISMATCH:
            # Image has a real fingerprint label and it disagrees with the
            # host. Hard-error - the label is authoritative when present.
            assert stamp.expconf_fingerprint is not None
            error = VersionMismatchError(
                f"Docker image '{image}' was built from llenergymeasure "
                f"{stamp.pkg_version or 'unknown'} (schema {stamp.expconf_fingerprint[:12]}) "
                f"but the host is running {_HOST_PKG_VERSION} (schema {host_fp[:12]}). "
                f"The container will reject ExperimentConfig fields added on the host "
                f"after the image was built.\n\n"
                f"To fix:\n"
                f"  {rebuild_hint(engine_name)}       # rebuild locally\n"
                f"  make docker-pull                  # or pull a newer tagged release\n\n"
                f"If you're certain the skew is harmless, set {ENV_SKIP_IMAGE_CHECK}=1."
            )
            return "mismatch", error

        # label_status is UNVERIFIED or UNREACHABLE: fall back to engine-
        # version probe. The probe takes one docker-run (a few seconds)
        # and is cached per (image, engine). The "expected" version comes
        # from the bundled invariants/schema envelopes (the artefacts llem
        # actually applies at experiment time), not from current.yaml -
        # current.yaml isn't shipped in the wheel, so an SSOT-based check
        # is silently UNREACHABLE for installed users. The bundled envelope
        # works in both wheel and editable installs.
        probed = probe_image_engine_version(image, engine_name)
        try:
            expected = read_bundled_engine_version(engine_name)
        except BundledEngineVersionMismatchError as exc:
            return "mismatch (bundled artefact disagreement)", exc
        probe_status = classify_engine_version(probed, expected)

        if probe_status is SchemaStatus.OK:
            logger.debug(
                "Image %s verified via engine-version probe: %s matches bundled envelope",
                image,
                probed,
            )
            return f"ok (engine={probed})", None

        if probe_status is SchemaStatus.UNREACHABLE:
            # Neither label nor probe yielded a definitive answer. Soft-warn
            # and proceed - the bind-mounted framework on the host still
            # defines the actual contract; we just can't independently
            # verify the engine version.
            logger.warning(
                "Image %s has no %s label and the engine-version probe was "
                "inconclusive (probed=%r, bundled=%r). Dispatch will proceed; "
                "the bind-mounted framework defines the contract regardless.",
                image,
                LABEL_SCHEMA_FINGERPRINT,
                probed,
                expected,
            )
            return "unverified (no labels, probe inconclusive)", None

        # probe_status is MISMATCH
        error = VersionMismatchError(
            f"Docker image '{image}' has {engine_name} library version "
            f"{probed!r} but the bundled invariants + discovered schemas "
            f"in this wheel were mined against {expected!r}. Applying "
            f"version-{expected!r} validation rules to a version-{probed!r} "
            f"substrate is not safe.\n\n"
            f"To fix:\n"
            f"  Pull an image matching {expected!r}, or install a wheel "
            f"built against {probed!r}.\n\n"
            f"If you're certain the skew is harmless, set "
            f"{ENV_SKIP_IMAGE_CHECK}=1."
        )
        return f"mismatch (engine {probed} vs bundled {expected})", error
