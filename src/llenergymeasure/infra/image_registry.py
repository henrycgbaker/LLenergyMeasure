"""Docker image resolution for engine containers.

Per-engine image sources
-------------------------
**Transformers** uses a first-party image published by CI on release tags
(``ghcr.io/henrycgbaker/llenergymeasure/transformers:v{package_version}``). It
is the only engine with a project-built image (its flash-attention kernel
compile has no upstream equivalent).

**vLLM and TensorRT-LLM** run inside the canonical UPSTREAM images
(``vllm/vllm-openai:v{engine_version}`` and
``nvcr.io/nvidia/tensorrt-llm/release:{engine_version}``) with the
llenergymeasure source bind-mounted at run time. There is no first-party GHCR
image for these engines, so their default must point at the upstream ref at the
pinned engine version - not a GHCR tag CI never publishes.

**Local images** (``llenergymeasure:{engine}``) produced by
``docker compose build`` / ``make docker-build-all`` are preferred when present
for fast local iteration.

``get_default_image(engine)`` checks for a local image first, then resolves the
per-engine default remote image. The engine version that tags the vLLM/TRT
upstream images is read from the wheel-shipped artefact envelope
(``version_handshake.read_bundled_engine_version``), which CI keeps equal to
``engine_versions/<engine>/current.yaml`` via
``scripts/check_discovered_schema_versions.py``. When that version cannot be
resolved (a partial or broken wheel), resolution raises an actionable
``ConfigError`` rather than emitting a tag that would 404 on ``docker pull``.

Overriding the image
--------------------
The ``runners:`` section in the study YAML accepts explicit image references::

    runners:
      transformers: local               # host execution (no Docker)
      vllm: docker                      # default resolution (local -> upstream)
      tensorrt: "docker:my/custom:tag"  # explicit image override

``parse_runner_value()`` converts these into a ``(runner_type, image_override)``
tuple consumed by the runner resolution chain.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from functools import cache, lru_cache

from llenergymeasure.config.ssot import (
    ALL_ENGINES,
    ENGINES,
    ENV_IMAGE_PREFIX,
    RUNNER_DOCKER,
    RUNNER_LOCAL,
    TIMEOUT_DOCKER_CLI,
    Engine,
    RunnerMode,
    engine_str,
)
from llenergymeasure.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_IMAGE_TEMPLATES",
    "get_default_image",
    "image_present_locally",
    "local_image_for",
    "parse_runner_value",
    "resolve_image",
    "resolve_image_digest",
    "shadowed_default_image",
    "show_image_resolution",
]

# ---------------------------------------------------------------------------
# Per-engine default image templates
# ---------------------------------------------------------------------------

# ``{version}`` is filled at runtime (see ``get_default_image``). Transformers
# is a first-party GHCR image tagged with the llenergymeasure package version
# (published by docker-publish.yml); vLLM and TensorRT-LLM are canonical
# upstream images tagged with the pinned engine version. The vLLM tag carries a
# ``v`` prefix and the NGC TRT-LLM tag does not - matching the refs the
# schema/rules producers pull (scripts/refresh_discovered_schemas.sh).
DEFAULT_IMAGE_TEMPLATES: dict[str, str] = {
    Engine.TRANSFORMERS.value: "ghcr.io/henrycgbaker/llenergymeasure/transformers:v{version}",
    Engine.VLLM.value: "vllm/vllm-openai:v{version}",
    Engine.TENSORRT.value: "nvcr.io/nvidia/tensorrt-llm/release:{version}",
}

# Local image tag produced by `docker compose build` (no registry prefix).
LOCAL_IMAGE_TEMPLATE = "llenergymeasure:{engine}"


def local_image_for(engine: str) -> str:
    """Return the local image tag (``llenergymeasure:{engine}``) for *engine*.

    Produced by ``docker compose build`` / ``make docker-build-all``; the single
    accessor for the tag every resolution path checks (and compares against) when
    a locally-built image is preferred.
    """
    return LOCAL_IMAGE_TEMPLATE.format(engine=engine_str(engine))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_default_image(engine: str) -> str:
    """Resolve the default Docker image for *engine*.

    Resolution order:

    1. **Local image** (``llenergymeasure:{engine}``): produced by
       ``docker compose build`` / ``make docker-build-all``. Always reflects
       current source code. Preferred for local development iteration.
    2. **Per-engine default remote image**:

       - transformers -> first-party GHCR image at the package version
         (``ghcr.io/.../transformers:v{package_version}``);
       - vllm -> upstream ``vllm/vllm-openai:v{engine_version}``;
       - tensorrt -> upstream
         ``nvcr.io/nvidia/tensorrt-llm/release:{engine_version}``.

    The engine version tagging the vLLM/TRT upstream images comes from the
    wheel-shipped artefact envelope, kept equal to the ``current.yaml`` pin by
    CI. To force a specific image, use ``runners: {engine}: "docker:<image:tag>"``
    in the study YAML.

    Args:
        engine: Engine name, e.g. ``"vllm"``, ``"transformers"``, ``"tensorrt"``.

    Returns:
        Image reference string, e.g. ``"llenergymeasure:vllm"`` (local) or
        ``"vllm/vllm-openai:v0.19.1"`` (upstream).

    Raises:
        ConfigError: The engine is unknown, or its pinned upstream version is
            unavailable at runtime (a partial or broken wheel). The message
            tells the user to set ``runners.<engine>`` to an explicit
            ``docker:<image>`` reference.
    """
    # 1. Prefer a locally-built image (fast local iteration). This precedence is
    #    intentional, but a months-stale dev tag can hijack resolution
    #    invisibly, so warn (once) and name the pinned default it bypasses.
    local_image = local_image_for(engine)
    if _image_exists_locally(local_image):
        _warn_local_shadow(engine)
        return local_image

    # 2. Per-engine default remote image at the resolved version.
    image = _resolve_pinned_default(engine_str(engine))
    logger.info("No local image found; using default image %s", image)
    return image


def _default_image_version(engine: str) -> str:
    """Return the version that tags *engine*'s default remote image.

    Transformers uses the llenergymeasure package version (its GHCR image is
    published per release). vLLM and TensorRT-LLM use the pinned ENGINE version
    read from the wheel-shipped artefact envelope - the same version the runtime
    schema handshake uses, kept equal to
    ``engine_versions/<engine>/current.yaml`` by
    ``scripts/check_discovered_schema_versions.py``.

    Raises:
        ConfigError: The pinned engine version is unavailable at runtime.
    """
    if ENGINES[Engine(engine)].image_version_source == "package":
        from llenergymeasure._version import __version__

        return __version__ if __version__ else "latest"

    # Lazy import: version_handshake imports this module at top level, so a
    # module-level import here would be circular.
    from llenergymeasure.infra.version_handshake import read_bundled_engine_version

    version = read_bundled_engine_version(engine)
    if not version:
        raise ConfigError(
            f"Cannot resolve a default Docker image for engine {engine!r}: its "
            f"pinned engine version is unavailable at runtime (the bundled rules "
            f"and schema artefacts are missing or disagree). Set "
            f'runners.{engine} to "docker:<image>:<tag>" in your study YAML to '
            f"pin an explicit image."
        )
    return version


def _resolve_pinned_default(engine: str) -> str:
    """Resolve *engine*'s version-pinned default remote image.

    Shared by ``get_default_image`` (step 2) and the shadow-warning path so the
    warning always names the exact image a local bare tag bypassed.

    Raises:
        ConfigError: the engine has no default template, or its pinned engine
            version is unavailable at runtime (a partial or broken wheel).
    """
    template = DEFAULT_IMAGE_TEMPLATES.get(engine)
    if template is None:
        raise ConfigError(
            f"No default Docker image is defined for engine {engine!r}. "
            f'Set runners.{engine} to "docker:<image>:<tag>" in your study '
            f"YAML to pin an explicit image."
        )
    return template.format(version=_default_image_version(engine))


@cache
def _pinned_default_or_none(engine: str) -> str | None:
    """Best-effort name of the version-pinned default a local bare tag bypasses.

    Returns None when it cannot be resolved (unknown engine or a broken wheel),
    so naming the bypassed default never fails a run.

    Cached on the engine so the shadow warning and ``shadowed_default_image``
    (both called per engine within ``run_doctor_checks``) resolve the bypassed
    default once, sharing a single file read instead of two. ``cache_clear()``
    resets the memoization (used by tests).
    """
    from llenergymeasure.infra.version_handshake import BundledEngineVersionMismatchError

    try:
        return _resolve_pinned_default(engine_str(engine))
    except (ConfigError, BundledEngineVersionMismatchError):
        return None


@cache
def _warn_local_shadow(engine: str) -> None:
    """Warn (once per process) that a local bare tag shadows the pinned default.

    ``get_default_image`` runs repeatedly per engine within a single study prep
    (preflight, then once per experiment and cycle), so the warning is
    deduplicated with ``functools.cache`` keyed on the engine: the same shadow is
    logged only once. ``cache_clear()`` resets the dedup (used by tests).
    """
    local_image = local_image_for(engine)
    pinned_default = _pinned_default_or_none(engine)
    shadowed = (
        f"the version-pinned default {pinned_default}"
        if pinned_default is not None
        else "this engine's version-pinned default"
    )
    logger.warning(
        "Using local Docker image %s, which shadows %s. A stale local tag can "
        "silently win image resolution. Run `docker rmi %s` to restore the "
        "pinned default, or pin an explicit image via runners.<engine> or "
        "LLEM_IMAGE_<ENGINE>.",
        local_image,
        shadowed,
        local_image,
    )


def shadowed_default_image(engine: str, resolved_image: str) -> str | None:
    """Return the version-pinned default *resolved_image* bypasses, or None.

    When *resolved_image* is the local bare tag for *engine* (a locally built
    image won resolution), return the version-pinned default it shadows;
    otherwise None. Lets diagnostics (``llem doctor``) surface the same
    shadowing fact the resolution warning logs. Best-effort: returns None when
    the pinned default cannot be resolved.
    """
    if resolved_image != local_image_for(engine):
        return None
    return _pinned_default_or_none(engine)


def image_present_locally(image: str) -> bool:
    """Return True iff *image* is present in the local Docker image cache.

    Thin public wrapper over the cached inspect check, for diagnostics (e.g.
    ``llem doctor``) that report whether a resolved image reference is already
    pulled locally.
    """
    return _image_exists_locally(image)


def inspect_image(image: str, *, timeout: float) -> subprocess.CompletedProcess[bytes] | None:
    """Run ``docker image inspect <image>``; return the result or None on failure.

    Returns None when docker is missing, the call times out, or the OS rejects
    it (``FileNotFoundError`` / ``TimeoutExpired`` / ``OSError``). A non-zero
    return code is NOT a failure here - the ``CompletedProcess`` is returned and
    the caller inspects ``returncode`` / ``stdout``.
    """
    try:
        return subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


@lru_cache(maxsize=8)
def _image_exists_locally(image: str) -> bool:
    """Check whether a Docker image tag exists in the local cache."""
    result = inspect_image(image, timeout=TIMEOUT_DOCKER_CLI)
    return result is not None and result.returncode == 0


def _inspect_first_dict(image: str, *, timeout: float) -> dict[str, object] | None:
    """Return the first record from ``docker image inspect <image>``, or None.

    Shared skeleton for the field extractors that read a single ``docker image
    inspect`` object (``resolve_image_digest`` here reads ``RepoDigests``;
    ``version_handshake._resolve_image_digest`` reads ``Id``). Requires a zero
    exit and a parseable JSON array whose first element is an object; returns
    None (never raises) on any failure - docker missing, non-zero exit,
    malformed/empty JSON - so callers extract their field from a known-good dict.
    ``json.JSONDecodeError`` subclasses ``ValueError``, so the one except covers both.
    """
    result = inspect_image(image, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return None
    if not data or not isinstance(data[0], dict):
        return None
    return data[0]


@lru_cache(maxsize=8)
def resolve_image_digest(image: str) -> str | None:
    """Return the registry digest of *image* (``repo@sha256:...``), or None.

    Reads ``docker image inspect``'s first ``RepoDigests`` entry - the registry
    content digest that pins the full image (base image, CUDA, torch, patches)
    and is portable across hosts, making it the reproducibility anchor recorded
    in ``system.json``.

    Returns None (never raises) when the digest cannot be resolved: docker
    missing, the image not pulled, a locally-built image with no registry
    digest, an inspect timeout / non-zero exit, or malformed JSON. Callers
    record the None verbatim so a run is never failed by digest resolution.

    RepoDigests (not the local ``Id`` config digest used by the version probe
    cache) is used deliberately: cross-host reproducibility needs the registry
    digest, and locally-built images without one honestly resolve to None.

    Cached (like the sibling ``_image_exists_locally``): an image's digest is
    invariant for the process lifetime, so the per-experiment-per-cycle call
    resolves it once. A cached None is retried only next process, matching the
    sibling's accepted trade-off.
    """
    record = _inspect_first_dict(image, timeout=TIMEOUT_DOCKER_CLI)
    if record is None:
        return None
    repo_digests = record.get("RepoDigests")
    if isinstance(repo_digests, list) and repo_digests:
        first = repo_digests[0]
        if isinstance(first, str) and first:
            return first
    return None


def resolve_image(
    engine: str,
    *,
    spec_image: str | None = None,
    yaml_images: dict[str, str] | None = None,
    user_config_images: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve the Docker image for *engine* using the full precedence chain.

    This is the **image axis** of the orthogonal runner/image resolution system.
    The runner axis (local vs docker) is handled by ``resolve_runner()`` in
    ``runner_resolution.py``.

    Precedence (highest to lowest):

    1. ``LLEM_IMAGE_{ENGINE}`` env var (from shell or ``.env`` file)
    2. Study YAML ``images:`` section
    3. Explicit image from runner spec (``docker:<image>`` shorthand)
    4. User config ``images:`` section
    5. Smart default: local image → registry fallback

    Args:
        engine:              Engine name (e.g. ``"vllm"``).
        spec_image:          Image override from ``docker:<image>`` runner
                             shorthand.  None when runner was bare ``"docker"``.
        yaml_images:         ``images:`` dict from the study YAML (optional).
        user_config_images:  ``images:`` dict from user config (optional).

    Returns:
        ``(image, image_source)`` tuple where *image_source* indicates provenance:
        ``"env"``, ``"yaml"``, ``"runner_override"``, ``"user_config"``,
        ``"local_build"``, or ``"registry"``.
    """
    # Load .env so LLEM_IMAGE_* vars are available
    from llenergymeasure.infra.runner_resolution import _load_dotenv

    _load_dotenv()

    # 1. Env var (includes .env via python-dotenv)
    env_key = f"{ENV_IMAGE_PREFIX}{engine.upper()}"
    if env_val := os.environ.get(env_key):
        logger.info("Image for %s resolved from env var %s: %s", engine, env_key, env_val)
        return (env_val, "env")

    # 2. Study YAML images: section
    if yaml_images and engine in yaml_images:
        img = yaml_images[engine]
        logger.info("Image for %s resolved from study YAML images: %s", engine, img)
        return (img, "yaml")

    # 3. Explicit image from runner spec (docker:<image> shorthand)
    if spec_image is not None:
        logger.info("Image for %s resolved from runner override: %s", engine, spec_image)
        return (spec_image, "runner_override")

    # 4. User config images: section
    if user_config_images and engine in user_config_images:
        img = user_config_images[engine]
        logger.info("Image for %s resolved from user config images: %s", engine, img)
        return (img, "user_config")

    # 5. Smart default: delegate to get_default_image() (local build → registry)
    image = get_default_image(engine)
    local_image = local_image_for(engine)
    if image == local_image:
        source = "local_build"
    elif _image_exists_locally(image):
        source = "registry_cached"
    else:
        source = "registry"
    return (image, source)


def show_image_resolution() -> None:
    """Print which Docker image each engine will resolve to.

    Shows local vs registry source for each engine.  Used by
    ``make docker-images`` for quick diagnostics.
    """
    print("=== Image resolution ===")
    for engine in sorted(ALL_ENGINES):
        image, source = resolve_image(engine)
        print(f"  {engine:10s} -> {image}  ({source})")


def parse_runner_value(value: str) -> tuple[RunnerMode, str | None]:
    """Parse a runner config value into ``(runner_type, image_override)``.

    Accepted forms::

        "local"                → ("local", None)
        "docker"               → ("docker", None)
        "docker:image/name:tag" → ("docker", "image/name:tag")

    Args:
        value: Raw string from ``runners.{engine}`` in YAML config.

    Returns:
        Tuple of ``(runner_type, image_override)`` where ``image_override`` is
        ``None`` when the built-in default image should be used.

    Raises:
        ValueError: If ``"docker:"`` is given with an empty image name, or if
                    the value is not one of the recognised runner types.
    """
    if value == RUNNER_LOCAL:
        return (RUNNER_LOCAL, None)

    if value == RUNNER_DOCKER:
        return (RUNNER_DOCKER, None)

    if value.startswith("docker:"):
        image = value[len("docker:") :]
        if not image:
            raise ValueError(
                "empty image name in runner value 'docker:' - "
                "use 'docker' (bare) to select the built-in default image, "
                "or 'docker:full/image:tag' for an explicit image."
            )
        return (RUNNER_DOCKER, image)

    raise ValueError(
        f"Unrecognised runner value {value!r}. "
        "Accepted values: 'local', 'docker', or 'docker:<image-tag>'."
    )
