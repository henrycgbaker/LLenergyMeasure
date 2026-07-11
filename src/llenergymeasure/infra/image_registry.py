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

import logging
import os
import subprocess
from functools import lru_cache

from llenergymeasure.config.ssot import (
    ALL_ENGINES,
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
    "parse_runner_value",
    "resolve_image",
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
    # 1. Prefer a locally-built image (fast local iteration).
    local_image = LOCAL_IMAGE_TEMPLATE.format(engine=engine)
    if _image_exists_locally(local_image):
        logger.info("Using local image %s (from docker compose build)", local_image)
        return local_image

    # 2. Per-engine default remote image at the resolved version.
    engine_name = engine_str(engine)
    template = DEFAULT_IMAGE_TEMPLATES.get(engine_name)
    if template is None:
        raise ConfigError(
            f"No default Docker image is defined for engine {engine_name!r}. "
            f'Set runners.{engine_name} to "docker:<image>:<tag>" in your study '
            f"YAML to pin an explicit image."
        )
    image = template.format(version=_default_image_version(engine_name))
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
    if engine == Engine.TRANSFORMERS.value:
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
    local_image = LOCAL_IMAGE_TEMPLATE.format(engine=engine)
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
