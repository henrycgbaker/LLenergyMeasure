"""Runner resolution - determine process vs container execution mode for each engine.

Precedence chain (highest wins):
  env var > study/experiment YAML > user config > auto-detection > default

Auto-detection is container-self-aware. Placement is relative: llenergymeasure may
itself run inside a container, so PATH inspection alone is not enough - a stray
``docker`` CLI on PATH would make dispatch attempt docker-in-docker. When llem runs
inside a container it resolves by Docker-socket availability (siblings via the host
daemon if a socket is mounted, otherwise process). On the host, if Docker + the
NVIDIA Container Toolkit are available it defaults to container mode for best
measurement isolation; otherwise it falls back to process mode with a nudge message.

User config: non-"auto" values in UserRunnersConfig are treated as explicit.
"auto" (the default) falls through to auto-detection, allowing Docker to be
picked up automatically when available. Pass ``user_config=None`` to skip
the user config step entirely.

This module is intentionally free of Docker dispatch mechanics - it only decides
*what* should run *where*. Dispatch is handled by DockerRunner (Plan 03).
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llenergymeasure.config.user_config import UserRunnersConfig

from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.config.ssot import (
    ENV_RUNNER_PREFIX,
    RUNNER_CONTAINER,
    RUNNER_PROCESS,
    SOURCE_AUTO_DETECTED,
    SOURCE_DEFAULT,
    SOURCE_ENV,
    SOURCE_USER_CONFIG,
    SOURCE_YAML,
)

# The NVIDIA Container Toolkit binary list lives in docker_preflight (its canonical
# home) and is reused here for the docker-availability check.
from llenergymeasure.infra.docker_preflight import NVIDIA_TOOLKIT_BINS

# Re-exported from image_registry for convenience - parse_runner_value is defined
# there (canonical home) but used heavily in this module and its tests.
from llenergymeasure.infra.image_registry import parse_runner_value

__all__ = [
    "is_container_socket_available",
    "is_docker_available",
    "is_running_in_container",
    "parse_runner_value",
    "resolve_runner",
    "resolve_study_runners",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# .env file loading
# ---------------------------------------------------------------------------


@functools.cache
def _load_dotenv() -> None:
    """Load ``.env`` from the working directory if present.

    Uses ``override=False`` so shell environment variables always win.
    Cached so the filesystem scan happens at most once per process.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(override=False)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Docker availability detection
# ---------------------------------------------------------------------------


@functools.cache
def is_docker_available() -> bool:
    """Return True if Docker CLI and NVIDIA Container Toolkit are both on PATH.

    This is a quick host-level check (PATH inspection only). Container-level GPU
    validation is done at pre-flight time.

    Checks:
        1. ``docker`` CLI is on PATH (via shutil.which)
        2. At least one NVIDIA Container Toolkit binary is on PATH:
           ``nvidia-container-runtime``, ``nvidia-ctk``, or ``nvidia-container-cli``
    """
    if shutil.which("docker") is None:
        return False

    return any(shutil.which(tool) is not None for tool in NVIDIA_TOOLKIT_BINS)


# ---------------------------------------------------------------------------
# Container-self detection (is llem itself running inside a container?)
# ---------------------------------------------------------------------------


@functools.cache
def is_running_in_container() -> bool:
    """Return True if llenergymeasure itself appears to be running inside a container.

    Layered FILE-EXISTENCE detection - the pattern real self-detection libraries use
    (Sindre Sorhus' widely-vendored ``is-inside-container``, Testcontainers' own
    ``inside_container`` helper):

        1. ``Path("/.dockerenv")``       - the empty marker Docker writes at the
           container root filesystem.
        2. ``Path("/run/.containerenv")`` - podman's equivalent marker.

    Deliberately NOT ``/proc/1/cgroup`` substring matching: that signal is unreliable
    on the cgroup v2 unified hierarchy, where the cgroup path is frequently just ``/``
    or is driver-dependent (systemd vs cgroupfs), producing false negatives on modern
    hosts. Deliberately NOT ``systemd-detect-virt --container``: it reads PID 1's
    ``container=`` environment / a systemd-only ``/run/systemd/container`` marker and
    returns ``none`` for a plain ``docker run`` container (systemd/systemd#15393 - the
    maintainers' own documented workaround is "check ``/.dockerenv`` instead"), which
    is exactly the case that matters here.

    Consumed by auto-detection so that llem running inside a container does not
    blindly attempt docker-in-docker when a stray ``docker`` CLI is on PATH.
    """
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


@functools.cache
def is_container_socket_available() -> bool:
    """Return True if a Docker control socket appears usable from this process.

    This is the docker-outside-of-docker (DooD) signal: when llem runs inside a
    container, a mounted host Docker socket lets it dispatch *sibling* containers via
    the host daemon (which performs GPU injection - llem's own container needs only
    the socket, not the NVIDIA Container Toolkit), rather than attempting
    docker-in-docker.

    Existence check only, matching ``is_docker_available``'s PATH-inspection
    philosophy - deep validation (socket reachability, daemon ping, ``--network host``
    routing) stays in pre-flight. Checks:

        1. ``DOCKER_HOST`` environment variable set (an explicit socket / TCP / SSH
           endpoint), or
        2. ``Path("/var/run/docker.sock")`` present (the default mounted-socket path).

    Note: any ``DOCKER_HOST`` value counts as available, including a remote TCP/SSH
    daemon where the sibling-container reasoning does not strictly hold. This shallow
    check matches the module's existence-only philosophy; validated remote and podman
    topologies are tracked in issue #891.
    """
    if os.environ.get("DOCKER_HOST"):
        return True
    return Path("/var/run/docker.sock").exists()


# ---------------------------------------------------------------------------
# Core resolution function
# ---------------------------------------------------------------------------


def resolve_runner(
    engine: str,
    yaml_runners: dict[str, str] | None = None,
    user_config: UserRunnersConfig | None = None,
) -> RunnerSpec:
    """Resolve the runner for a single engine using the full precedence chain.

    Precedence (highest to lowest):
        1. Env var   ``LLEM_RUNNER_{ENGINE}``   - source="env"
        2. Study YAML ``runners:`` section       - source="yaml"
        3. User config ``runners.{engine}``      - source="user_config"
           Only non-"auto" values are treated as explicit. "auto" falls through
           to step 4. Pass ``user_config=None`` to allow auto-detection.
        4. Auto-detection (container-self-aware)  - source="auto_detected"
           In a container: container mode iff a Docker socket is available (DooD
           siblings), else process. On the host: container iff Docker + NVIDIA CT.
        5. Built-in default: process            - source="default"

    When mode is "container" and image is None, the caller (DockerRunner) should
    resolve the image via ``get_default_image(engine)`` from image_registry.

    Args:
        engine:       Engine name, e.g. "transformers", "vllm", "tensorrt".
        yaml_runners: Runners dict from study YAML ``runners:`` section.
                      Keys are engine names, values are runner strings.
        user_config:  UserRunnersConfig from loaded user preferences.
                      None = no user config present (enables auto-detection).
                      When provided, "auto" values fall through to auto-detection;
                      explicit values ("process", "container", "container:<img>") are
                      honoured (the legacy "local"/"docker" vocabulary was renamed in
                      v0.7 and is now rejected with a migration error).

    Returns:
        RunnerSpec with mode, image, and source fields populated.
    """
    # Load .env (idempotent, shell env wins via override=False)
    _load_dotenv()

    # 1. Env var: LLEM_RUNNER_{ENGINE} (highest precedence)
    env_key = f"{ENV_RUNNER_PREFIX}{engine.upper()}"
    if env_val := os.environ.get(env_key):
        mode, image = parse_runner_value(env_val)
        return RunnerSpec(mode=mode, image=image, source=SOURCE_ENV)
    # 2. Study/experiment YAML runners section
    if yaml_runners is not None and engine in yaml_runners:
        yaml_val = yaml_runners[engine]
        if yaml_val is not None:
            mode, image = parse_runner_value(yaml_val)
            return RunnerSpec(mode=mode, image=image, source=SOURCE_YAML)
    # 3. User config - "auto" means no explicit preference, fall through to auto-detection.
    #    Passing user_config=None means "no user config file present" → auto-detect.
    if user_config is not None:
        user_val: str = getattr(user_config, engine, "auto")
        if user_val != "auto":
            mode, image = parse_runner_value(user_val)
            return RunnerSpec(
                mode=mode, image=image, source=SOURCE_USER_CONFIG
            )  # "auto" -> fall through to auto-detection

    # 4. Auto-detection.
    #    4a. Container-self-aware branch. If llem itself runs inside a container, PATH
    #    inspection alone is misleading: a docker CLI on PATH without a usable control
    #    socket would make dispatch attempt docker-in-docker. Resolve by socket
    #    availability (the DooD signal) instead. See is_running_in_container /
    #    is_container_socket_available for the detection rationale.
    if is_running_in_container():
        if is_container_socket_available():
            logger.info(
                "Running inside a container with a Docker socket - auto-selecting "
                "container mode (siblings dispatched via the host daemon, which does "
                "GPU injection)."
            )
            return RunnerSpec(mode=RUNNER_CONTAINER, image=None, source=SOURCE_AUTO_DETECTED)
        logger.info(
            "Running inside a container without a Docker socket - auto-selecting "
            "process mode (avoids attempting docker-in-docker)."
        )
        # auto_detected (not default): a positive detection ran and chose process,
        # so the CLI renders "(auto-detected)" rather than the misleading "(default)".
        return RunnerSpec(mode=RUNNER_PROCESS, image=None, source=SOURCE_AUTO_DETECTED)

    #    4b. On the host: Docker + NVIDIA Container Toolkit available?
    if is_docker_available():
        logger.info("Docker detected. Using containerised execution for reproducible measurements.")
        return RunnerSpec(mode=RUNNER_CONTAINER, image=None, source=SOURCE_AUTO_DETECTED)

    # 5. Default: process mode with nudge message
    logger.info(
        "Docker not detected. Install Docker + NVIDIA Container Toolkit "
        "for reproducible isolated measurements."
    )
    return RunnerSpec(mode=RUNNER_PROCESS, image=None, source=SOURCE_DEFAULT)


# ---------------------------------------------------------------------------
# Study-level runner resolution
# ---------------------------------------------------------------------------


def resolve_study_runners(
    engines: list[str],
    yaml_runners: dict[str, str] | None = None,
    user_config: UserRunnersConfig | None = None,
) -> dict[str, RunnerSpec]:
    """Resolve runners for all engines in a study.

    Calls ``resolve_runner`` for each unique engine and returns a mapping of
    engine name → RunnerSpec. The ``yaml_runners`` dict (from the study YAML
    ``runners:`` section) and ``user_config`` are passed through unchanged.

    Args:
        engines:      Unique engine names present in the study's experiments.
        yaml_runners: Study-level ``runners:`` section from YAML (optional).
        user_config:  Loaded UserRunnersConfig (optional, None = auto-detect).

    Returns:
        Dict mapping each engine name to its resolved RunnerSpec.
    """
    return {
        engine: resolve_runner(engine, yaml_runners=yaml_runners, user_config=user_config)
        for engine in engines
    }
