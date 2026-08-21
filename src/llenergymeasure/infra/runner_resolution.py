"""Runner mechanics - turn a runner pin, or no pin, into an execution mode.

Which config layer wins - env var versus study file versus user config - is settled
before this module runs, by the precedence chain in
:mod:`llenergymeasure.config.precedence`. This module holds what is left: parsing a
pinned runner value, and probing the host to pick a runner for an engine nothing
pinned.

Auto-detection is container-self-aware. Placement is relative: llenergymeasure may
itself run inside a container, so PATH inspection alone is not enough - a stray
``docker`` CLI on PATH would make dispatch attempt docker-in-docker. When llem runs
inside a container it resolves by Docker-socket availability (siblings via the host
daemon if a socket is mounted, otherwise process). On the host, if Docker + the
NVIDIA Container Toolkit are available it defaults to container mode for best
measurement isolation; otherwise it falls back to process mode with a nudge message.

This module is intentionally free of Docker dispatch mechanics - it only decides
*what* should run *where*. Dispatch is handled by DockerRunner.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

from llenergymeasure.config.runner_spec import RunnerPin, RunnerSpec
from llenergymeasure.config.ssot import (
    RUNNER_CONTAINER,
    RUNNER_PROCESS,
    SOURCE_AUTO_DETECTED,
    SOURCE_DEFAULT,
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


def resolve_runner(engine: str, pin: RunnerPin | None = None) -> RunnerSpec:
    """Turn an explicit runner pin, or the absence of one, into a RunnerSpec.

    The precedence question - env var versus study file versus user config - is
    settled before this is called, by the precedence chain
    (:func:`llenergymeasure.config.precedence.resolve_study_settings`). What is left
    here is the mechanics: parse the pinned value, or, when nothing pinned this
    engine, probe the host and pick a runner.

    Fallback when there is no pin (in order):
        1. Auto-detection (container-self-aware) - source="auto_detected"
           In a container: container mode iff a Docker socket is available (DooD
           siblings), else process. On the host: container iff Docker + NVIDIA CT.
        2. Built-in default: process             - source="default"

    When mode is "container" and image is None, the caller (DockerRunner) should
    resolve the image via ``get_default_image(engine)`` from image_registry.

    Args:
        engine: Engine name, e.g. "transformers", "vllm", "tensorrt".
        pin: The runner the chain resolved for this engine, with the layer that
            supplied it, or None when no layer pinned it. A pinned value is
            honoured verbatim (the legacy "local"/"docker" vocabulary was renamed
            in v0.7 and is rejected with a migration error).

    Returns:
        RunnerSpec with mode, image, and source fields populated.
    """
    if pin is not None:
        mode, image = parse_runner_value(pin.value)
        return RunnerSpec(mode=mode, image=image, source=pin.source)

    # No layer pinned this engine, so detect.
    #    Container-self-aware branch. If llem itself runs inside a container, PATH
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

    #    On the host: Docker + NVIDIA Container Toolkit available?
    if is_docker_available():
        logger.info("Docker detected. Using containerised execution for reproducible measurements.")
        return RunnerSpec(mode=RUNNER_CONTAINER, image=None, source=SOURCE_AUTO_DETECTED)

    # Default: process mode with nudge message
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
    pins: Mapping[str, RunnerPin] | None = None,
) -> dict[str, RunnerSpec]:
    """Resolve runners for all engines in a study.

    Calls :func:`resolve_runner` for each unique engine, handing it that engine's
    pin from the precedence chain when one exists. An engine with no pin
    auto-detects.

    Args:
        engines: Unique engine names present in the study's experiments.
        pins: Chain-resolved runner pins keyed by engine name.

    Returns:
        Dict mapping each engine name to its resolved RunnerSpec.
    """
    resolved_pins = pins or {}
    return {engine: resolve_runner(engine, resolved_pins.get(engine)) for engine in engines}
