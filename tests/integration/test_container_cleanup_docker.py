"""Real-container proof that the study cleanup path sees only its own containers.

Unlike the other docker-marked integration tests, this one needs no GPU, no
NVIDIA container runtime, and no engine image: it launches tiny ``busybox sleep``
containers wearing the study ownership labels and drives the real
``cleanup_study_containers`` against them. It is marked ``docker`` because it
needs a reachable daemon, and it skips when the daemon or the busybox image is
absent (it never pulls).

The startup reaper is deliberately NOT exercised here. Its ``docker ps`` filter
is host-wide (any ``llem.study_id``), so running it against a real daemon could
stop containers belonging to something else on the machine; its scoping is proved
against the in-memory docker fake in
tests/unit/study/test_container_lifecycle.py instead.

Run: pytest tests/integration/test_container_cleanup_docker.py -m docker -v
Requires: a running Docker daemon and a local busybox image.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid

import pytest

from llenergymeasure.study.container_lifecycle import (
    cleanup_study_containers,
    generate_container_labels,
)

pytestmark = pytest.mark.docker

IMAGE = "busybox:latest"


def _docker_daemon_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _image_present(image: str) -> bool:
    """True if the image is already local. Never pulls."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image], capture_output=True, timeout=10
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def _launch(study_id: str) -> str:
    """Start a detached, labelled busybox container and return its id."""
    argv = ["docker", "run", "-d"]
    for key, value in generate_container_labels(study_id).items():
        argv += ["--label", f"{key}={value}"]
    argv += [IMAGE, "sleep", "120"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True)
    return result.stdout.strip()


def _is_running(container_id: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip() == "true"


def _remove(container_id: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=60)


@pytest.mark.skipif(not _docker_daemon_available(), reason="Docker daemon not reachable")
@pytest.mark.skipif(
    not _image_present(IMAGE), reason=f"Image {IMAGE!r} not present locally (never pulled here)"
)
def test_cleanup_reaps_own_containers_and_spares_a_concurrent_study() -> None:
    """One study's cleanup stops its own labelled containers and nothing else.

    Two study ids stand in for two concurrent trials on one host. Cleaning up
    after the first must leave the second's container running - the container-kill
    hazard that a shared placeholder study id created.
    """
    study_a = f"llemtest{uuid.uuid4().hex}"
    study_b = f"llemtest{uuid.uuid4().hex}"
    launched: list[str] = []
    try:
        a_first = _launch(study_a)
        a_second = _launch(study_a)
        b_only = _launch(study_b)
        launched += [a_first, a_second, b_only]
        assert all(_is_running(c) for c in launched)

        cleanup_study_containers(study_a)

        assert not _is_running(a_first)
        assert not _is_running(a_second)
        assert _is_running(b_only), "a concurrent study's container was stopped"

        # The second study's own cleanup then stops exactly what is left.
        cleanup_study_containers(study_b)
        assert not _is_running(b_only)
    finally:
        for container_id in launched:
            _remove(container_id)
