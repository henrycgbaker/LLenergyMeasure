"""Real-container proof of the study cleanup path against a live daemon.

Two properties, both hard to trust from a fake: that cleanup sees only its own
study's containers, and that a container it stops has its log tail on disk before
it is removed.

Unlike the other docker-marked integration tests, this one needs no GPU, no
NVIDIA container runtime, and no engine image: it launches tiny busybox
containers wearing the study ownership labels and drives the real
``cleanup_study_containers`` against them. It is marked ``docker`` because it
needs a reachable daemon, and it skips when the daemon or the busybox image is
absent (it never pulls).

The startup reaper is deliberately NOT exercised here. Its ``docker ps`` filter
is host-wide (any ``llem.study_id``), so running it against a real daemon could
stop containers belonging to something else on the machine; its scoping is proved
against the in-memory docker fake in
tests/unit/docker/test_container_ownership.py instead.

Run: pytest tests/integration/test_container_cleanup_docker.py -m docker -v
Requires: a running Docker daemon and a local busybox image.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from llenergymeasure.infra.docker.ownership import (
    cleanup_study_containers,
    generate_container_labels,
)

pytestmark = pytest.mark.docker

IMAGE = "busybox:latest"

_OWNERSHIP_LOGGER = "llenergymeasure.infra.docker.ownership"


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


MARKER = "hello-from-container"


def _launch(study_id: str, *, auto_remove: bool = False) -> str:
    """Start a detached, labelled busybox container and return its id.

    The removal policy is the whole point of the parameter, because it is what
    separates the two container shapes the cleanup meets. The default is NOT
    ``--rm``, matching the engine-server shape: that is the container the cleanup
    has to stop, read, and only then remove. ``auto_remove=True`` stands in for
    an experiment or baseline container, which docker reaps on stop.

    The container prints a marker before sleeping, so a persisted log tail has
    something to prove.
    """
    argv = ["docker", "run", "-d"]
    if auto_remove:
        argv.append("--rm")
    for key, value in generate_container_labels(study_id).items():
        argv += ["--label", f"{key}={value}"]
    argv += [IMAGE, "sh", "-c", f"echo {MARKER}; sleep 120"]
    result = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=True)
    return result.stdout.strip()


def _exists(container_id: str) -> bool:
    result = subprocess.run(
        ["docker", "container", "inspect", container_id], capture_output=True, timeout=30
    )
    return result.returncode == 0


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
def test_cleanup_reaps_own_containers_and_spares_a_concurrent_study(tmp_path: Path) -> None:
    """One study's cleanup reclaims its own labelled containers and nothing else.

    Two study ids stand in for two concurrent trials on one host. Cleaning up
    after the first must leave the second's container running - the container-kill
    hazard that a shared placeholder study id created. Each container the cleanup
    does own is stopped, has its log tail written into the study's directory, and
    is then removed.
    """
    study_a = f"llemtest{uuid.uuid4().hex}"
    study_b = f"llemtest{uuid.uuid4().hex}"
    log_dir_a = tmp_path / "study-a" / "failed-runs"
    log_dir_b = tmp_path / "study-b" / "failed-runs"
    launched: list[str] = []
    try:
        a_first = _launch(study_a)
        a_second = _launch(study_a)
        b_only = _launch(study_b)
        launched += [a_first, a_second, b_only]
        assert all(_is_running(c) for c in launched)

        cleanup_study_containers(study_a, log_dir_a)

        # Reclaimed: stopped, log tail kept, container gone.
        for container_id in (a_first, a_second):
            assert not _is_running(container_id)
            assert not _exists(container_id)
        persisted = sorted(p.read_text() for p in log_dir_a.glob("*.log"))
        assert len(persisted) == 2
        assert all(MARKER in text for text in persisted)

        assert _is_running(b_only), "a concurrent study's container was stopped"
        assert not log_dir_b.exists(), "a concurrent study's logs were written"

        # The second study's own cleanup then reclaims exactly what is left.
        cleanup_study_containers(study_b, log_dir_b)
        assert not _is_running(b_only)
        assert not _exists(b_only)
        assert len(list(log_dir_b.glob("*.log"))) == 1
    finally:
        for container_id in launched:
            _remove(container_id)


@pytest.mark.skipif(not _docker_daemon_available(), reason="Docker daemon not reachable")
@pytest.mark.skipif(
    not _image_present(IMAGE), reason=f"Image {IMAGE!r} not present locally (never pulled here)"
)
def test_cleanup_leaves_no_complaint_for_an_auto_removing_container(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A ``--rm`` container is reaped by docker on stop, and that is not a problem.

    The run-to-completion shapes carry ``--rm``, so the daemon removes them the
    moment they stop and there is nothing left for the cleanup to persist or
    reclaim. The point of this test is the absence of a warning: the normal
    outcome for an experiment or baseline container must not look like a failure
    to keep its evidence.
    """
    study_id = f"llemtest{uuid.uuid4().hex}"
    container_id = _launch(study_id, auto_remove=True)
    try:
        assert _is_running(container_id)

        with caplog.at_level(logging.WARNING, logger=_OWNERSHIP_LOGGER):
            cleanup_study_containers(study_id, tmp_path / "failed-runs")

        assert not _exists(container_id), "docker did not reap the --rm container"
        assert caplog.text == ""
    finally:
        _remove(container_id)
