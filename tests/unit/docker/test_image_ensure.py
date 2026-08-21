"""Tests for the guarded image pull: ``ensure_image`` and ``ensure_images``.

Every ``docker pull`` the framework issues is guarded by a local
``docker image inspect``, so a warm image never reaches a registry. These tests
fake the docker CLI at this module's ``subprocess.run`` and pin both entry
points: the single interactive one (raises, streams docker's output) and the
concurrent one (reports per image, captures the output the caller classifies).
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable, Iterable
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest

from llenergymeasure.infra.docker import lifecycle
from llenergymeasure.infra.docker_errors import DockerImagePullError
from tests.helpers.docker_cli import fake_docker_pull_cli

_INSPECT_JSON = b'[{"Id": "sha256:abc"}]'


def _fake_docker_cli(
    pull: Callable[[str], CompletedProcess[bytes]],
    *,
    present: Iterable[str] = (),
) -> tuple[Callable[..., CompletedProcess[bytes]], list[list[str]]]:
    """The shared docker-CLI fake, pinned to this module's inspect payload."""
    return fake_docker_pull_cli(pull, present=present, inspect_stdout=_INSPECT_JSON)


def _pull_ok(image: str) -> CompletedProcess[bytes]:
    return CompletedProcess(["docker", "pull", image], 0, b"", b"")


def _pull_absent(image: str) -> CompletedProcess[bytes]:
    return CompletedProcess(["docker", "pull", image], 1, b"", b"manifest unknown")


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_cached_image_is_never_pulled(monkeypatch: pytest.MonkeyPatch):
    """A locally present image costs one inspect and no network call."""
    run, calls = _fake_docker_cli(_pull_ok, present={"img:1"})
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    outcomes = lifecycle.ensure_images([("k", "img:1")], on_outcome=lambda _k, o: o)

    assert [c[:2] for c in calls] == [["docker", "image"]]
    assert outcomes[0].cached is True
    assert outcomes[0].ok is True
    # The guard's own inspect output is handed on, so a cached image still
    # carries its metadata without a second call.
    assert outcomes[0].inspect_stdout == _INSPECT_JSON


def test_absent_image_is_pulled_then_reinspected(monkeypatch: pytest.MonkeyPatch):
    """A missing image is pulled, then inspected so its metadata comes back."""
    run, calls = _fake_docker_cli(_pull_ok)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    outcome = lifecycle.ensure_images([("k", "img:1")], on_outcome=lambda _k, o: o)[0]

    assert [c[1] for c in calls] == ["image", "pull", "image"]
    assert outcome.cached is False
    assert outcome.ok is True
    assert outcome.returncode == 0
    assert outcome.inspect_stdout == _INSPECT_JSON


# ---------------------------------------------------------------------------
# ensure_images: concurrency, reporting, ordering
# ---------------------------------------------------------------------------


def test_pulls_run_concurrently(monkeypatch: pytest.MonkeyPatch):
    """Three absent images are pulled simultaneously, not one after another.

    The barrier only clears if all three pulls are in flight at once, so a
    serial implementation would hang here until the barrier timed out.
    """
    barrier = threading.Barrier(3, timeout=5)

    def pull(image: str) -> CompletedProcess[bytes]:
        barrier.wait()
        return _pull_ok(image)

    run, _ = _fake_docker_cli(pull)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    outcomes = lifecycle.ensure_images(
        [("a", "a:1"), ("b", "b:1"), ("c", "c:1")], on_outcome=lambda _k, o: o
    )

    assert all(o.ok for o in outcomes)


def test_worker_count_capped(monkeypatch: pytest.MonkeyPatch):
    """More images than the cap still only runs `max_concurrent` at a time."""
    seen: list[int] = []
    real = lifecycle.ThreadPoolExecutor

    def spy(*, max_workers: int, thread_name_prefix: str = "") -> object:
        seen.append(max_workers)
        return real(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    run, _ = _fake_docker_cli(_pull_ok)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    monkeypatch.setattr(lifecycle, "ThreadPoolExecutor", spy)

    images = [f"img:{i}" for i in range(lifecycle.MAX_CONCURRENT_PULLS + 2)]
    lifecycle.ensure_images([(i, i) for i in images], on_outcome=lambda _k, o: o)

    assert seen == [lifecycle.MAX_CONCURRENT_PULLS]


def test_one_failure_does_not_cancel_siblings(monkeypatch: pytest.MonkeyPatch):
    """A failing pull is reported, not raised, and the others still complete."""

    def pull(image: str) -> CompletedProcess[bytes]:
        return _pull_absent(image) if image == "bad:1" else _pull_ok(image)

    run, _ = _fake_docker_cli(pull)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    outcomes = {
        o.image: o
        for o in lifecycle.ensure_images(
            [("good", "good:1"), ("bad", "bad:1"), ("also", "also:1")],
            on_outcome=lambda _k, o: o,
        )
    }

    assert outcomes["good:1"].ok and outcomes["also:1"].ok
    assert outcomes["bad:1"].ok is False
    # The failure's stderr comes back so the caller can classify it (an
    # unreachable registry needs a retry, an absent image needs a build).
    assert "manifest unknown" in outcomes["bad:1"].stderr


def test_timeout_is_reported_not_raised(monkeypatch: pytest.MonkeyPatch):
    """A pull that exceeds the budget comes back flagged, with no exception."""

    def pull(image: str) -> CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd=["docker", "pull", image], timeout=1)

    run, _ = _fake_docker_cli(pull)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    outcome = lifecycle.ensure_images([("k", "img:1")], on_outcome=lambda _k, o: o)[0]

    assert outcome.timed_out is True
    assert outcome.ok is False


def test_results_follow_input_order_not_completion_order(monkeypatch: pytest.MonkeyPatch):
    """Returned values track the order the images were given, so callers can zip."""
    import time

    def pull(image: str) -> CompletedProcess[bytes]:
        if image == "slow:1":
            time.sleep(0.05)
        return _pull_ok(image)

    run, _ = _fake_docker_cli(pull)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    order = lifecycle.ensure_images(
        [("slow", "slow:1"), ("fast", "fast:1")], max_concurrent=2, on_outcome=lambda k, _o: k
    )

    assert order == ["slow", "fast"]


def test_empty_image_list_starts_no_pool(monkeypatch: pytest.MonkeyPatch):
    """Nothing to ensure means no thread pool and no docker call at all."""
    executor = MagicMock()
    monkeypatch.setattr(lifecycle, "ThreadPoolExecutor", executor)

    assert lifecycle.ensure_images([], on_outcome=lambda _k, o: o) == []
    executor.assert_not_called()


def test_handler_exception_surfaces(monkeypatch: pytest.MonkeyPatch):
    """An unexpected failure inside the caller's handler must not vanish."""
    run, _ = _fake_docker_cli(_pull_ok)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    def boom(key: str, outcome: lifecycle.PullOutcome) -> None:
        raise RuntimeError("handler broke")

    with pytest.raises(RuntimeError, match="handler broke"):
        lifecycle.ensure_images([("k", "img:1")], on_outcome=boom)


def test_two_keys_on_one_image_get_one_pull_and_two_reports(
    monkeypatch: pytest.MonkeyPatch,
):
    """Items sharing an image tag are each reported, under their own key.

    Two engines can legitimately be pinned to the same image. Each still needs
    its own report - a caller that had to recover its context from the image
    reference alone would collapse the two onto whichever it saw last - but the
    tag must be pulled exactly ONCE, not raced against itself.
    """
    run, calls = _fake_docker_cli(_pull_ok)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    reported = lifecycle.ensure_images(
        [("vllm", "shared:1"), ("tensorrt", "shared:1")],
        on_outcome=lambda key, outcome: (key, outcome.image, outcome.ok),
    )

    assert reported == [
        ("vllm", "shared:1", True),
        ("tensorrt", "shared:1", True),
    ]
    pulls = [c for c in calls if c[:2] == ["docker", "pull"]]
    assert pulls == [["docker", "pull", "shared:1"]]


def test_a_shared_image_failing_is_reported_against_every_key(
    monkeypatch: pytest.MonkeyPatch,
):
    """One failed pull of a shared tag must not be attributed to one key only."""
    run, calls = _fake_docker_cli(_pull_absent)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    reported = lifecycle.ensure_images(
        [("vllm", "shared:1"), ("tensorrt", "shared:1")],
        on_outcome=lambda key, outcome: (key, outcome.ok),
    )

    assert reported == [("vllm", False), ("tensorrt", False)]
    assert len([c for c in calls if c[:2] == ["docker", "pull"]]) == 1


# ---------------------------------------------------------------------------
# ensure_image: the single interactive entry point
# ---------------------------------------------------------------------------


def test_ensure_image_cached_skips_the_pull_step(monkeypatch: pytest.MonkeyPatch):
    """A cached image reports the lookup and marks the pull step skipped."""
    run, calls = _fake_docker_cli(_pull_ok, present={"img:1"})
    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    progress = MagicMock()

    lifecycle.ensure_image("img:1", progress=progress)

    assert [c[1] for c in calls] == ["image"]
    progress.on_step_skip.assert_called_once_with("pull", "cached")
    progress.on_step_start.assert_called_once()  # image_check only, no pull step


def test_ensure_image_pulls_and_reports_both_steps(monkeypatch: pytest.MonkeyPatch):
    """An absent image closes the lookup step and opens a pull step."""
    run, _ = _fake_docker_cli(_pull_ok)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)
    progress = MagicMock()

    lifecycle.ensure_image("img:1", progress=progress)

    started = [c.args[0] for c in progress.on_step_start.call_args_list]
    assert started == ["image_check", "pull"]
    done = [c.args[0] for c in progress.on_step_done.call_args_list]
    assert done == ["image_check", "pull"]
    progress.on_step_skip.assert_not_called()


def test_ensure_image_raises_when_pull_fails(monkeypatch: pytest.MonkeyPatch):
    """A single run has nothing to salvage, so an unpullable image is terminal."""
    run, _ = _fake_docker_cli(_pull_absent)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    with pytest.raises(DockerImagePullError, match="Image not found or could not be pulled"):
        lifecycle.ensure_image("img:1")


def test_ensure_image_raises_on_timeout(monkeypatch: pytest.MonkeyPatch):
    """A timed-out pull raises with the timeout message, not the not-found one."""

    def pull(image: str) -> CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired(cmd=["docker", "pull", image], timeout=1)

    run, _ = _fake_docker_cli(pull)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    with pytest.raises(DockerImagePullError, match="timed out"):
        lifecycle.ensure_image("img:1")


def test_ensure_image_timeout_keeps_the_original_exception_as_cause(
    monkeypatch: pytest.MonkeyPatch,
):
    """The TimeoutExpired stays chained, so the traceback still shows the timeout.

    The pull reports its outcome rather than raising, which is what makes the
    concurrent path possible - but it is also how a cause gets dropped. The
    outcome carries the exception so the raising entry point can chain it.
    """
    timeout = subprocess.TimeoutExpired(cmd=["docker", "pull", "img:1"], timeout=1)

    def pull(image: str) -> CompletedProcess[bytes]:
        raise timeout

    run, _ = _fake_docker_cli(pull)
    monkeypatch.setattr(lifecycle.subprocess, "run", run)

    with pytest.raises(DockerImagePullError) as excinfo:
        lifecycle.ensure_image("img:1")

    assert excinfo.value.__cause__ is timeout
