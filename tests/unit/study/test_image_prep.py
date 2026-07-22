"""Tests for concurrent Docker image preparation in ``study.image_prep``.

Exercises ``_ImageMixin._prepare_images`` without a real Docker daemon by
patching ``inspect_image`` (local cache probe) and ``subprocess.run`` (the
``docker pull`` / ``docker image inspect`` calls). Fingerprint verification is
bypassed via ``LLEM_SKIP_IMAGE_CHECK`` so the tests isolate the pull /
concurrency / error-aggregation behaviour.
"""

from __future__ import annotations

import re
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from subprocess import CompletedProcess
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from llenergymeasure.cli._step_display import StudyStepDisplay
from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.infra.docker_errors import DockerImagePullError
from llenergymeasure.study import image_prep
from llenergymeasure.study.image_prep import (
    _aggregate_image_errors,
    _classify_pull_failure,
    _ImageMixin,
)

# =============================================================================
# Harness + helpers
# =============================================================================


class _Harness(_ImageMixin):
    """Minimal carrier of the three attributes ``_prepare_images`` relies on."""

    def __init__(
        self,
        runner_specs: dict[str, RunnerSpec] | None,
        progress: object | None = None,
    ) -> None:
        self._runner_specs = runner_specs  # type: ignore[assignment]
        self._progress = progress  # type: ignore[assignment]
        self._images_prepared = False


def _docker_spec(image: str) -> RunnerSpec:
    return RunnerSpec(mode="docker", image=image, source="yaml")


def _cached() -> CompletedProcess[bytes]:
    """An ``inspect_image`` result for an image present in the local cache."""
    return CompletedProcess(["docker", "image", "inspect"], 0, b"[]", b"")


def _missing() -> CompletedProcess[bytes]:
    """An ``inspect_image`` result for an image absent from the local cache."""
    return CompletedProcess(["docker", "image", "inspect"], 1, b"", b"Error: No such image")


@pytest.fixture(autouse=True)
def _bypass_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the schema-fingerprint handshake so no host/probe work runs."""
    monkeypatch.setenv("LLEM_SKIP_IMAGE_CHECK", "1")


# =============================================================================
# All cached: no pulls, no executor
# =============================================================================


def test_all_cached_no_pull_no_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every image cached -> no docker pull, no ThreadPoolExecutor, images ready."""
    specs = {name: _docker_spec(f"img/{name}:latest") for name in ("transformers", "vllm")}
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _cached())

    run_calls: list[list[str]] = []

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        run_calls.append(argv)
        return _cached()

    monkeypatch.setattr(image_prep.subprocess, "run", fake_run)
    executor_cls = MagicMock()
    monkeypatch.setattr(image_prep, "ThreadPoolExecutor", executor_cls)

    progress = MagicMock()
    harness = _Harness(specs, progress)
    harness._prepare_images()

    assert harness._images_prepared is True
    # No docker pull attempted for cached images.
    assert run_calls == [], f"expected no subprocess.run calls, got {run_calls}"
    # The concurrent-pull path must not even be entered when nothing is missing.
    executor_cls.assert_not_called()
    assert progress.image_ready.call_count == 2
    for c in progress.image_ready.call_args_list:
        assert c.kwargs["cached"] is True
    progress.begin_image_prep.assert_called_once()
    progress.end_image_prep.assert_called_once()


# =============================================================================
# Some missing: concurrent pulls invoked
# =============================================================================


def test_missing_images_pulled_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three missing images are pulled on distinct threads simultaneously.

    A ``Barrier(3)`` in the fake pull only clears if all three pulls are
    in-flight at once; a sequential implementation would block on the first
    ``wait()`` until the barrier times out and the test would fail.
    """
    engines = ("transformers", "vllm", "tensorrt")
    specs = {name: _docker_spec(f"img/{name}:latest") for name in engines}
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

    barrier = threading.Barrier(3, timeout=5)
    pulled: list[str] = []
    pulled_lock = threading.Lock()

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        if argv[:2] == ["docker", "pull"]:
            barrier.wait()  # requires all 3 threads present concurrently
            with pulled_lock:
                pulled.append(argv[2])
            return CompletedProcess(argv, 0, b"", b"")
        return _cached()  # the post-pull "docker image inspect"

    monkeypatch.setattr(image_prep.subprocess, "run", fake_run)

    progress = MagicMock()
    harness = _Harness(specs, progress)
    harness._prepare_images()

    assert harness._images_prepared is True
    assert sorted(pulled) == sorted(f"img/{name}:latest" for name in engines)
    assert progress.image_ready.call_count == 3
    for c in progress.image_ready.call_args_list:
        assert c.kwargs["cached"] is False


def test_pull_worker_count_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """max_workers is capped at _MAX_CONCURRENT_PULLS even with more missing images."""
    n = image_prep._MAX_CONCURRENT_PULLS + 2
    specs = {f"e{i}": _docker_spec(f"img/e{i}:latest") for i in range(n)}
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

    seen_max_workers: list[int] = []
    real_executor = image_prep.ThreadPoolExecutor

    def spy_executor(*, max_workers: int, thread_name_prefix: str = "") -> ThreadPoolExecutor:
        seen_max_workers.append(max_workers)
        return real_executor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)

    monkeypatch.setattr(image_prep, "ThreadPoolExecutor", spy_executor)
    monkeypatch.setattr(
        image_prep.subprocess,
        "run",
        lambda argv, **_kw: CompletedProcess(argv, 0, b"[]", b""),
    )

    _Harness(specs)._prepare_images()

    assert seen_max_workers == [image_prep._MAX_CONCURRENT_PULLS]


# =============================================================================
# One fails: others complete, aggregate error
# =============================================================================


def test_one_pull_fails_others_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single failing pull does not cancel siblings; failure surfaces afterwards."""
    specs = {
        "transformers": _docker_spec("img/transformers:latest"),
        "vllm": _docker_spec("img/vllm:latest"),
        "tensorrt": _docker_spec("img/tensorrt:latest"),
    }
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

    completed: list[str] = []
    completed_lock = threading.Lock()

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        if argv[:2] == ["docker", "pull"]:
            image = argv[2]
            if image == "img/vllm:latest":
                return CompletedProcess(argv, 1, b"", b"manifest unknown")
            with completed_lock:
                completed.append(image)
            return CompletedProcess(argv, 0, b"", b"")
        return _cached()

    monkeypatch.setattr(image_prep.subprocess, "run", fake_run)

    progress = MagicMock()
    harness = _Harness(specs, progress)

    with pytest.raises(DockerImagePullError) as excinfo:
        harness._prepare_images()

    # The two healthy images were pulled to completion despite the sibling failure.
    assert sorted(completed) == ["img/tensorrt:latest", "img/transformers:latest"]
    # Single failure -> original error is surfaced unchanged, naming the image.
    assert "img/vllm:latest" in str(excinfo.value)
    # The two successes were still finalised (reported ready).
    ready_images = {c.args[1] for c in progress.image_ready.call_args_list}
    assert ready_images == {"img/transformers:latest", "img/tensorrt:latest"}
    progress.image_failed.assert_called_once()
    assert progress.image_failed.call_args.args[1] == "img/vllm:latest"
    # end_image_prep still fires on the failure path (finally block).
    progress.end_image_prep.assert_called_once()
    # A failed prep must not mark images prepared.
    assert harness._images_prepared is False


def test_multiple_failures_aggregate_names_each(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two failing pulls collapse into one aggregate naming both images + causes."""
    specs = {
        "transformers": _docker_spec("img/transformers:latest"),
        "vllm": _docker_spec("img/vllm:latest"),
        "tensorrt": _docker_spec("img/tensorrt:latest"),
    }
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        if argv[:2] == ["docker", "pull"]:
            image = argv[2]
            if image == "img/vllm:latest":
                return CompletedProcess(argv, 1, b"", b"manifest unknown")  # absent
            if image == "img/tensorrt:latest":
                return CompletedProcess(argv, 1, b"", b"dial tcp: i/o timeout")  # network
            return CompletedProcess(argv, 0, b"", b"")
        return _cached()

    monkeypatch.setattr(image_prep.subprocess, "run", fake_run)

    harness = _Harness(specs)
    with pytest.raises(DockerImagePullError) as excinfo:
        harness._prepare_images()

    message = str(excinfo.value)
    assert "2 Docker images could not be prepared" in message
    assert "img/vllm:latest" in message
    assert "img/tensorrt:latest" in message
    # Network-vs-absent classification is preserved in the aggregate.
    assert "registry unreachable" in message


# =============================================================================
# Ordering independence of the result
# =============================================================================


def test_aggregate_error_is_order_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The raised aggregate is identical no matter which pull finishes first."""

    def run_with_completion_order(fast_image: str) -> str:
        specs = {
            "vllm": _docker_spec("img/vllm:latest"),
            "tensorrt": _docker_spec("img/tensorrt:latest"),
        }
        monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

        def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
            if argv[:2] == ["docker", "pull"]:
                image = argv[2]
                # Force a completion order: the non-fast image lingers briefly.
                if image != fast_image:
                    import time as _time

                    _time.sleep(0.05)
                return CompletedProcess(argv, 1, b"", b"manifest unknown")
            return _cached()

        monkeypatch.setattr(image_prep.subprocess, "run", fake_run)

        harness = _Harness(specs)
        with pytest.raises(DockerImagePullError) as excinfo:
            harness._prepare_images()
        return str(excinfo.value)

    first = run_with_completion_order("img/vllm:latest")
    second = run_with_completion_order("img/tensorrt:latest")
    assert first == second


def test_concurrent_failures_all_render_in_real_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two images failing concurrently both appear in the real display's panel.

    Drives the actual ``StudyStepDisplay`` (not a MagicMock) through
    ``_prepare_images`` with two failing pulls. Regression guard: the live
    panel formerly held a single failure slot, so the second concurrent
    failure overwrote the first and only one rendered.
    """
    specs = {
        "vllm": _docker_spec("img/vllm:latest"),
        "tensorrt": _docker_spec("img/tensorrt:latest"),
    }
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        if argv[:2] == ["docker", "pull"]:
            image = argv[2]
            stderr = b"manifest unknown" if image == "img/vllm:latest" else b"dial tcp: i/o timeout"
            return CompletedProcess(argv, 1, b"", stderr)
        return _cached()

    monkeypatch.setattr(image_prep.subprocess, "run", fake_run)

    # force_terminal drives the live-panel (TTY) code path so _render_image_prep
    # reflects accumulated state rather than the immediate non-TTY prints.
    display = StudyStepDisplay(
        total_experiments=2,
        console=Console(force_terminal=True, no_color=True, width=120),
    )
    harness = _Harness(specs, display)
    with pytest.raises(DockerImagePullError):
        harness._prepare_images()

    rendered = display._render_image_prep().plain
    assert "vllm" in rendered
    assert "tensorrt" in rendered
    # Both distinct causes survive, not just the last failure to land.
    assert "docker compose build vllm" in rendered  # absent-image cause
    assert "registry unreachable (network)" in rendered  # network cause
    # Unique, collision-free counters across both concurrent failures.
    counters = re.findall(r"\[(\d+)/2\]", rendered)
    assert sorted(counters) == ["1", "2"], f"expected unique counters, got {counters}"


def test_pull_timeout_reported_and_aggregated(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pull that times out becomes a DockerImagePullError, not an uncaught raise."""
    specs = {"vllm": _docker_spec("img/vllm:latest")}
    monkeypatch.setattr(image_prep, "inspect_image", lambda image, timeout: _missing())

    def fake_run(argv: list[str], **_kwargs: object) -> CompletedProcess[bytes]:
        if argv[:2] == ["docker", "pull"]:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
        return _cached()

    monkeypatch.setattr(image_prep.subprocess, "run", fake_run)

    progress = MagicMock()
    harness = _Harness(specs, progress)
    with pytest.raises(DockerImagePullError) as excinfo:
        harness._prepare_images()

    assert "timed out" in str(excinfo.value).lower()
    progress.image_failed.assert_called_once()


# =============================================================================
# Guard clauses
# =============================================================================


def test_no_runner_specs_is_noop() -> None:
    """No runner specs -> nothing to prepare, images not marked prepared."""
    harness = _Harness(None)
    harness._prepare_images()
    assert harness._images_prepared is False


def test_no_docker_engines_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only local-mode engines -> no image prep runs."""
    specs = {"transformers": RunnerSpec(mode="local", image=None, source="default")}
    called = MagicMock()
    monkeypatch.setattr(image_prep, "inspect_image", called)
    harness = _Harness(specs)
    harness._prepare_images()
    assert harness._images_prepared is False
    called.assert_not_called()


# =============================================================================
# Free-function unit coverage
# =============================================================================


def test_classify_pull_failure_network_vs_absent() -> None:
    """Network stderr -> retry hint; anything else -> build-locally hint."""
    reason_net, err_net = _classify_pull_failure("vllm", "img/vllm:latest", "dial tcp: i/o timeout")
    assert "network" in reason_net
    assert "connectivity" in err_net.fix_suggestion.lower()

    reason_abs, err_abs = _classify_pull_failure("vllm", "img/vllm:latest", "manifest unknown")
    assert "not found" in reason_abs
    assert err_abs.fix_suggestion == "docker compose build vllm"
    assert err_abs.args[0] == "Image not found: img/vllm:latest"


def test_aggregate_single_error_passthrough() -> None:
    """A lone error is returned unchanged (type, message, fix all intact)."""
    err = DockerImagePullError(message="only one", fix_suggestion="fix it")
    assert _aggregate_image_errors([err]) is err


def test_aggregate_multiple_errors_sorted_and_combined() -> None:
    """Multiple errors combine into one message with sorted, per-line entries."""
    e1 = DockerImagePullError(message="Image not found: zzz", fix_suggestion="build zzz")
    e2 = DockerImagePullError(message="Image not found: aaa", fix_suggestion="build aaa")
    combined = _aggregate_image_errors([e1, e2])
    assert isinstance(combined, DockerImagePullError)
    text = str(combined)
    assert "2 Docker images could not be prepared" in text
    # Sorted: aaa line precedes zzz line regardless of input order.
    assert text.index("aaa") < text.index("zzz")
