"""Container process lifecycle: image ensure, launch, wait, and the watchdog.

This module owns everything that runs a docker subprocess for offline dispatch.
The block-until-exit mode is expressed as two separable steps so the
stdout-silence watchdog stays isolated to the wait path (the same launch/wait
separation server-mode measurement follows in its own
:mod:`llenergymeasure.infra.server_lifecycle`):

- :func:`launch` starts the container process (``subprocess.Popen``) and returns
  it. It is detach-capable: a detached caller would call this, then add its own
  health-poll and explicit stop.
- :func:`wait_to_completion` streams stdout, forwards progress events, and owns
  the stdout-silence + wall-clock WATCHDOG. The watchdog lives ONLY here, in the
  wait path - an idle-between-requests server that reused it would look "stuck",
  so a detached long-lived server must not call this.

:func:`run_blocking` is the classic no-progress path (``subprocess.run`` blocking
until exit); :func:`ensure_image` is the image-availability pull guard. Neither
builds server dispatch - only the launch/wait seam is left clean.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.ssot import (
    DOCKER_PULL_TIMEOUT,
    TIMEOUT_DOCKER_INSPECT,
    TIMEOUT_THREAD_JOIN,
)
from llenergymeasure.domain.progress import resolve_container_step
from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerStdoutSilenceError,
    DockerTimeoutError,
)

if TYPE_CHECKING:
    from llenergymeasure.domain.progress import ProgressCallback

logger = logging.getLogger(__name__)

# Watchdog poll cadence: small enough to surface timeouts promptly, large
# enough to keep idle CPU near zero. 0.5s gives users at most a half-second
# tail beyond their configured ceiling and matches the progress-display
# heartbeat sampling cadence.
_WATCHDOG_POLL_INTERVAL = 0.5

# Sentinel for "budget disabled" in the deadline comparison.
_NO_DEADLINE = float("inf")

# Keywords in container stderr that indicate meaningful activity. When no JSON
# progress events arrive (old images), surface these as on_step_update to show
# the container is alive and working.
_ACTIVITY_KEYWORDS = (
    "loading",
    "downloading",
    "measuring",
    "warmup",
    "warming",
    "inference",
    "saving",
    "running",
    "model",
    "tokenizer",
)


def ensure_image(image: str, progress: ProgressCallback | None = None) -> None:
    """Check if the Docker image exists locally; pull with visible output if not.

    Always emits an ``image_check`` step so the user sees the cache lookup.
    If the image is not cached, emits a separate ``pull`` step.
    Substeps report image metadata (ID, size, age) for provenance visibility.
    """
    from llenergymeasure.utils.formatting import short_name

    short_image = short_name(image)

    if progress:
        progress.on_step_start("image_check", "Inspecting", short_image)
    t0 = time.perf_counter()

    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=TIMEOUT_DOCKER_INSPECT,
    )
    if check.returncode == 0:
        if progress:
            progress.on_step_update("image_check", f"{short_image} (cached)")
            progress.on_step_done("image_check", time.perf_counter() - t0)
            progress.on_step_skip("pull", "cached")
        return

    if progress:
        progress.on_step_done("image_check", time.perf_counter() - t0)

    # Image not cached - pull it
    if progress:
        progress.on_step_start("pull", "Pulling", image)
    t0_pull = time.perf_counter()

    print(f"Pulling image: {image}", file=sys.stderr)
    try:
        pull = subprocess.run(
            ["docker", "pull", image],
            stdout=sys.stderr,
            stderr=sys.stderr,
            timeout=DOCKER_PULL_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        if progress:
            progress.on_step_done("pull", time.perf_counter() - t0_pull)
        from llenergymeasure.infra.docker_errors import DockerImagePullError

        raise DockerImagePullError(
            message=f"Image pull timed out after {DOCKER_PULL_TIMEOUT}s: {image}",
            fix_suggestion=f"Pull manually: docker pull {image}",
        ) from exc
    if pull.returncode != 0:
        if progress:
            progress.on_step_done("pull", time.perf_counter() - t0_pull)
        from llenergymeasure.infra.docker_errors import DockerImagePullError

        raise DockerImagePullError(
            message=f"Image not found or could not be pulled: {image}",
            fix_suggestion=f"docker pull {image}",
        )

    if progress:
        progress.on_step_done("pull", time.perf_counter() - t0_pull)


def run_blocking(cmd: list[str], timeout: float | None) -> tuple[int, str]:
    """Run the container blocking until exit (classic, no-progress path).

    Backward-compatible ``subprocess.run`` capture. Returns
    ``(returncode, stderr_text)``. Raises :class:`DockerTimeoutError` when the
    wall-clock ``timeout`` is exceeded.
    """
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DockerTimeoutError(
            message=f"Container timed out after {timeout}s.",
            fix_suggestion="Increase timeout or reduce experiment size.",
        ) from exc
    return proc.returncode, proc.stderr


def launch(cmd: list[str]) -> subprocess.Popen[str]:
    """Start the container process and return the ``Popen`` handle.

    Detach-capable seam for the offline batch dispatch: this only starts the
    process (stdout/stderr piped) and is composed with
    :func:`wait_to_completion` (which owns the watchdog) by the streaming path.
    Server measurement mode does NOT reuse this: it owns an independent detached
    ``docker run -d`` lifecycle (launch / readiness poll / stop) in
    :mod:`llenergymeasure.infra.server_lifecycle`.
    """
    try:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        raise DockerContainerError(
            message=f"Failed to start docker process: {exc}",
            fix_suggestion="Is Docker installed and running?",
        ) from exc


def wait_to_completion(
    proc: subprocess.Popen[str],
    *,
    timeout: float | None,
    silence_timeout: float | None,
    progress: ProgressCallback | None = None,
    mask_secrets_fn: Callable[[str], str] | None = None,
    container_start_time: float | None = None,
) -> tuple[int, str]:
    """Block until ``proc`` exits, streaming stdout and running the watchdog.

    Container inner events (step_start, step_update, step_done) are forwarded as
    top-level progress steps so the CLI can display each measurement phase
    individually (Docker BuildKit-style granularity).

    The stdout-silence + wall-clock WATCHDOG is contained here, in the wait path
    only: it kills the container and raises the matching error if either budget
    is exhausted. The ``container_start`` step (started by the caller) is ended
    when the first inner event arrives, capturing the container boot time.

    Returns ``(returncode, stderr_text)``.
    """
    stderr_lines: list[str] = []

    # Thread-safe flag shared with stderr thread to track if JSON events arrived
    container_start_done_event = threading.Event()

    def _read_stderr(pipe: Any) -> None:
        """Read stderr in a background thread to prevent blocking.

        For old images that don't emit JSON progress events, surfaces
        interesting log lines as step updates on container_start.
        """
        for line in pipe:
            stderr_lines.append(line)
            stripped = line.strip()
            logger.debug("container stderr: %s", stripped)
            # Surface activity from old images as step updates
            if (
                progress is not None
                and not container_start_done_event.is_set()
                and stripped
                and any(kw in stripped.lower() for kw in _ACTIVITY_KEYWORDS)
            ):
                # Truncate long log lines and strip log prefix (e.g. "INFO:root:")
                display_text = stripped
                if ":" in display_text and display_text.split(":")[0].isupper():
                    display_text = display_text.split(":", 2)[-1].strip()
                progress.on_step_update("container_start", display_text[:50])
        pipe.close()

    # Read stderr in background thread to avoid deadlock
    stderr_thread = threading.Thread(target=_read_stderr, args=(proc.stderr,), daemon=True)
    stderr_thread.start()

    # Pump stdout into a queue from a daemon thread so the main loop
    # can wake up periodically to check both timeout budgets even
    # when the container is producing nothing. The blocking
    # ``for line in proc.stdout`` shape this replaces would hang
    # indefinitely on a stuck CUDA / NCCL / compile step - the
    # wall-clock proc.wait() never gets a chance to fire because we
    # never exit the for-loop. See issue #366.
    stdout_q: queue.Queue[str | None] = queue.Queue()

    def _pump_stdout(pipe: Any) -> None:
        try:
            for line in pipe:
                stdout_q.put(line)
        finally:
            stdout_q.put(None)  # sentinel: pipe closed (EOF or process exit)

    assert proc.stdout is not None
    stdout_thread = threading.Thread(target=_pump_stdout, args=(proc.stdout,), daemon=True)
    stdout_thread.start()

    container_start_done = False
    watchdog_start = time.monotonic()
    last_activity = watchdog_start

    try:
        while True:
            wall_remaining, silence_remaining = _check_watchdog_deadlines(
                proc, timeout, silence_timeout, watchdog_start, last_activity
            )

            # Cap the queue wait at the poll interval so a Ctrl-C
            # interrupt is observed within ~0.5s even with both
            # budgets large.
            wait_for = min(wall_remaining, silence_remaining, _WATCHDOG_POLL_INTERVAL)
            try:
                line = stdout_q.get(timeout=wait_for)
            except queue.Empty:
                continue
            if line is None:
                break  # stdout pipe closed
            last_activity = time.monotonic()

            container_start_done = _handle_stdout_line(
                line,
                progress,
                container_start_done,
                container_start_time,
                container_start_done_event,
                mask_secrets_fn,
            )
    finally:
        with suppress(Exception):
            proc.stdout.close()
        stdout_thread.join(timeout=TIMEOUT_THREAD_JOIN)

    # If no inner events arrived, end container_start now (old images)
    if not container_start_done and container_start_time is not None and progress is not None:
        progress.on_step_done("container_start", time.perf_counter() - container_start_time)

    # Wait for process exit. The watchdog above ensures we only get
    # here when stdout has closed; proc.wait() blocks only until the
    # process actually reaps, which is bounded.
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        proc.wait()
        raise DockerTimeoutError(
            message=f"Container timed out after {timeout}s.",
            fix_suggestion="Increase timeout or reduce experiment size.",
        ) from exc

    stderr_thread.join(timeout=TIMEOUT_THREAD_JOIN)
    stderr_text = "".join(stderr_lines)

    return proc.returncode, stderr_text


def _check_watchdog_deadlines(
    proc: subprocess.Popen[str],
    timeout: float | None,
    silence_timeout: float | None,
    watchdog_start: float,
    last_activity: float,
) -> tuple[float, float]:
    """Compute the remaining wall-clock and stdout-silence budgets.

    Kills the container and raises the matching watchdog error if either budget
    is exhausted. Returns ``(wall_remaining, silence_remaining)`` so the caller
    can cap its queue wait.
    """
    now = time.monotonic()
    wall_remaining = timeout - (now - watchdog_start) if timeout is not None else _NO_DEADLINE
    silence_remaining = (
        silence_timeout - (now - last_activity) if silence_timeout is not None else _NO_DEADLINE
    )

    if wall_remaining <= 0:
        _kill_container_for_watchdog(proc)
        raise DockerTimeoutError(
            message=f"Container timed out after {timeout}s (wall-clock).",
            fix_suggestion=(
                "Increase study_execution.experiment_timeout_seconds or reduce experiment size."
            ),
        )
    if silence_remaining <= 0:
        _kill_container_for_watchdog(proc)
        raise DockerStdoutSilenceError(
            message=(
                f"Container produced no stdout for {silence_timeout}s (likely stuck process)."
            ),
            fix_suggestion=(
                "Increase study_execution.stdout_silence_timeout_seconds "
                "if your workload legitimately goes silent for longer "
                "(e.g. fresh TRT-LLM engine builds), or investigate the "
                "stuck step. Check container logs in the exchange dir."
            ),
        )
    return wall_remaining, silence_remaining


def _handle_stdout_line(
    line: str,
    progress: ProgressCallback | None,
    container_start_done: bool,
    container_start_time: float | None,
    container_start_done_event: threading.Event,
    mask_secrets_fn: Callable[[str], str] | None,
) -> bool:
    """Process one container stdout line.

    JSON progress events are forwarded to ``progress``; plain output is logged
    (secrets masked). Returns the updated ``container_start_done`` flag - set True
    on the first inner event so the ``container_start`` step is ended exactly once.
    """
    stripped = line.strip()
    if stripped.startswith('{"event":') and progress is not None:
        try:
            event = json.loads(stripped)
            event_type = event.get("event")
            step = event.get("step", "")

            # End "container_start" on first inner event
            if not container_start_done and container_start_time is not None:
                container_start_done = True
                container_start_done_event.set()
                progress.on_step_done(
                    "container_start",
                    time.perf_counter() - container_start_time,
                )

            # Resolve container-boundary step id (e.g. the container's
            # "preflight" renders as "container_preflight" host-side). The
            # mapping lives in the progress registry.
            step = resolve_container_step(step)

            _dispatch_progress_event(progress, event_type, step, event)
        except (json.JSONDecodeError, KeyError):
            logger.debug("Unparseable progress line: %s", stripped)
    else:
        if stripped:
            masked = mask_secrets_fn(stripped) if mask_secrets_fn else stripped
            logger.debug("container stdout: %s", masked)
    return container_start_done


def _dispatch_progress_event(
    progress: ProgressCallback, event_type: str, step: str, event: dict[str, Any]
) -> None:
    """Forward a single decoded container progress event to the host callback."""
    if event_type == "step_start":
        progress.on_step_start(
            step,
            event.get("description", ""),
            event.get("detail", ""),
        )
    elif event_type == "step_update":
        progress.on_step_update(step, event.get("detail", ""))
    elif event_type == "step_done":
        progress.on_step_done(step, event.get("elapsed_sec", 0.0))
    elif event_type == "step_skip":
        progress.on_step_skip(step, event.get("reason", ""))
    elif event_type == "substep":
        progress.on_substep(
            step,
            event.get("text", ""),
            event.get("elapsed_sec", 0.0),
        )
    elif event_type == "substep_start":
        progress.on_substep_start(step, event.get("text", ""))
    elif event_type == "substep_done":
        progress.on_substep_done(
            step,
            event.get("text"),
            event.get("elapsed_sec"),
        )


def _kill_container_for_watchdog(proc: subprocess.Popen[str]) -> None:
    """Terminate then SIGKILL a hung container; swallow any wait failures.

    Used by the unified watchdog when a timeout fires. Mirrors the
    existing wall-clock kill path: best-effort terminate, then kill,
    then a final wait so the process group is fully reaped. Any
    exception from the cleanup path is logged at debug level - the
    watchdog's responsibility is to *raise the right error*, not to
    handle a cooperatively-shutting-down container.
    """
    for stage, action in (
        ("terminate", proc.terminate),
        ("wait-after-terminate", lambda: proc.wait(timeout=2.0)),
        ("kill", proc.kill),
        ("wait-after-kill", lambda: proc.wait(timeout=2.0)),
    ):
        try:
            action()
        except Exception as exc:
            logger.debug("Watchdog cleanup %s failed: %s", stage, exc)
