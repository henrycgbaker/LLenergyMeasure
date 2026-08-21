"""Container process lifecycle: image ensure, launch, wait, and the watchdog.

This module owns everything that runs a docker subprocess for offline dispatch.
The block-until-exit mode is expressed as two separable steps so the
stdout-silence watchdog stays isolated to the wait path (the same launch/wait
separation server-mode measurement follows in its own
:mod:`llenergymeasure.serving.lifecycle`):

- :func:`launch` starts the container process (``subprocess.Popen``) and returns
  it. It is detach-capable: a detached caller would call this, then add its own
  health-poll and explicit stop.
- :func:`wait_to_completion` streams stdout, forwards progress events, and owns
  the stdout-silence + wall-clock WATCHDOG. The watchdog lives ONLY here, in the
  wait path - an idle-between-requests server that reused it would look "stuck",
  so a detached long-lived server must not call this.

:func:`run_blocking` is the classic no-progress path (``subprocess.run`` blocking
until exit). Neither builds server dispatch - only the launch/wait seam is left
clean.

Image availability is the module's other job, and every ``docker pull`` this
framework issues goes through :func:`_pull_image_if_absent`: guarded by a local
``docker image inspect`` so an already-cached image never triggers a remote call.
Two entry points sit on it, because the two callers want opposite things from
docker's output:

- :func:`ensure_image` - one image, for a single interactive run. Docker's own
  progress output streams straight to stderr so a multi-GB download visibly
  moves, and a failure raises.
- :func:`ensure_images` - several images at once, for a study that spans multiple
  engines. Pulls run concurrently and docker's output is CAPTURED instead:
  interleaved progress bars from three simultaneous pulls are unreadable, and the
  caller needs the stderr text to tell an unreachable registry from an absent
  image. Failures are reported per item rather than raised, so one bad image does
  not cancel its siblings.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

from llenergymeasure.config.ssot import (
    DOCKER_PULL_TIMEOUT,
    TIMEOUT_DOCKER_INSPECT,
    TIMEOUT_THREAD_JOIN,
)
from llenergymeasure.domain.progress import resolve_container_step
from llenergymeasure.infra.docker_errors import (
    DockerContainerError,
    DockerImagePullError,
    DockerStdoutSilenceError,
    DockerTimeoutError,
)

if TYPE_CHECKING:
    from llenergymeasure.domain.progress import ProgressCallback

logger = logging.getLogger(__name__)

_K = TypeVar("_K")
_T = TypeVar("_T")

# Upper bound on simultaneous ``docker pull`` threads. A study rarely spans more
# than three engines, and the daemon serialises layer writes anyway, so a small
# cap keeps memory/disk pressure bounded without throttling the common case.
MAX_CONCURRENT_PULLS = 3

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


@dataclass(frozen=True)
class PullOutcome:
    """What one guarded ``docker pull`` did, and what docker said about it.

    Deliberately a report rather than a verdict: whether an absent image is a
    user error, and what to advise about it, depends on who asked. A single
    interactive run wants an exception; a multi-engine study wants to finish
    pulling its other images first and then name every failure at once, with a
    per-engine rebuild hint. Both read the same three facts from here - did a
    pull run, did it succeed, and what did docker print.

    Attributes:
        image:           The image reference that was ensured.
        cached:          The local cache already had it, so NO pull ran.
        returncode:      ``docker pull``'s exit status; ``None`` when no pull ran
                         (cached) or it never returned (timed out).
        stderr:          ``docker pull`` stderr on a FAILED pull, decoded - the
                         text that distinguishes an unreachable registry from an
                         absent image. Empty otherwise, and empty on any path
                         whose output was streamed rather than captured (see
                         :func:`ensure_images`).
        inspect_stdout:  ``docker image inspect`` JSON for the now-present image
                         (the guard's output when cached, a fresh inspect after a
                         successful pull). ``b""`` when unavailable, which callers
                         must tolerate - it is display/verification metadata, not
                         a success signal.
        elapsed:         Seconds the pull took; ``0.0`` when cached.
        timeout_exc:     The ``TimeoutExpired`` raised when the pull exceeded
                         ``DOCKER_PULL_TIMEOUT``, else ``None``. It is both the
                         record THAT the pull timed out (see :attr:`timed_out`)
                         and the cause a caller can chain when it turns this
                         report back into an exception - reporting the outcome
                         instead of raising is what would otherwise lose the
                         traceback.
    """

    image: str
    cached: bool = False
    returncode: int | None = None
    stderr: str = ""
    inspect_stdout: bytes = b""
    elapsed: float = 0.0
    timeout_exc: BaseException | None = None

    @property
    def timed_out(self) -> bool:
        """Whether the pull exceeded ``DOCKER_PULL_TIMEOUT``.

        Derived from :attr:`timeout_exc` rather than stored alongside it, so the
        flag and the exception behind it cannot disagree.
        """
        return self.timeout_exc is not None

    @property
    def ok(self) -> bool:
        """Whether the image is now present locally."""
        return self.cached or (not self.timed_out and self.returncode == 0)


def _image_is_cached(image: str) -> subprocess.CompletedProcess[bytes]:
    """Run the local ``docker image inspect`` that guards every pull.

    Kept in this module rather than delegating to ``image_registry`` so that all
    of the offline dispatch's docker subprocess calls stay behind one patchable
    boundary.
    """
    return subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=TIMEOUT_DOCKER_INSPECT,
    )


def _pull_image_if_absent(
    image: str,
    *,
    capture_output: bool,
    on_pull_start: Callable[[], None] | None = None,
) -> PullOutcome:
    """Pull *image* unless it is already cached locally. The single pull site.

    The guard is the point: ``docker image inspect`` is a local daemon call, so
    checking first means a warm image never reaches out to a registry at all.

    ``capture_output`` decides where docker's own progress output goes - to this
    process's stderr (visible, for one interactive pull) or into
    :attr:`PullOutcome.stderr` (quiet, for concurrent pulls and for callers that
    need to classify the failure text). ``on_pull_start`` fires once the guard has
    decided a pull is actually needed, before it begins, so a caller that reports
    the cache lookup and the download as separate phases can close one and open
    the other at the right moment.

    Never raises: the outcome carries what happened and the caller decides what it
    means.
    """
    check = _image_is_cached(image)
    if check.returncode == 0:
        return PullOutcome(image=image, cached=True, inspect_stdout=check.stdout)

    if on_pull_start is not None:
        on_pull_start()
    logger.info("Image %s not found locally, pulling...", image)
    if not capture_output:
        print(f"Pulling image: {image}", file=sys.stderr)

    t0 = time.perf_counter()
    sink: dict[str, Any] = (
        {"capture_output": True} if capture_output else {"stdout": sys.stderr, "stderr": sys.stderr}
    )
    try:
        pull = subprocess.run(["docker", "pull", image], timeout=DOCKER_PULL_TIMEOUT, **sink)
    except subprocess.TimeoutExpired as exc:
        return PullOutcome(
            image=image,
            elapsed=time.perf_counter() - t0,
            timeout_exc=exc,
        )
    elapsed = time.perf_counter() - t0

    if pull.returncode != 0:
        # Only a failure's stderr is worth carrying: it is what tells an
        # unreachable registry from an image that genuinely is not there.
        stderr = (pull.stderr or b"").decode("utf-8", "replace") if capture_output else ""
        return PullOutcome(image=image, returncode=pull.returncode, stderr=stderr, elapsed=elapsed)

    # The image is present now; re-inspect so the caller can read its metadata
    # and schema labels without issuing its own docker call. Unavailable metadata
    # is not a failure - the pull succeeded.
    try:
        inspect = _image_is_cached(image)
        inspect_stdout = inspect.stdout if inspect.returncode == 0 else b""
    except Exception:
        inspect_stdout = b""
    return PullOutcome(image=image, returncode=0, inspect_stdout=inspect_stdout, elapsed=elapsed)


def ensure_image(image: str, progress: ProgressCallback | None = None) -> None:
    """Ensure one image is present locally, pulling with visible output if not.

    The single-run entry point onto :func:`_pull_image_if_absent`. Always emits an
    ``image_check`` step so the user sees the cache lookup; a pull gets its own
    ``pull`` step. Docker's progress output goes straight to stderr, so a
    multi-GB download visibly moves rather than looking hung.

    Raises:
        DockerImagePullError: The image is absent and could not be pulled (or the
            pull timed out). A single run has nothing to salvage, so this is
            terminal here - contrast :func:`ensure_images`, which reports.
    """
    from llenergymeasure.utils.formatting import short_name

    short_image = short_name(image)

    if progress:
        progress.on_step_start("image_check", "Inspecting", short_image)
    t0 = time.perf_counter()

    def _pull_starting() -> None:
        if progress:
            progress.on_step_done("image_check", time.perf_counter() - t0)
            progress.on_step_start("pull", "Pulling", image)

    outcome = _pull_image_if_absent(image, capture_output=False, on_pull_start=_pull_starting)

    if outcome.cached:
        if progress:
            progress.on_step_update("image_check", f"{short_image} (cached)")
            progress.on_step_done("image_check", time.perf_counter() - t0)
            progress.on_step_skip("pull", "cached")
        return

    if progress:
        progress.on_step_done("pull", outcome.elapsed)
    if outcome.timed_out:
        raise DockerImagePullError(
            message=f"Image pull timed out after {DOCKER_PULL_TIMEOUT}s: {image}",
            fix_suggestion=f"Pull manually: docker pull {image}",
        ) from outcome.timeout_exc
    if not outcome.ok:
        raise DockerImagePullError(
            message=f"Image not found or could not be pulled: {image}",
            fix_suggestion=f"docker pull {image}",
        )


def ensure_images(
    items: Sequence[tuple[_K, str]],
    *,
    max_concurrent: int = MAX_CONCURRENT_PULLS,
    on_outcome: Callable[[_K, PullOutcome], _T],
) -> list[_T]:
    """Ensure several images are present, pulling the absent ones concurrently.

    One thread per distinct image, capped at *max_concurrent*, so a multi-engine
    study on a fresh box does not serialise several multi-GB downloads. Each pull
    is guarded by the same local inspect as the single-image path, so an image
    already in the cache costs one daemon call and no network.

    Each item pairs an image reference with a KEY of the caller's choosing, and
    that key comes back to *on_outcome* alongside the outcome. The key is what
    makes two items sharing one image tag distinguishable: that is expressible
    (two engines can be pinned to the same image), and a caller left to recover
    its own context from the image reference alone would collapse the two into
    one. The image is still pulled ONCE for all the items that name it - each of
    them gets its own report of that single outcome - because submitting the same
    tag twice would race two identical downloads past the guard.

    A failing pull does NOT cancel its siblings: every image runs to completion
    and *on_outcome* is called for every item, so the caller can report every
    failure at once instead of aborting on the first. Docker's pull output is
    captured rather than streamed - three interleaved progress bars are
    unreadable, and the captured stderr is what lets a caller tell an unreachable
    registry from a genuinely absent image.

    Args:
        items: ``(key, image)`` pairs to ensure. Distinct images are pulled in
            order of first appearance.
        max_concurrent: Ceiling on simultaneous pulls.
        on_outcome: Called once per ITEM with that item's key and the outcome for
            its image, from the worker thread that pulled it. Per-item follow-up
            work therefore stays concurrent (the point, when that work includes a
            cold verification probe); anything that must not interleave -
            terminal output above all - is the caller's to serialise. Its return
            values are collected.

    Returns:
        Whatever *on_outcome* returned, one per item, in the order *items* was
        given (not completion order).
    """
    if not items:
        return []

    # Group by image so one tag is pulled once however many items name it, while
    # every item still gets its own on_outcome call. Insertion order is the input
    # order of each image's first appearance, so submission order is stable.
    positions_by_image: dict[str, list[int]] = {}
    for position, (_key, image) in enumerate(items):
        positions_by_image.setdefault(image, []).append(position)

    def _ensure_one(image: str, positions: list[int]) -> list[tuple[int, _T]]:
        outcome = _pull_image_if_absent(image, capture_output=True)
        return [(position, on_outcome(items[position][0], outcome)) for position in positions]

    with ThreadPoolExecutor(
        max_workers=min(len(positions_by_image), max_concurrent),
        thread_name_prefix="image-pull",
    ) as executor:
        futures = [
            executor.submit(_ensure_one, image, positions)
            for image, positions in positions_by_image.items()
        ]
        # .result() re-raises anything a worker raised, so an unexpected failure
        # in on_outcome surfaces rather than vanishing into the pool.
        collected: dict[int, _T] = {}
        for future in futures:
            collected.update(future.result())
    return [collected[position] for position in range(len(items))]


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
    :mod:`llenergymeasure.serving.lifecycle`.
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
