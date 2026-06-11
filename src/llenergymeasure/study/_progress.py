"""Progress plumbing for subprocess-isolated experiments.

Two halves of a cross-process progress bridge:

- ``_QueueProgressCallback`` lives inside the worker subprocess and serialises
  step events onto a ``multiprocessing.Queue``.
- ``_consume_progress_events`` runs as a daemon thread in the parent and
  forwards those events to the study-level ``StudyProgressCallback``.

Pure plumbing: neither half touches ``StudyRunner`` state.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.domain.progress import StudyProgressCallback


class _QueueProgressCallback:
    """ProgressCallback that serialises step events onto a multiprocessing.Queue.

    Created inside the worker subprocess. Events are dicts consumed by the
    parent's _consume_progress_events thread and forwarded to StudyStepDisplay.
    """

    def __init__(self, queue: Any) -> None:
        self._queue = queue

    def _put(self, event: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._queue.put(event)

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        self._put(
            {"event": "step_start", "step": step, "description": description, "detail": detail}
        )

    def on_step_update(self, step: str, detail: str) -> None:
        self._put({"event": "step_update", "step": step, "detail": detail})

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        self._put({"event": "step_done", "step": step, "elapsed_sec": elapsed_sec})

    def on_step_skip(self, step: str, reason: str = "") -> None:
        self._put({"event": "step_skip", "step": step, "reason": reason})

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        self._put({"event": "substep", "step": step, "text": text, "elapsed_sec": elapsed_sec})

    def on_substep_start(self, step: str, text: str) -> None:
        self._put({"event": "substep_start", "step": step, "text": text})

    def on_substep_done(
        self,
        step: str,
        text: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        self._put(
            {
                "event": "substep_done",
                "step": step,
                "text": text,
                "elapsed_sec": elapsed_sec,
            }
        )


def _consume_progress_events(
    q: Any,
    study_progress: StudyProgressCallback | None = None,
) -> None:
    """Consume progress events from the queue and forward to study display.

    Runs as a daemon thread in the parent process. Receives step events from
    the child subprocess via multiprocessing.Queue and forwards them to the
    StudyProgressCallback (typically StudyStepDisplay).

    Coarse events (started/completed/failed) are ignored here - study-level
    begin/end experiment tracking is handled directly by _run_one().
    """
    while True:
        event = q.get()
        if event is None:
            break

        if not isinstance(event, dict) or study_progress is None:
            continue

        event_type = event.get("event")

        # Forward step-level events to study display
        if event_type == "step_start":
            study_progress.on_step_start(
                event["step"], event.get("description", ""), event.get("detail", "")
            )
        elif event_type == "step_update":
            study_progress.on_step_update(event["step"], event.get("detail", ""))
        elif event_type == "step_done":
            study_progress.on_step_done(event["step"], event.get("elapsed_sec", 0))
        elif event_type == "step_skip":
            study_progress.on_step_skip(event["step"], event.get("reason", ""))
        elif event_type == "substep":
            study_progress.on_substep(
                event["step"], event.get("text", ""), event.get("elapsed_sec", 0)
            )
        elif event_type == "substep_start":
            study_progress.on_substep_start(event["step"], event.get("text", ""))
        elif event_type == "substep_done":
            study_progress.on_substep_done(
                event["step"], event.get("text"), event.get("elapsed_sec")
            )
