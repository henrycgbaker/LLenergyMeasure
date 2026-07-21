"""Single-experiment step display (Rich ProgressCallback implementation).

Renders Docker-BuildKit-style hierarchical output with phase headers and
numbered sub-steps for one measurement run. Split out of the former
``_step_display`` god-module; the ``_step_display`` facade re-exports
``StepDisplay`` so callers are unchanged.
"""

from __future__ import annotations

import contextlib
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.text import Text

from llenergymeasure.cli._step_render import (
    _HEARTBEAT_INTERVAL,
    _HEARTBEAT_THRESHOLD,
    _SPINNER_FPS,
    _SPINNER_FRAMES,
    _DynamicRenderable,
    _phase_for_step,
    _render_substep_lines,
    _step_line,
    _step_line_prefix,
)
from llenergymeasure.domain.progress import STEP_LABELS
from llenergymeasure.utils.formatting import format_elapsed as _format_elapsed


class StepDisplay:
    """Rich-based step display with hierarchical phase grouping.

    Steps are automatically grouped into phases (Setup, Measurement)
    based on the STEP_PHASES mapping. Phase headers are rendered before
    their first sub-step.

    When steps are pre-registered via register_steps(), a fixed [x/y]
    counter is shown. Steps that don't apply are shown as SKIP.

    Thread-safe: harness calls on_step_start/update/done from a worker
    thread while Rich Live refreshes from its own thread.

    Usage::

        display = StepDisplay(header="Experiment: gpt2 | transformers | bf16")
        display.register_steps(
            docker_steps(images_prepared=False, host_baseline=True)
        )
        display.start()
        # ... pass display as ProgressCallback to harness ...
        display.finish()
    """

    def __init__(
        self, header: str = "", console: Console | None = None, force_plain: bool = False
    ) -> None:
        self._console = console or Console(stderr=True)
        self._header = header
        self._lock = threading.Lock()

        # Phase tracking: ordered list of phases seen, steps per phase
        self._phases: list[str] = []
        self._phase_steps: dict[str, list[str]] = {}
        self._explicitly_registered: bool = False

        # Step state: done, skipped, or active
        self._step_data: dict[str, tuple[str, str, float]] = {}  # step -> (label, detail, elapsed)
        self._completed_steps: set[str] = set()
        self._skipped_steps: set[str] = set()

        # Substeps per step: list of (text, elapsed_sec) tuples
        self._substeps: dict[str, list[tuple[str, float]]] = {}
        # Active substep per step: (text, start_monotonic) - drives the
        # heartbeat spinner for long-running sub-operations (e.g. CUDA init
        # inside the baseline container) so Rich Live animates them.
        self._active_substep: dict[str, tuple[str, float]] = {}

        # Active step
        self._active_step: str | None = None
        self._active_label: str = ""
        self._active_detail: str = ""
        self._active_start: float = 0.0

        # Rich Live (TTY only); force_plain disables Live mode even in a TTY
        self._live: Live | None = None
        self._is_tty = self._console.is_terminal and not force_plain
        self._total_start: float = 0.0

        # Non-TTY: track which phases have been printed
        self._printed_phases: set[str] = set()

        # Heartbeat thread (non-TTY only)
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_stop = threading.Event()

    @property
    def total_steps(self) -> int:
        return sum(len(steps) for steps in self._phase_steps.values())

    def _ensure_step_registered(self, step: str) -> str:
        """Register a step into its phase if not already present. Returns phase name.

        Must be called with self._lock held.
        """
        phase = _phase_for_step(step)
        if phase not in self._phases:
            self._phases.append(phase)
        if phase not in self._phase_steps:
            self._phase_steps[phase] = []
        if step not in self._phase_steps[phase]:
            self._phase_steps[phase].append(step)
        return phase

    def register_steps(self, steps: list[str]) -> None:
        """Pre-register steps for fixed [x/y] counting.

        When registered, the counter denominator is fixed from the start.
        Steps not started by the end are shown as SKIP.
        """
        with self._lock:
            self._explicitly_registered = True
            for step in steps:
                self._ensure_step_registered(step)

    def start(self) -> None:
        """Begin the display. Prints header and starts Rich Live if TTY."""
        self._total_start = time.monotonic()
        if self._header:
            self._console.print(self._header, highlight=False)
        if self._is_tty:
            self._live = Live(
                _DynamicRenderable(self._render),
                console=self._console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        else:
            # Start heartbeat thread for non-TTY mode
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop Rich Live and heartbeat thread."""
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
        if self._live is not None:
            self._live.stop()
            self._live = None

    def finish(
        self,
        total_elapsed: float | None = None,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
    ) -> None:
        """Print completion footer with optional key metrics."""
        self.stop()
        if total_elapsed is None:
            total_elapsed = time.monotonic() - self._total_start
        self._console.print(f"\nCompleted in {_format_elapsed(total_elapsed)}", highlight=False)
        # Key metrics line (Energy + Throughput)
        metrics_parts = []
        if energy_j is not None and energy_j > 0:
            metrics_parts.append(f"Energy: {energy_j:.1f} J")
        if throughput_tok_s is not None and throughput_tok_s > 0:
            metrics_parts.append(f"Throughput: {throughput_tok_s:.1f} tok/s")
        if metrics_parts:
            self._console.print("  ".join(metrics_parts), highlight=False)

    # -- ProgressCallback implementation --

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        with self._lock:
            phase = self._ensure_step_registered(step)
            # Clear any prior completion/skip/substeps for this step so a
            # re-fire (e.g. host dispatch failed and the experiment container
            # fell back to in-harness measurement) shows as an active spinner
            # instead of being masked by stale completed state.
            self._completed_steps.discard(step)
            self._skipped_steps.discard(step)
            self._step_data.pop(step, None)
            self._substeps.pop(step, None)
            self._active_substep.pop(step, None)
            self._active_step = step
            self._active_label = description or STEP_LABELS.get(step, step)
            self._active_detail = detail
            self._active_start = time.monotonic()

        if not self._is_tty:
            self._print_phase_header_if_new(phase)
            self._print_started_step(step, description or STEP_LABELS.get(step, step), detail)
        self._refresh()

    def on_step_update(self, step: str, detail: str) -> None:
        elapsed = 0.0
        with self._lock:
            if self._active_step == step:
                self._active_detail = detail
                elapsed = time.monotonic() - self._active_start
        if not self._is_tty and elapsed >= 1.0:
            # Only print updates in non-TTY for steps running > 1s
            # (avoids noisy duplicate lines for fast sub-second steps)
            self._print_update_line(step, detail)
        self._refresh()

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        with self._lock:
            label = self._active_label if self._active_step == step else STEP_LABELS.get(step, step)
            detail = self._active_detail if self._active_step == step else ""
            self._step_data[step] = (label, detail, elapsed_sec)
            self._completed_steps.add(step)
            if self._active_step == step:
                self._active_step = None
            # Freeze any dangling active substep so it doesn't keep animating
            # under a completed step. Uses its accumulated elapsed.
            dangling = self._active_substep.pop(step, None)
            if dangling is not None:
                d_text, d_start = dangling
                self._substeps.setdefault(step, []).append(
                    (d_text, max(0.0, time.monotonic() - d_start))
                )
        if not self._is_tty:
            self._print_completed_step(step, label, detail, elapsed_sec)
        self._refresh()

    def on_step_skip(self, step: str, reason: str = "") -> None:
        with self._lock:
            phase = self._ensure_step_registered(step)
            label = STEP_LABELS.get(step, step)
            # Store reason as detail, keep label as the verb
            self._step_data[step] = (label, reason or "-", 0.0)
            self._skipped_steps.add(step)

        if not self._is_tty:
            self._print_phase_header_if_new(phase)
            self._print_skipped_step(step, label, reason)
        self._refresh()

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Record a completed sub-operation within the active step.

        In TTY mode: stored and rendered as indented · lines below the parent step.
        In non-TTY mode: also printed immediately as they arrive.
        """
        with self._lock:
            if step not in self._substeps:
                self._substeps[step] = []
            self._substeps[step].append((text, elapsed_sec))
        if not self._is_tty:
            self._print_substep_line(step, text, elapsed_sec)
        self._refresh()

    def on_substep_start(self, step: str, text: str) -> None:
        """Begin a live sub-operation rendered with a dim spinner + counter."""
        with self._lock:
            # If a prior active substep for this step never got a matching
            # done, freeze it so the new active substep doesn't overwrite
            # silently. Uses its accumulated elapsed.
            prior = self._active_substep.pop(step, None)
            if prior is not None:
                prior_text, prior_start = prior
                self._substeps.setdefault(step, []).append(
                    (prior_text, max(0.0, time.monotonic() - prior_start))
                )
            self._active_substep[step] = (text, time.monotonic())
        if not self._is_tty:
            self._print_substep_line(step, text, 0.0)
        self._refresh()

    def on_substep_done(
        self,
        step: str,
        text: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Freeze the currently-active substep with final text + elapsed."""
        with self._lock:
            active = self._active_substep.pop(step, None)
            if active is None:
                # No matching start - fall through as a regular completed substep.
                final_text = text or ""
                final_elapsed = elapsed_sec if elapsed_sec is not None else 0.0
            else:
                start_text, start_ts = active
                final_text = text if text is not None else start_text
                final_elapsed = (
                    elapsed_sec
                    if elapsed_sec is not None
                    else max(0.0, time.monotonic() - start_ts)
                )
            if final_text:
                self._substeps.setdefault(step, []).append((final_text, final_elapsed))
        if not self._is_tty and final_text:
            self._print_substep_line(step, final_text, final_elapsed)
        self._refresh()

    # -- Heartbeat (non-TTY only) --

    def _heartbeat_loop(self) -> None:
        """Print periodic status for long-running steps (non-TTY mode)."""
        while not self._heartbeat_stop.wait(timeout=_HEARTBEAT_INTERVAL):
            with self._lock:
                if self._active_step is not None:
                    elapsed = time.monotonic() - self._active_start
                    if elapsed >= _HEARTBEAT_THRESHOLD:
                        phase = _phase_for_step(self._active_step)
                        idx = self._step_index_in_phase(self._active_step, phase)
                        total = self._phase_total(phase)
                        line = _step_line(
                            idx,
                            total,
                            self._active_label,
                            self._active_detail,
                            " ...",
                            _format_elapsed(elapsed),
                        )
                        self._console.print(line, highlight=False)

    # -- Rendering --

    def _step_index_in_phase(self, step: str, phase: str) -> int:
        """1-based index of step within its phase."""
        steps = self._phase_steps.get(phase, [])
        try:
            return steps.index(step) + 1
        except ValueError:
            return len(steps)

    def _phase_total(self, phase: str) -> int | None:
        """Total step count for a phase.

        Returns None in non-TTY mode when steps are auto-registered,
        to avoid misleading [1/1] -> [2/2] jitter as new steps arrive.
        TTY mode always returns a count (Live re-renders all lines).
        """
        if self._explicitly_registered or self._is_tty:
            return len(self._phase_steps.get(phase, []))
        return None

    def _render(self) -> Text:
        """Build current display state as a Rich Text renderable.

        Docker BuildKit-style: only show steps that have started, completed,
        or been skipped. Pending steps are NOT shown - they appear progressively
        as the harness reaches them. This gives a growing output that shows
        exactly where execution is.

        Colour coding:
        - Phase headers: bold white
        - Completed steps: green ✓ checkmark
        - Active steps: yellow braille spinner
        - Skipped steps: all dim grey
        - Substep lines: dim grey, indented with · prefix
        """
        lines = Text()
        with self._lock:
            for phase in self._phases:
                steps = self._phase_steps.get(phase, [])
                if not steps:
                    continue

                phase_total = len(steps)

                # Only show phase header if at least one step in this phase
                # has been started, completed, or skipped
                has_visible = any(
                    step in self._completed_steps
                    or step in self._skipped_steps
                    or step == self._active_step
                    for step in steps
                )
                if not has_visible:
                    continue

                lines.append(f"\n  {phase}\n", style="bold white")

                for step in steps:
                    idx = self._step_index_in_phase(step, phase)

                    if step in self._completed_steps:
                        label, detail, elapsed = self._step_data[step]
                        prefix = _step_line_prefix(idx, phase_total, label, detail)
                        lines.append(prefix)
                        lines.append("  ✓", style="bold green")
                        lines.append(f"  {_format_elapsed(elapsed)}\n")
                        _render_substep_lines(
                            lines,
                            self._substeps.get(step, []),
                            active=self._active_substep.get(step),
                        )
                    elif step in self._skipped_steps:
                        label, reason, _ = self._step_data[step]
                        prefix = _step_line_prefix(idx, phase_total, label, reason)
                        lines.append(prefix, style="dim")
                        lines.append("  SKIP", style="dim")
                        lines.append("\n")
                    elif step == self._active_step:
                        elapsed = time.monotonic() - self._active_start
                        frame_idx = int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)
                        spinner = _SPINNER_FRAMES[frame_idx]
                        prefix = _step_line_prefix(
                            idx, phase_total, self._active_label, self._active_detail
                        )
                        lines.append(prefix)
                        lines.append(f"  {spinner}", style="yellow")
                        lines.append(f"  {_format_elapsed(elapsed)}\n")
                        _render_substep_lines(
                            lines,
                            self._substeps.get(step, []),
                            active=self._active_substep.get(step),
                        )
                    # Pending steps: NOT shown (Docker BuildKit-style progressive output)

        return lines

    def _refresh(self) -> None:
        """Trigger immediate Live repaint (auto-refresh handles animation)."""
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.refresh()

    def _print_phase_header_if_new(self, phase: str) -> None:
        """Print phase header in non-TTY mode (once per phase)."""
        if phase not in self._printed_phases:
            self._printed_phases.add(phase)
            self._console.print(f"\n  {phase}", highlight=False)

    def _print_update_line(self, step: str, detail: str) -> None:
        """Print a step_update line in non-TTY mode (sub-step detail)."""
        phase = _phase_for_step(step)
        with self._lock:
            idx = self._step_index_in_phase(step, phase)
            total = self._phase_total(phase)
            label = self._active_label if self._active_step == step else STEP_LABELS.get(step, step)
            elapsed = time.monotonic() - self._active_start if self._active_step == step else 0.0
        line = _step_line(idx, total, label, detail, " ...", _format_elapsed(elapsed))
        self._console.print(line, highlight=False)

    def _print_started_step(self, step: str, label: str, detail: str) -> None:
        """Print a step start line (non-TTY mode) for immediate feedback."""
        phase = _phase_for_step(step)
        with self._lock:
            phase_total = self._phase_total(phase)
            idx = self._step_index_in_phase(step, phase)
        line = _step_line(idx, phase_total, label, detail, " ...", "")
        self._console.print(line, highlight=False)

    def _print_completed_step(self, step: str, label: str, detail: str, elapsed_sec: float) -> None:
        """Print a single completed step line (non-TTY mode)."""
        phase = _phase_for_step(step)
        with self._lock:
            phase_total = self._phase_total(phase)
            idx = self._step_index_in_phase(step, phase)
        line = _step_line(idx, phase_total, label, detail, "DONE", _format_elapsed(elapsed_sec))
        self._console.print(line, highlight=False)

    def _print_skipped_step(self, step: str, label: str, reason: str) -> None:
        """Print a skipped step line (non-TTY mode)."""
        phase = _phase_for_step(step)
        with self._lock:
            phase_total = self._phase_total(phase)
            idx = self._step_index_in_phase(step, phase)
        line = _step_line(idx, phase_total, label, reason or "-", "SKIP", "")
        self._console.print(line, highlight=False)

    def _print_substep_line(self, step: str, text: str, elapsed_sec: float) -> None:
        """Print a substep line in non-TTY mode (indented with · prefix)."""
        elapsed_str = f"  {_format_elapsed(elapsed_sec)}" if elapsed_sec > 0 else ""
        self._console.print(f"              · {text}{elapsed_str}", highlight=False)
