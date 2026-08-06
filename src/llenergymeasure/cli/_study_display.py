"""Study-level step display (Rich ProgressCallback implementation).

Renders a multi-experiment study: a completed-experiment results table, the
active experiment's nested step progress, and study-level Docker image
preparation. Split out of the former ``_step_display`` god-module; the
``_step_display`` facade re-exports ``StudyStepDisplay`` and the row/image
NamedTuples so callers are unchanged.
"""

from __future__ import annotations

import contextlib
import threading
import time
from typing import NamedTuple

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

from llenergymeasure.cli._step_render import (
    _SPINNER_FPS,
    _SPINNER_FRAMES,
    _VIEWPORT_RESERVED_LINES,
    _DynamicRenderable,
    _render_substep_lines,
)
from llenergymeasure.config.ssot import ENV_TABLE_ROWS
from llenergymeasure.domain.progress import STEP_LABELS
from llenergymeasure.utils.formatting import format_elapsed as _format_elapsed
from llenergymeasure.utils.formatting import short_name as _short_image
from llenergymeasure.utils.formatting import truncate_detail as _truncate_detail


class _ImagePrepResult(NamedTuple):
    """Result of a successfully prepared Docker image.

    ``idx`` is the shared monotonic display position assigned when the event
    arrives, so successes and failures render with unique, stable counters
    regardless of the order concurrent pulls finish in.
    """

    idx: int
    engine: str
    image: str
    cached: bool
    elapsed: float
    metadata: dict[str, str] | None


class _ImagePrepFailure(NamedTuple):
    """Result of a failed Docker image preparation.

    ``idx`` is the shared monotonic display position (see ``_ImagePrepResult``).
    """

    idx: int
    engine: str
    image: str
    error: str


class _CompletedRow(NamedTuple):
    """One completed or historical experiment row in the study results table.

    The first field is ``idx`` (not ``index``) because NamedTuple inherits the
    ``tuple.index`` method and a same-named field would shadow it.
    """

    idx: int
    status: str
    config: str
    elapsed: float
    inference_sec: float | None
    energy_j: float | None
    adj_energy_j: float | None
    throughput: float | None
    energy_per_token_mj: float | None


class StudyStepDisplay:
    """Step display for study mode using a Rich Table for completed experiments.

    Completed experiments appear as table rows with Config, Time, Energy, tok/s columns.
    The active experiment shows nested step progress below the table (deferred for
    multi-process study runs - see module docstring).

    Thread-safe: event methods may be called from worker threads.
    """

    def __init__(
        self,
        total_experiments: int,
        study_name: str = "",
        n_cycles: int = 1,
        console: Console | None = None,
        force_plain: bool = False,
    ) -> None:
        # Study results stream to stdout (the scientific record) so a study run
        # can be redirected or piped and still capture the results table. Setup
        # chrome (preflight panel, expansion spinner) stays on stderr in run.py.
        self._console = console or Console()
        self._total = total_experiments
        self._study_name = study_name
        self._n_cycles = n_cycles
        self._is_tty = self._console.is_terminal and not force_plain
        self._lock = threading.Lock()

        self._completed_rows: list[_CompletedRow] = []

        # Active experiment state
        self._active_index: int = 0
        self._active_header: str = ""
        self._inner_completed: list[tuple[str, str, str, float]] = []
        self._inner_active: tuple[str, str, str, float] | None = None  # step, label, detail, start
        self._inner_steps: list[str] = []
        self._inner_skipped: dict[str, str] = {}  # step -> reason
        self._inner_substeps: dict[str, list[tuple[str, float]]] = {}
        # Active substep per step: (text, start_monotonic) - same spinner
        # heartbeat treatment as StepDisplay, used for live baseline stages.
        self._inner_active_substep: dict[str, tuple[str, float]] = {}
        self._runner_info: dict[str, str | None] | None = None

        # Per-experiment save paths: (index, host_path, container_path | None)
        self._saved_paths: list[tuple[int, str, str | None]] = []

        self._live: Live | None = None
        self._total_start: float = 0.0
        self._gap_text: str = ""

        # Image prep state (study-level Docker image preparation). Successes and
        # failures share one monotonically increasing display index
        # (``_image_prep_seq``, bumped under ``_lock``) so concurrent pulls that
        # finish or fail in any order still render with unique, stable
        # [x/total] counters. Failures are a list, not a single slot: pulls no
        # longer cancel their siblings, so 2+ images can fail and every failure
        # must stay visible in the live panel.
        self._image_prep_active: bool = False
        self._image_prep_total: int = 0
        self._image_prep_seq: int = 0
        self._image_prep_done: list[_ImagePrepResult] = []
        self._image_prep_failed: list[_ImagePrepFailure] = []

    def start(self, *, print_header: bool = True) -> None:
        """Begin the display. Optionally prints study header and starts Rich Live if TTY.

        Args:
            print_header: When False, suppresses the header line (caller prints it
                separately, e.g. with a preflight summary in between).
        """
        self._total_start = time.monotonic()
        if print_header:
            # Print study header
            header = f"Study: {self._study_name}" if self._study_name else "Study"
            header += f" | {self._total} experiments | {self._n_cycles} cycles"
            self._console.print(header, highlight=False)
        if self._is_tty:
            self._live = Live(
                _DynamicRenderable(self._render),
                console=self._console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()

    def add_historical_rows(self, rows: list[_CompletedRow]) -> None:
        """Pre-populate the completed table with rows from a previous run.

        Used by --resume to show previously completed experiments in the same
        table format as live results. Must be called before start().

        Each row's status ("OK" / "FAIL") is prefixed to "PREV_OK" / "PREV_FAIL"
        so historical rows render dimmed with a distinct marker, visually
        separating them from live results.
        """
        with self._lock:
            for row in rows:
                self._completed_rows.append(row._replace(status=f"PREV_{row.status}"))

    def stop(self) -> None:
        """Stop Rich Live."""
        if self._live is not None:
            self._live.stop()
            self._live = None

    def begin_experiment(
        self,
        index: int,
        header: str,
        steps: list[str],
        runner_info: dict[str, str | None] | None = None,
    ) -> None:
        """Start tracking a new experiment within the study."""
        with self._lock:
            self._active_index = index
            self._active_header = header
            self._inner_completed = []
            self._inner_active = None
            self._inner_steps = steps
            self._inner_skipped = {}
            self._inner_substeps = {}
            self._inner_active_substep = {}
            self._runner_info = runner_info
        self._refresh()

    def end_experiment_ok(
        self,
        index: int,
        elapsed: float,
        energy_j: float | None = None,
        throughput_tok_s: float | None = None,
        inference_time_sec: float | None = None,
        adj_energy_j: float | None = None,
        energy_per_token_mj_adjusted: float | None = None,
        energy_per_token_mj_total: float | None = None,
    ) -> None:
        """Mark experiment as successfully completed."""
        # Prefer energy_per_token_mj_adjusted (baseline-subtracted) when available,
        # fall back to energy_per_token_mj_total. No recomputation - show "-" if both null.
        mj_tok: float | None
        if energy_per_token_mj_adjusted is not None:
            mj_tok = energy_per_token_mj_adjusted
        elif energy_per_token_mj_total is not None:
            mj_tok = energy_per_token_mj_total
        else:
            mj_tok = None

        with self._lock:
            self._inner_active = None
            row = _CompletedRow(
                index,
                "OK",
                self._active_header,
                elapsed,
                inference_time_sec,
                energy_j,
                adj_energy_j,
                throughput_tok_s,
                mj_tok,
            )
            self._completed_rows.append(row)
        if not self._is_tty:
            self._print_completed_row(*row)
        self._refresh()

    def end_experiment_fail(self, index: int, elapsed: float, error: str = "") -> None:
        """Mark experiment as failed."""
        with self._lock:
            self._inner_active = None
            row = _CompletedRow(
                index, "FAIL", self._active_header, elapsed, None, None, None, None, None
            )
            self._completed_rows.append(row)
        if not self._is_tty:
            self._print_completed_row(*row)
            if error:
                self._console.print(f"         {error}", highlight=False)
        self._refresh()

    def finish(
        self,
        save_path: str | None = None,
        total_elapsed: float | None = None,
    ) -> None:
        """Print study completion footer with final results table.

        When Live was active (TTY mode), the table is already on screen from the
        live display, so only the completion line and save path are printed.
        When used post-hoc (no Live), prints the full table.

        Args:
            save_path: Optional path to saved results directory.
            total_elapsed: Total elapsed time in seconds. If None, falls back to
                monotonic clock delta from start() (which may be wrong if start()
                was never called - callers constructing post-hoc should always pass this).
        """
        was_live = self._live is not None
        self.stop()
        if total_elapsed is not None:
            total = total_elapsed
        elif self._total_start > 0:
            total = time.monotonic() - self._total_start
        else:
            total = 0.0
        self._console.print(f"\nStudy completed in {_format_elapsed(total)}", highlight=False)

        # Only print table in post-hoc mode - Live already shows it on screen
        if not was_live and self._completed_rows:
            table, _hidden = self._build_table()
            self._console.print(table)

        # Print study results directory
        if save_path:
            self._console.print(f"\n  Results: {save_path}", style="dim", highlight=False)

        # Print per-experiment save paths (only in TTY mode - non-TTY prints inline)
        if was_live and self._saved_paths:
            for idx, host_path, container_path in self._saved_paths:
                if container_path:
                    self._console.print(
                        f"  [{idx}] container: {container_path}", style="dim", highlight=False
                    )
                self._console.print(
                    f"  [{idx}] host:      {host_path}", style="dim", highlight=False
                )

    # -- ProgressCallback for inner steps --

    def on_step_start(self, step: str, description: str, detail: str = "") -> None:
        with self._lock:
            # Clear any prior completion/skip/substeps for this step. A re-fire
            # means "this step is running again" (e.g. host dispatch failed and
            # the experiment container fell back to in-harness measurement), so
            # stale state from a prior attempt must not mask the new active
            # spinner - the renderer checks completed_map before _inner_active
            # and would otherwise hide the re-run.
            self._inner_completed = [c for c in self._inner_completed if c[0] != step]
            self._inner_skipped.pop(step, None)
            self._inner_substeps.pop(step, None)
            self._inner_active_substep.pop(step, None)
            label = description or STEP_LABELS.get(step, step)
            self._inner_active = (step, label, detail, time.monotonic())
        self._refresh()

    def on_step_update(self, step: str, detail: str) -> None:
        with self._lock:
            if self._inner_active and self._inner_active[0] == step:
                self._inner_active = (step, self._inner_active[1], detail, self._inner_active[3])
        self._refresh()

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        with self._lock:
            if self._inner_active and self._inner_active[0] == step:
                label = self._inner_active[1]
                detail = self._inner_active[2]
                self._inner_completed.append((step, label, detail, elapsed_sec))
                self._inner_active = None
            # Freeze any dangling active substep so it doesn't keep animating
            # under a completed step.
            dangling = self._inner_active_substep.pop(step, None)
            if dangling is not None:
                d_text, d_start = dangling
                self._inner_substeps.setdefault(step, []).append(
                    (d_text, max(0.0, time.monotonic() - d_start))
                )
        self._refresh()

    def on_step_skip(self, step: str, reason: str = "") -> None:
        """Record a skipped step (rendered dim grey in the step list)."""
        with self._lock:
            self._inner_skipped[step] = reason or "-"
        self._refresh()

    def on_substep(self, step: str, text: str, elapsed_sec: float = 0.0) -> None:
        """Record a completed sub-operation within the active step."""
        with self._lock:
            if step not in self._inner_substeps:
                self._inner_substeps[step] = []
            self._inner_substeps[step].append((text, elapsed_sec))
        self._refresh()

    def on_substep_start(self, step: str, text: str) -> None:
        """Begin a live sub-operation rendered with a dim spinner + counter."""
        with self._lock:
            prior = self._inner_active_substep.pop(step, None)
            if prior is not None:
                prior_text, prior_start = prior
                self._inner_substeps.setdefault(step, []).append(
                    (prior_text, max(0.0, time.monotonic() - prior_start))
                )
            self._inner_active_substep[step] = (text, time.monotonic())
        self._refresh()

    def on_substep_done(
        self,
        step: str,
        text: str | None = None,
        elapsed_sec: float | None = None,
    ) -> None:
        """Freeze the currently-active substep with final text + elapsed."""
        with self._lock:
            active = self._inner_active_substep.pop(step, None)
            if active is None:
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
                self._inner_substeps.setdefault(step, []).append((final_text, final_elapsed))
        self._refresh()

    def on_experiment_saved(
        self, index: int, host_path: str, container_path: str | None = None
    ) -> None:
        """Display save path info after experiment results are written to disk."""
        with self._lock:
            self._saved_paths.append((index, host_path, container_path))
        if not self._is_tty:
            if container_path:
                self._console.print(f"         \u00b7 container: {container_path}", highlight=False)
            self._console.print(f"         \u00b7 host:      {host_path}", highlight=False)

    # -- Gap display --

    def show_gap(self, text: str) -> None:
        """Show a gap countdown line below the table (e.g. 'Experiment gap: 7s')."""
        with self._lock:
            self._gap_text = text
        self._refresh()

    def clear_gap(self) -> None:
        """Clear the gap countdown line."""
        with self._lock:
            self._gap_text = ""
        self._refresh()

    # -- Image prep (study-level Docker image preparation) --

    def begin_image_prep(self, engines: list[str]) -> None:
        """Signal the start of study-level Docker image preparation."""
        with self._lock:
            self._image_prep_active = True
            self._image_prep_total = len(engines)
        if not self._is_tty:
            self._console.print("\n  Preparing Docker images", highlight=False)
        self._refresh()

    def image_ready(
        self,
        engine: str,
        image: str,
        cached: bool,
        elapsed: float,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Signal that a Docker image is ready."""
        with self._lock:
            self._image_prep_seq += 1
            idx = self._image_prep_seq
            self._image_prep_done.append(
                _ImagePrepResult(idx, engine, image, cached, elapsed, metadata)
            )
        if not self._is_tty:
            total = self._image_prep_total
            status = "cached" if cached else "pulled"
            short_img = _short_image(image)
            line = f"      [{idx}/{total}]  {engine:<16s}{short_img} ({status})"
            line += f"  \u2713  {_format_elapsed(elapsed)}"
            self._console.print(line, highlight=False)
            if metadata:
                parts = [f"{k}: {v}" for k, v in metadata.items()]
                meta_text = " \u00b7 ".join(parts)
                self._console.print(
                    f"                    \u00b7 {meta_text}",
                    style="dim",
                    highlight=False,
                )
        self._refresh()

    def image_failed(self, engine: str, image: str, error: str) -> None:
        """Signal that a Docker image could not be prepared."""
        with self._lock:
            self._image_prep_seq += 1
            idx = self._image_prep_seq
            self._image_prep_failed.append(_ImagePrepFailure(idx, engine, image, error))
        if not self._is_tty:
            total = self._image_prep_total
            short_img = _short_image(image)
            self._console.print(
                f"      [{idx}/{total}]  {engine:<16s}{short_img}  \u2717",
                highlight=False,
            )
            self._console.print(
                f"                    \u00b7 {error}",
                style="dim",
                highlight=False,
            )
        self._refresh()

    def end_image_prep(self) -> None:
        """Signal the end of study-level Docker image preparation."""
        with self._lock:
            self._image_prep_active = False
        self._refresh()

    def _render_image_prep(self) -> Text:
        """Render the Docker image preparation section."""
        lines = Text()
        if not self._image_prep_done and not self._image_prep_failed:
            if self._image_prep_active:
                lines.append("\n  Preparing Docker images\n", style="bold")
            return lines

        lines.append("\n  Preparing Docker images\n")
        total = self._image_prep_total

        # Successes and failures share one monotonic counter, so render them
        # interleaved in arrival order for a coherent [x/total] sequence. Every
        # failure is shown - concurrent pulls can produce more than one.
        successes = {r.idx: r for r in self._image_prep_done}
        failures = {f.idx: f for f in self._image_prep_failed}
        for idx in sorted(successes.keys() | failures.keys()):
            counter = f"[{idx}/{total}]"
            if idx in successes:
                r = successes[idx]
                status = "cached" if r.cached else "pulled"
                short_img = _short_image(r.image)
                lines.append(f"      {counter:>7s}  {r.engine:<16s}{short_img} ({status})")
                lines.append("  \u2713", style="bold green")
                lines.append(f"  {_format_elapsed(r.elapsed)}\n")
                if r.metadata:
                    parts = [f"{k}: {v}" for k, v in r.metadata.items()]
                    meta_text = " \u00b7 ".join(parts)
                    lines.append(f"                    \u00b7 {meta_text}\n", style="dim")
            else:
                f = failures[idx]
                short_img = _short_image(f.image)
                lines.append(f"      {counter:>7s}  {f.engine:<16s}{short_img}")
                lines.append("  \u2717", style="bold red")
                lines.append("\n")
                lines.append(f"                    \u00b7 {f.error}\n", style="dim")

        return lines

    # -- Rendering --

    def _viewport_size(self) -> int:
        """Maximum number of completed rows visible in the terminal.

        Respects LLEM_TABLE_ROWS env var if set (overrides terminal height calc).
        """
        import os

        env_rows = os.environ.get(ENV_TABLE_ROWS)
        if env_rows:
            try:
                return max(3, int(env_rows))
            except ValueError:
                pass
        return max(5, int(self._console.size.height) - _VIEWPORT_RESERVED_LINES)

    def _build_table(self) -> tuple[Table, int]:
        """Build the Rich Table of completed experiments with viewport limiting.

        Returns (table, hidden_count).
        """
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("#", width=3, justify="right")
        table.add_column("", width=2)
        table.add_column("Config", max_width=45, overflow="ellipsis", no_wrap=True)
        table.add_column("Total", justify="right")
        table.add_column("Infer", justify="right")
        table.add_column("Energy", justify="right")
        table.add_column("Adj. E", justify="right")
        table.add_column("tok/s", justify="right")
        table.add_column("mJ/tok", justify="right")
        rows = self._completed_rows
        available = self._viewport_size()
        hidden = max(0, len(rows) - available)
        visible = rows[max(0, len(rows) - available) :]
        for (
            idx,
            status,
            config,
            elapsed,
            infer_sec,
            energy,
            adj_energy,
            throughput,
            mj_tok,
        ) in visible:
            is_historical = status.startswith("PREV_")
            if status == "OK":
                status_text = Text("\u2713", style="bold green")
            elif status == "PREV_OK":
                status_text = Text("\u2713", style="dim green")
            elif status == "PREV_FAIL":
                status_text = Text("\u2717", style="dim red")
            else:
                status_text = Text("\u2717", style="bold red")
            row_style = "dim" if is_historical else None
            infer_str = _format_elapsed(infer_sec) if infer_sec is not None else "-"
            energy_str = f"{energy:.1f} J" if energy is not None else "-"
            adj_energy_str = f"{adj_energy:.1f} J" if adj_energy is not None else "-"
            throughput_str = f"{throughput:.1f}" if throughput is not None else "-"
            mj_str = f"{mj_tok:.1f}" if mj_tok is not None else "-"
            table.add_row(
                str(idx),
                status_text,
                config,
                _format_elapsed(elapsed),
                infer_str,
                energy_str,
                adj_energy_str,
                throughput_str,
                mj_str,
                style=row_style,
            )
        return table, hidden

    def _render_active_steps(self) -> Text:
        """Render the active experiment's step progress.

        Iterates registered steps in order so completed, skipped, and active
        steps all appear at their correct [x/N] position. Skipped steps
        render dim grey with SKIP label (mirrors StepDisplay behaviour).
        Pending steps are not shown (Docker BuildKit-style progressive output).
        """
        lines = Text()
        if not self._active_header:
            return lines

        lines.append(f"\n  [{self._active_index}/{self._total}] {self._active_header}\n")

        inner_total = len(self._inner_steps) or (
            len(self._inner_completed) + len(self._inner_skipped) + (1 if self._inner_active else 0)
        )

        # Index completed steps by name for O(1) lookup while preserving order
        completed_map: dict[str, tuple[str, str, float]] = {}
        for step, label, detail, elapsed in self._inner_completed:
            completed_map[step] = (label, detail, elapsed)

        idx = 0
        for step in self._inner_steps:
            if step in completed_map:
                idx += 1
                label, detail, elapsed = completed_map[step]
                counter = f"[{idx}/{inner_total}]"
                trunc_detail = _truncate_detail(detail)
                lines.append(f"      {counter:>7s}  {label:<16s} {trunc_detail:<34s}")
                lines.append("  \u2713", style="bold green")
                lines.append(f"  {_format_elapsed(elapsed)}\n")
                _render_substep_lines(
                    lines,
                    self._inner_substeps.get(step, []),
                    indent="                    ",
                    active=self._inner_active_substep.get(step),
                )
            elif step in self._inner_skipped:
                idx += 1
                label = STEP_LABELS.get(step, step)
                reason = self._inner_skipped[step]
                counter = f"[{idx}/{inner_total}]"
                trunc_reason = _truncate_detail(reason)
                lines.append(
                    f"      {counter:>7s}  {label:<16s} {trunc_reason:<34s}  SKIP\n",
                    style="dim",
                )
            elif self._inner_active and self._inner_active[0] == step:
                idx += 1
                _step, label, detail, start = self._inner_active
                elapsed = time.monotonic() - start
                frame_idx = int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)
                spinner = _SPINNER_FRAMES[frame_idx]
                counter = f"[{idx}/{inner_total}]"
                trunc_detail = _truncate_detail(detail)
                lines.append(f"      {counter:>7s}  {label:<16s} {trunc_detail:<34s}")
                lines.append(f"  {spinner}", style="yellow")
                lines.append(f"  {_format_elapsed(elapsed)}\n")
                _render_substep_lines(
                    lines,
                    self._inner_substeps.get(step, []),
                    indent="                    ",
                    active=self._inner_active_substep.get(step),
                )
            # Pending steps: not shown (Docker BuildKit-style progressive output)

        return lines

    def _render(self) -> Group:
        """Render image prep + hidden-row indicator + completed experiments table + active steps + gap."""
        with self._lock:
            image_prep = self._render_image_prep()
            table, hidden = self._build_table()
            step_text = self._render_active_steps()
            gap = Text(f"\n  {self._gap_text}", style="dim") if self._gap_text else Text("")
            if hidden > 0:
                indicator = Text(f"  ({hidden} earlier results not shown)\n", style="dim")
            else:
                indicator = Text("")
        return Group(image_prep, indicator, table, step_text, gap)

    def _refresh(self) -> None:
        """Trigger immediate Live repaint (auto-refresh handles animation)."""
        if self._live is not None:
            with contextlib.suppress(Exception):
                self._live.refresh()

    def _print_completed_row(
        self,
        index: int,
        status: str,
        config: str,
        elapsed: float,
        inference_sec: float | None,
        energy: float | None,
        adj_energy: float | None,
        throughput: float | None,
        mj_tok: float | None = None,
    ) -> None:
        """Print a completed experiment row in non-TTY mode."""
        status_icon = "\u2713" if status == "OK" else "\u2717"
        infer_str = f"  infer={_format_elapsed(inference_sec)}" if inference_sec is not None else ""
        energy_str = f"  {energy:.1f} J" if energy is not None else ""
        adj_energy_str = f"  adj={adj_energy:.1f} J" if adj_energy is not None else ""
        throughput_str = f"  {throughput:.1f} tok/s" if throughput is not None else ""
        mj_str = f"  {mj_tok:.1f} mJ/tok" if mj_tok is not None else ""
        line = (
            f" [{index:>2d}/{self._total}]  {status_icon}  {config:<42s}"
            f" {_format_elapsed(elapsed):>8s}{infer_str}{energy_str}{adj_energy_str}"
            f"{throughput_str}{mj_str}"
        )
        self._console.print(line, highlight=False)
