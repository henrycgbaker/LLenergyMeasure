"""Progress tracking with tqdm and verbosity awareness.

Provides progress bars for inference batches and experiment cycles,
with proper multi-process handling and verbosity levels.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import Enum, auto
from typing import TYPE_CHECKING, TextIO

from tqdm import tqdm

if TYPE_CHECKING:
    from tqdm.std import tqdm as TqdmType


class VerbosityLevel(Enum):
    """Output verbosity levels for CLI."""

    QUIET = auto()  # Warnings only, no progress bars
    NORMAL = auto()  # Progress bars + warnings, simplified logs
    VERBOSE = auto()  # Full logs with timestamps (current behaviour)


def get_verbosity_from_env() -> VerbosityLevel:
    """Get verbosity level from environment variable.

    Environment variable LLM_ENERGY_VERBOSITY can be:
    - "quiet": Warnings only
    - "normal": Default progress + simplified logs
    - "verbose": Full debug logs

    Returns:
        VerbosityLevel from environment or NORMAL default.
    """
    env_value = os.environ.get("LLM_ENERGY_VERBOSITY", "normal").lower()
    mapping = {
        "quiet": VerbosityLevel.QUIET,
        "normal": VerbosityLevel.NORMAL,
        "verbose": VerbosityLevel.VERBOSE,
    }
    return mapping.get(env_value, VerbosityLevel.NORMAL)


class ProgressTracker:
    """tqdm wrapper with verbosity and multi-process awareness.

    Only shows progress on main process (process_index=0).
    Uses tqdm.write() for warnings to preserve progress bar.

    Example:
        with ProgressTracker(len(batches), "Batches", is_main_process=True) as progress:
            for batch in batches:
                result = process_batch(batch)
                progress.update(1, latency_ms=result.latency_ms)
    """

    def __init__(
        self,
        total: int,
        desc: str,
        is_main_process: bool = True,
        verbosity: VerbosityLevel | None = None,
        unit: str = "it",
        leave: bool = True,
        position: int | None = None,
        file: TextIO | None = None,
    ) -> None:
        """Initialise progress tracker.

        Args:
            total: Total number of items to track.
            desc: Description shown on progress bar.
            is_main_process: Only show progress on main process.
            verbosity: Override verbosity level (default: from env).
            unit: Unit of iteration (e.g., "batch", "prompt").
            leave: Keep progress bar after completion.
            position: Position for nested progress bars (0=outer, 1=inner).
            file: File to write to (default: stderr for tqdm).
        """
        self._total = total
        self._desc = desc
        self._is_main_process = is_main_process
        self._verbosity = verbosity or get_verbosity_from_env()
        self._unit = unit
        self._leave = leave
        self._position = position
        self._file: TextIO = file or sys.stderr
        self._pbar: TqdmType | None = None
        self._current = 0
        self._latencies: list[float] = []

    @property
    def should_show(self) -> bool:
        """Whether to display progress bar."""
        return self._is_main_process and self._verbosity != VerbosityLevel.QUIET and self._total > 0

    def __enter__(self) -> ProgressTracker:
        """Start progress tracking."""
        if self.should_show:
            self._pbar = tqdm(
                total=self._total,
                desc=self._desc,
                unit=self._unit,
                leave=self._leave,
                position=self._position,
                file=self._file,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Finish progress tracking."""
        if self._pbar is not None:
            self._pbar.close()

    def update(self, n: int = 1, latency_ms: float | None = None) -> None:
        """Update progress by n items.

        Args:
            n: Number of items completed.
            latency_ms: Optional latency measurement to track.
        """
        self._current += n
        if latency_ms is not None:
            self._latencies.append(latency_ms)

        if self._pbar is not None:
            # Add latency to postfix if available
            if self._latencies:
                avg_latency = sum(self._latencies) / len(self._latencies)
                self._pbar.set_postfix({"avg_ms": f"{avg_latency:.1f}"}, refresh=False)
            self._pbar.update(n)

    def warning(self, msg: str) -> None:
        """Display warning inline with progress bar.

        Uses tqdm.write() to preserve progress bar position.

        Args:
            msg: Warning message to display.
        """
        formatted = f"\u26a0 {msg}"  # ⚠ prefix
        if self._pbar is not None:
            tqdm.write(formatted, file=self._file)
        elif self._is_main_process and self._verbosity != VerbosityLevel.QUIET:
            print(formatted, file=self._file)

    def info(self, msg: str) -> None:
        """Display info message inline with progress bar.

        Args:
            msg: Info message to display.
        """
        if self._pbar is not None:
            tqdm.write(msg, file=self._file)
        elif self._is_main_process and self._verbosity == VerbosityLevel.VERBOSE:
            print(msg, file=self._file)

    def set_description(self, desc: str) -> None:
        """Update progress bar description.

        Args:
            desc: New description.
        """
        if self._pbar is not None:
            self._pbar.set_description(desc)

    def set_postfix(self, **kwargs: object) -> None:
        """Update progress bar postfix.

        Args:
            **kwargs: Key-value pairs for postfix.
        """
        if self._pbar is not None:
            self._pbar.set_postfix(kwargs)


class CycleProgress:
    """Progress tracking for multi-cycle experiment runs.

    Shows outer progress for cycles with inner progress for batches.
    """

    def __init__(
        self,
        total_cycles: int,
        verbosity: VerbosityLevel | None = None,
    ) -> None:
        """Initialise cycle progress tracker.

        Args:
            total_cycles: Total number of cycles.
            verbosity: Override verbosity level.
        """
        self._total_cycles = total_cycles
        self._verbosity = verbosity or get_verbosity_from_env()
        self._current_cycle = 0
        self._pbar: TqdmType | None = None

    @property
    def should_show(self) -> bool:
        """Whether to display progress."""
        return self._verbosity != VerbosityLevel.QUIET and self._total_cycles > 1

    def __enter__(self) -> CycleProgress:
        """Start cycle tracking."""
        if self.should_show:
            self._pbar = tqdm(
                total=self._total_cycles,
                desc="Cycles",
                unit="cycle",
                leave=True,
                position=0,
                file=sys.stderr,
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Finish cycle tracking."""
        if self._pbar is not None:
            self._pbar.close()

    def advance(self, cycle_id: int | None = None) -> None:
        """Advance to next cycle.

        Args:
            cycle_id: Optional cycle ID for display.
        """
        self._current_cycle += 1
        if self._pbar is not None:
            self._pbar.update(1)

    def warning(self, msg: str) -> None:
        """Display warning inline with progress bar.

        Args:
            msg: Warning message.
        """
        formatted = f"\u26a0 {msg}"
        if self._pbar is not None:
            tqdm.write(formatted, file=sys.stderr)
        elif self._verbosity != VerbosityLevel.QUIET:
            print(formatted, file=sys.stderr)

    def info(self, msg: str) -> None:
        """Display info message.

        Args:
            msg: Info message.
        """
        if self._pbar is not None:
            tqdm.write(msg, file=sys.stderr)
        elif self._verbosity == VerbosityLevel.VERBOSE:
            print(msg, file=sys.stderr)


@contextmanager
def batch_progress(
    total: int,
    desc: str = "Batches",
    is_main_process: bool = True,
    verbosity: VerbosityLevel | None = None,
    position: int | None = None,
) -> Iterator[ProgressTracker]:
    """Context manager for batch progress tracking.

    Convenience wrapper around ProgressTracker.

    Args:
        total: Total number of batches.
        desc: Description (default: "Batches").
        is_main_process: Only show on main process.
        verbosity: Override verbosity level.
        position: Position for nested bars.

    Yields:
        ProgressTracker instance.
    """
    tracker = ProgressTracker(
        total=total,
        desc=desc,
        is_main_process=is_main_process,
        verbosity=verbosity,
        unit="batch",
        position=position,
    )
    with tracker:
        yield tracker


@contextmanager
def prompt_progress(
    total: int,
    desc: str = "Prompts",
    is_main_process: bool = True,
    verbosity: VerbosityLevel | None = None,
    position: int | None = None,
) -> Iterator[ProgressTracker]:
    """Context manager for prompt progress tracking.

    Convenience wrapper for prompt-level progress (vLLM, TensorRT).

    Args:
        total: Total number of prompts.
        desc: Description (default: "Prompts").
        is_main_process: Only show on main process.
        verbosity: Override verbosity level.
        position: Position for nested bars.

    Yields:
        ProgressTracker instance.
    """
    tracker = ProgressTracker(
        total=total,
        desc=desc,
        is_main_process=is_main_process,
        verbosity=verbosity,
        unit="prompt",
        position=position,
    )
    with tracker:
        yield tracker
