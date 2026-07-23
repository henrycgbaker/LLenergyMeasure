"""Formatting helpers and shared render primitives for the step displays.

Pure presentation helpers shared by ``StepDisplay`` (single experiment) and
``StudyStepDisplay`` (multi-experiment study): line formatters, the substep
renderer, spinner/heartbeat constants, the Rich auto-refresh proxy, and the
image-source provenance labels. No display state lives here.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from rich.console import Console, ConsoleOptions, Group, RenderResult
from rich.table import Table
from rich.text import Text

from llenergymeasure.config.ssot import RUNNER_CONTAINER, RUNNER_PROCESS
from llenergymeasure.domain.progress import PHASE_MEASUREMENT, STEP_PHASES
from llenergymeasure.utils.compat import StrEnum
from llenergymeasure.utils.formatting import format_elapsed as _format_elapsed
from llenergymeasure.utils.formatting import truncate_detail as _truncate_detail

# Braille spinner frames (same as Docker BuildKit / ora)
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_SPINNER_FPS = 8

# Heartbeat interval for non-TTY mode (seconds)
_HEARTBEAT_INTERVAL = 5.0

# Minimum step duration before heartbeat kicks in (seconds)
_HEARTBEAT_THRESHOLD = 3.0

# Lines to reserve when computing viewport height for the completed-rows table.
# Accounts for: study header (1), image prep block (~4), hidden indicator (1),
# table header (1), active experiment block (~8), gap (1), completion line (1).
_VIEWPORT_RESERVED_LINES = 12


class _DynamicRenderable:
    """Proxy that calls a render function on every Rich auto-refresh.

    Without this, ``Live`` holds a static ``Text`` snapshot and the
    spinner / elapsed counters freeze between callback events.
    """

    def __init__(self, render_fn: Callable[[], Text | Group | Table]) -> None:
        self._render_fn = render_fn

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self._render_fn()


def _step_line(
    idx: int,
    total: int | None,
    label: str,
    detail: str,
    status: str,
    elapsed_str: str,
) -> str:
    """Format a single step line.

    When total is None, the counter shows just ``[x]`` (no denominator).
    Detail is truncated to 34 chars to prevent line wrapping on 80-col terminals.
    """
    counter = f"[{idx}/{total}]" if total is not None else f"[{idx}]"
    detail = _truncate_detail(detail)
    return f"   {counter:>7s}  {label:<16s} {detail:<34s} {status:>4s}  {elapsed_str}"


def _step_line_prefix(
    idx: int,
    total: int | None,
    label: str,
    detail: str,
) -> str:
    """Format the counter/label/detail portion of a step line (without status/elapsed).

    Used by _render() to build styled output where status and elapsed are
    appended separately with colour styles. Non-TTY mode continues to use
    _step_line() which includes status and elapsed in the same string.
    """
    counter = f"[{idx}/{total}]" if total is not None else f"[{idx}]"
    detail = _truncate_detail(detail)
    return f"   {counter:>7s}  {label:<16s} {detail:<34s}"


def _render_substep_lines(
    lines: Text,
    substeps: list[tuple[str, float]],
    indent: str = "              ",
    active: tuple[str, float] | None = None,
) -> None:
    """Append substep lines (dim · prefix) to a Rich Text renderable.

    Shared between StepDisplay and StudyStepDisplay to avoid duplication.
    Frozen substeps render as dim ``· text ✓ elapsed``; the optional
    ``active`` substep (``(text, start_monotonic)``) renders with a dim
    spinner and rising elapsed counter so Rich Live animates it each frame.
    """
    for sub_text, sub_elapsed in substeps:
        lines.append(f"{indent}· {sub_text}", style="dim")
        if sub_elapsed > 0:
            lines.append("  ✓", style="dim")
            lines.append(f"  {_format_elapsed(sub_elapsed)}", style="dim")
        lines.append("\n")
    if active is not None:
        sub_text, start_ts = active
        elapsed = time.monotonic() - start_ts
        frame_idx = int(elapsed * _SPINNER_FPS) % len(_SPINNER_FRAMES)
        spinner = _SPINNER_FRAMES[frame_idx]
        lines.append(f"{indent}· {sub_text}", style="dim")
        lines.append(f"  {spinner}", style="dim")
        lines.append(f"  {_format_elapsed(elapsed)}", style="dim")
        lines.append("\n")


class ImageSource(StrEnum):
    """Provenance values for Docker image selection."""

    LOCAL_BUILD = "local_build"
    REGISTRY = "registry"
    REGISTRY_CACHED = "registry_cached"
    ENV = "env"
    YAML = "yaml"
    RUNNER_OVERRIDE = "runner_override"
    USER_CONFIG = "user_config"


_IMAGE_SOURCE_LABELS: dict[ImageSource, str] = {
    ImageSource.LOCAL_BUILD: "LOCAL BUILD - current source tree (via docker compose build)",
    ImageSource.REGISTRY: "REGISTRY - versioned release image",
    ImageSource.REGISTRY_CACHED: "REGISTRY - cached locally from prior pull",
    ImageSource.ENV: "OVERRIDE - image set via environment variable",
    ImageSource.YAML: "OVERRIDE - image set in study YAML images: section",
    ImageSource.RUNNER_OVERRIDE: "OVERRIDE - image set via container:<image> in runners:",
    ImageSource.USER_CONFIG: "OVERRIDE - image set in user config (~/.config/llenergymeasure/config.yaml)",
}

_RUNNER_SOURCE_LABELS: dict[str, str] = {
    "env": "env var",
    "yaml": "study YAML",
    "user_config": "user config",
    "auto_detected": "auto-detected",
    "default": "default",
    "multi_engine_elevation": "multi-engine auto-elevation",
}


def _render_runner_info(lines: Text, info: dict[str, str | None]) -> None:
    """Render runner/image provenance lines below the experiment header."""
    mode = info.get("mode", "unknown")
    source = info.get("source", "")
    image = info.get("image")
    image_source = info.get("image_source")

    source_label = _RUNNER_SOURCE_LABELS.get(source or "", source or "")

    if mode == RUNNER_PROCESS:
        lines.append(f"       mode:    process ({source_label})\n", style="dim")
        lines.append(
            "               no container isolation - running directly on host\n", style="dim"
        )
    elif mode == RUNNER_CONTAINER and image:
        lines.append(f"       mode:    container ({source_label})\n", style="dim")
        lines.append(f"       image:   {image}\n", style="dim")
        if image_source:
            try:
                key = ImageSource(image_source)
                detail = _IMAGE_SOURCE_LABELS.get(key, image_source)
            except ValueError:
                detail = image_source
            lines.append(f"               {detail}\n", style="dim")


def _phase_for_step(step: str) -> str:
    """Look up the phase for a step name, defaulting to Measurement."""
    return STEP_PHASES.get(step, PHASE_MEASUREMENT)
