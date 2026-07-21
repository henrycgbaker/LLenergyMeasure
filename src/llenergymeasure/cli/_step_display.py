"""Rich-based step display for CLI progress output.

Implements the ProgressCallback protocol from domain/progress.py. Renders
Docker-BuildKit-style hierarchical output with phase headers and numbered
sub-steps, using fixed step counts with SKIP for inapplicable steps.

TTY mode:    Rich Live for flicker-free in-place updates + spinner.
Non-TTY:     Phase headers + one line per completed/skipped step +
             heartbeat every 10s for long-running steps.

This module is a thin facade over the concern-split implementation:

- ``_step_render``        - formatting helpers + shared render primitives
- ``_experiment_display`` - ``StepDisplay`` (single experiment)
- ``_study_display``      - ``StudyStepDisplay`` + row/image NamedTuples

Import display types from here; the submodules are internal.
"""

from __future__ import annotations

from llenergymeasure.cli._experiment_display import StepDisplay
from llenergymeasure.cli._step_render import _VIEWPORT_RESERVED_LINES, _step_line
from llenergymeasure.cli._study_display import (
    StudyStepDisplay,
    _CompletedRow,
    _ImagePrepFailure,
    _ImagePrepResult,
)
from llenergymeasure.utils.formatting import format_elapsed as _format_elapsed

__all__ = [
    "_VIEWPORT_RESERVED_LINES",
    "StepDisplay",
    "StudyStepDisplay",
    "_CompletedRow",
    "_ImagePrepFailure",
    "_ImagePrepResult",
    "_format_elapsed",
    "_step_line",
]
