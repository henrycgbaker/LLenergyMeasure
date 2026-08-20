"""CLI-layer effective defaults for study execution.

The Pydantic ``ExecutionConfig`` defaults are deliberately conservative
(``n_cycles=1``, sequential order). The CLI applies research-appropriate
effective defaults (``n_cycles=3``, shuffle) for anything the study file does not
declare itself. Both ``llem run`` and ``llem study plan`` pass them into
``load_study``, so a plan previews exactly what a run executes. This stays at the
CLI layer on purpose - the library keeps its conservative defaults.

These are DEFAULTS, not overrides: they sit beneath the study file in the
precedence chain, so the file is the source of truth for anything it declares.
``resolve_study`` decides what the file declared from the parsed execution block,
so the CLI never re-reads the YAML to find out.
"""

from __future__ import annotations

from typing import Any, Final

__all__ = ["STUDY_EXECUTION_DEFAULTS"]

STUDY_EXECUTION_DEFAULTS: Final[dict[str, Any]] = {
    "n_cycles": 3,
    "experiment_order": "shuffle",
}
