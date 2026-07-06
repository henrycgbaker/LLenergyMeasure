"""CLI-layer effective defaults for study execution.

The Pydantic ``StudyExecution`` defaults are deliberately conservative
(``n_cycles=1``, sequential order). The CLI applies research-appropriate
effective defaults (``n_cycles=3``, shuffle) unless the study file already sets
them. Both ``llem run`` and ``llem study plan`` apply these identically, so a
plan previews exactly what a run executes. This stays at the CLI layer on
purpose - the library keeps its conservative defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["build_study_cli_overrides", "study_cli_overrides_for_file"]


def build_study_cli_overrides(yaml_execution: dict[str, Any]) -> dict[str, Any]:
    """Build the CLI-layer effective-default overrides for a study.

    ``n_cycles=3`` and ``experiment_order="shuffle"`` are injected only when the
    YAML ``study_execution`` block does not set them; the Pydantic model defaults
    are intentionally more conservative (n_cycles=1). These are unconditional
    research defaults, not flag overrides - the semantic-override flags were
    removed and the YAML is the source of truth for anything it declares.
    """
    exec_overrides: dict[str, Any] = {}
    if "n_cycles" not in yaml_execution:
        exec_overrides["n_cycles"] = 3
    if "experiment_order" not in yaml_execution:
        exec_overrides["experiment_order"] = "shuffle"
    if not exec_overrides:
        return {}
    return {"study_execution": exec_overrides}


def study_cli_overrides_for_file(study_file: Path) -> dict[str, Any] | None:
    """Effective-default overrides for a study file, or None if none apply.

    Reads only the ``study_execution`` block to decide which effective defaults
    to inject. Read or parse errors are ignored here and left for the
    authoritative ``load_study`` call to report, so callers can pass the result
    straight into ``load_study(path, cli_overrides=...)``.
    """
    import yaml

    try:
        raw = yaml.safe_load(study_file.read_text())
    except (OSError, yaml.YAMLError):
        return None
    yaml_execution = raw.get("study_execution") if isinstance(raw, dict) else None
    overrides = build_study_cli_overrides(yaml_execution or {})
    return overrides or None
