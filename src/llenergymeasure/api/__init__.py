"""Public library API for llenergymeasure."""

from llenergymeasure.api._impl import load_study, run_experiment, run_study
from llenergymeasure.api.diagnose import (
    DiagnoseError,
    DiagnoseResult,
    diagnose_carried_failures,
    diagnose_gaps,
)
from llenergymeasure.api.report_gaps import (
    ReportGapsError,
    find_runtime_gaps,
    render_yaml_fragment,
)
from llenergymeasure.results.persistence import save_result
from llenergymeasure.study.manifest import study_dir_name
from llenergymeasure.study.preflight import run_study_preflight
from llenergymeasure.study.resume import find_resumable_study, load_resume_state

__all__ = [
    "DiagnoseError",
    "DiagnoseResult",
    "ReportGapsError",
    "build_study_skeleton",
    "diagnose_carried_failures",
    "diagnose_gaps",
    "find_resumable_study",
    "find_runtime_gaps",
    "load_resume_state",
    "load_study",
    "probe_energy_sampler",
    "render_yaml_fragment",
    "run_experiment",
    "run_study",
    "run_study_preflight",
    "save_result",
    "study_dir_name",
]


def build_study_skeleton(engine: str) -> str:
    """Return a commented starting study YAML for *engine* (a trim-me template).

    Thin pass-through to
    :func:`llenergymeasure.config.sweep_autoexpand.build_study_skeleton`; see
    there for the auto-expansion policy. Raises ``ConfigError`` for an unknown
    engine.
    """
    from llenergymeasure.config.sweep_autoexpand import build_study_skeleton

    return build_study_skeleton(engine)


def probe_energy_sampler() -> str | None:
    """Best-effort probe for the auto-selected energy sampler on this host.

    Returns the class name of the selected sampler, or None if unavailable.
    """
    try:
        from llenergymeasure.energy import select_energy_sampler

        sampler = select_energy_sampler("auto")
        return type(sampler).__name__ if sampler else None
    except Exception:
        return None
