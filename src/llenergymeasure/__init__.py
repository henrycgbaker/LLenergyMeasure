"""LLenergyMeasure -- LLM inference efficiency measurement framework.

Public API:
    run_experiment, run_study, ExperimentConfig, StudyConfig,
    ExperimentResult, StudyResult, __version__

Stability contract: exports in __all__ follow SemVer. Names not in __all__
are internal and may change without notice. A deprecated __all__ export is
removed no earlier than one minor version after its deprecation.
"""

import logging

from llenergymeasure._version import __version__
from llenergymeasure.api._impl import run_experiment, run_study
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "StudyConfig",
    "StudyResult",
    "__version__",
    "run_experiment",
    "run_study",
]
