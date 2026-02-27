"""Internal API implementation for llenergymeasure.

This module is internal (underscore prefix). Import via llenergymeasure.__init__ only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, overload

from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.config.models import ExperimentConfig, StudyConfig
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult
from llenergymeasure.exceptions import ConfigError

# ---------------------------------------------------------------------------
# run_experiment — three overloaded forms
# ---------------------------------------------------------------------------


@overload
def run_experiment(config: str | Path) -> ExperimentResult: ...


@overload
def run_experiment(config: ExperimentConfig) -> ExperimentResult: ...


@overload
def run_experiment(
    config: None = None,
    *,
    model: str,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs: Any,
) -> ExperimentResult: ...


def run_experiment(
    config: str | Path | ExperimentConfig | None = None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs: Any,
) -> ExperimentResult:
    """Run a single LLM inference efficiency experiment.

    Side-effect free: no disk writes unless output_dir is specified in the config.

    Three call forms:
        run_experiment("config.yaml")              # YAML path
        run_experiment(ExperimentConfig(...))       # config object
        run_experiment(model="gpt2", backend="Y")  # kwargs convenience

    Args:
        config: YAML file path, ExperimentConfig object, or None (use kwargs).
        model: Model name/path (kwargs form only).
        backend: Inference backend (kwargs form only, defaults to ExperimentConfig default).
        n: Number of prompts (kwargs form only, default 100).
        dataset: Dataset name (kwargs form only, default "aienergyscore").
        **kwargs: Additional ExperimentConfig fields (kwargs form only).

    Returns:
        ExperimentResult: Experiment measurements and metadata.

    Raises:
        ConfigError: Invalid config path, missing model in kwargs form.
        pydantic.ValidationError: Invalid field values (passes through unchanged).
    """
    study = _to_study_config(config, model=model, backend=backend, n=n, dataset=dataset, **kwargs)
    study_result = _run(study)
    return study_result.experiments[0]


# ---------------------------------------------------------------------------
# run_study — stub for M1 (surface completeness)
# ---------------------------------------------------------------------------


def run_study(config: str | Path | StudyConfig) -> StudyResult:
    """Run a study (multiple experiments). Available in M2.

    Exported from v2.0.0 for API surface stability. Implementation ships in M2.

    Raises:
        NotImplementedError: Study execution is not yet implemented.
    """
    raise NotImplementedError(
        "Study execution is available in M2. Use run_experiment() for single experiments."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_study_config(
    config: str | Path | ExperimentConfig | None,
    *,
    model: str | None = None,
    backend: str | None = None,
    n: int = 100,
    dataset: str = "aienergyscore",
    **kwargs: Any,
) -> StudyConfig:
    """Convert any run_experiment() input form to a degenerate StudyConfig."""
    if isinstance(config, ExperimentConfig):
        experiment = config
    elif isinstance(config, (str, Path)):
        experiment = load_experiment_config(path=Path(config))
    elif config is None:
        if model is None:
            raise ConfigError(
                "run_experiment() requires either a config argument or model= keyword.\n"
                "Example: run_experiment(model='meta-llama/Llama-3.1-8B')"
            )
        # Build kwargs dict for ExperimentConfig — only include non-default values
        # to let Pydantic defaults apply for omitted fields.
        ec_kwargs: dict[str, Any] = {"model": model, "n": n, "dataset": dataset}
        if backend is not None:
            ec_kwargs["backend"] = backend
        ec_kwargs.update(kwargs)
        experiment = ExperimentConfig(**ec_kwargs)
    else:
        raise ConfigError(
            f"Expected str, Path, ExperimentConfig, or None; got {type(config).__name__}"
        )
    return StudyConfig(experiments=[experiment])


def _run(study: StudyConfig) -> StudyResult:
    """Internal runner -- always receives StudyConfig, returns StudyResult.

    Calls run_preflight() then get_backend().run() for each experiment config.
    Pre-flight runs once per experiment (each config may differ in model/backend).
    Errors propagate naturally -- PreFlightError and BackendError are not caught here.
    """
    from llenergymeasure.core.backends import get_backend
    from llenergymeasure.orchestration.preflight import run_preflight

    results = []
    for config in study.experiments:
        # Pre-flight runs once per experiment config (CM-29, CM-30, CM-31)
        run_preflight(config)

        backend = get_backend(config.backend)
        result = backend.run(config)
        results.append(result)

    return StudyResult(experiments=results, name=study.name)
