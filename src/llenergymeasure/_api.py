"""Internal API implementation for llenergymeasure.

This module is internal (underscore prefix). Import via llenergymeasure.__init__ only.
"""

from __future__ import annotations

import time
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
# run_study — M2 implementation
# ---------------------------------------------------------------------------


def run_study(config: str | Path | StudyConfig) -> StudyResult:
    """Run a multi-experiment study.

    Always writes manifest.json to disk (LA-05 — documented side-effect).
    Returns StudyResult with full schema (RES-13).

    Args:
        config: YAML file path or resolved StudyConfig.

    Returns:
        StudyResult with experiments, result_files, measurement_protocol, summary.

    Raises:
        ConfigError: Invalid config path or parse error.
        PreFlightError: Multi-backend study without Docker (CM-10).
        pydantic.ValidationError: Invalid field values (passes through unchanged).
    """
    if isinstance(config, (str, Path)):
        from llenergymeasure.config.loader import load_study_config

        study = load_study_config(path=Path(config))
    elif isinstance(config, StudyConfig):
        study = config
    else:
        raise ConfigError(f"Expected str, Path, or StudyConfig; got {type(config).__name__}")
    return _run(study)


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
    """Dispatcher: single experiment runs in-process; multi-experiment uses StudyRunner.

    Always:
    - Calls run_study_preflight(study) first (CM-10 multi-backend guard)
    - Creates study output directory and ManifestWriter (LA-05)
    - Returns fully populated StudyResult (RES-13 + RES-15)

    Single-experiment / n_cycles=1:  runs in-process (no subprocess overhead).
    Otherwise:                         delegates to StudyRunner.
    """
    from llenergymeasure.domain.experiment import StudySummary
    from llenergymeasure.orchestration.preflight import run_study_preflight
    from llenergymeasure.study.manifest import ManifestWriter, create_study_dir

    # Multi-backend guard — raises PreFlightError for multi-backend studies (CM-10)
    run_study_preflight(study)

    # Always create study dir + manifest (LA-05)
    study_dir = create_study_dir(study.name, Path("results"))
    manifest = ManifestWriter(study, study_dir)

    wall_start = time.monotonic()
    is_single = len(study.experiments) == 1 and study.execution.n_cycles == 1

    if is_single:
        result_files, experiment_results, warnings = _run_in_process(study, manifest, study_dir)
    else:
        result_files, experiment_results, warnings = _run_via_runner(study, manifest, study_dir)

    wall_time = time.monotonic() - wall_start

    completed = sum(1 for r in experiment_results if r is not None)
    failed = len(experiment_results) - completed
    total_energy = sum(r.total_energy_j for r in experiment_results if r is not None)

    summary = StudySummary(
        total_experiments=len(study.experiments) * study.execution.n_cycles,
        completed=completed,
        failed=failed,
        total_wall_time_s=wall_time,
        total_energy_j=total_energy,
        warnings=warnings,
    )

    measurement_protocol: dict[str, Any] = {
        "n_cycles": study.execution.n_cycles,
        "cycle_order": study.execution.cycle_order,
        "experiment_gap_seconds": study.execution.experiment_gap_seconds,
        "cycle_gap_seconds": study.execution.cycle_gap_seconds,
        "shuffle_seed": study.execution.shuffle_seed,
    }

    return StudyResult(
        experiments=[r for r in experiment_results if r is not None],
        name=study.name,
        study_design_hash=study.study_design_hash,
        measurement_protocol=measurement_protocol,
        result_files=result_files,
        summary=summary,
    )


def _run_in_process(
    study: StudyConfig,
    manifest: Any,
    study_dir: Path,
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Run a single experiment in-process. Returns (result_files, results, warnings)."""
    from llenergymeasure.core.backends import get_backend
    from llenergymeasure.domain.experiment import compute_measurement_config_hash
    from llenergymeasure.orchestration.preflight import run_preflight
    from llenergymeasure.results.persistence import save_result

    config = study.experiments[0]
    config_hash = compute_measurement_config_hash(config)
    cycle = 1

    manifest.mark_running(config_hash, cycle)

    result: ExperimentResult | None = None
    result_files: list[str] = []
    warnings: list[str] = []

    try:
        run_preflight(config)
        backend = get_backend(config.backend)
        result = backend.run(config)

        result_path = save_result(result, study_dir)
        rel_path = str(result_path.relative_to(study_dir))
        result_files.append(str(result_path))
        manifest.mark_completed(config_hash, cycle, rel_path)
    except Exception as exc:
        warnings.append(str(exc))
        manifest.mark_failed(config_hash, cycle, type(exc).__name__, str(exc))

    return result_files, [result], warnings


def _run_via_runner(
    study: StudyConfig,
    manifest: Any,
    study_dir: Path,
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Delegate to StudyRunner for multi-experiment / multi-cycle runs."""
    from llenergymeasure.study.runner import StudyRunner

    runner = StudyRunner(study, manifest, study_dir)
    raw_results = runner.run()

    warnings: list[str] = []
    experiment_results: list[ExperimentResult | None] = []
    for r in raw_results:
        if isinstance(r, dict):
            warnings.append(r.get("message", "Unknown error"))
            experiment_results.append(None)
        else:
            experiment_results.append(r)

    return runner.result_files, experiment_results, warnings
