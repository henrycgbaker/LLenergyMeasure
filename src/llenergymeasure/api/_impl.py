"""Internal API implementation for llenergymeasure.

This module is internal (underscore prefix). Import via llenergymeasure.__init__ only.

It is a thin adapter over the study-layer orchestrator
(:func:`llenergymeasure.study.orchestration.orchestrate_study`): it routes every
call form through the single study-resolution entry point
(:func:`llenergymeasure.study.loading.resolve_study`), translates the overloaded
public ``output_dir`` argument into the orchestrator's explicit internal
parameters, and delegates. A YAML path and the equivalent objects therefore
produce the same resolved study. The public API surface (``load_study`` /
``run_experiment`` / ``run_study`` and ``api.__all__``) is stable: call forms and
their meaning do not change, and it grows only by addition (an optional keyword),
never by removing or repurposing what callers already pass. The orchestration
itself lives in the study layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, overload

from llenergymeasure.config.loader import load_experiment_config
from llenergymeasure.config.models import (
    DatasetConfig,
    ExperimentConfig,
    MeasurementConfig,
    StudyConfig,
    TaskConfig,
)
from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult
from llenergymeasure.domain.progress import ProgressCallback
from llenergymeasure.utils.exceptions import ConfigError

# Single source of truth for n_prompts default - derived from DatasetConfig field default
# so run_experiment() kwargs and DatasetConfig always agree.
_N_PROMPTS_DEFAULT: int = DatasetConfig.model_fields["n_prompts"].default

# Derived from model fields so routing auto-updates when sub-models gain new fields.
_TASK_FIELDS: frozenset[str] = frozenset(TaskConfig.model_fields) - {"model", "dataset"}
_MEASUREMENT_FIELDS: frozenset[str] = frozenset(MeasurementConfig.model_fields)

# ---------------------------------------------------------------------------
# load_study - the YAML front door onto the single resolution entry point
# ---------------------------------------------------------------------------


def load_study(
    path: str | Path,
    cli_overrides: dict[str, Any] | None = None,
    *,
    execution_defaults: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Load a study YAML and resolve it into a runnable StudyConfig.

    Composes the config-layer parse/expand step
    (:func:`llenergymeasure.config.loader.load_study_config`) with the single
    study-resolution entry point
    (:func:`llenergymeasure.study.loading.resolve_study`). This is the YAML front
    door both the CLI and ``run_study`` use, so the config layer never imports
    upward into ``study`` and the CLI never imports ``study`` directly.

    It also loads the tool-wide user config and hands it to ``resolve_study``,
    which overlays its ``server.warmup`` defaults onto each declared server config
    and fills in the machine-local thermal gaps the study left unset. The warmup
    overlay shapes the resolved-config hash (which dedup binds on) but never the
    declared hash, so a shared study file keeps its declared identity across
    machines. Resume and drift-detection remain declared-hash-only, so they are
    blind to a user-config warmup change between an original run and a resume.

    Args:
        path: Path to study YAML file.
        cli_overrides: Optional dict of overrides applied ON TOP of the study file
            (e.g. {"study_execution": {"n_cycles": 5}}) - they win over what the
            file declares. This is the call-site layer of the precedence chain, so
            a study setting resolved from here is recorded with a ``call_site``
            provenance label.
        execution_defaults: Optional execution-block defaults applied BENEATH the
            study file: a field the file wrote wins, a field it omitted takes this
            value (e.g. {"n_cycles": 3}).
        overrides: Optional study-file-shaped call-site overrides for the settings
            the precedence chain owns (e.g. {"output": {"results_dir": "/data"}}).
            Wins over ``cli_overrides`` where both name the same setting.

    Returns:
        Resolved StudyConfig with ordered experiments, study_design_hash, dedup
        mode, and pre-run equivalence groups.

    Raises:
        ConfigError: File not found, parse error, all configs invalid, empty study,
            a study mixing serving_mode values, or invalid ``execution_defaults``.
        ValidationError: Pydantic structural errors pass through unchanged.
    """
    from llenergymeasure.config.loader import load_study_config
    from llenergymeasure.config.precedence import resolve_layers
    from llenergymeasure.config.user_config import load_user_config
    from llenergymeasure.study.loading import resolve_study

    # The production edge that folds the tool-wide user config into the study.
    return resolve_study(
        load_study_config(path, cli_overrides=cli_overrides),
        user_config=load_user_config(),
        execution_defaults=execution_defaults,
        overrides=resolve_layers(cli_overrides or {}, overrides or {}),
    )


# ---------------------------------------------------------------------------
# run_experiment - three overloaded forms
# ---------------------------------------------------------------------------


@overload
def run_experiment(
    config: str | Path,
    *,
    skip_preflight: bool = ...,
    progress: ProgressCallback | None = ...,
    output_dir: str | Path | None = ...,
) -> ExperimentResult: ...


@overload
def run_experiment(
    config: ExperimentConfig,
    *,
    skip_preflight: bool = ...,
    progress: ProgressCallback | None = ...,
    output_dir: str | Path | None = ...,
) -> ExperimentResult: ...


@overload
def run_experiment(
    config: None = None,
    *,
    model: str,
    engine: str | None = None,
    n_prompts: int = _N_PROMPTS_DEFAULT,
    dataset: str = "aienergyscore",
    skip_preflight: bool = ...,
    progress: ProgressCallback | None = ...,
    output_dir: str | Path | None = ...,
    **kwargs: Any,
) -> ExperimentResult: ...


def run_experiment(
    config: str | Path | ExperimentConfig | None = None,
    *,
    model: str | None = None,
    engine: str | None = None,
    n_prompts: int = _N_PROMPTS_DEFAULT,
    dataset: str = "aienergyscore",
    skip_preflight: bool = False,
    progress: ProgressCallback | None = None,
    output_dir: str | Path | None = None,
    **kwargs: Any,
) -> ExperimentResult:
    """Run a single LLM inference efficiency experiment.

    Three call forms:
        run_experiment("config.yaml")              # YAML path
        run_experiment(ExperimentConfig(...))       # config object
        run_experiment(model="gpt2", engine="Y")   # kwargs convenience

    Args:
        config: YAML file path, ExperimentConfig object, or None (use kwargs).
        model: Model name/path (kwargs form only).
        engine: Inference engine (kwargs form only, defaults to ExperimentConfig default).
        n_prompts: Number of prompts (kwargs form only, default 100).
        dataset: Dataset source name (kwargs form only, default "aienergyscore").
        skip_preflight: Skip Docker pre-flight checks (GPU visibility, CUDA/driver compat).
        progress: Optional callback for step-by-step progress reporting.
        output_dir: Base directory for results. When provided, overrides the
            default ``./results`` directory. A timestamped study subdirectory
            is created within this path.
        **kwargs: Additional ExperimentConfig fields (kwargs form only).

    Returns:
        ExperimentResult: Experiment measurements and metadata.

    Raises:
        ConfigError: Invalid config path, missing model in kwargs form.
        pydantic.ValidationError: Invalid field values (passes through unchanged).
    """
    from llenergymeasure.study.orchestration import orchestrate_study

    study = _to_study_config(
        config, model=model, engine=engine, n_prompts=n_prompts, dataset=dataset, **kwargs
    )
    study = _resolve_objects(study, overrides=_results_dir_override(output_dir))
    study_result = orchestrate_study(study, skip_preflight=skip_preflight, progress=progress)
    if not study_result.experiments:
        from llenergymeasure.utils.exceptions import ExperimentError

        error_msg = (
            study_result.summary.warnings[0]
            if study_result.summary.warnings
            else "Experiment produced no results"
        )
        raise ExperimentError(error_msg)
    return study_result.experiments[0]


# ---------------------------------------------------------------------------
# run_study
# ---------------------------------------------------------------------------


def run_study(
    config: str | Path | StudyConfig,
    *,
    skip_preflight: bool = False,
    progress: ProgressCallback | None = None,
    resume_dir: Path | None = None,
    resume: bool = False,
    output_dir: Path | None = None,
    skip_set: set[tuple[str, int]] | None = None,
    no_lock: bool = False,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    preresolved: tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]] | None = None,
) -> StudyResult:
    """Run a multi-experiment study.

    Always writes manifest.json to disk (documented side-effect).

    A StudyConfig built in memory is resolved here exactly as a YAML study is:
    equivalent configs are deduplicated, the ``study_design_hash`` is computed,
    ``n_cycles`` is expanded into the execution sequence, and the equivalence
    groups are recorded. An already-resolved StudyConfig (from ``load_study``)
    passes through unchanged.

    Resolving a caller-built StudyConfig is not free of side effects on its
    argument: for a server-mode experiment, the resolved warmup protocol is
    attached to the caller's OWN ExperimentConfig objects (as side-channel state,
    not a declared field), so those objects carry it after the call. The declared
    fields are never rewritten - the configs that actually run are copies.

    Args:
        config: YAML file path or a StudyConfig.
        skip_preflight: Skip Docker pre-flight checks (GPU visibility, CUDA/driver compat).
            CLI --skip-preflight flag and YAML execution.skip_preflight: true also bypass.
        progress: Optional StudyProgressCallback for live per-experiment display.
            When provided, the study runner emits begin/end experiment events and
            forwards per-step progress from worker subprocesses.
        resume_dir: Explicit study directory to resume. Overrides ``resume``.
        resume: When True and resume_dir is None, auto-detect the most recent
            resumable study in ``output_dir`` (default ``results/``).
        output_dir: Dual role by run mode. For a fresh run it is the results-dir
            override (precedence: ``output_dir`` > YAML ``output.results_dir`` >
            user config > ``./results``). For an auto-detect resume it is the base
            directory searched for the most recent resumable study. Ignored when
            ``resume_dir`` is given explicitly.
        skip_set: Set of (config_hash, cycle) pairs to skip (already completed in a
            previous run). Populated automatically when resuming; callers rarely
            need to set this directly.
        no_lock: Skip GPU advisory lock acquisition. Use with --no-lock CLI flag.
        config_path: Original YAML config file path for copying to study artefacts.
            When config is a StudyConfig object, callers should pass the original
            path separately so the YAML is preserved for reproducibility.
        cli_overrides: Overrides applied on top of the study file when ``config``
            is a path (forwarded to :func:`load_study` as its call-site layer, so
            they win over what the file declares and are recorded as ``call_site``
            in the provenance). Ignored for a StudyConfig input, which the caller
            has already built.
        preresolved: Optional ``(runner_specs, system_overrides)`` already
            computed by a prior ``run_study_preflight`` call (e.g. the CLI runs
            preflight to render the panel). When supplied, the orchestrator reuses
            it instead of re-running preflight. Must be paired with
            ``skip_preflight=True`` so the precomputed result is trusted.

    Returns:
        StudyResult with experiments, result_files, measurement_protocol, and inline summary fields.

    Raises:
        ConfigError: Invalid config path or parse error.
        PreFlightError: Multi-engine study without Docker.
        StudyError: No resumable study found (when resume=True).
        StudyError: Config drift detected (study_design_hash changed).
        pydantic.ValidationError: Invalid field values (passes through unchanged).
    """
    from llenergymeasure.study.orchestration import orchestrate_study

    # The public ``output_dir`` is overloaded (#842): for an auto-detect resume it
    # is the base directory searched for the most recent resumable study, and for a
    # fresh run it is the results-dir override. A resume consumes it below to
    # locate the study, so only a fresh run feeds it into resolution.
    resume_search_base = output_dir
    is_resume = resume_dir is not None or resume
    overrides = None if is_resume else _results_dir_override(output_dir)

    if isinstance(config, (str, Path)):
        config_path = config_path or Path(config).resolve()
        study = load_study(config_path, cli_overrides=cli_overrides, overrides=overrides)
    elif isinstance(config, StudyConfig):
        # config_path may have been passed by caller (e.g. CLI pre-loads config)
        study = _resolve_objects(config, overrides=overrides)
    else:
        raise ConfigError(f"Expected str, Path, or StudyConfig; got {type(config).__name__}")

    # Resolve resume state if requested.
    if is_resume:
        from llenergymeasure.study.resume import (
            find_resumable_study,
            load_resume_state,
            prepare_resume_manifest,
            validate_config_drift,
            validate_resolved_config_drift,
        )
        from llenergymeasure.utils.exceptions import StudyError

        if resume_dir is None:
            resume_dir = find_resumable_study(resume_search_base or Path("results"))
            if resume_dir is None:
                raise StudyError("No resumable study found. Run a study first or use --resume-dir.")

        old_manifest, skip_set = load_resume_state(resume_dir)
        validate_config_drift(old_manifest, study)
        # Declared-hash drift is only half the guard: catch a resolved-protocol
        # change (e.g. a user-config warmup overlay) that the declared family and the
        # skip-set are blind to, before it silently skips a differently-resolved cell.
        validate_resolved_config_drift(old_manifest, study)
        prepare_resume_manifest(resume_dir, old_manifest)

    return orchestrate_study(
        study,
        skip_preflight=skip_preflight,
        progress=progress,
        resume_dir=resume_dir,
        skip_set=skip_set,
        no_lock=no_lock,
        config_path=config_path,
        preresolved=preresolved,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _results_dir_override(output_dir: str | Path | None) -> dict[str, Any] | None:
    """The call-site results-dir layer for an explicit ``output_dir`` argument."""
    if output_dir is None:
        return None
    return {"output": {"results_dir": str(output_dir)}}


def _resolve_objects(study: StudyConfig, *, overrides: dict[str, Any] | None = None) -> StudyConfig:
    """Route a caller-built StudyConfig through the single resolution entry point.

    A StudyConfig assembled in memory - by ``run_study(StudyConfig(...))``, by
    ``run_experiment``, or by any programmatic caller - has had none of the
    resolution the YAML path performs: no dedup, no ``study_design_hash``, no
    cycle expansion, no equivalence groups.
    :func:`llenergymeasure.study.loading.resolve_study_objects` gives it exactly
    the resolution a YAML study gets, with no file touched (#886).

    An already-resolved study (one that carries a ``study_design_hash``, i.e. it
    came from ``load_study``) is returned untouched, before the user config is
    even read: its experiment list is already cycle-expanded, so resolving twice
    would re-expand it. A study whose hash was written by hand rather than by
    resolution is therefore NOT quietly resolved here - the orchestrator refuses
    it, which is the louder outcome.
    """
    if study.study_design_hash is not None:
        return study

    from llenergymeasure.config.user_config import load_user_config
    from llenergymeasure.study.loading import resolve_study_objects

    return resolve_study_objects(study, user_config=load_user_config(), overrides=overrides)


def _to_study_config(
    config: str | Path | ExperimentConfig | None,
    *,
    model: str | None = None,
    engine: str | None = None,
    n_prompts: int = _N_PROMPTS_DEFAULT,
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
        # Build kwargs dict for ExperimentConfig - route fields into sub-models.
        task_kwargs: dict[str, Any] = {
            "model": model,
            "dataset": DatasetConfig(source=dataset, n_prompts=n_prompts),
        }
        ec_kwargs: dict[str, Any] = {"task": task_kwargs}
        if engine is not None:
            ec_kwargs["engine"] = engine
        # Route remaining kwargs into correct sub-model or top-level
        measurement_kwargs: dict[str, Any] = {}
        for key, value in kwargs.items():
            if key in _TASK_FIELDS:
                task_kwargs[key] = value
            elif key in _MEASUREMENT_FIELDS:
                measurement_kwargs[key] = value
            else:
                ec_kwargs[key] = value
        if measurement_kwargs:
            ec_kwargs["measurement"] = measurement_kwargs
        # serving_mode is required with no model default; the model= convenience
        # path is the offline batch one-liner, so declare offline explicitly here
        # (an explicit serving_mode= kwarg routed above still wins).
        ec_kwargs.setdefault("serving_mode", "offline")
        experiment = ExperimentConfig(**ec_kwargs)
    else:
        raise ConfigError(
            f"Expected str, Path, ExperimentConfig, or None; got {type(config).__name__}"
        )
    return StudyConfig(experiments=[experiment])
