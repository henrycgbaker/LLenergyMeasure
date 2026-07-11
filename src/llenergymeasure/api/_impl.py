"""Internal API implementation for llenergymeasure.

This module is internal (underscore prefix). Import via llenergymeasure.__init__ only.
"""

from __future__ import annotations

import json
import logging
import time
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
from llenergymeasure.config.ssot import RUNNER_DOCKER
from llenergymeasure.domain.bundle_artefacts import (
    ENVIRONMENT_FILENAME,
    STUDY_ARTEFACTS_DIR,
    SYSTEM_OVERRIDES_FILENAME,
)
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult, StudySummary
from llenergymeasure.domain.progress import ProgressCallback
from llenergymeasure.infra.runner_resolution import RunnerSpec
from llenergymeasure.study.single import run_single_experiment
from llenergymeasure.utils.exceptions import ConfigError

logger = logging.getLogger(__name__)

# Single source of truth for n_prompts default - derived from DatasetConfig field default
# so run_experiment() kwargs and DatasetConfig always agree.
_N_PROMPTS_DEFAULT: int = DatasetConfig.model_fields["n_prompts"].default

# Derived from model fields so routing auto-updates when sub-models gain new fields.
_TASK_FIELDS: frozenset[str] = frozenset(TaskConfig.model_fields) - {"model", "dataset"}
_MEASUREMENT_FIELDS: frozenset[str] = frozenset(MeasurementConfig.model_fields)

# ---------------------------------------------------------------------------
# load_study - parse + finalise composition
# ---------------------------------------------------------------------------


def load_study(
    path: str | Path,
    cli_overrides: dict[str, Any] | None = None,
) -> StudyConfig:
    """Load a study YAML and finalise it into a resolved StudyConfig.

    Composes the config-layer parse/expand step
    (:func:`llenergymeasure.config.loader.load_study_config`) with the
    study-layer finalisation
    (:func:`llenergymeasure.study.loading.finalise_study`). This is the single
    public entry both the CLI and ``run_study`` use, so the config layer never
    imports upward into ``study`` and the CLI never imports ``study`` directly.

    Args:
        path: Path to study YAML file.
        cli_overrides: Optional dict of CLI flag overrides for the execution
            block (e.g. {"study_execution": {"n_cycles": 5}}).

    Returns:
        Resolved StudyConfig with ordered experiments, study_design_hash, dedup
        mode, and pre-run equivalence groups.

    Raises:
        ConfigError: File not found, parse error, all configs invalid, empty study.
        ValidationError: Pydantic structural errors pass through unchanged.
    """
    from llenergymeasure.config.loader import load_study_config
    from llenergymeasure.study.loading import finalise_study

    return finalise_study(load_study_config(path, cli_overrides=cli_overrides))


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
    study = _to_study_config(
        config, model=model, engine=engine, n_prompts=n_prompts, dataset=dataset, **kwargs
    )
    if output_dir is not None:
        study.output = study.output.model_copy(update={"results_dir": str(output_dir)})
    study_result = _run(study, skip_preflight=skip_preflight, progress=progress)
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

    Args:
        config: YAML file path or resolved StudyConfig.
        skip_preflight: Skip Docker pre-flight checks (GPU visibility, CUDA/driver compat).
            CLI --skip-preflight flag and YAML execution.skip_preflight: true also bypass.
        progress: Optional StudyProgressCallback for live per-experiment display.
            When provided, the study runner emits begin/end experiment events and
            forwards per-step progress from worker subprocesses.
        resume_dir: Explicit study directory to resume. Overrides ``resume``.
        resume: When True and resume_dir is None, auto-detect the most recent
            resumable study in ``output_dir`` (default ``results/``).
        output_dir: Base output directory used by auto-detect resume. Ignored when
            ``resume_dir`` is given explicitly.
        skip_set: Set of (config_hash, cycle) pairs to skip (already completed in a
            previous run). Populated automatically when resuming; callers rarely
            need to set this directly.
        no_lock: Skip GPU advisory lock acquisition. Use with --no-lock CLI flag.
        config_path: Original YAML config file path for copying to study artefacts.
            When config is a StudyConfig object, callers should pass the original
            path separately so the YAML is preserved for reproducibility.
        cli_overrides: Flat dict of CLI flag overrides (e.g. {"model": "gpt2"}).
            Used to build per-experiment ``_resolution.json`` sidecars showing
            which fields were overridden by CLI flags vs YAML vs sweep.
        preresolved: Optional ``(runner_specs, system_overrides)`` already
            computed by a prior ``run_study_preflight`` call (e.g. the CLI runs
            preflight to render the panel). When supplied, ``_run`` reuses it
            instead of re-running preflight. Must be paired with
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
    if isinstance(config, (str, Path)):
        config_path = config_path or Path(config).resolve()
        study = load_study(config_path)
    elif isinstance(config, StudyConfig):
        # config_path may have been passed by caller (e.g. CLI pre-loads config)
        study = config
    else:
        raise ConfigError(f"Expected str, Path, or StudyConfig; got {type(config).__name__}")

    # Resolve resume state if requested.
    if resume_dir is not None or resume:
        from llenergymeasure.study.resume import (
            find_resumable_study,
            load_resume_state,
            prepare_resume_manifest,
            validate_config_drift,
        )
        from llenergymeasure.utils.exceptions import StudyError

        if resume_dir is None:
            _output = output_dir or Path("results")
            resume_dir = find_resumable_study(_output)
            if resume_dir is None:
                raise StudyError("No resumable study found. Run a study first or use --resume-dir.")

        old_manifest, skip_set = load_resume_state(resume_dir)
        validate_config_drift(old_manifest, study)
        prepare_resume_manifest(resume_dir, old_manifest)

    return _run(
        study,
        skip_preflight=skip_preflight,
        progress=progress,
        resume_dir=resume_dir,
        skip_set=skip_set,
        no_lock=no_lock,
        config_path=config_path,
        cli_overrides=cli_overrides,
        preresolved=preresolved,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
        experiment = ExperimentConfig(**ec_kwargs)
    else:
        raise ConfigError(
            f"Expected str, Path, ExperimentConfig, or None; got {type(config).__name__}"
        )
    return StudyConfig(experiments=[experiment])


def _ensure_study_artefacts_dir(study_dir: Path) -> Path:
    """Create and return the _study-artefacts/ subdirectory."""
    artefacts_dir = study_dir / STUDY_ARTEFACTS_DIR
    artefacts_dir.mkdir(exist_ok=True)
    return artefacts_dir


def _write_skipped_configs_log(
    skipped_configs: list[dict[str, Any]],
    artefacts_dir: Path,
    study_design_hash: str = "",
    study_name: str = "",
) -> None:
    """Write detailed skipped-config information to the _study-artefacts directory."""
    log_path = artefacts_dir / "skipped_configs.log"
    lines = [
        f"# study_design_hash: {study_design_hash} | study_name: {study_name}",
        f"Skipped {len(skipped_configs)} config(s) due to validation errors\n",
    ]
    for s in skipped_configs:
        label = s.get("short_label", "unknown")
        reason = s.get("reason", "unknown error")
        lines.append(f"  {label}")
        lines.append(f"    {reason}")
        lines.append("")
    log_path.write_text("\n".join(lines))


def _run(
    study: StudyConfig,
    skip_preflight: bool = False,
    progress: ProgressCallback | None = None,
    resume_dir: Path | None = None,
    skip_set: set[tuple[str, int]] | None = None,
    no_lock: bool = False,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    preresolved: tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]] | None = None,
) -> StudyResult:
    """Dispatcher: single experiment runs in-process; multi-experiment uses StudyRunner.

    Always:
    - Calls run_study_preflight() first (multi-engine guard and Docker pre-flight checks)
    - Resolves runner specs for all engines in the study
    - Creates study output directory and ManifestWriter
    - Returns fully populated StudyResult

    Single-experiment / n_cycles=1:  runs in-process or via DockerRunner directly.
    Otherwise:                         delegates to StudyRunner.
    """
    from llenergymeasure.config.user_config import load_user_config
    from llenergymeasure.study.manifest import ManifestWriter, create_study_dir

    # Preresolved runner specs bypass preflight; demanding skip_preflight makes the
    # contract explicit rather than silently ignoring the precomputed result.
    if preresolved is not None and not skip_preflight:
        raise ValueError("preresolved requires skip_preflight=True")

    # Load user config first so runner context can be forwarded to preflight,
    # ensuring preflight uses the same runner resolution as the actual dispatch path.
    user_config = load_user_config()

    runner_specs, system_overrides = _resolve_runner_specs(
        study, user_config, preresolved, skip_preflight, progress
    )

    # Resolve results_dir: resume_dir takes priority, then YAML > user config > built-in default
    if resume_dir is not None:
        study_dir = resume_dir
        # Resume: load the existing manifest written by prepare_resume_manifest()
        # and wrap it without rebuilding or overwriting the prepared manifest.
        from llenergymeasure.study.resume import load_resume_state

        loaded_manifest, _ = load_resume_state(study_dir)
        manifest = ManifestWriter.from_existing(study_dir, loaded_manifest)
    else:
        results_dir_str = study.output.results_dir or user_config.output.results_dir or "./results"
        study_dir = create_study_dir(study.study_name, Path(results_dir_str))
        manifest = ManifestWriter(study, study_dir)

    # Create _study-artefacts/ once for config copy, skipped log, and study-level env.
    artefacts_dir = _ensure_study_artefacts_dir(study_dir)

    _write_study_artefacts(study, artefacts_dir, system_overrides, config_path)

    resolution_logs = _build_resolution_logs(study, cli_overrides)

    wall_start = time.monotonic()
    is_single = len(study.experiments) == 1 and study.study_execution.n_cycles == 1

    if is_single:
        result_files, experiment_results, warnings = _run_single_experiment_dispatch(
            study, manifest, study_dir, runner_specs, progress, resolution_logs
        )
    else:
        result_files, experiment_results, warnings = _run_via_runner(
            study,
            manifest,
            study_dir,
            runner_specs=runner_specs,
            progress=progress,
            skip_set=skip_set,
            no_lock=no_lock,
            resolution_logs=resolution_logs,
        )

    wall_time = time.monotonic() - wall_start

    # Mark the study completed - but never overwrite a terminal abort status the
    # runner already set (wall-clock timeout or circuit-breaker). Doing so would make
    # an aborted study look completed and therefore non-resumable, leaving its skipped
    # experiments to never re-run. The SIGINT path calls mark_interrupted() then
    # sys.exit(130) before returning here, so 'interrupted' is not normally seen.
    if manifest.status not in ("timed_out", "circuit_breaker", "interrupted"):
        manifest.mark_study_completed()

    completed = sum(1 for r in experiment_results if r is not None)
    failed = len(experiment_results) - completed
    total_energy = sum(r.total_energy_j for r in experiment_results if r is not None)

    # study.experiments is already cycle-expanded by apply_cycles(), so len() is the true total
    n_cycles = study.study_execution.n_cycles
    unique_configs = len(study.experiments) // n_cycles if n_cycles > 0 else len(study.experiments)

    measurement_protocol: dict[str, Any] = {
        "n_cycles": study.study_execution.n_cycles,
        "experiment_order": study.study_execution.experiment_order,
        "experiment_gap_seconds": study.study_execution.experiment_gap_seconds,
        "cycle_gap_seconds": study.study_execution.cycle_gap_seconds,
        "shuffle_seed": study.study_execution.shuffle_seed,
        "experiment_timeout_seconds": study.study_execution.experiment_timeout_seconds,
    }

    summary = StudySummary(
        total_experiments=len(study.experiments),
        completed=completed,
        failed=failed,
        total_wall_time_s=wall_time,
        total_energy_j=total_energy,
        unique_configurations=unique_configs,
        warnings=warnings,
    )

    return StudyResult(
        experiments=[r for r in experiment_results if r is not None],
        study_name=study.study_name,
        study_design_hash=study.study_design_hash,
        measurement_protocol=measurement_protocol,
        result_files=result_files,
        summary=summary,
        skipped_experiments=study.skipped_configs,
    )


def _resolve_runner_specs(
    study: StudyConfig,
    user_config: Any,
    preresolved: tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]] | None,
    skip_preflight: bool,
    progress: ProgressCallback | None,
) -> tuple[dict[str, RunnerSpec], dict[str, dict[str, str]]]:
    """Resolve runner specs via preflight, or reuse a caller-provided preresolved result.

    Runs the multi-engine guard + Docker preflight (raising PreFlightError for
    multi-engine studies without Docker, or auto-elevating when available), emits
    preflight progress, and warns when the study mixes local and Docker runners.
    """
    from llenergymeasure.study.preflight import run_study_preflight

    if preresolved is not None:
        runner_specs, system_overrides = preresolved
    else:
        if progress:
            progress.on_step_start("preflight", "Checking", "environment and Docker")
            t0_pf = time.perf_counter()
        try:
            runner_specs, system_overrides = run_study_preflight(
                study,
                skip_preflight=skip_preflight,
                yaml_runners=study.runners,
                user_config=user_config.runners,
                yaml_images=study.images,
                user_config_images=user_config.images or None,
            )
        except Exception:
            if progress:
                progress.on_step_done("preflight", time.perf_counter() - t0_pf)
            raise
        if progress:
            progress.on_step_done("preflight", time.perf_counter() - t0_pf)

    # Warn on mixed runners (some local, some docker)
    modes = {spec.mode for spec in runner_specs.values()}
    if len(modes) > 1:
        logger.warning(
            "Mixed runners detected. For consistent measurements, "
            "consider running all engines in Docker."
        )
    return runner_specs, system_overrides


def _write_study_artefacts(
    study: StudyConfig,
    artefacts_dir: Path,
    system_overrides: dict[str, dict[str, str]],
    config_path: Path | None,
) -> None:
    """Persist study-level artefacts to ``_study-artefacts/``.

    Writes the original YAML copy (with an identity header), the skipped-config log,
    the system-overrides record, and the software environment snapshot. Each write is
    best-effort and logs on failure rather than aborting the run.
    """

    # Identity fields for all study-level artefacts
    _study_hash = study.study_design_hash or ""
    _study_name = study.study_name or ""

    # Copy original YAML config to _study-artefacts/ with identity header.
    if config_path is not None:
        dest = artefacts_dir / "study_config.yaml"
        try:
            original = Path(config_path).read_text(encoding="utf-8")
            header = f"# study_design_hash: {_study_hash} | study_name: {_study_name}\n"
            dest.write_text(header + original, encoding="utf-8")
            logger.info("Config YAML copied to %s", dest)
        except FileNotFoundError:
            logger.warning("Config YAML %s not found, skipping copy", config_path)

    # Persist skipped config details to _study-artefacts/.
    if study.skipped_configs:
        _write_skipped_configs_log(study.skipped_configs, artefacts_dir, _study_hash, _study_name)

    # Persist system overrides to _study-artefacts/ (runner auto-elevation, etc.)
    if system_overrides:
        overrides_with_identity = {
            "study_design_hash": _study_hash,
            "study_name": _study_name,
            **system_overrides,
        }
        overrides_path = artefacts_dir / SYSTEM_OVERRIDES_FILENAME
        try:
            overrides_path.write_text(
                json.dumps(overrides_with_identity, indent=2), encoding="utf-8"
            )
            logger.info("System overrides written to %s", overrides_path)
        except OSError as exc:
            logger.warning("Failed to write system_overrides.json: %s", exc)

    # Write study-level environment.json (installed_packages + software constants).
    try:
        from llenergymeasure.harness.environment import collect_software_environment

        sw_env = collect_software_environment()
        study_env = {
            "study_design_hash": _study_hash,
            "study_name": _study_name,
            **sw_env,
        }
        env_path = artefacts_dir / ENVIRONMENT_FILENAME
        env_path.write_text(json.dumps(study_env, indent=2), encoding="utf-8")
        logger.info("Study-level environment written to %s", env_path)
    except Exception as exc:
        logger.warning("Failed to write study-level environment.json: %s", exc)


def _build_resolution_logs(
    study: StudyConfig, cli_overrides: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    """Build per-experiment resolution logs keyed by declared-config hash.

    Computed once here so runners don't need to know about resolution logic.
    Best-effort: returns whatever was built before any failure.
    """
    resolution_logs: dict[str, dict[str, Any]] = {}
    try:
        from llenergymeasure.config.introspection import get_swept_field_paths
        from llenergymeasure.config.resolution import build_resolution_log
        from llenergymeasure.domain.experiment import compute_declared_config_hash

        swept_fields = get_swept_field_paths(study.experiments)
        seen_hashes: set[str] = set()
        for exp in study.experiments:
            h = compute_declared_config_hash(exp)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            resolution_logs[h] = build_resolution_log(
                exp.model_dump(),
                cli_overrides=cli_overrides,
                swept_fields=swept_fields,
            )
    except Exception as exc:
        logger.debug("Failed to build resolution logs: %s", exc)
    return resolution_logs


def _run_single_experiment_dispatch(
    study: StudyConfig,
    manifest: Any,
    study_dir: Path,
    runner_specs: Any,
    progress: ProgressCallback | None,
    resolution_logs: dict[str, dict[str, Any]],
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Run a single-experiment study in-process, emitting study-table begin/end events.

    For a StudyProgressCallback, emits begin_experiment / end_experiment_(ok|fail) so
    the study display shows the single experiment's table row.
    """
    from llenergymeasure.domain.progress import STEPS_LOCAL, StudyProgressCallback, docker_steps

    study_cb: StudyProgressCallback | None = (
        progress if isinstance(progress, StudyProgressCallback) else None
    )
    if study_cb is not None:
        from llenergymeasure.utils.formatting import format_experiment_header

        config = study.experiments[0]
        spec = runner_specs.get(config.engine) if runner_specs else None
        is_docker = spec and spec.mode == RUNNER_DOCKER
        if is_docker:
            host_baseline = (
                config.measurement.baseline.enabled
                and config.measurement.baseline.strategy != "fresh"
            )
            steps = docker_steps(images_prepared=False, host_baseline=host_baseline)
        else:
            steps = list(STEPS_LOCAL)
        study_cb.begin_experiment(
            1,
            format_experiment_header(config),
            steps,
            runner_info=spec.to_runner_info() if spec else None,
        )

    exp_start = time.monotonic()
    result_files, experiment_results, warnings = run_single_experiment(
        study,
        manifest,
        study_dir,
        runner_specs=runner_specs,
        progress=progress,
        resolution_logs=resolution_logs,
    )
    exp_elapsed = time.monotonic() - exp_start

    if study_cb is not None:
        r = experiment_results[0] if experiment_results else None
        if r is not None:
            energy = r.total_energy_j if r.total_energy_j > 0 else None
            tp = r.avg_tokens_per_second if r.avg_tokens_per_second > 0 else None
            infer = r.total_inference_time_sec if r.total_inference_time_sec > 0 else None
            adj_e = r.energy_adjusted_j if r.energy_adjusted_j and r.energy_adjusted_j > 0 else None
            study_cb.end_experiment_ok(
                1,
                exp_elapsed,
                energy_j=energy,
                throughput_tok_s=tp,
                inference_time_sec=infer,
                adj_energy_j=adj_e,
                mj_per_tok_adjusted=r.mj_per_tok_adjusted,
                mj_per_tok_total=r.mj_per_tok_total,
            )
        else:
            study_cb.end_experiment_fail(1, exp_elapsed)

    return result_files, experiment_results, warnings


def _run_via_runner(
    study: StudyConfig,
    manifest: Any,
    study_dir: Path,
    runner_specs: Any = None,
    progress: ProgressCallback | None = None,
    skip_set: set[tuple[str, int]] | None = None,
    no_lock: bool = False,
    resolution_logs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Delegate to StudyRunner for multi-experiment / multi-cycle runs."""
    from llenergymeasure.domain.progress import StudyProgressCallback
    from llenergymeasure.study.runner import StudyRunner

    study_progress = progress if isinstance(progress, StudyProgressCallback) else None
    runner = StudyRunner(
        study,
        manifest,
        study_dir,
        runner_specs=runner_specs,
        progress=study_progress,
        no_lock=no_lock,
        skip_set=skip_set,
        resolution_logs=resolution_logs,
    )
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
