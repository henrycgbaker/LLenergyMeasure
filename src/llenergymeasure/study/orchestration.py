"""Study-layer orchestration: the single owner of the run loop's setup + assembly.

``orchestrate_study`` is the study layer's orchestration entry point. It owns
what used to live in ``api/_impl._run``: resolve runner specs (preflight),
create the study directory + manifest (or reattach one for resume), write the
study-level artefacts, build the per-experiment resolution logs, branch between
the single-experiment in-process path and the multi-experiment ``StudyRunner``,
and assemble the final ``StudyResult``.

The ``api`` layer is a thin adapter over this module: it translates the public
call forms into a resolved ``StudyConfig`` and the orchestrator's explicit
internal parameters, then delegates. Keeping the orchestration here (not in
``api``) makes the layer boundary honest - the config layer never imports upward
into ``study`` and the orchestration never re-imports downward from ``api``.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from llenergymeasure.config.models import StudyConfig
from llenergymeasure.config.runner_spec import RunnerSpec
from llenergymeasure.config.ssot import RUNNER_CONTAINER
from llenergymeasure.domain.bundle_artefacts import (
    STUDY_ARTEFACTS_DIR,
    SYSTEM_FILENAME,
    SYSTEM_OVERRIDES_FILENAME,
)
from llenergymeasure.domain.experiment import ExperimentResult, StudyResult, StudySummary
from llenergymeasure.domain.progress import ProgressCallback
from llenergymeasure.study.single import run_single_experiment

logger = logging.getLogger(__name__)


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


def orchestrate_study(
    study: StudyConfig,
    skip_preflight: bool = False,
    progress: ProgressCallback | None = None,
    resume_dir: Path | None = None,
    results_dir_override: Path | None = None,
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

    The two directory parameters are single-purpose (the ``api`` adapter maps the
    overloaded public ``output_dir`` onto them):
    - ``resume_dir``: an explicit study directory to reattach and resume into.
    - ``results_dir_override``: the results-dir override for a fresh run
      (precedence head, above YAML ``output.results_dir`` and user config).
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

    # Server-capable-entry-path contract (R7W CONTRACT NOTE): every entry that
    # dispatches a server config must apply the user-config warmup overlay (or
    # reject). ``api.load_study`` overlays before dedup, but ``run_experiment`` and
    # ``run_study(StudyConfig)`` bypass it; this single choke point overlays every
    # server experiment here so the ServerSession always reads the overlay-resolved
    # protocol. Idempotent (re-applying on the load_study path recomputes the same
    # value), a no-op for offline configs, and dedup-safe (the tool-wide overlay is
    # uniform across a run, so it never regroups within-run dedup).
    _apply_server_warmup_overlay_to_study(study, user_config)

    runner_specs, system_overrides = _resolve_runner_specs(
        study, user_config, preresolved, skip_preflight, progress
    )

    # Resolve results_dir: resume_dir takes priority, then the fresh-run chain
    # (CLI -o override > YAML output.results_dir > user config > built-in default).
    if resume_dir is not None:
        study_dir = resume_dir
        # Resume: load the existing manifest written by prepare_resume_manifest()
        # and wrap it without rebuilding or overwriting the prepared manifest.
        from llenergymeasure.study.resume import load_resume_state

        loaded_manifest, _ = load_resume_state(study_dir)
        manifest = ManifestWriter.from_existing(study_dir, loaded_manifest)
    else:
        if results_dir_override is not None:
            results_dir_str = str(results_dir_override)
        else:
            results_dir_str = (
                study.output.results_dir or user_config.output.results_dir or "./results"
            )
        study_dir = create_study_dir(study.study_name, Path(results_dir_str))
        manifest = ManifestWriter(study, study_dir)

    # Create _study-artefacts/ once for config copy, skipped log, and study-level env.
    artefacts_dir = _ensure_study_artefacts_dir(study_dir)

    _write_study_artefacts(study, artefacts_dir, system_overrides, config_path)

    resolution_logs = _build_resolution_logs(study, cli_overrides)

    wall_start = time.monotonic()
    # Server experiments always route through StudyRunner (the single server call
    # site is StudyRunner._run_one -> ServerSession); the in-process single path
    # is offline-only. Offline single experiments still take the fast path
    # unchanged (byte-identical behaviour).
    is_server_study = any(exp.serving_mode == "server" for exp in study.experiments)
    is_single = (
        len(study.experiments) == 1 and study.study_execution.n_cycles == 1 and not is_server_study
    )

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

    Runs precedence-based multi-engine elevation + Docker preflight (explicit
    runner pins win; auto-resolved engines elevate to Docker, raising
    PreFlightError when a local-pinned engine is not importable on the host or an
    auto-resolved engine needs Docker but it is unavailable), emits preflight
    progress, and warns on two resolved-plan conflicts: mixed local/Docker
    runners, and a GPU selector set both via LLEM_DOCKER_GPUS and
    study_execution.gpu_indices when a Docker container will launch.
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

    # Physical GPU selector precedence (env>config): warn once per study dispatch
    # when both LLEM_DOCKER_GPUS and study_execution.gpu_indices are set and a
    # Docker container will actually launch. GPU scoping only affects containers,
    # so a study with no Docker runner never triggers the warning. Single choke
    # point for both dispatch paths (single-experiment and StudyRunner).
    if RUNNER_CONTAINER in modes:
        from llenergymeasure.utils.env_config import warn_on_gpu_selector_conflict

        warn_on_gpu_selector_conflict(study.study_execution.gpu_indices)

    return runner_specs, system_overrides


def _apply_server_warmup_overlay_to_study(study: StudyConfig, user_config: Any) -> None:
    """Overlay the tool-wide user-config server warmup onto every server experiment.

    The server-capable-entry-path contract (R7W): ``api.load_study`` applies the
    overlay before dedup, but ``run_experiment`` / ``run_study(StudyConfig)`` reach
    the runner without it. Applying it here - the universal orchestration choke
    point that already loaded the user config - guarantees the ServerSession reads
    the overlay-resolved warmup on EVERY entry path. Idempotent and a no-op for
    offline configs and when the user config carries no warmup layer.
    """
    if not any(exp.serving_mode == "server" for exp in study.experiments):
        return
    from llenergymeasure.config.precedence import apply_server_warmup_overlay

    for exp in study.experiments:
        if exp.serving_mode == "server":
            apply_server_warmup_overlay(exp, user_config)


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

    # Write study-level system.json (installed_packages + software constants).
    try:
        from llenergymeasure.harness.environment import collect_software_environment

        sw_env = collect_software_environment()
        study_env = {
            "study_design_hash": _study_hash,
            "study_name": _study_name,
            **sw_env,
        }
        env_path = artefacts_dir / SYSTEM_FILENAME
        env_path.write_text(json.dumps(study_env, indent=2), encoding="utf-8")
        logger.info("Study-level system snapshot written to %s", env_path)
    except Exception as exc:
        logger.warning("Failed to write study-level system.json: %s", exc)


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
        is_docker = spec and spec.mode == RUNNER_CONTAINER
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
                energy_per_token_mj_adjusted=r.energy_per_token_mj_adjusted,
                energy_per_token_mj_total=r.energy_per_token_mj_total,
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

    from llenergymeasure.study.server_session import ServerSessionResult

    warnings: list[str] = []
    experiment_results: list[ExperimentResult | None] = []
    for r in raw_results:
        if isinstance(r, dict):
            warnings.append(r.get("message", "Unknown error"))
            experiment_results.append(None)
        elif isinstance(r, ServerSessionResult):
            # A server session's N window results are not (yet) an ExperimentResult
            # - per-window bundle persistence is SM10 and metrics derivation is SM12
            # - so they do not enter StudyResult.experiments at SM9. Surface a
            # one-line session summary and keep the offline energy sum untouched.
            experiment_results.append(None)
            warnings.append(_server_session_summary(r))
        else:
            experiment_results.append(r)

    return runner.result_files, experiment_results, warnings


def _server_session_summary(result: Any) -> str:
    """One-line human summary of a server session's outcome (SM9-interim surfacing)."""
    valid = sum(1 for level in result.levels if level.valid)
    total = len(result.levels)
    verdict = "valid" if result.valid else ("aborted" if result.aborted else "invalid")
    return (
        f"server session ({result.engine}) {verdict}: {valid}/{total} level(s) passed "
        f"the stability gate, {result.window_count} measured window(s). Per-window "
        "bundles + metrics land with SM10/SM12."
    )
