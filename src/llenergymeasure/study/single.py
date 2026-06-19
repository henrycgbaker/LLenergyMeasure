"""Single-experiment in-process execution path - shared seam in the study layer.

A study with exactly one experiment and ``n_cycles == 1`` runs in-process (or
dispatches to a DockerRunner directly) rather than spawning a subprocess via
StudyRunner. This module is that path. It lives in the study layer so the
result-saving helper (``_save_and_record``) is an intra-package call rather than
an api-layer reach into a study-layer private symbol.

The api layer (``api/_impl._run``) keeps the single-experiment dispatch decision
and the study-level progress begin/end block; it delegates the actual execution
body to ``run_single_experiment`` here.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.models import StudyConfig
from llenergymeasure.config.ssot import RUNNER_DOCKER, TEMP_PREFIX_TIMESERIES
from llenergymeasure.device.gpu_info import _resolve_gpu_indices
from llenergymeasure.domain.experiment import ExperimentResult
from llenergymeasure.domain.progress import ProgressCallback

if TYPE_CHECKING:
    from llenergymeasure.infra.runner_resolution import RunnerSpec
    from llenergymeasure.study.manifest import ManifestWriter


def run_single_experiment(
    study: StudyConfig,
    manifest: ManifestWriter,
    study_dir: Path,
    *,
    runner_specs: dict[str, RunnerSpec] | None = None,
    progress: ProgressCallback | None = None,
    resolution_logs: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[str], list[ExperimentResult | None], list[str]]:
    """Run a single experiment in-process or via DockerRunner directly.

    When runner_specs resolves the engine to "docker", uses DockerRunner directly
    (no subprocess spawning). Otherwise runs in-process via the engine.

    Errors from run_preflight() and harness.run(engine, config) propagate unchanged (PreFlightError,
    EngineError). Only result-saving errors are caught so a save failure does not
    discard a completed measurement.
    """
    from llenergymeasure.domain.experiment import compute_declared_config_hash
    from llenergymeasure.study.runner import _save_and_record

    config = study.experiments[0]
    config_hash = compute_declared_config_hash(config)
    cycle = 1

    # Pre-dispatch GPU memory residual check (MEAS-01, MEAS-02)
    # Mirrors the pattern used in StudyRunner._run_one() and _run_one_docker().
    from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

    check_gpu_memory_residual()

    # Collect environment snapshot once - used for both harness and environment.json sidecar
    from llenergymeasure.harness.environment import collect_environment_snapshot

    snapshot = collect_environment_snapshot()

    # Check runner spec for this engine
    spec = runner_specs.get(config.engine) if runner_specs else None

    manifest.mark_running(config_hash, cycle)

    save_ts = study.output.save_timeseries
    ts_tmpdir: Path | None = None  # Only set for local path

    if spec is not None and spec.mode == RUNNER_DOCKER:
        # Docker path: dispatch to container directly (no subprocess)
        from llenergymeasure.infra.docker_errors import docker_exc_to_failure
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.container_lifecycle import persist_failure_artefacts
        from llenergymeasure.utils.exceptions import DockerError

        image = spec.image if spec.image is not None else get_default_image(config.engine)

        docker_runner = DockerRunner(
            image=image,
            timeout=study.study_execution.experiment_timeout_seconds,
            silence_timeout=study.study_execution.stdout_silence_timeout_seconds,
            source=spec.source,
            extra_mounts=spec.extra_mounts,
        )
        docker_ts_dir: Path | None = None
        try:
            result, docker_ts_dir = docker_runner.run(
                config, progress=progress, save_timeseries=save_ts
            )
        except DockerError as exc:
            # Same failure classification and artefact persistence as the
            # multi-experiment runner path, so single- and multi-experiment
            # Docker failures report identical shapes and both leave a
            # container.log + error JSON in failed-runs/ for debugging.
            failure: dict[str, Any] = docker_exc_to_failure(exc, config_hash)
            persist_failure_artefacts(exc, study_dir, config_hash, cycle, failure)
            manifest.mark_failed(config_hash, cycle, failure["type"], failure["message"])
            return [], [None], [failure["message"]]
        # Docker path: ts_tmpdir comes from DockerRunner
        ts_tmpdir = docker_ts_dir
    else:
        # Local in-process path - errors propagate naturally (PreFlightError, EngineError)
        import tempfile

        from llenergymeasure.engines import get_engine
        from llenergymeasure.harness import MeasurementHarness
        from llenergymeasure.harness.preflight import run_preflight

        if progress:
            progress.on_step_start("container_preflight", "Checking", "CUDA, model access")
        t0 = time.perf_counter()
        run_preflight(config)
        if progress:
            progress.on_step_done("container_preflight", time.perf_counter() - t0)

        # Create temp dir for timeseries parquet (if enabled)
        ts_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES)) if save_ts else None

        try:
            engine = get_engine(config.engine)
            harness = MeasurementHarness()
            gpu_indices = _resolve_gpu_indices(config)
            result = harness.run(
                engine,
                config,
                snapshot=snapshot,
                gpu_indices=gpu_indices,
                progress=progress,
                output_dir=str(ts_tmpdir) if ts_tmpdir else None,
                save_timeseries=save_ts,
            )
        except Exception:
            if ts_tmpdir is not None:
                shutil.rmtree(ts_tmpdir, ignore_errors=True)
            raise

    # Handle error payload returned from Docker container (exit 0 but wrote error JSON)
    if isinstance(result, dict) and "type" in result:
        error_type = result.get("type", "UnknownError")
        error_message = result.get("message", "")
        manifest.mark_failed(config_hash, cycle, error_type, error_message)
        return [], [None], [error_message]

    result_files: list[str] = []
    warnings: list[str] = []
    _save_and_record(
        result,
        study_dir,
        manifest,
        config_hash,
        cycle,
        result_files,
        ts_source_dir=ts_tmpdir,
        environment_snapshot=snapshot,
        resolution_log=(resolution_logs or {}).get(config_hash),
    )

    # Clean up temp dirs
    if ts_tmpdir is not None and ts_tmpdir.exists():
        shutil.rmtree(ts_tmpdir, ignore_errors=True)

    return result_files, [result], warnings
