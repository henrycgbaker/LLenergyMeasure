"""Single-experiment in-process execution path - shared seam in the study layer.

A study with exactly one experiment and ``n_cycles == 1`` runs in-process (or
dispatches to a DockerRunner directly) rather than spawning a subprocess via
StudyRunner. This module is that path. It lives in the study layer so the
result-saving helper (``_save_and_record``) is an intra-package call rather than
an api-layer reach into a study-layer private symbol.

The study-layer orchestrator (``study.orchestration.orchestrate_study``) keeps the
single-experiment dispatch decision and the study-level progress begin/end block;
it delegates the actual execution body to ``run_single_experiment`` here.
"""

from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.config.models import StudyConfig
from llenergymeasure.config.ssot import RUNNER_CONTAINER, TEMP_PREFIX_TIMESERIES, engine_str
from llenergymeasure.device.gpu_info import _resolve_gpu_indices
from llenergymeasure.domain.experiment import ExperimentResult
from llenergymeasure.domain.progress import ProgressCallback

if TYPE_CHECKING:
    from llenergymeasure.config.runner_spec import RunnerSpec
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
    from llenergymeasure.study.runner import (
        _provenance_from_spec,
        _save_and_record,
    )

    config = study.experiments[0]
    config_hash = compute_declared_config_hash(config)
    cycle = 1

    # Pre-dispatch GPU memory residual check (MEAS-01, MEAS-02)
    # Mirrors the pattern used in StudyRunner._run_one() and _run_one_docker().
    from llenergymeasure.study.gpu_memory import check_gpu_memory_residual

    check_gpu_memory_residual()

    # Collect environment snapshot once - used for both harness and system.json sidecar
    from llenergymeasure.harness.environment import collect_environment_snapshot

    snapshot = collect_environment_snapshot()

    # Check runner spec for this engine
    spec = runner_specs.get(config.engine) if runner_specs else None

    manifest.mark_running(config_hash, cycle)

    save_ts = study.output.save_timeseries
    # Artefact staging dir. Local path always creates one (below); docker path
    # inherits the DockerRunner rescue dir, which may be None.
    ts_tmpdir: Path | None = None
    # Image that actually ran under docker dispatch (needed for the system.json
    # runner block's digest resolution); None on the local path.
    resolved_docker_image: str | None = None

    if spec is not None and spec.mode == RUNNER_CONTAINER:
        # Docker path: dispatch to container directly (no subprocess)
        from llenergymeasure.infra.docker_errors import docker_exc_to_failure
        from llenergymeasure.infra.docker_runner import DockerRunner
        from llenergymeasure.infra.image_registry import get_default_image
        from llenergymeasure.study.container_lifecycle import persist_failure_artefacts
        from llenergymeasure.utils.exceptions import DockerError

        image = spec.image if spec.image is not None else get_default_image(config.engine)
        resolved_docker_image = image

        # Physical GPU scoping precedence (env>config).
        gpu_indices = study.study_execution.gpu_indices

        docker_runner = DockerRunner(
            image=image,
            timeout=study.study_execution.experiment_timeout_seconds,
            silence_timeout=study.study_execution.stdout_silence_timeout_seconds,
            source=spec.source,
            gpu_indices=gpu_indices,
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
            # persist_failure_artefacts sets failure["log_file"]; pass it through so
            # the manifest actually points at the persisted container.log/error JSON.
            manifest.mark_failed(
                config_hash,
                cycle,
                failure["type"],
                failure["message"],
                log_file=failure.get("log_file"),
            )
            return [], [None], [failure["message"]]
        # Docker path: ts_tmpdir comes from DockerRunner
        ts_tmpdir = docker_ts_dir
    else:
        # Local in-process path - errors propagate naturally (PreFlightError, EngineError)
        import tempfile
        import uuid

        from llenergymeasure.engines import get_engine
        from llenergymeasure.harness import MeasurementHarness
        from llenergymeasure.harness.preflight import run_preflight
        from llenergymeasure.study.runtime_observations import capture_runtime_observations

        # Staging dir for harness artefacts. The harness writes config.json here
        # always (sole home of provenance, authoritative home of identity) and
        # timeseries.parquet when save_timeseries is on, so the dir is created
        # regardless of save_ts.
        ts_tmpdir = Path(tempfile.mkdtemp(prefix=TEMP_PREFIX_TIMESERIES))

        # Wrap the in-process body (preflight -> engine import -> harness.run) in
        # capture_runtime_observations so single-experiment studies emit
        # runtime_observations.jsonl like the multi-experiment worker path
        # (study/worker.py). Without it, report-gaps finds nothing for single-exp studies.
        obs_ctx = capture_runtime_observations(
            config,
            study_dir=study_dir,
            study_run_id=str(uuid.uuid4()),
            cycle=cycle,
            config_hash=config_hash,
        )
        try:
            with obs_ctx:
                if progress:
                    progress.on_step_start("container_preflight", "Checking", "CUDA, model access")
                t0 = time.perf_counter()
                run_preflight(config)
                if progress:
                    progress.on_step_done("container_preflight", time.perf_counter() - t0)

                engine = get_engine(config.engine)
                harness = MeasurementHarness()
                gpu_indices = _resolve_gpu_indices(config)
                result = harness.run(
                    engine,
                    config,
                    snapshot=snapshot,
                    gpu_indices=gpu_indices,
                    progress=progress,
                    output_dir=str(ts_tmpdir),
                    save_timeseries=save_ts,
                )
        except Exception as exc:
            # Persist the captured traceback into failed-runs/ and mark the
            # manifest failed, mirroring the Docker single-experiment branch so
            # a local failure leaves the same on-disk breadcrumb. The original
            # exception is then re-raised unchanged: the real type, message and
            # traceback still reach the CLI (format_error surfaces the live
            # traceback under -v), so display fidelity is preserved while the
            # failure also becomes debuggable from disk.
            from llenergymeasure.study.container_lifecycle import persist_failure_traceback

            local_failure: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
            persist_failure_traceback(
                study_dir, config_hash, cycle, traceback.format_exc(), local_failure
            )
            manifest.mark_failed(
                config_hash,
                cycle,
                local_failure["type"],
                local_failure["message"],
                log_file=local_failure.get("log_file"),
            )
            if ts_tmpdir is not None:
                shutil.rmtree(ts_tmpdir, ignore_errors=True)
            raise

    # Handle error payload returned from Docker container (exit 0 but wrote error JSON)
    if isinstance(result, dict) and "type" in result:
        error_type = result.get("type", "UnknownError")
        error_message = result.get("message", "")
        manifest.mark_failed(config_hash, cycle, error_type, error_message)
        return [], [None], [error_message]

    # Resolved-config hash for the config.json sidecar - mirrors the multi-experiment
    # runner path (StudyRunner._build_resolved_hashes) so single- and multi-experiment
    # studies write identical sidecar fields. Best-effort: None on failure.
    resolved_config_hash: str | None = None
    try:
        from llenergymeasure.study.hashing import build_resolved_view, hash_config

        resolved_config_hash = hash_config(build_resolved_view(config))
    except Exception:  # pragma: no cover - best-effort, mirrors the runner
        resolved_config_hash = None

    result_files: list[str] = []
    warnings: list[str] = []
    _save_and_record(
        result,
        study_dir,
        manifest,
        config_hash,
        cycle,
        result_files,
        model_name=config.task.model,
        engine=engine_str(config.engine),
        ts_source_dir=ts_tmpdir,
        environment_snapshot=snapshot,
        resolution_log=(resolution_logs or {}).get(config_hash),
        resolved_config_hash=resolved_config_hash,
        runner_provenance=_provenance_from_spec(spec, resolved_image=resolved_docker_image),
    )

    # Clean up temp dirs
    if ts_tmpdir is not None and ts_tmpdir.exists():
        shutil.rmtree(ts_tmpdir, ignore_errors=True)

    return result_files, [result], warnings
