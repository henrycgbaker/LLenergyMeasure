"""Stage the measured window into a temp dir: warnings, result assembly, sidecars.

:func:`persist_and_assemble` is the post-window orchestration: it decides the
timeseries sidecar path, collects the mode-agnostic measurement warnings, calls
the result-assembly seam, then writes the timeseries Parquet and config.json
sidecars into the staging directory. It reads per-model-load state off the
``_EngineLifetime`` and the per-window products off the ``_MeasuredWindow``. The
files land in a temp staging area; the S4 ``BundleWriter`` finalises them into
the per-experiment bundle - this module stages, it does not own bundle layout.

The lazy ``write_timeseries_parquet`` wrapper here is a monkeypatch surface:
tests patch it at ``llenergymeasure.harness.staging.write_timeseries_parquet``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure.domain.bundle_artefacts import TIMESERIES_FILENAME
from llenergymeasure.domain.progress import emit_substep
from llenergymeasure.engines.protocol import EnginePlugin
from llenergymeasure.harness.lifecycle import _EngineLifetime
from llenergymeasure.harness.measurement_warnings import collect_warnings
from llenergymeasure.harness.result_assembly import _ConfigMethodology, build_result
from llenergymeasure.harness.window import _MeasuredWindow
from llenergymeasure.results.persistence import save_config_sidecar

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.domain.experiment import ExperimentResult
    from llenergymeasure.domain.progress import ProgressCallback

logger = logging.getLogger(__name__)


def write_timeseries_parquet(
    samples: list[PowerThermalSample],
    path: Path,
    *,
    experiment_id: str | None = None,
    declared_config_hash: str | None = None,
) -> Path:  # pragma: no cover
    from llenergymeasure.harness.timeseries import write_timeseries_parquet as _wts

    return _wts(
        samples,
        path,
        experiment_id=experiment_id,
        declared_config_hash=declared_config_hash,
    )


def persist_and_assemble(
    *,
    engine: EnginePlugin,
    config: ExperimentConfig,
    lifetime: _EngineLifetime,
    gpu_indices: list[int] | None,
    window: _MeasuredWindow,
    output_dir: Path | str | None,
    save_timeseries: bool,
    progress: ProgressCallback | None,
) -> ExperimentResult:
    """Write the timeseries + config sidecars and assemble the ExperimentResult.

    Reads the per-model-load state (snapshot, baseline, warmup, memory/load
    timings, prompt count) off ``lifetime`` and the per-window measurement
    products off ``window``.
    """
    _p = progress

    # 14. Decide the timeseries sidecar path. The Parquet file is written after
    # the result is assembled (step 17) so it can carry the experiment identity
    # as file-level metadata, mirroring the JSON sidecars.
    resolved_output_dir = Path(output_dir) if output_dir is not None else None
    if _p:
        _p.on_step_start(
            "save",
            "Saving",
            "writing results",
        )
        t0_save = time.perf_counter()

    core = window.core
    write_timeseries = bool(
        save_timeseries and resolved_output_dir is not None and core.timeseries_samples
    )
    # Relative name recorded in result JSON; the file lands at this basename.
    timeseries_path: str | None = TIMESERIES_FILENAME if write_timeseries else None

    # 15. Collect measurement quality warnings
    duration_sec = (core.end_time - core.start_time).total_seconds()
    measurement_warnings = collect_warnings(
        duration_sec,
        core.timeseries_samples,
        gpu_indices,
        core.energy_measurement,
        energy_sampler_reasons=core.energy_sampler_reasons,
    )

    # 16. Assemble ExperimentResult
    result, methodology = build_result(
        engine_name=engine.name,
        config=config,
        output=window.output,
        model_memory_mb=lifetime.model_memory_mb,
        start_time=core.start_time,
        end_time=core.end_time,
        duration_sec=duration_sec,
        throttle_info=core.throttle_info,
        energy_measurement=core.energy_measurement,
        baseline=lifetime.baseline,
        flops_result=window.flops_result,
        timeseries_path=timeseries_path,
        timeseries_samples=core.timeseries_samples,
        measurement_warnings=measurement_warnings,
        warmup_result=lifetime.warmup_result,
        prompt_count=len(lifetime.prompts),
        model_load_time_sec=lifetime.model_load_time_sec,
    )
    emit_substep(_p, "save", "result assembled")

    # 17. Write timeseries Parquet sidecar, tagged with the assembled identity.
    if write_timeseries and resolved_output_dir is not None:
        write_timeseries_parquet(
            core.timeseries_samples,
            resolved_output_dir / TIMESERIES_FILENAME,
            experiment_id=result.experiment_id,
            declared_config_hash=result.declared_config_hash,
        )
        emit_substep(_p, "save", "timeseries parquet written")

    # 18. Write config.json sidecar (observed-params + observed_config_hash).
    # Written to output_dir (temp dir, same as timeseries.parquet) so the runner
    # can move it to the per-experiment directory.
    if resolved_output_dir is not None:
        write_config_sidecar(
            output=window.output,
            config=config,
            result=result,
            engine_name=engine.name,
            methodology=methodology,
            output_dir=resolved_output_dir,
        )
        emit_substep(_p, "save", "config sidecar written")

    if _p:
        _p.on_step_done("save", time.perf_counter() - t0_save)

    return result


def write_config_sidecar(
    output: Any,
    config: Any,
    result: Any,
    engine_name: str,
    methodology: _ConfigMethodology,
    output_dir: Path,
) -> None:
    """Write ``config.json`` sidecar to ``output_dir`` (temp staging area).

    Extracts observed params from ``output.extras`` (populated by each engine's
    ``_capture_effective_params`` after inference), computes ``observed_config_hash``
    from the observed-config hashing pipeline, and writes the sidecar atomically.
    The runner's ``_save_and_record`` moves this file to the per-experiment
    directory alongside ``result.json``.

    This sidecar is the authoritative home for engine identity (``engine`` /
    ``engine_version``), ``model_name``, and the measurement ``methodology``
    fields. ``result.json`` carries ``engine`` and ``model_name`` as convenience
    copies only; the rest live here exclusively.

    Best-effort - failures are logged at DEBUG to avoid masking measurement results.
    """
    try:
        from llenergymeasure.domain.hashing import build_observed_view, hash_config

        obs_engine = output.extras.get("observed_engine_params", {}) or {}
        obs_sampling = output.extras.get("observed_sampling_params", {}) or {}
        lib_ver = output.extras.get("library_version", "unknown") or "unknown"

        # Compute observed_config_hash from extracted native-type state.
        # llem_execution + measurement come from the ran config so the observed
        # hash covers the same identity dimensions as the resolved hash.
        task_dict = config.task.model_dump(mode="python")
        execution = config.active_llem_execution()
        execution_dump = execution.model_dump(mode="python") if execution is not None else {}
        observed_view = build_observed_view(
            engine=engine_name,
            task=task_dict,
            observed_engine_params=obs_engine,
            observed_sampling_params=obs_sampling,
            llem_execution=execution_dump,
            measurement=config.measurement.model_dump(mode="python"),
        )
        obs_hash = hash_config(observed_view)

        # Full user-declared config, recorded so the observed-collision miner can
        # attribute a shared observed_config_hash to the declared fields that
        # varied. Guarded separately: a declared-dump failure must not cost us the
        # observed hash written below.
        try:
            declared_config: dict[str, object] | None = config.model_dump(mode="json")
        except Exception:  # pragma: no cover - declared dump is best-effort
            declared_config = None

        save_config_sidecar(
            output_dir,
            experiment_id=result.experiment_id,
            config_hash=result.declared_config_hash,
            engine=engine_name,
            engine_version=lib_ver,
            model_name=config.task.model,
            measurement_methodology=methodology.measurement_methodology,
            steady_state_window=methodology.steady_state_window,
            measurement_window_discard_fraction=methodology.measurement_window_discard_fraction,
            steady_state_not_detected=methodology.steady_state_not_detected,
            observed_engine_params=obs_engine if obs_engine else None,
            observed_sampling_params=obs_sampling if obs_sampling else None,
            observed_config_hash=obs_hash,
            declared_config=declared_config,
        )
    except Exception as exc:
        logger.debug("Config sidecar write failed (non-fatal): %s", exc)
