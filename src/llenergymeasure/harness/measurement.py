"""MeasurementHarness implementation - owns the measurement lifecycle for any EnginePlugin.

The harness extracts the ~600 lines of identical measurement infrastructure
duplicated across transformers.py and vllm.py into a single location.
Engines become thin plugins implementing the 4-method EnginePlugin protocol.

This module holds the implementation; ``llenergymeasure.harness`` re-exports the
public surface plus the module-level helpers that tests patch.
"""

from __future__ import annotations

import importlib.util
import logging
import time
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from llenergymeasure._version import __version__
from llenergymeasure.config.ssot import TIMEOUT_ENV_SNAPSHOT
from llenergymeasure.datasets import load_prompts
from llenergymeasure.domain.bundle_artefacts import TIMESERIES_FILENAME
from llenergymeasure.domain.experiment import (
    AggregationMetadata,
    ExperimentResult,
    compute_declared_config_hash,
    mj_per_token,
)
from llenergymeasure.domain.progress import STEP_BASELINE
from llenergymeasure.energy import select_energy_sampler
from llenergymeasure.engines.protocol import EnginePlugin, InferenceOutput
from llenergymeasure.harness.warmup import thermal_floor_wait, warmup_until_converged
from llenergymeasure.results.persistence import save_config_sidecar
from llenergymeasure.utils.formatting import bytes_to_mb

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.metrics import FlopsResult, WarmupResult
    from llenergymeasure.domain.progress import ProgressCallback
    from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module-level helpers (extracted from both engines - byte-identical copies)
# ---------------------------------------------------------------------------


def _cuda_sync() -> None:
    """Synchronise CUDA at measurement boundary.

    Best-effort - failures are non-fatal and silently ignored.
    """
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass  # Non-fatal - best effort sync


def _capture_observed_params_into_output(
    engine: Any,
    config: Any,
    model: Any,
    output: Any,
) -> None:
    """Call ``engine.capture_observed_params`` and merge the result into ``output.extras``.

    Invoked post-window (after ``t_inference_end`` + ``_cuda_sync``) so capture
    overhead does not perturb energy or timing measurements.

    Best-effort: engines that haven't implemented the method yet silently skip.
    The result dict is merged into ``output.extras`` under the standard keys
    ``observed_engine_params``, ``observed_sampling_params``, and
    ``library_version``.
    """
    try:
        capture = getattr(engine, "capture_observed_params", None)
        if capture is None:
            return
        params = capture(config, model, output)
        if isinstance(params, dict):
            output.extras["observed_engine_params"] = params.get("engine", {})
            output.extras["observed_sampling_params"] = params.get("sampling", {})
            output.extras["library_version"] = params.get("library_version", "unknown")
    except Exception as exc:
        logger.debug("capture_observed_params failed (non-fatal): %s", exc)


def _check_persistence_mode(gpu_indices: list[int] | None = None) -> bool:
    """Check whether GPU persistence mode is enabled on all specified GPUs.

    Checks every GPU in gpu_indices (defaults to [0] when None). Returns False
    only if persistence mode is definitively off on at least one GPU.

    Args:
        gpu_indices: GPU device indices to check. Defaults to [0] when None.

    Returns:
        True if persistence mode is on (or unknown) for all GPUs,
        False if definitively off on any GPU.
    """
    indices = gpu_indices if gpu_indices is not None else [0]
    try:
        import pynvml

        from llenergymeasure.device.gpu_info import nvml_context

        with nvml_context():
            for idx in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mode = pynvml.nvmlDeviceGetPersistenceMode(handle)
                if mode == pynvml.NVML_FEATURE_DISABLED:
                    return False
        return True
    except Exception:
        return True  # Unknown - don't generate spurious warning


# ---------------------------------------------------------------------------
# Top-level imports used in harness (lazy in engines, top-level here for
# patching in tests).  The actual heavy work is inside the engine plugins.
# ---------------------------------------------------------------------------


def collect_environment_snapshot() -> EnvironmentSnapshot:  # pragma: no cover
    from llenergymeasure.harness.environment import (
        collect_environment_snapshot as _snap,
    )

    return _snap()


def collect_environment_snapshot_async() -> Future[EnvironmentSnapshot]:  # pragma: no cover
    from llenergymeasure.harness.environment import (
        collect_environment_snapshot_async as _snap_async,
    )

    return _snap_async()


def measure_baseline_power(
    duration_sec: float,
    gpu_indices: list[int] | None = None,
) -> BaselineCache | None:  # pragma: no cover
    from llenergymeasure.harness.baseline import measure_baseline_power as _mbp

    return _mbp(duration_sec=duration_sec, gpu_indices=gpu_indices)


def estimate_flops_palm(
    model: Any, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult:  # pragma: no cover
    from llenergymeasure.harness.flops import estimate_flops_palm as _efp

    return _efp(model=model, n_input_tokens=n_input_tokens, n_output_tokens=n_output_tokens)


def estimate_flops_palm_from_config(
    model_name: str, n_input_tokens: int, n_output_tokens: int
) -> FlopsResult | None:  # pragma: no cover
    from llenergymeasure.harness.flops import estimate_flops_palm_from_config as _efpc

    return _efpc(
        model_name=model_name,
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
    )


def write_timeseries_parquet(
    samples: list[PowerThermalSample], path: Path
) -> Path:  # pragma: no cover
    from llenergymeasure.harness.timeseries import write_timeseries_parquet as _wts

    return _wts(samples, path)


def collect_measurement_warnings(
    duration_sec: float,
    gpu_persistence_mode: bool,
    temp_start_c: float | None,
    temp_end_c: float | None,
    nvml_sample_count: int,
    energy_measurement_present: bool = True,
) -> list[str]:  # pragma: no cover
    from llenergymeasure.harness.measurement_warnings import (
        collect_measurement_warnings as _cmw,
    )

    return _cmw(
        duration_sec=duration_sec,
        gpu_persistence_mode=gpu_persistence_mode,
        temp_start_c=temp_start_c,
        temp_end_c=temp_end_c,
        nvml_sample_count=nvml_sample_count,
        energy_measurement_present=energy_measurement_present,
    )


class PowerThermalSampler:  # pragma: no cover
    """Thin re-export so tests can patch llenergymeasure.harness.PowerThermalSampler."""

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        from llenergymeasure.device.power_thermal import PowerThermalSampler as _PTS

        return _PTS(*args, **kwargs)

    def __enter__(self) -> Any: ...  # type: ignore[empty-body]

    def __exit__(self, *args: Any) -> None: ...  # type: ignore[empty-body]


def _emit_substep(
    progress: ProgressCallback | None, step: str, text: str, elapsed: float = 0.0
) -> None:
    """Emit a substep event to the progress callback when one is registered."""
    if progress:
        progress.on_substep(step, text, elapsed)


@dataclass
class _MeasuredWindow:
    """Outputs of the measured inference window (steps 7-13 of the run lifecycle)."""

    output: InferenceOutput
    thermal_info: Any
    timeseries_samples: list[PowerThermalSample]
    energy_measurement: Any
    flops_result: Any
    start_time: datetime
    end_time: datetime


# ---------------------------------------------------------------------------
# MeasurementHarness
# ---------------------------------------------------------------------------


class MeasurementHarness:
    """Orchestrates the measurement lifecycle for any EnginePlugin.

    Engines are thin plugins implementing EnginePlugin (load_model, warmup,
    run_inference, cleanup). The harness owns everything else:
    environment snapshot, baseline power, energy tracking, CUDA sync,
    thermal floor wait, FLOPs estimation, timeseries, warnings, and result assembly.
    """

    def run(
        self,
        engine: EnginePlugin,
        config: ExperimentConfig,
        snapshot: EnvironmentSnapshot | None = None,
        gpu_indices: list[int] | None = None,
        progress: ProgressCallback | None = None,
        output_dir: Path | str | None = None,
        save_timeseries: bool = True,
        baseline: BaselineCache | None = None,
    ) -> ExperimentResult:
        """Run a complete measurement using the given engine plugin.

        Args:
            engine: EnginePlugin instance (transformers, vllm, tensorrt, ...).
            config: Fully resolved experiment configuration.
            snapshot: Pre-collected environment snapshot (study-level cache).
                      When None, collected in a background thread during model load.
            gpu_indices: GPU device indices to monitor for energy/thermal measurement.
                         Defaults to [0] (single GPU, backward compatible) when None.
            progress: Optional callback for step-by-step progress reporting.
                      When None, no progress events are emitted (backward compatible).
            output_dir: Directory for timeseries parquet output. None = no disk writes.
                        Passed as runtime param by the study runner, not from config.
            save_timeseries: Whether to persist GPU timeseries to Parquet sidecar.
                             Controlled by OutputConfig.save_timeseries at study level.
            baseline: Pre-measured baseline power (study-level cache). When provided
                      and config.measurement.baseline.enabled, skips fresh measurement and reuses
                      this value (marked as cached in the energy breakdown).

        Returns:
            ExperimentResult with all measurement fields populated.

        Raises:
            EngineError: If model loading or inference fails.
            PreFlightError: If pre-flight checks fail before GPU allocation.
        """
        # 1. Environment snapshot - start background thread (before model loading)
        snapshot_future: Future[EnvironmentSnapshot] | None = None
        if snapshot is None:
            logger.debug("Collecting environment snapshot (background thread)")
            snapshot_future = collect_environment_snapshot_async()

        # 2. Baseline power measurement (before model load)
        baseline = self._measure_or_reuse_baseline(config, baseline, gpu_indices, progress)

        # 3-4. Load model, join env snapshot, capture model memory baseline
        model, snapshot, model_memory_mb, model_load_time_sec = self._load_model_and_snapshot(
            engine, config, snapshot, snapshot_future, gpu_indices, progress
        )

        # 4b. Load prompts - BEFORE the measurement window (methodology fix)
        prompts = self._load_prompts(config, progress)

        try:
            # 5-6. Warmup convergence + thermal floor wait
            warmup_result = self._run_warmup(engine, config, model, prompts, progress)

            # 7-13. Measured inference window (energy, timing, FLOPs)
            window = self._run_measured_inference(
                engine, config, model, prompts, gpu_indices, progress
            )

        finally:
            # Always release model from memory even on exception
            engine.cleanup(model)

        # 14-17. Persist sidecars + assemble the ExperimentResult
        return self._persist_and_assemble(
            engine=engine,
            config=config,
            snapshot=snapshot,
            baseline=baseline,
            warmup_result=warmup_result,
            model_memory_mb=model_memory_mb,
            model_load_time_sec=model_load_time_sec,
            prompt_count=len(prompts),
            gpu_indices=gpu_indices,
            window=window,
            output_dir=output_dir,
            save_timeseries=save_timeseries,
            progress=progress,
        )

    # -------------------------------------------------------------------------
    # Run lifecycle phases (extracted from run())
    # -------------------------------------------------------------------------

    def _measure_or_reuse_baseline(
        self,
        config: ExperimentConfig,
        baseline: BaselineCache | None,
        gpu_indices: list[int] | None,
        progress: ProgressCallback | None,
    ) -> BaselineCache | None:
        """Measure baseline idle power, or mark a study-supplied baseline as cached.

        Returns the baseline to thread into the energy breakdown (None when baseline
        measurement is disabled or unavailable).
        """
        _p = progress

        if baseline is not None and config.measurement.baseline.enabled:
            # For cached/validated strategies, progress events are emitted by
            # study/runner.py::_get_baseline. Here we only mark from_cache for
            # create_energy_breakdown method attribution.
            from dataclasses import replace as _dc_replace

            baseline = _dc_replace(baseline, from_cache=True)
            logger.debug(
                "Using study-level baseline cache: %.1fW (%d samples)",
                baseline.power_w,
                baseline.sample_count,
            )
        elif config.measurement.baseline.enabled:
            dur = config.measurement.baseline.duration_seconds
            logger.debug("Measuring baseline power (%.0fs)...", dur)
            if _p:
                _p.on_step_start(STEP_BASELINE, "Measuring", f"baseline idle power ({dur:.0f}s)")
                t0 = time.perf_counter()
            baseline = measure_baseline_power(dur, gpu_indices=gpu_indices)
            if baseline is not None:
                cache_label = " (cached)" if baseline.from_cache else ""
                _emit_substep(
                    _p,
                    STEP_BASELINE,
                    f"baseline: {baseline.power_w:.1f}W"
                    f" ({baseline.sample_count} samples{cache_label})",
                )
                if _p and baseline.from_cache:
                    _p.on_step_update(STEP_BASELINE, f"cached baseline {baseline.power_w:.1f}W")
            if _p:
                _p.on_step_done(STEP_BASELINE, time.perf_counter() - t0)
        elif _p:
            _p.on_step_skip(STEP_BASELINE, "disabled")
        return baseline

    def _load_model_and_snapshot(
        self,
        engine: EnginePlugin,
        config: ExperimentConfig,
        snapshot: EnvironmentSnapshot | None,
        snapshot_future: Future[EnvironmentSnapshot] | None,
        gpu_indices: list[int] | None,
        progress: ProgressCallback | None,
    ) -> tuple[Any, EnvironmentSnapshot | None, float, float]:
        """Load the model, join the background environment snapshot, and capture the
        post-load GPU memory baseline.

        Returns (model, snapshot, model_memory_mb, load_time_sec). The memory
        baseline must be captured here - before warmup allocates KV cache.
        load_time_sec brackets engine.load_model() alone: model load plus any
        engine build/compile the plugin performs there. It lands on the result
        as model_load_time_sec (non-energy metadata; this phase precedes the
        NVML measurement window).
        """
        _p = progress

        # 3. Load model via engine plugin
        if _p:
            _p.on_step_start("model", "Loading", f"model {config.task.model}")
        t0 = time.perf_counter()

        # Build model substep callback
        def _on_model_substep(text: str, elapsed: float) -> None:
            _emit_substep(_p, "model", text, elapsed)

        model = engine.load_model(config, on_substep=_on_model_substep)

        load_time_sec = time.perf_counter() - t0
        if _p:
            _p.on_step_done("model", load_time_sec)

        # 3b. Join snapshot future - collection hidden behind model loading
        if snapshot_future is not None:
            snapshot = snapshot_future.result(timeout=TIMEOUT_ENV_SNAPSHOT)

        # 4. Capture model memory baseline immediately after model load.
        # Must happen BEFORE warmup, which allocates KV cache.
        model_memory_mb = self._capture_model_memory_mb(gpu_indices=gpu_indices)
        if model_memory_mb > 0:
            _emit_substep(_p, "model", f"model memory: {model_memory_mb:.0f}MB")

        return model, snapshot, model_memory_mb, load_time_sec

    def _load_prompts(
        self,
        config: ExperimentConfig,
        progress: ProgressCallback | None,
    ) -> list[str]:
        """Load and tokenise prompts before the measurement window (methodology fix)."""
        _p = progress

        if _p:
            _p.on_step_start(
                "prompts",
                "Loading",
                f"prompts ({config.task.dataset.n_prompts} {config.task.dataset.source})",
            )
            t0_prompts = time.perf_counter()
        prompts = load_prompts(config)
        logger.debug("Loaded %d prompts via dataset loader", len(prompts))
        _emit_substep(_p, "prompts", f"tokenised {len(prompts)} prompts")
        if _p:
            _p.on_step_done("prompts", time.perf_counter() - t0_prompts)
        return prompts

    def _run_warmup(
        self,
        engine: EnginePlugin,
        config: ExperimentConfig,
        model: Any,
        prompts: list[str],
        progress: ProgressCallback | None,
    ) -> WarmupResult:
        """Run engine warmup to convergence, then wait out the thermal floor.

        Returns the WarmupResult with thermal_floor_wait_s populated.
        """
        _p = progress

        # 5. Warmup
        if _p:
            _p.on_step_start(
                "warmup", "Warming up", f"up to {config.measurement.warmup.max_prompts} prompts"
            )
            t0_warmup = time.perf_counter()

        # Probe call determines warmup strategy:
        # > 0.0 → CV-based convergence (Transformers), 0.0 → kernel warmup (vLLM/TRT-LLM)
        if config.measurement.warmup.enabled:
            first_latency = engine.run_warmup_prompt(config, model, prompts[0])
        else:
            first_latency = 0.0

        if first_latency > 0.0:
            warmup_substep = (
                (lambda text, elapsed: _p.on_substep("warmup", text, elapsed)) if _p else None
            )
            warmup_result = warmup_until_converged(
                lambda: engine.run_warmup_prompt(config, model, prompts[0]),
                config.measurement.warmup,
                on_substep=warmup_substep,
            )
            # The probe at line above ran one extra discarded inference that
            # warmup_until_converged() does not count; fold it into the warmup
            # total so warmup_excluded_samples reflects every discarded inference.
            warmup_result.iterations_completed += 1
        else:
            from llenergymeasure.domain.metrics import WarmupResult

            warmup_result = WarmupResult(
                converged=True,
                final_cv=0.0,
                iterations_completed=1 if config.measurement.warmup.enabled else 0,
                target_cv=config.measurement.warmup.cv_threshold,
                max_prompts=config.measurement.warmup.max_prompts,
            )

        if _p:
            iters = warmup_result.iterations_completed
            cv_info = f"  CV={warmup_result.final_cv:.1%}" if warmup_result.final_cv > 0 else ""
            converged = "converged" if warmup_result.converged else "not converged"
            iters_label = f"{iters} iteration{'s' if iters != 1 else ''}"
            # Kernel-only warmup (vLLM/TRT-LLM): no CV-based convergence
            if first_latency == 0.0 and config.measurement.warmup.enabled:
                _p.on_step_update("warmup", f"engine kernel warmup ({iters_label})")
            else:
                _p.on_step_update("warmup", f"{iters_label} ({converged}{cv_info})")
            _p.on_step_done("warmup", time.perf_counter() - t0_warmup)
        iters = warmup_result.iterations_completed
        _emit_substep(
            _p,
            "warmup",
            f"{iters} iteration{'s' if iters != 1 else ''}"
            + (f"  CV={warmup_result.final_cv:.1%}" if warmup_result.final_cv > 0 else ""),
        )

        # 6. Thermal floor - show step before sleeping
        floor_secs = (
            config.measurement.warmup.thermal_floor_seconds
            if config.measurement.warmup.enabled
            else 0
        )
        if _p:
            if floor_secs > 0:
                _p.on_step_start("thermal_floor", "Waiting", f"thermal floor ({floor_secs:.0f}s)")
            else:
                _p.on_step_skip("thermal_floor", "wait=0s")

        wait_s = thermal_floor_wait(config)
        warmup_result.thermal_floor_wait_s = wait_s

        if _p and wait_s > 0:
            _p.on_step_done("thermal_floor", wait_s)

        return warmup_result

    def _run_measured_inference(
        self,
        engine: EnginePlugin,
        config: ExperimentConfig,
        model: Any,
        prompts: list[str],
        gpu_indices: list[int] | None,
        progress: ProgressCallback | None,
    ) -> _MeasuredWindow:
        """Run the measured inference window: select + start energy tracking, sync
        CUDA, run inference under the thermal sampler, capture observed params, stop
        tracking, and estimate FLOPs.

        Returns the window outputs needed to assemble the ExperimentResult.
        """
        _p = progress

        # 7. Select energy sampler
        if _p:
            _p.on_step_start("energy_select", "Selecting", "energy sampler")
            t0_energy = time.perf_counter()
        energy_sampler = select_energy_sampler(
            config.measurement.energy_sampler, gpu_indices=gpu_indices
        )
        sampler_name = type(energy_sampler).__name__ if energy_sampler else "none"
        _emit_substep(_p, "energy_select", f"selected: {sampler_name}")
        if _p:
            _p.on_step_update("energy_select", f"energy sampler ({sampler_name})")
            _p.on_step_done("energy_select", time.perf_counter() - t0_energy)

        # 8. Start energy tracking (after warmup + thermal floor)
        energy_tracker = None
        if energy_sampler is not None:
            energy_tracker = energy_sampler.start_tracking()

        # 9. CUDA sync before inference (Zeus best practice)
        _cuda_sync()
        _emit_substep(_p, "measure", "CUDA sync (pre)")

        if _p:
            _p.on_step_start(
                "measure", "Measuring", f"inference ({config.task.dataset.n_prompts} prompts)"
            )
        _emit_substep(_p, "measure", "energy tracker started")

        t_inference_start = time.perf_counter()
        start_time = datetime.now()

        # Start thermal sampler around inference for timeseries + throttle detection
        with PowerThermalSampler(gpu_indices=gpu_indices) as thermal_sampler:
            output = engine.run_inference(config, model, prompts)

        t_inference_end = time.perf_counter()

        # 11. CUDA sync after inference, before stopping energy
        _cuda_sync()
        _emit_substep(_p, "measure", "CUDA sync (post)")

        # 11a. Capture observed params post-window (outside NVML sampling).
        # Must run here - model is still alive; capture overhead (~5-50 ms pure
        # Python) must not land inside the PowerThermalSampler context above.
        _capture_observed_params_into_output(engine, config, model, output)

        if _p:
            _p.on_step_done("measure", t_inference_end - t_inference_start)

        thermal_info = thermal_sampler.get_thermal_throttle_info()
        timeseries_samples = thermal_sampler.get_samples()

        # Harness sets canonical inference timer - overrides engine's elapsed_time_sec
        output.inference_time_sec = t_inference_end - t_inference_start

        # 12. Stop energy tracking
        energy_measurement = None
        if energy_sampler is not None and energy_tracker is not None:
            energy_measurement = energy_sampler.stop_tracking(energy_tracker)
            tracker_duration = energy_measurement.duration_sec if energy_measurement else 0.0
            _emit_substep(_p, "measure", f"energy tracker stopped  {tracker_duration:.1f}s")
        end_time = datetime.now()

        # 13. FLOPs estimation (warmup tokens excluded)
        if _p:
            _p.on_step_start("flops", "Estimating", "FLOPs (PaLM formula)")
            t0_flops = time.perf_counter()
        flops_result = self._estimate_flops(engine, config, output)
        if _p:
            if flops_result is not None:
                _p.on_step_update("flops", f"FLOPs: {flops_result.value:.2e}")
            _p.on_step_done("flops", time.perf_counter() - t0_flops)
        if flops_result is not None:
            _emit_substep(_p, "flops", f"FLOPs: {flops_result.value:.2e}")

        return _MeasuredWindow(
            output=output,
            thermal_info=thermal_info,
            timeseries_samples=timeseries_samples,
            energy_measurement=energy_measurement,
            flops_result=flops_result,
            start_time=start_time,
            end_time=end_time,
        )

    def _persist_and_assemble(
        self,
        *,
        engine: EnginePlugin,
        config: ExperimentConfig,
        snapshot: EnvironmentSnapshot | None,
        baseline: BaselineCache | None,
        warmup_result: WarmupResult,
        model_memory_mb: float,
        model_load_time_sec: float | None,
        prompt_count: int,
        gpu_indices: list[int] | None,
        window: _MeasuredWindow,
        output_dir: Path | str | None,
        save_timeseries: bool,
        progress: ProgressCallback | None,
    ) -> ExperimentResult:
        """Write the timeseries + config sidecars and assemble the ExperimentResult."""
        _p = progress

        # 14. Write timeseries Parquet sidecar (if output_dir set and save_timeseries enabled)
        resolved_output_dir = Path(output_dir) if output_dir is not None else None
        if _p:
            _p.on_step_start(
                "save",
                "Saving",
                "writing results",
            )
            t0_save = time.perf_counter()

        timeseries_path: str | None = None
        if save_timeseries and resolved_output_dir is not None and window.timeseries_samples:
            ts_file = write_timeseries_parquet(
                window.timeseries_samples,
                resolved_output_dir / TIMESERIES_FILENAME,
            )
            timeseries_path = ts_file.name  # relative name in result JSON
            _emit_substep(_p, "save", "timeseries parquet written")

        # 15. Collect measurement quality warnings
        duration_sec = (window.end_time - window.start_time).total_seconds()
        measurement_warnings = self._collect_warnings(
            duration_sec, window.timeseries_samples, gpu_indices, window.energy_measurement
        )

        # 16. Assemble ExperimentResult
        engine_version = getattr(engine, "version", None)
        result = self._build_result(
            engine_name=engine.name,
            engine_version=engine_version,
            config=config,
            output=window.output,
            model_memory_mb=model_memory_mb,
            snapshot=snapshot,
            start_time=window.start_time,
            end_time=window.end_time,
            duration_sec=duration_sec,
            thermal_info=window.thermal_info,
            energy_measurement=window.energy_measurement,
            baseline=baseline,
            flops_result=window.flops_result,
            timeseries_path=timeseries_path,
            timeseries_samples=window.timeseries_samples,
            measurement_warnings=measurement_warnings,
            warmup_result=warmup_result,
            prompt_count=prompt_count,
            model_load_time_sec=model_load_time_sec,
        )
        _emit_substep(_p, "save", "result assembled")

        # 17. Write config.json sidecar (observed-params + observed_config_hash)
        # Written to output_dir (temp dir, same as timeseries.parquet) so the
        # runner can move it to the per-experiment directory.
        if resolved_output_dir is not None:
            self._write_config_sidecar(
                output=window.output,
                config=config,
                result=result,
                output_dir=resolved_output_dir,
            )
            _emit_substep(_p, "save", "config sidecar written")

        if _p:
            _p.on_step_done("save", time.perf_counter() - t0_save)

        return result

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _write_config_sidecar(
        self,
        output: Any,
        config: Any,
        result: Any,
        output_dir: Path,
    ) -> None:
        """Write ``config.json`` sidecar to ``output_dir`` (temp staging area).

        Extracts observed params from ``output.extras`` (populated by each engine's
        ``_capture_effective_params`` after inference), computes ``observed_config_hash``
        from the observed-config hashing pipeline, and writes the sidecar atomically. The runner's
        ``_save_and_record`` moves this file to the per-experiment directory alongside
        ``result.json``.

        Best-effort - failures are logged at DEBUG to avoid masking measurement results.
        """
        try:
            from llenergymeasure.domain.hashing import build_observed_view, hash_config

            obs_engine = output.extras.get("observed_engine_params", {}) or {}
            obs_sampling = output.extras.get("observed_sampling_params", {}) or {}
            lib_ver = output.extras.get("library_version", "unknown") or "unknown"

            # Compute observed_config_hash from extracted native-type state.
            # harness + measurement come from the ran config so the observed hash
            # covers the same identity dimensions as the resolved hash.
            engine_name = result.engine
            task_dict = config.task.model_dump(mode="python")
            active_harness = config.active_harness()
            harness_dump = (
                active_harness.model_dump(mode="python") if active_harness is not None else {}
            )
            observed_view = build_observed_view(
                engine=engine_name,
                task=task_dict,
                observed_engine_params=obs_engine,
                observed_sampling_params=obs_sampling,
                harness=harness_dump,
                measurement=config.measurement.model_dump(mode="python"),
            )
            obs_hash = hash_config(observed_view)

            # Full user-declared config, recorded so the observed-collision
            # miner can attribute a shared observed_config_hash to the declared
            # fields that varied. Guarded separately: a declared-dump failure
            # must not cost us the observed hash written below.
            try:
                declared_config: dict[str, object] | None = config.model_dump(mode="json")
            except Exception:  # pragma: no cover - declared dump is best-effort
                declared_config = None

            save_config_sidecar(
                output_dir,
                experiment_id=result.experiment_id,
                config_hash=result.measurement_config_hash,
                engine=engine_name,
                library_version=lib_ver,
                observed_engine_params=obs_engine if obs_engine else None,
                observed_sampling_params=obs_sampling if obs_sampling else None,
                observed_config_hash=obs_hash,
                declared_config=declared_config,
            )
        except Exception as exc:
            logger.debug("Config sidecar write failed (non-fatal): %s", exc)

    def _capture_model_memory_mb(self, gpu_indices: list[int] | None = None) -> float:
        """Capture the post-load model-memory baseline in MB.

        In-process engines (Transformers): torch's per-process allocator sees the
        loaded weights, so ``torch.cuda.max_memory_allocated(device=idx)`` per
        rank (max across ``gpu_indices``, defaulting to ``[0]``) is authoritative
        and captures the tensor-parallel peak.

        Out-of-process engines (vLLM V1 EngineCore, TRT-LLM executor): the weights
        are loaded in a child process, so torch here reports a silent 0.0. Fall
        back to NVML device-used memory (whole-device; see
        :func:`get_nvml_device_memory_mb` for the tenancy caveat).

        Returns 0.0 only when NEITHER source is available (no torch/CUDA and no
        NVML, or a CPU run). The domain layer coerces a 0.0 baseline to null so
        the field is never a silently-wrong zero.
        """
        torch_mb = 0.0
        if importlib.util.find_spec("torch") is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    indices = gpu_indices if gpu_indices is not None else [0]
                    if indices:
                        peak = max(torch.cuda.max_memory_allocated(device=idx) for idx in indices)
                        torch_mb = bytes_to_mb(peak)
            except Exception:
                torch_mb = 0.0
        if torch_mb > 0:
            return torch_mb

        # torch saw nothing: out-of-process engine or no in-process CUDA activity.
        from llenergymeasure.engines._cuda import get_nvml_device_memory_mb

        nvml_mb = get_nvml_device_memory_mb(gpu_indices)
        return nvml_mb if nvml_mb is not None else 0.0

    def _estimate_flops(
        self,
        engine: EnginePlugin,
        config: ExperimentConfig,
        output: InferenceOutput,
    ) -> Any:
        """Estimate FLOPs from model and token counts.

        Fallback chain (highest confidence first):
        1. hf_model path - uses estimate_flops_palm(hf_model) when extras['hf_model']
           is set. Higher confidence: counts the actual loaded parameters.
        2. AutoConfig path - uses estimate_flops_palm_from_config(config.task.model).
           Works for engines that do not expose a model object (vLLM, TensorRT-LLM).
        3. None - FLOPs unavailable.
        """
        # Step 1: hf_model path (higher confidence - uses actual loaded parameters)
        model_obj = output.extras.get("hf_model")
        if model_obj is not None:
            try:
                return estimate_flops_palm(
                    model=model_obj,
                    n_input_tokens=output.input_tokens,
                    n_output_tokens=output.output_tokens,
                )
            except Exception as e:
                logger.debug("hf_model FLOPs estimation failed: %s", e)

        # Step 2: AutoConfig path (works without a loaded model object)
        try:
            result = estimate_flops_palm_from_config(
                model_name=config.task.model,
                n_input_tokens=output.input_tokens,
                n_output_tokens=output.output_tokens,
            )
            if result is not None:
                return result
        except Exception as e:
            logger.debug("AutoConfig FLOPs estimation failed: %s", e)

        # Step 3: FLOPs unavailable
        return None

    def _collect_warnings(
        self,
        duration_sec: float,
        timeseries_samples: list[PowerThermalSample],
        gpu_indices: list[int] | None = None,
        energy_measurement: Any = None,
    ) -> list[str]:
        """Collect measurement quality warnings from timeseries samples.

        ``energy_measurement`` is the authoritative energy backend result (or None).
        Its presence is a separate signal from ``timeseries_samples`` (which come from
        the thermal-telemetry sampler) - an absent energy measurement must be flagged
        even when thermal telemetry sampled fine.
        """
        temp_start: float | None = None
        temp_end: float | None = None
        if timeseries_samples:
            temps = [s.temperature_c for s in timeseries_samples if s.temperature_c is not None]
            if temps:
                temp_start = temps[0]
                temp_end = temps[-1]

        persistence_on = _check_persistence_mode(gpu_indices)
        nvml_count = len(timeseries_samples)

        return collect_measurement_warnings(
            duration_sec=duration_sec,
            gpu_persistence_mode=persistence_on,
            temp_start_c=temp_start,
            temp_end_c=temp_end,
            nvml_sample_count=nvml_count,
            energy_measurement_present=energy_measurement is not None,
        )

    @staticmethod
    def _configured_batch_size(engine_name: str, config: ExperimentConfig) -> int | None:
        """Return the configured static batch size for the engine, or None.

        transformers -> harness.transformers.batch_size, defaulting to 1 (mirrors
                        the engine plugin, which defaults batch_size to 1 when the
                        harness block or its batch_size is unset)
        tensorrt     -> tensorrt.engine_params.max_batch_size (its analogous field)
        anything else (incl. vllm continuous batching) -> None
        """
        if engine_name == "transformers":
            harness = config.active_harness()
            if harness is not None and harness.batch_size is not None:
                return int(harness.batch_size)
            return 1
        if engine_name == "tensorrt" and config.tensorrt is not None:
            engine_params = config.active_engine_params()
            return engine_params.max_batch_size if engine_params is not None else None
        return None

    @staticmethod
    def _resolve_measurement_mode(
        declared_mode: str | None,
        measurement_warnings: list[str],
    ) -> Any:
        """Map an engine-declared latency mode string to LatencyMeasurementMode.

        The engine sets ``output.latency_measurement_mode`` explicitly whenever it
        emits TTFT. If it is missing - or an unrecognised string - that is an engine
        bug: log a warning and fall back to the field's default (TRUE_STREAMING),
        noting it in measurement_warnings. A bad engine string must never crash
        result assembly.
        """
        from llenergymeasure.domain.metrics import LatencyMeasurementMode

        if declared_mode is None:
            logger.warning(
                "Engine emitted TTFT samples but no latency_measurement_mode; "
                "defaulting provenance to TRUE_STREAMING."
            )
            measurement_warnings.append(
                "latency_measurement_mode missing despite TTFT capture; "
                "provenance defaulted to true_streaming (engine should set it explicitly)."
            )
            return LatencyMeasurementMode.TRUE_STREAMING
        try:
            return LatencyMeasurementMode(declared_mode)
        except ValueError:
            logger.warning(
                "Engine emitted unrecognised latency_measurement_mode %r; "
                "defaulting provenance to TRUE_STREAMING.",
                declared_mode,
            )
            measurement_warnings.append(
                f"latency_measurement_mode {declared_mode!r} unrecognised; "
                "provenance defaulted to true_streaming."
            )
            return LatencyMeasurementMode.TRUE_STREAMING

    @staticmethod
    def _append_latency_profiling_warnings(
        config: ExperimentConfig,
        output: InferenceOutput,
        engine_name: str,
        measurement_warnings: list[str],
    ) -> None:
        """Append latency-profiling provenance warnings to measurement_warnings.

        Only fires when ``config.measurement.latency_profiling`` is enabled. Adds
        the fixed provenance note plus, when relevant, the batch-size-forcing note
        (transformers) and the unsupported-engine note (tensorrt).
        """
        if not config.measurement.latency_profiling:
            return
        measurement_warnings.append(
            "latency_profiling enabled: per-token timing capture (streamer/decode-average "
            "ITL) may perturb energy and latency; energy figures emitted as-is and are not "
            "directly comparable to non-profiled runs."
        )
        if output.extras.get("profiling_forced_batch_size"):
            measurement_warnings.append(
                "latency_profiling forced batch_size=1 for per-token timing capture; "
                "throughput is not comparable to the configured batch size."
            )
        if output.extras.get("latency_profiling_unsupported"):
            measurement_warnings.append(
                f"latency_profiling is not supported by the {engine_name} engine; "
                "no per-token timing was captured."
            )

    def _resolve_measurement_window(
        self,
        config: ExperimentConfig,
        output: InferenceOutput,
        energy_measurement: Any,
        timeseries_samples: list[PowerThermalSample] | None,
    ) -> Any:
        """Apply the configured measurement window, or None for total mode.

        Prefers the energy sampler's own power samples (NVML) for re-integration, and
        falls back to the harness PowerThermalSampler timeseries (always present even
        with Zeus/CodeCarbon, which expose no raw samples). Returns a WindowResult, or
        None when the window cannot be applied (keeping the unchanged total figures).
        """
        from llenergymeasure.harness.windowing import apply_measurement_window

        if config.measurement.measurement_methodology == "total":
            return None

        sampler_samples = getattr(energy_measurement, "samples", None) or []
        power_samples = sampler_samples if len(sampler_samples) >= 2 else (timeseries_samples or [])
        return apply_measurement_window(
            power_samples, config.measurement, output.inference_time_sec
        )

    def _build_result(
        self,
        engine_name: str,
        engine_version: str | None,
        config: ExperimentConfig,
        output: InferenceOutput,
        model_memory_mb: float,
        snapshot: Any,
        start_time: datetime,
        end_time: datetime,
        duration_sec: float,
        thermal_info: Any,
        energy_measurement: Any,
        baseline: Any,
        flops_result: Any,
        timeseries_path: str | None,
        measurement_warnings: list[str],
        warmup_result: Any = None,
        timeseries_samples: list[PowerThermalSample] | None = None,
        prompt_count: int = 0,
        model_load_time_sec: float | None = None,
    ) -> ExperimentResult:
        """Assemble ExperimentResult from measurement data.

        Energy/FLOPs fields carry measured values. The one exception: when no energy
        sampler produced a measurement, total_energy_j keeps a 0.0 placeholder that is
        made loud via a warning log and the energy_measurement_unavailable measurement
        warning (absence, not a measured zero). Energy breakdown uses baseline
        adjustment when available.

        Args:
            engine_name: Engine identifier string (from engine.name).
            engine_version: Engine version string (from engine.version), or None.
            config: Experiment configuration.
            output: InferenceOutput from engine.run_inference().
            model_memory_mb: GPU memory after model load, before inference (MB).
            snapshot: EnvironmentSnapshot captured before model load.
            start_time: Measurement start time.
            end_time: Measurement end time.
            duration_sec: Pre-computed (end_time - start_time).total_seconds().
            thermal_info: ThermalThrottleInfo from PowerThermalSampler.
            energy_measurement: EnergyMeasurement from energy sampler, or None.
            baseline: BaselineCache from baseline measurement, or None.
            flops_result: FlopsResult from estimate_flops_palm(), or None.
            timeseries_path: Relative path to Parquet sidecar, or None.
            measurement_warnings: List of quality warning strings.
            warmup_result: WarmupResult from warmup phase, or None.
            timeseries_samples: Raw PowerThermalSample list for GPU-utilisation /
                memory-bandwidth / total-VRAM extraction. None = no samples.
            prompt_count: Number of prompts in the run (for batch effective size).
            model_load_time_sec: Wall-clock seconds spent in engine.load_model()
                (model load + any engine build/compile). Non-energy metadata;
                the phase precedes the NVML measurement window.

        Returns:
            Fully assembled ExperimentResult.
        """
        from llenergymeasure.domain.extended_metrics import compute_extended_metrics
        from llenergymeasure.domain.metrics import (
            MultiGPUMetrics,
            compute_latency_statistics,
        )
        from llenergymeasure.harness.baseline import create_energy_breakdown

        experiment_id = f"{config.task.model}_{start_time.strftime('%Y%m%d_%H%M%S')}"

        # Resolve the measurement window (None for total mode = unchanged path).
        window_result = self._resolve_measurement_window(
            config, output, energy_measurement, timeseries_samples
        )

        # Reported inference time: window duration for windowed/steady_state, else full run.
        measured_time_sec = (
            window_result.window_duration_sec
            if window_result is not None
            else output.inference_time_sec
        )

        # Real energy values from energy sampler (windowed re-integration overrides
        # the sampler total when a window is in effect).
        if window_result is not None:
            total_energy_j = window_result.energy_j
        elif energy_measurement is not None:
            total_energy_j = energy_measurement.total_j
        else:
            # No authoritative energy measurement (sampler unavailable or disabled). The
            # schema requires a non-null total_energy_j, so we keep a 0.0 placeholder -
            # but this is absence, NOT a measured zero. It is made loud here and, in the
            # persisted result, via the energy_measurement_unavailable measurement warning
            # (see collect_measurement_warnings); it can never be silent.
            logger.warning(
                "No energy measurement available for %s; reporting total_energy_j=0.0 as a "
                "placeholder for absence, not a measured zero (see the "
                "'energy_measurement_unavailable' measurement warning).",
                experiment_id,
            )
            total_energy_j = 0.0
        # duration_sec is passed in from run() - computed once, not recalculated here

        # Token counts reported describe the full workload; for a sub-window, energy and
        # throughput are normalised by the window-attributed token share (proportional by
        # time - the harness has no absolute per-token timestamps).
        output_tokens = output.output_tokens if output.output_tokens > 0 else output.total_tokens
        token_fraction = window_result.token_fraction if window_result is not None else 1.0
        windowed_output_tokens = output_tokens * token_fraction
        windowed_total_tokens = output.total_tokens * token_fraction

        avg_tokens_per_second = (
            windowed_total_tokens / measured_time_sec if measured_time_sec > 0 else 0.0
        )

        # Energy per token: output tokens only (input tokens are not "generated")
        avg_energy_per_token_j = (
            total_energy_j / windowed_output_tokens
            if (total_energy_j > 0 and windowed_output_tokens > 0)
            else 0.0
        )

        # Energy breakdown with baseline adjustment.
        # Use energy sampler's window duration for baseline adjustment,
        # not harness datetime duration, to avoid CUDA sync latency skew. For a
        # sub-window, the realised window duration is the correct baseline span.
        energy_duration = (
            measured_time_sec
            if window_result is not None
            else (
                energy_measurement.duration_sec if energy_measurement is not None else duration_sec
            )
        )
        energy_breakdown = create_energy_breakdown(total_energy_j, baseline, energy_duration)

        # Per-GPU energy breakdown. A window re-integrates per-GPU energy; otherwise the
        # sampler's per-GPU totals are used.
        per_gpu_source = (
            window_result.per_gpu_j
            if window_result is not None
            else (energy_measurement.per_gpu_j if energy_measurement is not None else None)
        )
        energy_per_device_j = None
        multi_gpu = None
        if per_gpu_source:
            sorted_indices = sorted(per_gpu_source.keys())
            energy_per_device_j = [per_gpu_source[i] for i in sorted_indices]
            if len(sorted_indices) > 1:
                multi_gpu = MultiGPUMetrics(
                    num_gpus=len(sorted_indices),
                    energy_per_gpu_j=energy_per_device_j,
                    energy_total_j=total_energy_j,
                    energy_per_output_token_j=(
                        total_energy_j / windowed_output_tokens
                        if windowed_output_tokens > 0
                        else 0.0
                    ),
                )

        # mJ/tok derived fields (energy in millijoules per OUTPUT token, matching
        # avg_energy_per_token_j; input tokens are prefilled, not "generated").
        _mj_total = mj_per_token(total_energy_j, windowed_output_tokens)
        energy_adjusted_j = energy_breakdown.adjusted_j if energy_breakdown else None
        _mj_adjusted = (
            mj_per_token(energy_adjusted_j, windowed_output_tokens)
            if energy_adjusted_j is not None
            else None
        )

        # FLOPs from PaLM formula (0.0 if estimation unavailable)
        total_flops = flops_result.value if flops_result is not None else 0.0

        # FLOPs derived fields
        flops_per_output_token = (
            total_flops / output.output_tokens
            if (total_flops > 0 and output.output_tokens > 0)
            else None
        )
        flops_per_input_token = (
            total_flops / output.input_tokens
            if (total_flops > 0 and output.input_tokens > 0)
            else None
        )
        flops_per_second = (
            total_flops / output.inference_time_sec
            if (total_flops > 0 and output.inference_time_sec > 0)
            else None
        )

        # Memory metrics: inference-window-only peak and derived delta. Both peak
        # and model baseline are 0.0 when neither torch nor NVML could measure
        # them (out-of-process engine with NVML unavailable, or a CPU run); the
        # delta is only meaningful when both are real, otherwise it stays null
        # rather than reporting a silently-wrong number.
        inference_memory_mb: float | None
        if output.peak_memory_mb > 0 and model_memory_mb > 0:
            inference_memory_mb = max(0.0, output.peak_memory_mb - model_memory_mb)
        else:
            inference_memory_mb = None
        logger.debug(
            "Memory: model=%.1fMB, peak_inference=%.1fMB, inference_delta=%s",
            model_memory_mb,
            output.peak_memory_mb,
            f"{inference_memory_mb:.1f}MB" if inference_memory_mb is not None else "null",
        )

        # --- Extended efficiency metrics ---
        samples = timeseries_samples or []
        gpu_utilisation_samples = [
            s.sm_utilisation for s in samples if s.sm_utilisation is not None
        ]
        memory_bandwidth_samples = [
            s.memory_bandwidth_utilisation
            for s in samples
            if s.memory_bandwidth_utilisation is not None
        ]
        total_vram_mb = max(
            (s.memory_total_mb for s in samples if s.memory_total_mb is not None),
            default=0.0,
        )

        kv_cache_stats = output.kv_cache_stats
        kv_cache_mb = kv_cache_stats.get("kv_cache_mb") if kv_cache_stats else None
        memory_stats: dict[str, float] = {
            "peak_mb": output.peak_memory_mb,
            "model_mb": model_memory_mb,
            "total_vram_mb": total_vram_mb,
        }
        if kv_cache_mb is not None:
            memory_stats["kv_cache_mb"] = kv_cache_mb

        # Batch stats: continuous-batching engines (e.g. vLLM) report num_batches
        # as None, so the truthiness guard skips them. Static-batching engines
        # report num_batches + padding; effective batch size derives from prompt count.
        batch_stats: dict[str, Any] | None = None
        if output.num_batches:
            configured_batch_size = self._configured_batch_size(engine_name, config)
            effective_batch_size: float | None = None
            if output.num_batches > 0:
                effective_batch_size = prompt_count / output.num_batches
            padding_overhead: float | None = None
            if output.padding_tokens is not None and output.input_tokens > 0:
                total_positions = output.input_tokens + output.padding_tokens
                if total_positions > 0:
                    padding_overhead = output.padding_tokens / total_positions
            batch_stats = {
                "num_batches": output.num_batches,
                "effective_batch_size": effective_batch_size,
                "configured_batch_size": configured_batch_size,
                "padding_overhead": padding_overhead,
            }

        # Latency stats from streaming TTFT/ITL. Computed BEFORE extended metrics
        # so the ITL mean can feed tpot_ms. measurement_mode is mapped from the
        # engine-declared capture mode (provenance). vLLM populates TTFT-only stats
        # even without profiling; ITL (and thus tpot_ms) needs profiling.
        latency_stats = None
        if output.ttft_ms:
            measurement_mode = self._resolve_measurement_mode(
                output.latency_measurement_mode, measurement_warnings
            )
            latency_stats = compute_latency_statistics(
                output.ttft_ms,
                itl_trimmed_ms=output.itl_ms or None,
                measurement_mode=measurement_mode,
            )

        itl_mean_ms = latency_stats.itl_mean_ms if latency_stats is not None else None

        extended_metrics = compute_extended_metrics(
            output_tokens=output.output_tokens,
            total_energy_j=total_energy_j,
            tokens_per_second=avg_tokens_per_second,
            precision_factor=1.0,  # No precision-scaling applied (default)
            itl_mean_ms=itl_mean_ms,  # populates tpot_ms when ITL was captured
            per_request_latencies_ms=output.per_request_latencies_ms or None,
            gpu_utilisation_samples=gpu_utilisation_samples or None,
            memory_bandwidth_samples=memory_bandwidth_samples or None,
            memory_stats=memory_stats,
            batch_stats=batch_stats,
            kv_cache_stats=kv_cache_stats,
        )
        # Preserve inference-only memory delta (compute_extended_metrics does not
        # know the model baseline split).
        extended_metrics.memory.inference_memory_mb = inference_memory_mb

        # Latency profiling provenance warnings (appended to measurement_warnings).
        self._append_latency_profiling_warnings(config, output, engine_name, measurement_warnings)

        # Measurement-methodology provenance. For total mode the window spans the whole
        # run (unchanged); for windowed/steady_state the realised window is recorded.
        if window_result is not None:
            measurement_methodology = window_result.methodology
            steady_state_window = window_result.window
            steady_state_not_detected = window_result.steady_state_not_detected
            measurement_warnings.extend(window_result.warnings)
            discard_fraction = (
                window_result.window[0] / output.inference_time_sec
                if (window_result.methodology == "steady_state" and output.inference_time_sec > 0)
                else None
            )
        else:
            measurement_methodology = "total"
            steady_state_window = (0.0, output.inference_time_sec)
            steady_state_not_detected = False
            discard_fraction = None

        warmup_excluded_samples = (
            warmup_result.iterations_completed if warmup_result is not None else None
        )

        return ExperimentResult(
            experiment_id=experiment_id,
            measurement_config_hash=compute_declared_config_hash(config),
            llenergymeasure_version=__version__,
            measurement_methodology=measurement_methodology,
            engine=engine_name,
            engine_version=engine_version,
            aggregation=AggregationMetadata(
                method="single_process",
                num_processes=1,
            ),
            total_tokens=output.total_tokens,
            total_energy_j=total_energy_j,
            total_inference_time_sec=measured_time_sec,
            avg_tokens_per_second=avg_tokens_per_second,
            avg_energy_per_token_j=avg_energy_per_token_j,
            total_flops=total_flops,
            flops_per_output_token=flops_per_output_token,
            flops_per_input_token=flops_per_input_token,
            flops_per_second=flops_per_second,
            start_time=start_time,
            end_time=end_time,
            thermal_throttle=thermal_info,
            energy_breakdown=energy_breakdown,
            model_name=config.task.model,
            timeseries=timeseries_path,
            mj_per_tok_total=_mj_total,
            mj_per_tok_adjusted=_mj_adjusted,
            baseline_power_w=energy_breakdown.baseline_power_w if energy_breakdown else None,
            energy_adjusted_j=energy_adjusted_j,
            energy_per_device_j=energy_per_device_j,
            multi_gpu=multi_gpu,
            warmup_result=warmup_result,
            measurement_warnings=measurement_warnings,
            extended_metrics=extended_metrics,
            latency_stats=latency_stats,
            steady_state_window=steady_state_window,
            measurement_window_discard_fraction=discard_fraction,
            steady_state_not_detected=steady_state_not_detected,
            warmup_excluded_samples=warmup_excluded_samples,
            model_load_time_sec=model_load_time_sec,
            engine_build_cache_hit=output.extras.get("engine_build_cache_hit"),
        )
