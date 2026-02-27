"""Experiment orchestrator for LLM benchmarking.

This module provides the main experiment runner that coordinates
model loading, inference, and metrics collection. It saves raw
per-process results and does NOT aggregate - aggregation is a
separate step (see results/aggregation.py).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from llenergymeasure.constants import COMPLETION_MARKER_PREFIX
from llenergymeasure.domain.experiment import RawProcessResult, Timestamps
from llenergymeasure.domain.metrics import (
    CombinedMetrics,
    ComputeMetrics,
    EnergyMetrics,
    ExtendedEfficiencyMetrics,
)
from llenergymeasure.orchestration.context import ExperimentContext

if TYPE_CHECKING:
    from llenergymeasure.protocols import (
        EnergyBackend,
        InferenceEngine,
        MetricsCollector,
        ModelLoader,
        ResultsRepository,
    )


def _compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file for integrity verification."""
    if not file_path.exists():
        return "sha256:file_not_found"
    hash_obj = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_obj.update(chunk)
    return f"sha256:{hash_obj.hexdigest()[:16]}"


def _write_completion_marker(
    experiment_id: str,
    process_index: int,
    gpu_id: int,
    result_path: Path,
) -> Path:
    """Write process completion marker after successful result save.

    Args:
        experiment_id: Experiment identifier.
        process_index: Process index (0-based).
        gpu_id: GPU device index.
        result_path: Path to the saved result file.

    Returns:
        Path to the completion marker file.
    """
    marker_dir = result_path.parent
    marker_path = marker_dir / f"{COMPLETION_MARKER_PREFIX}{process_index}"

    marker_data = {
        "process_index": process_index,
        "gpu_id": gpu_id,
        "result_path": str(result_path),
        "timestamp": datetime.now().isoformat(),
        "result_hash": _compute_file_hash(result_path),
    }

    # Atomic write
    temp_path = marker_path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(marker_data, indent=2))
    temp_path.rename(marker_path)

    logger.debug(f"Completion marker written: {marker_path}")
    return marker_path


class ExperimentOrchestrator:
    """Orchestrates LLM benchmark experiments.

    Coordinates the experiment lifecycle using dependency-injected components.
    Saves raw per-process results only - aggregation happens separately.

    Attributes:
        model_loader: Component for loading models and tokenizers.
        inference_engine: Component for running inference.
        metrics_collector: Component for collecting metrics.
        energy_backend: Component for energy measurement.
        repository: Component for persisting results.
        backend_name: Name of the inference backend (for result metadata).
        backend_version: Version of the inference backend (for reproducibility).
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        inference_engine: InferenceEngine,
        metrics_collector: MetricsCollector,
        energy_backend: EnergyBackend,
        repository: ResultsRepository,
        backend_name: str = "pytorch",
        backend_version: str | None = None,
    ) -> None:
        self._loader = model_loader
        self._inference = inference_engine
        self._metrics = metrics_collector
        self._energy = energy_backend
        self._repository = repository
        self._backend_name = backend_name
        self._backend_version = backend_version

    def run(self, ctx: ExperimentContext, prompts: list[str]) -> Path:
        """Run experiment and save raw results.

        Lifecycle: environment -> baseline -> model load -> warmup ->
        sampler+inference -> energy breakdown -> thermal check ->
        save results -> export timeseries.

        Args:
            ctx: Experiment context with config and accelerator.
            prompts: List of prompts to process.

        Returns:
            Path to the saved raw result file.
        """
        start = datetime.now()
        logger.info(f"Starting experiment {ctx.experiment_id} on process {ctx.process_index}")

        # --- Phase 1: Collect environment metadata (MEAS-01) ---
        environment = None
        try:
            from llenergymeasure.core.environment import collect_environment_metadata

            environment = collect_environment_metadata(ctx.device.index or 0)
            logger.info(f"Environment: {environment.summary_line}")
        except Exception as e:
            logger.warning(f"Environment metadata collection failed (non-fatal): {e}")

        # --- Phase 1: Baseline power measurement (MEAS-02) ---
        baseline = None
        try:
            from llenergymeasure.core.baseline import measure_baseline_power

            if ctx.config.baseline.enabled:
                baseline = measure_baseline_power(
                    device_index=ctx.device.index or 0,
                    duration_sec=ctx.config.baseline.duration_sec,
                    sample_interval_ms=ctx.config.baseline.sample_interval_ms,
                    cache_ttl_sec=ctx.config.baseline.cache_ttl_sec,
                )
                if baseline:
                    logger.info(f"Baseline power: {baseline.power_w:.1f}W")
                elif ctx.config.baseline.required:
                    raise RuntimeError("Baseline measurement required but failed")
                else:
                    logger.warning("Baseline measurement failed, continuing with raw energy only")
        except RuntimeError:
            raise
        except Exception as e:
            if ctx.config.baseline.required:
                raise RuntimeError(f"Baseline measurement required but failed: {e}") from e
            logger.warning(f"Baseline measurement failed (non-fatal): {e}")

        # Load model
        model, tokenizer = self._loader.load(ctx.config)
        logger.info(f"Model loaded: {ctx.config.model_name}")

        # --- Phase 1: Warmup convergence (MEAS-05) ---
        warmup_result = None
        try:
            if ctx.config.warmup.enabled and model is not None and tokenizer is not None:
                from llenergymeasure.core.warmup import (
                    create_warmup_inference_fn,
                    warmup_until_converged,
                )

                warmup_fn = create_warmup_inference_fn(
                    model,
                    tokenizer,
                    prompts[0] if prompts else "Hello",
                    max_new_tokens=min(32, ctx.config.max_output_tokens),
                )
                warmup_result = warmup_until_converged(
                    warmup_fn, ctx.config.warmup, show_progress=True
                )
                if warmup_result.converged:
                    logger.info(
                        f"Warmup converged: {warmup_result.iterations_completed} prompts, "
                        f"CV={warmup_result.final_cv:.3f}"
                    )
                else:
                    logger.warning(
                        f"Warmup did not converge after {warmup_result.iterations_completed} "
                        f"prompts (CV={warmup_result.final_cv:.3f})"
                    )
            elif ctx.config.warmup.enabled:
                logger.debug(
                    "Warmup convergence skipped: backend manages model internally "
                    "(using backend's own warmup)"
                )
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

        # --- Phase 1: PowerThermalSampler (MEAS-03, MEAS-04) ---
        power_sampler = None
        try:
            from llenergymeasure.core.power_thermal import PowerThermalSampler

            power_sampler = PowerThermalSampler(
                device_index=ctx.device.index or 0,
                sample_interval_ms=(
                    ctx.config.timeseries.sample_interval_ms
                    if ctx.config.timeseries.enabled
                    else 100
                ),
            )
        except Exception as e:
            logger.warning(f"PowerThermalSampler creation failed (non-fatal): {e}")

        # Start energy tracking
        tracker = self._energy.start_tracking()
        logger.debug("Energy tracking started")

        # Run inference (with PowerThermalSampler if available)
        if power_sampler is not None:
            try:
                with power_sampler:
                    inference_result = self._inference.run(model, tokenizer, prompts, ctx.config)
            except Exception:
                # If sampler context fails, run inference without it
                logger.warning("PowerThermalSampler context failed, running without it")
                power_sampler = None
                inference_result = self._inference.run(model, tokenizer, prompts, ctx.config)
        else:
            inference_result = self._inference.run(model, tokenizer, prompts, ctx.config)
        logger.info(f"Inference complete: {len(prompts)} prompts processed")

        # Stop energy tracking
        # Note: Energy tracking can fail with vLLM due to CUDA context issues
        # We continue with placeholder metrics if this happens
        energy_tracking_failed = False
        try:
            energy_metrics = self._energy.stop_tracking(tracker)
            logger.debug("Energy tracking stopped")
        except Exception as e:
            logger.warning(f"Energy tracking failed (non-fatal): {e}")
            energy_metrics = EnergyMetrics.placeholder(
                duration_sec=inference_result.metrics.inference_time_sec
            )
            energy_tracking_failed = True

        # Collect metrics
        # Note: Can fail with vLLM due to CUDA context issues
        try:
            combined = self._metrics.collect(model, inference_result, ctx.config)
        except Exception as e:
            logger.warning(f"Metrics collection failed (non-fatal): {e}")
            combined = CombinedMetrics(
                inference=inference_result.metrics,
                energy=energy_metrics,
                compute=ComputeMetrics(
                    flops_total=0.0,
                    flops_per_token=0.0,
                    flops_per_second=0.0,
                    peak_memory_mb=0.0,
                    model_memory_mb=0.0,
                    flops_method="unavailable",
                    flops_confidence="low",
                    compute_precision=ctx.config.fp_precision,
                ),
            )
        end = datetime.now()

        # Get GPU/MIG info for result metadata
        # Note: Can fail with vLLM due to CUDA context issues
        try:
            from llenergymeasure.core.gpu_info import get_device_mig_info

            gpu_id = ctx.device.index or 0
            mig_info = get_device_mig_info(gpu_id)
        except Exception as e:
            logger.warning(f"GPU info collection failed (non-fatal): {e}")
            gpu_id = 0
            mig_info = {"gpu_name": "unknown", "gpu_is_mig": False, "gpu_mig_profile": None}

        # Generate energy measurement warning if on MIG
        energy_warning = None
        if mig_info.get("gpu_is_mig", False):
            profile = mig_info.get("gpu_mig_profile") or "unknown"
            energy_warning = (
                f"Running on MIG instance ({profile}). Energy measurement reflects "
                "parent GPU total, not per-instance consumption."
            )
            logger.warning(energy_warning)

        # Compute extended efficiency metrics
        from llenergymeasure.core.extended_metrics import compute_extended_metrics

        # Get precision factor from backend result if available
        precision_factor = 1.0
        backend_result = getattr(inference_result, "backend_result", None)
        if backend_result and backend_result.precision_metadata:
            precision_factor = backend_result.precision_metadata.precision_factor

        # Get ITL mean from latency measurements (for TPOT)
        itl_mean_ms = None
        latency_measurements = getattr(combined.inference, "latency_measurements", None)
        if latency_measurements and latency_measurements.itl_trimmed_ms:
            import numpy as np

            itl_mean_ms = float(np.mean(latency_measurements.itl_trimmed_ms))

        # Extract raw data from backend result for late aggregation
        per_request_latencies = []
        gpu_utilisation_samples = []
        memory_stats = None
        batch_stats = None
        kv_cache_stats = None

        if backend_result:
            per_request_latencies = backend_result.per_request_latencies_ms
            gpu_utilisation_samples = backend_result.gpu_utilisation_samples
            memory_stats = backend_result.memory_stats or None
            batch_stats = backend_result.batch_stats or None
            kv_cache_stats = backend_result.kv_cache_stats or None

        # Add memory stats from compute metrics if not from backend
        if not memory_stats:
            memory_stats = {
                "peak_mb": combined.compute.peak_memory_mb,
                "model_mb": combined.compute.model_memory_mb,
            }
            # Get total VRAM from GPU info
            try:
                import torch

                if torch.cuda.is_available():
                    device_props = torch.cuda.get_device_properties(ctx.device)
                    memory_stats["total_vram_mb"] = device_props.total_memory / (1024 * 1024)
            except Exception:
                pass

        try:
            extended_metrics = compute_extended_metrics(
                output_tokens=combined.inference.output_tokens,
                total_energy_j=energy_metrics.total_energy_j,
                tokens_per_second=combined.inference.tokens_per_second,
                precision_factor=precision_factor,
                itl_mean_ms=itl_mean_ms,
                per_request_latencies_ms=per_request_latencies or None,
                gpu_utilisation_samples=gpu_utilisation_samples or None,
                memory_stats=memory_stats,
                batch_stats=batch_stats,
                kv_cache_stats=kv_cache_stats,
            )
        except Exception as e:
            logger.warning(f"Extended metrics computation failed (non-fatal): {e}")
            extended_metrics = ExtendedEfficiencyMetrics()

        # --- Phase 1: Energy breakdown (MEAS-02) ---
        energy_breakdown = None
        try:
            from llenergymeasure.core.baseline import create_energy_breakdown

            # Use actual experiment duration from timestamps, not energy_metrics.duration_sec
            # which may be 0.0 from CodeCarbon's reporting
            experiment_duration_sec = (end - start).total_seconds()
            energy_breakdown = create_energy_breakdown(
                total_energy_j=energy_metrics.total_energy_j,
                baseline=baseline,
                duration_sec=experiment_duration_sec,
            )
        except Exception as e:
            logger.warning(f"Energy breakdown creation failed (non-fatal): {e}")

        # --- Phase 1: Thermal throttle info (MEAS-03) ---
        thermal_throttle = None
        try:
            if power_sampler is not None:
                thermal_throttle = power_sampler.get_thermal_throttle_info()
                if thermal_throttle.detected:
                    logger.warning(
                        f"Thermal throttling detected during experiment "
                        f"({thermal_throttle.throttle_duration_sec:.1f}s)"
                    )
        except Exception as e:
            logger.warning(f"Thermal throttle info failed (non-fatal): {e}")

        # Build raw result with effective_config, cli_overrides, and provenance
        raw_result = RawProcessResult(
            experiment_id=ctx.experiment_id,
            backend=self._backend_name,
            backend_version=self._backend_version,
            process_index=ctx.process_index,
            gpu_id=gpu_id,
            gpu_name=mig_info.get("gpu_name", ""),
            gpu_is_mig=mig_info.get("gpu_is_mig", False),
            gpu_mig_profile=mig_info.get("gpu_mig_profile"),
            energy_measurement_warning=energy_warning,
            energy_tracking_failed=energy_tracking_failed,
            config_name=ctx.config.config_name,
            model_name=ctx.config.model_name,
            timestamps=Timestamps(
                start=start,
                end=end,
                duration_sec=(end - start).total_seconds(),
            ),
            inference_metrics=combined.inference,
            energy_metrics=energy_metrics,
            compute_metrics=combined.compute,
            effective_config=ctx.effective_config,
            cli_overrides=ctx.cli_overrides,
            config_warnings=ctx.config_warnings,
            parameter_provenance=ctx.parameter_provenance,
            preset_chain=ctx.preset_chain,
            extended_metrics=extended_metrics,
            per_request_latencies_ms=per_request_latencies,
            gpu_utilisation_samples=gpu_utilisation_samples,
            environment=environment,
            energy_breakdown=energy_breakdown,
            thermal_throttle=thermal_throttle,
            warmup_result=warmup_result,
        )

        # Save raw result
        result_path = self._repository.save_raw(ctx.experiment_id, raw_result)
        logger.info(f"Raw result saved to {result_path}")

        # --- Phase 1: Time-series export (MEAS-04) ---
        timeseries_path_str = None
        if power_sampler is not None and power_sampler.is_available and ctx.config.timeseries.save:
            try:
                from llenergymeasure.results.timeseries import export_timeseries

                ts_path = export_timeseries(
                    samples=power_sampler.get_samples(),
                    experiment_id=ctx.experiment_id,
                    process_index=ctx.process_index,
                    output_dir=result_path.parent,
                    sample_interval_ms=ctx.config.timeseries.sample_interval_ms,
                )
                timeseries_path_str = str(ts_path)
                logger.info(f"Time-series saved: {ts_path}")
            except Exception as e:
                logger.warning(f"Time-series export failed (non-fatal): {e}")

        # Update result with timeseries path if exported
        # Since RawProcessResult is frozen, we need to reconstruct if timeseries was saved
        if timeseries_path_str is not None:
            raw_result = raw_result.model_copy(update={"timeseries_path": timeseries_path_str})
            # Re-save with timeseries path
            result_path = self._repository.save_raw(ctx.experiment_id, raw_result)
            logger.debug("Result re-saved with timeseries path")

        # Write completion marker (Phase 5)
        _write_completion_marker(
            ctx.experiment_id,
            ctx.process_index,
            ctx.device.index or 0,
            result_path,
        )

        return result_path
