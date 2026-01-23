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

        Args:
            ctx: Experiment context with config and accelerator.
            prompts: List of prompts to process.

        Returns:
            Path to the saved raw result file.
        """
        start = datetime.now()
        logger.info(f"Starting experiment {ctx.experiment_id} on process {ctx.process_index}")

        # Load model
        model, tokenizer = self._loader.load(ctx.config)
        logger.info(f"Model loaded: {ctx.config.model_name}")

        # Start energy tracking
        tracker = self._energy.start_tracking()
        logger.debug("Energy tracking started")

        # Run inference
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
            from llenergymeasure.domain.metrics import EnergyMetrics

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
            from llenergymeasure.domain.metrics import CombinedMetrics, ComputeMetrics

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
        )

        # Save raw result
        result_path = self._repository.save_raw(ctx.experiment_id, raw_result)
        logger.info(f"Raw result saved to {result_path}")

        # Write completion marker (Phase 5)
        _write_completion_marker(
            ctx.experiment_id,
            ctx.process_index,
            ctx.device.index or 0,
            result_path,
        )

        return result_path
