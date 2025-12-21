"""Experiment orchestrator for LLM benchmarking.

This module provides the main experiment runner that coordinates
model loading, inference, and metrics collection. It saves raw
per-process results and does NOT aggregate - aggregation is a
separate step (see results/aggregation.py).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from llm_energy_measure.domain.experiment import RawProcessResult, Timestamps
from llm_energy_measure.orchestration.context import ExperimentContext

if TYPE_CHECKING:
    from llm_energy_measure.protocols import (
        EnergyBackend,
        InferenceEngine,
        MetricsCollector,
        ModelLoader,
        ResultsRepository,
    )


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
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        inference_engine: InferenceEngine,
        metrics_collector: MetricsCollector,
        energy_backend: EnergyBackend,
        repository: ResultsRepository,
    ) -> None:
        self._loader = model_loader
        self._inference = inference_engine
        self._metrics = metrics_collector
        self._energy = energy_backend
        self._repository = repository

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
        energy_metrics = self._energy.stop_tracking(tracker)
        logger.debug("Energy tracking stopped")

        # Collect metrics
        combined = self._metrics.collect(model, inference_result, ctx.config)
        end = datetime.now()

        # Build raw result
        raw_result = RawProcessResult(
            experiment_id=ctx.experiment_id,
            process_index=ctx.process_index,
            gpu_id=ctx.device.index or 0,
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
        )

        # Save raw result
        result_path = self._repository.save_raw(ctx.experiment_id, raw_result)
        logger.info(f"Raw result saved to {result_path}")

        return result_path
