"""MeasurementHarness - owns the measurement lifecycle for any EnginePlugin.

The harness extracts the ~600 lines of identical measurement infrastructure
duplicated across transformers.py and vllm.py into a single location. Engines
become thin plugins implementing the 4-method EnginePlugin protocol.

This module is the facade: :class:`MeasurementHarness` sequences three phase
modules, each a coherent single concern -

- :mod:`llenergymeasure.harness.lifecycle` - per-model-load phases (baseline,
  model + snapshot load, prompts, warmup), producing the ``_EngineLifetime``;
- :mod:`llenergymeasure.harness.window` - the measured-window orchestration over
  :mod:`llenergymeasure.harness.bracket`, producing the ``_MeasuredWindow``;
- :mod:`llenergymeasure.harness.staging` + :mod:`llenergymeasure.harness.result_assembly`
  - the source-branched result assembly and the temp-dir sidecar staging.

``llenergymeasure.harness`` re-exports the public surface plus the module-level
helpers tests patch; those helpers now live in the phase modules and are patched
at their use sites (see each module's docstring).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from llenergymeasure.harness.lifecycle import _EngineLifetime, build_lifetime
from llenergymeasure.harness.result_assembly import SourceMetrics, _ConfigMethodology
from llenergymeasure.harness.staging import persist_and_assemble
from llenergymeasure.harness.window import _MeasuredWindow, run_measured_inference

if TYPE_CHECKING:
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.environment import EnvironmentSnapshot
    from llenergymeasure.domain.experiment import ExperimentResult
    from llenergymeasure.domain.progress import ProgressCallback
    from llenergymeasure.engines.protocol import EnginePlugin
    from llenergymeasure.harness.baseline import BaselineCache

logger = logging.getLogger(__name__)

# Re-exported so ``from llenergymeasure.harness.measurement import _ConfigMethodology``
# (and the carrier dataclasses) keeps resolving after the decomposition.
__all__ = [
    "MeasurementHarness",
    "SourceMetrics",
    "_ConfigMethodology",
    "_EngineLifetime",
    "_MeasuredWindow",
]


class MeasurementHarness:
    """Orchestrates the measurement lifecycle for any EnginePlugin.

    Engines are thin plugins implementing EnginePlugin (load_model, warmup,
    run_inference, cleanup). The harness owns everything else: environment
    snapshot, baseline power, energy tracking, CUDA sync, thermal floor wait,
    FLOPs estimation, timeseries, warnings, and result assembly - split across
    the lifecycle / window / persistence phase modules this facade sequences.
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
        # Build the per-model-load lifetime (snapshot, baseline, model, prompts,
        # warmup); take exactly one measured window against it; then persist.
        lifetime = build_lifetime(
            engine,
            config,
            snapshot=snapshot,
            gpu_indices=gpu_indices,
            progress=progress,
            baseline=baseline,
        )

        try:
            # 7-13. Measured inference window (energy, timing, FLOPs)
            window = run_measured_inference(engine, config, lifetime, gpu_indices, progress)
        finally:
            # Always release the model from memory, even on inference failure.
            engine.cleanup(lifetime.model)

        # 14-18. Persist sidecars + assemble the ExperimentResult
        return persist_and_assemble(
            engine=engine,
            config=config,
            lifetime=lifetime,
            gpu_indices=gpu_indices,
            window=window,
            output_dir=output_dir,
            save_timeseries=save_timeseries,
            progress=progress,
        )
