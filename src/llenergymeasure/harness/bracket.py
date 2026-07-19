"""MeasurementBracket - the mode-agnostic measured-window mechanics.

The bracket owns the measured window's boundary: energy-sampler selection,
energy-tracker start/stop, the pre/post CUDA syncs, the PowerThermalSampler
lifetime, and the wall-clock + perf-counter timestamps. It is deliberately
independent of WHAT runs inside it - no engine, no prompts, no InferenceOutput in
its signature - so server mode (v0.8.0) can reuse it around a load-gen-timed run
unchanged.

Window-width subtlety (behavior-frozen): the energy tracker is stopped in
``finish()``, strictly after ``__exit__`` has stopped the thermal sampler. The
offline caller interleaves its observed-params capture between context exit and
``finish()`` so capture lands INSIDE the energy window but OUTSIDE the thermal
timeseries - the energy reading is deliberately slightly wider than the thermal
window. See ``tests/unit/harness/test_window_ordering.py``.

The window primitives (``select_energy_sampler``,
``select_energy_sampler_with_diagnostics``, ``_cuda_sync``,
``PowerThermalSampler``) are resolved through ``llenergymeasure.harness.
measurement`` - the harness's canonical monkeypatch surface - rather than
imported directly, so that module stays the single place tests patch them and the
durable ordering test reads identically across this extraction. The perf clock
(``time``) is a plain module import here, patchable at ``bracket.time``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llenergymeasure.config.models import MeasurementConfig
    from llenergymeasure.device.power_thermal import PowerThermalSample
    from llenergymeasure.domain.progress import ProgressCallback

STEP_ENERGY_SELECT = "energy_select"
STEP_MEASURE = "measure"


@dataclass
class MeasuredWindowCore:
    """Mode-agnostic product of one measured window.

    Holds only what the window mechanics produce - energy, thermal, timeseries,
    and the wall-clock span. Mode-specific extras (the InferenceOutput, FLOPs)
    are composed on top by the caller.
    """

    thermal_info: Any
    timeseries_samples: list[PowerThermalSample]
    energy_measurement: Any
    # Per-sampler probe reasons, populated only when auto-selection found no
    # available sampler; the caller folds them into structured measurement_warnings.
    energy_sampler_reasons: list[str]
    start_time: datetime
    end_time: datetime


class MeasurementBracket:
    """Context manager owning the measured-window boundary mechanics.

    Usage (offline)::

        bracket = MeasurementBracket(
            config.measurement, gpu_indices, progress, measure_detail="inference (N prompts)"
        )
        with bracket:
            output = engine.run_inference(config, model, prompts)
        # capture runs here: after the thermal sampler stops, before the tracker
        core = bracket.finish()

    ``__enter__`` selects the energy sampler, starts the tracker, runs the pre
    CUDA sync, and starts the thermal sampler. ``__exit__`` stops the thermal
    sampler and runs the post CUDA sync. ``finish()`` stops the energy tracker
    and returns the :class:`MeasuredWindowCore`.
    """

    def __init__(
        self,
        measurement_config: MeasurementConfig,
        gpu_indices: list[int] | None,
        progress: ProgressCallback | None,
        *,
        measure_detail: str = "",
    ) -> None:
        self._measurement_config = measurement_config
        self._gpu_indices = gpu_indices
        self._progress = progress
        self._measure_detail = measure_detail

        self._energy_sampler: Any = None
        self._energy_tracker: Any = None
        self._energy_sampler_reasons: list[str] = []
        self._thermal_sampler: Any = None
        self._thermal_info: Any = None
        self._timeseries_samples: list[Any] = []
        self._t_inference_start: float = 0.0
        self._t_inference_end: float = 0.0
        self._start_time: datetime | None = None

    def _substep(self, step: str, text: str) -> None:
        """Emit a substep event when a progress callback is registered."""
        if self._progress is not None:
            self._progress.on_substep(step, text)

    def __enter__(self) -> MeasurementBracket:
        from llenergymeasure.harness import measurement as _m

        _p = self._progress

        # 7. Select energy sampler.
        if _p:
            _p.on_step_start(STEP_ENERGY_SELECT, "Selecting", "energy sampler")
            t0_energy = time.perf_counter()
        self._energy_sampler = _m.select_energy_sampler(
            self._measurement_config.energy_sampler, gpu_indices=self._gpu_indices
        )
        # Capture the per-sampler probe reasons when auto-selection came up empty,
        # so they reach structured measurement_warnings (not only the log). Kept
        # off the patchable select_energy_sampler seam (harness tests patch it);
        # the diagnostics re-probe only runs on the rare no-sampler path.
        self._energy_sampler_reasons = []
        if self._energy_sampler is None and self._measurement_config.energy_sampler is not None:
            _, self._energy_sampler_reasons = _m.select_energy_sampler_with_diagnostics(
                self._measurement_config.energy_sampler, gpu_indices=self._gpu_indices
            )
        sampler_name = type(self._energy_sampler).__name__ if self._energy_sampler else "none"
        self._substep(STEP_ENERGY_SELECT, f"selected: {sampler_name}")
        if _p:
            _p.on_step_update(STEP_ENERGY_SELECT, f"energy sampler ({sampler_name})")
            _p.on_step_done(STEP_ENERGY_SELECT, time.perf_counter() - t0_energy)

        # 8. Start energy tracking (after warmup + thermal floor).
        self._energy_tracker = None
        if self._energy_sampler is not None:
            self._energy_tracker = self._energy_sampler.start_tracking()

        # 9. CUDA sync before inference (Zeus best practice).
        _m._cuda_sync()
        self._substep(STEP_MEASURE, "CUDA sync (pre)")

        if _p:
            _p.on_step_start(STEP_MEASURE, "Measuring", self._measure_detail)
        self._substep(STEP_MEASURE, "energy tracker started")

        self._t_inference_start = time.perf_counter()
        self._start_time = datetime.now()

        # Start the thermal sampler around inference (timeseries + throttle).
        self._thermal_sampler = _m.PowerThermalSampler(gpu_indices=self._gpu_indices)
        self._thermal_sampler.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        # Returns None (falsy) so an exception raised inside the window is never
        # suppressed - the caller's cleanup still runs.
        from llenergymeasure.harness import measurement as _m

        # Stop the thermal sampler; the energy tracker stays open until finish().
        self._thermal_sampler.stop()
        self._t_inference_end = time.perf_counter()

        # 11. CUDA sync after inference, before stopping energy.
        _m._cuda_sync()
        self._substep(STEP_MEASURE, "CUDA sync (post)")

        if self._progress:
            self._progress.on_step_done(STEP_MEASURE, self.inference_duration_sec)

        self._thermal_info = self._thermal_sampler.get_thermal_throttle_info()
        self._timeseries_samples = self._thermal_sampler.get_samples()

    @property
    def inference_duration_sec(self) -> float:
        """Perf-counter delta bracketing the run inside the window."""
        return self._t_inference_end - self._t_inference_start

    def finish(self) -> MeasuredWindowCore:
        """Stop the energy tracker and return the finalised window core.

        Called after the caller's post-window work (e.g. observed-params capture)
        so that work lands inside the energy window but after the thermal window.
        """
        energy_measurement = None
        if self._energy_sampler is not None and self._energy_tracker is not None:
            energy_measurement = self._energy_sampler.stop_tracking(self._energy_tracker)
            tracker_duration = energy_measurement.duration_sec if energy_measurement else 0.0
            self._substep(STEP_MEASURE, f"energy tracker stopped  {tracker_duration:.1f}s")
        end_time = datetime.now()

        assert self._start_time is not None  # set in __enter__
        return MeasuredWindowCore(
            thermal_info=self._thermal_info,
            timeseries_samples=self._timeseries_samples,
            energy_measurement=energy_measurement,
            energy_sampler_reasons=self._energy_sampler_reasons,
            start_time=self._start_time,
            end_time=end_time,
        )
