"""Durable NVML measured-window ordering contract.

This locks the FULL boundary order of the measured inference window - the missing
guard the S8 brief calls out. It threads one shared recorder through a fake
engine (``FakeBackendWithCapture``), a fake energy sampler (a ``FakeEnergySampler``
subclass), and a fake thermal sampler, then asserts the recorded call sequence
rather than any private method name, so it survives the later measurement.py
decomposition.

The window primitives live on ``llenergymeasure.harness.bracket`` and are patched
there (patch at the use site). The load-bearing assertion is the window-width
subtlety: observed-params capture runs INSIDE the energy-tracker window (before
the tracker stops) but OUTSIDE the thermal sampler (after it stops), so the energy
reading is deliberately slightly wider than the thermal timeseries. Any reorder
that moves capture past the tracker stop shifts energy readings at the margin and
must fail here.

A companion progress-order test locks that the "measure" step-done event fires
AFTER the observed-params capture (unchanged from before the bracket extraction).
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Any
from unittest.mock import patch

from llenergymeasure.config.models import DatasetConfig, ExperimentConfig
from llenergymeasure.domain.metrics import ThrottleInfo
from llenergymeasure.harness import MeasurementHarness
from tests.fakes import FakeEnergySampler
from tests.unit.harness.conftest import FakeBackendWithCapture


class _RecordingSampler(FakeEnergySampler):
    """FakeEnergySampler that records tracker start/stop into a shared log."""

    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self._events = events

    def start_tracking(self) -> str:
        self._events.append("tracker_start")
        return super().start_tracking()

    def stop_tracking(self, tracker: Any) -> Any:
        self._events.append("tracker_stop")
        return super().stop_tracking(tracker)


class _RecordingThermal:
    """Fake PowerThermalSampler recording start/stop into a shared log.

    Records on both the context-manager protocol and direct start/stop calls, so
    the same fake locks the order however the bracket drives the thermal sampler.
    """

    def __init__(self, events: list[str], **kwargs: Any) -> None:
        self._events = events

    def start(self) -> None:
        self._events.append("thermal_start")

    def stop(self) -> None:
        self._events.append("thermal_stop")

    def __enter__(self) -> _RecordingThermal:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def get_throttle_info(self) -> ThrottleInfo:
        return ThrottleInfo()

    def get_samples(self) -> list[Any]:
        return []


class _StepDoneRecorder:
    """Progress callback that records only the 'measure' step-done event."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def on_step_done(self, step: str, elapsed_sec: float) -> None:
        if step == "measure":
            self._events.append("step_done:measure")

    def on_step_start(self, *args: Any, **kwargs: Any) -> None: ...
    def on_step_update(self, *args: Any, **kwargs: Any) -> None: ...
    def on_step_skip(self, *args: Any, **kwargs: Any) -> None: ...
    def on_substep(self, *args: Any, **kwargs: Any) -> None: ...
    def on_substep_start(self, *args: Any, **kwargs: Any) -> None: ...
    def on_substep_done(self, *args: Any, **kwargs: Any) -> None: ...


def _resolved_future(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


def _idx_lt(events: list[str], first: str, second: str) -> bool:
    """True when ``first`` is recorded strictly before ``second``."""
    return events.index(first) < events.index(second)


def _run_recorded(*, record_progress: bool = False) -> list[str]:
    """Run one measurement with recording fakes; return the ordered event log.

    With ``record_progress`` a :class:`_StepDoneRecorder` is threaded in so the
    'measure' step-done event lands in the same timeline as the mechanical events.
    """
    events: list[str] = []
    engine = FakeBackendWithCapture(engine_name="transformers", call_log=events)
    config = ExperimentConfig(
        task={
            "model": "fake/model",
            "dataset": DatasetConfig(n_prompts=1),
            "max_input_tokens": 32,
            "max_output_tokens": 32,
        },
        engine="transformers",
        measurement={
            "baseline": {"enabled": True, "duration_seconds": 5.0},
        },
        offline={"warmup": {"enabled": True}},
        serving_mode="offline",
    )
    harness = MeasurementHarness()

    def _baseline(*args: Any, **kwargs: Any) -> None:
        events.append("baseline")
        return None

    def _thermal_floor(config: Any) -> float:
        events.append("thermal_floor")
        return 0.0

    def _select(*args: Any, **kwargs: Any) -> _RecordingSampler:
        events.append("energy_select")
        return _RecordingSampler(events)

    def _cuda_sync() -> None:
        events.append("cuda_sync")

    def _thermal(**kwargs: Any) -> _RecordingThermal:
        return _RecordingThermal(events, **kwargs)

    progress = _StepDoneRecorder(events) if record_progress else None

    with (
        patch(
            "llenergymeasure.harness.lifecycle.collect_environment_snapshot_async",
            return_value=_resolved_future(None),
        ),
        patch("llenergymeasure.harness.lifecycle.load_prompts", return_value=["prompt"]),
        patch(
            "llenergymeasure.harness.lifecycle.measure_baseline_power",
            side_effect=_baseline,
        ),
        patch(
            "llenergymeasure.harness.lifecycle.thermal_floor_wait",
            side_effect=_thermal_floor,
        ),
        patch("llenergymeasure.harness.bracket.select_energy_sampler", side_effect=_select),
        patch("llenergymeasure.harness.bracket._cuda_sync", side_effect=_cuda_sync),
        patch("llenergymeasure.harness.bracket.PowerThermalSampler", side_effect=_thermal),
        patch(
            "llenergymeasure.harness.window.estimate_flops_palm_from_config",
            return_value=None,
        ),
        patch(
            "llenergymeasure.harness.measurement_warnings.collect_measurement_warnings",
            return_value=[],
        ),
    ):
        harness.run(engine, config, gpu_indices=[0], progress=progress)

    return events


def test_measured_window_full_order_contract() -> None:
    """The full pre-window -> window -> boundary order is locked end to end."""
    events = _run_recorded()

    # Two CUDA syncs bracket the run: pre (before thermal start) and post (after
    # thermal stop). Split them out before ordering the rest.
    cuda_positions = [i for i, e in enumerate(events) if e == "cuda_sync"]
    assert len(cuda_positions) == 2, f"expected two CUDA syncs, got {events}"
    pre_sync, post_sync = cuda_positions

    def idx(name: str) -> int:
        return events.index(name)

    order = [
        idx("baseline"),
        idx("load_model"),
        idx("run_warmup_prompt"),
        idx("thermal_floor"),
        idx("energy_select"),
        idx("tracker_start"),
        pre_sync,
        idx("thermal_start"),
        idx("run_inference"),
        idx("thermal_stop"),
        post_sync,
        idx("capture_observed_params"),
        idx("tracker_stop"),
    ]
    assert order == sorted(order), f"measured-window ordering violated: {events}"


def test_capture_inside_energy_window_outside_thermal_window() -> None:
    """Width subtlety: capture sits after thermal stop but before tracker stop.

    This is the deliberate energy>thermal window-width gap. It must never regress:
    capture landing after the tracker stop would drop capture overhead out of the
    energy reading and shift results at the margin.
    """
    events = _run_recorded()
    assert _idx_lt(events, "thermal_stop", "capture_observed_params"), events
    assert _idx_lt(events, "capture_observed_params", "tracker_stop"), events


def test_engine_work_strictly_outside_tracker_window() -> None:
    """Model load and warmup finish before the energy tracker ever starts."""
    events = _run_recorded()
    assert _idx_lt(events, "load_model", "tracker_start"), events
    assert _idx_lt(events, "run_warmup_prompt", "tracker_start"), events


def test_run_inference_strictly_inside_both_windows() -> None:
    """The single inference call is nested inside the thermal and energy windows."""
    events = _run_recorded()
    assert _idx_lt(events, "thermal_start", "run_inference"), events
    assert _idx_lt(events, "run_inference", "thermal_stop"), events
    assert _idx_lt(events, "tracker_start", "run_inference"), events
    assert _idx_lt(events, "run_inference", "tracker_stop"), events


def test_measure_step_done_fires_after_capture() -> None:
    """The 'measure' step-done progress event fires AFTER observed-params capture.

    Pre-refactor the harness fired on_step_done('measure') after the capture; the
    bracket must preserve that (it fires in finish(), after the caller's capture).
    """
    events = _run_recorded(record_progress=True)
    assert _idx_lt(events, "capture_observed_params", "step_done:measure"), events
