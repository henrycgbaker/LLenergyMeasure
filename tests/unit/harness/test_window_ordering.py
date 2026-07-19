"""Durable NVML measured-window ordering contract.

This locks the FULL boundary order of the measured inference window - the missing
guard the S8 brief calls out. It threads one shared recorder through a fake
engine, a fake energy sampler, and a fake thermal sampler, then asserts the
recorded call sequence rather than any private method name, so it survives the
later measurement.py decomposition.

The harness's window primitives (``select_energy_sampler``, ``_cuda_sync``,
``PowerThermalSampler``) are patched on ``llenergymeasure.harness.measurement`` -
the harness's canonical monkeypatch surface. The MeasurementBracket extraction
keeps resolving those primitives through that module, so this test reads
identically before and after the extraction.

The load-bearing assertion is the window-width subtlety: observed-params capture
runs INSIDE the energy-tracker window (before the tracker stops) but OUTSIDE the
thermal sampler (after the thermal sampler stops). The energy reading is
therefore deliberately slightly wider than the thermal timeseries. Any reorder
that moves capture past the tracker stop shifts energy readings at the margin
and must fail here.
"""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import patch

from llenergymeasure.config.models import DatasetConfig, ExperimentConfig
from llenergymeasure.domain.metrics import ThermalThrottleInfo
from llenergymeasure.energy.nvml import EnergyMeasurement
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.harness import MeasurementHarness


@dataclass
class _RecordingEngine:
    """Fake EnginePlugin that records every lifecycle call into a shared log."""

    events: list[str]
    name: str = "transformers"

    def load_model(self, config: Any, **kwargs: Any) -> dict:
        self.events.append("model_load")
        return {"model": "fake"}

    def run_warmup_prompt(self, config: Any, model: Any, prompt: str) -> float:
        self.events.append("warmup")
        return 0.0  # kernel-warmup branch: one discarded probe, no CV loop

    def run_inference(self, config: Any, model: Any, prompts: list[str]) -> InferenceOutput:
        self.events.append("run_inference")
        return InferenceOutput(
            elapsed_time_sec=1.0,
            input_tokens=8,
            output_tokens=8,
            peak_memory_mb=0.0,
            model_memory_mb=0.0,
        )

    def capture_observed_params(self, config: Any, model: Any, output: Any) -> dict:
        self.events.append("observed_capture")
        return {"engine": {}, "sampling": {}, "library_version": "test"}

    def cleanup(self, model: Any) -> None:
        pass


@dataclass
class _RecordingSampler:
    """Fake energy sampler recording tracker start/stop into the shared log."""

    events: list[str]

    def start_tracking(self) -> str:
        self.events.append("tracker_start")
        return "tracker-handle"

    def stop_tracking(self, tracker: Any) -> EnergyMeasurement:
        self.events.append("tracker_stop")
        return EnergyMeasurement(total_j=10.0, duration_sec=1.0, per_gpu_j={0: 10.0})


@dataclass
class _RecordingThermal:
    """Fake PowerThermalSampler recording start/stop into the shared log.

    Records on both the context-manager protocol (current code wraps the run in a
    ``with``) and direct start/stop calls (the bracket splits the thermal lifetime
    across its own enter and exit), so the same fake locks the order on either
    shape.
    """

    events: list[str]
    kwargs: dict[str, Any] = field(default_factory=dict)

    def start(self) -> None:
        self.events.append("thermal_start")

    def stop(self) -> None:
        self.events.append("thermal_stop")

    def __enter__(self) -> _RecordingThermal:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def get_thermal_throttle_info(self) -> ThermalThrottleInfo:
        return ThermalThrottleInfo()

    def get_samples(self) -> list[Any]:
        return []


def _resolved_future(value: Any) -> Future:
    future: Future = Future()
    future.set_result(value)
    return future


def _idx_lt(events: list[str], first: str, second: str) -> bool:
    """True when ``first`` is recorded strictly before ``second``."""
    return events.index(first) < events.index(second)


def _run_recorded() -> list[str]:
    """Run one measurement with recording fakes; return the ordered event log."""
    events: list[str] = []
    engine = _RecordingEngine(events)
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
            "warmup": {"enabled": True},
        },
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
        return _RecordingThermal(events, kwargs)

    with (
        patch(
            "llenergymeasure.harness.measurement.collect_environment_snapshot_async",
            return_value=_resolved_future(None),
        ),
        patch("llenergymeasure.harness.measurement.load_prompts", return_value=["prompt"]),
        patch(
            "llenergymeasure.harness.measurement.measure_baseline_power",
            side_effect=_baseline,
        ),
        patch(
            "llenergymeasure.harness.measurement.thermal_floor_wait",
            side_effect=_thermal_floor,
        ),
        patch(
            "llenergymeasure.harness.measurement.select_energy_sampler",
            side_effect=_select,
        ),
        patch("llenergymeasure.harness.measurement._cuda_sync", side_effect=_cuda_sync),
        patch("llenergymeasure.harness.measurement.PowerThermalSampler", side_effect=_thermal),
        patch(
            "llenergymeasure.harness.measurement.estimate_flops_palm_from_config",
            return_value=None,
        ),
        patch(
            "llenergymeasure.harness.measurement.collect_measurement_warnings",
            return_value=[],
        ),
    ):
        harness.run(engine, config, gpu_indices=[0])

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
        idx("model_load"),
        idx("warmup"),
        idx("thermal_floor"),
        idx("energy_select"),
        idx("tracker_start"),
        pre_sync,
        idx("thermal_start"),
        idx("run_inference"),
        idx("thermal_stop"),
        post_sync,
        idx("observed_capture"),
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
    assert _idx_lt(events, "thermal_stop", "observed_capture"), events
    assert _idx_lt(events, "observed_capture", "tracker_stop"), events


def test_engine_work_strictly_outside_tracker_window() -> None:
    """Model load and warmup finish before the energy tracker ever starts."""
    events = _run_recorded()
    assert _idx_lt(events, "model_load", "tracker_start"), events
    assert _idx_lt(events, "warmup", "tracker_start"), events


def test_run_inference_strictly_inside_both_windows() -> None:
    """The single inference call is nested inside the thermal and energy windows."""
    events = _run_recorded()
    assert _idx_lt(events, "thermal_start", "run_inference"), events
    assert _idx_lt(events, "run_inference", "thermal_stop"), events
    assert _idx_lt(events, "tracker_start", "run_inference"), events
    assert _idx_lt(events, "run_inference", "tracker_stop"), events
