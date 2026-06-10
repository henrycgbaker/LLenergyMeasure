"""Harness-level tests for latency profiling provenance and wiring.

Uses the same FakeBackend/_apply_patches pattern as test_harness.py but drives a
controllable InferenceOutput so the harness's latency_stats/tpot_ms/provenance
logic can be asserted without GPU or engine libraries.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.config.models import DatasetConfig, ExperimentConfig
from llenergymeasure.domain.metrics import LatencyMeasurementMode
from llenergymeasure.engines.protocol import InferenceOutput
from llenergymeasure.harness import MeasurementHarness
from tests.unit.harness.test_harness import FakeBackend, _make_mock_pts


def _config(latency_profiling: bool) -> ExperimentConfig:
    return ExperimentConfig(
        task={
            "model": "fake/model",
            "dataset": DatasetConfig(n_prompts=1),
            "max_input_tokens": 32,
            "max_output_tokens": 32,
        },
        engine="transformers",
        measurement={"latency_profiling": latency_profiling},
    )


def _apply_patches():
    import contextlib

    stack = contextlib.ExitStack()
    patches = [
        patch(
            "llenergymeasure.harness.measurement.collect_environment_snapshot", return_value=None
        ),
        patch("llenergymeasure.harness.measurement.measure_baseline_power", return_value=None),
        patch("llenergymeasure.harness.measurement.load_prompts", return_value=["test prompt"]),
        patch("llenergymeasure.harness.measurement.select_energy_sampler", return_value=None),
        patch("llenergymeasure.harness.measurement.thermal_floor_wait", return_value=0.0),
        patch(
            "llenergymeasure.harness.measurement.estimate_flops_palm",
            return_value=MagicMock(value=1e9),
        ),
        patch("llenergymeasure.harness.measurement._cuda_sync"),
        patch("llenergymeasure.harness.measurement.PowerThermalSampler", new=_make_mock_pts()),
        patch(
            "llenergymeasure.harness.measurement.write_timeseries_parquet",
            return_value=MagicMock(name="timeseries.parquet"),
        ),
        patch("llenergymeasure.harness.measurement.collect_measurement_warnings", return_value=[]),
    ]
    for p in patches:
        stack.enter_context(p)
    return stack


def _profiled_output() -> InferenceOutput:
    """An InferenceOutput as a profiling-enabled streaming engine would emit."""
    return InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
        batch_times=[1.0],
        ttft_ms=[100.0, 110.0],
        itl_ms=[5.0, 6.0, 7.0, 8.0],
        latency_measurement_mode=LatencyMeasurementMode.PROPORTIONAL_ESTIMATE.value,
    )


def test_profiling_on_populates_latency_stats_mode_and_tpot():
    engine = FakeBackend(inference_output=_profiled_output())
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, _config(True))

    assert result.latency_stats is not None
    assert result.latency_stats.measurement_mode is LatencyMeasurementMode.PROPORTIONAL_ESTIMATE
    # tpot_ms is the ITL mean (5,6,7,8 -> 6.5)
    assert result.extended_metrics.tpot_ms == pytest.approx(6.5)
    # Provenance warning present
    assert any("latency_profiling enabled" in w for w in result.measurement_warnings)


def test_profiling_on_forced_batch_warning():
    out = _profiled_output()
    out.extras["profiling_forced_batch_size"] = True
    engine = FakeBackend(inference_output=out)
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, _config(True))

    assert any("forced batch_size=1" in w for w in result.measurement_warnings)


def test_profiling_on_unsupported_engine_warning():
    out = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
        batch_times=[1.0],
    )
    out.extras["latency_profiling_unsupported"] = True
    engine = FakeBackend(inference_output=out)
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, _config(True))

    assert any("not supported" in w for w in result.measurement_warnings)
    # No streaming signal -> no latency_stats
    assert result.latency_stats is None


def test_profiling_off_no_warning_and_nulls():
    out = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
        batch_times=[1.0],
    )
    engine = FakeBackend(inference_output=out)
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, _config(False))

    assert result.latency_stats is None
    assert result.extended_metrics.tpot_ms is None
    assert not any("latency_profiling" in w for w in result.measurement_warnings)


def test_ttft_without_mode_defaults_and_warns():
    """Defensive path: ttft present but mode None -> default + warning (should be rare)."""
    out = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=10,
        output_tokens=20,
        peak_memory_mb=512.0,
        model_memory_mb=256.0,
        batch_times=[1.0],
        ttft_ms=[100.0, 110.0],
        latency_measurement_mode=None,
    )
    engine = FakeBackend(inference_output=out)
    harness = MeasurementHarness()

    with _apply_patches():
        result = harness.run(engine, _config(False))

    assert result.latency_stats is not None
    assert result.latency_stats.measurement_mode is LatencyMeasurementMode.TRUE_STREAMING
    assert any("latency_measurement_mode missing" in w for w in result.measurement_warnings)
