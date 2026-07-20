"""Unit tests for measurement integration: timeseries, warnings, and TransformersEngine wiring.

All tests are mocked - no GPU or real model required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llenergymeasure.device.power_thermal import PowerThermalSample
from llenergymeasure.harness.lifecycle import run_warmup
from llenergymeasure.harness.measurement_warnings import (
    collect_measurement_warnings,
    collect_warnings,
)
from llenergymeasure.harness.result_assembly import (
    build_result,
    resolve_measurement_mode,
)
from llenergymeasure.harness.timeseries import write_timeseries_parquet
from llenergymeasure.harness.window import estimate_flops

# =============================================================================
# Timeseries writer tests
# =============================================================================


def _make_samples(n_seconds: int, start_ts: float = 0.0) -> list[PowerThermalSample]:
    """Generate synthetic PowerThermalSamples at 100ms intervals spanning n_seconds."""
    samples = []
    for sec in range(n_seconds):
        for ms_offset in range(0, 1000, 100):  # 10 samples per second
            ts = start_ts + sec + ms_offset / 1000.0
            samples.append(
                PowerThermalSample(
                    timestamp=ts,
                    power_w=100.0 + sec * 2.0,
                    temperature_c=45.0 + sec * 0.5,
                    memory_used_mb=8192.0,
                    memory_total_mb=40960.0,
                    sm_utilisation=85.0,
                    throttle_reasons=0,
                )
            )
    return samples


def test_timeseries_parquet_write() -> None:
    """write_timeseries_parquet() produces 1 Hz rows with correct schema."""
    import pyarrow.parquet as pq

    samples = _make_samples(n_seconds=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "timeseries.parquet"
        result_path = write_timeseries_parquet(samples, output, gpu_index=0)

        assert result_path == output
        assert output.exists()

        table = pq.read_table(output)
        # Should have 3-4 rows (one per second bucket)
        assert 3 <= len(table) <= 4, f"Expected 3-4 rows, got {len(table)}"

        # Check schema columns
        expected_columns = {
            "timestamp_s",
            "gpu_index",
            "power_w",
            "temperature_c",
            "memory_used_mb",
            "memory_total_mb",
            "sm_utilisation_pct",
            "throttle_reasons",
        }
        assert set(table.schema.names) == expected_columns

        # Check gpu_index column value
        gpu_indices = table.column("gpu_index").to_pylist()
        assert all(g == 0 for g in gpu_indices)

        # Power values should be non-null (we provided real values)
        power_values = table.column("power_w").to_pylist()
        assert all(p is not None for p in power_values)


def test_timeseries_parquet_empty() -> None:
    """write_timeseries_parquet() with empty samples creates 0-row file with correct schema."""
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "timeseries.parquet"
        write_timeseries_parquet([], output, gpu_index=1)

        assert output.exists()
        table = pq.read_table(output)
        assert len(table) == 0

        # Schema must still be correct even with 0 rows
        expected_columns = {
            "timestamp_s",
            "gpu_index",
            "power_w",
            "temperature_c",
            "memory_used_mb",
            "memory_total_mb",
            "sm_utilisation_pct",
            "throttle_reasons",
        }
        assert set(table.schema.names) == expected_columns


def test_timeseries_parquet_creates_parent_dir() -> None:
    """write_timeseries_parquet() creates parent directories as needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "nested" / "dirs" / "timeseries.parquet"
        write_timeseries_parquet([], output)
        assert output.exists()


def test_timeseries_throttle_reasons_ored() -> None:
    """Throttle reason bitmasks are OR'd within each 1s bucket."""
    import pyarrow.parquet as pq

    # Two samples in same second with different throttle bits
    samples = [
        PowerThermalSample(timestamp=0.0, throttle_reasons=0b01),
        PowerThermalSample(timestamp=0.1, throttle_reasons=0b10),
        PowerThermalSample(timestamp=0.2, throttle_reasons=0b00),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "ts.parquet"
        write_timeseries_parquet(samples, output)
        table = pq.read_table(output)
        throttle = table.column("throttle_reasons").to_pylist()
        assert throttle[0] == 0b11  # 0b01 | 0b10 | 0b00


# =============================================================================
# Measurement warnings tests
# =============================================================================


def test_warnings_short_duration() -> None:
    """Short measurement duration triggers short_measurement_duration warning."""
    warnings = collect_measurement_warnings(
        duration_sec=5.0,
        gpu_persistence_mode=True,
        temp_start_c=45.0,
        temp_end_c=46.0,
        nvml_sample_count=100,
    )
    assert any("short_measurement_duration" in w for w in warnings)


def test_warnings_persistence_mode() -> None:
    """Persistence mode off triggers gpu_persistence_mode_off warning."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=False,
        temp_start_c=45.0,
        temp_end_c=46.0,
        nvml_sample_count=100,
    )
    assert any("gpu_persistence_mode_off" in w for w in warnings)


def test_warnings_thermal_drift() -> None:
    """Temperature drift above threshold triggers thermal_drift_detected warning."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=30.0,
        temp_end_c=45.0,  # 15C drift > 10C threshold
        nvml_sample_count=100,
        thermal_drift_threshold_c=10.0,
    )
    assert any("thermal_drift_detected" in w for w in warnings)
    # Warning should include the actual drift and threshold
    drift_warning = next(w for w in warnings if "thermal_drift_detected" in w)
    assert "15.0C" in drift_warning
    assert "10.0C" in drift_warning


def test_warnings_thermal_drift_below_threshold() -> None:
    """Temperature drift below threshold does NOT trigger thermal_drift_detected."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=44.0,
        temp_end_c=45.0,  # 1C drift < 10C threshold
        nvml_sample_count=100,
        thermal_drift_threshold_c=10.0,
    )
    assert not any("thermal_drift_detected" in w for w in warnings)


def test_warnings_low_sample_count() -> None:
    """Fewer than 10 NVML samples triggers nvml_low_sample_count warning."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=45.0,
        temp_end_c=46.0,
        nvml_sample_count=5,
    )
    assert any("nvml_low_sample_count" in w for w in warnings)


def test_warnings_clean_measurement() -> None:
    """With nvml_sample_count > 0, only the throttle-subsampling methodology warning fires.

    The throttle-subsampling warning is always present when NVML is active - it is an
    all other conditions optimal and nvml_sample_count=600, no warnings fire.
    """
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=44.0,
        temp_end_c=45.0,
        nvml_sample_count=600,
        thermal_drift_threshold_c=10.0,
    )
    assert len(warnings) == 0


def test_warnings_all_four_triggered() -> None:
    """All four warning conditions triggered simultaneously."""
    warnings = collect_measurement_warnings(
        duration_sec=5.0,  # < 10s
        gpu_persistence_mode=False,  # off
        temp_start_c=30.0,  # 15C drift
        temp_end_c=45.0,
        nvml_sample_count=5,  # < 10 samples
        thermal_drift_threshold_c=10.0,
    )
    assert len(warnings) == 4
    warning_str = " ".join(warnings)
    assert "short_measurement_duration" in warning_str
    assert "gpu_persistence_mode_off" in warning_str
    assert "thermal_drift_detected" in warning_str
    assert "nvml_low_sample_count" in warning_str


def test_warnings_none_temps_no_drift_warning() -> None:
    """When temperatures are None, thermal_drift_detected is NOT triggered."""
    warnings = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=None,
        temp_end_c=None,
        nvml_sample_count=100,
    )
    assert not any("thermal_drift_detected" in w for w in warnings)


def test_warnings_custom_threshold() -> None:
    """Custom thermal_drift_threshold_c is applied correctly."""
    # 5C drift should only warn if threshold is 4C
    warnings_4 = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=40.0,
        temp_end_c=45.0,
        nvml_sample_count=100,
        thermal_drift_threshold_c=4.0,
    )
    warnings_6 = collect_measurement_warnings(
        duration_sec=60.0,
        gpu_persistence_mode=True,
        temp_start_c=40.0,
        temp_end_c=45.0,
        nvml_sample_count=100,
        thermal_drift_threshold_c=6.0,
    )
    assert any("thermal_drift_detected" in w for w in warnings_4)
    assert not any("thermal_drift_detected" in w for w in warnings_6)


# =============================================================================
# Harness and engine wiring tests
# =============================================================================


def test_cuda_sync_called() -> None:
    """_cuda_sync() calls torch.cuda.synchronize() when CUDA is available."""
    from llenergymeasure.harness import _cuda_sync

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        _cuda_sync()

    mock_torch.cuda.synchronize.assert_called_once()


def test_cuda_sync_skipped_when_cuda_unavailable() -> None:
    """_cuda_sync() skips synchronize() when torch.cuda.is_available() is False."""
    from llenergymeasure.harness import _cuda_sync

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    with (
        patch.dict("sys.modules", {"torch": mock_torch}),
        patch("importlib.util.find_spec", return_value=MagicMock()),
    ):
        _cuda_sync()

    mock_torch.cuda.synchronize.assert_not_called()


def test_warmup_disabled_skips_probe_call() -> None:
    """Harness skips run_warmup_prompt() probe when warmup.enabled=False."""
    from llenergymeasure.config.models import WarmupConfig
    from llenergymeasure.domain.metrics import WarmupResult
    from llenergymeasure.harness.warmup import warmup_until_converged

    wc = WarmupConfig(enabled=False)
    result = warmup_until_converged(lambda: 1.0, wc)
    assert isinstance(result, WarmupResult)
    assert result.iterations_completed == 0
    assert result.converged is True
    assert result.final_cv == 0.0


def test_harness_build_result_uses_real_energy_values() -> None:
    """MeasurementHarness._build_result() populates total_energy_j from EnergyMeasurement."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo
    from llenergymeasure.energy.nvml import EnergyMeasurement
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "test/model"})
    output = InferenceOutput(
        elapsed_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )
    energy_measurement = EnergyMeasurement(total_j=42.5, duration_sec=10.0)
    flops_result = FlopsResult(
        value=1e12, method="palm_formula", confidence="medium", precision="n/a"
    )
    now = datetime.now()
    result, _ = build_result(
        engine_name="transformers",
        config=config,
        output=output,
        model_memory_mb=0.0,
        start_time=now,
        end_time=now,
        duration_sec=10.0,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=energy_measurement,
        baseline=None,
        flops_result=flops_result,
        timeseries_path=None,
        measurement_warnings=[],
    )

    assert result.total_energy_j == 42.5
    assert result.total_flops == 1e12
    assert result.avg_energy_per_token_j == pytest.approx(42.5 / 50)
    assert result.measurement_warnings == []
    assert result.energy_breakdown is not None


def test_harness_build_result_absent_energy_placeholder_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_build_result() with no energy measurement keeps a 0.0 placeholder and warns loudly.

    The schema requires a non-null total_energy_j, so absence keeps 0.0 - but it is made
    distinguishable from a measured zero via a WARNING log naming the absence (item 2 of
    the silent-zero hardening; the persisted measurement_warnings list carries the
    corresponding energy_measurement_unavailable flag, tested separately).
    """
    import logging
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "test/model"})
    output = InferenceOutput(
        elapsed_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )
    flops_result = FlopsResult(
        value=1e12, method="palm_formula", confidence="medium", precision="n/a"
    )
    now = datetime.now()
    with caplog.at_level(logging.WARNING, logger="llenergymeasure.harness.result_assembly"):
        result, _ = build_result(
            engine_name="transformers",
            config=config,
            output=output,
            model_memory_mb=0.0,
            start_time=now,
            end_time=now,
            duration_sec=10.0,
            thermal_info=ThermalThrottleInfo(),
            energy_measurement=None,
            baseline=None,
            flops_result=flops_result,
            timeseries_path=None,
            measurement_warnings=[],
        )

    assert result.total_energy_j == 0.0
    assert any("No energy measurement available" in rec.message for rec in caplog.records)


def test_harness_collect_warnings_flags_absent_energy() -> None:
    """_collect_warnings() emits energy_measurement_unavailable when energy is absent.

    Thermal telemetry sampled fine (30 samples), so this proves the absent-energy flag is
    keyed off the authoritative backend, not the thermal sampler's sample count.
    """

    warnings = collect_warnings(
        duration_sec=30.0,
        timeseries_samples=_make_samples(n_seconds=3),
        gpu_indices=None,
        energy_measurement=None,
    )
    assert any("energy_measurement_unavailable" in w for w in warnings)
    assert not any("nvml_low_sample_count" in w for w in warnings)


def test_harness_collect_warnings_present_energy_no_flag() -> None:
    """_collect_warnings() does not flag energy absence when a measurement is present."""
    from llenergymeasure.energy.nvml import EnergyMeasurement

    warnings = collect_warnings(
        duration_sec=30.0,
        timeseries_samples=_make_samples(n_seconds=3),
        gpu_indices=None,
        energy_measurement=EnergyMeasurement(total_j=42.5, duration_sec=30.0),
    )
    assert not any("energy_measurement_unavailable" in w for w in warnings)


def test_harness_build_result_uses_energy_measurement_duration_for_baseline() -> None:
    """Baseline energy adjustment uses energy_measurement.duration_sec, not datetime delta."""
    from datetime import datetime, timedelta

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import ThermalThrottleInfo
    from llenergymeasure.energy.nvml import EnergyMeasurement
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "test/model"})
    output = InferenceOutput(
        elapsed_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )

    # Energy engine measured 8.0s (sampler window), but datetime delta is 10.0s.
    # Baseline adjustment should use 8.0s, not 10.0s.
    energy_measurement = EnergyMeasurement(total_j=100.0, duration_sec=8.0)
    now = datetime.now()
    result, _ = build_result(
        engine_name="transformers",
        config=config,
        output=output,
        model_memory_mb=0.0,
        start_time=now,
        end_time=now + timedelta(seconds=10),
        duration_sec=10.0,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=energy_measurement,
        baseline=None,
        flops_result=None,
        timeseries_path=None,
        measurement_warnings=[],
    )

    # With no baseline, energy_breakdown.total_energy_j == raw total.
    # The key check: the function received energy_measurement with duration_sec=8.0
    # and should pass 8.0 (not 10.0) to create_energy_breakdown.
    assert result.energy_breakdown is not None
    assert result.total_energy_j == 100.0


def test_harness_build_result_zero_energy_when_no_engine() -> None:
    """MeasurementHarness._build_result() returns total_energy_j=0.0 when energy_measurement is None."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "test/model"})
    output = InferenceOutput(
        elapsed_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )
    flops_result = FlopsResult(value=0.0, method="palm_formula", confidence="low", precision="n/a")
    now = datetime.now()

    result, _ = build_result(
        engine_name="transformers",
        config=config,
        output=output,
        model_memory_mb=0.0,
        start_time=now,
        end_time=now,
        duration_sec=10.0,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=None,
        baseline=None,
        flops_result=flops_result,
        timeseries_path=None,
        measurement_warnings=["short_measurement_duration: ..."],
    )

    assert result.total_energy_j == 0.0
    assert result.avg_energy_per_token_j == 0.0
    assert len(result.measurement_warnings) == 1


# =============================================================================
# Measurement-methodology wiring (total / windowed / steady_state)
# =============================================================================


def _methodology_build_result(measurement: dict, *, samples, output_tokens=100):
    """Run _build_result with a flat-100W timeseries; return (result, methodology)."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import ThermalThrottleInfo
    from llenergymeasure.energy.nvml import EnergyMeasurement
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "test/model"}, measurement=measurement)
    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=0,
        output_tokens=output_tokens,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )
    output.inference_time_sec = 1.0
    # Sampler total (100 J over the full run) - the windowed path re-integrates from
    # the timeseries and overrides this; total mode keeps it.
    energy_measurement = EnergyMeasurement(total_j=100.0, duration_sec=1.0)
    now = datetime.now()
    return build_result(
        engine_name="transformers",
        config=config,
        output=output,
        model_memory_mb=0.0,
        start_time=now,
        end_time=now,
        duration_sec=1.0,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=energy_measurement,
        baseline=None,
        flops_result=None,
        timeseries_path=None,
        measurement_warnings=[],
        timeseries_samples=samples,
    )


def _flat_samples():
    """11 samples 0..1.0s at constant 100 W (full-run energy = 100 J)."""
    from llenergymeasure.device.power_thermal import PowerThermalSample

    return [PowerThermalSample(timestamp=i * 0.1, power_w=100.0, gpu_index=0) for i in range(11)]


def test_methodology_total_is_unchanged_default() -> None:
    """Default total mode keeps the sampler total and spans the whole run."""
    result, methodology = _methodology_build_result({}, samples=_flat_samples())
    assert methodology.measurement_methodology == "total"
    assert result.total_energy_j == pytest.approx(100.0)
    assert result.total_inference_time_sec == pytest.approx(1.0)
    assert methodology.steady_state_window == (0.0, 1.0)
    assert methodology.steady_state_not_detected is False
    assert methodology.measurement_window_discard_fraction is None
    # output_tokens=100 over 100 J -> 1.0 J/token, mj 1000.
    assert result.avg_energy_per_token_j == pytest.approx(1.0)


def test_methodology_windowed_reintegrates_and_attributes_tokens() -> None:
    """windowed [0.2,0.7] re-integrates 50 J and attributes 50% of tokens."""
    result, methodology = _methodology_build_result(
        {"measurement_methodology": "windowed", "measurement_window": (0.2, 0.7)},
        samples=_flat_samples(),
    )
    assert methodology.measurement_methodology == "windowed"
    assert result.total_energy_j == pytest.approx(50.0)
    assert methodology.steady_state_window == (0.2, 0.7)
    assert result.total_inference_time_sec == pytest.approx(0.5)
    # 50 J over 50% of 100 output tokens = 50 J / 50 tokens = 1.0 J/token.
    assert result.avg_energy_per_token_j == pytest.approx(1.0)
    # throughput: 50% of 100 total tokens over 0.5s window = 100 tok/s.
    assert result.avg_tokens_per_second == pytest.approx(100.0)
    assert any("proportionally by time" in w for w in result.measurement_warnings)


def test_methodology_steady_state_fixed_discard() -> None:
    """steady_state fixed fraction 0.3 -> window [0.3,1.0], 70 J, discard fraction recorded."""
    result, methodology = _methodology_build_result(
        {"measurement_methodology": "steady_state", "warmup_discard_fraction": 0.3},
        samples=_flat_samples(),
    )
    assert methodology.measurement_methodology == "steady_state"
    assert result.total_energy_j == pytest.approx(70.0)
    assert methodology.steady_state_window[0] == pytest.approx(0.3)
    assert methodology.steady_state_window[1] == pytest.approx(1.0)
    assert methodology.measurement_window_discard_fraction == pytest.approx(0.3)
    assert methodology.steady_state_not_detected is False


def test_methodology_steady_state_auto_not_detected_flag() -> None:
    """auto-detect on a never-stable series sets the not-detected flag in the methodology."""
    import math

    from llenergymeasure.device.power_thermal import PowerThermalSample

    noisy = [
        PowerThermalSample(
            timestamp=t / 10,
            power_w=50.0 + 40.0 * math.sin(t) * (1 + t / 10) + (t * 3),
            gpu_index=0,
        )
        for t in range(60)
    ]
    result, methodology = _methodology_build_result(
        {
            "measurement_methodology": "steady_state",
            "steady_state_auto_detect": True,
            "warmup_discard_fraction": 0.1,
        },
        samples=noisy,
    )
    assert methodology.measurement_methodology == "steady_state"
    assert methodology.steady_state_not_detected is True
    assert any("auto-detection found no stable region" in w for w in result.measurement_warnings)


def test_methodology_falls_back_to_total_without_samples() -> None:
    """No timeseries samples -> windowing cannot apply, total figures retained."""
    result, methodology = _methodology_build_result(
        {"measurement_methodology": "windowed", "measurement_window": (0.2, 0.7)},
        samples=[],
    )
    # Keeps the sampler total and reports total methodology (window not applied).
    assert methodology.measurement_methodology == "total"
    assert result.total_energy_j == pytest.approx(100.0)


def test_inference_output_tracks_input_output_tokens() -> None:
    """InferenceOutput separates input_tokens and output_tokens."""
    from llenergymeasure.engines.protocol import InferenceOutput

    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=50,
        output_tokens=30,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )
    assert output.input_tokens == 50
    assert output.output_tokens == 30
    assert output.total_tokens == 80


# =============================================================================
# _build_result() field wiring tests (CM-16, RES-16, RES-06)
# =============================================================================


def _make_build_result_args():
    """Shared helper: return kwargs for MeasurementHarness._build_result() with minimal data."""
    from datetime import datetime

    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.domain.metrics import FlopsResult, ThermalThrottleInfo
    from llenergymeasure.energy.nvml import EnergyMeasurement
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "gpt2"})
    output = InferenceOutput(
        elapsed_time_sec=10.0,
        input_tokens=50,
        output_tokens=50,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )
    energy_measurement = EnergyMeasurement(total_j=100.0, duration_sec=10.0)
    flops_result = FlopsResult(
        value=1e12, method="palm_formula", confidence="medium", precision="n/a"
    )
    now = datetime(2026, 1, 1, 12, 0, 0)
    return dict(
        engine_name="transformers",
        config=config,
        output=output,
        model_memory_mb=0.0,
        start_time=now,
        end_time=now,
        duration_sec=10.0,
        thermal_info=ThermalThrottleInfo(),
        energy_measurement=energy_measurement,
        baseline=None,
        flops_result=flops_result,
        timeseries_path=None,
        measurement_warnings=[],
    )


def test_harness_build_result_populates_timeseries_field() -> None:
    """_build_result() with timeseries_path='timeseries.parquet' sets result.timeseries (CM-16)."""

    kwargs = _make_build_result_args()
    kwargs["timeseries_path"] = "timeseries.parquet"

    result, _ = build_result(**kwargs)

    assert result.timeseries == "timeseries.parquet", (
        "timeseries field should be populated from timeseries_path argument"
    )


def test_harness_build_result_propagates_baseline_fields() -> None:
    """_build_result() with a baseline populates baseline_power_w and energy_adjusted_j (RES-06)."""
    from llenergymeasure.harness.baseline import BaselineCache

    kwargs = _make_build_result_args()
    kwargs["baseline"] = BaselineCache(
        power_w=30.0,
        timestamp=0.0,
        gpu_indices=[0],
        sample_count=300,
        duration_sec=30.0,
    )

    result, _ = build_result(**kwargs)

    assert result.baseline_power_w == pytest.approx(30.0), (
        "baseline_power_w should be populated from EnergyBreakdown.baseline_power_w"
    )
    assert result.energy_adjusted_j is not None, (
        "energy_adjusted_j should be populated when baseline is provided"
    )


# =============================================================================
# D1: per-token energy headline divides by OUTPUT tokens only
# =============================================================================


def test_mj_per_tok_uses_output_tokens_only() -> None:
    """mj_per_tok_total divides energy by OUTPUT tokens, not input+output (D1)."""
    from llenergymeasure.engines.protocol import InferenceOutput

    kwargs = _make_build_result_args()
    # Asymmetric: 900 input, 100 output. total=1000. energy=100 J.
    kwargs["output"] = InferenceOutput(
        elapsed_time_sec=10.0,
        input_tokens=900,
        output_tokens=100,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )

    result, _ = build_result(**kwargs)

    # 100 J / 100 output tokens * 1000 = 1000.0 mJ/output-token.
    # Old (buggy) denominator total_tokens=1000 would give 100.0.
    assert result.mj_per_tok_total == pytest.approx(1000.0)
    # Consistent with the existing output-token-only headline.
    assert result.avg_energy_per_token_j == pytest.approx(100.0 / 100)


# =============================================================================
# H1: FLOPs estimator tries the hf_model (actual params) path FIRST
# =============================================================================


def test_estimate_flops_prefers_hf_model_over_autoconfig() -> None:
    """_estimate_flops uses the actual-param hf_model path before AutoConfig (H1)."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.protocol import InferenceOutput

    class _StubParam:
        def __init__(self, n: int) -> None:
            self._n = n

        def numel(self) -> int:
            return self._n

    class _StubModel:
        def named_parameters(self):
            # Non-embedding param count that is distinctive and would never
            # coincide with the AutoConfig estimate for the same model.
            return iter([("decoder.layer.weight", _StubParam(777))])

    config = ExperimentConfig(task={"model": "gpt2"})
    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=10,
        output_tokens=5,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
        extras={"hf_model": _StubModel()},
    )

    result = estimate_flops(harness_engine := MagicMock(), config, output)
    del harness_engine

    # PaLM: 2 * 777 * (10 + 5) = 23310. High confidence proves the hf_model
    # path ran (AutoConfig would be 'medium' and a different value).
    assert result is not None
    assert result.confidence == "high"
    assert result.value == float(2 * 777 * (10 + 5))


def test_estimate_flops_falls_back_to_autoconfig_without_model() -> None:
    """_estimate_flops falls back to AutoConfig when no hf_model is present (H1)."""
    from llenergymeasure.config.models import ExperimentConfig
    from llenergymeasure.engines.protocol import InferenceOutput

    config = ExperimentConfig(task={"model": "gpt2"})
    output = InferenceOutput(
        elapsed_time_sec=1.0,
        input_tokens=10,
        output_tokens=5,
        peak_memory_mb=0.0,
        model_memory_mb=0.0,
    )

    fake = MagicMock(value=5e11, confidence="medium")
    with patch(
        "llenergymeasure.harness.window.estimate_flops_palm_from_config",
        return_value=fake,
    ) as mock_cfg:
        result = estimate_flops(MagicMock(), config, output)

    mock_cfg.assert_called_once()
    assert result is fake


# =============================================================================
# H6: a bad latency-measurement-mode string must not crash result assembly
# =============================================================================


def test_resolve_measurement_mode_guards_bad_string() -> None:
    """An unrecognised mode string falls back to TRUE_STREAMING with a warning (H6)."""
    from llenergymeasure.domain.metrics import LatencyMeasurementMode

    warnings: list[str] = []
    mode = resolve_measurement_mode("not_a_real_mode", warnings)

    assert mode is LatencyMeasurementMode.TRUE_STREAMING
    assert any("not_a_real_mode" in w for w in warnings)


def test_resolve_measurement_mode_accepts_valid_string() -> None:
    """A valid mode string maps to its enum member (H6 regression guard)."""
    from llenergymeasure.domain.metrics import LatencyMeasurementMode

    warnings: list[str] = []
    mode = resolve_measurement_mode("proportional", warnings)

    assert mode is LatencyMeasurementMode.PROPORTIONAL_ESTIMATE
    assert warnings == []


# =============================================================================
# H5: warmup_excluded_samples counts the discarded probe inference
# =============================================================================


def test_warmup_excluded_samples_includes_probe() -> None:
    """_run_warmup counts the discarded strategy-probe inference (H5).

    Fixed mode runs n_prompts loop inferences; the harness also runs one probe
    inference up front to pick the warmup strategy. That probe is discarded but
    must still be counted, so iterations_completed == n_prompts + 1.
    """
    from llenergymeasure.config.models import ExperimentConfig

    n_prompts = 3
    config = ExperimentConfig(
        task={"model": "gpt2"},
        measurement={
            "warmup": {
                "enabled": True,
                "convergence_detection": False,
                "n_prompts": n_prompts,
                "thermal_floor_seconds": 30.0,
            }
        },
    )

    engine = MagicMock()
    # Positive latency -> CV/fixed branch (the one with the extra probe).
    engine.run_warmup_prompt.return_value = 12.5

    with patch("llenergymeasure.harness.lifecycle.thermal_floor_wait", return_value=0.0):
        warmup_result = run_warmup(engine, config, MagicMock(), ["p"], None)

    # 1 probe + n_prompts loop inferences all ran and were discarded.
    assert engine.run_warmup_prompt.call_count == n_prompts + 1
    assert warmup_result.iterations_completed == n_prompts + 1
