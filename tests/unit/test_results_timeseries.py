"""Tests for time-series export and load functionality.

Tests export_timeseries and load_timeseries round-trip, empty samples,
summary statistics, compact keys, and relative timestamps.
"""

from pathlib import Path

import pytest

from llenergymeasure.core.power_thermal import PowerThermalSample
from llenergymeasure.results.timeseries import export_timeseries, load_timeseries


def _make_samples(count: int = 10, base_time: float = 1000.0) -> list[PowerThermalSample]:
    """Create a list of mock PowerThermalSample instances."""
    samples = []
    for i in range(count):
        samples.append(
            PowerThermalSample(
                timestamp=base_time + i * 0.1,
                power_w=250.0 + i * 2.0,
                memory_used_mb=4096.0 + i * 10.0,
                memory_total_mb=81920.0,
                temperature_c=70.0 + i * 0.5,
                sm_utilisation=80.0 + i,
                thermal_throttle=i >= 8,  # last 2 samples throttled
                throttle_reasons=0x40 if i >= 8 else 0,
            )
        )
    return samples


class TestExportAndLoad:
    """Tests for round-trip export/load of time-series data."""

    def test_export_and_load_round_trip(self, tmp_path: Path):
        samples = _make_samples(5)
        output_path = export_timeseries(
            samples=samples,
            experiment_id="exp_001",
            process_index=0,
            output_dir=tmp_path,
            sample_interval_ms=100,
        )
        assert output_path.exists()
        assert output_path.name == "process_0_timeseries.json"

        # Load back
        data = load_timeseries(output_path)
        assert data["experiment_id"] == "exp_001"
        assert data["process_index"] == 0
        assert data["sample_count"] == 5
        assert data["sample_interval_ms"] == 100
        assert len(data["samples"]) == 5

    def test_load_preserves_values(self, tmp_path: Path):
        samples = _make_samples(3)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_002",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)

        # First sample
        s0 = data["samples"][0]
        assert s0["power_w"] == pytest.approx(250.0, abs=0.1)
        assert s0["mem_mb"] == pytest.approx(4096.0, abs=1.0)
        assert s0["temp_c"] == pytest.approx(70.0)
        assert s0["throttle"] is False

    def test_load_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_timeseries(tmp_path / "nonexistent.json")


class TestExportEmptySamples:
    """Tests for exporting with no samples."""

    def test_empty_samples_creates_file(self, tmp_path: Path):
        path = export_timeseries(
            samples=[],
            experiment_id="exp_empty",
            process_index=0,
            output_dir=tmp_path,
        )
        assert path.exists()

        data = load_timeseries(path)
        assert data["sample_count"] == 0
        assert data["samples"] == []
        assert data["summary"] == {}
        assert data["duration_sec"] == 0.0


class TestSummaryStatistics:
    """Tests for summary statistics in the exported data."""

    def test_summary_has_power_stats(self, tmp_path: Path):
        samples = _make_samples(10)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_summary",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)
        summary = data["summary"]

        assert "power_mean_w" in summary
        assert "power_min_w" in summary
        assert "power_max_w" in summary
        assert summary["power_min_w"] <= summary["power_mean_w"] <= summary["power_max_w"]

    def test_summary_has_memory_stats(self, tmp_path: Path):
        samples = _make_samples(10)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_mem",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)
        summary = data["summary"]

        assert "memory_max_mb" in summary
        assert "memory_mean_mb" in summary

    def test_summary_has_temperature_stats(self, tmp_path: Path):
        samples = _make_samples(10)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_temp",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)
        summary = data["summary"]

        assert "temperature_mean_c" in summary
        assert "temperature_max_c" in summary

    def test_summary_thermal_throttle(self, tmp_path: Path):
        samples = _make_samples(10)  # last 2 are throttled
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_throttle",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)
        summary = data["summary"]

        assert summary["thermal_throttle_detected"] is True
        assert summary["thermal_throttle_sample_count"] == 2


class TestCompactKeys:
    """Tests for compact sample key names."""

    def test_compact_keys_used(self, tmp_path: Path):
        samples = _make_samples(1)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_keys",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)
        sample = data["samples"][0]

        expected_keys = {"t", "power_w", "mem_mb", "temp_c", "sm_pct", "throttle"}
        assert set(sample.keys()) == expected_keys


class TestRelativeTimestamps:
    """Tests that timestamps are relative (first sample at t=0)."""

    def test_first_sample_at_zero(self, tmp_path: Path):
        samples = _make_samples(5, base_time=1000.0)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_rel",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)

        assert data["samples"][0]["t"] == pytest.approx(0.0)

    def test_relative_intervals(self, tmp_path: Path):
        samples = _make_samples(5, base_time=1000.0)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_intervals",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)

        # Samples are 0.1s apart
        for i, s in enumerate(data["samples"]):
            assert s["t"] == pytest.approx(i * 0.1, abs=0.001)

    def test_duration_matches_span(self, tmp_path: Path):
        samples = _make_samples(10, base_time=500.0)
        path = export_timeseries(
            samples=samples,
            experiment_id="exp_dur",
            process_index=0,
            output_dir=tmp_path,
        )
        data = load_timeseries(path)

        # Duration should be (last - first) timestamp
        expected_duration = 9 * 0.1  # 10 samples, 0.1s apart
        assert data["duration_sec"] == pytest.approx(expected_duration, abs=0.001)
