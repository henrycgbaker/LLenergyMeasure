"""Tests for extended CSV export functionality (schema v3).

Tests the v3 columns in aggregated CSV export: energy breakdown,
thermal throttling, and environment metadata columns.
"""

import csv
from datetime import datetime
from pathlib import Path

from llenergymeasure.domain.environment import (
    ContainerEnvironment,
    CPUEnvironment,
    CUDAEnvironment,
    EnvironmentMetadata,
    GPUEnvironment,
    ThermalEnvironment,
)
from llenergymeasure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
)
from llenergymeasure.domain.metrics import (
    EnergyBreakdown,
    ThermalThrottleInfo,
)
from llenergymeasure.results.exporters import (
    export_aggregated_to_csv,
)


def _make_aggregated_v3(
    experiment_id: str = "exp_v3",
    energy_breakdown: EnergyBreakdown | None = None,
    thermal_throttle: ThermalThrottleInfo | None = None,
    environment: EnvironmentMetadata | None = None,
) -> AggregatedResult:
    """Create an AggregatedResult with optional v3 fields."""
    return AggregatedResult(
        experiment_id=experiment_id,
        aggregation=AggregationMetadata(
            method="sum_energy_avg_throughput",
            num_processes=1,
            temporal_overlap_verified=True,
            gpu_attribution_verified=True,
            warnings=[],
        ),
        total_tokens=1000,
        total_energy_j=100.0,
        total_inference_time_sec=10.0,
        avg_tokens_per_second=100.0,
        avg_energy_per_token_j=0.1,
        total_flops=1e12,
        process_results=[],
        start_time=datetime(2024, 1, 1, 12, 0, 0),
        end_time=datetime(2024, 1, 1, 12, 0, 10),
        energy_breakdown=energy_breakdown,
        thermal_throttle=thermal_throttle,
        environment=environment,
    )


def _make_environment() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        gpu=GPUEnvironment(name="NVIDIA A100-SXM4-80GB", vram_total_mb=81920.0),
        cuda=CUDAEnvironment(version="12.4", driver_version="535.104"),
        thermal=ThermalEnvironment(temperature_c=42.0, power_limit_w=400.0),
        cpu=CPUEnvironment(governor="performance", platform="Linux"),
        container=ContainerEnvironment(detected=False),
        collected_at=datetime(2024, 1, 1, 12, 0, 0),
    )


class TestAggregatedRowV3Fields:
    """Tests that v3 fields appear in the aggregated row dict."""

    def test_energy_breakdown_columns(self, tmp_path: Path):
        result = _make_aggregated_v3(
            energy_breakdown=EnergyBreakdown(
                raw_j=100.0,
                adjusted_j=50.0,
                baseline_power_w=10.0,
                baseline_method="fresh",
            ),
        )
        output_path = tmp_path / "results.csv"
        export_aggregated_to_csv([result], output_path)

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["energy_adjusted_j"] == "50.0"
        assert rows[0]["energy_baseline_w"] == "10.0"
        assert rows[0]["energy_baseline_method"] == "fresh"

    def test_thermal_throttle_columns(self, tmp_path: Path):
        result = _make_aggregated_v3(
            thermal_throttle=ThermalThrottleInfo(
                detected=True,
                throttle_duration_sec=5.0,
                max_temperature_c=95.0,
            ),
        )
        output_path = tmp_path / "results.csv"
        export_aggregated_to_csv([result], output_path)

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["thermal_throttle_detected"] == "True"
        assert rows[0]["thermal_throttle_duration_sec"] == "5.0"
        assert rows[0]["thermal_max_temp_c"] == "95.0"

    def test_environment_columns(self, tmp_path: Path):
        result = _make_aggregated_v3(environment=_make_environment())
        output_path = tmp_path / "results.csv"
        export_aggregated_to_csv([result], output_path)

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert rows[0]["env_gpu_name"] == "NVIDIA A100-SXM4-80GB"
        assert rows[0]["env_gpu_vram_mb"] == "81920.0"
        assert rows[0]["env_cuda_version"] == "12.4"
        assert rows[0]["env_driver_version"] == "535.104"
        assert rows[0]["env_gpu_temp_c"] == "42.0"
        assert rows[0]["env_power_limit_w"] == "400.0"
        assert rows[0]["env_cpu_governor"] == "performance"
        assert rows[0]["env_in_container"] == "False"


class TestAggregatedRowBackwardsCompat:
    """Tests that results WITHOUT v3 fields still export correctly."""

    def test_no_v3_fields_no_crash(self, tmp_path: Path):
        """Result without v3 fields should not crash on export."""
        result = _make_aggregated_v3()  # no v3 fields
        output_path = tmp_path / "results.csv"
        export_aggregated_to_csv([result], output_path)

        assert output_path.exists()
        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        # v3 columns should exist but be None/empty
        assert rows[0]["energy_adjusted_j"] == ""
        assert rows[0]["energy_baseline_w"] == ""
        assert rows[0]["thermal_throttle_detected"] == "False"
        assert rows[0]["env_gpu_name"] == ""


class TestColumnOrdering:
    """Tests that v3 columns are grouped in the correct order."""

    def test_energy_thermal_env_columns_grouped(self, tmp_path: Path):
        result = _make_aggregated_v3(
            energy_breakdown=EnergyBreakdown(raw_j=100.0, adjusted_j=50.0),
            thermal_throttle=ThermalThrottleInfo(detected=True),
            environment=_make_environment(),
        )
        output_path = tmp_path / "results.csv"
        export_aggregated_to_csv([result], output_path)

        with output_path.open() as f:
            reader = csv.reader(f)
            headers = next(reader)

        # Verify grouping order: energy_* before thermal_* before env_*
        energy_idx = headers.index("energy_adjusted_j")
        thermal_idx = headers.index("thermal_throttle_detected")
        env_idx = headers.index("env_gpu_name")

        assert energy_idx < thermal_idx, "energy_ columns should come before thermal_"
        assert thermal_idx < env_idx, "thermal_ columns should come before env_"

    def test_experiment_id_is_first(self, tmp_path: Path):
        result = _make_aggregated_v3()
        output_path = tmp_path / "results.csv"
        export_aggregated_to_csv([result], output_path)

        with output_path.open() as f:
            reader = csv.reader(f)
            headers = next(reader)

        assert headers[0] == "experiment_id"


class TestCsvWriteWithV3Fields:
    """Full CSV write integration test with v3 fields."""

    def test_full_csv_write(self, tmp_path: Path):
        result = _make_aggregated_v3(
            energy_breakdown=EnergyBreakdown(
                raw_j=100.0,
                adjusted_j=50.0,
                baseline_power_w=10.0,
                baseline_method="fresh",
            ),
            thermal_throttle=ThermalThrottleInfo(
                detected=True,
                thermal=True,
                throttle_duration_sec=2.5,
                max_temperature_c=90.0,
            ),
            environment=_make_environment(),
        )
        output_path = tmp_path / "full_v3.csv"
        export_aggregated_to_csv([result], output_path)

        assert output_path.exists()
        content = output_path.read_text()

        # Verify key headers present
        assert "energy_adjusted_j" in content
        assert "energy_baseline_w" in content
        assert "energy_baseline_method" in content
        assert "thermal_throttle_detected" in content
        assert "thermal_max_temp_c" in content
        assert "env_gpu_name" in content
        assert "env_cuda_version" in content
        assert "env_summary" in content

        # Verify values present
        assert "50.0" in content  # adjusted energy
        assert "fresh" in content  # baseline method
        assert "True" in content  # throttle detected
        assert "NVIDIA A100" in content  # GPU name
