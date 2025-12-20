"""Tests for results export functionality."""

import csv
import json
from datetime import datetime
from pathlib import Path

from llm_bench.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
    Timestamps,
)
from llm_bench.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llm_bench.results.exporters import (
    ResultsExporter,
    export_aggregated_to_csv,
    export_raw_to_csv,
    flatten_model,
)


def make_aggregated_result(
    experiment_id: str = "exp_001",
    total_tokens: int = 1000,
    total_energy_j: float = 100.0,
) -> AggregatedResult:
    """Create an AggregatedResult for testing."""
    return AggregatedResult(
        experiment_id=experiment_id,
        aggregation=AggregationMetadata(
            method="sum_energy_avg_throughput",
            num_processes=2,
            temporal_overlap_verified=True,
            gpu_attribution_verified=True,
            warnings=[],
        ),
        total_tokens=total_tokens,
        total_energy_j=total_energy_j,
        total_inference_time_sec=10.0,
        avg_tokens_per_second=100.0,
        avg_energy_per_token_j=total_energy_j / total_tokens,
        total_flops=1e12,
        process_results=[],
        start_time=datetime(2024, 1, 1, 12, 0, 0),
        end_time=datetime(2024, 1, 1, 12, 0, 10),
    )


def make_raw_result(
    process_index: int = 0,
    gpu_id: int = 0,
) -> RawProcessResult:
    """Create a RawProcessResult for testing."""
    return RawProcessResult(
        experiment_id="exp_001",
        process_index=process_index,
        gpu_id=gpu_id,
        timestamps=Timestamps(
            start=datetime(2024, 1, 1, 12, 0, 0),
            end=datetime(2024, 1, 1, 12, 0, 10),
            duration_sec=10.0,
        ),
        inference_metrics=InferenceMetrics(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            inference_time_sec=10.0,
            tokens_per_second=10.0,
            latency_per_token_ms=100.0,
        ),
        energy_metrics=EnergyMetrics(
            total_energy_j=50.0,
            gpu_energy_j=40.0,
            cpu_energy_j=8.0,
            ram_energy_j=2.0,
            gpu_power_w=100.0,
            cpu_power_w=50.0,
            duration_sec=10.0,
            emissions_kg_co2=0.001,
            energy_per_token_j=0.5,
        ),
        compute_metrics=ComputeMetrics(
            flops_total=5e11,
            flops_per_token=5e9,
            flops_per_second=5e10,
            peak_memory_mb=2048.0,
            model_memory_mb=1024.0,
            flops_method="ptflops",
            flops_confidence="medium",
            compute_precision="fp16",
        ),
    )


class TestFlattenModel:
    """Tests for flatten_model function."""

    def test_flatten_simple_dict(self) -> None:
        data = {"a": 1, "b": 2}
        result = flatten_model(data)

        assert result == {"a": 1, "b": 2}

    def test_flatten_nested_dict(self) -> None:
        data = {"outer": {"inner": 42}}
        result = flatten_model(data)

        assert result == {"outer_inner": 42}

    def test_flatten_list(self) -> None:
        data = {"items": [1, 2, 3]}
        result = flatten_model(data)

        assert result["items_0"] == 1
        assert result["items_1"] == 2
        assert result["items_2"] == 3

    def test_flatten_pydantic_model(self) -> None:
        metrics = InferenceMetrics(
            total_tokens=100,
            input_tokens=50,
            output_tokens=50,
            inference_time_sec=1.0,
            tokens_per_second=100.0,
            latency_per_token_ms=10.0,
        )
        result = flatten_model(metrics)

        assert result["total_tokens"] == 100
        assert result["tokens_per_second"] == 100.0


class TestExportAggregatedToCsv:
    """Tests for export_aggregated_to_csv function."""

    def test_export_single_result(self, tmp_path: Path) -> None:
        result = make_aggregated_result("exp_001")
        output_path = tmp_path / "results.csv"

        export_aggregated_to_csv([result], output_path)

        assert output_path.exists()

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp_001"
        assert rows[0]["total_tokens"] == "1000"

    def test_export_multiple_results(self, tmp_path: Path) -> None:
        results = [
            make_aggregated_result("exp_001", total_tokens=1000),
            make_aggregated_result("exp_002", total_tokens=2000),
        ]
        output_path = tmp_path / "results.csv"

        export_aggregated_to_csv(results, output_path)

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["experiment_id"] == "exp_001"
        assert rows[1]["experiment_id"] == "exp_002"

    def test_export_empty_returns_path(self, tmp_path: Path) -> None:
        output_path = tmp_path / "results.csv"

        result_path = export_aggregated_to_csv([], output_path)

        assert result_path == output_path

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        output_path = tmp_path / "subdir" / "deep" / "results.csv"
        result = make_aggregated_result()

        export_aggregated_to_csv([result], output_path)

        assert output_path.exists()

    def test_column_ordering(self, tmp_path: Path) -> None:
        result = make_aggregated_result()
        output_path = tmp_path / "results.csv"

        export_aggregated_to_csv([result], output_path)

        with output_path.open() as f:
            reader = csv.reader(f)
            headers = next(reader)

        # experiment_id should be first
        assert headers[0] == "experiment_id"


class TestExportRawToCsv:
    """Tests for export_raw_to_csv function."""

    def test_export_single_raw_result(self, tmp_path: Path) -> None:
        result = make_raw_result(0, 0)
        output_path = tmp_path / "raw_results.csv"

        export_raw_to_csv([result], output_path)

        assert output_path.exists()

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp_001"

    def test_export_multiple_raw_results(self, tmp_path: Path) -> None:
        results = [make_raw_result(0, 0), make_raw_result(1, 1)]
        output_path = tmp_path / "raw_results.csv"

        export_raw_to_csv(results, output_path)

        with output_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2


class TestResultsExporter:
    """Tests for ResultsExporter class."""

    def test_init_creates_output_dir(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "exports"
        _exporter = ResultsExporter(output_dir)

        assert output_dir.exists()

    def test_export_aggregated(self, tmp_path: Path) -> None:
        exporter = ResultsExporter(tmp_path)
        result = make_aggregated_result()

        path = exporter.export_aggregated([result], "test.csv")

        assert path == tmp_path / "test.csv"
        assert path.exists()

    def test_export_raw(self, tmp_path: Path) -> None:
        exporter = ResultsExporter(tmp_path)
        result = make_raw_result()

        path = exporter.export_raw([result], "raw.csv")

        assert path == tmp_path / "raw.csv"
        assert path.exists()

    def test_export_json(self, tmp_path: Path) -> None:
        exporter = ResultsExporter(tmp_path)
        result = make_aggregated_result()

        path = exporter.export_json([result], "results.json")

        assert path == tmp_path / "results.json"
        assert path.exists()

        with path.open() as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["experiment_id"] == "exp_001"
