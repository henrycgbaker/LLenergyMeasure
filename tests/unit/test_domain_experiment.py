"""Tests for domain experiment models."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from llm_energy_measure.constants import SCHEMA_VERSION
from llm_energy_measure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
    Timestamps,
)
from llm_energy_measure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics


class TestTimestamps:
    """Tests for Timestamps model."""

    def test_create_directly(self):
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 10)
        ts = Timestamps(start=start, end=end, duration_sec=10.0)
        assert ts.duration_sec == 10.0

    def test_from_times_factory(self):
        start = datetime(2024, 1, 1, 12, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 30)
        ts = Timestamps.from_times(start, end)
        assert ts.start == start
        assert ts.end == end
        assert ts.duration_sec == 30.0

    def test_from_times_subsecond(self):
        start = datetime(2024, 1, 1, 12, 0, 0, 0)
        end = datetime(2024, 1, 1, 12, 0, 0, 500000)  # 0.5 seconds
        ts = Timestamps.from_times(start, end)
        assert ts.duration_sec == pytest.approx(0.5)


class TestRawProcessResult:
    """Tests for RawProcessResult model."""

    @pytest.fixture
    def sample_raw_result(self):
        return RawProcessResult(
            experiment_id="exp_001",
            process_index=0,
            gpu_id=0,
            config_name="test_config",
            model_name="test/model",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 12, 0, 0),
                end=datetime(2024, 1, 1, 12, 0, 10),
                duration_sec=10.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=500,
                input_tokens=50,
                output_tokens=450,
                inference_time_sec=10.0,
                tokens_per_second=50.0,
                latency_per_token_ms=20.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=250.0,
                gpu_power_w=100.0,
                duration_sec=10.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=5e11,
            ),
        )

    def test_schema_version_default(self, sample_raw_result):
        assert sample_raw_result.schema_version == SCHEMA_VERSION

    def test_all_fields_accessible(self, sample_raw_result):
        assert sample_raw_result.experiment_id == "exp_001"
        assert sample_raw_result.process_index == 0
        assert sample_raw_result.gpu_id == 0
        assert sample_raw_result.inference_metrics.total_tokens == 500

    def test_frozen_model(self, sample_raw_result):
        with pytest.raises(ValidationError):  # Pydantic frozen model
            sample_raw_result.experiment_id = "changed"

    def test_serialization_roundtrip(self, sample_raw_result):
        json_str = sample_raw_result.model_dump_json()
        restored = RawProcessResult.model_validate_json(json_str)
        assert restored.experiment_id == sample_raw_result.experiment_id
        assert restored.inference_metrics.total_tokens == 500

    def test_effective_config_default_empty(self, sample_raw_result):
        """effective_config defaults to empty dict."""
        assert sample_raw_result.effective_config == {}

    def test_cli_overrides_default_empty(self, sample_raw_result):
        """cli_overrides defaults to empty dict."""
        assert sample_raw_result.cli_overrides == {}

    def test_effective_config_field_exists(self):
        """effective_config field is present on model."""
        assert "effective_config" in RawProcessResult.model_fields

    def test_cli_overrides_field_exists(self):
        """cli_overrides field is present on model."""
        assert "cli_overrides" in RawProcessResult.model_fields

    def test_effective_config_accepts_dict(self):
        """effective_config accepts dictionary values."""
        result = RawProcessResult(
            experiment_id="exp_002",
            process_index=0,
            gpu_id=0,
            config_name="test_config",
            model_name="test/model",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 12, 0, 0),
                end=datetime(2024, 1, 1, 12, 0, 10),
                duration_sec=10.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=500,
                input_tokens=50,
                output_tokens=450,
                inference_time_sec=10.0,
                tokens_per_second=50.0,
                latency_per_token_ms=20.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=250.0,
                gpu_power_w=100.0,
                duration_sec=10.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=5e11,
            ),
            effective_config={
                "model_name": "test/model",
                "batch_size": 4,
                "fp_precision": "float16",
            },
        )
        assert result.effective_config["model_name"] == "test/model"
        assert result.effective_config["batch_size"] == 4

    def test_cli_overrides_accepts_dict(self):
        """cli_overrides accepts dictionary values."""
        result = RawProcessResult(
            experiment_id="exp_003",
            process_index=0,
            gpu_id=0,
            config_name="test_config",
            model_name="test/model",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 12, 0, 0),
                end=datetime(2024, 1, 1, 12, 0, 10),
                duration_sec=10.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=500,
                input_tokens=50,
                output_tokens=450,
                inference_time_sec=10.0,
                tokens_per_second=50.0,
                latency_per_token_ms=20.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=250.0,
                gpu_power_w=100.0,
                duration_sec=10.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=5e11,
            ),
            cli_overrides={
                "batch_size": {"new": 8, "original": 1},
                "fp_precision": {"new": "float16", "original": "float32"},
            },
        )
        assert result.cli_overrides["batch_size"]["new"] == 8
        assert result.cli_overrides["fp_precision"]["original"] == "float32"

    def test_effective_config_serialization_roundtrip(self):
        """effective_config survives JSON serialization."""
        result = RawProcessResult(
            experiment_id="exp_004",
            process_index=0,
            gpu_id=0,
            config_name="test_config",
            model_name="test/model",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 12, 0, 0),
                end=datetime(2024, 1, 1, 12, 0, 10),
                duration_sec=10.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=500,
                input_tokens=50,
                output_tokens=450,
                inference_time_sec=10.0,
                tokens_per_second=50.0,
                latency_per_token_ms=20.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=250.0,
                gpu_power_w=100.0,
                duration_sec=10.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=5e11,
            ),
            effective_config={"key": "value", "nested": {"a": 1}},
            cli_overrides={"override": {"new": 10, "original": 5}},
        )
        json_str = result.model_dump_json()
        restored = RawProcessResult.model_validate_json(json_str)
        assert restored.effective_config == {"key": "value", "nested": {"a": 1}}
        assert restored.cli_overrides == {"override": {"new": 10, "original": 5}}


class TestAggregationMetadata:
    """Tests for AggregationMetadata model."""

    def test_defaults(self):
        meta = AggregationMetadata(num_processes=4)
        assert meta.method == "sum_energy_avg_throughput"
        assert meta.temporal_overlap_verified is False
        assert meta.warnings == []

    def test_with_warnings(self):
        meta = AggregationMetadata(
            num_processes=4,
            warnings=["GPU 0 had low utilization", "Temporal overlap incomplete"],
        )
        assert len(meta.warnings) == 2


class TestAggregatedResult:
    """Tests for AggregatedResult model."""

    @pytest.fixture
    def sample_aggregated(self):
        return AggregatedResult(
            experiment_id="exp_001",
            aggregation=AggregationMetadata(
                num_processes=2,
                temporal_overlap_verified=True,
                gpu_attribution_verified=True,
            ),
            total_tokens=1000,
            total_energy_j=500.0,
            total_inference_time_sec=10.0,
            avg_tokens_per_second=100.0,
            avg_energy_per_token_j=0.5,
            total_flops=1e12,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 10),
        )

    def test_schema_version(self, sample_aggregated):
        assert sample_aggregated.schema_version == SCHEMA_VERSION

    def test_duration_sec_property(self, sample_aggregated):
        assert sample_aggregated.duration_sec == 10.0

    def test_tokens_per_joule_property(self, sample_aggregated):
        # 1000 tokens / 500 J = 2 tokens/J
        assert sample_aggregated.tokens_per_joule == 2.0

    def test_tokens_per_joule_zero_energy(self):
        result = AggregatedResult(
            experiment_id="exp_002",
            aggregation=AggregationMetadata(num_processes=1),
            total_tokens=100,
            total_energy_j=0.0,
            total_inference_time_sec=1.0,
            avg_tokens_per_second=100.0,
            avg_energy_per_token_j=0.0,
            total_flops=1e10,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 1),
        )
        assert result.tokens_per_joule == 0.0

    def test_frozen_model(self, sample_aggregated):
        with pytest.raises(ValidationError):  # Pydantic frozen model
            sample_aggregated.experiment_id = "changed"

    def test_effective_config_default_empty(self, sample_aggregated):
        """effective_config defaults to empty dict."""
        assert sample_aggregated.effective_config == {}

    def test_cli_overrides_default_empty(self, sample_aggregated):
        """cli_overrides defaults to empty dict."""
        assert sample_aggregated.cli_overrides == {}

    def test_effective_config_field_exists(self):
        """effective_config field is present on AggregatedResult."""
        assert "effective_config" in AggregatedResult.model_fields

    def test_cli_overrides_field_exists(self):
        """cli_overrides field is present on AggregatedResult."""
        assert "cli_overrides" in AggregatedResult.model_fields

    def test_effective_config_accepts_dict(self):
        """effective_config accepts dictionary values on AggregatedResult."""
        result = AggregatedResult(
            experiment_id="exp_with_config",
            aggregation=AggregationMetadata(num_processes=1),
            total_tokens=100,
            total_energy_j=50.0,
            total_inference_time_sec=1.0,
            avg_tokens_per_second=100.0,
            avg_energy_per_token_j=0.5,
            total_flops=1e10,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 1),
            effective_config={
                "model_name": "test/model",
                "batch_size": 4,
                "preset": "benchmark",
            },
        )
        assert result.effective_config["model_name"] == "test/model"
        assert result.effective_config["batch_size"] == 4
        assert result.effective_config["preset"] == "benchmark"

    def test_cli_overrides_accepts_dict(self):
        """cli_overrides accepts dictionary values on AggregatedResult."""
        result = AggregatedResult(
            experiment_id="exp_with_overrides",
            aggregation=AggregationMetadata(num_processes=1),
            total_tokens=100,
            total_energy_j=50.0,
            total_inference_time_sec=1.0,
            avg_tokens_per_second=100.0,
            avg_energy_per_token_j=0.5,
            total_flops=1e10,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 1),
            cli_overrides={
                "batch_size": {"new": 16, "original": 1},
                "max_tokens": {"new": 512, "original": 256},
            },
        )
        assert result.cli_overrides["batch_size"]["new"] == 16
        assert result.cli_overrides["max_tokens"]["original"] == 256

    def test_effective_config_serialization_roundtrip(self):
        """effective_config survives JSON serialization on AggregatedResult."""
        result = AggregatedResult(
            experiment_id="exp_serialize",
            aggregation=AggregationMetadata(num_processes=1),
            total_tokens=100,
            total_energy_j=50.0,
            total_inference_time_sec=1.0,
            avg_tokens_per_second=100.0,
            avg_energy_per_token_j=0.5,
            total_flops=1e10,
            start_time=datetime(2024, 1, 1, 12, 0, 0),
            end_time=datetime(2024, 1, 1, 12, 0, 1),
            effective_config={"config_name": "test", "nested": {"key": "value"}},
            cli_overrides={"override": {"new": 42, "original": 21}},
        )
        json_str = result.model_dump_json()
        restored = AggregatedResult.model_validate_json(json_str)
        assert restored.effective_config == {"config_name": "test", "nested": {"key": "value"}}
        assert restored.cli_overrides == {"override": {"new": 42, "original": 21}}
