"""Integration tests for the config → aggregation → export pipeline.

Tests the full flow from loading config through aggregation to export,
without requiring GPU access.
"""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from llm_energy_measure.config.loader import load_config
from llm_energy_measure.domain.experiment import (
    RawProcessResult,
    Timestamps,
)
from llm_energy_measure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llm_energy_measure.results.aggregation import aggregate_results, calculate_efficiency_metrics
from llm_energy_measure.results.exporters import ResultsExporter
from llm_energy_measure.results.repository import FileSystemRepository


class TestConfigToAggregationPipeline:
    """End-to-end tests for config → aggregation → export."""

    @pytest.fixture
    def config_with_inheritance(self, tmp_path: Path) -> Path:
        """Create config files with inheritance."""
        # Base config
        base = tmp_path / "base.yaml"
        base.write_text("""
max_input_tokens: 1024
max_output_tokens: 256
decoder:
  temperature: 0.7
  top_p: 0.9
""")

        # Model config extending base
        model_config = tmp_path / "llama-7b.yaml"
        model_config.write_text("""
_extends: base.yaml
config_name: llama-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf
num_processes: 2
gpus: [0, 1]
""")
        return model_config

    @pytest.fixture
    def simulated_raw_results(self) -> list[RawProcessResult]:
        """Create simulated raw results from 2 processes."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        results = []

        for i in range(2):
            results.append(
                RawProcessResult(
                    experiment_id="exp_llama_001",
                    process_index=i,
                    gpu_id=i,
                    config_name="llama-7b-benchmark",
                    model_name="meta-llama/Llama-2-7b-hf",
                    timestamps=Timestamps(
                        start=base_time + timedelta(seconds=i * 0.5),
                        end=base_time + timedelta(seconds=60 + i * 0.5),
                        duration_sec=60.0,
                    ),
                    inference_metrics=InferenceMetrics(
                        total_tokens=1000 + i * 100,
                        input_tokens=500,
                        output_tokens=500 + i * 100,
                        inference_time_sec=55.0 + i * 2,
                        tokens_per_second=18.0 - i * 0.5,
                        latency_per_token_ms=55.0 + i * 2,
                    ),
                    energy_metrics=EnergyMetrics(
                        total_energy_j=150.0 + i * 10,
                        gpu_energy_j=120.0 + i * 8,
                        cpu_energy_j=30.0 + i * 2,
                        duration_sec=60.0,
                    ),
                    compute_metrics=ComputeMetrics(
                        flops_total=1.5e12 + i * 1e11,
                        flops_per_second=2.5e10,
                        flops_method="calflops",
                        flops_confidence="high",
                    ),
                )
            )
        return results

    def test_full_pipeline_config_to_export(
        self, tmp_path: Path, config_with_inheritance: Path, simulated_raw_results: list
    ):
        """Test full pipeline: load config → save raw → aggregate → export."""
        results_dir = tmp_path / "results"
        export_dir = tmp_path / "exports"

        # Step 1: Load config with inheritance
        config = load_config(config_with_inheritance)
        assert config.config_name == "llama-7b-benchmark"
        assert config.max_input_tokens == 1024  # Inherited
        assert config.decoder.temperature == 0.7  # Inherited

        # Step 2: Save simulated raw results
        repo = FileSystemRepository(results_dir)
        for result in simulated_raw_results:
            repo.save_raw("exp_llama_001", result)

        # Verify raw results saved
        assert repo.has_raw("exp_llama_001")
        assert len(repo.list_raw("exp_llama_001")) == 2

        # Step 3: Aggregate results
        loaded_raw = repo.load_all_raw("exp_llama_001")
        aggregated = aggregate_results("exp_llama_001", loaded_raw)

        # Verify aggregation
        assert aggregated.aggregation.num_processes == 2
        assert aggregated.total_tokens == 2100  # 1000 + 1100
        assert aggregated.total_energy_j == 310.0  # 150 + 160
        assert aggregated.aggregation.gpu_attribution_verified  # Unique GPU IDs

        # Step 4: Save aggregated result
        agg_path = repo.save_aggregated(aggregated)
        assert agg_path.exists()

        # Step 5: Export to CSV
        exporter = ResultsExporter(export_dir)
        csv_path = exporter.export_aggregated([aggregated], "benchmark_results.csv")
        assert csv_path.exists()

        # Verify CSV content
        csv_content = csv_path.read_text()
        assert "exp_llama_001" in csv_content
        assert "2100" in csv_content  # Total tokens

    def test_pipeline_with_efficiency_metrics(self, tmp_path: Path, simulated_raw_results: list):
        """Test that efficiency metrics are correctly calculated."""
        repo = FileSystemRepository(tmp_path)

        # Save and aggregate
        for result in simulated_raw_results:
            repo.save_raw("efficiency_test", result)

        raw = repo.load_all_raw("efficiency_test")
        aggregated = aggregate_results("efficiency_test", raw)

        # Calculate efficiency
        metrics = calculate_efficiency_metrics(aggregated)

        # Verify metrics
        assert metrics["tokens_per_joule"] > 0
        assert metrics["joules_per_token"] > 0
        assert metrics["tokens_per_second"] > 0
        assert metrics["effective_batch_throughput"] > metrics["tokens_per_second"]

        # Efficiency = total_tokens / total_energy
        expected_tpj = aggregated.total_tokens / aggregated.total_energy_j
        assert abs(metrics["tokens_per_joule"] - expected_tpj) < 0.001

    def test_pipeline_preserves_process_breakdown(
        self, tmp_path: Path, simulated_raw_results: list
    ):
        """Test that aggregation preserves per-process details."""
        repo = FileSystemRepository(tmp_path)

        for result in simulated_raw_results:
            repo.save_raw("breakdown_test", result)

        raw = repo.load_all_raw("breakdown_test")
        aggregated = aggregate_results("breakdown_test", raw)

        # Verify process results preserved
        assert len(aggregated.process_results) == 2
        assert aggregated.process_results[0].gpu_id == 0
        assert aggregated.process_results[1].gpu_id == 1

        # Can still access individual process metrics
        p0_tokens = aggregated.process_results[0].inference_metrics.total_tokens
        p1_tokens = aggregated.process_results[1].inference_metrics.total_tokens
        assert p0_tokens + p1_tokens == aggregated.total_tokens


class TestMultiLevelConfigInheritance:
    """Test complex config inheritance scenarios."""

    def test_three_level_inheritance(self, tmp_path: Path):
        """Test config inheritance through multiple levels."""
        # Level 1: Base defaults
        base = tmp_path / "defaults.yaml"
        base.write_text("""
max_input_tokens: 512
max_output_tokens: 128
decoder:
  temperature: 1.0
  top_p: 1.0
  do_sample: true
""")

        # Level 2: Model family defaults
        family = tmp_path / "llama-family.yaml"
        family.write_text("""
_extends: defaults.yaml
decoder:
  temperature: 0.7
fp_precision: float16
""")

        # Level 3: Specific model config
        specific = tmp_path / "llama-7b-4bit.yaml"
        specific.write_text("""
_extends: llama-family.yaml
config_name: llama-7b-4bit
model_name: meta-llama/Llama-2-7b-hf
quantization:
  quantization: true
  load_in_4bit: true
""")

        config = load_config(specific)

        # Level 1 inherited
        assert config.max_input_tokens == 512
        assert config.decoder.do_sample is True

        # Level 2 overrides
        assert config.decoder.temperature == 0.7
        assert config.fp_precision == "float16"

        # Level 3 specific
        assert config.config_name == "llama-7b-4bit"
        assert config.quantization.load_in_4bit is True


class TestAggregationValidation:
    """Test aggregation validation logic without GPU."""

    def test_temporal_overlap_detection(self, tmp_path: Path):
        """Test that non-overlapping processes generate warnings."""
        # Create non-overlapping results
        results = [
            RawProcessResult(
                experiment_id="overlap_test",
                process_index=0,
                gpu_id=0,
                config_name="test",
                model_name="test-model",
                timestamps=Timestamps(
                    start=datetime(2024, 1, 1, 10, 0, 0),
                    end=datetime(2024, 1, 1, 10, 1, 0),
                    duration_sec=60.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=100,
                    input_tokens=50,
                    output_tokens=50,
                    inference_time_sec=60.0,
                    tokens_per_second=1.67,
                    latency_per_token_ms=600.0,
                ),
                energy_metrics=EnergyMetrics(
                    total_energy_j=50.0,
                    duration_sec=60.0,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=1e10,
                    flops_method="parameter_estimate",
                    flops_confidence="low",
                ),
            ),
            RawProcessResult(
                experiment_id="overlap_test",
                process_index=1,
                gpu_id=1,
                config_name="test",
                model_name="test-model",
                timestamps=Timestamps(
                    start=datetime(2024, 1, 1, 10, 5, 0),  # 5 minutes later
                    end=datetime(2024, 1, 1, 10, 6, 0),
                    duration_sec=60.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=100,
                    input_tokens=50,
                    output_tokens=50,
                    inference_time_sec=60.0,
                    tokens_per_second=1.67,
                    latency_per_token_ms=600.0,
                ),
                energy_metrics=EnergyMetrics(
                    total_energy_j=50.0,
                    duration_sec=60.0,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=1e10,
                    flops_method="parameter_estimate",
                    flops_confidence="low",
                ),
            ),
        ]

        aggregated = aggregate_results("overlap_test", results)

        # Should have temporal overlap warning
        assert not aggregated.aggregation.temporal_overlap_verified
        assert any(
            "overlap" in w.lower() or "concurrent" in w.lower()
            for w in aggregated.aggregation.warnings
        )

    def test_duplicate_gpu_detection(self, tmp_path: Path):
        """Test that duplicate GPU IDs generate warnings."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Create results with same GPU ID
        results = []
        for i in range(2):
            results.append(
                RawProcessResult(
                    experiment_id="dup_gpu_test",
                    process_index=i,
                    gpu_id=0,  # Same GPU!
                    config_name="test",
                    model_name="test-model",
                    timestamps=Timestamps(
                        start=base_time,
                        end=base_time + timedelta(seconds=60),
                        duration_sec=60.0,
                    ),
                    inference_metrics=InferenceMetrics(
                        total_tokens=100,
                        input_tokens=50,
                        output_tokens=50,
                        inference_time_sec=60.0,
                        tokens_per_second=1.67,
                        latency_per_token_ms=600.0,
                    ),
                    energy_metrics=EnergyMetrics(
                        total_energy_j=50.0,
                        duration_sec=60.0,
                    ),
                    compute_metrics=ComputeMetrics(
                        flops_total=1e10,
                        flops_method="parameter_estimate",
                        flops_confidence="low",
                    ),
                )
            )

        aggregated = aggregate_results("dup_gpu_test", results)

        # Should have GPU attribution warning
        assert not aggregated.aggregation.gpu_attribution_verified
        assert any(
            "gpu" in w.lower() or "double" in w.lower() for w in aggregated.aggregation.warnings
        )
