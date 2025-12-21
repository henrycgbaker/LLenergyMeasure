"""Tests for the CLI module."""

from datetime import datetime

import pytest
from typer.testing import CliRunner

from llm_energy_measure.cli import app
from llm_energy_measure.domain.experiment import (
    AggregatedResult,
    AggregationMetadata,
    RawProcessResult,
    Timestamps,
)
from llm_energy_measure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llm_energy_measure.results.repository import FileSystemRepository

runner = CliRunner()


class TestMainApp:
    """Tests for main CLI app."""

    def test_help_displays(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "LLM inference efficiency measurement framework" in result.stdout

    def test_version_displays(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "llm-energy-measure" in result.stdout


class TestRunCommand:
    """Tests for the run command."""

    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "Run an LLM efficiency experiment" in result.stdout

    def test_run_requires_config(self):
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0  # Should fail - missing required arg

    def test_run_with_invalid_config(self, tmp_path):
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid_field: true")
        result = runner.invoke(app, ["run", str(config_file)])
        assert result.exit_code == 1

    def test_run_dry_run(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: meta-llama/Llama-2-7b-hf
""")
        result = runner.invoke(app, ["run", str(config_file), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.stdout
        assert "Config loaded" in result.stdout


class TestAggregateCommand:
    """Tests for the aggregate command."""

    def test_aggregate_help(self):
        result = runner.invoke(app, ["aggregate", "--help"])
        assert result.exit_code == 0
        assert "Aggregate raw per-process results" in result.stdout

    def test_aggregate_requires_id_or_all(self):
        result = runner.invoke(app, ["aggregate"])
        assert result.exit_code == 1
        assert "Provide experiment ID or use --all" in result.stdout

    def test_aggregate_nonexistent_experiment(self, tmp_path):
        result = runner.invoke(app, ["aggregate", "nonexistent", "--results-dir", str(tmp_path)])
        assert "No raw results found" in result.stdout

    def test_aggregate_all_no_pending(self, tmp_path):
        result = runner.invoke(app, ["aggregate", "--all", "--results-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No pending experiments" in result.stdout


class TestConfigCommands:
    """Tests for config subcommands."""

    def test_config_validate_help(self):
        result = runner.invoke(app, ["config", "validate", "--help"])
        assert result.exit_code == 0
        assert "Validate an experiment configuration file" in result.stdout

    def test_config_validate_valid(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: meta-llama/Llama-2-7b-hf
max_input_tokens: 512
max_output_tokens: 256
""")
        result = runner.invoke(app, ["config", "validate", str(config_file)])
        assert result.exit_code == 0
        assert "Valid configuration" in result.stdout

    def test_config_validate_invalid(self, tmp_path):
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("config_name: test")  # Missing model_name
        result = runner.invoke(app, ["config", "validate", str(config_file)])
        assert result.exit_code == 1
        assert "Invalid configuration" in result.stdout

    def test_config_show(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: meta-llama/Llama-2-7b-hf
num_processes: 2
gpu_list: [0, 1]
""")
        result = runner.invoke(app, ["config", "show", str(config_file)])
        assert result.exit_code == 0
        assert "test_experiment" in result.stdout
        assert "meta-llama/Llama-2-7b-hf" in result.stdout


class TestResultsCommands:
    """Tests for results subcommands."""

    @pytest.fixture
    def sample_raw_result(self) -> RawProcessResult:
        """Create a sample raw process result."""
        return RawProcessResult(
            experiment_id="test_exp_001",
            process_index=0,
            gpu_id=0,
            config_name="test_config",
            model_name="test-model",
            timestamps=Timestamps(
                start=datetime(2024, 1, 1, 10, 0, 0),
                end=datetime(2024, 1, 1, 10, 1, 0),
                duration_sec=60.0,
            ),
            inference_metrics=InferenceMetrics(
                total_tokens=1000,
                input_tokens=500,
                output_tokens=500,
                inference_time_sec=60.0,
                tokens_per_second=16.67,
                latency_per_token_ms=60.0,
            ),
            energy_metrics=EnergyMetrics(
                total_energy_j=100.0,
                gpu_energy_j=80.0,
                cpu_energy_j=20.0,
                duration_sec=60.0,
            ),
            compute_metrics=ComputeMetrics(
                flops_total=1e12,
                flops_per_second=1.67e10,
                flops_method="calflops",
                flops_confidence="high",
            ),
        )

    @pytest.fixture
    def sample_aggregated_result(self, sample_raw_result) -> AggregatedResult:
        """Create a sample aggregated result."""
        return AggregatedResult(
            experiment_id="test_exp_001",
            aggregation=AggregationMetadata(
                method="sum_energy_avg_throughput",
                num_processes=1,
                temporal_overlap_verified=True,
                gpu_attribution_verified=True,
            ),
            total_tokens=1000,
            total_energy_j=100.0,
            total_inference_time_sec=60.0,
            avg_tokens_per_second=16.67,
            avg_energy_per_token_j=0.1,
            total_flops=1e12,
            process_results=[sample_raw_result],
            start_time=datetime(2024, 1, 1, 10, 0, 0),
            end_time=datetime(2024, 1, 1, 10, 1, 0),
        )

    def test_results_list_help(self):
        result = runner.invoke(app, ["results", "list", "--help"])
        assert result.exit_code == 0
        assert "List all experiments" in result.stdout

    def test_results_list_empty(self, tmp_path):
        result = runner.invoke(app, ["results", "list", "--results-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No experiments found" in result.stdout

    def test_results_list_with_data(self, tmp_path, sample_raw_result, sample_aggregated_result):
        # Set up repository with data
        repo = FileSystemRepository(tmp_path)
        repo.save_raw("test_exp_001", sample_raw_result)
        repo.save_aggregated(sample_aggregated_result)

        result = runner.invoke(app, ["results", "list", "--results-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "test_exp_001" in result.stdout

    def test_results_show_help(self):
        result = runner.invoke(app, ["results", "show", "--help"])
        assert result.exit_code == 0
        assert "Show detailed results" in result.stdout

    def test_results_show_not_found(self, tmp_path):
        result = runner.invoke(
            app, ["results", "show", "nonexistent", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "No aggregated result" in result.stdout

    def test_results_show_aggregated(self, tmp_path, sample_raw_result, sample_aggregated_result):
        repo = FileSystemRepository(tmp_path)
        repo.save_raw("test_exp_001", sample_raw_result)
        repo.save_aggregated(sample_aggregated_result)

        result = runner.invoke(
            app, ["results", "show", "test_exp_001", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "test_exp_001" in result.stdout
        assert "1,000" in result.stdout  # Total tokens

    def test_results_show_raw(self, tmp_path, sample_raw_result):
        repo = FileSystemRepository(tmp_path)
        repo.save_raw("test_exp_001", sample_raw_result)

        result = runner.invoke(
            app, ["results", "show", "test_exp_001", "--raw", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert "Process 0" in result.stdout
        assert "GPU 0" in result.stdout

    def test_results_show_json_output(self, tmp_path, sample_raw_result, sample_aggregated_result):
        repo = FileSystemRepository(tmp_path)
        repo.save_raw("test_exp_001", sample_raw_result)
        repo.save_aggregated(sample_aggregated_result)

        result = runner.invoke(
            app, ["results", "show", "test_exp_001", "--json", "--results-dir", str(tmp_path)]
        )
        assert result.exit_code == 0
        assert '"experiment_id"' in result.stdout
        assert '"test_exp_001"' in result.stdout


class TestAggregationWorkflow:
    """Integration tests for the aggregate workflow."""

    @pytest.fixture
    def sample_raw_results(self) -> list[RawProcessResult]:
        """Create sample raw results for multiple processes."""
        base_time = datetime(2024, 1, 1, 10, 0, 0)
        results = []
        for i in range(2):
            results.append(
                RawProcessResult(
                    experiment_id="multi_gpu_exp",
                    process_index=i,
                    gpu_id=i,
                    config_name="test_config",
                    model_name="test-model",
                    timestamps=Timestamps(
                        start=base_time,
                        end=datetime(2024, 1, 1, 10, 1, 0),
                        duration_sec=60.0,
                    ),
                    inference_metrics=InferenceMetrics(
                        total_tokens=500,
                        input_tokens=250,
                        output_tokens=250,
                        inference_time_sec=30.0,
                        tokens_per_second=16.67,
                        latency_per_token_ms=60.0,
                    ),
                    energy_metrics=EnergyMetrics(
                        total_energy_j=50.0,
                        gpu_energy_j=40.0,
                        cpu_energy_j=10.0,
                        duration_sec=30.0,
                    ),
                    compute_metrics=ComputeMetrics(
                        flops_total=5e11,
                        flops_per_second=1.67e10,
                        flops_method="calflops",
                        flops_confidence="high",
                    ),
                )
            )
        return results

    def test_aggregate_multiple_processes(self, tmp_path, sample_raw_results):
        """Test aggregating results from multiple processes."""
        repo = FileSystemRepository(tmp_path)
        for result in sample_raw_results:
            repo.save_raw("multi_gpu_exp", result)

        result = runner.invoke(app, ["aggregate", "multi_gpu_exp", "--results-dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Aggregated multi_gpu_exp" in result.stdout
        assert "2 processes" in result.stdout

        # Verify aggregated result was created
        assert repo.has_aggregated("multi_gpu_exp")

    def test_aggregate_skip_existing(self, tmp_path, sample_raw_results):
        """Test that existing aggregated results are skipped."""
        repo = FileSystemRepository(tmp_path)
        for result in sample_raw_results:
            repo.save_raw("multi_gpu_exp", result)

        # First aggregation
        runner.invoke(app, ["aggregate", "multi_gpu_exp", "--results-dir", str(tmp_path)])

        # Second should skip
        result = runner.invoke(app, ["aggregate", "multi_gpu_exp", "--results-dir", str(tmp_path)])
        assert "Skipping" in result.stdout
        assert "already aggregated" in result.stdout

    def test_aggregate_force(self, tmp_path, sample_raw_results):
        """Test force re-aggregation."""
        repo = FileSystemRepository(tmp_path)
        for result in sample_raw_results:
            repo.save_raw("multi_gpu_exp", result)

        # First aggregation
        runner.invoke(app, ["aggregate", "multi_gpu_exp", "--results-dir", str(tmp_path)])

        # Force re-aggregation
        result = runner.invoke(
            app, ["aggregate", "multi_gpu_exp", "--force", "--results-dir", str(tmp_path)]
        )
        assert "Aggregated multi_gpu_exp" in result.stdout
