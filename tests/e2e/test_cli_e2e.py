"""End-to-end tests for the CLI.

These tests simulate complete user workflows from config creation
through result analysis, without requiring GPU access.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llm_energy_measure.cli import app
from llm_energy_measure.domain.experiment import RawProcessResult, Timestamps
from llm_energy_measure.domain.metrics import ComputeMetrics, EnergyMetrics, InferenceMetrics
from llm_energy_measure.results.repository import FileSystemRepository

# Pass NO_COLOR to disable Rich colors in test output for consistent assertions
runner = CliRunner(env={"NO_COLOR": "1"})


class TestBenchmarkWorkflowE2E:
    """E2E test: Complete benchmark workflow."""

    @pytest.fixture
    def workspace(self, tmp_path: Path) -> dict[str, Path]:
        """Set up a complete workspace with configs, prompts, and results dirs."""
        workspace = {
            "root": tmp_path,
            "configs": tmp_path / "configs",
            "prompts": tmp_path / "prompts",
            "results": tmp_path / "results",
            "exports": tmp_path / "exports",
        }
        for path in workspace.values():
            path.mkdir(parents=True, exist_ok=True)

        # Create base config
        base_config = workspace["configs"] / "base.yaml"
        base_config.write_text("""
# Base configuration for all experiments
max_input_tokens: 1024
max_output_tokens: 512
decoder_config:
  temperature: 0.7
  top_p: 0.9
  do_sample: true
fp_precision: float16
""")

        # Create model-specific configs
        llama_config = workspace["configs"] / "llama-7b.yaml"
        llama_config.write_text("""
_extends: base.yaml
config_name: llama-7b-benchmark
model_name: meta-llama/Llama-2-7b-hf
num_processes: 2
gpu_list: [0, 1]
""")

        mistral_config = workspace["configs"] / "mistral-7b.yaml"
        mistral_config.write_text("""
_extends: base.yaml
config_name: mistral-7b-benchmark
model_name: mistralai/Mistral-7B-v0.1
num_processes: 2
gpu_list: [0, 1]
""")

        # Create prompts file
        prompts = workspace["prompts"] / "test_prompts.txt"
        prompts.write_text("""What is machine learning?
Explain neural networks in simple terms.
How does backpropagation work?
What is the difference between supervised and unsupervised learning?
""")

        workspace["llama_config"] = llama_config
        workspace["mistral_config"] = mistral_config
        workspace["prompts_file"] = prompts

        return workspace

    def _create_simulated_results(
        self, repo: FileSystemRepository, experiment_id: str, num_processes: int = 2
    ) -> None:
        """Simulate experiment results (what would be created by actual run)."""
        base_time = datetime.now()

        for proc in range(num_processes):
            result = RawProcessResult(
                experiment_id=experiment_id,
                process_index=proc,
                gpu_id=proc,
                config_name=experiment_id,
                model_name="test-model",
                timestamps=Timestamps(
                    start=base_time,
                    end=base_time + timedelta(minutes=5),
                    duration_sec=300.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=2500 + proc * 100,
                    input_tokens=1000,
                    output_tokens=1500 + proc * 100,
                    inference_time_sec=300.0,
                    tokens_per_second=8.33 + proc * 0.5,
                    latency_per_token_ms=120.0 - proc * 5,
                ),
                energy_metrics=EnergyMetrics(
                    total_energy_j=250.0 + proc * 20,
                    gpu_energy_j=200.0 + proc * 15,
                    cpu_energy_j=50.0 + proc * 5,
                    duration_sec=300.0,
                    gpu_power_w=40.0 + proc * 3,
                    cpu_power_w=10.0 + proc,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=5e12 + proc * 1e11,
                    flops_per_second=1.67e10,
                    flops_method="calflops",
                    flops_confidence="high",
                ),
            )
            repo.save_raw(experiment_id, result)

    def test_complete_benchmark_workflow(self, workspace: dict[str, Path]):
        """Test complete workflow: validate → run(dry) → simulate → aggregate → analyze."""
        results_dir = workspace["results"]
        llama_config = workspace["llama_config"]

        # Step 1: Validate configuration
        result = runner.invoke(app, ["config", "validate", str(llama_config)])
        assert result.exit_code == 0
        assert "Valid configuration" in result.stdout
        assert "llama-7b-benchmark" in result.stdout

        # Step 2: Show resolved config
        result = runner.invoke(app, ["config", "show", str(llama_config)])
        assert result.exit_code == 0
        assert "1024" in result.stdout  # Inherited max_input_tokens
        assert "float16" in result.stdout  # Inherited precision

        # Step 3: Dry run to verify setup
        result = runner.invoke(
            app,
            ["run", str(llama_config), "--dry-run", "--results-dir", str(results_dir)],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.stdout

        # Step 4: Simulate experiment results
        repo = FileSystemRepository(results_dir)
        self._create_simulated_results(repo, "llama-7b-benchmark")

        # Step 5: Verify results exist
        result = runner.invoke(app, ["results", "list", "--all", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        assert "llama-7b-benchmark" in result.stdout
        assert "pending" in result.stdout

        # Step 6: Aggregate results
        result = runner.invoke(
            app, ["aggregate", "llama-7b-benchmark", "--results-dir", str(results_dir)]
        )
        assert result.exit_code == 0
        assert "Aggregated llama-7b-benchmark" in result.stdout
        assert "2 processes" in result.stdout

        # Step 7: View aggregated results
        result = runner.invoke(
            app, ["results", "show", "llama-7b-benchmark", "--results-dir", str(results_dir)]
        )
        assert result.exit_code == 0
        assert "Total Tokens" in result.stdout
        assert "Total Energy" in result.stdout
        assert "Efficiency" in result.stdout

        # Step 8: Export as JSON
        result = runner.invoke(
            app,
            ["results", "show", "llama-7b-benchmark", "--json", "--results-dir", str(results_dir)],
        )
        assert result.exit_code == 0
        # Verify valid JSON
        json_data = json.loads(result.stdout)
        assert json_data["experiment_id"] == "llama-7b-benchmark"
        assert json_data["aggregation"]["num_processes"] == 2

    def test_multi_model_comparison_workflow(self, workspace: dict[str, Path]):
        """Test workflow comparing multiple models."""
        results_dir = workspace["results"]
        repo = FileSystemRepository(results_dir)

        models = ["llama-7b-benchmark", "mistral-7b-benchmark"]

        # Simulate results for both models
        for model in models:
            self._create_simulated_results(repo, model)

        # Aggregate all
        result = runner.invoke(app, ["aggregate", "--all", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        assert "Aggregating 2 experiments" in result.stdout

        # List all results
        result = runner.invoke(app, ["results", "list", "--results-dir", str(results_dir)])
        assert result.exit_code == 0
        for model in models:
            assert model in result.stdout

        # Compare by viewing each
        for model in models:
            result = runner.invoke(
                app, ["results", "show", model, "--json", "--results-dir", str(results_dir)]
            )
            assert result.exit_code == 0
            data = json.loads(result.stdout)
            assert data["total_tokens"] > 0
            assert data["total_energy_j"] > 0


class TestConfigManagementE2E:
    """E2E tests for configuration management workflows."""

    def test_config_inheritance_validation(self, tmp_path: Path):
        """Test validating a complex config inheritance chain."""
        configs = tmp_path / "configs"
        configs.mkdir()

        # Create inheritance chain: defaults → model-family → specific
        (configs / "defaults.yaml").write_text("""
max_input_tokens: 512
max_output_tokens: 256
decoder_config:
  temperature: 1.0
  top_p: 1.0
""")

        (configs / "llama-family.yaml").write_text("""
_extends: defaults.yaml
decoder_config:
  temperature: 0.7
fp_precision: float16
""")

        (configs / "llama-7b-4bit.yaml").write_text("""
_extends: llama-family.yaml
config_name: llama-7b-4bit
model_name: meta-llama/Llama-2-7b-hf
quantization_config:
  quantization: true
  load_in_4bit: true
num_processes: 1
gpu_list: [0]
""")

        # Validate the final config
        result = runner.invoke(app, ["config", "validate", str(configs / "llama-7b-4bit.yaml")])
        assert result.exit_code == 0
        assert "llama-7b-4bit" in result.stdout
        assert "load_in_4bit: True" in result.stdout

        # Show resolved config
        result = runner.invoke(app, ["config", "show", str(configs / "llama-7b-4bit.yaml")])
        assert result.exit_code == 0
        # Check inherited values
        assert "512" in result.stdout  # From defaults
        assert "float16" in result.stdout  # From llama-family

    def test_config_validation_errors(self, tmp_path: Path):
        """Test that config validation catches errors correctly."""
        configs = tmp_path / "configs"
        configs.mkdir()

        # Invalid: processes > GPUs
        (configs / "invalid_gpus.yaml").write_text("""
config_name: invalid
model_name: test-model
num_processes: 4
gpu_list: [0, 1]
""")

        result = runner.invoke(app, ["config", "validate", str(configs / "invalid_gpus.yaml")])
        assert result.exit_code == 1

        # Invalid: min > max tokens
        (configs / "invalid_tokens.yaml").write_text("""
config_name: invalid
model_name: test-model
min_output_tokens: 1000
max_output_tokens: 100
""")

        result = runner.invoke(app, ["config", "validate", str(configs / "invalid_tokens.yaml")])
        assert result.exit_code == 1


class TestResultsAnalysisE2E:
    """E2E tests for results analysis workflows."""

    @pytest.fixture
    def results_with_data(self, tmp_path: Path) -> Path:
        """Create results directory with multiple experiments."""
        results_dir = tmp_path / "results"
        repo = FileSystemRepository(results_dir)
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        # Create 3 experiments with varying characteristics
        experiments = [
            ("exp_baseline", 1, 100.0, 500),
            ("exp_optimized", 2, 80.0, 600),
            ("exp_quantized", 2, 60.0, 450),
        ]

        for exp_id, num_proc, energy_base, tokens_base in experiments:
            for proc in range(num_proc):
                result = RawProcessResult(
                    experiment_id=exp_id,
                    process_index=proc,
                    gpu_id=proc,
                    config_name=exp_id,
                    model_name="test-model",
                    timestamps=Timestamps(
                        start=base_time,
                        end=base_time + timedelta(minutes=5),
                        duration_sec=300.0,
                    ),
                    inference_metrics=InferenceMetrics(
                        total_tokens=tokens_base + proc * 50,
                        input_tokens=200,
                        output_tokens=tokens_base - 200 + proc * 50,
                        inference_time_sec=300.0,
                        tokens_per_second=tokens_base / 300.0,
                        latency_per_token_ms=300000.0 / tokens_base,
                    ),
                    energy_metrics=EnergyMetrics(
                        total_energy_j=energy_base + proc * 10,
                        gpu_energy_j=energy_base * 0.8,
                        cpu_energy_j=energy_base * 0.2,
                        duration_sec=300.0,
                    ),
                    compute_metrics=ComputeMetrics(
                        flops_total=1e12,
                        flops_per_second=3.33e9,
                        flops_method="calflops",
                        flops_confidence="high",
                    ),
                )
                repo.save_raw(exp_id, result)

        return results_dir

    def test_results_comparison_workflow(self, results_with_data: Path):
        """Test comparing results across experiments."""
        # Aggregate all experiments
        result = runner.invoke(app, ["aggregate", "--all", "--results-dir", str(results_with_data)])
        assert result.exit_code == 0

        # List experiments
        result = runner.invoke(app, ["results", "list", "--results-dir", str(results_with_data)])
        assert result.exit_code == 0
        assert "exp_baseline" in result.stdout
        assert "exp_optimized" in result.stdout
        assert "exp_quantized" in result.stdout

        # Collect metrics for comparison
        metrics = {}
        for exp_id in ["exp_baseline", "exp_optimized", "exp_quantized"]:
            result = runner.invoke(
                app, ["results", "show", exp_id, "--json", "--results-dir", str(results_with_data)]
            )
            assert result.exit_code == 0
            metrics[exp_id] = json.loads(result.stdout)

        # Verify we can compare key metrics
        for _exp_id, data in metrics.items():
            assert "total_tokens" in data
            assert "total_energy_j" in data
            assert "avg_tokens_per_second" in data

    def test_raw_vs_aggregated_view(self, results_with_data: Path):
        """Test viewing raw process results vs aggregated."""
        # Aggregate first
        runner.invoke(app, ["aggregate", "exp_optimized", "--results-dir", str(results_with_data)])

        # View raw results
        result = runner.invoke(
            app,
            ["results", "show", "exp_optimized", "--raw", "--results-dir", str(results_with_data)],
        )
        assert result.exit_code == 0
        assert "Process 0" in result.stdout
        assert "Process 1" in result.stdout

        # View aggregated
        result = runner.invoke(
            app, ["results", "show", "exp_optimized", "--results-dir", str(results_with_data)]
        )
        assert result.exit_code == 0
        assert "Aggregated Metrics" in result.stdout
        assert "Processes: 2" in result.stdout


class TestErrorRecoveryE2E:
    """E2E tests for error handling and recovery."""

    def test_recovery_from_partial_aggregation(self, tmp_path: Path):
        """Test recovering when some experiments fail to aggregate."""
        results_dir = tmp_path / "results"
        repo = FileSystemRepository(results_dir)
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Create valid experiment
        for proc in range(2):
            result = RawProcessResult(
                experiment_id="valid_exp",
                process_index=proc,
                gpu_id=proc,
                config_name="test",
                model_name="test-model",
                timestamps=Timestamps(
                    start=base_time,
                    end=base_time + timedelta(minutes=1),
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
                energy_metrics=EnergyMetrics(total_energy_j=50.0, duration_sec=60.0),
                compute_metrics=ComputeMetrics(
                    flops_total=1e10,
                    flops_method="parameter_estimate",
                    flops_confidence="low",
                ),
            )
            repo.save_raw("valid_exp", result)

        # Create empty experiment directory (will fail to aggregate)
        (results_dir / "raw" / "empty_exp").mkdir(parents=True)

        # Aggregate all - should succeed for valid, skip empty
        result = runner.invoke(app, ["aggregate", "--all", "--results-dir", str(results_dir)])
        # Valid should succeed
        assert "valid_exp" in result.stdout

        # Verify valid was aggregated
        result = runner.invoke(
            app, ["results", "show", "valid_exp", "--results-dir", str(results_dir)]
        )
        assert result.exit_code == 0

    def test_force_reaggregation_after_new_data(self, tmp_path: Path):
        """Test force-reaggregating after adding new process data."""
        results_dir = tmp_path / "results"
        repo = FileSystemRepository(results_dir)
        base_time = datetime(2024, 1, 1, 10, 0, 0)

        # Initial single-process result
        result = RawProcessResult(
            experiment_id="growing_exp",
            process_index=0,
            gpu_id=0,
            config_name="test",
            model_name="test-model",
            timestamps=Timestamps(
                start=base_time,
                end=base_time + timedelta(minutes=1),
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
            energy_metrics=EnergyMetrics(total_energy_j=50.0, duration_sec=60.0),
            compute_metrics=ComputeMetrics(
                flops_total=1e10,
                flops_method="parameter_estimate",
                flops_confidence="low",
            ),
        )
        repo.save_raw("growing_exp", result)

        # First aggregation
        cli_result = runner.invoke(
            app, ["aggregate", "growing_exp", "--results-dir", str(results_dir)]
        )
        assert "1 processes" in cli_result.stdout

        # Add second process
        result2 = RawProcessResult(**{**result.model_dump(), "process_index": 1, "gpu_id": 1})
        repo.save_raw("growing_exp", result2)

        # Force reaggregation
        cli_result = runner.invoke(
            app, ["aggregate", "growing_exp", "--force", "--results-dir", str(results_dir)]
        )
        assert "2 processes" in cli_result.stdout
