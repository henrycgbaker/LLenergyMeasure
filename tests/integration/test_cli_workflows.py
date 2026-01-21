"""Integration tests for CLI end-to-end workflows.

Tests complete CLI command chains without requiring GPU access.
"""

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


@pytest.fixture
def config_hierarchy(tmp_path: Path) -> dict[str, Path]:
    """Create a hierarchy of config files."""
    configs = {}

    # Base config
    base = tmp_path / "configs" / "base.yaml"
    base.parent.mkdir(parents=True)
    base.write_text("""
max_input_tokens: 1024
max_output_tokens: 256
decoder:
  temperature: 0.7
  top_p: 0.9
""")
    configs["base"] = base

    # Model config
    model = tmp_path / "configs" / "llama-7b.yaml"
    model.write_text("""
_extends: base.yaml
config_name: llama-7b-test
model_name: meta-llama/Llama-2-7b-hf
num_processes: 2
gpus: [0, 1]
""")
    configs["model"] = model

    # Quantized variant
    quant = tmp_path / "configs" / "llama-7b-4bit.yaml"
    quant.write_text("""
_extends: llama-7b.yaml
config_name: llama-7b-4bit
quantization:
  quantization: true
  load_in_4bit: true
""")
    configs["quant"] = quant

    return configs


@pytest.fixture
def populated_results(tmp_path: Path) -> Path:
    """Create a results directory with sample data."""
    results_dir = tmp_path / "results"
    repo = FileSystemRepository(results_dir)
    base_time = datetime(2024, 1, 15, 10, 0, 0)

    # Create 2 experiments with 2 processes each
    for exp_num in range(2):
        exp_id = f"exp_{exp_num:03d}"
        for proc in range(2):
            result = RawProcessResult(
                experiment_id=exp_id,
                process_index=proc,
                gpu_id=proc,
                config_name="test_config",
                model_name="test-model",
                timestamps=Timestamps(
                    start=base_time + timedelta(minutes=exp_num * 10),
                    end=base_time + timedelta(minutes=exp_num * 10 + 1),
                    duration_sec=60.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=500 + proc * 50,
                    input_tokens=200,
                    output_tokens=300 + proc * 50,
                    inference_time_sec=60.0,
                    tokens_per_second=8.33,
                    latency_per_token_ms=120.0,
                ),
                energy_metrics=EnergyMetrics(
                    total_energy_j=50.0 + proc * 5,
                    gpu_energy_j=40.0,
                    cpu_energy_j=10.0 + proc * 5,
                    duration_sec=60.0,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=5e11,
                    flops_per_second=8.33e9,
                    flops_method="calflops",
                    flops_confidence="high",
                ),
            )
            repo.save_raw(exp_id, result)

    return results_dir


class TestConfigWorkflow:
    """Test config validation and inspection workflow."""

    def test_validate_then_show_workflow(self, config_hierarchy):
        """Test: validate config → show resolved config."""
        model_config = config_hierarchy["model"]

        # Step 1: Validate
        result = runner.invoke(app, ["config", "validate", str(model_config)])
        assert result.exit_code == 0
        assert "Valid configuration" in result.stdout
        assert "llama-7b-test" in result.stdout

        # Step 2: Show resolved config
        result = runner.invoke(app, ["config", "show", str(model_config)])
        assert result.exit_code == 0
        assert "1024" in result.stdout  # Inherited max_input_tokens
        assert "meta-llama/Llama-2-7b-hf" in result.stdout

    def test_validate_inheritance_chain(self, config_hierarchy):
        """Test validating config with multi-level inheritance."""
        quant_config = config_hierarchy["quant"]

        result = runner.invoke(app, ["config", "validate", str(quant_config)])
        assert result.exit_code == 0
        assert "llama-7b-4bit" in result.stdout
        assert "load_in_4bit: True" in result.stdout  # Shows quantization

    def test_validate_invalid_config(self, tmp_path: Path):
        """Test validation catches invalid configs."""
        invalid = tmp_path / "invalid.yaml"
        invalid.write_text("""
config_name: test
# Missing model_name!
""")

        result = runner.invoke(app, ["config", "validate", str(invalid)])
        assert result.exit_code == 1
        assert "Invalid configuration" in result.stdout


class TestAggregationWorkflow:
    """Test aggregation CLI workflow."""

    def test_list_then_aggregate_workflow(self, populated_results: Path):
        """Test: list experiments → aggregate → show results."""
        # Step 1: List experiments (should show pending)
        result = runner.invoke(
            app, ["results", "list", "--all", "--results-dir", str(populated_results)]
        )
        assert result.exit_code == 0
        assert "exp_000" in result.stdout
        assert "exp_001" in result.stdout
        assert "pending" in result.stdout

        # Step 2: Aggregate first experiment
        result = runner.invoke(
            app, ["aggregate", "exp_000", "--results-dir", str(populated_results)]
        )
        assert result.exit_code == 0
        assert "Aggregated exp_000" in result.stdout
        assert "2 processes" in result.stdout

        # Step 3: Show aggregated result
        result = runner.invoke(
            app, ["results", "show", "exp_000", "--results-dir", str(populated_results)]
        )
        assert result.exit_code == 0
        assert "exp_000" in result.stdout
        assert "Total Tokens" in result.stdout

    def test_aggregate_all_workflow(self, populated_results: Path):
        """Test aggregating all pending experiments."""
        # Aggregate all
        result = runner.invoke(app, ["aggregate", "--all", "--results-dir", str(populated_results)])
        assert result.exit_code == 0
        assert "Aggregating 2 experiments" in result.stdout
        assert "exp_000" in result.stdout
        assert "exp_001" in result.stdout

        # Verify both aggregated
        result = runner.invoke(app, ["results", "list", "--results-dir", str(populated_results)])
        assert result.exit_code == 0
        # Both should now show as aggregated (green checkmarks)
        assert result.stdout.count("exp_") >= 2

    def test_aggregate_force_reaggregation(self, populated_results: Path):
        """Test force re-aggregation of existing results."""
        # First aggregation
        runner.invoke(app, ["aggregate", "exp_000", "--results-dir", str(populated_results)])

        # Second attempt without force - should skip
        result = runner.invoke(
            app, ["aggregate", "exp_000", "--results-dir", str(populated_results)]
        )
        assert "Skipping" in result.stdout
        assert "already aggregated" in result.stdout

        # With force - should re-aggregate
        result = runner.invoke(
            app, ["aggregate", "exp_000", "--force", "--results-dir", str(populated_results)]
        )
        assert "Aggregated exp_000" in result.stdout


class TestResultsInspectionWorkflow:
    """Test results inspection workflow."""

    def test_show_raw_results(self, populated_results: Path):
        """Test viewing raw per-process results."""
        result = runner.invoke(
            app,
            ["results", "show", "exp_000", "--raw", "--results-dir", str(populated_results)],
        )
        assert result.exit_code == 0
        assert "Process 0" in result.stdout
        assert "Process 1" in result.stdout
        assert "GPU 0" in result.stdout
        assert "GPU 1" in result.stdout

    def test_show_json_output(self, populated_results: Path):
        """Test JSON output format."""
        # First aggregate
        runner.invoke(app, ["aggregate", "exp_000", "--results-dir", str(populated_results)])

        # Get JSON output
        result = runner.invoke(
            app,
            ["results", "show", "exp_000", "--json", "--results-dir", str(populated_results)],
        )
        assert result.exit_code == 0
        assert '"experiment_id"' in result.stdout
        assert '"exp_000"' in result.stdout
        assert '"total_tokens"' in result.stdout

    def test_show_not_aggregated_suggests_action(self, populated_results: Path):
        """Test that showing non-aggregated experiment suggests action."""
        result = runner.invoke(
            app, ["results", "show", "exp_000", "--results-dir", str(populated_results)]
        )
        assert result.exit_code == 1
        assert "No aggregated result" in result.stdout
        assert "aggregate" in result.stdout.lower()


class TestDryRunWorkflow:
    """Test dry-run workflow for config validation."""

    def test_run_dry_run(self, config_hierarchy):
        """Test run command with --dry-run."""
        model_config = config_hierarchy["model"]

        result = runner.invoke(app, ["run", str(model_config), "--dry-run"])
        assert result.exit_code == 0
        assert "Config loaded" in result.stdout
        assert "Dry run" in result.stdout
        assert "llama-7b-test" in result.stdout

    def test_run_dry_run_with_warnings(self, tmp_path: Path):
        """Test dry-run shows config warnings."""
        config_file = tmp_path / "high_tokens.yaml"
        config_file.write_text("""
config_name: high-tokens-test
model_name: test-model
max_output_tokens: 4096
""")

        result = runner.invoke(app, ["run", str(config_file), "--dry-run"])
        assert result.exit_code == 0
        assert "Warning" in result.stdout
        assert "max_output_tokens" in result.stdout


class TestCompleteWorkflow:
    """Test complete end-to-end workflows."""

    def test_full_benchmark_workflow(self, config_hierarchy, tmp_path: Path):
        """Test complete workflow: validate → (mock run) → aggregate → export."""
        results_dir = tmp_path / "benchmark_results"
        model_config = config_hierarchy["model"]

        # Step 1: Validate config
        result = runner.invoke(app, ["config", "validate", str(model_config)])
        assert result.exit_code == 0

        # Step 2: Simulate run by creating raw results
        repo = FileSystemRepository(results_dir)
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        for proc in range(2):
            raw = RawProcessResult(
                experiment_id="benchmark_001",
                process_index=proc,
                gpu_id=proc,
                config_name="llama-7b-test",
                model_name="meta-llama/Llama-2-7b-hf",
                timestamps=Timestamps(
                    start=base_time,
                    end=base_time + timedelta(minutes=5),
                    duration_sec=300.0,
                ),
                inference_metrics=InferenceMetrics(
                    total_tokens=5000,
                    input_tokens=2000,
                    output_tokens=3000,
                    inference_time_sec=300.0,
                    tokens_per_second=16.67,
                    latency_per_token_ms=60.0,
                ),
                energy_metrics=EnergyMetrics(
                    total_energy_j=500.0,
                    gpu_energy_j=400.0,
                    cpu_energy_j=100.0,
                    duration_sec=300.0,
                ),
                compute_metrics=ComputeMetrics(
                    flops_total=1e13,
                    flops_per_second=3.33e10,
                    flops_method="calflops",
                    flops_confidence="high",
                ),
            )
            repo.save_raw("benchmark_001", raw)

        # Step 3: Aggregate
        result = runner.invoke(
            app, ["aggregate", "benchmark_001", "--results-dir", str(results_dir)]
        )
        assert result.exit_code == 0
        assert "Aggregated benchmark_001" in result.stdout

        # Step 4: View results
        result = runner.invoke(
            app, ["results", "show", "benchmark_001", "--results-dir", str(results_dir)]
        )
        assert result.exit_code == 0
        assert "10,000" in result.stdout  # Total tokens (5000 * 2)
        assert "1000.00" in result.stdout  # Energy (500 * 2 = 1000 J)
