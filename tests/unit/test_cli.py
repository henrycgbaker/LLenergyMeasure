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


class TestPresetsCommand:
    """Tests for the presets command."""

    def test_presets_help(self):
        result = runner.invoke(app, ["presets", "--help"])
        assert result.exit_code == 0
        assert "List built-in experiment presets" in result.stdout

    def test_presets_lists_all(self):
        result = runner.invoke(app, ["presets"])
        assert result.exit_code == 0
        # Check all preset names are displayed
        assert "quick-test" in result.stdout
        assert "benchmark" in result.stdout
        assert "throughput" in result.stdout

    def test_presets_shows_descriptions(self):
        result = runner.invoke(app, ["presets"])
        assert result.exit_code == 0
        # Check that descriptions/settings are shown
        assert "max_in=" in result.stdout or "batch=" in result.stdout


class TestExperimentCommand:
    """Tests for the experiment command with presets and CLI overrides."""

    def test_experiment_help(self):
        result = runner.invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "--preset" in result.stdout
        assert "--model" in result.stdout
        assert "--batch-size" in result.stdout
        assert "--precision" in result.stdout
        assert "--max-tokens" in result.stdout
        assert "--seed" in result.stdout
        assert "--gpu-list" in result.stdout
        assert "--temperature" in result.stdout
        assert "--quantization" in result.stdout

    def test_experiment_requires_config_or_preset(self):
        """Error when neither config nor preset provided."""
        result = runner.invoke(app, ["experiment"])
        assert result.exit_code == 1
        assert "config file, --preset, or --resume" in result.stdout

    def test_experiment_preset_requires_model(self):
        """Error when --preset used without --model."""
        result = runner.invoke(app, ["experiment", "--preset", "quick-test"])
        assert result.exit_code == 1
        assert "--model is required" in result.stdout

    def test_experiment_unknown_preset_error(self):
        """Error for unknown preset name."""
        result = runner.invoke(
            app, ["experiment", "--preset", "nonexistent", "--model", "test/model"]
        )
        assert result.exit_code == 1
        assert "Unknown preset" in result.stdout
        assert "nonexistent" in result.stdout
        # Should show available presets
        assert "quick-test" in result.stdout or "benchmark" in result.stdout

    def test_experiment_valid_preset_with_model(self, tmp_path, monkeypatch):
        """Preset with model proceeds to subprocess (mock subprocess)."""
        import subprocess
        from unittest.mock import MagicMock

        calls = []

        def mock_popen(cmd, **kwargs):
            calls.append(cmd)
            # Return a mock process
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        result = runner.invoke(
            app,
            [
                "experiment",
                "--preset",
                "quick-test",
                "--model",
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "--dataset",
                "alpaca",
                "-n",
                "10",
            ],
        )
        # Should exit with the mocked subprocess return code
        assert result.exit_code == 0
        # Check that subprocess was called with accelerate
        assert len(calls) == 1
        cmd = calls[0]
        # Preset settings are baked into a temp config file
        assert "--config" in cmd
        assert "--dataset" in cmd
        assert "alpaca" in cmd
        # Temp config file should exist and be a yaml file
        config_idx = cmd.index("--config") + 1
        assert cmd[config_idx].endswith(".yaml")

    def test_experiment_config_file_with_overrides(self, tmp_path, monkeypatch):
        """Config file with CLI overrides baked into temp config."""
        import subprocess
        from unittest.mock import MagicMock

        import yaml

        calls = []

        def mock_popen(cmd, **kwargs):
            calls.append(cmd)
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: meta-llama/Llama-2-7b-hf
batching_options:
  batch_size: 1
fp_precision: float32
""")
        result = runner.invoke(
            app,
            [
                "experiment",
                str(config_file),
                "--batch-size",
                "8",
                "--precision",
                "float16",
                "--max-tokens",
                "256",
                "--seed",
                "42",
                "--dataset",
                "alpaca",
            ],
        )
        assert result.exit_code == 0
        # Check subprocess was called with accelerate
        assert len(calls) == 1
        cmd = calls[0]
        assert "--config" in cmd
        assert "--dataset" in cmd
        # CLI overrides are baked into temp config file (not original)
        config_idx = cmd.index("--config") + 1
        temp_config_path = cmd[config_idx]
        # Should be a temp file, not the original
        assert temp_config_path != str(config_file)
        assert temp_config_path.endswith(".yaml")
        # Check the temp config contains overrides
        with open(temp_config_path) as f:
            temp_config = yaml.safe_load(f)
        assert temp_config["batching_options"]["batch_size"] == 8
        assert temp_config["fp_precision"] == "float16"
        assert temp_config["max_output_tokens"] == 256
        assert temp_config["random_seed"] == 42

    def test_experiment_gpu_list_parsing(self, tmp_path, monkeypatch):
        """GPU list is parsed from comma-separated string."""
        import subprocess
        from unittest.mock import MagicMock

        calls = []

        def mock_popen(cmd, **kwargs):
            calls.append(cmd)
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        config_file = tmp_path / "config.yaml"
        # Use num_processes <= len(gpu_list) after override
        # Original config has 4 GPUs, 4 processes
        # Override to gpu_list [0,1,2] means 3 GPUs, but we keep num_processes=4 from config
        # So we need a config where the override makes sense
        config_file.write_text("""
config_name: test_experiment
model_name: test/model
gpu_list: [0, 1]
num_processes: 2
""")
        runner.invoke(
            app,
            [
                "experiment",
                str(config_file),
                "--gpu-list",
                "0,1,2,3",
                "--dataset",
                "alpaca",
            ],
        )
        # Should parse "0,1,2,3" into list [0, 1, 2, 3]
        # The config will be validated and subprocess called
        assert len(calls) == 1

    def test_experiment_temperature_override(self, tmp_path, monkeypatch):
        """Temperature override is passed correctly."""
        import subprocess
        from unittest.mock import MagicMock

        calls = []

        def mock_popen(cmd, **kwargs):
            calls.append(cmd)
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: test/model
""")
        result = runner.invoke(
            app,
            [
                "experiment",
                str(config_file),
                "--temperature",
                "0.7",
                "--dataset",
                "alpaca",
                "--fresh",  # Skip incomplete experiment detection
            ],
        )
        assert result.exit_code == 0

    def test_experiment_quantization_flag(self, tmp_path, monkeypatch):
        """Quantization boolean flag works."""
        import subprocess
        from unittest.mock import MagicMock

        calls = []

        def mock_popen(cmd, **kwargs):
            calls.append(cmd)
            mock_proc = MagicMock()
            mock_proc.wait.return_value = 0
            mock_proc.returncode = 0
            mock_proc.pid = 12345
            return mock_proc

        monkeypatch.setattr(subprocess, "Popen", mock_popen)

        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
config_name: test_experiment
model_name: test/model
""")
        # Test --quantization
        result = runner.invoke(
            app,
            [
                "experiment",
                str(config_file),
                "--quantization",
                "--dataset",
                "alpaca",
                "--fresh",  # Skip incomplete experiment detection
            ],
        )
        assert result.exit_code == 0

        # Test --no-quantization
        result = runner.invoke(
            app,
            [
                "experiment",
                str(config_file),
                "--no-quantization",
                "--dataset",
                "alpaca",
                "--fresh",  # Skip incomplete experiment detection
            ],
        )
        assert result.exit_code == 0


class TestConfigNewCommand:
    """Tests for config new command (interactive config builder)."""

    def test_config_new_help(self):
        result = runner.invoke(app, ["config", "new", "--help"])
        assert result.exit_code == 0
        assert "Interactive config builder" in result.stdout
        assert "--preset" in result.stdout
        assert "--output" in result.stdout


class TestConfigGenerateGridCommand:
    """Tests for config generate-grid command."""

    def test_generate_grid_help(self):
        result = runner.invoke(app, ["config", "generate-grid", "--help"])
        assert result.exit_code == 0
        assert "Generate a grid of configs" in result.stdout
        assert "--vary" in result.stdout
        assert "--output-dir" in result.stdout

    def test_generate_grid_requires_vary(self, tmp_path):
        """Error when no --vary parameter provided."""
        config_file = tmp_path / "base.yaml"
        config_file.write_text("""
config_name: base_config
model_name: test/model
""")
        result = runner.invoke(app, ["config", "generate-grid", str(config_file)])
        assert result.exit_code == 1
        assert "--vary parameter is required" in result.stdout

    def test_generate_grid_invalid_vary_format(self, tmp_path):
        """Error when --vary has invalid format."""
        config_file = tmp_path / "base.yaml"
        config_file.write_text("""
config_name: base_config
model_name: test/model
""")
        result = runner.invoke(
            app,
            [
                "config",
                "generate-grid",
                str(config_file),
                "--vary",
                "invalid_no_equals",
            ],
        )
        assert result.exit_code == 1
        assert "Invalid --vary format" in result.stdout

    def test_generate_grid_single_param(self, tmp_path):
        """Generate grid with single parameter variation."""
        config_file = tmp_path / "base.yaml"
        config_file.write_text("""
config_name: base_config
model_name: test/model
batching_options:
  batch_size: 1
""")
        output_dir = tmp_path / "grid"

        result = runner.invoke(
            app,
            [
                "config",
                "generate-grid",
                str(config_file),
                "--vary",
                "batching_options.batch_size=1,2,4,8",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Generated 4 configs" in result.stdout

        # Check files were created
        generated = list(output_dir.glob("*.yaml"))
        assert len(generated) == 4

    def test_generate_grid_multiple_params(self, tmp_path):
        """Generate grid with Cartesian product of multiple parameters."""
        config_file = tmp_path / "base.yaml"
        config_file.write_text("""
config_name: base_config
model_name: test/model
fp_precision: float32
batching_options:
  batch_size: 1
""")
        output_dir = tmp_path / "grid"

        result = runner.invoke(
            app,
            [
                "config",
                "generate-grid",
                str(config_file),
                "--vary",
                "batching_options.batch_size=1,2,4",
                "--vary",
                "fp_precision=float16,float32",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        # 3 batch sizes x 2 precisions = 6 configs
        assert "Generated 6 configs" in result.stdout

        generated = list(output_dir.glob("*.yaml"))
        assert len(generated) == 6

    def test_generate_grid_nested_param(self, tmp_path):
        """Nested parameter (dot notation) works correctly."""
        config_file = tmp_path / "base.yaml"
        config_file.write_text("""
config_name: base_config
model_name: test/model
batching_options:
  batch_size: 1
""")
        output_dir = tmp_path / "grid"

        result = runner.invoke(
            app,
            [
                "config",
                "generate-grid",
                str(config_file),
                "--vary",
                "batching_options.batch_size=2,4",
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
        assert "Generated 2 configs" in result.stdout

        # Verify content of generated configs
        import yaml

        for yaml_file in output_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                content = yaml.safe_load(f)
            assert "batching_options" in content
            assert content["batching_options"]["batch_size"] in [2, 4]


class TestBatchCommand:
    """Tests for batch run command."""

    def test_batch_help(self):
        result = runner.invoke(app, ["batch", "--help"])
        assert result.exit_code == 0
        assert "Run multiple experiment configs" in result.stdout
        assert "--parallel" in result.stdout
        assert "--dry-run" in result.stdout

    def test_batch_no_matching_configs(self, tmp_path):
        """Error when glob pattern matches nothing."""
        result = runner.invoke(app, ["batch", str(tmp_path / "*.yaml")])
        assert result.exit_code == 1
        assert "No configs match pattern" in result.stdout

    def test_batch_dry_run_lists_configs(self, tmp_path):
        """Dry run lists configs without executing."""
        # Create some valid configs
        for i in range(3):
            config_file = tmp_path / f"config_{i}.yaml"
            config_file.write_text(f"""
config_name: test_config_{i}
model_name: test/model
""")

        result = runner.invoke(
            app,
            ["batch", str(tmp_path / "*.yaml"), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "Dry run" in result.stdout
        assert "would execute 3 experiments" in result.stdout

    def test_batch_validates_configs(self, tmp_path):
        """Batch command validates configs and reports invalid ones."""
        # Create one valid and one invalid config
        valid_config = tmp_path / "valid.yaml"
        valid_config.write_text("""
config_name: valid_config
model_name: test/model
""")

        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("""
config_name: invalid_config
# Missing model_name
""")

        result = runner.invoke(
            app,
            ["batch", str(tmp_path / "*.yaml"), "--dry-run"],
        )
        # Should complete with warnings about invalid configs
        assert "invalid configs skipped" in result.stdout or "would execute 1" in result.stdout

    def test_batch_glob_pattern_matching(self, tmp_path):
        """Glob pattern matches correctly."""
        # Create configs with different patterns
        (tmp_path / "experiment_a.yaml").write_text("""
config_name: exp_a
model_name: test/model
""")
        (tmp_path / "experiment_b.yaml").write_text("""
config_name: exp_b
model_name: test/model
""")
        (tmp_path / "other.yaml").write_text("""
config_name: other
model_name: test/model
""")

        # Match only experiment_*.yaml
        result = runner.invoke(
            app,
            ["batch", str(tmp_path / "experiment_*.yaml"), "--dry-run"],
        )
        assert result.exit_code == 0
        assert "would execute 2 experiments" in result.stdout
