"""
Unit tests for CLI commands.

These tests verify that the CLI commands work correctly without requiring
actual model downloads or GPU access.
"""

import json
import pytest
from pathlib import Path
from typer.testing import CliRunner
from unittest.mock import Mock, patch, MagicMock

from llm_efficiency.cli.main import app
from llm_efficiency.config import ExperimentConfig, BatchingConfig, QuantizationConfig


runner = CliRunner()


class TestInitCommand:
    """Test the init command."""

    def test_init_creates_default_config(self, tmp_path, monkeypatch):
        """Test that init creates config.json by default."""
        monkeypatch.chdir(tmp_path)

        # Mock the prompts to provide automatic answers
        with patch('llm_efficiency.cli.main.Prompt.ask') as mock_ask, \
             patch('llm_efficiency.cli.main.Confirm.ask') as mock_confirm:

            mock_ask.side_effect = [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # model_name
                "float16",  # precision
                "8",  # batch_size
                "100",  # num_prompts
                "512",  # max_input
                "128",  # max_output
            ]
            mock_confirm.return_value = False  # quantize

            result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert (tmp_path / "config.json").exists()

        # Verify config contents
        with open(tmp_path / "config.json") as f:
            config_data = json.load(f)

        assert config_data["model_name"] == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config_data["precision"] == "float16"
        assert config_data["batching"]["batch_size"] == 8
        assert config_data["num_input_prompts"] == 100

    def test_init_creates_custom_named_config(self, tmp_path, monkeypatch):
        """Test that init can create custom-named config files."""
        monkeypatch.chdir(tmp_path)

        with patch('llm_efficiency.cli.main.Prompt.ask') as mock_ask, \
             patch('llm_efficiency.cli.main.Confirm.ask') as mock_confirm:

            mock_ask.side_effect = [
                "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "float16",
                "8",
                "100",
                "512",
                "128",
            ]
            mock_confirm.return_value = False

            result = runner.invoke(app, ["init", "my-experiment.json"])

        assert result.exit_code == 0
        assert (tmp_path / "my-experiment.json").exists()
        assert not (tmp_path / "config.json").exists()


class TestRunCommand:
    """Test the run command."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all heavy dependencies for run command."""
        with patch('llm_efficiency.cli.main.setup_logging'), \
             patch('llm_efficiency.cli.main.setup_accelerator') as mock_accelerator, \
             patch('llm_efficiency.cli.main.generate_experiment_id') as mock_exp_id, \
             patch('llm_efficiency.cli.main.load_model_and_tokenizer') as mock_load_model, \
             patch('llm_efficiency.cli.main.load_prompts_from_dataset') as mock_load_prompts, \
             patch('llm_efficiency.cli.main.filter_prompts_by_length') as mock_filter, \
             patch('llm_efficiency.cli.main.run_inference_experiment') as mock_inference, \
             patch('llm_efficiency.cli.main.FLOPsCalculator') as mock_flops, \
             patch('llm_efficiency.cli.main.EnergyTracker') as mock_energy, \
             patch('llm_efficiency.cli.main.get_gpu_memory_stats') as mock_gpu, \
             patch('llm_efficiency.cli.main.create_results') as mock_create_results, \
             patch('llm_efficiency.cli.main.ResultsManager') as mock_results_manager, \
             patch('llm_efficiency.cli.main.torch') as mock_torch:

            # Setup mock returns
            mock_exp_id.return_value = "test-exp-123"

            # Mock model
            mock_model = MagicMock()
            mock_param = MagicMock()
            mock_param.device = "cpu"
            mock_param.numel.return_value = 1000000
            mock_param.requires_grad = True
            mock_model.parameters.return_value = [mock_param]

            mock_tokenizer = MagicMock()
            mock_load_model.return_value = (mock_model, mock_tokenizer)

            # Mock prompts
            mock_load_prompts.return_value = ["test prompt 1", "test prompt 2"]
            mock_filter.return_value = ["test prompt 1", "test prompt 2"]

            # Mock inference
            mock_inference.return_value = (
                ["output 1", "output 2"],
                {
                    "tokens_per_second": 100.0,
                    "avg_latency_per_query": 0.01,
                    "queries_per_second": 10.0,
                    "total_tokens": 200,
                    "num_prompts": 2,
                }
            )

            # Mock FLOPs
            mock_flops_instance = MagicMock()
            mock_flops_instance.get_flops.return_value = 1000000
            mock_flops.return_value = mock_flops_instance

            # Mock GPU stats
            mock_gpu.return_value = {
                "gpu_current_memory_allocated_bytes": 1024 * 1024 * 100,
                "gpu_peak_memory_allocated_bytes": 1024 * 1024 * 150,
            }

            # Mock torch
            mock_torch.cuda.is_available.return_value = False

            # Mock results manager
            mock_manager = MagicMock()
            mock_manager.save_experiment.return_value = Path("results/test-exp-123.json")
            mock_results_manager.return_value = mock_manager

            yield {
                'accelerator': mock_accelerator,
                'exp_id': mock_exp_id,
                'load_model': mock_load_model,
                'load_prompts': mock_load_prompts,
                'filter': mock_filter,
                'inference': mock_inference,
                'flops': mock_flops,
                'energy': mock_energy,
                'gpu': mock_gpu,
                'create_results': mock_create_results,
                'results_manager': mock_results_manager,
            }

    def test_run_requires_model_or_config(self):
        """Test that run command requires either --model or --config."""
        result = runner.invoke(app, ["run"])

        assert result.exit_code == 1
        assert "--model is required when not using --config" in result.stdout

    def test_run_with_model_argument(self, mock_dependencies):
        """Test run command with --model argument."""
        result = runner.invoke(app, [
            "run",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--batch-size", "8",
            "--num-prompts", "10",
        ])

        # Should succeed (exit code 0)
        assert result.exit_code == 0

        # Verify model was loaded
        mock_dependencies['load_model'].assert_called_once()

        # Verify prompts were loaded
        mock_dependencies['load_prompts'].assert_called_once()

    def test_run_with_config_file(self, tmp_path, mock_dependencies):
        """Test run command with --config file."""
        # Create a config file
        config_file = tmp_path / "test-config.json"
        config = ExperimentConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            precision="float16",
            num_input_prompts=10,
            max_input_tokens=512,
            max_output_tokens=128,
            batching=BatchingConfig(batch_size=8),
            quantization=QuantizationConfig(enabled=False),
        )

        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f)

        result = runner.invoke(app, [
            "run",
            "--config", str(config_file),
        ])

        # Should succeed
        assert result.exit_code == 0

        # Verify model was loaded with correct config
        mock_dependencies['load_model'].assert_called_once()
        call_args = mock_dependencies['load_model'].call_args
        loaded_config = call_args[0][0]
        assert loaded_config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert loaded_config.batching.batch_size == 8

    def test_run_config_with_cli_overrides(self, tmp_path, mock_dependencies):
        """Test that CLI arguments override config file values."""
        # Create a config file with batch_size=8
        config_file = tmp_path / "test-config.json"
        config = ExperimentConfig(
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            precision="float16",
            num_input_prompts=10,
            max_input_tokens=512,
            max_output_tokens=128,
            batching=BatchingConfig(batch_size=8),
        )

        with open(config_file, "w") as f:
            json.dump(config.to_dict(), f)

        # Run with --batch-size override
        result = runner.invoke(app, [
            "run",
            "--config", str(config_file),
            "--batch-size", "16",
            "--num-prompts", "50",
        ])

        assert result.exit_code == 0

        # Verify overrides were applied
        call_args = mock_dependencies['load_model'].call_args
        loaded_config = call_args[0][0]
        assert loaded_config.batching.batch_size == 16
        assert loaded_config.num_input_prompts == 50

    def test_run_with_nonexistent_config_file(self):
        """Test that run fails gracefully with missing config file."""
        result = runner.invoke(app, [
            "run",
            "--config", "nonexistent.json",
        ])

        assert result.exit_code == 1
        assert "Config file" in result.stdout
        assert "not found" in result.stdout

    def test_run_with_all_options(self, mock_dependencies):
        """Test run command with all CLI options."""
        result = runner.invoke(app, [
            "run",
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--precision", "bfloat16",
            "--batch-size", "16",
            "--num-prompts", "50",
            "--max-input", "256",
            "--max-output", "64",
            "--quantize",
            "--dataset", "custom/dataset",
            "--no-energy",
            "--verbose",
        ])

        assert result.exit_code == 0

        # Verify config was created with all options
        call_args = mock_dependencies['load_model'].call_args
        loaded_config = call_args[0][0]
        assert loaded_config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert loaded_config.precision == "bfloat16"
        assert loaded_config.batching.batch_size == 16
        assert loaded_config.num_input_prompts == 50
        assert loaded_config.max_input_tokens == 256
        assert loaded_config.max_output_tokens == 64
        assert loaded_config.quantization.enabled is True


class TestListCommand:
    """Test the list command."""

    def test_list_with_no_experiments(self, tmp_path):
        """Test list command with no experiments."""
        with patch('llm_efficiency.cli.main.ResultsManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.list_experiments.return_value = []
            mock_manager.return_value = mock_instance

            result = runner.invoke(app, ["list", "--results-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No experiments found" in result.stdout


class TestShowCommand:
    """Test the show command."""

    def test_show_nonexistent_experiment(self):
        """Test show command with nonexistent experiment."""
        with patch('llm_efficiency.cli.main.ResultsManager') as mock_manager:
            mock_instance = MagicMock()
            mock_instance.load_experiment.return_value = None
            mock_manager.return_value = mock_instance

            result = runner.invoke(app, ["show", "nonexistent-id"])

        assert result.exit_code == 1
        assert "not found" in result.stdout


class TestVersionCommand:
    """Test version display."""

    def test_version_flag(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "LLM Efficiency Measurement Tool" in result.stdout
        assert "version" in result.stdout.lower()
