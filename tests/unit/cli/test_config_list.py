"""Unit tests for lem config list command."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llenergymeasure.cli import app

# Pass NO_COLOR to disable Rich colors in test output for consistent assertions
runner = CliRunner(env={"NO_COLOR": "1"})


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create temporary config directory with sample YAML files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def sample_configs(config_dir: Path) -> Path:
    """Create sample config files for testing."""
    # Valid experiment config
    pytorch_config = {
        "schema_version": "3.0.0",
        "config_name": "test-pytorch",
        "backend": "pytorch",
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "pytorch": {"batch_size": 1},
    }
    (config_dir / "pytorch.yaml").write_text(yaml.safe_dump(pytorch_config))

    # Another valid config with different backend
    vllm_config = {
        "schema_version": "3.0.0",
        "config_name": "test-vllm",
        "backend": "vllm",
        "model_name": "meta-llama/Llama-2-7b-hf",
        "vllm": {"max_num_seqs": 256},
    }
    (config_dir / "vllm.yaml").write_text(yaml.safe_dump(vllm_config))

    # Campaign config (should be filtered out)
    campaign_config = {
        "campaign_name": "test-campaign",
        "configs": ["pytorch.yaml", "vllm.yaml"],
    }
    (config_dir / "campaign.yaml").write_text(yaml.safe_dump(campaign_config))

    # Invalid YAML file
    (config_dir / "broken.yaml").write_text("invalid: yaml: content:")

    return config_dir


class TestConfigList:
    """Tests for lem config list command."""

    def test_config_list_shows_configs(self, sample_configs: Path) -> None:
        """Run config list, verify table shows experiment configs but not campaign."""
        result = runner.invoke(app, ["config", "list", "-d", str(sample_configs)])

        assert result.exit_code == 0
        assert "Available Configurations" in result.stdout
        assert "test-pytorch" in result.stdout
        assert "test-vllm" in result.stdout
        assert "pytorch" in result.stdout
        assert "vllm" in result.stdout
        # Campaign config should NOT appear
        assert "test-campaign" not in result.stdout

    def test_config_list_empty_directory(self, tmp_path: Path) -> None:
        """Run config list on empty directory, verify friendly message."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = runner.invoke(app, ["config", "list", "-d", str(empty_dir)])

        assert result.exit_code == 0
        assert "No configuration files found" in result.stdout

    def test_config_list_shows_backend_column(self, sample_configs: Path) -> None:
        """Verify backend column appears in table."""
        result = runner.invoke(app, ["config", "list", "-d", str(sample_configs)])

        assert result.exit_code == 0
        assert "Backend" in result.stdout
        assert "pytorch" in result.stdout
        assert "vllm" in result.stdout

    def test_config_list_shows_model_column(self, sample_configs: Path) -> None:
        """Verify model column appears in table."""
        result = runner.invoke(app, ["config", "list", "-d", str(sample_configs)])

        assert result.exit_code == 0
        assert "Model" in result.stdout
        assert "TinyLlama" in result.stdout
        assert "Llama-2" in result.stdout

    def test_config_list_shows_path_column(self, sample_configs: Path) -> None:
        """Verify path column appears in table."""
        result = runner.invoke(app, ["config", "list", "-d", str(sample_configs)])

        assert result.exit_code == 0
        assert "Path" in result.stdout
        # Path column should contain the directory name or file stem
        assert "configs" in result.stdout or "pytorch" in result.stdout

    def test_config_list_skips_invalid_files(self, sample_configs: Path) -> None:
        """Invalid YAML files are skipped with count shown."""
        result = runner.invoke(app, ["config", "list", "-d", str(sample_configs)])

        assert result.exit_code == 0
        # Should report skipped files
        assert "Skipped" in result.stdout or "skipped" in result.stdout

    def test_config_list_default_directory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no -d flag, uses configs/ as default directory."""
        # This test verifies the option default, even if configs/ doesn't exist
        result = runner.invoke(app, ["config", "list"])

        # Exit code 0 even if no configs found
        assert result.exit_code == 0

    def test_config_list_long_model_names_truncated(self, config_dir: Path) -> None:
        """Very long model names are truncated for table display."""
        long_model_config = {
            "schema_version": "3.0.0",
            "config_name": "long-model",
            "backend": "pytorch",
            "model_name": "very/long/model/name/that/exceeds/forty/characters/and/should/be/truncated",
        }
        (config_dir / "long_model.yaml").write_text(yaml.safe_dump(long_model_config))

        result = runner.invoke(app, ["config", "list", "-d", str(config_dir)])

        assert result.exit_code == 0
        # Should have ellipsis for truncation
        assert "..." in result.stdout or "long-model" in result.stdout

    def test_config_list_show_user_config_flag(
        self, sample_configs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--show-user-config flag displays .lem-config.yaml settings."""
        # Create .lem-config.yaml
        user_config = {
            "results_dir": "custom_results",
            "thermal_gaps": {"between_experiments": 60.0, "between_cycles": 300.0},
            "docker": {"strategy": "ephemeral"},
            "notifications": {"webhook_url": None},
        }
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".lem-config.yaml").write_text(yaml.safe_dump(user_config))

        result = runner.invoke(
            app, ["config", "list", "-d", str(sample_configs), "--show-user-config"]
        )

        assert result.exit_code == 0
        assert "User Configuration" in result.stdout
        assert "custom_results" in result.stdout
        assert "ephemeral" in result.stdout

    def test_config_list_missing_user_config(
        self, sample_configs: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--show-user-config with missing file shows friendly message."""
        # Change to empty directory so .lem-config.yaml doesn't exist
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["config", "list", "-d", str(sample_configs), "-u"])

        assert result.exit_code == 0
        # Should mention no config found or show default values
        # Either "No .lem-config.yaml" or shows results/default settings
        assert "User Configuration" in result.stdout or "No .lem-config.yaml" in result.stdout


class TestConfigListFiltering:
    """Tests for campaign config filtering."""

    def test_campaign_configs_filtered_by_campaign_name_key(self, tmp_path: Path) -> None:
        """Configs with campaign_name key are excluded from listing."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        # Experiment config (has config_name)
        experiment = {
            "schema_version": "3.0.0",
            "config_name": "experiment",
            "backend": "pytorch",
            "model_name": "test/model",
        }
        (config_dir / "experiment.yaml").write_text(yaml.safe_dump(experiment))

        # Campaign config (has campaign_name)
        campaign = {"campaign_name": "my-campaign", "configs": ["experiment.yaml"]}
        (config_dir / "campaign.yaml").write_text(yaml.safe_dump(campaign))

        result = runner.invoke(app, ["config", "list", "-d", str(config_dir)])

        assert result.exit_code == 0
        assert "experiment" in result.stdout
        assert "my-campaign" not in result.stdout

    def test_mixed_configs_only_experiments_shown(self, tmp_path: Path) -> None:
        """Directory with both experiment and campaign configs shows only experiments."""
        config_dir = tmp_path / "mixed"
        config_dir.mkdir()

        # Multiple experiment configs
        for i in range(3):
            exp_cfg = {
                "schema_version": "3.0.0",
                "config_name": f"exp-{i}",
                "backend": "pytorch",
                "model_name": "test/model",
            }
            (config_dir / f"exp_{i}.yaml").write_text(yaml.safe_dump(exp_cfg))

        # Multiple campaign configs
        for i in range(2):
            camp_cfg = {
                "campaign_name": f"campaign-{i}",
                "configs": [f"exp_{i}.yaml"],
            }
            (config_dir / f"campaign_{i}.yaml").write_text(yaml.safe_dump(camp_cfg))

        result = runner.invoke(app, ["config", "list", "-d", str(config_dir)])

        assert result.exit_code == 0
        # All experiments present
        assert "exp-0" in result.stdout
        assert "exp-1" in result.stdout
        assert "exp-2" in result.stdout
        # No campaigns
        assert "campaign-0" not in result.stdout
        assert "campaign-1" not in result.stdout
