"""Unit tests for schema_version field and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from llenergymeasure.cli import app
from llenergymeasure.config.loader import load_config
from llenergymeasure.config.models import CURRENT_SCHEMA_VERSION, ExperimentConfig

# Pass NO_COLOR to disable Rich colors in test output
runner = CliRunner(env={"NO_COLOR": "1"})


class TestSchemaVersionField:
    """Tests for schema_version field in config models."""

    def test_current_schema_version_constant_exists(self) -> None:
        """CURRENT_SCHEMA_VERSION constant is defined."""
        assert CURRENT_SCHEMA_VERSION is not None
        assert isinstance(CURRENT_SCHEMA_VERSION, str)
        assert "." in CURRENT_SCHEMA_VERSION  # Should be semver

    def test_experiment_config_has_schema_version_field(self) -> None:
        """ExperimentConfig model has schema_version field."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test/model",
        )
        # schema_version should exist
        assert hasattr(config, "schema_version")
        # Default is None (not auto-populated)
        assert config.schema_version is None

    def test_schema_version_can_be_set_explicitly(self) -> None:
        """schema_version can be set to a specific value."""
        custom_version = "2.0.0"
        config = ExperimentConfig(
            config_name="test",
            model_name="test/model",
            schema_version=custom_version,
        )
        assert config.schema_version == custom_version

    def test_schema_version_persists_in_model_dump(self) -> None:
        """schema_version appears in model_dump() output when set."""
        config = ExperimentConfig(
            config_name="test",
            model_name="test/model",
            schema_version=CURRENT_SCHEMA_VERSION,
        )
        dumped = config.model_dump()
        assert "schema_version" in dumped
        assert dumped["schema_version"] == CURRENT_SCHEMA_VERSION


class TestSchemaVersionInConfigs:
    """Tests for schema_version in YAML config files."""

    @pytest.fixture
    def temp_config_dir(self, tmp_path: Path) -> Path:
        """Create temporary config directory."""
        config_dir = tmp_path / "configs"
        config_dir.mkdir()
        return config_dir

    def test_config_with_schema_version_loads_correctly(self, temp_config_dir: Path) -> None:
        """Config with schema_version field loads without warnings."""
        config_path = temp_config_dir / "test.yaml"
        config_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "config_name": "test-config",
            "model_name": "test/model",
            "backend": "pytorch",
        }
        config_path.write_text(yaml.safe_dump(config_data))

        config = load_config(config_path)
        assert config.schema_version == CURRENT_SCHEMA_VERSION

    def test_config_without_schema_version_is_none(self, temp_config_dir: Path) -> None:
        """Config without schema_version loads with None value."""
        config_path = temp_config_dir / "test.yaml"
        config_data = {
            "config_name": "test-config",
            "model_name": "test/model",
            "backend": "pytorch",
        }
        config_path.write_text(yaml.safe_dump(config_data))

        config = load_config(config_path)
        # No default - remains None if not specified
        assert config.schema_version is None

    def test_config_with_old_schema_version_loads(self, temp_config_dir: Path) -> None:
        """Config with older schema_version loads (may produce warnings)."""
        config_path = temp_config_dir / "test.yaml"
        old_version = "2.0.0"
        config_data = {
            "schema_version": old_version,
            "config_name": "test-config",
            "model_name": "test/model",
            "backend": "pytorch",
        }
        config_path.write_text(yaml.safe_dump(config_data))

        # Should still load successfully
        config = load_config(config_path)
        assert config.schema_version == old_version

    def test_example_configs_have_schema_version(self) -> None:
        """Example configs in configs/examples/ have schema_version field."""
        examples_dir = Path("configs/examples")
        if not examples_dir.exists():
            pytest.skip("configs/examples directory not found")

        yaml_files = list(examples_dir.glob("*.yaml"))
        # Filter out campaign configs
        experiment_configs = []
        for yaml_path in yaml_files:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            # Skip campaign configs
            if isinstance(data, dict) and "campaign_name" not in data:
                experiment_configs.append((yaml_path, data))

        # At least some experiment configs should exist
        assert len(experiment_configs) > 0, "No experiment configs found in examples/"

        # All should have schema_version
        for yaml_path, config_data in experiment_configs:
            assert "schema_version" in config_data, f"{yaml_path.name} missing schema_version"
            # Should be current version for examples
            assert (
                config_data["schema_version"] == CURRENT_SCHEMA_VERSION
            ), f"{yaml_path.name} has outdated schema_version"


class TestSchemaVersionValidation:
    """Tests for schema_version validation warnings."""

    def test_config_validate_command_shows_schema_info(self, tmp_path: Path) -> None:
        """lem config validate shows schema version in output."""
        config_path = tmp_path / "test.yaml"
        config_data = {
            "schema_version": CURRENT_SCHEMA_VERSION,
            "config_name": "test-config",
            "model_name": "test/model",
            "backend": "pytorch",
        }
        config_path.write_text(yaml.safe_dump(config_data))

        result = runner.invoke(app, ["config", "validate", str(config_path)])

        assert result.exit_code == 0
        # Should show config is valid
        assert "Valid configuration" in result.stdout or "✓" in result.stdout

    def test_config_show_command_runs_without_error(self, tmp_path: Path) -> None:
        """lem config show displays config without errors."""
        config_path = tmp_path / "test.yaml"
        config_data = {
            "schema_version": "2.0.0",
            "config_name": "test-config",
            "model_name": "test/model",
            "backend": "pytorch",
        }
        config_path.write_text(yaml.safe_dump(config_data))

        result = runner.invoke(app, ["config", "show", str(config_path)])

        assert result.exit_code == 0
        # Config should display successfully
        assert "test-config" in result.stdout or "test/model" in result.stdout

    def test_config_new_command_works(self, tmp_path: Path) -> None:
        """lem config new generates valid configs."""
        output_path = tmp_path / "generated.yaml"
        result = runner.invoke(
            app,
            [
                "config",
                "new",
                "--output",
                str(output_path),
            ],
            input="\n".join(
                [
                    "test-config",  # config name
                    "test/model",  # model name
                    "n",  # start from preset
                    "pytorch",  # backend
                    "1",  # num gpus
                    "float16",  # precision
                    "512",  # max input tokens
                    "128",  # max output tokens
                    "1",  # batch size
                    "n",  # quantization
                ]
            ),
        )

        assert result.exit_code == 0
        assert output_path.exists()

        with open(output_path) as f:
            generated_config = yaml.safe_load(f)

        # Config should be valid
        assert "config_name" in generated_config
        assert "model_name" in generated_config


class TestSchemaVersionBackwardsCompatibility:
    """Tests for handling configs with different schema versions."""

    def test_loading_v1_config_without_schema_field(self, tmp_path: Path) -> None:
        """Old v1 configs without schema_version field still load."""
        config_path = tmp_path / "v1_config.yaml"
        # V1 config: no schema_version field
        v1_config = {
            "config_name": "old-config",
            "model_name": "test/model",
            "backend": "pytorch",
            "pytorch": {"batch_size": 1},
        }
        config_path.write_text(yaml.safe_dump(v1_config))

        # Should load successfully (schema_version will be None)
        config = load_config(config_path)
        assert config.schema_version is None  # Not auto-populated

    def test_loading_v2_config_with_old_schema(self, tmp_path: Path) -> None:
        """V2 config with explicit old schema_version loads."""
        config_path = tmp_path / "v2_config.yaml"
        v2_config = {
            "schema_version": "2.0.0",
            "config_name": "v2-config",
            "model_name": "test/model",
            "backend": "pytorch",
        }
        config_path.write_text(yaml.safe_dump(v2_config))

        config = load_config(config_path)
        assert config.schema_version == "2.0.0"
