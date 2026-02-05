"""Tests for user configuration loading."""

from pathlib import Path

import pytest

from llenergymeasure.config.user_config import (
    DockerConfig,
    NotificationsConfig,
    ThermalGapConfig,
    UserConfig,
    load_user_config,
)


class TestUserConfig:
    """Tests for UserConfig model."""

    def test_default_values(self) -> None:
        """UserConfig has sensible defaults."""
        config = UserConfig()

        assert config.verbosity == "normal"
        assert config.thermal_gaps.between_experiments == 60.0
        assert config.thermal_gaps.between_cycles == 300.0
        assert config.docker.strategy == "ephemeral"
        assert config.docker.warmup_delay == 0.0
        assert config.docker.auto_teardown is True
        assert config.results_dir == "results"
        # Verify notifications defaults
        assert config.notifications.on_complete is True
        assert config.notifications.on_failure is True
        assert config.notifications.webhook_url is None

    def test_verbosity_values(self) -> None:
        """UserConfig validates verbosity values."""
        # Valid values
        for level in ["quiet", "normal", "verbose"]:
            config = UserConfig(verbosity=level)
            assert config.verbosity == level

    def test_verbosity_invalid_value(self) -> None:
        """UserConfig rejects invalid verbosity."""
        with pytest.raises(ValueError, match="Input should be 'quiet', 'normal' or 'verbose'"):
            UserConfig(verbosity="invalid")

    def test_user_config_no_default_backend(self) -> None:
        """Verify default_backend not in UserConfig.model_fields (removed in 2.3-01)."""
        assert "default_backend" not in UserConfig.model_fields

    def test_user_config_has_notifications(self) -> None:
        """Verify notifications field exists in UserConfig."""
        assert "notifications" in UserConfig.model_fields

    def test_thermal_gaps_config(self) -> None:
        """ThermalGapConfig validates values."""
        config = ThermalGapConfig(between_experiments=30.0, between_cycles=120.0)
        assert config.between_experiments == 30.0
        assert config.between_cycles == 120.0

    def test_docker_config_strategy(self) -> None:
        """DockerConfig validates strategy values."""
        config = DockerConfig(strategy="persistent")
        assert config.strategy == "persistent"

        config = DockerConfig(strategy="ephemeral")
        assert config.strategy == "ephemeral"

    def test_docker_config_invalid_strategy(self) -> None:
        """DockerConfig rejects invalid strategy."""
        with pytest.raises(ValueError, match="Input should be 'ephemeral' or 'persistent'"):
            DockerConfig(strategy="invalid")


class TestLoadUserConfig:
    """Tests for load_user_config function."""

    def test_missing_file_returns_defaults(self) -> None:
        """Missing config file returns defaults without error."""
        config = load_user_config(Path("/nonexistent/path.yaml"))

        assert config.thermal_gaps.between_experiments == 60.0
        assert config.docker.strategy == "ephemeral"

    def test_load_valid_config(self, tmp_path: Path) -> None:
        """Valid config file is loaded correctly."""
        config_path = tmp_path / ".lem-config.yaml"
        config_path.write_text("""
thermal_gaps:
  between_experiments: 30
  between_cycles: 180
docker:
  strategy: persistent
  warmup_delay: 5.0
notifications:
  webhook_url: https://example.com/hook
  on_complete: true
  on_failure: false
""")

        config = load_user_config(config_path)

        assert config.thermal_gaps.between_experiments == 30.0
        assert config.thermal_gaps.between_cycles == 180.0
        assert config.docker.strategy == "persistent"
        assert config.docker.warmup_delay == 5.0
        assert config.notifications.webhook_url == "https://example.com/hook"
        assert config.notifications.on_complete is True
        assert config.notifications.on_failure is False

    def test_partial_config_uses_defaults(self, tmp_path: Path) -> None:
        """Partial config file fills in missing values with defaults."""
        config_path = tmp_path / ".lem-config.yaml"
        config_path.write_text("""
thermal_gaps:
  between_experiments: 45
""")

        config = load_user_config(config_path)

        assert config.thermal_gaps.between_experiments == 45.0
        assert config.thermal_gaps.between_cycles == 300.0  # Default
        assert config.docker.strategy == "ephemeral"  # Default

    def test_empty_file_returns_defaults(self, tmp_path: Path) -> None:
        """Empty config file returns defaults."""
        config_path = tmp_path / ".lem-config.yaml"
        config_path.write_text("")

        config = load_user_config(config_path)

        assert config.thermal_gaps.between_experiments == 60.0
        assert config.docker.strategy == "ephemeral"

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """Invalid YAML raises ValueError (fail-fast validation)."""
        config_path = tmp_path / ".lem-config.yaml"
        config_path.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_user_config(config_path)

    def test_invalid_config_raises_value_error(self, tmp_path: Path) -> None:
        """Invalid config values raise ValueError (fail-fast validation)."""
        config_path = tmp_path / ".lem-config.yaml"
        config_path.write_text("""
docker:
  strategy: invalid_strategy
""")

        with pytest.raises(ValueError, match="Invalid config"):
            load_user_config(config_path)


class TestNotificationsConfig:
    """Tests for NotificationsConfig model."""

    def test_notifications_config_defaults(self) -> None:
        """NotificationsConfig has correct defaults."""
        config = NotificationsConfig()

        assert config.on_complete is True
        assert config.on_failure is True
        assert config.webhook_url is None

    def test_notifications_config_with_url(self) -> None:
        """NotificationsConfig accepts webhook URL."""
        config = NotificationsConfig(
            webhook_url="https://example.com/hook",
            on_complete=False,
            on_failure=True,
        )

        assert config.webhook_url == "https://example.com/hook"
        assert config.on_complete is False
        assert config.on_failure is True
