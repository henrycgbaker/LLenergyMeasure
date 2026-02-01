"""Unit tests for extended campaign configuration models (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llenergymeasure.config.campaign_config import (
    CampaignColdStartConfig,
    CampaignConfig,
    CampaignDaemonConfig,
    CampaignGridConfig,
    CampaignHealthCheckConfig,
    CampaignIOConfig,
)


class TestCampaignConfigBackwardsCompat:
    """Tests for backwards compatibility with existing config-list mode."""

    @pytest.fixture
    def temp_configs(self, tmp_path: Path) -> list[Path]:
        """Create temporary config files for testing."""
        configs = []
        for name in ["pytorch_base", "vllm_base"]:
            config_path = tmp_path / f"{name}.yaml"
            config_path.write_text(
                yaml.dump(
                    {"config_name": name, "model_name": "test/model", "backend": name.split("_")[0]}
                )
            )
            configs.append(config_path)
        return configs

    def test_campaign_config_backwards_compat(self, temp_configs: list[Path]) -> None:
        """Existing config-list mode still works with Phase 2 additions."""
        config = CampaignConfig(
            campaign_name="legacy-campaign",
            configs=[str(p) for p in temp_configs],
        )
        assert len(config.configs) == 2
        # Phase 2 defaults present
        assert config.grid is None
        assert config.health_check.enabled is True
        assert config.cold_start.force_cold_start is False
        assert config.daemon is None
        assert config.io.results_dir == "results"

    def test_campaign_config_grid_mode(self) -> None:
        """Grid config without configs list is valid."""
        config = CampaignConfig(
            campaign_name="grid-campaign",
            grid=CampaignGridConfig(
                backends=["pytorch", "vllm"],
                shared={"fp_precision": ["float16"]},
            ),
        )
        assert config.grid is not None
        assert len(config.configs) == 0
        assert config.grid.backends == ["pytorch", "vllm"]

    def test_campaign_config_requires_source(self) -> None:
        """Neither configs nor grid nor experiments raises validation error."""
        with pytest.raises(ValueError, match="At least one"):
            CampaignConfig(campaign_name="empty-campaign")


class TestCampaignGridConfig:
    """Tests for CampaignGridConfig validation."""

    def test_campaign_grid_config_validation_valid(self) -> None:
        """Valid backend names accepted."""
        grid = CampaignGridConfig(
            backends=["pytorch", "vllm", "tensorrt"],
        )
        assert len(grid.backends) == 3

    def test_campaign_grid_config_validation_invalid(self) -> None:
        """Invalid backend name rejected."""
        with pytest.raises(ValueError, match="Unknown backends"):
            CampaignGridConfig(backends=["pytorch", "nonexistent_backend"])

    def test_campaign_grid_config_empty_backends(self) -> None:
        """Empty backends list rejected."""
        with pytest.raises(ValueError):
            CampaignGridConfig(backends=[])

    def test_campaign_grid_config_with_overrides(self) -> None:
        """Backend overrides stored correctly."""
        grid = CampaignGridConfig(
            backends=["pytorch"],
            backend_overrides={"pytorch": {"batch_size": [1, 2, 4]}},
        )
        assert "pytorch" in grid.backend_overrides
        assert grid.backend_overrides["pytorch"]["batch_size"] == [1, 2, 4]


class TestCampaignHealthCheckConfig:
    """Tests for CampaignHealthCheckConfig defaults."""

    def test_campaign_health_check_defaults(self) -> None:
        """Default values are correct."""
        config = CampaignHealthCheckConfig()
        assert config.enabled is True
        assert config.interval_experiments == 0
        assert config.gpu_memory_threshold_pct == 90.0
        assert config.restart_on_failure is True
        assert config.max_restarts == 3

    def test_health_check_custom_threshold(self) -> None:
        """Custom GPU memory threshold accepted."""
        config = CampaignHealthCheckConfig(gpu_memory_threshold_pct=80.0)
        assert config.gpu_memory_threshold_pct == 80.0

    def test_health_check_threshold_bounds(self) -> None:
        """GPU memory threshold must be between 50 and 100."""
        with pytest.raises(ValueError):
            CampaignHealthCheckConfig(gpu_memory_threshold_pct=30.0)


class TestCampaignColdStartConfig:
    """Tests for CampaignColdStartConfig defaults."""

    def test_campaign_cold_start_defaults(self) -> None:
        """Default is force_cold_start=False."""
        config = CampaignColdStartConfig()
        assert config.force_cold_start is False
        assert config.restart_container is False

    def test_cold_start_enabled(self) -> None:
        """Cold start can be enabled."""
        config = CampaignColdStartConfig(force_cold_start=True, restart_container=True)
        assert config.force_cold_start is True
        assert config.restart_container is True


class TestCampaignIOConfig:
    """Tests for CampaignIOConfig defaults and paths."""

    def test_campaign_io_defaults(self) -> None:
        """Default paths correct."""
        config = CampaignIOConfig()
        assert config.results_dir == "results"
        assert config.configs_dir == "configs"
        assert config.state_dir == ".state"
        assert config.manifest_filename == "campaign_manifest.json"

    def test_campaign_io_manifest_path(self) -> None:
        """manifest_path property constructs correct path."""
        config = CampaignIOConfig()
        assert config.manifest_path == Path(".state") / "campaign_manifest.json"

    def test_campaign_io_custom_paths(self) -> None:
        """Custom paths stored correctly."""
        config = CampaignIOConfig(
            results_dir="/data/results",
            state_dir="/data/state",
        )
        assert config.manifest_path == Path("/data/state") / "campaign_manifest.json"


class TestCampaignDaemonConfig:
    """Tests for CampaignDaemonConfig validation and parsing."""

    def test_campaign_daemon_defaults(self) -> None:
        """Defaults: disabled, no scheduling."""
        config = CampaignDaemonConfig()
        assert config.enabled is False
        assert config.at is None
        assert config.interval is None
        assert config.total_duration is None
        assert config.quiet is True

    def test_campaign_daemon_interval_parsing(self) -> None:
        """Interval parsing works for Nh and Nm formats."""
        config = CampaignDaemonConfig(enabled=True, interval="6h")
        assert config.interval == "6h"
        assert config.interval_seconds == 21600  # 6 * 3600

        config_m = CampaignDaemonConfig(enabled=True, interval="30m")
        assert config_m.interval_seconds == 1800  # 30 * 60

    def test_campaign_daemon_total_duration_parsing(self) -> None:
        """Total duration parsing works."""
        config = CampaignDaemonConfig(enabled=True, total_duration="48h")
        assert config.total_duration_seconds == 172800  # 48 * 3600

    def test_campaign_daemon_invalid_interval(self) -> None:
        """Invalid interval format rejected."""
        with pytest.raises(ValueError, match="Nh or Nm"):
            CampaignDaemonConfig(enabled=True, interval="bad_format")

    def test_campaign_daemon_invalid_time(self) -> None:
        """Invalid time format rejected."""
        with pytest.raises(ValueError, match="Invalid time"):
            CampaignDaemonConfig(enabled=True, at="25:00")

    def test_campaign_daemon_valid_time(self) -> None:
        """Valid time format accepted."""
        config = CampaignDaemonConfig(enabled=True, at="03:00")
        assert config.at == "03:00"

    def test_campaign_daemon_no_interval_returns_none(self) -> None:
        """interval_seconds returns None when interval not set."""
        config = CampaignDaemonConfig()
        assert config.interval_seconds is None
        assert config.total_duration_seconds is None
