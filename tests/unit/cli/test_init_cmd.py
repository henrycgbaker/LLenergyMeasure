"""Unit tests for lem init command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml
from typer.testing import CliRunner

from llenergymeasure.cli import app

if TYPE_CHECKING:
    from pytest import MonkeyPatch

# Pass NO_COLOR to disable Rich colors in test output for consistent assertions
runner = CliRunner(env={"NO_COLOR": "1"})


@pytest.fixture
def work_dir(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Change to tmp_path for isolated config file creation."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


class TestInitNonInteractive:
    """Tests for lem init command in non-interactive mode."""

    def test_init_non_interactive_creates_config(self, work_dir: Path) -> None:
        """Run --non-interactive, verify .lem-config.yaml created with defaults."""
        result = runner.invoke(app, ["init", "--non-interactive"])

        assert result.exit_code == 0
        assert "Config written" in result.stdout

        config_path = work_dir / ".lem-config.yaml"
        assert config_path.exists()

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check default values are present
        assert config.get("results_dir") == "results"
        assert "thermal_gaps" in config
        assert "docker" in config

    def test_init_non_interactive_with_results_dir(self, work_dir: Path) -> None:
        """Run --non-interactive --results-dir /custom, verify custom value in config."""
        result = runner.invoke(
            app, ["init", "--non-interactive", "--results-dir", "/custom/results"]
        )

        assert result.exit_code == 0

        config_path = work_dir / ".lem-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["results_dir"] == "/custom/results"

    def test_init_non_interactive_with_webhook(self, work_dir: Path) -> None:
        """Run --non-interactive --webhook-url, verify webhook in config."""
        webhook_url = "https://example.com/webhook"
        result = runner.invoke(app, ["init", "--non-interactive", "--webhook-url", webhook_url])

        assert result.exit_code == 0

        config_path = work_dir / ".lem-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["notifications"]["webhook_url"] == webhook_url

    def test_init_non_interactive_with_all_flags(self, work_dir: Path) -> None:
        """Run --non-interactive with multiple flags, verify all values in config."""
        result = runner.invoke(
            app,
            [
                "init",
                "--non-interactive",
                "--results-dir",
                "/data/results",
                "--webhook-url",
                "https://slack.example.com/hook",
            ],
        )

        assert result.exit_code == 0

        config_path = work_dir / ".lem-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["results_dir"] == "/data/results"
        assert config["notifications"]["webhook_url"] == "https://slack.example.com/hook"


class TestInitExistingConfig:
    """Tests for lem init command when config already exists."""

    def test_init_non_interactive_updates_existing(self, work_dir: Path) -> None:
        """Non-interactive mode silently updates existing config."""
        # Create existing config
        existing_config = {
            "results_dir": "old_results",
            "thermal_gaps": {"between_experiments": 30.0},
        }
        config_path = work_dir / ".lem-config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(existing_config, f)

        # Run init with new results dir
        result = runner.invoke(app, ["init", "--non-interactive", "--results-dir", "/new/results"])

        assert result.exit_code == 0

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # New value applied
        assert config["results_dir"] == "/new/results"
        # Existing value preserved from file (read back by load_user_config)
        assert config["thermal_gaps"]["between_experiments"] == 30.0


class TestInitHelp:
    """Tests for lem init command help output."""

    def test_init_help_shows_options(self) -> None:
        """Run --help, verify --non-interactive, --results-dir, --webhook-url documented."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "--non-interactive" in result.stdout
        assert "--results-dir" in result.stdout
        assert "--webhook-url" in result.stdout
        assert "Initialize project" in result.stdout or "configuration wizard" in result.stdout


class TestInitDoctorIntegration:
    """Tests for init command running doctor after config creation."""

    def test_init_runs_doctor_after_config(self, work_dir: Path) -> None:
        """Init command runs doctor diagnostics after creating config."""
        result = runner.invoke(app, ["init", "--non-interactive"])

        assert result.exit_code == 0
        # Doctor output should be present
        assert "Running diagnostics" in result.stdout
        # Doctor command shows environment info
        assert "Python" in result.stdout or "GPU" in result.stdout


class TestInitConfigContent:
    """Tests for the content of generated config files."""

    def test_init_config_has_expected_structure(self, work_dir: Path) -> None:
        """Generated config has all expected sections."""
        runner.invoke(app, ["init", "--non-interactive"])

        config_path = work_dir / ".lem-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Expected top-level keys
        assert "results_dir" in config
        assert "thermal_gaps" in config
        assert "docker" in config
        assert "notifications" in config

        # thermal_gaps structure
        assert "between_experiments" in config["thermal_gaps"]
        assert "between_cycles" in config["thermal_gaps"]

        # docker structure
        assert "strategy" in config["docker"]
        assert config["docker"]["strategy"] in ("ephemeral", "persistent")

        # notifications structure
        assert "on_complete" in config["notifications"]
        assert "on_failure" in config["notifications"]

    def test_init_config_no_default_backend(self, work_dir: Path) -> None:
        """Generated config does not include default_backend field."""
        runner.invoke(app, ["init", "--non-interactive"])

        config_path = work_dir / ".lem-config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # default_backend was removed in Phase 2.3-01
        assert "default_backend" not in config
