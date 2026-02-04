"""Unit tests for lem resume command."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from typer.testing import CliRunner

from llenergymeasure.cli import app
from llenergymeasure.orchestration.manifest import (
    CampaignManifest,
    CampaignManifestEntry,
    ManifestManager,
)

if TYPE_CHECKING:
    from pytest import MonkeyPatch

# Pass NO_COLOR to disable Rich colors in test output for consistent assertions
runner = CliRunner(env={"NO_COLOR": "1"})


@pytest.fixture
def state_dir(tmp_path: Path, monkeypatch: MonkeyPatch) -> Path:
    """Create and configure .state directory in tmp_path."""
    state = tmp_path / ".state"
    state.mkdir()
    # Change working directory to tmp_path so .state/ lookups find our temp dir
    monkeypatch.chdir(tmp_path)
    return state


@pytest.fixture
def incomplete_manifest(state_dir: Path) -> CampaignManifest:
    """Create an incomplete campaign manifest in state_dir."""
    manifest_path = state_dir / "campaign_manifest.json"
    manifest_mgr = ManifestManager(manifest_path)

    experiments = [
        CampaignManifestEntry(
            exp_id="exp-001",
            config_name="pytorch_base",
            config_path="configs/pytorch_base.yaml",
            config_hash="abc123",
            backend="pytorch",
            container="pytorch",
            cycle_index=0,
            status="completed",
        ),
        CampaignManifestEntry(
            exp_id="exp-002",
            config_name="vllm_base",
            config_path="configs/vllm_base.yaml",
            config_hash="def456",
            backend="vllm",
            container="vllm",
            cycle_index=0,
            status="pending",
        ),
    ]

    manifest = manifest_mgr.create_manifest(
        campaign_id="test-campaign-001",
        campaign_name="Test Campaign",
        config_hash="hash123",
        experiments=experiments,
    )
    manifest_mgr.save(manifest)
    return manifest


class TestResumeNoState:
    """Tests for resume command when no state exists."""

    def test_resume_no_state_directory(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """No .state/ exists -> exits with 'No interrupted work found'."""
        # Change to tmp_path where no .state exists
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["resume"])

        assert result.exit_code == 1
        assert "No interrupted work found" in result.stdout

    def test_resume_no_manifests(self, state_dir: Path) -> None:
        """Empty .state/ directory -> exits with 'No interrupted campaigns found'."""
        # state_dir fixture creates empty .state/ and chdir to parent

        result = runner.invoke(app, ["resume"])

        assert result.exit_code == 1
        assert "No interrupted campaigns found" in result.stdout


class TestResumeDryRun:
    """Tests for resume command --dry-run flag."""

    def test_resume_dry_run_shows_campaign(
        self, state_dir: Path, incomplete_manifest: CampaignManifest
    ) -> None:
        """Create mock manifest, run --dry-run, verify output shows campaign info."""
        result = runner.invoke(app, ["resume", "--dry-run"])

        assert result.exit_code == 0
        assert "Dry run" in result.stdout
        assert "Test Campaign" in result.stdout
        assert "test-campaign-001" in result.stdout
        # Check progress info
        assert "1/2" in result.stdout or "Completed: 1/2" in result.stdout
        assert "Pending: 1" in result.stdout


class TestResumeWipe:
    """Tests for resume command --wipe flag."""

    def test_resume_wipe_clears_state(self, state_dir: Path) -> None:
        """Create .state/ with files, run --wipe with confirm, verify directory deleted."""
        # Create some files in state_dir
        (state_dir / "campaign_manifest.json").write_text("{}")
        (state_dir / "temp_file.txt").write_text("temp")

        # Confirm 'y' for the wipe confirmation
        result = runner.invoke(app, ["resume", "--wipe"], input="y\n")

        assert result.exit_code == 0
        assert "Cleared all state files" in result.stdout
        assert not state_dir.exists()

    def test_resume_wipe_cancelled(self, state_dir: Path) -> None:
        """Run --wipe, decline confirmation, verify state still exists."""
        (state_dir / "campaign_manifest.json").write_text("{}")

        # Decline with 'n'
        result = runner.invoke(app, ["resume", "--wipe"], input="n\n")

        # Typer aborts on declined confirm
        assert result.exit_code != 0
        assert state_dir.exists()
        assert (state_dir / "campaign_manifest.json").exists()

    def test_resume_wipe_no_state(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        """Wipe with no .state/ directory shows nothing to clear."""
        monkeypatch.chdir(tmp_path)

        result = runner.invoke(app, ["resume", "--wipe"])

        assert result.exit_code == 0
        assert "No state directory found" in result.stdout or "Nothing to clear" in result.stdout


class TestResumeHelp:
    """Tests for resume command help output."""

    def test_resume_help_shows_options(self) -> None:
        """Run --help, verify --dry-run and --wipe documented."""
        result = runner.invoke(app, ["resume", "--help"])

        assert result.exit_code == 0
        assert "--dry-run" in result.stdout
        assert "--wipe" in result.stdout
        assert "Discover and resume interrupted campaigns" in result.stdout


class TestResumeWithFailedExperiments:
    """Tests for resume command with failed experiments in manifest."""

    def test_resume_dry_run_shows_failed_count(
        self, state_dir: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Manifest with failed experiments shows failure count in dry-run."""
        manifest_path = state_dir / "campaign_manifest.json"
        manifest_mgr = ManifestManager(manifest_path)

        experiments = [
            CampaignManifestEntry(
                exp_id="exp-001",
                config_name="config1",
                config_path="configs/config1.yaml",
                config_hash="hash1",
                backend="pytorch",
                container="pytorch",
                cycle_index=0,
                status="completed",
            ),
            CampaignManifestEntry(
                exp_id="exp-002",
                config_name="config2",
                config_path="configs/config2.yaml",
                config_hash="hash2",
                backend="pytorch",
                container="pytorch",
                cycle_index=0,
                status="failed",
                error="Out of memory",
            ),
            CampaignManifestEntry(
                exp_id="exp-003",
                config_name="config3",
                config_path="configs/config3.yaml",
                config_hash="hash3",
                backend="pytorch",
                container="pytorch",
                cycle_index=0,
                status="pending",
            ),
        ]

        manifest = manifest_mgr.create_manifest(
            campaign_id="campaign-with-failures",
            campaign_name="Campaign With Failures",
            config_hash="hash123",
            experiments=experiments,
        )
        manifest_mgr.save(manifest)

        result = runner.invoke(app, ["resume", "--dry-run"])

        assert result.exit_code == 0
        assert "Failed: 1" in result.stdout


class TestResumeCompletedCampaign:
    """Tests for resume when all campaigns are complete."""

    def test_resume_all_complete_shows_message(
        self, state_dir: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """All campaigns completed -> shows 'No interrupted campaigns found'."""
        manifest_path = state_dir / "campaign_manifest.json"
        manifest_mgr = ManifestManager(manifest_path)

        experiments = [
            CampaignManifestEntry(
                exp_id="exp-001",
                config_name="config1",
                config_path="configs/config1.yaml",
                config_hash="hash1",
                backend="pytorch",
                container="pytorch",
                cycle_index=0,
                status="completed",
            ),
            CampaignManifestEntry(
                exp_id="exp-002",
                config_name="config2",
                config_path="configs/config2.yaml",
                config_hash="hash2",
                backend="pytorch",
                container="pytorch",
                cycle_index=0,
                status="completed",
            ),
        ]

        manifest = manifest_mgr.create_manifest(
            campaign_id="complete-campaign",
            campaign_name="Complete Campaign",
            config_hash="hash123",
            experiments=experiments,
        )
        manifest_mgr.save(manifest)

        result = runner.invoke(app, ["resume"])

        assert result.exit_code == 1
        assert "No interrupted campaigns found" in result.stdout
        assert "All discovered campaigns have completed" in result.stdout
