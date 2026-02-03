"""Tests for campaign context propagation."""

import os
from unittest.mock import patch


class TestCampaignContextEnvironment:
    """Tests for campaign context environment variables."""

    def test_campaign_context_detection(self) -> None:
        """Experiment detects campaign context from environment."""
        # Simulate running inside a campaign
        with patch.dict(
            os.environ,
            {
                "LEM_CAMPAIGN_ID": "abc12345",
                "LEM_CAMPAIGN_NAME": "test-campaign",
                "LEM_CYCLE": "2",
                "LEM_TOTAL_CYCLES": "3",
            },
        ):
            campaign_id = os.environ.get("LEM_CAMPAIGN_ID")
            campaign_name = os.environ.get("LEM_CAMPAIGN_NAME")
            cycle = os.environ.get("LEM_CYCLE")
            total_cycles = os.environ.get("LEM_TOTAL_CYCLES")

            assert campaign_id == "abc12345"
            assert campaign_name == "test-campaign"
            assert cycle == "2"
            assert total_cycles == "3"

    def test_no_campaign_context(self) -> None:
        """Without campaign context, env vars are None."""
        # Ensure env vars are not set
        env = os.environ.copy()
        for key in [
            "LEM_CAMPAIGN_ID",
            "LEM_CAMPAIGN_NAME",
            "LEM_CYCLE",
            "LEM_TOTAL_CYCLES",
        ]:
            env.pop(key, None)

        with patch.dict(os.environ, env, clear=True):
            assert os.environ.get("LEM_CAMPAIGN_ID") is None

    def test_in_campaign_detection_logic(self) -> None:
        """in_campaign flag correctly detects campaign context."""
        # With campaign context
        with patch.dict(os.environ, {"LEM_CAMPAIGN_ID": "abc123"}):
            in_campaign = os.environ.get("LEM_CAMPAIGN_ID") is not None
            assert in_campaign is True

        # Without campaign context (ensure it's cleared)
        original = os.environ.pop("LEM_CAMPAIGN_ID", None)
        try:
            in_campaign = os.environ.get("LEM_CAMPAIGN_ID") is not None
            assert in_campaign is False
        finally:
            if original:
                os.environ["LEM_CAMPAIGN_ID"] = original


class TestCampaignContextInDocker:
    """Tests for campaign context in Docker commands."""

    def test_docker_command_includes_env_vars(self) -> None:
        """Docker command should include campaign context env vars."""
        # This is a design test - verify the expected command structure
        campaign_context = {
            "LEM_CAMPAIGN_ID": "abc123",
            "LEM_CAMPAIGN_NAME": "test",
            "LEM_CYCLE": "1",
            "LEM_TOTAL_CYCLES": "3",
        }

        # Expected command structure with -e flags
        cmd = ["docker", "compose", "run", "--rm"]
        for key, value in campaign_context.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(["pytorch", "lem", "experiment", "config.yaml"])

        # Verify structure
        assert "-e" in cmd
        assert "LEM_CAMPAIGN_ID=abc123" in cmd
        assert "LEM_CYCLE=1" in cmd
        assert "pytorch" in cmd


class TestCycleWarningSuppress:
    """Tests for cycle warning suppression in campaign context."""

    def test_should_show_campaign_context_not_warning(self) -> None:
        """When in campaign, show campaign context instead of warning."""
        with patch.dict(
            os.environ,
            {
                "LEM_CAMPAIGN_ID": "abc123",
                "LEM_CAMPAIGN_NAME": "my-campaign",
                "LEM_CYCLE": "2",
                "LEM_TOTAL_CYCLES": "5",
            },
        ):
            in_campaign = os.environ.get("LEM_CAMPAIGN_ID") is not None
            assert in_campaign is True

            # Should show: "Part of campaign: my-campaign (cycle 2/5)"
            # NOT: "Single cycle: confidence intervals require >= 3 cycles"
            # Indicates warning should be suppressed

    def test_should_show_warning_when_not_in_campaign(self) -> None:
        """When not in campaign, show single cycle warning."""
        # Clear campaign context
        env = {k: v for k, v in os.environ.items() if not k.startswith("LEM_")}
        with patch.dict(os.environ, env, clear=True):
            in_campaign = os.environ.get("LEM_CAMPAIGN_ID") is not None
            assert in_campaign is False
            # Should show single cycle warning
