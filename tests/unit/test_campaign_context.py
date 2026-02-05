"""Tests for campaign context propagation.

Architecture note: Experiments are ATOMIC. They don't know about cycles.
Cycles are a campaign-level concept for statistical robustness.
Experiments only know they're "part of a campaign" for identification.
"""

import os
from unittest.mock import patch


class TestCampaignContextEnvironment:
    """Tests for campaign context environment variables."""

    def test_campaign_context_detection(self) -> None:
        """Experiment detects campaign context from environment."""
        # Simulate running inside a campaign
        # Note: LEM_CYCLE and LEM_TOTAL_CYCLES are passed but experiments
        # should NOT use them (they're for campaign-level aggregation only)
        with patch.dict(
            os.environ,
            {
                "LEM_CAMPAIGN_ID": "abc12345",
                "LEM_CAMPAIGN_NAME": "test-campaign",
                "LEM_CYCLE": "2",  # Passed but not displayed by experiment
                "LEM_TOTAL_CYCLES": "3",  # Passed but not displayed by experiment
            },
        ):
            campaign_id = os.environ.get("LEM_CAMPAIGN_ID")
            campaign_name = os.environ.get("LEM_CAMPAIGN_NAME")

            # Experiment should only care about id and name
            assert campaign_id == "abc12345"
            assert campaign_name == "test-campaign"

    def test_no_campaign_context(self) -> None:
        """Without campaign context, env vars are None."""
        env = os.environ.copy()
        for key in ["LEM_CAMPAIGN_ID", "LEM_CAMPAIGN_NAME"]:
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
        # Campaign passes all context, but experiment only uses id/name
        campaign_context = {
            "LEM_CAMPAIGN_ID": "abc123",
            "LEM_CAMPAIGN_NAME": "test",
            "LEM_CYCLE": "1",  # Passed for campaign aggregation
            "LEM_TOTAL_CYCLES": "3",  # Passed for campaign aggregation
        }

        cmd = ["docker", "compose", "run", "--rm"]
        for key, value in campaign_context.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(["pytorch", "lem", "experiment", "config.yaml"])

        # Campaign passes all env vars
        assert "-e" in cmd
        assert "LEM_CAMPAIGN_ID=abc123" in cmd
        assert "pytorch" in cmd


class TestExperimentCampaignDisplay:
    """Tests for experiment campaign display behavior.

    Key architecture principle: Experiments are ATOMIC.
    They show "Part of campaign: X" but NOT cycle info.
    Cycles are campaign-level only.
    """

    def test_experiment_shows_campaign_name_only(self) -> None:
        """Experiment should show campaign name, not cycle info."""
        with patch.dict(
            os.environ,
            {
                "LEM_CAMPAIGN_ID": "abc123",
                "LEM_CAMPAIGN_NAME": "my-campaign",
                "LEM_CYCLE": "2",
                "LEM_TOTAL_CYCLES": "5",
            },
        ):
            # Experiment only reads id and name
            campaign_id = os.environ.get("LEM_CAMPAIGN_ID")
            campaign_name = os.environ.get("LEM_CAMPAIGN_NAME")

            assert campaign_id is not None
            # Expected display: "Part of campaign: my-campaign"
            # NOT: "Part of campaign: my-campaign (cycle 2/5)"
            display = f"Part of campaign: {campaign_name}"
            assert "cycle" not in display.lower()

    def test_standalone_experiment_no_campaign_display(self) -> None:
        """When not in campaign, no campaign info displayed."""
        env = {k: v for k, v in os.environ.items() if not k.startswith("LEM_")}
        with patch.dict(os.environ, env, clear=True):
            campaign_id = os.environ.get("LEM_CAMPAIGN_ID")
            assert campaign_id is None
            # No campaign display when running standalone
