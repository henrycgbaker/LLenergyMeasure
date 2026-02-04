"""Unit tests for webhook notification sender."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from llenergymeasure.config.user_config import NotificationsConfig, UserConfig
from llenergymeasure.notifications.webhook import send_webhook_notification


class TestWebhookNotConfigured:
    """Tests for webhook when not configured."""

    def test_webhook_not_configured_returns_false(self) -> None:
        """No notifications webhook configured -> returns False, no HTTP call."""
        # Config with no webhook URL
        mock_config = UserConfig(notifications=NotificationsConfig(webhook_url=None))

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch("httpx.post") as mock_post,
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
            )

            assert result is False
            mock_post.assert_not_called()


class TestWebhookDisabledEvents:
    """Tests for webhook with specific events disabled."""

    def test_webhook_disabled_on_complete_returns_false(self) -> None:
        """on_complete=False, event='complete' -> returns False."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
                on_complete=False,
                on_failure=True,
            )
        )

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch("httpx.post") as mock_post,
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
            )

            assert result is False
            mock_post.assert_not_called()

    def test_webhook_disabled_on_failure_returns_false(self) -> None:
        """on_failure=False, event='failure' -> returns False."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
                on_complete=True,
                on_failure=False,
            )
        )

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch("httpx.post") as mock_post,
        ):
            result = send_webhook_notification(
                event_type="failure",
                experiment_id="exp-001",
            )

            assert result is False
            mock_post.assert_not_called()


class TestWebhookSuccess:
    """Tests for successful webhook delivery."""

    def test_webhook_success_returns_true(self) -> None:
        """Mock httpx.post success -> returns True."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
                on_complete=True,
            )
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch("httpx.post", return_value=mock_response) as mock_post,
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
                campaign_id="campaign-001",
            )

            assert result is True
            mock_post.assert_called_once()

            # Verify call arguments
            call_args = mock_post.call_args
            assert call_args[0][0] == "https://example.com/hook"
            json_data = call_args[1]["json"]
            assert json_data["event_type"] == "complete"
            assert json_data["experiment_id"] == "exp-001"
            assert json_data["campaign_id"] == "campaign-001"
            assert "timestamp" in json_data

    def test_webhook_success_with_payload(self) -> None:
        """Additional payload data included in notification."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
            )
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch("httpx.post", return_value=mock_response) as mock_post,
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
                payload={"tokens": 1000, "energy_j": 50.5},
            )

            assert result is True
            json_data = mock_post.call_args[1]["json"]
            assert json_data["data"]["tokens"] == 1000
            assert json_data["data"]["energy_j"] == 50.5


class TestWebhookErrors:
    """Tests for webhook error handling."""

    def test_webhook_timeout_returns_false(self) -> None:
        """Mock httpx.post raises TimeoutException -> returns False."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
            )
        )

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch("httpx.post", side_effect=httpx.TimeoutException("Request timed out")),
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
            )

            assert result is False

    def test_webhook_http_error_returns_false(self) -> None:
        """Mock httpx.post raises HTTPStatusError -> returns False."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
            )
        )

        mock_response = MagicMock()
        mock_response.status_code = 500

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch(
                "httpx.post",
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=mock_response,
                ),
            ),
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
            )

            assert result is False

    def test_webhook_request_error_returns_false(self) -> None:
        """Mock httpx.post raises RequestError -> returns False."""
        mock_config = UserConfig(
            notifications=NotificationsConfig(
                webhook_url="https://example.com/hook",
            )
        )

        with (
            patch(
                "llenergymeasure.notifications.webhook.load_user_config",
                return_value=mock_config,
            ),
            patch(
                "httpx.post",
                side_effect=httpx.RequestError("Connection failed"),
            ),
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
            )

            assert result is False

    def test_webhook_config_load_error_returns_false(self) -> None:
        """Failed to load user config -> returns False."""
        with patch(
            "llenergymeasure.notifications.webhook.load_user_config",
            side_effect=ValueError("Invalid config"),
        ):
            result = send_webhook_notification(
                event_type="complete",
                experiment_id="exp-001",
            )

            assert result is False
