"""Webhook notification sender for experiment events.

Sends HTTP POST notifications to a configured webhook URL when
experiments complete or fail. Handles timeouts and errors gracefully
without interrupting campaign execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from loguru import logger

from llenergymeasure.config.user_config import load_user_config


def send_webhook_notification(
    event_type: Literal["complete", "failure"],
    experiment_id: str,
    campaign_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Send webhook notification if configured.

    Loads user config to check if webhooks are enabled, then sends
    a POST request to the configured URL. Handles all errors gracefully
    without raising exceptions.

    Args:
        event_type: "complete" or "failure"
        experiment_id: Experiment identifier
        campaign_id: Optional campaign identifier
        payload: Additional data to include in the notification

    Returns:
        True if sent successfully, False otherwise (or if not configured)
    """
    try:
        config = load_user_config()
    except ValueError as e:
        logger.warning("Failed to load user config for webhook: {}", e)
        return False

    notifications = config.notifications

    # Check if webhook is configured
    if not notifications.webhook_url:
        return False

    # Check if this event type is enabled
    if event_type == "complete" and not notifications.on_complete:
        logger.debug("Webhook notification skipped: on_complete disabled")
        return False
    if event_type == "failure" and not notifications.on_failure:
        logger.debug("Webhook notification skipped: on_failure disabled")
        return False

    # Build the notification payload
    notification_data: dict[str, Any] = {
        "event_type": event_type,
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if campaign_id:
        notification_data["campaign_id"] = campaign_id

    if payload:
        notification_data["data"] = payload

    # Send the webhook
    try:
        response = httpx.post(
            notifications.webhook_url,
            json=notification_data,
            timeout=10.0,
            follow_redirects=True,
        )
        response.raise_for_status()
        logger.debug(
            "Webhook notification sent: {} {} -> {}",
            event_type,
            experiment_id,
            response.status_code,
        )
        return True
    except httpx.TimeoutException:
        logger.warning(
            "Webhook notification timed out: {} {}",
            event_type,
            experiment_id,
        )
        return False
    except httpx.HTTPStatusError as e:
        logger.warning(
            "Webhook notification failed with status {}: {} {}",
            e.response.status_code,
            event_type,
            experiment_id,
        )
        return False
    except httpx.RequestError as e:
        logger.warning(
            "Webhook notification request failed: {} {} - {}",
            event_type,
            experiment_id,
            e,
        )
        return False


__all__ = ["send_webhook_notification"]
