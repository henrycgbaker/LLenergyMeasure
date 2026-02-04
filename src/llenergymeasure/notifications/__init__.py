"""Notification services for experiments and campaigns.

This module provides webhook notifications for experiment completion
and failure events, configured via .lem-config.yaml.
"""

from llenergymeasure.notifications.webhook import send_webhook_notification

__all__ = ["send_webhook_notification"]
