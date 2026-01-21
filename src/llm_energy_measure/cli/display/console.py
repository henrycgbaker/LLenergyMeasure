"""Rich console setup and basic styling.

This module provides the shared console instance and basic formatting utilities
that are used throughout the CLI display system.
"""

from __future__ import annotations

import os

from rich.console import Console

# Respect NO_COLOR environment variable for testing and accessibility
console = Console(no_color=os.environ.get("NO_COLOR") == "1")


def format_duration(seconds: float) -> str:
    """Format seconds as human-readable duration.

    Args:
        seconds: Duration in seconds.

    Returns:
        Human-readable string (e.g., "5.2s", "3.1m", "1.5h").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
