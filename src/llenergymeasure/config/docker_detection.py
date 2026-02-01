"""Docker container detection and campaign dispatch logic."""

from __future__ import annotations

from pathlib import Path


def is_inside_docker() -> bool:
    """Check if code is running inside a Docker container.

    Uses two detection methods:
    1. Presence of /.dockerenv file
    2. Docker/containerd strings in /proc/1/cgroup

    Returns:
        True if running inside Docker container, False otherwise.
    """
    # Method 1: Check for /.dockerenv
    if Path("/.dockerenv").exists():
        return True

    # Method 2: Check /proc/1/cgroup for docker/containerd
    try:
        with open("/proc/1/cgroup") as f:
            content = f.read()
            if "docker" in content or "containerd" in content:
                return True
    except (FileNotFoundError, PermissionError):
        # Expected on Windows/macOS or when /proc not accessible
        pass

    return False


def should_use_docker_for_campaign(backends: list[str]) -> bool:
    """Determine if campaign should dispatch to Docker or run locally.

    Dispatch logic:
    - Already in Docker → run locally (no nested containers)
    - Single backend + installed locally → run locally
    - Multi-backend OR backend not installed → dispatch to Docker

    Args:
        backends: List of backend names (e.g., ["pytorch", "vllm"]).

    Returns:
        True if campaign should use Docker, False if should run locally.
    """
    # Import inside function to avoid circular deps at module level
    from llenergymeasure.config.backend_detection import is_backend_available

    # Already in Docker → no nested containers
    if is_inside_docker():
        return False

    # Single backend installed locally → run locally
    # Multi-backend or backend not installed → use Docker
    return not (len(backends) == 1 and is_backend_available(backends[0]))
