#!/bin/bash
# Production entrypoint with PUID/PGID support
# Based on LinuxServer.io pattern for seamless host/container permission mapping
#
# Usage:
#   docker run -e PUID=$(id -u) -e PGID=$(id -g) ...
#
# PUID/PGID are REQUIRED - no auto-detection magic that can fail silently.
# Generate .env with: ./setup.sh (or copy from .env.example)

set -e

# =============================================================================
# Validate PUID/PGID are set (required - no auto-detection)
# =============================================================================
if [ -z "$PUID" ] || [ -z "$PGID" ]; then
    echo "ERROR: PUID and PGID environment variables are required."
    echo ""
    echo "Quick fix:"
    echo "  1. Run: ./setup.sh  (creates .env automatically)"
    echo "  2. Or:  cp .env.example .env && edit .env"
    echo "  3. Or:  PUID=\$(id -u) PGID=\$(id -g) docker compose run ..."
    echo ""
    echo "See docs/deployment.md for details."
    exit 1
fi

# =============================================================================
# Set up user/group if running as root with non-root PUID
# =============================================================================
if [ "$(id -u)" = "0" ] && [ "$PUID" != "0" ]; then
    # Create group if it doesn't exist
    if ! getent group appgroup >/dev/null 2>&1; then
        groupadd -g "$PGID" appgroup 2>/dev/null || true
    fi

    # Create user if it doesn't exist
    if ! getent passwd appuser >/dev/null 2>&1; then
        useradd -u "$PUID" -g "$PGID" -s /bin/bash -m appuser 2>/dev/null || true
    fi

    # ==========================================================================
    # Create and own directories BEFORE attempting chown
    # This fixes the bootstrap problem where directories don't exist on first run
    # ==========================================================================

    # Results directory: bind mount, needs ownership for writing
    # Single-level chown only (not recursive) - fast even with many results
    mkdir -p /app/results
    chown appuser:appgroup /app/results 2>/dev/null || true

    # State directory: named volume, just ensure it exists
    # Named volumes are already owned correctly by Docker
    mkdir -p /app/.state

    # HF cache: named volume, no chown needed (Docker-managed)
    # Just ensure the directory exists
    mkdir -p /app/.cache/huggingface

    # TensorRT cache: named volume, no chown needed (Docker-managed)
    mkdir -p /app/.cache/tensorrt-engines

    # ==========================================================================
    # Run command as appuser
    # ==========================================================================
    exec gosu appuser "$@"
else
    # Running as non-root or PUID=0, just exec the command
    exec "$@"
fi
