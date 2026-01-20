#!/bin/bash
# Production entrypoint with PUID/PGID support
# Based on LinuxServer.io pattern for seamless host/container permission mapping
#
# Usage:
#   docker run -e PUID=$(id -u) -e PGID=$(id -g) ...
#
# If PUID/PGID not set, auto-detects from mounted /app/results directory ownership.
# This allows zero-config usage while still supporting explicit overrides.

set -e

# Auto-detect PUID/PGID from mounted results directory if not explicitly set
# Precedence: explicit env var > auto-detect from mount > root (fallback)
if [ -z "$PUID" ] && [ -d "/app/results" ]; then
    PUID=$(stat -c %u /app/results 2>/dev/null || echo "0")
    PGID=$(stat -c %g /app/results 2>/dev/null || echo "0")
fi

# Fall back to root if detection fails or directory doesn't exist
PUID=${PUID:-0}
PGID=${PGID:-0}

# If running as root and PUID/PGID specified, set up the user
if [ "$(id -u)" = "0" ] && [ "$PUID" != "0" ]; then
    # Create group if it doesn't exist
    if ! getent group appgroup >/dev/null 2>&1; then
        groupadd -g "$PGID" appgroup
    fi

    # Create user if it doesn't exist
    if ! getent passwd appuser >/dev/null 2>&1; then
        useradd -u "$PUID" -g appgroup -s /bin/bash -m appuser
    fi

    # Ensure writable directories have correct ownership
    for dir in /app/results /app/.state /app/.cache/huggingface; do
        if [ -d "$dir" ]; then
            chown -R appuser:appgroup "$dir" 2>/dev/null || true
        fi
    done

    # Run command as appuser using gosu
    exec gosu appuser "$@"
else
    # Running as non-root or PUID=0, just exec the command
    exec "$@"
fi
