#!/bin/bash
# API entrypoint with PUID/PGID support
# Based on LinuxServer.io pattern for seamless host/container permission mapping
#
# Usage:
#   docker run -e PUID=$(id -u) -e PGID=$(id -g) ...
#
# If PUID/PGID not set, auto-detects from mounted /app/api directory ownership.

set -e

# Auto-detect PUID/PGID from mounted api directory if not explicitly set
if [ -z "$PUID" ] && [ -d "/app/api" ]; then
    PUID=$(stat -c %u /app/api 2>/dev/null || echo "0")
    PGID=$(stat -c %g /app/api 2>/dev/null || echo "0")
fi

# Fall back to root if detection fails
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

    # Ensure app directory has correct ownership for hot reload
    chown -R appuser:appgroup /app 2>/dev/null || true

    # Run command as appuser using gosu
    exec gosu appuser "$@"
else
    # Running as non-root or PUID=0, just exec the command
    exec "$@"
fi
