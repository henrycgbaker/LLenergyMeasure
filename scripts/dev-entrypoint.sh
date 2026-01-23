#!/bin/bash
# Development container entrypoint with PUID/PGID support
# Installs package in editable mode, then runs command or drops to shell

set -e

# Auto-detect PUID/PGID from mounted results directory if not explicitly set
if [ -z "$PUID" ] && [ -d "/app/results" ]; then
    PUID=$(stat -c %u /app/results 2>/dev/null || echo "0")
    PGID=$(stat -c %g /app/results 2>/dev/null || echo "0")
fi

PUID=${PUID:-0}
PGID=${PGID:-0}

# Install in editable mode if not already installed (do this as root first)
if ! pip show llm-energy-measure &>/dev/null; then
    echo "Installing llm-energy-measure in editable mode..."
    pip install -e /app --quiet
fi

# If running as root and PUID/PGID specified, set up the user
if [ "$(id -u)" = "0" ] && [ "$PUID" != "0" ]; then
    # Create group if it doesn't exist
    if ! getent group appgroup >/dev/null 2>&1; then
        groupadd -g "$PGID" appgroup 2>/dev/null || true
    fi

    # Create user if it doesn't exist
    if ! getent passwd appuser >/dev/null 2>&1; then
        useradd -u "$PUID" -g appgroup -s /bin/bash -m appuser 2>/dev/null || true
    fi

    # Ensure writable directories have correct ownership
    for dir in /app/results /app/configs /app/.state /app/.cache/huggingface; do
        if [ -d "$dir" ]; then
            chown -R appuser:appgroup "$dir" 2>/dev/null || true
        fi
    done

    # Run command as appuser using gosu (if available) or su
    if command -v gosu &>/dev/null; then
        if [ $# -eq 0 ]; then
            exec gosu appuser /bin/bash
        else
            exec gosu appuser "$@"
        fi
    else
        if [ $# -eq 0 ]; then
            exec su - appuser
        else
            exec su - appuser -c "$*"
        fi
    fi
else
    # Running as non-root or PUID=0, just exec the command
    if [ $# -eq 0 ]; then
        exec /bin/bash
    else
        exec "$@"
    fi
fi
