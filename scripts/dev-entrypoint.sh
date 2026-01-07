#!/bin/bash
# Development container entrypoint
# Installs package in editable mode, then runs command or drops to shell

set -e

# Install in editable mode if not already installed
if ! pip show llm-energy-measure &>/dev/null; then
    echo "Installing llm-energy-measure in editable mode..."
    pip install -e /app --quiet
fi

# Run provided command or drop to shell
if [ $# -eq 0 ]; then
    exec /bin/bash
else
    exec "$@"
fi
