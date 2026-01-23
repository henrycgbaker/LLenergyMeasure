#!/bin/bash
# Wrapper script for running experiments in Docker
# Reads num_processes from config file and launches accelerate

set -e

if [ -z "$1" ]; then
    echo "Usage: docker-experiment.sh <config_path> [additional args...]"
    echo "Example: docker-experiment.sh /app/configs/test_tiny.yaml --dataset alpaca -n 100"
    exit 1
fi

CONFIG="$1"
shift

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Extract num_processes from config (defaults to 1)
PROCS=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('num_processes', 1))" "$CONFIG" 2>/dev/null || echo 1)

echo "Running experiment with num_processes=$PROCS from config: $CONFIG"

exec accelerate launch --num_processes "$PROCS" \
    -m llenergymeasure.orchestration.launcher \
    --config "$CONFIG" "$@"
