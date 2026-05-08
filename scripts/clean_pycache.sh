#!/usr/bin/env bash
# Remove all __pycache__ directories under the repo tree.
#
# Pre-existing root-owned __pycache__ entries (written by Docker-run probes)
# can hold stale .pyc files referencing module shapes that have moved during
# a session's edits. The pipeline's determinism check after a run treats any
# diff in tracked files as a fingerprint shift; stale .pyc-loaded code can
# spoof "diff present" even after a clean re-run.
#
# Run before any determinism-sensitive command:
#   ./scripts/clean_pycache.sh
#
# Idempotent and safe to run multiple times.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

# Descend into all places Python may have compiled bytecode: src/, tests/,
# scripts/, engine_versions/. -prune on .git and node_modules to skip them.
find . \
    \( -path ./.git -o -path ./node_modules -o -path ./.venv -o -path ./venv \) -prune \
    -o -type d -name __pycache__ -print0 \
    | xargs -0 -r rm -rf 2>/dev/null || true

echo "Cleaned __pycache__ trees under $REPO_ROOT" >&2
