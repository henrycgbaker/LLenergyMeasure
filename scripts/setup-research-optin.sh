#!/usr/bin/env bash
# Opt out of the research/ directory in this clone.
#
# The research/ directory holds long-running research artefacts that
# aren't part of the production codebase. It is included by default in
# fresh clones; this script lets casual contributors who don't need
# the research corpus skip it (saves ~5 MB on disk + ~1300 files).
#
# Run from the repo root. To restore research/ later, run:
#   git sparse-checkout disable
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

git sparse-checkout init --no-cone
git sparse-checkout set '/*' '!/research/'

echo "research/ excluded from checkout."
echo "Run 'git sparse-checkout disable' to restore."
