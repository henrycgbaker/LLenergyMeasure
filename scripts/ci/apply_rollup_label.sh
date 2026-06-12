#!/usr/bin/env bash
# Apply the cross-pipeline `safe-bump` rollup label for one engine cell.
#
# The invariants + schemas cells share one rollup label, so a no-changes re-run
# of either must not clobber the other's verdict: only a run that actually
# classified the diff (breaking / safe) ever touches the label. A no-changes
# classification is a deliberate no-op (it carries no fresh signal and must not
# downgrade an existing "breaking" back to "safe-bump").
#
# Usage:
#   PR_NUMBER=123 CLASSIFICATION=safe GH_TOKEN=... \
#     scripts/ci/apply_rollup_label.sh
#
# Reads from env: PR_NUMBER, CLASSIFICATION (breaking|safe|no-changes).

set -euo pipefail

: "${PR_NUMBER:?PR_NUMBER required}"
: "${CLASSIFICATION:?CLASSIFICATION required (breaking|safe|no-changes)}"

case "$CLASSIFICATION" in
  breaking)   gh pr edit "$PR_NUMBER" --remove-label "safe-bump" 2>/dev/null || true ;;
  safe)       gh pr edit "$PR_NUMBER" --add-label "safe-bump" 2>/dev/null || true ;;
  no-changes) ;;
esac
