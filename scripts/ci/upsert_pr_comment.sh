#!/usr/bin/env bash
# Idempotent PR-comment upsert keyed on a stable HTML marker.
#
# Reads body from stdin, looks up an existing PR comment by the marker,
# PATCHes if found, otherwise POSTs new. Byte-compares before PATCH so
# re-runs that produce identical bodies are no-ops (no `updated_at`
# churn). Requires the body itself be deterministic across re-runs -
# diff-emitting workflows must use `diff -u --label <stable> --label
# <stable>` (or equivalent) to suppress filesystem timestamps.
#
# Usage (upsert):
#   echo "<body containing <!-- bot-id: foo -->" | \
#     PR=123 MARKER="bot-id: foo" REPO=owner/name \
#     scripts/ci/upsert_pr_comment.sh
#
# Usage (delete stale comment, e.g. on probe-blocked → pass transition):
#   PR=123 MARKER="bot-id: foo" REPO=owner/name MODE=delete \
#     scripts/ci/upsert_pr_comment.sh
#
# Requires: gh CLI authenticated (GH_TOKEN env), jq.

set -euo pipefail

: "${PR:?PR number required}"
: "${MARKER:?MARKER required (e.g. 'bot-id: rules-check-vllm')}"
: "${REPO:=${GITHUB_REPOSITORY:-}}"
: "${REPO:?REPO required (owner/name)}"
MODE="${MODE:-upsert}"

existing_id=$(
  gh api --paginate "repos/${REPO}/issues/${PR}/comments" --jq \
    "[.[] | select(.body | contains(\"<!-- ${MARKER} -->\")) | .id] | first // empty"
)

case "${MODE}" in
  delete)
    if [[ -n "${existing_id}" ]]; then
      gh api --method DELETE "repos/${REPO}/issues/comments/${existing_id}" >/dev/null
      echo "Deleted stale comment ${existing_id} (marker: ${MARKER})"
    else
      echo "No comment with marker '${MARKER}' to delete; no-op."
    fi
    ;;
  upsert)
    body=$(cat)
    if [[ "${body}" != *"<!-- ${MARKER} -->"* ]]; then
      echo "::error::Comment body missing marker '<!-- ${MARKER} -->'; refusing to post." >&2
      exit 1
    fi
    if [[ -n "${existing_id}" ]]; then
      existing_body=$(
        gh api "repos/${REPO}/issues/comments/${existing_id}" --jq '.body'
      )
      if [[ "${existing_body}" == "${body}" ]]; then
        echo "Comment body unchanged; no-op (marker: ${MARKER})."
      else
        jq -n --arg b "${body}" '{body: $b}' \
          | gh api --method PATCH \
              "repos/${REPO}/issues/comments/${existing_id}" \
              --input - >/dev/null
        echo "Patched comment ${existing_id} (marker: ${MARKER})"
      fi
    else
      printf '%s' "${body}" \
        | gh pr comment "${PR}" --repo "${REPO}" --body-file - >/dev/null
      echo "Created new comment (marker: ${MARKER})"
    fi
    ;;
  *)
    echo "::error::Unknown MODE='${MODE}'; expected 'upsert' or 'delete'." >&2
    exit 1
    ;;
esac
