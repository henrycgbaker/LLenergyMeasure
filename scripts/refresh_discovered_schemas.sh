#!/usr/bin/env bash
# Rediscover an engine schema by running discovery inside the
# appropriate Docker image.
#
# Usage: ./scripts/refresh_discovered_schemas.sh {vllm|tensorrt|transformers}
#
# Always writes to engine_versions/<engine>/v<safe>/outputs/schema.discovered.json
# and prints the resulting `git diff`. Does NOT commit. The discovered JSON
# file IS the canonical SSOT - authority comes from `git commit`, not from
# who ran discovery.
#
# Legitimate refresh (e.g. you bumped library.current_version in current.yaml):
#   review the diff, `git add`, and open a PR.
# Exploring a fork or stale image:
#   `git checkout engine_versions/<engine>/v<safe>/outputs/schema.discovered.json`
#
# Discovery image selection:
#   vllm         -> pristine vllm/vllm-openai:v<ver> (vllm pre-installed)
#   tensorrt     -> pristine nvcr.io/nvidia/tensorrt-llm/release:<ver>
#                   (works around llenergymeasure:tensorrt's cuKernelGetName bug)
#   transformers -> llenergymeasure:transformers-<ver> (base pytorch image has no
#                   transformers package; our Dockerfile pip-installs it at the
#                   version pinned by ARG TRANSFORMERS_VERSION)
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/refresh_discovered_schemas.sh {vllm|tensorrt|transformers}

Builds or pulls the engine's discovery image, runs discovery inside it,
writes engine_versions/<engine>/v<safe>/outputs/schema.discovered.json, and
prints the git diff. Does NOT commit.
EOF
}

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage >&2
    exit 1
fi

ENGINE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SSOT="${REPO_ROOT}/engine_versions/${ENGINE}/current.yaml"

if [[ ! -f "$SSOT" ]]; then
    echo "[$ENGINE] SSOT $SSOT not found; unsupported engine?" >&2
    usage >&2
    exit 1
fi

VER="$(yq '.library.current_version' "$SSOT")"
if [[ -z "$VER" || "$VER" == "null" ]]; then
    echo "[$ENGINE] Could not resolve library.current_version from $SSOT" >&2
    exit 1
fi
SAFE_VER="v${VER//[.-]/_}"
OUTPUTS_DIR="engine_versions/${ENGINE}/${SAFE_VER}/outputs"

case "$ENGINE" in
    vllm)
        IMAGE="vllm/vllm-openai:v${VER}"
        ;;
    tensorrt)
        IMAGE="nvcr.io/nvidia/tensorrt-llm/release:${VER}"
        ;;
    transformers)
        IMAGE="llenergymeasure:transformers-${VER}"
        if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
            echo "[$ENGINE] Image $IMAGE not found; building from docker/Dockerfile.transformers..." >&2
            docker build -f "$REPO_ROOT/docker/Dockerfile.transformers" -t "$IMAGE" "$REPO_ROOT"
        fi
        ;;
    *)
        echo "Unknown engine: $ENGINE" >&2
        usage >&2
        exit 1
        ;;
esac

OUTPUT_REL="${OUTPUTS_DIR}/schema.discovered.json"
mkdir -p "$REPO_ROOT/$OUTPUTS_DIR"

echo "[$ENGINE] Running discovery inside $IMAGE..." >&2
# Forward LLENERGY_DISCOVERY_FROZEN_AT into the container if the caller (CI)
# set it. The introspectors use it to pin `discovered_at` to a
# stable anchor, breaking the writeback->resync loop on unchanged source.
docker run --rm --gpus all \
    --user "$(id -u):$(id -g)" \
    --entrypoint python3 \
    -e LLENERGY_DISCOVERY_FROZEN_AT="${LLENERGY_DISCOVERY_FROZEN_AT:-}" \
    -v "$REPO_ROOT:/repo" \
    -w /repo \
    "$IMAGE" \
    -m scripts.engine_producers._schemas_runner \
    --engine "$ENGINE" \
    --image-ref "$IMAGE" \
    --output "/repo/$OUTPUT_REL"

cd "$REPO_ROOT"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "[$ENGINE] Not inside a git repo - skipping diff output." >&2
    exit 0
fi

if git diff --quiet -- "$OUTPUT_REL" 2>/dev/null; then
    if [[ -z "$(git status --porcelain -- "$OUTPUT_REL")" ]]; then
        echo "[$ENGINE] No changes to discovered schema." >&2
        exit 0
    fi
fi

echo "" >&2
echo "=== git diff --stat $OUTPUT_REL ===" >&2
git diff --stat -- "$OUTPUT_REL" || true
echo "" >&2
echo "=== git diff $OUTPUT_REL (first 200 lines) ===" >&2
git --no-pager diff -- "$OUTPUT_REL" | head -200 || true
echo "" >&2
cat <<EOF >&2
Schema changed.
  - Legitimate refresh? Review the diff, \`git add $OUTPUT_REL\`, and open a PR.
  - Exploring a custom fork or stale image? Revert with:
      git checkout -- $OUTPUT_REL
EOF
