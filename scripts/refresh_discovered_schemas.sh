#!/usr/bin/env bash
# Rediscover an engine schema by running discovery inside the appropriate
# Docker image, then promote the result into the packaged copy.
#
# Usage: ./scripts/refresh_discovered_schemas.sh {vllm|tensorrt|transformers}
#
# Write path (one writer per step):
#   1. Discovery writes the versioned snapshot
#      engine_versions/<engine>/v<safe>/outputs/schema.discovered.json - the
#      ONLY discovery write target. v<safe> is derived from the current.yaml pin
#      via engine_versions/_outputs.py (the one place that name-mangling lives;
#      this mirrors the scaffold-snapshot make target).
#   2. scripts/promote_schemas.py byte-copies that snapshot into the packaged
#      shadow src/llenergymeasure/engines/<engine>/schema.discovered.json - the
#      ONLY writer of the src copy, with no transformation.
# Codegen (scripts/engine_producers/regen_engine_configs.py) reads the outputs/
# snapshot. This script runs both steps, then prints the `git diff` of the two
# files. It does NOT commit. The committed JSON IS the canonical SSOT - authority
# comes from `git commit`, not from who ran discovery.
#
# Legitimate refresh (e.g. you bumped an engine pin in current.yaml):
#   review the diff, `git add` both files, and open a PR.
# Exploring a fork or stale image:
#   `git checkout --` the two schema paths printed in the diff.
#
# Image selection matches scripts/probe_candidates.sh and the CI engine cells:
# the pinned version is read from engine_versions/<engine>/current.yaml (the
# Renovate-writable SSOT), then
#   vllm         -> pristine vllm/vllm-openai:v<version> (vllm pre-installed)
#   tensorrt     -> pristine nvcr.io/nvidia/tensorrt-llm/release:<version>
#                   (works around llenergymeasure:tensorrt's cuKernelGetName bug)
#   transformers -> llenergymeasure:transformers-<version> (base pytorch image
#                   has no transformers package; our Dockerfile pip-installs it
#                   at the version passed via --build-arg TRANSFORMERS_VERSION)
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/refresh_discovered_schemas.sh {vllm|tensorrt|transformers}

Builds or pulls the engine's discovery image and runs discovery inside it,
writing the versioned snapshot
engine_versions/<engine>/v<safe>/outputs/schema.discovered.json, then promotes
it byte-for-byte into
src/llenergymeasure/engines/<engine>/schema.discovered.json. Prints the git diff
of both files. Does NOT commit.
EOF
}

if [[ $# -ne 1 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    usage >&2
    exit 1
fi

ENGINE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v yq >/dev/null 2>&1; then
    echo "yq is required to read the engine pin from current.yaml but was not found on PATH" >&2
    exit 1
fi

# Pinned engine version from the per-engine SSOT (same locus probe_candidates.sh
# and the CI cells read).
_pinned_version() {
    yq '.library.current_version' "$REPO_ROOT/engine_versions/$1/current.yaml"
}

VER="$(_pinned_version "$ENGINE")"
if [[ -z "$VER" || "$VER" == "null" ]]; then
    echo "Could not resolve $ENGINE version from engine_versions/$ENGINE/current.yaml" >&2
    exit 1
fi

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
            docker build --build-arg "TRANSFORMERS_VERSION=${VER}" \
                -f "$REPO_ROOT/docker/Dockerfile.transformers" -t "$IMAGE" "$REPO_ROOT"
        fi
        ;;
    *)
        echo "Unknown engine: $ENGINE" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ -z "${IMAGE:-}" ]]; then
    echo "Failed to resolve image for engine '$ENGINE'" >&2
    exit 1
fi

# Discovery write target: the versioned snapshot under engine_versions/. Derive
# the v<safe> directory from the pin via engine_versions/_outputs.py (the one
# place name-mangling lives; mirrors the scaffold-snapshot make target). The path
# is emitted repo-relative so it can be handed to both docker (-v mount) and git.
OUTPUT_REL="$(cd "$REPO_ROOT" && python3 -c "
from pathlib import Path
from engine_versions import _outputs
print(_outputs.schema_path('$ENGINE', '$VER').relative_to(Path.cwd()))
")"
if [[ -z "$OUTPUT_REL" ]]; then
    echo "[$ENGINE] Failed to derive the snapshot output path from pin $VER" >&2
    exit 1
fi
# The packaged copy the promotion step writes (never written directly here).
SRC_REL="src/llenergymeasure/engines/${ENGINE}/schema.discovered.json"

echo "[$ENGINE] Running discovery inside $IMAGE..." >&2
echo "[$ENGINE] Discovery writes $OUTPUT_REL" >&2
# Forward LLENERGY_DISCOVERY_FROZEN_AT into the container if the caller (CI)
# set it. The introspectors use it to pin `discovered_at` to a
# stable anchor, breaking the writeback->resync loop on unchanged source.
#
# --user maps the host uid, which has no /etc/passwd entry in the image; torch
# calls getpass.getuser() at import and raises "uid not found" without USER set.
# HOME=/tmp keeps any torch/HF cache writes on a writable ephemeral path.
DOCKER_ARGS=(
    --rm --gpus all
    --user "$(id -u):$(id -g)"
    -e HOME=/tmp -e USER=llem-discovery
    -e LLENERGY_DISCOVERY_FROZEN_AT="${LLENERGY_DISCOVERY_FROZEN_AT:-}"
    -v "$REPO_ROOT:/repo" -w /repo
)
RUNNER_ARGS=(
    -m scripts.engine_producers._schemas_runner
    --engine "$ENGINE"
    --image-ref "$IMAGE"
    --output "/repo/$OUTPUT_REL"
)
if [[ "$ENGINE" == "tensorrt" ]]; then
    # Keep the NVIDIA entrypoint (matches scripts/probe_candidates.sh): from
    # 1.2.1 the NGC image sets up /usr/local/tensorrt/lib on LD_LIBRARY_PATH
    # in /etc/shinit_v2, not in the static image env, so bypassing the
    # entrypoint makes `import tensorrt` fail with a missing libnvonnxparser.
    docker run "${DOCKER_ARGS[@]}" "$IMAGE" python3 "${RUNNER_ARGS[@]}"
else
    docker run "${DOCKER_ARGS[@]}" --entrypoint python3 "$IMAGE" "${RUNNER_ARGS[@]}"
fi

cd "$REPO_ROOT"

# Promote the versioned snapshot into the packaged src copy. This is the ONLY
# writer of the src copy - discovery above never touches src directly.
echo "[$ENGINE] Promoting $OUTPUT_REL -> $SRC_REL" >&2
python3 "$REPO_ROOT/scripts/promote_schemas.py" --engine "$ENGINE"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "[$ENGINE] Not inside a git repo - skipping diff output." >&2
    exit 0
fi

if git diff --quiet -- "$OUTPUT_REL" "$SRC_REL" 2>/dev/null \
    && [[ -z "$(git status --porcelain -- "$OUTPUT_REL" "$SRC_REL")" ]]; then
    echo "[$ENGINE] No changes to discovered schema." >&2
    exit 0
fi

echo "" >&2
echo "=== git diff --stat (snapshot + packaged copy) ===" >&2
git diff --stat -- "$OUTPUT_REL" "$SRC_REL" || true
echo "" >&2
echo "=== git diff (first 200 lines) ===" >&2
git --no-pager diff -- "$OUTPUT_REL" "$SRC_REL" | head -200 || true
echo "" >&2
cat <<EOF >&2
Schema changed.
  - Legitimate refresh? Review the diff, \`git add $OUTPUT_REL $SRC_REL\`, and open a PR.
  - Exploring a custom fork or stale image? Revert with:
      git checkout -- $OUTPUT_REL $SRC_REL
EOF
